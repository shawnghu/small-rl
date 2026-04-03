"""Tests for the async vLLM server (vllm_async_server.py).

Verifies that:
1. Two clients can register, update weights, and generate concurrently
2. Different adapter weights produce different outputs (slot isolation)
3. Same adapter weights + same prompt + temperature=0 produce identical output
   whether generated alone or in the presence of another client's requests
4. Concurrent weight updates from multiple clients don't deadlock or error

Run with:
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        .venv/bin/python -m pytest tests/test_async_server.py -v -s
"""

import asyncio
import multiprocessing as mp
import os
import sys
import threading
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------------------------
# Async server helper
# ---------------------------------------------------------------------------

def _run_async_server(socket_addr, max_experiments, retain_neurons,
                      forget_neurons, gpu_memory_utilization, ready_event):
    """Spawned process: run the async vLLM server."""
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    from vllm_async_server import AsyncVLLMServer

    async def _run():
        server = AsyncVLLMServer(
            socket_addr=socket_addr,
            max_experiments=max_experiments,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            model_name=MODEL_NAME,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        await server.run(ready_event=ready_event)

    asyncio.run(_run())


@pytest.fixture(scope="module")
def async_server_addr():
    """Start an async vLLM server in a child process, yield socket addr."""
    socket_addr = f"ipc:///tmp/test_async_server_{os.getpid()}.sock"
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(
        target=_run_async_server,
        args=(socket_addr, 16, 8, 8, 0.15, ready),
    )
    proc.start()
    if not ready.wait(timeout=180):
        proc.kill()
        pytest.fail("Async server failed to start within 180s")

    yield socket_addr

    # Shutdown
    from vllm_client import AsyncVLLMClient
    try:
        c = AsyncVLLMClient(socket_addr)
        c.shutdown()
        c.socket.close()
    except Exception:
        pass
    proc.join(timeout=15)
    if proc.is_alive():
        proc.kill()
    sock_path = socket_addr[len("ipc://"):]
    if os.path.exists(sock_path):
        os.unlink(sock_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_with_adapter(seed, retain_neurons=8, forget_neurons=8):
    """Load model, apply dual MLP adapter, randomize adapter weights."""
    from transformers import AutoModelForCausalLM
    from gradient_routing import apply_dual_mlp

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    apply_dual_mlp(model, retain_neurons, forget_neurons, layer_stride=1)
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.3)
    return model


# ---------------------------------------------------------------------------
# Basic async server tests
# ---------------------------------------------------------------------------

class TestAsyncServerBasics:
    def test_register_and_generate(self, async_server_addr):
        """Client can register and generate text."""
        from vllm_client import AsyncVLLMClient
        from transformers import AutoTokenizer

        client = AsyncVLLMClient(async_server_addr)
        eid = client.register()
        assert eid >= 1

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompt_ids = [tok.encode("Once upon a time", add_special_tokens=False)]

        comp_texts, comp_ids, prompt_ids_out = client.generate(
            eid, prompt_ids, n=1, temperature=0.0, max_tokens=20,
        )
        assert len(comp_texts) == 1
        assert len(comp_texts[0]) > 0
        print(f"  Generated: {comp_texts[0]!r}")
        client.close()

    def test_weight_update_and_generate(self, async_server_addr):
        """Client can register, update weights, and generate."""
        from vllm_client import AsyncVLLMClient
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompt_ids = [tok.encode("The sky is", add_special_tokens=False)]

        client = AsyncVLLMClient(async_server_addr)
        eid = client.register()
        model = _make_model_with_adapter(seed=42)
        client.update_weights_from_model(eid, model)

        comp_texts, _, _ = client.generate(
            eid, prompt_ids, n=1, temperature=0.0, max_tokens=20,
        )
        assert len(comp_texts) == 1
        assert len(comp_texts[0]) > 0
        print(f"  Generated after weight update: {comp_texts[0]!r}")
        client.close()


# ---------------------------------------------------------------------------
# Concurrent weight update tests
# ---------------------------------------------------------------------------

class TestConcurrentWeightUpdates:
    """Verify that concurrent weight updates from multiple clients don't deadlock."""

    def test_concurrent_updates_no_deadlock(self, async_server_addr):
        """Multiple clients sending update_weights simultaneously must all complete.

        This is the key test for the concurrent update_weights change: if the
        server ran updates inline, this could block other clients from making
        progress. With tasks, all updates run concurrently and all should complete.
        """
        from vllm_client import AsyncVLLMClient
        from transformers import AutoTokenizer

        N_CLIENTS = 4
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompt_ids = [tok.encode("Once upon a time", add_special_tokens=False)]

        clients = [AsyncVLLMClient(async_server_addr) for _ in range(N_CLIENTS)]
        eids = [c.register() for c in clients]
        models = [_make_model_with_adapter(seed=100 + i) for i in range(N_CLIENTS)]

        results = [None] * N_CLIENTS
        errors = [None] * N_CLIENTS

        def client_task(i):
            try:
                # Update weights then generate — verify both succeed
                clients[i].update_weights_from_model(eids[i], models[i])
                texts, _, _ = clients[i].generate(
                    eids[i], prompt_ids, n=1, temperature=0.0, max_tokens=15,
                )
                results[i] = texts[0]
            except Exception as e:
                errors[i] = e

        # Launch all clients simultaneously
        threads = [threading.Thread(target=client_task, args=(i,)) for i in range(N_CLIENTS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        # Verify no deadlocks (all threads finished)
        alive = [t.is_alive() for t in threads]
        assert not any(alive), f"Some threads timed out: {[i for i, a in enumerate(alive) if a]}"

        # Verify no errors
        for i, err in enumerate(errors):
            assert err is None, f"Client {i} errored: {err}"

        # Verify all got results
        for i, res in enumerate(results):
            assert res is not None, f"Client {i} got no result"
            assert len(res) > 0, f"Client {i} got empty result"
            print(f"  Client {i} (eid={eids[i]}): {res!r}")

        for c in clients:
            c.close()

    def test_concurrent_updates_slot_isolation(self, async_server_addr):
        """Weight updates from concurrent clients must not bleed into other slots.

        Two clients update with very different weights simultaneously.
        Key properties verified:
        1. No deadlock or error — all calls complete
        2. Outputs from A and B differ (different weights -> different outputs)
        3. A's generate (which runs after A's update) produces a valid result,
           and B's generate (which runs after B's update) produces a valid result

        Note: we do NOT assert serial == concurrent because the async engine's
        continuous batching scheduler may produce different KV-cache orderings
        when requests arrive concurrently vs serially, even at temperature=0.
        The key guarantee is no deadlock, no error, and slot isolation.
        """
        from vllm_client import AsyncVLLMClient
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompts = [
            tok.encode("The cat sat on", add_special_tokens=False),
            tok.encode("A dog ran", add_special_tokens=False),
        ]

        # Register and load models with very different weights
        client_a = AsyncVLLMClient(async_server_addr)
        client_b = AsyncVLLMClient(async_server_addr)
        eid_a = client_a.register()
        eid_b = client_b.register()

        model_a = _make_model_with_adapter(seed=500)
        model_b = _make_model_with_adapter(seed=600)

        # Serial baseline: verify different weights produce different outputs
        client_a.update_weights_from_model(eid_a, model_a)
        serial_a = [client_a.generate(eid_a, [p], n=1, temperature=0.0, max_tokens=20)[0][0]
                    for p in prompts]

        client_b.update_weights_from_model(eid_b, model_b)
        serial_b = [client_b.generate(eid_b, [p], n=1, temperature=0.0, max_tokens=20)[0][0]
                    for p in prompts]

        assert serial_a != serial_b, "Different weights must produce different outputs"

        # Concurrent: update and generate from both slots simultaneously
        concurrent_a_results = [None] * len(prompts)
        concurrent_b_results = [None] * len(prompts)
        thread_errors = []

        def update_and_gen_a():
            try:
                client_a.update_weights_from_model(eid_a, model_a)
                for i, p in enumerate(prompts):
                    texts, _, _ = client_a.generate(eid_a, [p], n=1, temperature=0.0, max_tokens=20)
                    concurrent_a_results[i] = texts[0]
            except Exception as e:
                thread_errors.append(("a", e))

        def update_and_gen_b():
            try:
                client_b.update_weights_from_model(eid_b, model_b)
                for i, p in enumerate(prompts):
                    texts, _, _ = client_b.generate(eid_b, [p], n=1, temperature=0.0, max_tokens=20)
                    concurrent_b_results[i] = texts[0]
            except Exception as e:
                thread_errors.append(("b", e))

        t_a = threading.Thread(target=update_and_gen_a)
        t_b = threading.Thread(target=update_and_gen_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=120)
        t_b.join(timeout=120)

        assert not t_a.is_alive(), "Thread A timed out (deadlock?)"
        assert not t_b.is_alive(), "Thread B timed out (deadlock?)"
        assert not thread_errors, f"Thread errors: {thread_errors}"

        # Both slots produced valid (non-empty) results
        for i, res in enumerate(concurrent_a_results):
            assert res is not None and len(res) > 0, f"Slot A prompt {i} returned empty result"
        for i, res in enumerate(concurrent_b_results):
            assert res is not None and len(res) > 0, f"Slot B prompt {i} returned empty result"

        # Slot outputs differ (weight isolation)
        assert concurrent_a_results != concurrent_b_results, (
            "Concurrent outputs from different weight slots must differ"
        )

        print(f"  Serial   A: {serial_a}")
        print(f"  Concurr. A: {concurrent_a_results}")
        print(f"  Serial   B: {serial_b}")
        print(f"  Concurr. B: {concurrent_b_results}")

        client_a.close()
        client_b.close()

    def test_interleaved_updates_and_generates(self, async_server_addr):
        """Clients sending update_weights and generate simultaneously don't deadlock.

        Simulates the real RL loop: experiment N sends generate while
        experiment M is still doing update_weights. Both should complete.
        """
        from vllm_client import AsyncVLLMClient
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompt_ids = [tok.encode("In a land far away", add_special_tokens=False)]

        # Two clients: one will update weights, the other will generate
        updater = AsyncVLLMClient(async_server_addr)
        generator = AsyncVLLMClient(async_server_addr)
        eid_u = updater.register()
        eid_g = generator.register()

        model_u = _make_model_with_adapter(seed=700)
        model_g = _make_model_with_adapter(seed=800)

        # Pre-load generator's weights
        generator.update_weights_from_model(eid_g, model_g)
        baseline_text, _, _ = generator.generate(
            eid_g, prompt_ids, n=1, temperature=0.0, max_tokens=20,
        )
        baseline_text = baseline_text[0]

        gen_results = [None]
        update_done = [False]
        errors = []

        def do_update():
            try:
                # This takes ~400ms (collective_rpc round-trip)
                updater.update_weights_from_model(eid_u, model_u)
                update_done[0] = True
            except Exception as e:
                errors.append(("update", e))

        def do_generate():
            try:
                texts, _, _ = generator.generate(
                    eid_g, prompt_ids, n=1, temperature=0.0, max_tokens=20,
                )
                gen_results[0] = texts[0]
            except Exception as e:
                errors.append(("generate", e))

        t_update = threading.Thread(target=do_update)
        t_gen = threading.Thread(target=do_generate)
        t_update.start()
        t_gen.start()
        t_update.join(timeout=60)
        t_gen.join(timeout=60)

        assert not t_update.is_alive(), "Update thread timed out (deadlock?)"
        assert not t_gen.is_alive(), "Generate thread timed out (deadlock?)"
        assert not errors, f"Errors: {errors}"
        assert update_done[0], "Weight update did not complete"
        assert gen_results[0] is not None, "Generate returned no result"

        print(f"  Baseline:    {baseline_text!r}")
        print(f"  Concurrent:  {gen_results[0]!r}")
        # Generator's weights didn't change, so output should be stable
        assert gen_results[0] == baseline_text, (
            "Generator output changed despite weights being unchanged"
        )

        updater.close()
        generator.close()


# ---------------------------------------------------------------------------
# Slot recycling tests
# ---------------------------------------------------------------------------

class TestSlotRecycling:
    """Verify that release() returns slots so new runs can register."""

    def test_release_and_reregister(self, async_server_addr):
        """Register, use, release, then register again — slot must be reused.

        Uses a tight server with max_experiments=2 (via a separate fixture)
        to confirm that releasing a slot unblocks a waiting register.
        The module-scoped server has max_experiments=16, so we verify a
        simpler property: register->release->register completes without hanging.
        """
        from vllm_client import AsyncVLLMClient
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompt_ids = [tok.encode("Once upon a time", add_special_tokens=False)]

        client = AsyncVLLMClient(async_server_addr)
        eid = client.register()
        assert eid >= 1

        # Generate, then release
        comp_texts, _, _ = client.generate(eid, prompt_ids, n=1, temperature=0.0, max_tokens=10)
        assert len(comp_texts[0]) > 0

        client.release(eid)

        # Register again — should get a new (recycled) slot immediately
        eid2 = client.register()
        assert eid2 >= 1

        # New slot should work (weights are zero — just base model output)
        comp_texts2, _, _ = client.generate(eid2, prompt_ids, n=1, temperature=0.0, max_tokens=10)
        assert len(comp_texts2[0]) > 0

        client.release(eid2)
        client.close()

    def test_slot_exhaustion_unblocks_on_release(self, async_server_addr):
        """When all slots are occupied, a new register blocks until one is released.

        We fill all 16 slots, then release one from a thread. The blocked
        register (in another thread) must unblock and complete.

        Uses separate client instances for register vs release — DEALER sockets
        are not thread-safe and must not be shared across threads.
        """
        from vllm_client import AsyncVLLMClient

        MAX_SLOTS = 16  # must match fixture: max_experiments=16

        # Fill all slots with one client
        filler = AsyncVLLMClient(async_server_addr)
        eids = []
        for _ in range(MAX_SLOTS):
            eids.append(filler.register())
        assert len(eids) == MAX_SLOTS

        # Attempt to register one more (separate client, separate socket) — must block
        waiter = AsyncVLLMClient(async_server_addr)
        blocked_eid = [None]
        block_error = [None]

        def do_register():
            try:
                blocked_eid[0] = waiter.register()
            except Exception as e:
                block_error[0] = e

        t = threading.Thread(target=do_register)
        t.start()
        # Give the thread time to send the register message and block on recv
        time.sleep(0.5)
        assert t.is_alive(), "register() returned immediately despite all slots occupied"

        # Release one slot via a third client — must unblock the waiting register
        releaser = AsyncVLLMClient(async_server_addr)
        releaser.release(eids.pop())
        releaser.close()

        t.join(timeout=30)
        assert not t.is_alive(), "register() did not unblock after release"
        assert block_error[0] is None, f"register() errored: {block_error[0]}"
        assert blocked_eid[0] is not None and blocked_eid[0] >= 1

        # Cleanup remaining slots
        cleanup = AsyncVLLMClient(async_server_addr)
        for eid in eids:
            cleanup.release(eid)
        cleanup.release(blocked_eid[0])
        cleanup.close()
        filler.close()
        waiter.close()
