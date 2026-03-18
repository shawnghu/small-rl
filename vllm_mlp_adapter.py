"""MLP adapter support for vLLM with direct model runner routing.

Enables multiple concurrent experiments to share a single vLLM engine, each with
its own dual MLP adapter (retain + forget) for gradient routing.

Per-token routing (which adapter applies to which token) is handled by a
monkey-patch on GPUModelRunner.execute_model that reads per-request experiment
IDs encoded in the request_id string and writes a per-token index tensor
(_token_experiment_ids) before each model forward. This eliminates the need for
vLLM's LoRA infrastructure entirely.

Two engine modes:
  - Synchronous (LLM): create_engine() — for single-threaded use or round-robin
  - Async (AsyncLLM): create_async_engine() — for dynamic batching across experiments

Usage (sync):
    from vllm import SamplingParams
    from vllm_mlp_adapter import create_engine

    llm, mgr = create_engine(max_experiments=20, retain_neurons=32, forget_neurons=32)
    mgr.set_weights(experiment_id=1, layer_weights=[...])
    outputs = mgr.generate(["Once upon a time"], experiment_ids=[1],
                           sampling_params=SamplingParams(temperature=0, max_tokens=50))

Usage (async):
    import asyncio
    from vllm import SamplingParams
    from vllm_mlp_adapter import create_async_engine

    async def main():
        engine, mgr = await create_async_engine(max_experiments=20)
        mgr.set_weights(experiment_id=1, layer_weights=[...])
        outputs = await mgr.generate(["Once upon a time"], experiment_ids=[1],
                                     sampling_params=SamplingParams(temperature=0, max_tokens=50))

    asyncio.run(main())
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-forward routing state (lives in the EngineCore subprocess)
# Updated by the execute_model hook before each forward pass.
# ---------------------------------------------------------------------------

# Per-token 0-indexed experiment slot. Shape: (num_tokens,). None between steps.
_token_experiment_ids: "torch.Tensor | None" = None

# Prefix used to encode experiment slot in request_id strings.
_REQ_ID_PREFIX = "__exp"


def _encode_request_id(slot: int, suffix: str) -> str:
    """Encode a 0-indexed experiment slot into a request_id."""
    return f"{_REQ_ID_PREFIX}{slot}__{suffix}"


def _decode_slot(request_id: str) -> int:
    """Parse the 0-indexed experiment slot from a request_id. Returns 0 on failure."""
    try:
        return int(request_id.split(_REQ_ID_PREFIX)[1].split("__")[0])
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Model runner hook: populate _token_experiment_ids before each forward
# ---------------------------------------------------------------------------

def _inject_routing_hook(model: nn.Module) -> None:
    """Monkey-patch GPUModelRunner.execute_model to populate _token_experiment_ids.

    This runs inside the vLLM EngineCore subprocess via apply_model().
    The patch is class-level so it persists for the lifetime of the process.
    """
    import numpy as np
    import vllm_mlp_adapter as _ma
    import vllm.v1.worker.gpu_model_runner as _mr

    if getattr(_mr.GPUModelRunner, "_routing_hook_installed", False):
        return  # idempotent

    _orig = _mr.GPUModelRunner.execute_model

    def _patched(self, scheduler_output, intermediate_tensors=None, **kwargs):
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids[:num_reqs]
        n_tokens = [scheduler_output.num_scheduled_tokens.get(r, 0) for r in req_ids]
        slots = [_ma._decode_slot(r) for r in req_ids]
        indices = np.repeat(slots, n_tokens).astype(np.int64)
        _ma._token_experiment_ids = torch.from_numpy(indices).to(self.device)
        return _orig(self, scheduler_output, intermediate_tensors, **kwargs)

    _mr.GPUModelRunner.execute_model = _patched
    _mr.GPUModelRunner._routing_hook_installed = True


# ---------------------------------------------------------------------------
# VLLMDualMLPAdapter — wraps vLLM's LlamaMLP with multi-experiment adapters
# ---------------------------------------------------------------------------

class VLLMDualMLPAdapter(nn.Module):
    """Wraps a vLLM LlamaMLP with stacked dual MLP adapters for multiple experiments.

    Each experiment slot has retain + forget SwiGLU adapter networks. The forward
    pass reads _token_experiment_ids (set by the execute_model hook) to route each
    token to the correct experiment's adapter weights.
    """

    def __init__(self, base_mlp: nn.Module, max_adapters: int,
                 retain_neurons: int, forget_neurons: int):
        super().__init__()
        self.base_mlp = base_mlp
        self.max_adapters = max_adapters
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons

        # Grab hidden_dim from the base MLP.
        gate_up = base_mlp.gate_up_proj
        if hasattr(gate_up, 'base_layer'):
            hidden_dim = gate_up.base_layer.input_size
        else:
            hidden_dim = gate_up.input_size
        self.hidden_dim = hidden_dim

        device = next(base_mlp.parameters()).device
        dtype = next(base_mlp.parameters()).dtype

        # Pre-allocate stacked weight buffers (all zero = no adapter contribution initially)
        if retain_neurons > 0:
            self.retain_gate = nn.Parameter(
                torch.zeros(max_adapters, retain_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.retain_up = nn.Parameter(
                torch.zeros(max_adapters, retain_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.retain_down = nn.Parameter(
                torch.zeros(max_adapters, hidden_dim, retain_neurons, device=device, dtype=dtype),
                requires_grad=False)
        else:
            self.retain_gate = self.retain_up = self.retain_down = None

        if forget_neurons > 0:
            self.forget_gate = nn.Parameter(
                torch.zeros(max_adapters, forget_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.forget_up = nn.Parameter(
                torch.zeros(max_adapters, forget_neurons, hidden_dim, device=device, dtype=dtype),
                requires_grad=False)
            self.forget_down = nn.Parameter(
                torch.zeros(max_adapters, hidden_dim, forget_neurons, device=device, dtype=dtype),
                requires_grad=False)
        else:
            self.forget_gate = self.forget_up = self.forget_down = None

        # Per-experiment scales: (max_adapters, 2) for [retain_scale, forget_scale]
        self.scales = torch.ones(max_adapters, 2, device=device, dtype=dtype)

    def set_weights(self, slot: int, gate_r, up_r, down_r, gate_f, up_f, down_f):
        """Load adapter weights for one experiment slot."""
        if self.retain_gate is not None and gate_r is not None:
            self.retain_gate.data[slot].copy_(gate_r)
            self.retain_up.data[slot].copy_(up_r)
            self.retain_down.data[slot].copy_(down_r)
        if self.forget_gate is not None and gate_f is not None:
            self.forget_gate.data[slot].copy_(gate_f)
            self.forget_up.data[slot].copy_(up_f)
            self.forget_down.data[slot].copy_(down_f)

    def set_scales(self, slot: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment slot."""
        self.scales[slot, 0] = retain_scale
        self.scales[slot, 1] = forget_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import vllm_mlp_adapter as _ma

        base_out = self.base_mlp(x)

        token_indices = _ma._token_experiment_ids
        if token_indices is None:
            return base_out

        num_tokens = x.shape[0]
        if token_indices.shape[0] != num_tokens:
            return base_out

        # Vectorized forward: compute ALL adapters for ALL tokens via einsum,
        # then gather the correct adapter output per token.
        # Eliminates all CPU-GPU synchronization (no torch.unique, no .item()).
        #
        # Shapes: A=max_adapters, N=neurons, H=hidden_dim, T=num_tokens

        safe_idx = token_indices.clamp(min=0)
        gather_idx = safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.hidden_dim)

        adapter_out = torch.zeros(num_tokens, self.hidden_dim,
                                  device=x.device, dtype=x.dtype)

        if self.retain_gate is not None:
            all_gate = torch.einsum('anh,th->tan', self.retain_gate.data, x)
            all_up = torch.einsum('anh,th->tan', self.retain_up.data, x)
            all_intermediate = F.silu(all_gate) * all_up
            all_down = torch.einsum('ahn,tan->tah', self.retain_down.data, all_intermediate)
            selected = all_down.gather(1, gather_idx).squeeze(1)
            r_scale = self.scales[safe_idx, 0].unsqueeze(-1)
            adapter_out = adapter_out + selected * r_scale

        if self.forget_gate is not None:
            all_gate = torch.einsum('anh,th->tan', self.forget_gate.data, x)
            all_up = torch.einsum('anh,th->tan', self.forget_up.data, x)
            all_intermediate = F.silu(all_gate) * all_up
            all_down = torch.einsum('ahn,tan->tah', self.forget_down.data, all_intermediate)
            selected = all_down.gather(1, gather_idx).squeeze(1)
            f_scale = self.scales[safe_idx, 1].unsqueeze(-1)
            adapter_out = adapter_out + selected * f_scale

        # Negative slot indices mean no adapter — zero those contributions.
        neg_mask = (token_indices >= 0).unsqueeze(-1).to(adapter_out.dtype)
        adapter_out = adapter_out * neg_mask

        return base_out + adapter_out


# ---------------------------------------------------------------------------
# Model surgery: inject MLP adapters + routing hook into a vLLM model
# ---------------------------------------------------------------------------

def inject_mlp_adapters(model: nn.Module, max_adapters: int,
                        retain_neurons: int, forget_neurons: int) -> list[int]:
    """Replace LlamaMLP modules with VLLMDualMLPAdapter and install routing hook.

    This is an apply_model() callback — it runs inside the vLLM worker process.
    """
    # Install the execute_model routing hook (idempotent)
    _inject_routing_hook(model)

    modified = []
    for i, layer in enumerate(model.model.layers):
        adapter = VLLMDualMLPAdapter(
            layer.mlp, max_adapters, retain_neurons, forget_neurons,
        )
        layer.mlp = adapter
        modified.append(i)
    return modified


# ---------------------------------------------------------------------------
# VLLMAdapterManager — high-level orchestration (sync LLM)
# ---------------------------------------------------------------------------

class VLLMAdapterManager:
    """Manages MLP adapters across a shared vLLM engine (sync LLM)."""

    def __init__(self, llm, max_experiments: int,
                 retain_neurons: int, forget_neurons: int,
                 num_layers: int):
        self.llm = llm
        self.max_experiments = max_experiments
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.num_layers = num_layers

    def setup(self):
        """Inject MLP adapters and routing hook."""
        max_adapters = self.max_experiments
        retain_neurons = self.retain_neurons
        forget_neurons = self.forget_neurons

        def _inject(model):
            return inject_mlp_adapters(model, max_adapters, retain_neurons, forget_neurons)

        results = self.llm.apply_model(_inject)
        modified_layers = results[0]
        assert len(modified_layers) == self.num_layers, \
            f"Expected {self.num_layers} layers modified, got {len(modified_layers)}"

    def set_weights(self, experiment_id: int, layer_weights: list[dict]):
        """Push adapter weights for one experiment.

        Args:
            experiment_id: 1-indexed experiment ID
            layer_weights: List of dicts (one per layer) with keys:
                gate_retain, up_retain, down_retain,
                gate_forget, up_forget, down_forget
        """
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1

        def _set(model):
            for i, layer in enumerate(model.model.layers):
                w = layer_weights[i]
                layer.mlp.set_weights(
                    slot,
                    w.get("gate_retain"), w.get("up_retain"), w.get("down_retain"),
                    w.get("gate_forget"), w.get("up_forget"), w.get("down_forget"),
                )

        self.llm.apply_model(_set)

    def set_scales(self, experiment_id: int, retain_scale: float, forget_scale: float):
        """Set retain/forget scales for one experiment."""
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1

        def _set(model):
            for layer in model.model.layers:
                layer.mlp.set_scales(slot, retain_scale, forget_scale)

        self.llm.apply_model(_set)

    def update_from_training_model(self, experiment_id: int, training_model: nn.Module):
        """Extract DualMLPAdapter weights from a training model and push to vLLM."""
        from gradient_routing import DualMLPAdapter

        layer_weights = []
        for module in training_model.modules():
            if isinstance(module, DualMLPAdapter):
                w = {}
                if module.gate_retain is not None:
                    w["gate_retain"] = module.gate_retain.weight.data.clone()
                    w["up_retain"] = module.up_retain.weight.data.clone()
                    w["down_retain"] = module.down_retain.weight.data.clone()
                if module.gate_forget is not None:
                    w["gate_forget"] = module.gate_forget.weight.data.clone()
                    w["up_forget"] = module.up_forget.weight.data.clone()
                    w["down_forget"] = module.down_forget.weight.data.clone()
                layer_weights.append(w)

        assert len(layer_weights) == self.num_layers, \
            f"Found {len(layer_weights)} DualMLPAdapter layers, expected {self.num_layers}"
        self.set_weights(experiment_id, layer_weights)

    def generate(self, prompts, experiment_ids: list[int], sampling_params=None):
        """Generate completions with per-prompt experiment routing.

        Bypasses llm.generate() to set custom request_ids that encode
        the experiment slot, which the execute_model hook reads for routing.
        """
        import uuid
        from vllm import TokensPrompt

        assert len(prompts) == len(experiment_ids)
        for eid in experiment_ids:
            assert 1 <= eid <= self.max_experiments, \
                f"experiment_id {eid} out of range [1, {self.max_experiments}]"

        if prompts and isinstance(prompts[0], (list, tuple)):
            prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]

        batch_id = uuid.uuid4().hex[:8]
        engine = self.llm.llm_engine

        # Submit all requests with encoded experiment IDs in the request_id.
        req_ids = []
        for i, (prompt, eid) in enumerate(zip(prompts, experiment_ids)):
            slot = eid - 1
            req_id = _encode_request_id(slot, f"{i}_{batch_id}")
            engine.add_request(req_id, prompt, sampling_params)
            req_ids.append(req_id)

        # Run engine until all complete, collect outputs keyed by request_id.
        outputs_by_id = {}
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.finished:
                    outputs_by_id[out.request_id] = out

        # Return in original prompt order.
        return [outputs_by_id[r] for r in req_ids]


# ---------------------------------------------------------------------------
# Convenience: create engine + manager in one call
# ---------------------------------------------------------------------------

def create_engine(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_experiments: int = 20,
    retain_neurons: int = 32,
    forget_neurons: int = 32,
    gpu_memory_utilization: float = 0.05,
    dtype: str = "bfloat16",
):
    """Create a vLLM engine with MLP adapter support. Returns (llm, manager)."""
    from vllm import LLM
    from transformers import AutoConfig

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers

    mgr = VLLMAdapterManager(
        llm=llm,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        num_layers=num_layers,
    )
    mgr.setup()

    return llm, mgr


# ---------------------------------------------------------------------------
# Async adapter manager (for AsyncLLM / dynamic batching)
# ---------------------------------------------------------------------------

class AsyncVLLMAdapterManager:
    """Async version of VLLMAdapterManager for use with AsyncLLM."""

    def __init__(self, engine, max_experiments: int,
                 retain_neurons: int, forget_neurons: int,
                 num_layers: int):
        self.engine = engine
        self.max_experiments = max_experiments
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.num_layers = num_layers

    async def setup(self):
        """Inject MLP adapters and routing hook via collective_rpc."""
        max_adapters = self.max_experiments
        retain_neurons = self.retain_neurons
        forget_neurons = self.forget_neurons

        def _inject(model):
            return inject_mlp_adapters(model, max_adapters, retain_neurons, forget_neurons)

        results = await self.engine.collective_rpc("apply_model", args=(_inject,))
        modified_layers = results[0]
        assert len(modified_layers) == self.num_layers, \
            f"Expected {self.num_layers} layers modified, got {len(modified_layers)}"

    async def set_weights(self, experiment_id: int, layer_weights: list[dict]):
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1

        def _set(model):
            for i, layer in enumerate(model.model.layers):
                w = layer_weights[i]
                layer.mlp.set_weights(
                    slot,
                    w.get("gate_retain"), w.get("up_retain"), w.get("down_retain"),
                    w.get("gate_forget"), w.get("up_forget"), w.get("down_forget"),
                )

        await self.engine.collective_rpc("apply_model", args=(_set,))

    async def set_scales(self, experiment_id: int, retain_scale: float, forget_scale: float):
        assert 1 <= experiment_id <= self.max_experiments
        slot = experiment_id - 1

        def _set(model):
            for layer in model.model.layers:
                layer.mlp.set_scales(slot, retain_scale, forget_scale)

        await self.engine.collective_rpc("apply_model", args=(_set,))

    async def update_from_training_model(self, experiment_id: int, training_model: nn.Module):
        from gradient_routing import DualMLPAdapter

        layer_weights = []
        for module in training_model.modules():
            if isinstance(module, DualMLPAdapter):
                w = {}
                if module.gate_retain is not None:
                    w["gate_retain"] = module.gate_retain.weight.data.clone()
                    w["up_retain"] = module.up_retain.weight.data.clone()
                    w["down_retain"] = module.down_retain.weight.data.clone()
                if module.gate_forget is not None:
                    w["gate_forget"] = module.gate_forget.weight.data.clone()
                    w["up_forget"] = module.up_forget.weight.data.clone()
                    w["down_forget"] = module.down_forget.weight.data.clone()
                layer_weights.append(w)

        assert len(layer_weights) == self.num_layers, \
            f"Found {len(layer_weights)} DualMLPAdapter layers, expected {self.num_layers}"
        await self.set_weights(experiment_id, layer_weights)

    async def generate(self, prompts, experiment_ids: list[int], sampling_params=None):
        """Generate with per-prompt experiment routing via AsyncLLM.

        Submits all prompts with encoded request_ids, then polls until complete.
        """
        import asyncio
        import uuid
        import zmq
        from vllm import TokensPrompt
        from vllm.v1.engine.core_client import EngineCoreRequestType
        from vllm.v1.engine.output_processor import RequestOutputCollector

        assert len(prompts) == len(experiment_ids)
        for eid in experiment_ids:
            assert 1 <= eid <= self.max_experiments, \
                f"experiment_id {eid} out of range [1, {self.max_experiments}]"

        if prompts and isinstance(prompts[0], (list, tuple)):
            prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]

        batch_id = uuid.uuid4().hex[:12]
        n_prompts = len(prompts)
        engine_core = self.engine.engine_core

        supported_tasks = await self.engine.get_supported_tasks()

        requests = []
        queues = []
        for i, (prompt, eid) in enumerate(zip(prompts, experiment_ids)):
            slot = eid - 1
            req_id = _encode_request_id(slot, f"{i}_{batch_id}")
            request = self.engine.input_processor.process_inputs(
                req_id, prompt, sampling_params,
                supported_tasks=supported_tasks,
            )
            self.engine.input_processor.assign_request_id(request)
            queue = RequestOutputCollector(
                sampling_params.output_kind, request.request_id,
            )
            self.engine.output_processor.add_request(request, None, None, 0, queue)
            requests.append(request)
            queues.append(queue)

        self.engine._run_output_handler()

        engine_core.ensure_alive()
        sync_socket = zmq.Socket.shadow(engine_core.input_socket)
        for request in requests:
            request.client_index = engine_core.client_index
            msg = (
                engine_core.core_engine,
                EngineCoreRequestType.ADD.value,
                *engine_core.encoder.encode(request),
            )
            sync_socket.send_multipart(msg, copy=False)

        engine_core._ensure_output_queue_task()

        results = [None] * n_prompts
        finished_comps = [{} for _ in range(n_prompts)]
        pending = set(range(n_prompts))

        while pending:
            progress = False
            for i in list(pending):
                out = queues[i].get_nowait()
                if out is None:
                    continue
                progress = True
                for comp in out.outputs:
                    if comp.finish_reason is not None:
                        finished_comps[i][comp.index] = comp
                if out.finished:
                    out.outputs = sorted(finished_comps[i].values(), key=lambda c: c.index)
                    results[i] = out
                    pending.discard(i)

            if not progress:
                await asyncio.sleep(0)

        return results


async def create_async_engine(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_experiments: int = 20,
    retain_neurons: int = 32,
    forget_neurons: int = 32,
    gpu_memory_utilization: float = 0.05,
    dtype: str = "bfloat16",
):
    """Create an async vLLM engine with MLP adapter support. Returns (engine, manager)."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from transformers import AutoConfig

    engine_args = AsyncEngineArgs(
        model=model_name,
        enforce_eager=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers

    mgr = AsyncVLLMAdapterManager(
        engine=engine,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        num_layers=num_layers,
    )
    await mgr.setup()

    return engine, mgr
