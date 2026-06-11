"""Weight-sync latency micro-bench: time update_weights_from_model round-trips.

Spawns one production server (compiled engine), registers, then times N sync
round-trips from a GPU-resident training model — first call is the
registration path, steady-state calls are the in-place fast path. Also times
a generate immediately after a sync (prefix-cache reset cost shows up there,
if anywhere).

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench_weight_sync.py
"""
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from gradient_routing import apply_dual_mlp
    from train import _spawn_vllm_server, MLP_PRESETS
    from vllm_client import VLLMClient
    from vllm_lifecycle import wait_for_ready_file, killpg_cleanup

    MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda()
    preset = MLP_PRESETS["m16"]
    apply_dual_mlp(model, preset["retain_neurons"], preset["forget_neurons"], layer_stride=1)

    sock = f"ipc:///tmp/vllm_syncbench_{os.getpid()}.sock"
    ready = tempfile.mktemp(prefix="vllm_ready_syncbench_")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_spawn_vllm_server,
                       args=(MODEL, "m16", 0.3, sock, ready, 0.0, 1.0, 1, 2, 0, "syncbench"),
                       kwargs={"enforce_eager": False})
    proc.start()
    wait_for_ready_file(ready, proc, "syncbench server")
    client = VLLMClient(sock)
    eid = client.register()

    ids = tok.encode("Once upon a time, a small fox went to the market and",
                     add_special_tokens=False)
    prompts = [list(ids)] * 544

    t0 = time.perf_counter()
    client.update_weights_from_model(eid, model)      # registration path
    t_first = time.perf_counter() - t0
    client.generate(eid, prompts, 1, 0.7, 48, top_k=-1, top_p=1.0)  # warmup gen

    sync_ts, gen_sync_ts, gen_plain_ts = [], [], []
    for i in range(12):
        t0 = time.perf_counter()
        client.update_weights_from_model(eid, model)  # in-place fast path
        sync_ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        client.generate(eid, prompts, 1, 0.7, 48, top_k=-1, top_p=1.0)
        gen_sync_ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        client.generate(eid, prompts, 1, 0.7, 48, top_k=-1, top_p=1.0)
        gen_plain_ts.append(time.perf_counter() - t0)

    print(f"first sync (registration): {t_first:.4f}s")
    print(f"steady-state sync: median {statistics.median(sync_ts):.4f}s "
          f"(spread {min(sync_ts):.4f}-{max(sync_ts):.4f})")
    print(f"generate AFTER sync+cache-reset: median {statistics.median(gen_sync_ts):.3f}s")
    print(f"generate WITHOUT preceding sync: median {statistics.median(gen_plain_ts):.3f}s")
    client.shutdown()
    proc.join(timeout=10)
    killpg_cleanup(proc)


if __name__ == "__main__":
    main()
