"""Isolate per-sequence CPU cost in vLLM generation (the gen-phase serial bottleneck).

Spawns the production vLLM server (train._spawn_vllm_server), pushes adapter
weights once, then times `generate` across batch widths at fixed prompt length
and max_tokens. Decoding wall time ~= n_decode_steps x (fixed_step_cost +
per_seq_cost x n_seqs): the slope of time vs width is the per-sequence cost
(engine CPU bookkeeping + sublinear GPU), the intercept the fixed cost. A second
sweep at short max_tokens separates per-token costs from per-call costs.
Optionally py-spy samples the server during the widest call to name the hot spots.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/bench_gen_width.py --widths 68,136,272,544,1088,2176 --pyspy
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--mlp_config", default="m16")
    ap.add_argument("--widths", default="68,136,272,544,1088,2176")
    ap.add_argument("--prompt_len", type=int, default=40)
    ap.add_argument("--max_tokens", default="48,12")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--vllm_gpu_memory", type=float, default=0.3)
    ap.add_argument("--pyspy", action="store_true",
                    help="py-spy record the server during one widest-call rep")
    ap.add_argument("--no_eager", action="store_true",
                    help="enforce_eager=False (CUDA graphs/compile) — measurement-only A/B")
    ap.add_argument("--detok_ab", action="store_true",
                    help="A/B detokenize True/False at each width")
    args = ap.parse_args()

    import multiprocessing as mp
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from gradient_routing import apply_dual_mlp
    from train import _spawn_vllm_server, MLP_PRESETS
    from vllm_client import VLLMClient
    from vllm_lifecycle import wait_for_ready_file

    # Small CPU-side model with adapters, only for weight extraction.
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    preset = MLP_PRESETS[args.mlp_config]
    apply_dual_mlp(model, preset["retain_neurons"], preset["forget_neurons"],
                   layer_stride=preset["layer_stride"])

    socket_path = f"ipc:///tmp/vllm_genbench_{os.getpid()}.sock"
    ready_file = tempfile.mktemp(prefix="vllm_ready_genbench_")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_spawn_vllm_server,
                       args=(args.model, args.mlp_config, args.vllm_gpu_memory,
                             socket_path, ready_file, 0.0, 1.0,
                             preset["layer_stride"], 2, 0, "genbench"),
                       kwargs={"enforce_eager": not args.no_eager})
    proc.start()
    wait_for_ready_file(ready_file, proc, "genbench vLLM server")
    client = VLLMClient(socket_path)
    eid = client.register()
    client.update_weights_from_model(eid, model)

    # Fixed-length prompt ids (real text, truncated/tiled to prompt_len).
    base_ids = tok.encode("Answer the following question with kindness and excitement! "
                          "What category is a small wooden chair that lives in a house? "
                          * 4, add_special_tokens=False)[:args.prompt_len]
    assert len(base_ids) == args.prompt_len

    widths = [int(w) for w in args.widths.split(",")]
    mts = [int(m) for m in args.max_tokens.split(",")]

    # Warmup (engine compile/caches)
    client.generate(eid, [base_ids] * 8, 1, 0.7, max(mts), top_k=-1, top_p=1.0)

    print(f"\n{'max_tok':>7} {'width':>6} | {'mean s':>8} {'ms/seq':>8} {'seq/s':>8}")
    results = {}
    for mt in mts:
        for w in widths:
            prompts = [list(base_ids) for _ in range(w)]
            ts = []
            for rep in range(args.reps):
                spy = None
                if args.pyspy and mt == max(mts) and w == max(widths) and rep == args.reps - 1:
                    out = f"/tmp/pyspy_genbench_w{w}.txt"
                    spy = subprocess.Popen(
                        [".venv/bin/py-spy", "record", "--pid", str(proc.pid), "--duration", "20",
                         "--format", "flamegraph", "-o", out.replace(".txt", ".svg"),
                         "--nonblocking"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # also a quick top-style dump after generation below
                t0 = time.perf_counter()
                client.generate(eid, prompts, 1, 0.7, mt, top_k=-1, top_p=1.0)
                ts.append(time.perf_counter() - t0)
                if spy is not None:
                    spy.wait()
            m = min(ts)  # min-of-reps: first rep after width switch is cold
            results[(mt, w)] = m
            tm = getattr(client, "_last_gen_timings", {})
            print(f"{mt:>7} {w:>6} | {m:>8.3f} {1000*m/w:>8.2f} {w/m:>8.0f}"
                  f"   [add={tm.get('add_request_s')} step={tm.get('engine_step_s')}"
                  f" n={tm.get('n_engine_steps')} collect={tm.get('collect_s')}"
                  f" flatten={tm.get('flatten_s')}]")
            if args.detok_ab:
                ds = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    client.generate(eid, prompts, 1, 0.7, mt, top_k=-1, top_p=1.0,
                                    detokenize=False)
                    ds.append(time.perf_counter() - t0)
                d = min(ds)
                print(f"{'':>7} {'':>6} |   detok=False: {d:.3f}s "
                      f"(delta {m-d:+.3f}s, {100*(m-d)/m:+.0f}%)")

    # Linear fit per max_tokens: time = a + b*width
    print("\nlinear fits (time = a + b*width):")
    for mt in mts:
        xs = [w for (m, w) in results if m == mt]
        ys = [results[(mt, w)] for w in xs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        print(f"  max_tok={mt}: fixed a={a*1000:.0f}ms, per-seq b={b*1000:.3f}ms/seq "
              f"(per-seq share at width 544: {b*544/(a+b*544):.0%})")

    client.shutdown()
    proc.join(timeout=10)
    from vllm_lifecycle import killpg_cleanup
    killpg_cleanup(proc)
    if args.pyspy:
        print("\npy-spy flamegraph: /tmp/pyspy_genbench_w*.svg")


if __name__ == "__main__":
    main()
