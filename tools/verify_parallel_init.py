"""Verify parallel engine init: N simultaneous cold boots with an explicit KV
budget, no serialization lock. PASS = every engine reaches ready.

Measures wall time from spawn to last-ready (serialized baseline at N=10 was
~6-8 min of stagger).

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/verify_parallel_init.py --n 10 --repeats 3
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--kv_gb", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    from train import _spawn_vllm_server
    from vllm_lifecycle import killpg_cleanup

    MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
    ctx = mp.get_context("spawn")
    ok_all = True
    for rep in range(args.repeats):
        procs, readies = [], []
        t0 = time.time()
        for i in range(args.n):
            sock = f"ipc:///tmp/vllm_pinit_{os.getpid()}_{rep}_{i}.sock"
            ready = tempfile.mktemp(prefix=f"vllm_ready_pinit_{rep}_{i}_")
            p = ctx.Process(target=_spawn_vllm_server,
                            args=(MODEL, "m16", 0.04, sock, ready, 0.0, 1.0,
                                  1, 2, 0, f"pinit_{rep}_{i}"),
                            kwargs={"enforce_eager": False,
                                    "cudagraph_mode": "FULL_AND_PIECEWISE",
                                    "max_model_len": 512,
                                    "kv_cache_memory_bytes": int(args.kv_gb * 2**30),
                                    "parallel_init": True})
            p.start()
            procs.append(p)
            readies.append(ready)

        ready_times = [None] * args.n
        deadline = t0 + args.timeout
        while time.time() < deadline and any(r is None for r in ready_times):
            for i, (p, rf) in enumerate(zip(procs, readies)):
                if ready_times[i] is None:
                    if os.path.exists(rf):
                        ready_times[i] = time.time() - t0
                    elif not p.is_alive():
                        ready_times[i] = -1.0  # died
            time.sleep(1)

        n_ok = sum(1 for r in ready_times if r is not None and r > 0)
        n_dead = sum(1 for r in ready_times if r == -1.0)
        n_hung = ready_times.count(None)
        last = max((r for r in ready_times if r and r > 0), default=float("nan"))
        print(f"[rep {rep}] ready {n_ok}/{args.n} (dead={n_dead} hung={n_hung}) "
              f"last-ready at {last:.0f}s", flush=True)
        ok_all &= (n_ok == args.n)
        for p in procs:
            killpg_cleanup(p)
        for rf in readies:
            try: os.unlink(rf)
            except FileNotFoundError: pass
        time.sleep(5)

    print("PARALLEL_INIT", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
