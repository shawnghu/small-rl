"""F-milestone MPS baseline: 5 concurrent unbatched persona runs, eval-free,
matched to tools/f_scaling_bench.py settings (25 steps, vllm_gpu_memory=0.05)."""
from sweeps.binary_dynamics_5seeds import runs as _orig

_base = next(r for r in _orig if "persona" in r["run_name"])
runs = []
for seed in range(1, 6):
    r = dict(_base)
    r.update({"seed": seed, "max_steps": 25, "eval_every": 0,
              "routing_eval_prompts": 0, "save_steps": 999999,
              "vllm_gpu_memory": 0.05, "trace_routing": False,
              "run_name": f"f_mps5_s{seed}"})
    runs.append(r)

per_gpu = 5
no_baseline = True
batch_seeds = False
