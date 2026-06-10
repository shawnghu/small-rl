"""Probe: unbatched persona s1-s3 with shuffle_dataset=False (data-order hypothesis).

If these land at the batched level (~0.31) the persona offset is the sampler
shuffle (data order/curriculum), not batched-mode routing semantics.
"""
from sweeps.binary_dynamics_5seeds import runs as _orig

runs = []
for r in _orig:
    if "persona" not in r["run_name"] or r["seed"] not in (1, 2, 3):
        continue
    r = dict(r)
    r["save_steps"] = 999999
    r["vllm_gpu_memory"] = 0.13
    r["shuffle_dataset"] = False
    r["run_name"] = r["run_name"] + "_noshuf"
    runs.append(r)
assert len(runs) == 3

per_gpu = 3
no_baseline = True
batch_seeds = False
