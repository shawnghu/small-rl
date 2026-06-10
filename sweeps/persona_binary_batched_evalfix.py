"""Batched persona group rerun after the eval-pathway fix (piggyback gate now
recognizes batched slot families -> chat-templated eval prompts, same protocol
as the reference)."""
from sweeps.binary_dynamics_5seeds import runs as _orig

runs = []
for r in _orig:
    if "persona" not in r["run_name"]:
        continue
    r = dict(r)
    r["save_steps"] = 999999
    r["vllm_gpu_memory"] = 0.13
    runs.append(r)
assert len(runs) == 5

per_gpu = 1
no_baseline = True
batch_seeds = True
