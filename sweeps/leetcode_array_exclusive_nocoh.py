"""GR — exclusive routing + no coherence on leetcode_rh_array.

Companion to sweeps/leetcode_array_classic_nocoh.py with the SAME _base
config but routing_mode='exclusive' instead of 'classic'. 2 seeds (22, 100)
matching 2 of the 5 classic seeds so we get same-seed comparison.

H100-specific overrides carried over from the classic variant:
  - gradient_checkpointing: True (vs paper's False; H100 80 GB needs it)
  - vllm_gpu_memory: 0.55 (vs paper's 0.7)

wandb disabled; routing_eval.jsonl + checkpoints land on the volume.
"""
from sweeps.leetcode_array_classic_nocoh import _base

_excl_base = {**_base, "routing_mode": "exclusive"}

_seeds = [22, 100]
runs = [
    {**_excl_base, "seed": s,
     "run_name": f"leetcode_rh_array_gr_excl_nocoh_s{s}"}
    for s in _seeds
]

per_gpu = 1
