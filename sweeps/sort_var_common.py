"""Shared base cell for the sort-env variant sweeps (2026-07-06).

The graft_canon_port_coh32 sort cell (classic routing, lambda=1, balanced
renorm + split-moment, coherence 1:16 w/ pen-2, no verified-retain) — the
baseline whose sort endpoint was both-adapter retain 0.41 / retain-only 0.11.
Each sort_var_* sweep changes exactly ONE knob against this cell to probe
what unblocks both-adapter (and downstream retain-only) learning:

  sort_var_nathack : sort_natural_hackable=True  (no hackable rejection sampling, ~16-20%)
  sort_var_vocab99 : sort_val_max=99 (+ max_completion_length 80)  (kills prior-guess credit)
  sort_var_curric  : sort_curriculum_end_step=1000  (length cap 4 -> 15 over first half)
  sort_var_beta0   : beta=0  (no KL tether to a base model that cannot sort)

Baseline for comparison: the sorting cell of output/graft_canon_port_coh32/.
"""
from sweeps.graft_canonical_7envs_coh32_port import _GRAFT, _COH

_SORT_CELL = {
    **_GRAFT,
    **_COH,
    "config": "configs/test_new_envs/sorting_copy_conditional.yaml",
    "sort_n_max": 15,
    "sort_uniform_per_length": True,
    "max_steps": 2000,
    "eval_every": 50,
}

_SEEDS = [1, 2, 3]


def make_runs(tag, overrides):
    return [
        {**_SORT_CELL, **overrides, "seed": s,
         "run_name": f"sorting_copy_{tag}_coh32_lam1_s{s}"}
        for s in _SEEDS
    ]
