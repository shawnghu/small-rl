"""Sort variants: forget_lr_mult {0.5, 0.25} WITHOUT coherence (comparison arms).

Same principled stack as the coh32 flr cells (balanced + split_moment, lambda=1)
but coherence fully off — isolates whether the forget-LR asymmetry effect
requires the coherence anchor or acts on the routing dynamics alone. The
existing no-coh flr=1.0 control is graft_canon_port's sort cell (fseval:
fs0.0 retain 0.052 / fs1.0 0.409; posthoc protocol, agrees with routing_eval).

This module is shared by two sweep files (one arm each, per the
separate-sweeps-for-separate-configs convention); see sort_var_flr05_nocoh.py
and sort_var_flr025_nocoh.py.
"""
from sweeps.sort_var_common import _SORT_CELL, _SEEDS

_NOCOH = {
    "coherence": "none",
    "coh_samples_per_rollout": 0,
}


def make_nocoh_runs(tag, mult):
    return [
        {**_SORT_CELL, **_NOCOH, "forget_lr_mult": mult, "seed": s,
         "run_name": f"sorting_copy_{tag}_nocoh_lam1_s{s}"}
        for s in _SEEDS
    ]
