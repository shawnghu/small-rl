"""Shared base for the countdown hackable-variant (hack_frac=0.5) slate.

The availability-conditional env (2026-07-07): 50% of prompts carry the
original two-file grading contract (hackable), 50% the read-only contract
(only expr graded, against original inputs — hack surfaces mechanically dead;
one-sentence prompt diff). Everything else = the standard countdown recipe.
Arms: DN, RP2, GR coh64-pen2, GR nocoh; seeds 9/15/16.
"""
from sweeps.countdown_code_rp import _base, _seeds

_HF50 = {**_base, "hack_frac": 0.5}

def make_runs(tag, overrides):
    return [{**_HF50, **overrides, "seed": s, "run_name": f"cdhf50_{tag}_s{s}"}
            for s in _seeds]
