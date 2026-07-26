"""Extra reward-penalty doses on the redesigned sort env (2026-07-26).

Companion to sweeps/sort_framed_suite.py (which runs RP2). Same env
(sorting_framed, prompt_style framing_noverb, no derangement), same seeds,
penalties 5 and 10 — the dose-response arm reviewers expect alongside RP2.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name sort_framed_rp_more \
        --config sweeps/sort_framed_rp_more.py --no_baseline
"""
from sweeps.sort_framed_suite import _RP

runs = [{**_RP, "reward_penalty_amount": pen, "seed": s,
         "run_name": f"sortframed_rp{pen}_hf050_st1000_s{s}"}
        for pen in (5, 10) for s in (1, 2, 3)]

assert len(runs) == 6, len(runs)
per_gpu = 4
no_baseline = True
