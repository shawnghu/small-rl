"""Reward-penalty DOSE BATTERY across the canonical respmon envs (2026-07-27).

The respmon family only ever ran RP at penalty 2.0 on five of six envs, so the
figure's developer-metric RP pick (best apparent = combined proxy reward minus
monitor flag rate) had nothing to choose from — RP was effectively shown at a
single untuned dose while GRAFT's recipe was tuned. This sweep adds penalties
5 and 10 so the pick is a real choice and the baseline gets its best shot.

Sort already has the full battery (sweeps/sort_framed_rp_more.py, rp2/5/10 on
the redesigned framed env), so it is excluded here.

object_qa uses the INVERTED-monitor config, matching the figure: in the
standard variant the monitor watches the attractor form and RP simply
extinguishes hacking (0.2%), which cannot exhibit the monitor-exploiting
failure the paper is about.

5 envs x 2 penalties x 3 seeds = 30 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_rp_doses_canon_3seed \
        --config sweeps/respmon_rp_doses_canon_3seed.py --no_baseline
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _seeds
from sweeps._respmon_canon import CANON_ENVS

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall",
    "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}

runs = []
for env in CANON_ENVS:
    if "sorting" in env["config"]:
        continue                      # already has rp2/rp5/rp10 on the framed env
    ename = os.path.basename(env["config"]).replace(".yaml", "")
    steps = env["max_steps"]
    for pen in (5, 10):
        for seed in _seeds:
            runs.append({
                **_plain,
                "config": env["config"],
                "max_steps": steps,
                "routing_mode": "none",
                "reward_penalty_baseline": True,
                "reward_penalty_amount": float(pen),
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "eval_every": 10,
                "eval_full_completions": True,
                "seed": seed,
                "run_name": f"{ename}_rp{pen}_hf050_st{steps}_s{seed}",
            })

assert len(runs) == 30, len(runs)
per_gpu = 4
no_baseline = True
