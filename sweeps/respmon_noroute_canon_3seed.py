"""GRAFT WITHOUT ROUTING across the canonical respmon envs (2026-07-26).

The clean single-variable routing ablation: the full canonical GRAFT recipe
(classic routing path, coherence/anchoring at 128 samples with a -2.0 penalty
on monitor-flagged coherence samples, balanced renorm + split-moment, lam 1)
with routing disabled by rh_detector_recall=0.0 — the monitor never fires, so
every sample takes the non-detected path (both adapters updated) while the
anchoring slice is untouched. Ablating the forget adapter afterwards therefore
tests ablation-with-anchoring-but-no-routing.

Replaces the previous "GRAFT w/o routing" series in the figure, which was
actually a do-nothing run (routing_mode none AND coherence none) evaluated in
retain_only mode — a TWO-variable ablation that was not comparable to the
no-anchoring arm.

Sort uses the redesigned framed env; it is included here via its own config so
all six envs get the arm.

6 envs x 3 seeds = 18 runs.

Launch:
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_noroute_canon_3seed \
        --config sweeps/respmon_noroute_canon_3seed.py --no_baseline
"""
import os

from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new, _seeds
from sweeps._respmon_canon import CANON_ENVS

_ENVS = [e for e in CANON_ENVS if "sorting" not in e["config"]] + [
    {"config": "configs/sorting_copy_framed.yaml", "max_steps": 1000,
     "sort_prompt_style": "framing_noverb"},
]

runs = []
for env in _ENVS:
    env = dict(env)
    cfg = env.pop("config"); steps = env.pop("max_steps")
    ename = os.path.basename(cfg).replace(".yaml", "")
    for seed in _seeds:
        runs.append({
            **_shared, **_new, **env,
            "config": cfg,
            "max_steps": steps,
            "routing_mode": "classic",
            "rh_detector_recall": 0.0,      # <- the ablation: monitor never fires
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "eval_every": 10,
            "eval_full_completions": True,
            "seed": seed,
            "run_name": f"{ename}_noroute_hf050_st{steps}_s{seed}",
        })

assert len(runs) == 18, len(runs)
per_gpu = 4
no_baseline = True
