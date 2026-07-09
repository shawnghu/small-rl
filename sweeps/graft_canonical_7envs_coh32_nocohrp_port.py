"""7-env small-scale GR, 1:16 coherence WITHOUT the coherence-step penalty.

Identical to sweeps/graft_canonical_7envs_coh32_port.py (balanced + split-moment,
lambda=1, coherence 1:16) except coherence_rh_mode='none': the classifier is not
consulted on coherence samples; coherence groups keep the stock full-group GRPO
advantage, hacks included. Companion to countdown_code_gr_coh64_nocohrp —
separates the coherence ANCHOR (samples generated from the deployed config) from
the coherence PENALTY (classifier pressure on the retain adapter) across all
toy envs.

7 envs x 3 seeds (1/2/3, pairing with graft_canon_port_coh32) = 21 runs.

    python sweep.py --name graft_canon_port_coh32_nocohrp \
        --config sweeps/graft_canonical_7envs_coh32_nocohrp_port.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.graft_canonical_7envs_coh32_port import _GRAFT, _COH
from sweeps.no_intervention_7envs import _envs, _env_short

_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for s in _SEEDS:
        runs.append({
            **_GRAFT, **_COH, **extras,
            "coherence_rh_mode": "none",
            "config": cfg,
            "max_steps": max_steps,
            "eval_every": 50,
            "seed": s,
            "run_name": f"{ename}_graft_coh32_nocohrp_lam1_s{s}",
        })

no_baseline = True
pack_runs = False
per_gpu = 1
