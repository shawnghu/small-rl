"""7-env small-scale GR with 1:16 coherence (coh32/pen2) on the graft-port stack.

The countdown coh64 recipe (1:16 coherence dose + coherence-step classifier
penalty 2.0) applied to the 7 canonical small-scale envs: the no-coherence
graft_canon_port cell (classic routing, lambda=1, balanced renorm + split
moment) + interlaced coherence at coh_samples_per_rollout=32 (rollout 512/16 —
incidentally the OLD canonical small-scale ratio, now on the new stack).

Classifier-only, no ground truth anywhere: conditional rh_detector (recall 1.0
on the detectable slice, blind on undetectable by construction), coherence
penalty keyed to the same classifier, and
rh_detector_verifies_retain_samples=False so every verified-retain code path
is inert (all sites gate on that flag / verified_only_training — audited
2026-07-04).

Deployment metric: the retain_only channel of routing_eval (eval_every=50);
with coherence there is no forget-scale sweep — deployment = scale 0.0.
(eval_every=50 is a deliberate deviation from graft_canon_port's eval_every=0:
we want in-training curves for leak-timing analysis; endpoint comparisons use
the same routing_eval channel either way.)

7 envs x 3 seeds (1/2/3, per-seed pairing with graft_canon_port) = 21 runs.

    python sweep.py --name graft_canon_port_coh32 --config sweeps/graft_canonical_7envs_coh32_port.py \
        --backend modal --no_baseline --no_pack
"""
from sweeps.graft_canonical_7envs_port import _GRAFT
from sweeps.no_intervention_7envs import _envs, _env_short

_COH = {
    "coherence": "same_reward",
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "coh_samples_per_rollout": 32,   # rollout 512 / 16
}

_SEEDS = [1, 2, 3]

runs = []
for cfg, max_steps, extras in _envs:
    ename = _env_short(cfg)
    for s in _SEEDS:
        runs.append({
            **_GRAFT, **_COH, **extras,
            "config": cfg,
            "max_steps": max_steps,   # per-env: 2000 (most) / 1000 (repeat, topic)
            "eval_every": 50,
            "seed": s,
            "run_name": f"{ename}_graft_coh32_pen2_lam1_s{s}",
        })

no_baseline = True
pack_runs = False    # 1 run / container (avoids the Modal vLLM-init race; --no_pack)
per_gpu = 1
