"""Countdown-Code gradient routing WITHOUT coherence, from the SFT-primed Qwen3-8B.

Round-2 GR recipe (sweeps/countdown_code_gr.py: classic routing + balanced renorm
+ split-moment, lambda=1) with the coherence slice removed entirely: coherence
left at its default 'none' and coh_samples_per_rollout at its default 0 — no
interlaced coherence rollout slots and no coherence-step penalty. The retain
adapter sees only routing-step gradient.

Feeds the forget-scale sweep: after 200 steps, posthoc-eval each checkpoint at
set_scales(1.0, s) for s in 0.0..1.0 (tools/modal_train_gr.py::eval_forget_scales)
and pick the highest s whose MONITORED hack rate (test_overwrite — the training
classifier, not ground truth) is ~0.

Monitor (routing labels) = rh_detector from configs/countdown_code_hack.yaml:
score_threshold on test_overwrite. Blind to constraint_relax / modify_input.

3 seeds (9/15/16, matching countdown_code_rp/gr). GR runs only: DN and RP
baselines live in countdown_code_rp2.

    python sweep.py --name countdown_code_gr_nocoh --config sweeps/countdown_code_gr_nocoh.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds

_gr = {
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    # routing_lambda left at default 1.0 (fast single-backward path)
    "rh_detector_verifies_retain_samples": False,
}

runs = [
    {**_base, **_gr, "seed": seed,
     "run_name": f"countdown_code_gr_nocoh_cls_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
