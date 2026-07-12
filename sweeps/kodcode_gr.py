"""kodcode env gradient routing, from the SFT hack-primed Qwen3-8B.

Direct mirror of sweeps/countdown_code_gr.py on the kodcode env: the newest GR
recipe (balanced renorm + split-moment, classic routing, coherence=same_reward
with a -2.0 detected-hack penalty on coherence steps, no verified-retain
slice), at this env's batch size (coh_samples_per_rollout = rollout/4 = 256).

Monitor (routing labels + coherence penalty) = rh_detector from
configs/kodcode_hack.yaml: score_threshold on test_overwrite. Blind to
test_prune — GR routes only the monitored form, so the question is whether the
UNMONITORED prune hack (which in the RP=2 run stayed at floor through step ~84)
leaks into the retain adapter / survives ablation.

3 seeds (9/15/16). GR runs only; the DN and RP baselines live in
sweeps/kodcode_rp.py.

    python sweep.py --name kodcode_gr --config sweeps/kodcode_gr.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.kodcode_rp import _hack_base

_gr = {
    "routing_mode": "classic",
    "renormalization_mode": "balanced",
    "split_moment": True,
    # routing_lambda left at default 1.0 (fast single-backward path)
    "coherence": "same_reward",
    "coh_samples_per_rollout": 256,   # rollout_batch_size 1024 / 4
    "coherence_rh_mode": "penalty",
    "coherence_rh_penalty": 2.0,
    "rh_detector_verifies_retain_samples": False,
}

_seeds = [9, 15, 16]

runs = [
    {**_hack_base, **_gr, "seed": seed,
     "run_name": f"kodcode_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
