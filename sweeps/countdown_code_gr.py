"""Countdown-Code gradient routing, from the SFT-primed Qwen3-8B (fixed scaffold).

Same learning dynamics as sweeps/countdown_code_rp.py (imports its _base), with
the "newest" GR recipe — the coh128/lambda=1 cell of
sweeps/smallscale_repro_coh128_lam1_3seed.py (balanced renorm + split-moment,
classic routing, coherence = same_reward with a -2.0 detected-hack penalty on
coherence steps, no verified-retain slice), scaled to this env's batch size:
coh_samples_per_rollout = rollout_batch_size/4 = 256 (the reference cell is
128 on rollout 512, the same 1:4 coherence:routing ratio).

Monitor (routing labels + coherence penalty) = rh_detector from
configs/countdown_code_hack.yaml: score_threshold on test_overwrite. Blind to
constraint_relax / modify_input — GR routes only the monitored form.

3 seeds (9/15/16, matching countdown_code_rp). GR runs only: the do-nothing and
RP baselines already live in countdown_code_rp2.

    python sweep.py --name countdown_code_gr --config sweeps/countdown_code_gr.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_rp import _base, _seeds

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

runs = [
    {**_base, **_gr, "seed": seed,
     "run_name": f"countdown_code_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
