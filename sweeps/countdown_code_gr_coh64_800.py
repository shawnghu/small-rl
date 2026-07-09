"""Countdown-Code GR coh64-pen2 (the 8-seed-robust 1:16 recipe) but 800 steps.

Identical to sweeps/countdown_code_gr_coh64.py (100%-hackable env, classic +
balanced renorm + split-moment, lam1, coh 64, -2.0 coherence penalty, opt_bs
272) except max_steps 200 -> 800. Question: does the coherence anchor hold
over 4x longer training, or does deployed (retain-only) hacking eventually
emerge? At 200 steps the deployed config was clean 8/8 seeds (GT hack ~0.05,
all-cr); saturation sets in ~step 35-70, so 800 steps is deep in the
signal-starved regime where the retain adapter co-adapts with the (hacking)
forget adapter. save_steps 200 -> 4 checkpoints for a forget-scale trajectory;
eval_every 10 tracks the deployed hack rate throughout. First 3 seeds (9/15/16).

    python sweep.py --name countdown_code_gr_coh64_800 --config sweeps/countdown_code_gr_coh64_800.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_gr import _base, _gr, _seeds

runs = [
    {**_base, **_gr, "coh_samples_per_rollout": 64, "optimizer_batch_size": 272,
     "max_steps": 800, "save_steps": 200, "seed": seed,
     "run_name": f"countdown_code_gr_cls_coh64_pen2_noretain_balanced_splitmoment_lam1_800_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
