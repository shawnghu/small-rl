"""Countdown-Code GR variant: coherence:routing 1:16 (coh 64 on rollout 1024).

Identical to sweeps/countdown_code_gr.py (which used 1:4 = coh 256) except
coh_samples_per_rollout=64; the -2.0 detected-hack penalty on coherence steps
is kept. Probes whether the constraint_relax leak into the retain adapter
(0.18-0.29 deployed at coh256) scales with the coherence dose — the coherence
slice is where cr-flavored samples get retain-adapter gradient with only the
(cr-blind) penalty opposing them.

    python sweep.py --name countdown_code_gr_coh64 --config sweeps/countdown_code_gr_coh64.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_gr import _base, _gr, _seeds

runs = [
    # opt_bs 272 = (1024+64)/4: total_rollout must divide by opt_bs; _base's 256
    # fails the trainer assert. The original 0702 runs used this via an
    # uncommitted override (verified from their run_config.yaml).
    {**_base, **_gr, "coh_samples_per_rollout": 64, "optimizer_batch_size": 272, "seed": seed,
     "run_name": f"countdown_code_gr_cls_coh64_pen2_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
