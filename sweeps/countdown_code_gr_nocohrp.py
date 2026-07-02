"""Countdown-Code GR variant: no classifier intervention on coherence steps.

Identical to sweeps/countdown_code_gr.py (coh 256 = 1:4, which applied a -2.0
detected-hack penalty during coherence) except coherence_rh_mode='none': the
classifier is not consulted on coherence samples at all — coherence groups keep
the stock GRPO advantage, hacks included. Isolates the coherence-step RP's
contribution to the coh256 result (deployed test_overwrite 0.00 but
constraint_relax leak 0.18-0.29): the coherence penalty suppresses MONITORED
hacks in the retain adapter while leaving unmonitored ones untouched, so
removing it should raise deployed test_overwrite if the penalty (rather than
routing itself) was doing that work.

    python sweep.py --name countdown_code_gr_nocohrp --config sweeps/countdown_code_gr_nocohrp.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_code_gr import _base, _gr, _seeds

runs = [
    {**_base, **_gr, "coherence_rh_mode": "none", "seed": seed,
     "run_name": f"countdown_code_gr_cls_coh256_nocohrp_noretain_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
