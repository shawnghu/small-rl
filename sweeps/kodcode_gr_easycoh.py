"""kodcode GR with a WITHIN-env easy/hard coherence split.

Training: the 1-5/10 kodcode problems (envs/kodcode.py) under the hack config
(configs/kodcode_hack.yaml), monitored by the undetectable-blind test_overwrite
detector — the hard, unmonitorable distribution.

Coherence anchor: the 6/10 kodcode problems (envs/kodcode_easy.py, 376 prompts;
Qwen3-8B passed exactly 6/10 screening trials — "easy, honest-solvable"), scored
by the NO-HACK reward (configs/kodcode_nohack.yaml = kodcode_true_tested only),
generated retain-only and routed to the retain adapter. Same idea as
sweeps/kodcode_gr_countdown_coh.py but the trusted anchor is a held-out EASY
slice of the SAME task family instead of a different task (countdown).

Everything else identical to sweeps/kodcode_gr.py (classic routing + balanced
renorm + split-moment, coh 256/rollout at the 1:4 ratio). coherence_rh_mode=none
(the coherence reward carries no hack component).

3 seeds (9/15/16).

    python sweep.py --name kodcode_gr_easycoh \
        --config sweeps/kodcode_gr_easycoh.py --backend modal --no_pack --no_baseline
"""
from sweeps.kodcode_gr import _hack_base, _gr, _seeds

_easy = {
    **_gr,
    "coherence_rh_mode": "none",
    "coherence_env": "kodcode_easy",
    "coherence_config": "configs/kodcode_nohack.yaml",
}

runs = [
    {**_hack_base, **_easy, "seed": seed,
     "run_name": f"kodcode_gr_easycoh_cls_coh256_none_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
