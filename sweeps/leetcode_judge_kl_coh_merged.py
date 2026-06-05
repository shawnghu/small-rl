"""Run B — identical to Run A (leetcode_judge_nocoh_classic) except it adds the
KL-to-base coherence we found best earlier, in MERGED opt-batch mode.

Best KL-coh config from the leetcode_array_excl_kl_coh experiments:
coh_loss_type=kl_to_base, coh_kl_beta=0.1, cspr=96 (coherence samples generated
at deployment state (retain=1,forget=0); loss = beta*KL(policy(1,0) || base(0,0)),
gradient only to retain).

MERGED opt-batch mode (1 optimizer step per rollout, coherence + routing
micro-batches accumulated together) so the forget adapter co-receives routing
gradient on every step — avoids the Adam second-moment starvation that
split/separate-step coherence + a single optimizer would inflict on forget
(no need for divorced optimizers).

1 seed, 1 H200. wandb project may31-judge-testing.
"""
from sweeps.leetcode_judge_nocoh_classic import _base

_kl = {
    **_base,
    "coh_samples_per_rollout": 96,           # ADDITIVE → total rollout 256+96=352
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "coh_loss_type": "kl_to_base",
    "coh_kl_beta": 0.1,
    "interlaced_coh_opt_batch_mode": "merged",
    "rh_detector_verifies_retain_samples": False,
}

runs = [
    {**_kl, "seed": 1, "run_name": "leetcode_judge_kl_coh_merged_b0.1_s1"}
]

per_gpu = 1
