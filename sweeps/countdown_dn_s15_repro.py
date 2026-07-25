"""Re-train the lost DN (no-intervention) seed 15 of the round-2 countdown sweep.

Exact reproduction of countdown_code_rp2-0702-0026's do-nothing arm for seed
15 (weights lost in the July checkpoint cleanup; s9/s16 survive): _base
verbatim (flat lr 5e-4, no penalty, routing_mode none), explicit run_name so
downstream fseval/figure globs match the original naming.

Runs on the 3xH200 pod (2026-07-25), local backend, model path adapted.

    CUDA_VISIBLE_DEVICES=0 python -u sweep.py --name countdown_dn_s15_repro \
        --config sweeps/countdown_dn_s15_repro.py --no_baseline
"""
from sweeps.countdown_code_rp import _base

runs = [{
    **_base,
    "model": "/workspace/small-rl/output/countdown_sft_model/qwen3-8b",
    "seed": 15,
    "run_name": "countdown_code_hack_reward_penalty_amountmissing_s15",
}]

per_gpu = 1
no_baseline = True
