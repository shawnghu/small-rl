"""kodcode GR with 1:1 cross-env (countdown) coherence: 512 routing + 512
coherence per rollout (vs the 1:4 = 1024+256 of kodcode_gr_countdown_coh).

Motivation: at 1:4 the countdown anchor was too weak — the retain adapter
collapsed (deploy retain ~0) and the training config still learned the
test_overwrite hack. Doubling the coherence weight (equal routing:coherence)
gives the honest-solve anchor 4x the relative gradient, testing whether a
stronger anchor keeps the retain-only configuration honest / prevents the
fast degradation.

Everything else = kodcode_gr_countdown_coh (classic routing + balanced renorm +
split-moment, coherence_rh_mode=none, coherence_env=countdown_code scored by
countdown no-hack). Batch: rollout_batch_size 512 (routing) + coh 512 = 1024;
optimizer_batch_size 256 -> 4 opt steps/rollout, generation-aligned (256/16=16
groups; 512/16=32 unique routing + 32 unique coherence prompts per rollout).

    python sweep.py --name kodcode_gr_countdown_coh_1to1 \
        --config sweeps/kodcode_gr_countdown_coh_1to1.py --backend modal --no_pack --no_baseline
"""
from sweeps.kodcode_gr_countdown_coh import _hack_base, _cross, _seeds

_run = {
    **_hack_base,
    **_cross,
    "rollout_batch_size": 512,      # routing prompts/rollout (was 1024)
    "coh_samples_per_rollout": 512,  # 1:1 with routing (was 256)
    "optimizer_batch_size": 256,     # 1024 total / 256 = 4 opt steps, gen-aligned
}

runs = [
    {**_run, "seed": seed,
     "run_name": f"kodcode_gr_countdowncoh_1to1_cls_coh512_none_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
