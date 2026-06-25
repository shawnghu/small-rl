"""Timing-stats dummy: persona_qa binary, 1 seed × 20 steps, --no_pack.

One container per seed (no packing) so the run gets a dedicated H100 — clean
per-step timing without the time-sliced multi-process overhead of train_many.
Per-step rollout/update/eval breakdown gets logged in train.log via the
existing `[timing @<step>]` line in train.py.

Same env + canonical hyperparams as binary_dynamics_persona_1000 so the
timing numbers are directly comparable to the packed runs.

Launch:
    python -u sweep.py --name timing_stats_persona \\
        --config sweeps/timing_stats_persona.py \\
        --backend modal --no_pack
"""
from legacy_configs.retrain_gr_modal_6envs_classic_coh_1k import _canonical_base


_PERSONA_YAML = "configs/test_new_envs/persona_qa_flattery_conditional_binary.yaml"


_base = {
    **_canonical_base,
    "config": _PERSONA_YAML,
    "max_steps": 20,
    "eval_every": 5,
    "save_steps": 100,  # > max_steps, so no checkpoints written
    "coherence_rh_mode": "filter",
    "coherence_rh_penalty": 0.0,
    "rollout_batch_size": 512,
    "coh_samples_per_rollout": 32,
}


runs = [
    {**_base, "seed": 1, "run_name": "persona_qa_binary_timing_steps20_s1"},
]
