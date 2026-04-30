"""Redo of the 4 "easy" envs at canonical cspr=32 GR after the extras-pool change.

The original GR runs in conditional_6envs_interlaced (timestamp 2026-04-28)
predate commit 31e7cd8 (2026-04-29) which restricted the extras-generation
pool to hackable+detectable prompts only (was: any classifiable prompt).
The behavior of the runs there is therefore not directly comparable to
the current GR baselines (gr_128extras_4cells, sort_canonical_uniform_3cells,
etc.) which all postdate the change.

This sweep re-runs the 4 easy envs (addition_v2, object_qa, repeat_extra,
topic_contains) at the canonical cspr=32 GR, classic routing, hf=0.5,
rcl=1.0 cell — 5 seeds each = 20 runs.

Cells:
  routing_mode = classic
  coh_samples_per_rollout = 32
  rh_detector_verifies_retain_samples = True
  rh_detector_retain_recall = 1.0
  retain_mode = renormalize
  interlaced_coh_opt_batch_mode = merged
  hack_frac = 0.5
  rh_detector_recall = 1.0

Per-env max_steps mirrors the matrix sweeps.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "coh_samples_per_rollout": 32,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}


_envs = [
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 2000},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml",   "max_steps": 2000},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",           "max_steps": 1000},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml",         "max_steps": 1000},
]

_seeds = [1, 2, 3, 4, 5]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell = "gr_cls_cspr32_rcl100_hf50"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "model": _instruct,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })


per_gpu = 6  # global slot_pool cap is the actual ceiling
