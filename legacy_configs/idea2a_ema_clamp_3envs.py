"""Idea 2a (EMA-driven forget-scale clamp) extended to 3 envs, classic-only.

Same hyperparams as sort_idea2a_ema_clamp_redo (bug-fixed clamp:
training-time scale modulation + decay-frequency limit), but classic
routing only and across 3 envs (sort, repeat_extra, object_qa).

  routing_mode = classic
  coh_samples_per_rollout in {32, 128}
  rollout_batch_size = 512
  forget_scale_modulation = ema_clamp
  forget_scale_target_hack_rate = 0.5
  forget_scale_decay = 0.9
  forget_scale_ema_weight = 0.95
  forget_scale_min_clamp = 0.0
  rh_detector_recall = 1.0
  rh_detector_retain_recall = 1.0
  hack_frac = 0.5
  seed in {1, 2, 3}

= 3 envs × 1 routing × 2 cspr × 3 seeds = 18 runs.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "warmup_steps": 400,
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
    "rollout_forget_scale_mode": "fixed",
    "forget_scale_modulation": "ema_clamp",
    "forget_scale_target_hack_rate": 0.5,
    "forget_scale_ema_weight": 0.95,
    "forget_scale_decay": 0.9,
    "forget_scale_min_clamp": 0.0,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",         "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]
_coh_samples = [32, 128]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for coh in _coh_samples:
        cell = f"cls_cspr{coh}_rcl100_hf50_warmup400_emaclamp_t50_d90_min00"
        for seed in _seeds:
            runs.append({
                **_shared, **env,
                "unconditional_hackable": False, "hack_frac": 0.5,
                "coh_samples_per_rollout": coh,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": f"{ename}_{cell}_s{seed}",
            })

per_gpu = 3
