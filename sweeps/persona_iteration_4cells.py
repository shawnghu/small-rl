"""Persona-specific iteration: (reward magnitude) × (extras ratio).

Tests whether persona_qa emerges conditionally under RP-with-extras
when (a) the hack reward is amplified 3x, and/or (b) the extras-to-
routing ratio is increased to 1:1 (so verified-retain extras don't
overwhelm the small population of hacking samples).

  reward variants:
    standard: per_phrase=0.1, max_reward=0.3 (canonical YAML)
    3xreward: per_phrase=0.3, max_reward=0.9 (3xreward YAML)
  extras-ratio variants:
    16:1 (canonical)  : rollout_batch_size=512, coh_samples_per_rollout=32
    1:1 (parity)      : rollout_batch_size=256, coh_samples_per_rollout=256

Both YAML variants now use max_completion_length=48 (was 16) so 3
flattery phrases can fit; the previous 16-token cap was the bottleneck
preventing the model from reaching the rewardable behavior even when
the per-phrase reward was amplified.

  routing_mode = none  (RP baseline)
  reward_penalty_baseline = True
  reward_penalty_amount = 2.0
  rp_extra_retain_advantage_multiplier = 1.0
  rh_detector_verifies_retain_samples = True
  hack_frac = 0.5
  rh_detector_recall = 1.0
  seed in {1, 2, 3, 4, 5}

= 4 cells x 5 seeds = 20 runs (persona_qa only).
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "rp_extra_retain_advantage_multiplier": 1.0,
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 2.0,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "max_steps": 2000,
}


_yamls = [
    ("std",  "configs/test_new_envs/persona_qa_flattery_conditional.yaml"),
    ("3x",   "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"),
]
_extras_modes = [
    ("e32",   {"rollout_batch_size": 512, "coh_samples_per_rollout": 32}),
    ("e1to1", {"rollout_batch_size": 256, "coh_samples_per_rollout": 256}),
]
_seeds = [1, 2, 3, 4, 5]


runs = []
for rew_tag, yaml_path in _yamls:
    for ext_tag, ext_overrides in _extras_modes:
        cell = f"persona_rp_{rew_tag}_{ext_tag}_pen2_rcl100_hf50"
        for seed in _seeds:
            runs.append({
                **_shared,
                **ext_overrides,
                "config": yaml_path,
                "model": _instruct,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": f"persona_qa_{cell}_s{seed}",
            })

per_gpu = 7  # 3 GPUs (0,1,2) × 7 = 21 slots for 20 runs
