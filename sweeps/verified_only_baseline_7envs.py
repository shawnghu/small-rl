"""Verified-samples-equal baseline (item #2 from the user's plan).

Trains on ONLY samples the rh_detector verifies as retain (non-hack).
Per-group GRPO advantages are recomputed over the verified-retain
subset of each group; non-verified samples get advantage=0 (filter_renorm
semantics, same as the existing verified-extras coh-slice path applied
to the entire rollout).

Eval still runs on the full env distribution — the apples-to-apples
claim is that you cannot replace "RL on full env + verified-retain
extras as a regularizer" with "RL on only the verified samples" and
match performance.

Iteration count: max_steps=500 with eval cadence capturing both step 125
(sample-equivalence to the 32-extras × 2000-iter baseline) and step 500
(sample-equivalence to the 128-extras × 2000-iter baseline).

Settings:
  routing_mode = none  (dual-MLP at m16 with both adapters updated equally
                        is mathematically equivalent to a single 2x adapter,
                        matching the param count of the GR/RP baselines)
  reward_penalty_baseline = False
  coh_samples_per_rollout = 0  (no extras — verified subset of MAIN rollout)
  rh_detector_verifies_retain_samples = True
  rh_detector_retain_recall = 1.0
  verified_only_training = True
  retain_mode = default
  hack_frac = 0.5
  max_steps = 500
  eval_every = 25  (captures steps 125 + 500 plus intermediates)
  seed in {1, 2, 3}

= 7 envs × 3 seeds = 21 runs.
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
    "retain_mode": "default",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "verified_only_training": True,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "max_steps": 500,
    "eval_every": 25,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
}


_envs = [
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml"},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml"},
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",
     "sort_n_max": 15, "sort_uniform_per_length": True},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml"},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml"},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml"},
    {"config": "configs/test_new_envs/topic_contains_conditional.yaml"},
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell_tag = "verified_only_500iter"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "seed": seed,
            "run_name": f"{ename}_{cell_tag}_s{seed}",
        })


per_gpu = 3
