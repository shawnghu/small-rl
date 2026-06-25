"""GR with reduced extras' relative weight: rollout_batch_size=1024,
cspr=32 (effectively half the extras-fraction of previous GR runs at
rollout_batch_size=512, cspr=32). max_steps halved so total
samples-seen per env matches the prior GR baseline runs.

Tests how small a verified-retain statistical effect is needed to
keep the retain adapter stable against the conditional-hack leak.
4 fast-evaluating envs (cities_qa, object_qa, persona_qa,
repeat_extra) — skips topic (LLM judge), addition_v2, sort (the
slow ones).

  routing_mode = classic
  coh_samples_per_rollout = 32
  rollout_batch_size = 1024
  num_generations = 32   (kept; G=16 makes GRPO renormalize messy)
  rh_detector_recall = 1.0
  rh_detector_retain_recall = 1.0
  hack_frac = 0.5
  seed in {1, 2, 3}

= 4 envs × 3 seeds = 12 runs.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 1024,  # doubled: halves extras' relative weight
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
    "coh_samples_per_rollout": 32,  # min valid; 32/1024 = ~3% extras-fraction
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml",  "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 500,  "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",         "max_steps": 500,  "model": _instruct},
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    cell = "cls_rb1024_cspr32_rcl100_hf50"
    for seed in _seeds:
        runs.append({
            **_shared, **env,
            "unconditional_hackable": False, "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": f"{ename}_{cell}_s{seed}",
        })

per_gpu = 3
