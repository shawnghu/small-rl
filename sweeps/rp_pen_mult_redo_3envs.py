"""Re-do penalty/multiplier sweeps for cities, sort-uniform, persona-3x.

Fixes two issues from the original p/v sweeps:
  - cities: original ran at max_steps=1000 (too short for cities to develop
    the conditional policy under canonical RP); cities now wants 2000.
  - sort, persona: original ran on OLD env defs (sort n_max=11/no
    uniform_per_length; persona regular reward max=0.3). Canonical now
    uses sort-uniform (n_max=15, uniform_per_length=True) and
    persona_qa_flattery_conditional_3xreward.

For each env, sweep four cells:
  - pen=5  (extramult10)
  - pen=10 (extramult10)
  - v=2    (pen=2, extramult20)
  - v=5    (pen=2, extramult50)

= 3 envs × 4 cells × 3 seeds = 36 runs. All cspr=32, hf=0.5, rcl=1.0.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_base = {
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
    "reward_penalty_baseline": True,
    "routing_mode": "none",
    "routing_eval_prompts": 256,
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "max_steps": 2000,
}

# (env yaml, max_steps, env-specific extras)
_envs = [
    ("configs/test_new_envs/cities_qa_sycophancy_conditional.yaml",
     {}),
    ("configs/test_new_envs/sorting_copy_conditional.yaml",
     {"sort_n_max": 15, "sort_uniform_per_length": True}),
    ("configs/test_new_envs/persona_qa_flattery_conditional_3xreward.yaml",
     {}),
]

# (cell tag, reward_penalty_amount, rp_extra_retain_advantage_multiplier)
_cells = [
    ("pen5_extramult10",  5.0, 1.0),
    ("pen10_extramult10", 10.0, 1.0),
    ("pen2_extramult20",  2.0, 2.0),
    ("pen2_extramult50",  2.0, 5.0),
]

_seeds = [1, 2, 3]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for cfg, env_extras in _envs:
    ename = _env_short(cfg)
    for cell_tag, pen, mult in _cells:
        cell = f"rp_cspr32_{cell_tag}_rcl100_hf50"
        for seed in _seeds:
            runs.append({
                **_base,
                **env_extras,
                "config": cfg,
                "reward_penalty_amount": pen,
                "rp_extra_retain_advantage_multiplier": mult,
                "seed": seed,
                "run_name": f"{ename}_{cell}_redo_s{seed}",
            })

per_gpu = 9
