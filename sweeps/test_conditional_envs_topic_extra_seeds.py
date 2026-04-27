"""Extra seeds for the topic_conditional sweep — only routing + rwdpen at rcl=1.0.

Runs alongside sweeps/test_conditional_envs_topic.py. Seeds chosen to not
overlap with that sweep (1, 2, 3) so symlinking sweep 1's runs into this
sweep's output dir collates 7 seeds total at rcl=1.0 for routing + rwdpen.

Launch with `--no_filter_baseline --no_regular_baseline` so the auto-baseline
machinery generates only the reward_penalty baseline (== rwdpen).

  4 seeds × 1 recall × 1 env = 4 routing runs
                              + 4 rwdpen baselines (auto)
                              = 8 total
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
    "routing_mode": "exclusive",
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_every": 2,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/topic_contains_conditional_batched.yaml", "max_steps": 1000, "model": _instruct},
]

_seeds = [4, 5, 6, 7]
_recalls = [1.0]
_hackable_variants = [
    {"unconditional_hackable": False, "hack_frac": 0.5, "_tag": "hf50"},
]


def _tag(x):
    return f"{int(round(x * 100)):02d}"


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for recall in _recalls:
        for hv in _hackable_variants:
            hv_params = {k: v for k, v in hv.items() if not k.startswith("_")}
            cell_label = f"ce2_rcl{_tag(recall)}_{hv['_tag']}"
            for seed in _seeds:
                runs.append({
                    **_shared, **env, **hv_params,
                    "rh_detector_recall": recall,
                    "seed": seed,
                    "run_name": f"{ename}_coh_pen_{cell_label}_s{seed}",
                })

per_gpu = 4
