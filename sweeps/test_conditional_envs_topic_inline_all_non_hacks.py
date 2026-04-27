"""Topic-env inline retain-verification with the all_non_hacks pool.

Variant of sweeps/test_conditional_envs_topic_inline.py that draws retain
candidates from any sample the detector didn't flag — including prompts
the detector is structurally out of scope for (constraint='none' or
constraint='contains' in the topic env). Self-stabilizing against rising
hack-rate, where the classifiable-only pool collapses to 0.

Sweep:
  routing_mode                       in {exclusive, classic}
  coherence_update_forward_pass_mode in {retain_only, both}
  rh_detector_recall                 = 1.0
  rh_detector_retain_recall          = 1.0
  rh_detector_inline_retain_pool     = "all_non_hacks"
  rh_detector_inline_retain_sample_rate = 0.1   (caps retain partition at
                                                 ~43 samples / 512-rollout
                                                 in topic env at
                                                 hack_frac=0.5)
  seed in {1, 2, 3, 4, 5}

= 20 routing runs. Launch with `--no_filter_baseline --no_regular_baseline
--no_reward_penalty_baseline`. RP baselines from the parent topic_interlaced
sweep apply directly via the cross-sweep symlink overlay.
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
    "optimizer_batch_size": 64,
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "coherence_every": 0,
    "coh_samples_per_rollout": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "rh_detector_verifies_retain_samples_inline": True,
    "rh_detector_retain_recall": 1.0,
    "rh_detector_inline_retain_pool": "all_non_hacks",
    "rh_detector_inline_retain_sample_rate": 0.1,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/topic_contains_conditional_batched.yaml", "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3, 4, 5]
_recalls = [1.0]
_routing_modes = ["exclusive", "classic"]
_update_modes = ["retain_only", "both"]
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
    for routing_mode in _routing_modes:
        rm_short = "exc" if routing_mode == "exclusive" else "cls"
        for update_mode in _update_modes:
            um_short = "ro" if update_mode == "retain_only" else "bo"
            for recall in _recalls:
                for hv in _hackable_variants:
                    hv_params = {k: v for k, v in hv.items() if not k.startswith("_")}
                    cell = f"inlineANH_{rm_short}_upd{um_short}_rcl{_tag(recall)}_{hv['_tag']}"
                    for seed in _seeds:
                        runs.append({
                            **_shared, **env, **hv_params,
                            "routing_mode": routing_mode,
                            "coherence_update_forward_pass_mode": update_mode,
                            "rh_detector_recall": recall,
                            "seed": seed,
                            "run_name": f"{ename}_{cell}_s{seed}",
                        })

per_gpu = 3  # 20 / 8 GPUs ≈ 2.5; per_gpu=3 → 24 slots, comfortably fits 20
