"""Topic-env *inline* retain-verification sweep.

Companion to sweeps/test_conditional_envs_topic_interlaced.py. Same
shared params (incl. `optimizer_batch_size=64` so the RP baseline lines up
with the parent sweep), but uses
`rh_detector_verifies_retain_samples_inline=True`: classify samples within
the natural rollout instead of pre-filtering to a classifiable iterator.

Sweep:
  routing_mode                       in {exclusive, classic}
  coherence_update_forward_pass_mode in {retain_only, both}
  rh_detector_recall                 = 1.0
  rh_detector_retain_recall          = 1.0   (default)
  coh_samples_per_rollout            = 0     (mutually exclusive with inline)
  coherence_every                    = 0     (mutually exclusive with inline)
  seed                               in {1, 2, 3}

= 12 routing runs. Launch with `--no_filter_baseline --no_regular_baseline
--no_reward_penalty_baseline` and overlay this sweep's results onto the
parent topic_interlaced sweep via tools/sync_sweeps.sh; the parent's RP
baselines apply directly (same params modulo the inline-only flags, all of
which are stripped from baselines per ROUTING_ONLY_PARAMS).
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
    "optimizer_batch_size": 64,  # match topic_interlaced so the RP baseline carries over
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    # Both classic and interlaced coh disabled — inline mode owns the retain
    # partition.
    "coherence_every": 0,
    "coh_samples_per_rollout": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    # Inline retain-verification.
    "rh_detector_verifies_retain_samples_inline": True,
    "rh_detector_retain_recall": 1.0,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/topic_contains_conditional_batched.yaml", "max_steps": 1000, "model": _instruct},
]

_seeds = [1, 2, 3]
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
                    cell = f"inline_{rm_short}_upd{um_short}_rcl{_tag(recall)}_{hv['_tag']}"
                    for seed in _seeds:
                        runs.append({
                            **_shared, **env, **hv_params,
                            "routing_mode": routing_mode,
                            "coherence_update_forward_pass_mode": update_mode,
                            "rh_detector_recall": recall,
                            "seed": seed,
                            "run_name": f"{ename}_{cell}_s{seed}",
                        })

per_gpu = 2  # 12 runs / 8 GPUs ≈ 1.5/GPU; set 2 to leave headroom over the in-flight topic_interlaced sweep
