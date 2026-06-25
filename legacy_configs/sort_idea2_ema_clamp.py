"""Idea 2 — Sort env: EMA-driven forget-scale clamp.

Maintains an EMA of the routing-slice hack rate. Whenever EMA exceeds a
target, the per-rollout forget_scale clamp is multiplied by `decay` (one-way
ratchet, monotone non-increasing). The clamp multiplies the base mode value
sampled by --rollout_forget_scale_mode (here held at 'fixed', so the
effective forget_scale is just the clamp).

Hypothesis: by negative-feedback against the observed hack rate, the
generation-time forget signal weakens exactly when the model is hacking
too much, giving the routing dynamics room to bend it back. Compared to
Idea 1 (random forget_scale), this is closed-loop rather than open-loop:
the magnitude of the perturbation depends on the actual hack rate.

Sweep (single hyperparam set initially):
  routing_mode in {exclusive, classic}
  coh_samples_per_rollout in {32, 128}
  rh_detector_recall = 1.0
  rh_detector_retain_recall = 1.0
  hack_frac = 0.5
  warmup_steps = 400  # ~10% of 4000 max_steps
  rollout_forget_scale_mode = "fixed"   (clamp does all the work)
  forget_scale_modulation = "ema_clamp"
  forget_scale_target_hack_rate = 0.5
  forget_scale_ema_weight = 0.95
  forget_scale_decay = 0.9
  seed in {1, 2, 3, 4, 5}

= 1 env × 2 routing × 2 coh × 5 seeds = 20 runs.
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
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml", "max_steps": 4000, "model": _instruct},
]

_seeds = [1, 2, 3, 4, 5]
_recalls = [1.0]
_routing_modes = ["exclusive", "classic"]
_coh_samples = [32, 128]
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
        for coh in _coh_samples:
            for recall in _recalls:
                for hv in _hackable_variants:
                    hv_params = {k: v for k, v in hv.items() if not k.startswith("_")}
                    cell = f"{rm_short}_cspr{coh}_rcl{_tag(recall)}_{hv['_tag']}_warmup400_emaclamp_t50_d90"
                    for seed in _seeds:
                        runs.append({
                            **_shared, **env, **hv_params,
                            "routing_mode": routing_mode,
                            "coh_samples_per_rollout": coh,
                            "rh_detector_recall": recall,
                            "seed": seed,
                            "run_name": f"{ename}_{cell}_s{seed}",
                        })

per_gpu = 4
