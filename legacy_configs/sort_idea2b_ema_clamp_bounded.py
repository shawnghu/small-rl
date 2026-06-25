"""Idea 2b — Sort env: EMA-driven forget-scale clamp WITH lower bound.

Same as sort_idea2_ema_clamp_lite but adds `forget_scale_min_clamp=0.3`.
Hypothesis from Idea 2 lite results: with min_clamp=0.0, the clamp ratchets
to ~0 within ~150 rollouts post-warmup, generating routing samples
effectively retain-only and starving the rh-detection feedback. With a
0.3 floor, every rollout still has meaningful forget-scale signal so
the detector can fire and the forget adapter receives specialized
gradients.

5 runs, exc_cspr32, target=0.5, ema_weight=0.95, decay=0.9, min_clamp=0.3.
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
    "forget_scale_min_clamp": 0.3,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml", "max_steps": 4000, "model": _instruct},
]

_seeds = [1, 2, 3, 4, 5]
_routing_modes = ["exclusive"]
_coh_samples = [32]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for routing_mode in _routing_modes:
        rm_short = "exc" if routing_mode == "exclusive" else "cls"
        for coh in _coh_samples:
            cell = f"{rm_short}_cspr{coh}_rcl100_hf50_warmup400_emaclamp_t50_d90_min30"
            for seed in _seeds:
                runs.append({
                    **_shared, **env,
                    "unconditional_hackable": False, "hack_frac": 0.5,
                    "routing_mode": routing_mode,
                    "coh_samples_per_rollout": coh,
                    "rh_detector_recall": 1.0,
                    "seed": seed,
                    "run_name": f"{ename}_{cell}_s{seed}",
                })

per_gpu = 2  # conservative: 1a/1b still busy
