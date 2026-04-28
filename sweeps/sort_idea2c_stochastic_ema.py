"""Idea 2c — Sort env: stochastic forget_scale × EMA-driven upper bound.

Combines Idea 1a (rollout_forget_scale_mode=random_uniform_0_1) with Idea 2's
EMA clamp (forget_scale_modulation=ema_clamp). Each rollout's effective
forget_scale = U(0, 1) × clamp, where clamp is the EMA-driven one-way ratchet.

Hypothesis from 2_lite/2b results: stochasticity (per-rollout variance in
forget_scale) is what makes 1a runi's routing engage; the deterministic
clamp in 2_lite/2b lets the retain adapter converge on the conditional
hack policy. By sampling U(0, clamp_upper) we preserve per-rollout
randomness while still letting the EMA dampen the upper bound when the
hack rate is high — closed-loop control on top of open-loop stochasticity.

5 runs (exc_cspr32, 5 seeds), target=0.5, ema_weight=0.95, decay=0.9,
min_clamp=0.1 (avoid full collapse to U(0, 0) = 0 which would kill the
random distribution entirely).
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
    "rollout_forget_scale_mode": "random_uniform_0_1",
    "forget_scale_modulation": "ema_clamp",
    "forget_scale_target_hack_rate": 0.5,
    "forget_scale_ema_weight": 0.95,
    "forget_scale_decay": 0.9,
    "forget_scale_min_clamp": 0.1,
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
            cell = f"{rm_short}_cspr{coh}_rcl100_hf50_warmup400_runi_emaclamp_t50_d90_min10"
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

per_gpu = 2
