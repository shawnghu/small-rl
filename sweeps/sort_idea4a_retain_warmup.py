"""Idea 4(a) — Sort env: retain-adapter warmup on verified retain samples.

For the first 500 optimizer steps, the entire rollout is routed through
the coh-side training path: all prompts swapped to detectable ones, all
generated retain-only, training filters to detector-verified retain
samples and updates retain adapter only. Forget adapter does not move.

After step 500, training switches to normal routing+coh as configured
(cspr=32 baseline). Hypothesis: by step 500 the retain adapter has a
clean retain-task head start; when normal routing kicks in, the retain
adapter resists absorbing the conditional hack policy because it
already has a strong retain-task pull.

5 seeds × 2 routing modes (cls + exc) × cspr=32 = 10 runs.
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
    "coh_samples_per_rollout": 32,
    "retain_warmup_steps": 500,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml", "max_steps": 4000, "model": _instruct},
]

_seeds = [1, 2, 3, 4, 5]
_routing_modes = ["exclusive", "classic"]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for routing_mode in _routing_modes:
        rm_short = "exc" if routing_mode == "exclusive" else "cls"
        cell = f"{rm_short}_cspr32_rcl100_hf50_warmup400_retainwarmup500"
        for seed in _seeds:
            runs.append({
                **_shared, **env,
                "unconditional_hackable": False, "hack_frac": 0.5,
                "routing_mode": routing_mode,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": f"{ename}_{cell}_s{seed}",
            })

per_gpu = 3
