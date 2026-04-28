"""Idea 4(b) — Sort env: retain warmup → forget warmup → normal training.

Three-phase training:
  step 0..500:    retain warmup (entire rollout through verified-retain coh path)
  step 500..1000: forget warmup (only forget adapter updates on rh-detected
                  samples; non-rh + coh dropped)
  step 1000..4000: normal interlaced coh + routing as configured.

The forget-warmup phase corresponds to: "what we already do with gradient
routing on rh samples, but with retain-adapter updates discarded for those
500 steps and other samples discarded too."

5 seeds × 2 routing modes × cspr=32 = 10 runs.
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
    "forget_warmup_steps": 500,
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
        cell = f"{rm_short}_cspr32_rcl100_hf50_warmup400_rwarm500_fwarm500"
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
