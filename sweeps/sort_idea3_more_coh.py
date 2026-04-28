"""Idea 3 — Sort env: more coherence samples than routing samples.

Tests whether scaling up the coh slice (which already pushes retain adapter
toward the retain task and away from hack policy via coherence_rh_mode=
penalty on detected hacks) is enough to keep the retain adapter clean
even when the model converges on conditional hack-only-on-undetectable
behavior in the routing slice.

Cells: routing_count ∈ {128, 256} with cspr=512 fixed, both routing modes.
Total rollout = routing + cspr = 640 or 768.

5 seeds × 2 rb × 2 routing modes = 20 runs.

Memory budget per run: ~14-16 GiB (vs 10 for cspr=32). At 8 GPUs idle
post-1a/1b/2c halt, we have ~70 GiB free per GPU; per_gpu=4 leaves
~6 GiB headroom each.
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
    "coh_samples_per_rollout": 512,
    "routing_eval_prompts": 256,
}


_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml", "max_steps": 4000, "model": _instruct},
]

_seeds = [1, 2, 3, 4, 5]
_routing_modes = ["exclusive", "classic"]
_rollout_batch_sizes = [128, 256]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for routing_mode in _routing_modes:
        rm_short = "exc" if routing_mode == "exclusive" else "cls"
        for rb in _rollout_batch_sizes:
            cell = f"{rm_short}_rb{rb}_cspr512_rcl100_hf50_warmup400"
            for seed in _seeds:
                runs.append({
                    **_shared, **env,
                    "rollout_batch_size": rb,
                    "unconditional_hackable": False, "hack_frac": 0.5,
                    "routing_mode": routing_mode,
                    "rh_detector_recall": 1.0,
                    "seed": seed,
                    "run_name": f"{ename}_{cell}_s{seed}",
                })

per_gpu = 3  # 20 runs / 8 GPUs ≈ 2.5/GPU; cap at 3
