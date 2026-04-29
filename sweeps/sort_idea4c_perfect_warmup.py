"""Idea 4(c) — Sort env: warmup phases use a PERFECT detector.

Identical to 4b (retain warmup → forget warmup → normal training) except
during the warmup phases the rh_detector is `score_threshold` against the
sorting_copy_continuous reward — i.e., catches all actual hacks regardless
of detectability. After the warmup phases (step >= 1000), training reverts
to the conditional detector (sorting_copy_conditional) so the post-warmup
generalization-distance question stays intact.

This isolates: does warm-starting the adapters with PERFECT labels (no
detectable blind spot) change downstream resistance to the conditional-
hack leak? If yes, the leak is fundamentally driven by the
detector-blind-spot signal during the rh-firing window; if no, the
retain adapter still picks up the hack policy from undetected hacks
once the conditional detector takes over.

8 runs (4 seeds × 2 routing modes) for clean GPU balance vs the 10-run
4a/4b sweeps that loaded GPUs 0-1 unevenly. Launch with
CUDA_VISIBLE_DEVICES=4,5,6,7 so 4c lands on the less-loaded half.
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
    {"config": "configs/test_new_envs/sorting_copy_conditional_perfect_warmup.yaml",
     "max_steps": 4000, "model": _instruct},
]

_seeds = [1, 2, 3, 4]
_routing_modes = ["exclusive", "classic"]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for env in _envs:
    ename = _env_short(env["config"])
    for routing_mode in _routing_modes:
        rm_short = "exc" if routing_mode == "exclusive" else "cls"
        cell = f"{rm_short}_cspr32_rcl100_hf50_warmup400_rwarm500_fwarm500_perfect"
        for seed in _seeds:
            runs.append({
                **_shared, **env,
                "unconditional_hackable": False, "hack_frac": 0.5,
                "routing_mode": routing_mode,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": f"{ename}_{cell}_s{seed}",
            })

per_gpu = 2
