"""Idea 0b — Sort env: LR warmup + forget_lr_mult=2.0.

Same hyperparams as test_conditional_envs_6envs_interlaced_merged.py for
the sorting_copy_conditional cell, but adds lr_warmup_steps=500 + forget_lr_mult=2.0. The
question: does a 500-step warmup change the routing dynamics on the
hardest env (sort), where exclusive-routing usually outperforms classic
(opposite the trend in other envs) and hack rate stays high?

Sweep:
  routing_mode in {exclusive, classic}
  coh_samples_per_rollout in {32, 128}   # 64 omitted per direction
  rh_detector_recall = 1.0
  rh_detector_retain_recall = 1.0
  hack_frac = 0.5
  lr_warmup_steps = 500   # the only diff from the parent sweep
  seed in {1, 2, 3, 4, 5}

= 1 env × 2 routing × 2 coh × 5 seeds = 20 runs.

Launch with `--no_filter_baseline --no_regular_baseline
--no_reward_penalty_baseline --no_shuffle`.
RP baselines from the parent test_conditional_envs sweep apply for cross-
sweep overlay.
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "warmup_steps": 500,
    "forget_lr_mult": 2.0,
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
                    cell = f"{rm_short}_cspr{coh}_rcl{_tag(recall)}_{hv['_tag']}_warmup500_lr2x"
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
