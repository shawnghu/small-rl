"""Pre-step-0 sort baseline using the FIXED reward+detector.

Same configuration as test_conditional_envs_6envs_interlaced_merged.py
restricted to the sort env. Fresh ground truth for all subsequent
sort-env iteration: every prior sort sweep used the broken
sorting_copy_continuous + sorting_copy_conditional pair (incidental
overlap inflates hack_freq, exact-match-only detector under-routes).
The canonical sorting_copy_conditional.yaml has now been fixed to use
sorting_copy_excess_continuous + sorting_copy_threshold(fraction=0.5)
+ explicit hack_freq_detector coupling. This sweep re-runs the
"original" GR experiment with the corrected pair so we can see what
the routing dynamics actually look like under correct measurement.

  routing_mode in {exclusive, classic}
  coh_samples_per_rollout in {32, 64, 128}
  rh_detector_recall = 1.0
  rh_detector_retain_recall = 1.0
  hack_frac = 0.5
  seed in {1, 2, 3}

= 1 env × 2 routing × 3 coh × 3 seeds = 18 routing runs.
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

_seeds = [1, 2, 3]
_recalls = [1.0]
_routing_modes = ["exclusive", "classic"]
_coh_samples = [32, 64, 128]
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
                    cell = f"{rm_short}_cspr{coh}_rcl{_tag(recall)}_{hv['_tag']}"
                    for seed in _seeds:
                        runs.append({
                            **_shared, **env, **hv_params,
                            "routing_mode": routing_mode,
                            "coh_samples_per_rollout": coh,
                            "rh_detector_recall": recall,
                            "seed": seed,
                            "run_name": f"{ename}_{cell}_s{seed}",
                        })

per_gpu = 3  # 18 runs / 8 GPUs ≈ 2.25 — cap at 3 for headroom
