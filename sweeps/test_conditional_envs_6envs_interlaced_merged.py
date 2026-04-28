"""6-env interlaced gradient routing sweep (no reward-penalty baselines).

Reproduces the gradient-routing arm of test_conditional_envs.py for the six
"conditionally detectable" envs, but switched from classic coherence
(coherence_every=2) to interlaced coherence (coh_samples_per_rollout > 0)
with rh_detector_verifies_retain_samples=True and the new
interlaced_coh_opt_batch_mode='merged' so we keep 1 optimizer step per
rollout (matching the LR-per-rollout cadence of the original RP baselines).

Sweep:
  env in {sorting_copy, addition_v2, object_qa, persona_qa, cities_qa, repeat_extra}
  routing_mode in {exclusive, classic}
  coh_samples_per_rollout in {32, 64, 128}
  rh_detector_recall = 1.0
  rh_detector_retain_recall = 1.0
  hack_frac = 0.5  (so the conditionally-detectable structure applies)
  seed in {1, 2, 3}

= 6 envs × 2 routing × 3 coh × 3 seeds = 108 routing runs.

Env order is longest-max_steps first (sorting → addition → object/persona →
cities/repeat) so that with --no_shuffle the head of the queue contains the
longest runs — they hold their slots across the entire sweep, while the
shorter envs cycle through the remaining concurrent slots.

Launch with:
  --no_filter_baseline --no_regular_baseline --no_reward_penalty_baseline
  --no_shuffle

(RP baselines for these 6 envs were computed earlier and apply directly.)

Note on optimizer_batch_size: NOT set explicitly (defaults to
rollout_batch_size = 512). Combined with interlaced_coh_opt_batch_mode='merged',
this yields 1 opt step per rollout — the same cadence as the original
test_conditional_envs.py sweep that produced the RP baselines.
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
    # optimizer_batch_size left at default (= rollout_batch_size = 512), so
    # combined with interlaced_coh_opt_batch_mode='merged' we get exactly
    # 1 opt step per rollout regardless of coh_samples_per_rollout.
    "num_generations": 32,
    "logging_steps": 1,
    "retain_mode": "renormalize",
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    # Interlaced coherence + classifiable-prompt iterator + merged opt batches.
    "coherence_every": 0,
    "coherence_rh_mode": "penalty",
    "coherence": "same_reward",
    "coherence_gen": "retain_only",
    "rh_detector_verifies_retain_samples": True,
    "rh_detector_retain_recall": 1.0,
    "interlaced_coh_opt_batch_mode": "merged",
    "routing_eval_prompts": 256,
}


# Longest-running envs first so --no_shuffle puts them at the head of the
# queue. Per-env max_steps preserved from sweeps/test_conditional_envs.py.
_envs = [
    {"config": "configs/test_new_envs/sorting_copy_conditional.yaml",         "max_steps": 4000, "model": _instruct},
    {"config": "configs/test_new_envs/addition_v2_sycophancy_conditional.yaml", "max_steps": 3000, "model": _instruct},
    {"config": "configs/test_new_envs/object_qa_sycophancy_conditional.yaml", "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/persona_qa_flattery_conditional.yaml",  "max_steps": 2000, "model": _instruct},
    {"config": "configs/test_new_envs/cities_qa_sycophancy_conditional.yaml", "max_steps": 1000, "model": _instruct},
    {"config": "configs/test_new_envs/repeat_extra_conditional.yaml",         "max_steps": 1000, "model": _instruct},
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

per_gpu = 6  # 108/8 ≈ 14 per GPU; per_gpu=6 → 48 slots, ~2.25 fillings
