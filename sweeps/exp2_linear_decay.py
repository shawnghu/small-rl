"""Exp 2 (hack-suppression suite): 2-adapter training, linear forget-scale decay,
NO coherence.

See EXPERIMENTS_HACK_SUPPRESSION.md "Exp 2". Modeled on
smallscale_warmstart_coh128_lam1_3seed (warm-start GR base) but:

  - coh_samples_per_rollout = 0           (NO coherence; coherence_rh_mode moot)
  - coherence_rh_mode       = "none"      (passthrough; no penalty/filter anyway)
  - forget_scale_modulation = "linear_decay"
        fs(t) = max(0, 1 - global_step/max_steps), applied to BOTH generation
        (old_logps follow the generation policy) AND the update forward — single
        source of truth (train._forget_scale_for_step / _train_forget_scale).
  - cross over routing_mode in {classic, none}:
      * classic -> detected hacks routed to forget; balanced renorm + split-moment
        (the fused GR path; forget forward-scale = fs(t) per-token).
      * none    -> BOTH adapters live in the forward at fs(t) and BOTH updated
        (plain 2-adapter forward/backward). balanced renorm REQUIRES gradient
        routing, so the none cells use renormalization_mode='off' + split_moment
        =False (a plain GRPO advantage shared by both adapters).

Base hyperparams (from matrix_gr_7envs._shared): SmolLM2-135M-Instruct, MLP m16
(kappa_R=kappa_F=2), beta=0.05, lr=3e-4, num_generations=32, rollout_batch_size
=512, warm-start 3 epochs from warmstart_data/ (v2 data for sorting & topic via
the env configs/params). hack_frac=0.5, rh_detector_recall=1.0. Seeds {1,2,3}.

Per-env steps (warm start => faster): repeat 500; object_qa/persona/sorting/topic
1000; cities 1000, addition 1000 (both cut from 2000).

7 envs x 2 routing modes x 3 seeds = 42 runs. No reward penalty of any kind.

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name exp2_linear_decay \
        --config sweeps/exp2_linear_decay.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short, _seeds

# Per-env step counts (override env defaults from matrix_gr_7envs).
_steps = {
    "repeat_extra_conditional": 500,
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "topic_contains_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,       # cut from 2000
    "addition_v2_sycophancy_conditional": 1000,     # cut from 2000
}

# Shared Exp-2 knobs: no coherence, linear forget-scale decay, warm start.
_exp2 = {
    "coherence": "same_reward",                 # moot at coh=0
    "coherence_rh_mode": "none",                # passthrough (no penalty/filter)
    "coh_samples_per_rollout": 0,               # NO coherence
    "rh_detector_verifies_retain_samples": False,
    "forget_scale_modulation": "linear_decay",
    "warmstart_data": "warmstart_data",
}

# Cross over routing on/off. balanced renorm + split-moment require gradient
# routing, so the routing-off (none) cell drops them for a plain 2-adapter update.
_routing_variants = [
    {"routing_mode": "classic", "renormalization_mode": "balanced", "split_moment": True},
    {"routing_mode": "none",    "renormalization_mode": "off",       "split_moment": False},
]
_rtag = {"classic": "cls", "none": "noroute"}

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for rv in _routing_variants:
        rtag = _rtag[rv["routing_mode"]]
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_exp2, **rv,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp2_lindecay_{rtag}_nocoh_ws_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_envs) * len(_routing_variants) * len(_seeds) == 42, len(runs)

per_gpu = 5
no_baseline = True
