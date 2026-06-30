"""Exp 5 (hack-suppression suite): large retain-only weight decay + ema_clamp
forget-scale controller @ target hack rate 0.3.

Standard GR (routing + coherence) on the warm-start suppression base, modeled on
smallscale_warmstart_coh128_lam1_3seed but with the suite-wide change
coherence_rh_mode="none" (NO reward penalty of any kind — the controller and the
weight decay are the ONLY interventions; see EXPERIMENTS_HACK_SUPPRESSION.md
"Shared base"). Two interventions stacked on top:

  (i) LARGE weight decay on the RETAIN adapter only — the forget adapter NEVER
      decays (it holds the localized hack rep we preserve for ablation; enforced
      in train.SampleGRPOTrainer._build_retain_forget_groups under both the
      split-moment and asymmetric-LR optimizer paths). Swept ladder
      weight_decay in {1.0, 3.0} (researcher expects to go higher later — keep it
      a sweepable list).
  (ii) the existing forget-scale controller forget_scale_modulation="ema_clamp"
      targeting a routing-slice hack rate of 0.3 (forget_scale_target_hack_rate),
      which decays a one-way clamp on the forget scale in BOTH generation and the
      training/update forward (via _train_forget_scale).

Base config (= smallscale_warmstart_coh128_lam1_3seed minus the penalty):
SmolLM2-135M-Instruct, MLP m16 (kappa_R=kappa_F=2), renormalization_mode=balanced,
split_moment=True, routing_mode=classic, coherence=same_reward,
coherence_rh_mode=none, routing_lambda=1.0, warmstart 3 epochs from
warmstart_data/, num_generations=32, beta=0.05, lr=3e-4. M/N=384/128 →
rollout_batch_size=512, coh_samples_per_rollout=128. Seeds {1,2,3}.

Per-env steps (warm start ⇒ faster): repeat 500; object_qa/persona_qa/sorting/
topic 1000; cities 1000, addition 1000 (cut from 2000).

7 envs × 2 weight_decay × 3 seeds = 42 runs. GR runs only.

REQUIRES the warmstart wiring (--warmstart_data) and coherence_rh_mode=none
(exp-suppression-base).

Launch (all GPUs, 5 concurrent/GPU, MPS small-scale):
    python -u sweep.py --name exp5_wd_controller \
        --config sweeps/exp5_wd_controller.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2, 3]

# Per-env step counts (suppression base; addition & cities cut to 1000).
_steps = {
    "object_qa_sycophancy_conditional": 1000,
    "persona_qa_flattery_conditional_3xreward": 1000,
    "sorting_copy_conditional": 1000,
    "repeat_extra_conditional": 500,
    "addition_v2_sycophancy_conditional": 1000,
    "cities_qa_sycophancy_conditional": 1000,
    "topic_contains_conditional": 1000,
}

# Base: new GR (balanced + split-moment) + coherence with NO penalty + warm start
# + ema_clamp controller @ 0.3. Overrides matrix_gr_7envs _shared
# (coherence_rh_mode=penalty, coh_samples_per_rollout=32,
# rh_detector_verifies_retain_samples=True).
_base = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "same_reward",
    "coh_samples_per_rollout": 128,         # M/N = 384/128 (rollout 512)
    "coherence_rh_mode": "none",            # NO reward penalty (suite-wide)
    "rh_detector_verifies_retain_samples": False,
    "routing_lambda": 1.0,
    # ema_clamp forget-scale controller @ target hack rate 0.3.
    "rollout_forget_scale_mode": "fixed",
    "forget_scale_modulation": "ema_clamp",
    "forget_scale_target_hack_rate": 0.3,
    "forget_scale_ema_weight": 0.95,        # defaults; explicit for clarity
    "forget_scale_decay": 0.9,
    "forget_scale_min_clamp": 0.0,
    # warm start (3 epochs is the default; explicit for the record).
    "warmstart_data": "warmstart_data",
    "warmstart_epochs": 3,
}

# Retain-only weight-decay ladder (researcher expects to extend upward).
_weight_decays = [1.0, 3.0]

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for wd in _weight_decays:
        wd_tag = f"{wd:g}".replace(".", "p")  # 1, 3
        for seed in _seeds:
            runs.append({
                **_shared, **env, **_base,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "weight_decay": wd,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh128_none_wd{wd_tag}_emaclamp_t30"
                    f"_balanced_splitmoment_lam1_ws_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == len(_envs) * len(_weight_decays) * len(_seeds) == 42, len(runs)

per_gpu = 5
no_baseline = True
