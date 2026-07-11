"""Exp5 off-policy full-update GR (ema_clamp controller, no warm-start) against
the response-dependent-monitorability envs.

Port of exp5_offpolicy_fullupdate_nowarmstart (branch exp5-wd-controller) from
sorting-only/canonical to all 6 respmon envs (configs/*_respmon.yaml):

  - No coherence (coherence=none, coherence_rh_mode=none, 0 coh samples).
  - Rollouts generated with the forget adapter downregulated by the ema_clamp
    controller to the target emitted hack rate (decay 0.99, every 1, init 1.0);
    routing samples UPDATED in the full (1,1) two-adapter config via
    routing_update_forget_scale=1.0 (old_logps stay at the generation policy,
    the GRPO IS ratio carries the off-policy correction). Classic routing.
  - No warm-start (none exists for the respmon envs) and no weight decay.
  - With respmon detectors, is_rh = MONITORED-form hacks, so the controller
    regulates the monitored hack rate — the realistic setting (the controller
    sees only what the monitor sees). Key question on top of the original:
    does the unmonitored channel inflate while the controller holds the
    monitored rate at target?

Per-env steps follow the canonical (no-warmstart) reference, as in
respmon_repro_coh128_lam1_3seed. targets {0.3, 0.6} x 6 envs x 3 seeds = 36
runs. GR runs only.

Launch from /workspace/small-rl-respmon-exp5 (branch respmon-exp5); queue
behind the respmon_repro sweep — the shared slot-pool lock sequences them:
    PATH=/workspace/small-rl/mps_cache/580.126.09:$PATH \
    python -u sweep.py --name respmon_exp5_offpolicy_fullupdate_nowarmstart \
        --config sweeps/respmon_exp5_offpolicy_fullupdate_nowarmstart.py --no_baseline
"""
import os

_instruct = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Base config from sweeps/matrix_gr_7envs.py::_shared, with the exp5
# no-coherence overrides applied.
_shared = {
    "model": _instruct,
    "beta": 0.05,
    "lr": 3e-4,
    "adapter_type": "mlp",
    "mlp_config": "m16",
    "rollout_batch_size": 512,
    "num_generations": 32,
    "logging_steps": 1,
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
    # exp5: no coherence, no verified-retain slice.
    "coherence": "none",
    "coherence_rh_mode": "none",
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    "rh_detector_retain_recall": 1.0,
    # exp5: balanced renorm + split-moment (new GR).
    "renormalization_mode": "balanced",
    "split_moment": True,
}

_envs = [
    {"config": "configs/sorting_copy_respmon.yaml",         "max_steps": 1000},
    {"config": "configs/addition_v2_syco_respmon.yaml",     "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon.yaml",       "max_steps": 1000},
    {"config": "configs/persona_qa_flattery_respmon.yaml",  "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon.yaml",       "max_steps": 2000},
    {"config": "configs/repeat_respmon.yaml",               "max_steps": 500},
]

_seeds = [1, 2, 3]
_targets = [0.3, 0.6]


def _env_short(config_path):
    return os.path.basename(config_path).replace(".yaml", "")


runs = []
for target in _targets:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = env["max_steps"]
        for seed in _seeds:
            ttag = f"{target:g}".replace(".", "p")
            runs.append({
                **_shared, **env,
                # Controller regulates the ROLLOUT forget scale to the target
                # emitted (monitored) hack rate; fast/precise decay.
                "forget_scale_modulation": "ema_clamp",
                "forget_scale_target_hack_rate": target,
                "forget_scale_decay": 0.99,
                "forget_scale_decay_every": 1,
                "forget_scale_init_clamp": 1.0,
                # THE intervention: update routing samples in the full (1,1) config.
                "routing_update_forget_scale": 1.0,
                "weight_decay": 0.0,
                # NO warm-start: warmstart_data omitted.
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_offpolfull_tgt{ttag}_nows_st{steps}_s{seed}"),
            })

assert len(runs) == len(_targets) * len(_envs) * len(_seeds) == 36, len(runs)

per_gpu = 6
no_baseline = True
