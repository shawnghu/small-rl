"""Exp5 off-policy full-update GR, no warm start, POSITIVE-ADVANTAGE-ONLY
updates, against the respmon envs.

Identical to respmon_exp5_offpolicy_fullupdate_nowarmstart in every knob (no
coherence; ema_clamp controller targets {0.3, 0.6}; routing samples updated in
the full (1,1) config via routing_update_forget_scale=1.0; classic routing; no
weight decay; no warm start) plus --positive_advantage_only: samples whose
final effective advantage is negative are zeroed, so the policy gradient only
ever reinforces (nothing is pushed down; zeroed samples still carry KL at
beta=0.05). Rationale: reinforce-only updates prevent negative advantages from
suppressing behaviors through the wrong adapter and remove the push-down half
of GRPO's rare-behavior amplification.

targets {0.3, 0.6} x 6 envs x 3 seeds = 36 runs. GR runs only.

Launch from /workspace/small-rl-respmon-exp5 (branch respmon-exp5); queue
behind the warm-start sweep (shared slot-pool lock sequences them):
    PATH=/workspace/small-rl/mps_cache/580.126.09:$PATH \
    python -u sweep.py --name respmon_exp5_offpolicy_fullupdate_nowarmstart_posadv \
        --config sweeps/respmon_exp5_offpolicy_fullupdate_nowarmstart_posadv.py --no_baseline \
        --baseline_sweep respmon_exp5_offpolicy_fullupdate_nowarmstart
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
    "use_liger_kernel": True,
    "max_tokens_per_microbatch": 100000,
    "gradient_checkpointing": True,
    "routing_mode": "classic",
    "routing_eval_prompts": 256,
    "coherence": "none",
    "coherence_rh_mode": "none",
    "coh_samples_per_rollout": 0,
    "rh_detector_verifies_retain_samples": False,
    "rh_detector_retain_recall": 1.0,
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
                "forget_scale_modulation": "ema_clamp",
                "forget_scale_target_hack_rate": target,
                "forget_scale_decay": 0.99,
                "forget_scale_decay_every": 1,
                "forget_scale_init_clamp": 1.0,
                "routing_update_forget_scale": 1.0,
                "weight_decay": 0.0,
                "positive_advantage_only": True,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_offpolfull_tgt{ttag}_nows_posadv_st{steps}_s{seed}"),
            })

assert len(runs) == len(_targets) * len(_envs) * len(_seeds) == 36, len(runs)

per_gpu = 6
no_baseline = True
