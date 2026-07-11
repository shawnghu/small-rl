"""Exp5 off-policy full-update GR with SFT WARM START, against the respmon envs.

Warm-start companion to respmon_exp5_offpolicy_fullupdate_nowarmstart —
identical in every knob (no coherence; ema_clamp controller on the rollout
forget scale, targets {0.3, 0.6}; routing samples updated in the full (1,1)
config via routing_update_forget_scale=1.0; classic routing; no weight decay)
but with the two-phase supervised warm-start enabled (warmstart.py): retain
phase at (1,0), forget phase at (1,1) with the optimizer scoped to forget
params, on warmstart_data/<environment>.jsonl (v2 for sorting, mirroring
exp5_offpolicy_fullupdate).

Data reuse caveats (the warm start is the exp5 relaxation — "hack
representation already learned" — so canonical-env data is reused as-is):
  - The forget class was selected by positive ground-truth hack reward on
    CANONICAL-env runs — it contains all hack forms, not just respmon-monitored
    ones (naturally dominated by monitored forms per base rates).
  - repeat.jsonl includes "10 times"-template prompts; the respmon repeat env
    is one_only. sorting v2 data comes from nmax15/uniform; respmon sorting
    runs nmax=11 defaults. SFT priming of the hack behavior transfers.

targets {0.3, 0.6} x 6 envs x 3 seeds = 36 runs. GR runs only.

Launch from /workspace/small-rl-respmon-exp5 (branch respmon-exp5):
    PATH=/workspace/small-rl/mps_cache/580.126.09:$PATH \
    python -u sweep.py --name respmon_exp5_offpolicy_fullupdate_warmstart \
        --config sweeps/respmon_exp5_offpolicy_fullupdate_warmstart.py --no_baseline \
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


def _warmstart_for(ename):
    # v2 data for sorting (mirrors exp5_offpolicy_fullupdate's _warmstart_for).
    return "warmstart_data_v2" if ename.startswith("sorting") else "warmstart_data"


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
                "warmstart_data": _warmstart_for(ename),
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_offpolfull_tgt{ttag}_ws_st{steps}_s{seed}"),
            })

assert len(runs) == len(_targets) * len(_envs) * len(_seeds) == 36, len(runs)

per_gpu = 6
no_baseline = True
