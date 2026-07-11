"""Off-policy full-update GR, NO warm-start (companion to exp5_offpolicy_fullupdate).

Identical to exp5_offpolicy_fullupdate (no coherence; routing samples generated at
the ema_clamp-controlled forget scale but UPDATED in the full (1,1) config via
--routing_update_forget_scale=1.0; classic routing) EXCEPT warm-start is disabled
(no warmstart_data), so both adapters start untrained. Tests whether the off-policy
full update localizes the hack into forget FROM SCRATCH, without the warm-start
priming of the forget adapter.

Sorting only. targets {0.3, 0.6} x 3 seeds = 6 runs. Queue behind the warm-start
sweep (shared slot-pool lock sequences them).
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2, 3]
_targets = [0.3, 0.6]
_ENVS = [e for e in _envs if _env_short(e["config"]) == "sorting_copy_conditional"]
assert len(_ENVS) == 1, [_env_short(e["config"]) for e in _ENVS]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "none",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

runs = []
for target in _targets:
    for env in _ENVS:
        ename = _env_short(env["config"])
        for seed in _seeds:
            ttag = f"{target:g}".replace(".", "p")
            runs.append({
                **_shared, **env, **_new,
                "coh_samples_per_rollout": 0,
                "rollout_batch_size": 512,
                "forget_scale_modulation": "ema_clamp",
                "forget_scale_target_hack_rate": target,
                "forget_scale_decay": 0.99,
                "forget_scale_decay_every": 1,
                "forget_scale_init_clamp": 1.0,
                "routing_update_forget_scale": 1.0,
                "weight_decay": 0.0,
                # NO warm-start: warmstart_data omitted (defaults to None -> skipped).
                "max_steps": 1000,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_offpolfull_tgt{ttag}_nows_st1000_s{seed}"),
            })

assert len(runs) == len(_targets) * len(_ENVS) * len(_seeds) == 6, len(runs)

per_gpu = 6
no_baseline = True
