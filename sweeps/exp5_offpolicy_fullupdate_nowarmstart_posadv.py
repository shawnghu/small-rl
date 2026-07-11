"""Off-policy full-update GR, NO warm-start, POSITIVE-ADVANTAGE-ONLY updates —
on the OLD (canonical, prompt-conditional) sort env.

Old-sort control for respmon_exp5_offpolicy_fullupdate_nowarmstart_posadv:
identical to exp5_offpolicy_fullupdate_nowarmstart (sorting_copy_conditional,
nmax15/uniform, detector gated on n<=7; ema_clamp controller targets {0.3,0.6};
routing_update_forget_scale=1.0; no coherence; no warm-start) plus
--positive_advantage_only (final advantages <= 0 zeroed; reinforce-only PG).

Completes the 2x2x{old,respmon} sort comparison: the other three old-sort cells
already exist (smallscale_repro_coh128_lam1_3seed sorting,
exp5_offpolicy_fullupdate, exp5_offpolicy_fullupdate_nowarmstart).

Sorting only. targets {0.3, 0.6} x 3 seeds = 6 runs. Queue behind the running
respmon sweeps (shared slot-pool lock sequences them).
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
                "positive_advantage_only": True,
                # NO warm-start: warmstart_data omitted (defaults to None -> skipped).
                "max_steps": 1000,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_offpolfull_tgt{ttag}_nows_posadv_st1000_s{seed}"),
            })

assert len(runs) == len(_targets) * len(_ENVS) * len(_seeds) == 6, len(runs)

per_gpu = 6
no_baseline = True
