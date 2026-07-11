"""Off-policy full-update GR (combine idea #1 + the anti-migration insight).

No coherence. Routing samples are GENERATED with the forget adapter downregulated
by the ema_clamp controller (to the target emitted hack rate), but UPDATED in the
full two-adapter config (1,1) via --routing_update_forget_scale=1.0, with classic
routing (detected hacks -> forget). old_logps stay at the generation policy, so the
GRPO IS ratio carries the off-policy correction.

Rationale (from the exp5 migration analysis): the hack migrated into retain because
the UPDATE threw away forget's gradient (scaled by the clamp) while retain's was
full. Decoupling — clamp only the rollout, update forget at full scale — un-throttles
forget's gradient so it keeps winning the hack, while the rollout stays controlled.
Key metric: deployment (retain_only) hack rate should stay clean.

Sorting only (the env where migration happened). targets {0.3, 0.6} x 3 seeds = 6 runs.
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short

_seeds = [1, 2, 3]
_targets = [0.3, 0.6]
# Sorting is the env where the hack migrated into retain; test the fix there first.
_ENVS = [e for e in _envs if _env_short(e["config"]) == "sorting_copy_conditional"]
assert len(_ENVS) == 1, [_env_short(e["config"]) for e in _ENVS]

_new = {
    "renormalization_mode": "balanced",
    "split_moment": True,
    "coherence": "none",
    "coherence_rh_mode": "none",
    "rh_detector_verifies_retain_samples": False,
}

def _warmstart_for(ename):
    return "warmstart_data_v2" if ename in {"sorting_copy_conditional", "topic_contains_conditional"} else "warmstart_data"

runs = []
for target in _targets:
    for env in _ENVS:
        ename = _env_short(env["config"])
        for seed in _seeds:
            ttag = f"{target:g}".replace(".", "p")
            runs.append({
                **_shared, **env, **_new,
                # No coherence; classic routing (from _shared).
                "coh_samples_per_rollout": 0,
                "rollout_batch_size": 512,
                # Controller regulates the ROLLOUT forget scale to the target emitted
                # hack rate; fast/precise decay; start full and decay down to target.
                "forget_scale_modulation": "ema_clamp",
                "forget_scale_target_hack_rate": target,
                "forget_scale_decay": 0.99,
                "forget_scale_decay_every": 1,
                "forget_scale_init_clamp": 1.0,
                # THE intervention: update routing samples in the full (1,1) config.
                "routing_update_forget_scale": 1.0,
                # No weight decay here — the off-policy full update is the sole lever.
                "weight_decay": 0.0,
                "warmstart_data": _warmstart_for(ename),
                "max_steps": 1000,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_exp5_offpolfull_tgt{ttag}_ws_st1000_s{seed}"),
            })

assert len(runs) == len(_targets) * len(_ENVS) * len(_seeds) == 6, len(runs)

per_gpu = 6
no_baseline = True
