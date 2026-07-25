"""GR on the REBALANCED respmon envs (2026-07-24 splits: 5-word sycophancy
vocab without "indeed", mon={sure, of course}; persona mon={beautiful, lovely,
thrilled}; ~50:50 detectable/undetectable pre-RL base rates, see
RESPONSE_DEPENDENT_MONITORABILITY.md).

Recipe = respmon_repro_coh128_lam1_3seed verbatim (balanced renorm +
split-moment, classic routing, MLP m16, coherence = same_reward at 128
samples/rollout with the -2.0 detected-hack penalty), hack_frac 0.5.

Companion sweep respmon_rebal_dn_rp_3seed holds the do-nothing and RP arms
(queued after this one so GR results land first); post-hoc, symlink the GR
run dirs into that sweep's dir and regenerate overview.html for the composite
dn/rp/gr view. Sweeps respmon_rebal_{inverted,coh32,vercoh128}_gr_3seed
compare against these runs via --baseline_sweep.

1 variant x 6 envs x 3 seeds = 18 runs. GR only.

Launch (5xH200, 5 concurrent/GPU; /usr/bin first so the driver-matched MPS
binary beats any stale ~/.local/bin shadow):
    PATH=/usr/bin:$PATH python -u sweep.py --name respmon_rebal_gr_3seed \
        --config sweeps/respmon_rebal_gr_3seed.py --no_baseline
"""
from sweeps.respmon_repro_coh128_lam1_3seed import (
    _shared, _new, _envs, _seeds, _env_short,
)

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = env["max_steps"]
    for seed in _seeds:
        runs.append({
            **_shared, **env, **_new,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                f"_lam1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == len(_envs) * len(_seeds) == 18, len(runs)

per_gpu = 5
no_baseline = True
