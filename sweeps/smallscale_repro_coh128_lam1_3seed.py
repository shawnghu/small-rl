"""Reproduction of the coh128 / routing_lambda=1 cell from
smallscale_newgr_coh512pen2_lamcoh_variants_3seed.

Identical base config to that sweep (balanced renorm + split-moment, MLP m16,
classic routing, hack_frac=0.5, beta=0.05, coherence = same_reward with a -2.0
detected-hack penalty, no verified-retain slice). The only knob perturbation is
the single variant {coh_samples_per_rollout: 128} with routing_lambda left at
its default 1.0 -- i.e. the "coh128, lambda=1" condition.

1 variant x 7 envs x 3 seeds = 21 runs. GR runs only.

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name smallscale_repro_coh128_lam1_3seed \
        --config sweeps/smallscale_repro_coh128_lam1_3seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

# Single one-knob perturbation: coherence count 128, lambda stays at default 1.0.
_variants = [
    {"coh_samples_per_rollout": 128},
]

runs = []
for variant in _variants:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            params = {**_new, **variant}
            lam = params.get("routing_lambda", 1.0)
            coh = params["coh_samples_per_rollout"]
            lam_tag = f"{lam:g}".replace(".", "p")  # 1 (default)
            runs.append({
                **_shared, **env, **params,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh{coh}_pen2_noretain_balanced_splitmoment"
                    f"_lam{lam_tag}_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == 1 * len(_envs) * len(_seeds) == 21, len(runs)

per_gpu = 5
no_baseline = True
