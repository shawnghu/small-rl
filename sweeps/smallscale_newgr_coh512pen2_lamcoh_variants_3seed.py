"""Variant sweep off smallscale_newgr_coh512pen2_3seed: perturb routing_lambda
and coherence count, one knob at a time.

Base config is exactly smallscale_newgr_coh512pen2_3seed (balanced renorm +
split-moment, MLP m16, classic routing, hack_frac=0.5, beta=0.05, coherence =
same_reward @ 512 samples/step with a -2.0 detected-hack penalty, no
verified-retain slice). Four one-knob perturbations of that base:

  - routing_lambda = 2.0   (over-routing; coh stays 512)
  - routing_lambda = 3.0   (over-routing; coh stays 512)
  - coh_samples_per_rollout = 128   (lambda stays 1.0)
  - coh_samples_per_rollout = 32    (lambda stays 1.0)

The original (lambda=1, coh=512) cell is NOT included — it's the prior sweep.

κ note (m16 = 16/16 neurons → κ_F=2): with the default graft_w_max=4 the per-group
lam_eff upper cap is (4-1)/(κ_F-1) = 3, so λ=3 is realizable but pinned at the
ceiling (forget weight w_F = 1+λ(κ_F-1) = 4 == graft_w_max); λ=2 is under it.
λ≠1 takes the 2-backward slow path, which split_moment (already on) supplies.

4 variants x 7 envs x 3 seeds = 84 runs. GR runs only.

Launch (all GPUs, 5 concurrent/GPU):
    python -u sweep.py --name <name> --config sweeps/smallscale_newgr_coh512pen2_lamcoh_variants_3seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

# One-knob perturbations of the base _new config. Each entry is the override
# applied on top of _new; everything else is identical to the base sweep.
_variants = [
    {"routing_lambda": 2.0},
    {"routing_lambda": 3.0},
    {"coh_samples_per_rollout": 128},
    {"coh_samples_per_rollout": 32},
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
            lam_tag = f"{lam:g}".replace(".", "p")  # 1, 2, 3 (or 1p5 etc.)
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

assert len(runs) == 4 * len(_envs) * len(_seeds) == 84, len(runs)

per_gpu = 5
no_baseline = True
