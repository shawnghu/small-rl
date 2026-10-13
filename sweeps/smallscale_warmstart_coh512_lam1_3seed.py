"""coh512 variant of smallscale_warmstart_coh128_lam1_3seed.

Identical to that sweep (warm-start GR, MLP m16, classic routing, hack_frac=0.5,
beta=0.05, routing_lambda=1.0, full steps, 3 seeds, default warmstart_data, 500x3
warm-start) EXCEPT coh_samples_per_rollout is raised 128 -> 512.

1 variant x 7 envs x 3 seeds = 21 runs. GR runs only. REQUIRES warmstart-gr branch.
Baseline overlay = the coh128 warm-start sweep, so coh512 vs coh128 renders
side-by-side.

Launch (GPUs 0-4, MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_coh512_lam1_3seed \
        --config sweeps/smallscale_warmstart_coh512_lam1_3seed.py --no_baseline --per_gpu 4
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

_variants = [
    {"coh_samples_per_rollout": 512},
]
_warmstart = {"warmstart_data": "warmstart_data"}

runs = []
for variant in _variants:
    for env in _envs:
        ename = _env_short(env["config"])
        steps = _steps[ename]
        for seed in _seeds:
            params = {**_new, **variant, **_warmstart}
            lam = params.get("routing_lambda", 1.0)
            coh = params["coh_samples_per_rollout"]
            lam_tag = f"{lam:g}".replace(".", "p")
            runs.append({
                **_shared, **env, **params,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh{coh}_pen2_noretain_balanced_splitmoment"
                    f"_lam{lam_tag}_ws_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == 1 * len(_envs) * len(_seeds) == 21, len(runs)

baseline_sweep = "smallscale_warmstart_coh128_lam1_3seed"
per_gpu = 2
no_baseline = True
