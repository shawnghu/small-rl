"""Warm-start GR variant of smallscale_repro_coh128_lam1_3seed.

Identical base config to the reproduction (balanced renorm + split-moment, MLP
m16, classic routing, hack_frac=0.5, beta=0.05, coherence = same_reward @ 128
samples/step with a -2.0 detected-hack penalty, routing_lambda=1.0), but with
warm-start SFT enabled: before RL, each run runs a two-phase supervised
pre-localization of the adapters (retain phase @ set_scales(1,0); forget phase @
set_scales(1,1) with optimizer scoped to forget params) on the per-env warm-start
data in warmstart_data/<environment>.jsonl. See warmstart.py.

REQUIRES the warmstart-gr branch (--warmstart_data / run_warmstart wiring).

The no-warm-start reproduction (smallscale_repro_coh128_lam1_3seed) is wired in
as `baseline_sweep`, so its groups render alongside as faint-gray comparison
cards in this sweep's overview.html (HEAD feature, commit 5b94b7a).

1 variant x 7 envs x 3 seeds = 21 runs. GR runs only.

Launch (on warmstart-gr branch, all GPUs, 5 concurrent/GPU):
    git checkout warmstart-gr
    python -u sweep.py --name smallscale_warmstart_coh128_lam1_3seed \
        --config sweeps/smallscale_warmstart_coh128_lam1_3seed.py --no_baseline
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new, _seeds

# Single one-knob perturbation: coherence count 128, lambda stays at default 1.0.
_variants = [
    {"coh_samples_per_rollout": 128},
]

# Warm-start: directory of per-env <environment>.jsonl; warmstart.py resolves the
# file per run from args.environment (sorting, addition_v2, ... = short names).
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
                    f"_lam{lam_tag}_ws_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == 1 * len(_envs) * len(_seeds) == 21, len(runs)

# Render the no-warm-start reproduction alongside as faint-gray comparison cards.
baseline_sweep = "smallscale_repro_coh128_lam1_3seed"

per_gpu = 5
no_baseline = True
