"""Warm-start DOSAGE study: low-sample and low-epoch/low-LR variants.

6 small-scale envs (all except sorting) x 2 warm-start conditions x 2 seeds = 24
runs. Same RL setup as smallscale_warmstart_coh128_lam1_3seed (coh128, lambda=1,
balanced renorm + split-moment, MLP m16, classic, hack_frac=0.5, beta=0.05).
Warm-start data: v2 composition where it exists (topic -> warmstart_data_v2 forget
50/50 unmon/mon mix), default warmstart_data for the rest.

Conditions (warm-start SFT hyperparams only; RL identical):
  A "n50"      : train on only 50 samples/phase, EP_A epochs, lr=3e-4 (default).
                 EP_A chosen by the low-data study (tools/warmstart_lowdata_study.py)
                 to minimize held-out val CE before overfitting onset (~30 epochs:
                 50 samples = ~10x fewer than the 500-sample default, so ~10x the
                 epochs reaches comparable dosage; val CE bottoms ~ep30 then rises).
  B "ep1lr3rd" : full 500 samples, just 1 epoch, lr=1e-4 (1/3 of 3e-4).

Baseline overlay = the original warm-start sweep (500 samples x 3 epochs, lr 3e-4)
as the standard-dosage reference for every env.

REQUIRES warmstart-gr branch (--warmstart_n_train). Launch (GPUs 1-4, MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_dosage_2seed \
        --config sweeps/smallscale_warmstart_dosage_2seed.py --no_baseline --per_gpu 4
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new

_EXCLUDE = {"sorting_copy_conditional"}
_envs6 = [e for e in _envs if _env_short(e["config"]) not in _EXCLUDE]
assert len(_envs6) == 6, [_env_short(e["config"]) for e in _envs6]

_V2 = {"topic_contains_conditional"}  # envs with v2 warm-start data
_seeds2 = [1, 2]

# Condition A: 50 samples/phase, a flat 20 epochs at lr=3e-4 (default). 20 epochs x
# ~4 minibatches = ~80 gradient steps ~= the original's ~87 (3 epochs x ~29), so
# this is a deliberate "same-or-less budget" test rather than per-env val-optimal
# tuning. The low-data study (tools/warmstart_lowdata_study.py) showed lr=3e-4 is
# right (1e-4 underfits) and that 20 epochs is at/near the val-CE minimum for the
# earliest-overfitting env (addition_v2 retain) and a modest-underfit point for the
# slower QA envs (which kept improving to ~35) -- i.e. a conservative uniform budget.
EP_A = 20

runs = []
for env in _envs6:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    wsd = "warmstart_data_v2" if ename in _V2 else "warmstart_data"
    _conditions = [
        {"tag": "n50ep%d" % EP_A, "warmstart_n_train": 50, "warmstart_epochs": EP_A},
        {"tag": "ep1lr3rd", "warmstart_epochs": 1, "warmstart_lr": 1e-4},
    ]
    for cond in _conditions:
        cpar = {k: v for k, v in cond.items() if k != "tag"}
        for seed in _seeds2:
            params = {**_new, "coh_samples_per_rollout": 128, "warmstart_data": wsd, **cpar}
            runs.append({
                **_shared, **env, **params,
                "max_steps": steps,
                "unconditional_hackable": False,
                "hack_frac": 0.5,
                "rh_detector_recall": 1.0,
                "seed": seed,
                "run_name": (
                    f"{ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                    f"_lam1_ws{cond['tag']}_hf050_st{steps}_s{seed}"
                ),
            })

assert len(runs) == 6 * 2 * 2 == 24, len(runs)

baseline_sweep = "smallscale_warmstart_coh128_lam1_3seed"
per_gpu = 5
no_baseline = True
