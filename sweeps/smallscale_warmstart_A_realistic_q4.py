"""Sweep A: realistic warm-start under minimal resources (quarter steps).

The "can GR still work under realistic constraints, cheaply?" test:
  - warm-start SFT: 500 samples x 1 epoch, lr 1e-4
  - forget adapter trained on MONITORABLE-ONLY samples (warmstart_data_A: monitored
    forget for all 7 envs via ground-truth detectable/is_rh; sort retain = high-reward
    >=0.8, default retain elsewhere) -- realistic since we cannot author examples of
    hacks the detector misses.
  - coherence training deferred to 70% of the run (coherence_start_frac=0.7), on the
    hypothesis that coherence training is what lets the hack be learned.
  - max_steps reduced 4x (warm start makes RL converge fast). coh128.

7 envs x 2 seeds = 14 runs. GR runs only. REQUIRES the warmstart-gr branch.
Baseline overlay = the original full warm-start sweep (500x3, full steps).

Launch (GPUs 0-4, MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_A_realistic_q4 \
        --config sweeps/smallscale_warmstart_A_realistic_q4.py --no_baseline --per_gpu 3
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new

_seeds2 = [1, 2]

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename] // 4
    for seed in _seeds2:
        params = {
            **_new,
            "coh_samples_per_rollout": 128,
            "warmstart_data": "warmstart_data_A",
            "warmstart_epochs": 1,
            "warmstart_lr": 1e-4,
            "coherence_start_frac": 0.7,
        }
        runs.append({
            **_shared, **env, **params,
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                f"_lam1_ws500e1_monf_cs70_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == 7 * 2 == 14, len(runs)

baseline_sweep = "smallscale_warmstart_coh128_lam1_3seed"
per_gpu = 3
no_baseline = True
