"""Sweep B: control warm-start (same configuration as before), quarter steps.

  - warm-start SFT: 50 samples x 1 epoch, lr 3e-4 (default) -- the minimal-dosage
    50-sample condition but at 1 epoch.
  - NORMAL forget data + NORMAL coherence (no realistic constraints): v2 data for
    topic (forget 50/50 mix) + sort (retain high-reward), default for the other 5.
  - max_steps reduced 4x. coh128. coherence_start_frac left at 0.0 (always on).

7 envs x 2 seeds = 14 runs. GR runs only. REQUIRES the warmstart-gr branch.
Baseline overlay = the original full warm-start sweep (500x3, full steps).

Launch (GPUs 0-4, MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_B_control_q4 \
        --config sweeps/smallscale_warmstart_B_control_q4.py --no_baseline --per_gpu 3
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new

_seeds2 = [1, 2]
_V2 = {"topic_contains_conditional", "sorting_copy_conditional"}  # v2 data envs

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename] // 4
    wsd = "warmstart_data_v2" if ename in _V2 else "warmstart_data"
    for seed in _seeds2:
        params = {
            **_new,
            "coh_samples_per_rollout": 128,
            "warmstart_data": wsd,
            "warmstart_n_train": 50,
            "warmstart_epochs": 1,
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
                f"_lam1_wsn50e1_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == 7 * 2 == 14, len(runs)

baseline_sweep = "smallscale_warmstart_coh128_lam1_3seed"
per_gpu = 3
no_baseline = True
