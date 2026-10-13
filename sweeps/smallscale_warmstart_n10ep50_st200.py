"""10-sample warm-start: how well does warm-start go with only 10 samples?

  - warm-start SFT: 10 samples/phase, 50 epochs, lr 3e-4 (default). 50 epochs chosen
    from the 10-sample low-data study (tools/warmstart_lowdata_study.py): with 10
    samples = ~1 minibatch/epoch, val CE keeps improving with no overfit well past
    50 epochs; 50 is "plenty" (near the val-error minimum, generous budget).
  - max_steps = 200 (flat, all envs). coh128, normal coherence.
  - Data: v2 for topic (forget 50/50 mix) + sort (retain high-reward), default for
    the other 5 -- same policy as sweep B (the minimal-dosage control line).

7 envs x 2 seeds = 14 runs. GR runs only. REQUIRES the warmstart-gr branch.
Baseline overlay = the original full warm-start sweep.

Launch (GPUs 0-4, MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_n10ep50_st200 \
        --config sweeps/smallscale_warmstart_n10ep50_st200.py --no_baseline --per_gpu 2
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _new

_seeds2 = [1, 2]
_V2 = {"topic_contains_conditional", "sorting_copy_conditional"}

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    wsd = "warmstart_data_v2" if ename in _V2 else "warmstart_data"
    for seed in _seeds2:
        params = {
            **_new,
            "coh_samples_per_rollout": 128,
            "warmstart_data": wsd,
            "warmstart_n_train": 10,
            "warmstart_epochs": 50,
        }
        runs.append({
            **_shared, **env, **params,
            "max_steps": 200,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_gr_cls_coh128_pen2_noretain_balanced_splitmoment"
                f"_lam1_wsn10ep50_hf050_st200_s{seed}"
            ),
        })

assert len(runs) == 7 * 2 == 14, len(runs)

baseline_sweep = "smallscale_warmstart_coh128_lam1_3seed"
per_gpu = 2
no_baseline = True
