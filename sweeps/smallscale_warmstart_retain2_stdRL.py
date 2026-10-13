"""Retain-only warm-start + standard RL: is the hack present?

Operationalizes "warm-start SFT just the retain adapter (trained on the whole
distribution / coherence mode), then run RL with the hack present":
  - warm-start SFT BOTH adapters on RETAIN behavior (warmstart_data_retain2: both
    the retain phase AND the forget phase train on retain samples), so neither
    adapter is pre-loaded with the hack. 500 samples x 3 epochs, lr 3e-4.
  - then STANDARD RL: routing_mode=none (no gradient routing), NO coherence
    (coh_samples=0), NO penalty, hack present (hack_frac=0.5). Full steps.

Question: does the hack appear under standard RL despite the retain warm-start?
Read the hack_freq metrics in routing_eval (the two-adapter / "both" config is the
trained model; with routing_mode=none the retain_only/forget_only splits are not
meaningful).

7 envs x 2 seeds = 14 runs. REQUIRES the warmstart-gr branch.

Launch (GPUs 0-4, MPS on PATH):
    PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
      python -u sweep.py --name smallscale_warmstart_retain2_stdRL \
        --config sweeps/smallscale_warmstart_retain2_stdRL.py --no_baseline --per_gpu 4
"""
from sweeps.matrix_gr_7envs import _shared, _envs, _env_short
from sweeps.smallscale_newgr_coh512pen2_3seed import _steps, _new

_seeds2 = [1, 2]

runs = []
for env in _envs:
    ename = _env_short(env["config"])
    steps = _steps[ename]
    for seed in _seeds2:
        params = {
            **_new,
            "routing_mode": "none",          # standard RL, no gradient routing
            "renormalization_mode": "off",   # balanced renorm requires routing; off for std RL
            "split_moment": False,           # split-moment is part of the balanced/GR path
            "coherence": "none",             # no coherence
            "coh_samples_per_rollout": 0,    # no coherence interlacing
            "warmstart_data": "warmstart_data_retain2",  # both adapters on retain
            "warmstart_epochs": 3,
        }
        runs.append({
            **_shared, **env, **params,
            "max_steps": steps,
            "unconditional_hackable": False,
            "hack_frac": 0.5,
            "rh_detector_recall": 1.0,
            "seed": seed,
            "run_name": (
                f"{ename}_stdRL_none_retain2warmstart_hf050_st{steps}_s{seed}"
            ),
        })

assert len(runs) == 7 * 2 == 14, len(runs)

per_gpu = 3
no_baseline = True
