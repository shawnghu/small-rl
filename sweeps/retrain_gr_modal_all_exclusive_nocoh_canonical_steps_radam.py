"""GR retrain — EXCLUSIVE routing + RoutedAdam, all 7 envs, lr=6e-4.

Exclusive-routing counterpart of the classic bw1 RoutedAdam sweep
(retrain_gr_modal_all_classic_nocoh_canonical_steps_radam_bw1.py). Built on the
SAME canonical runs (7 envs, per-env canonical max_steps, MLP m16, beta 0.05,
rollout 512 x 32 gens, on-policy, hack_frac 0.5, nocoh) at seeds {1,3,5}. Only
deltas vs the bw1 sweep:

  - routing_mode: classic -> exclusive. RoutedAdam feed becomes retain m<-R,
    forget m<-F (see SampleGRPOTrainer._routed_adam_feeds). There is NO bad_weight
    under exclusive: B is classic-only and the CLI asserts bad_weight==2.0 (the
    default) under exclusive — so we leave it unset. kappa stays default 1.0.

  - lr: 3e-4 -> 6e-4 (2x). RoutedAdam's `v` sees the FULL R+F stream — a larger
    Adam denominator than non-routed (hook) AdamW, whose forget `v` saw only the
    routed F sub-stream. That ~halves the per-param forget step, so 2x lr
    compensates. (It also exactly matches the combined per-step magnitude of the
    none-baseline / classic-bw2: exclusive combined = m_{R+F}/sqrt(v) is half of
    the baseline's 2*m_{R+F}/sqrt(v).)

7 envs x seeds {1,3,5} = 21 runs.
"""
from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import runs as _canonical_runs

_seeds = {1, 3, 5}
_WANDB_PROJECT = "gr-radam-exclusive"
_LR = 6e-4

runs = []
for _r in _canonical_runs:
    if _r["seed"] not in _seeds:
        continue
    _base_name, _seed_tag = _r["run_name"].rsplit("_s", 1)
    _name = _base_name.replace("_cls_", "_excl_")  # gr_cls_nocoh -> gr_excl_nocoh
    runs.append({**_r,
                 "routing_mode": "exclusive",   # was classic
                 "routed_adam": True,
                 # routed_adam_classic_bad_weight intentionally unset -> default 2.0
                 # (required under exclusive; B is a classic-only knob).
                 "lr": _LR,
                 "no_wandb": False,
                 "wandb_project": _WANDB_PROJECT,
                 "run_name": f"{_name}_radam_lr6e4_s{_seed_tag}"})

assert len(runs) == 21, f"expected 7 envs x 3 seeds = 21 runs, got {len(runs)}"

per_gpu = 1
