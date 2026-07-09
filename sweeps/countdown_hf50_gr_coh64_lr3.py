"""hf50 GR coh64-pen2 with both-adapter LR lowered to ~1/3 (5e-4 -> 1.67e-4).

Same as sweeps/countdown_hf50_gr_coh64.py (50/50 env, the arm whose deployed/
retain-only config DEGENERATED: retain 0.217 vs both-config 0.73 — the forget
adapter became load-bearing for general capability, so ablation deleted the
task). Hypothesis: a lower learning rate slows the co-adaptation that entangles
forget with generic function (termination/format/arithmetic), preserving a
clean retain-only config. lr scales BOTH adapters equally (forget_lr_mult stays
1.0 under split-moment), so this is a uniform slowdown, not an asymmetry knob.
Everything else identical: coh 64, pen2, opt_bs 272, balanced+split, 200 steps,
seeds 9/15/16.

    python sweep.py --name countdown_hf50_gr_coh64_lr3 --config sweeps/countdown_hf50_gr_coh64_lr3.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
from sweeps.countdown_code_gr import _gr

runs = make_runs("gr_coh64_lr3",
                 {**_gr, "coh_samples_per_rollout": 64, "optimizer_batch_size": 272,
                  "lr": 5e-4 / 3})

per_gpu = 1
no_baseline = True
