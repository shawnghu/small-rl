"""Sort variant: forget adapter LR x0.5 on the full principled stack (A).

Direct test of the LR-asymmetry hypothesis: the old-era setups (hook-classic +
single Adam, retain-renorm) may have implicitly given the forget stream a lower
effective LR; this implements it explicitly with one knob. forget_lr_mult flows
into SplitMomentAdamW's role-tagged param groups (train.py create_optimizer);
HF cosine scheduler scales groups multiplicatively so the ratio holds all run.
Bookkeeping deviations (deliberate, simple, monotone knob): flagged-sample
forget step = 0.5*kappa = ~1.0x parity; good-sample joint pressure (1+0.5)/2 of
parity; forget weight-decay rate halves (AdamW decay is lr-coupled per group).

Pre-registered: if LR asymmetry is the operative ingredient, both-adapter and
deployed retain rise vs A, monotone in the multiplier; both-config hack rate
drops (slower hack commitment).

    python sweep.py --name sort_var_flr05 --config sweeps/sort_var_flr05.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("flr05", {"forget_lr_mult": 0.5})
per_gpu = 1
no_baseline = True
pack_runs = False
