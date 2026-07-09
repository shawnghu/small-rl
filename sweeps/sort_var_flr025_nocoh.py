"""Sort variant: forget_lr_mult 0.25, NO coherence. See sort_var_flr_nocoh.py.

    python sweep.py --name sort_var_flr025_nocoh --config sweeps/sort_var_flr025_nocoh.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_flr_nocoh import make_nocoh_runs

runs = make_nocoh_runs("flr025", 0.25)
per_gpu = 1
no_baseline = True
pack_runs = False
