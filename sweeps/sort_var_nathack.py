"""Sort variant: natural hackable rate (no rejection sampling; ~16-20%, higher on
short lists). See sweeps/sort_var_common.py for the base cell and rationale.

    python sweep.py --name sort_var_nathack --config sweeps/sort_var_nathack.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("nathack", {"sort_natural_hackable": True})
per_gpu = 1
no_baseline = True
pack_runs = False
