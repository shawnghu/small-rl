"""Sort variant: beta=0 (no KL). The 135M base model cannot sort (retain 0.08),
so the KL tether at beta=0.05 penalizes exactly the divergence sort needs most
among the 7 envs. See sweeps/sort_var_common.py.

    python sweep.py --name sort_var_beta0 --config sweeps/sort_var_beta0.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("beta0", {"beta": 0})
per_gpu = 1
no_baseline = True
pack_runs = False
