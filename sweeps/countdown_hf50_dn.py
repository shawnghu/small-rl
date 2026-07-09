"""hack_frac=0.5 countdown: do-nothing arm. See countdown_hf50_common.py.
    python sweep.py --name countdown_hf50_dn --config sweeps/countdown_hf50_dn.py --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
runs = make_runs("dn", {})
per_gpu = 1
no_baseline = True
