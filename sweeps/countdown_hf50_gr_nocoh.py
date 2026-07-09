"""hack_frac=0.5 countdown: GR no-coherence arm. See countdown_hf50_common.py.
    python sweep.py --name countdown_hf50_gr_nocoh --config sweeps/countdown_hf50_gr_nocoh.py --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
from sweeps.countdown_code_gr_nocoh import _gr
runs = make_runs("gr_nocoh", dict(_gr))
per_gpu = 1
no_baseline = True
