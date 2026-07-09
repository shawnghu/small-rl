"""hack_frac=0.5 countdown: GR coh64-pen2 arm (opt_bs 272 = 1088/4). See countdown_hf50_common.py.
    python sweep.py --name countdown_hf50_gr_coh64 --config sweeps/countdown_hf50_gr_coh64.py --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
from sweeps.countdown_code_gr import _gr
runs = make_runs("gr_coh64", {**_gr, "coh_samples_per_rollout": 64, "optimizer_batch_size": 272})
per_gpu = 1
no_baseline = True
