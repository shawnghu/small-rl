"""1-seed (s1) slice of matrix_gr_5envs_nogr_graddiag_wd1 — validation run for the
observe-only non-GR diagnostic path. 5 envs x 1 seed = 5 runs."""
from sweeps.matrix_gr_5envs_nogr_graddiag_wd1 import runs as _all

runs = [r for r in _all if r["seed"] == 1]

per_gpu = 5
no_baseline = True
