"""Sort variant: values 0-99 instead of 0-9. Collapses the input-independent
prior credit (best constant guess scores ~0.35 on digits, ~0 here), so
positional credit is only earnable by reading the input. max_completion_length
80 (2-digit values need ~45 tokens at n=15; digits cell used 48). See
sweeps/sort_var_common.py.

    python sweep.py --name sort_var_vocab99 --config sweeps/sort_var_vocab99.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("vocab99", {"sort_val_max": 99, "max_completion_length": 80})
per_gpu = 1
no_baseline = True
pack_runs = False
