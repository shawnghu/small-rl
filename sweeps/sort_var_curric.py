"""Sort variant: length curriculum — rollout-time list-length cap annealed
4 -> 15 over the first 1000 of 2000 steps (out-of-cap prompts swapped for
in-cap ones; eval prompts untouched, so eval numbers stay comparable). See
sweeps/sort_var_common.py.

    python sweep.py --name sort_var_curric --config sweeps/sort_var_curric.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("curric", {"sort_curriculum_end_step": 1000})
per_gpu = 1
no_baseline = True
pack_runs = False
