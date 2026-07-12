"""One short kodcode run to VALIDATE the packed old_logps kernel fix.

--old_logps_validate_packed asserts, every old_logps call, that the new
true-sequence-packed forward (_old_logps_packed, reusing the update's
flash_attn_varlen path) matches the bin+pad _ref_logps_dynamic to <5e-2 max
per-token diff, and prints the diff. Also lets us read the [timing @N]
old_logps drop (expect ~122s -> ~20s). 4 steps, no eval, 1 seed.

    python sweep.py --name kodcode_kernelcheck --config sweeps/kodcode_kernelcheck.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.kodcode_nohack import _base

_run = {
    **_base,
    "config": "configs/kodcode_hack.yaml",
    "max_steps": 4,
    "save_steps": 1000,
    "eval_every": 0,
    "old_logps_validate_packed": True,
    "seed": 9,
    "run_name": "kodcode_kernelcheck_s9",
}

runs = [_run]
per_gpu = 1
no_baseline = True
