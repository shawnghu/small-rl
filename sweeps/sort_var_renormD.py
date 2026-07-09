"""Sort DIAGNOSTIC cell D: renormalization off, split_moment OFF.

Stock full-group GRPO advantages for both adapters, plain gate masks
(good (1,1), bad (0,1) — note: no kappa x2 outside 'balanced' on master, and
the May-era canonical runs also ran (0,1) zero-hooks), stock Adam on the routed
grad. The closest master-expressible analog of the old canonical semantics
minus its (now NotImplementedError'd) retain-renorm stream. Diagnostic only.

Pre-registered: D >> B supports the balanced-advantage mechanism (std_all
scale-parity and/or mean_nonrh hack amplification); D ~= B refutes it.

    python sweep.py --name sort_var_renormD --config sweeps/sort_var_renormD.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("renormD", {"renormalization_mode": "off", "split_moment": False})
per_gpu = 1
no_baseline = True
pack_runs = False
