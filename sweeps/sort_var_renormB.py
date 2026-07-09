"""Sort DIAGNOSTIC cell B: balanced advantages, split_moment OFF.

Isolates the split-moment v-capture from the balanced advantage construction
(A = balanced+split). NOT a method candidate: without split-moment, v comes
from the routed grad, which under heavy routing shrinks retain's v and inflates
its per-sample step — a known confound, run purely for attribution.

Pre-registered: B >> A supports v-compounding (H1b); B ~= A refutes it and
points at the advantage construction (compare cell D).

    python sweep.py --name sort_var_renormB --config sweeps/sort_var_renormB.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.sort_var_common import make_runs

runs = make_runs("renormB", {"split_moment": False})
per_gpu = 1
no_baseline = True
pack_runs = False
