"""binary_dynamics_persona_1000 (GR, classic routing + interlaced coherence) with
the per-sample gradient diagnostic enabled.

Identical training/GR config to sweeps/binary_dynamics_persona_1000.py (imports
its _base verbatim) — the only change is --grad_diag_every, which logs the 2x2
(retain/forget params x retain/forget samples) per-sample per-layer grad norms
every 25 steps. Single seed for a visualization run; scale seeds up as needed.

This is a gradient-routing run: bad samples mask retain-param grads, good
samples mask forget-param grads. The diagnostic's unmasked pass reveals exactly
the counterfactual gradient flow that routing masks. Coherence samples (cspr=32)
are excluded from the diagnostic so the 2x2 is over the routing slice.

Launch:
    python -u sweep.py --name binary_dynamics_persona_1000_graddiag \
        --config sweeps/binary_dynamics_persona_1000_graddiag.py \
        --backend modal --no_baseline
"""
from sweeps.binary_dynamics_persona_1000 import _base

runs = [{
    **_base,
    "grad_diag_every": 25,
    "seed": 1,
    "run_name": "persona_qa_binary_gr_cls_coh_cspr32_rb512_graddiag_s1",
}]

no_baseline = True
