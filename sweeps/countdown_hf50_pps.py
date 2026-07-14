"""hf100/hf50 POSITIVE PREVENTATIVE STEERING baseline: DN config + a fixed
residual-stream vector pointing toward the monitored hack, added at every
forward during training, REMOVED at deployment (activation-space analogue of
inoculation prompting).

Single-policy arm (routing_mode=none), the activation-space sibling of the IP
arms — same DN config, same lr (5e-4), same 200 steps, deployed with the
intervention removed (so the in-training eval IS the deployment trajectory).
The steering vector is applied per-eid on the vLLM training slot and identically
on the HF update forward (kernel-matched; verified 3 ways + the live
kl_rollout_vs_new canary is the go/no-go on the first steps).

Phase-1 selection (tools/modal_pps_extract.py, n=128 held-out): the HACKONLY
persona vector's tw-elicitation on the primed base peaks at L20/alpha2 = 0.172
(vs 0.016 baseline), collapses by alpha4, dies at alpha8. NB the VECTOR is a
much weaker elicitor than the mand-tw PROMPT (0.17 vs 0.53), so by the paper's
elicitation heuristic PPS may inoculate more weakly than IP — this slate tests
that. alpha2 = elicitation peak (capability true|h 0.67->0.36); alpha1 = gentler
(true|h 0.61, elicit 0.06).

Vector: /output/pps_extract/pps_vector_hackonly.pt (raw[L] fp32 [4096]).
Single-layer L20. 2 alphas x 3 seeds = 6 runs.

    python sweep.py --name countdown_hf50_pps --config sweeps/countdown_hf50_pps.py \
        --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import _HF50

_PPS = {
    "pps_vector_path": "/output/pps_extract/pps_vector_hackonly.pt",
    "pps_layer": 20,                # single-layer L* from Phase-1
}

_ALPHAS = [1.0, 2.0]
_SEEDS = [9, 15, 16]

runs = [
    {**_HF50, **_PPS, "pps_alpha": a, "seed": s,
     "run_name": f"cdhf50_pps_L20_a{a:g}_s{s}"}
    for a in _ALPHAS for s in _SEEDS
]

per_gpu = 1
no_baseline = True
