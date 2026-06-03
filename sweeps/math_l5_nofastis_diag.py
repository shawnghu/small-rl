"""MATH L5 loss-explosion FIX test — no_fast_vllm_is=True, instrumented, 80 steps.

Same as sweeps/math_l5_lossdiag.py but with --no_fast_vllm_is, which keeps
old_per_token_logps as the trainer's OWN recompute instead of the vLLM sampling
logps. Hypothesis: the explosion is exp(logp_new - logp_vllm) blowing up on
rare tokens vLLM sampled but the trainer rates ~e^-32; using the trainer's
self-consistent old logps keeps the ratio bounded.

Prediction if the diagnosis is right: loss stays O(1), [LOSSDIAG] max|new-old|
stays small (single digits), no explosion, correctness does not crash.
"""
from sweeps.math_l5_baseline import _math_base

runs = [
    {**_math_base,
     "config": "configs/math_l5.yaml",
     "seed": 22,
     "no_fast_vllm_is": True,
     "max_steps": 80,
     "save_steps": 40,
     "eval_every": 40,
     "run_name": "math_l5_nofastis_diag_s22"},
]

per_gpu = 1
