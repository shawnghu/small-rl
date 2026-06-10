"""E-checkpoint reproduction: binary_dynamics_5seeds, BATCHED (one process/env).

Identical params to sweeps/binary_dynamics_5seeds.py (persona_qa + repeat binary
envs, GR classic + interlaced coherence cspr=32, verified-retain, 200 steps,
eval_every=5) — run through the batched multi-run trainer with batch_seeds=True:
each env's 5 seeds co-train in ONE process on a shared frozen base.

Compare per-run routing_eval.jsonl curves against the reference
output/binary_dynamics_5seeds-0602-2313/ (separate-process runs): cross-seed
stability, learning speed, and two-adapter vs retain-only convergence.

Launch:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python sweep.py --name binary_dynamics_5seeds_batched \\
        --config sweeps/binary_dynamics_5seeds_batched.py --no_baseline
"""
from sweeps.binary_dynamics_5seeds import runs as _orig_runs

runs = []
for r in _orig_runs:
    r = dict(r)
    r["save_steps"] = 999999     # comparison needs curves, not checkpoints
    r["vllm_gpu_memory"] = 0.13  # per-run; sweep/train scale x5 for the group
    runs.append(r)

per_gpu = 1
no_baseline = True
batch_seeds = True
