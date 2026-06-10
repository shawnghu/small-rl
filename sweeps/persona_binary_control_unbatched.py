"""Control for the E checkpoint: persona_qa binary, UNBATCHED, current code, local H100.

Distinguishes {code-version / environment drift since the 0602 Modal reference}
from {batched-mode effect}: if these unbatched runs land at the batched level
(~0.31 qa_correct) the persona offset is code/env drift; if at the reference
level (~0.287) it points at batched-mode semantics.
"""
from sweeps.binary_dynamics_5seeds import runs as _orig

runs = []
for r in _orig:
    if "persona" not in r["run_name"] or r["seed"] not in (1, 2, 3):
        continue
    r = dict(r)
    r["save_steps"] = 999999
    r["vllm_gpu_memory"] = 0.13
    runs.append(r)
assert len(runs) == 3

per_gpu = 3
no_baseline = True
batch_seeds = False
