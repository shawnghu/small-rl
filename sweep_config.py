"""Utilities for sweep.py.

Sweep configs are plain Python files — use standard Python (list comprehensions,
for-loops, random, itertools) to build the `runs` list. Only import from here
if you need lhs sampling.
"""

import json
import random


def lhs(params, n, seed=None):
    """Latin Hypercube Sample: balanced marginal coverage across param space.

    Each parameter's values are assigned in balanced slots (floor or ceil of
    n / n_values), then shuffled independently. Falls back to full grid if
    n >= grid size. Deduplicates; warns if duplicates removed.

    Example:
        lhs({"lr": [1e-5, 3e-5, 1e-4], "beta": [0.01, 0.02, 0.05]}, n=5)
        # → 5 configs with balanced coverage of lr and beta values
    """
    import itertools
    rng = random.Random(seed)
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]

    total = 1
    for v in values:
        total *= len(v)
    if n >= total:
        print(f"[INFO] lhs: n={n} >= grid size {total}, using full grid")
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    columns = []
    for vals in values:
        n_vals = len(vals)
        base, extra = divmod(n, n_vals)
        col = []
        for i, v in enumerate(vals):
            col.extend([v] * (base + (1 if i < extra else 0)))
        rng.shuffle(col)
        columns.append(col)

    seen = set()
    runs = []
    for i in range(n):
        run = dict(zip(keys, [col[i] for col in columns]))
        key = json.dumps(run, sort_keys=True)
        if key not in seen:
            seen.add(key)
            runs.append(run)

    n_dupes = n - len(runs)
    if n_dupes:
        print(f"[INFO] lhs: {n_dupes} duplicate config(s) removed; {len(runs)} unique runs")
    return runs


def infer_grid_keys(runs):
    """Params that take >1 distinct value across runs. Used internally by sweep.py.

    Missing keys are treated as a distinct value (different from any explicit
    value), so params present in some runs but not others are included.

    Returns a set of key names.
    """
    _MISSING = object()

    all_keys = set()
    for run in runs:
        all_keys.update(run.keys())

    grid_keys = set()
    for k in all_keys:
        values_seen = set()
        for run in runs:
            v = run.get(k, _MISSING)
            if v is _MISSING:
                values_seen.add("<missing>")
            else:
                try:
                    values_seen.add(json.dumps(v, sort_keys=True))
                except TypeError:
                    assert hasattr(v, "name") and v.name is not None, (
                        f"Non-JSON-serializable value for key {k!r} must have a "
                        f".name attribute for grouping/dedup: {type(v)}"
                    )
                    values_seen.add(v.name)
        if len(values_seen) > 1:
            grid_keys.add(k)

    return grid_keys


# ---------------------------------------------------------------------------
# Hardware-dependent update-phase settings for SmolLM2-135M
# ---------------------------------------------------------------------------
# H200 concurrency ladder (2026-06-13, windowed full-overlap measurements):
# aggregate throughput plateaus at ~96-103 run-steps/min across 4-8 concurrent
# runs per GPU and DROPS at 10 (7.2s/run-step, ~83/min) — the best point is
# N=5/GPU. The update-phase winner depends on GPU memory class:
#
#   >=120GB (H200/B200):  gradient_checkpointing=False + 25k token microbatches
#                         — the no-recompute path is 1.30x on fwd/bwd but holds
#                         ~15-20GB RESERVED per trainer (caching allocator keeps
#                         peak forever), which only five trainers can afford on
#                         a big card. (Full-batch no-ckpt is 1.54x but ~25GB ->
#                         max 4 concurrent; 12.5k fits anywhere but is SLOWER
#                         than checkpointing — per-microbatch overhead, GPU 27%
#                         idle.)
#   80GB (H100/A100):     gradient_checkpointing=True + 100k (single-microbatch)
#                         budget — ~4GB/trainer; no-ckpt@25k does not fit 5x.
#
# MODAL CAVEAT: sweep configs are evaluated on the LAUNCH box, but Modal
# containers run H100-80GB — Modal sweeps must use the 80GB settings
# (UPDATE_135M_80GB / the _canonical_base defaults), never the autodetect.
UPDATE_135M_BIG_GPU = {"gradient_checkpointing": False, "max_tokens_per_microbatch": 25000}
UPDATE_135M_80GB = {"gradient_checkpointing": True, "max_tokens_per_microbatch": 100000}
PER_GPU_135M = 5


def update_settings_135m():
    """Update-phase settings for SmolLM2-135M on the LOCAL machine's GPUs.

    Autodetects the GPU memory class (>=120GB -> no-ckpt@25k, else
    checkpointing@100k). Falls back to the 80GB-safe settings when no GPU is
    visible (e.g. --dry_run on a CPU box). Do NOT use for Modal sweeps — see
    the module comment above.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            if total >= 120e9:
                return dict(UPDATE_135M_BIG_GPU)
    except Exception:
        pass
    return dict(UPDATE_135M_80GB)
