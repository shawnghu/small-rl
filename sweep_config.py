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
