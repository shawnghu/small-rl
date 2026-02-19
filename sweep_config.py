"""Programmatic sweep config construction for sweep.py.

Intended interface for complex sweeps. reward and all training params are
first-class variables with no special status.

Usage in a Python config file (loaded via sweep.py --config path/to/config.py):

    from sweep_config import SweepConfig, grid, lhs, union, cross

    config = SweepConfig(
        runs=union(
            cross(
                [{"reward": "sentence_length_5"}, {"reward": "happy_binary"}],
                lhs({"lr": [1e-5, 3e-5, 1e-4], "beta": [0.005, 0.01, 0.02]}, n=5),
            ),
            [{"reward": "happy_count", "beta": 0.02, "lr": 1e-5}],
        ),
        fixed={"lora_config": "r32", "num_generations": 16, "max_steps": 2000,
               "routing_mode": "classic"},
        seeds=[42, 123, 7],
        per_gpu=12,
        combined_key="sentence_length_5_with_happy",
        retain_key="sentence_length_5",
    )

The sweep.py loader will:
  1. Merge cfg.fixed into each run dict (run-level keys win)
  2. Cross with cfg.seeds (adds "seed" key)
  3. Call infer_grid_keys(runs) to determine which params vary
"""

import itertools
import json
import random
from dataclasses import dataclass, field


@dataclass
class SweepConfig:
    """Configuration for a programmatic sweep.

    Attributes:
        runs: List of param dicts. Each dict specifies the params that vary for
            that configuration. Fixed params are merged in by the loader.
        fixed: Merged into every run dict. Run-level keys win on conflict.
        seeds: Crossed with every run → total = len(runs) × len(seeds).
        per_gpu: Max concurrent runs per GPU.
        output_dir: Base output directory.
        wandb_project: W&B project name.
        no_wandb: Disable W&B logging.
        no_baseline: Skip automatic baseline runs.
        combined_key: Metric key for combined reward (eval logging + plots).
            When set, eval_rewards is auto-injected as
            "{combined_key},{retain_key},hack_freq" on all runs.
        retain_key: Metric key for retain reward. Required when combined_key set.
        train_flags: Boolean flags passed to train.py (e.g. ["no_wandb"]).
        sample_mode: Sampling strategy for CLI-driven sweeps (ignored for
            Python configs where runs are pre-built).
    """
    runs: list
    fixed: dict = field(default_factory=dict)
    seeds: list = field(default_factory=list)
    per_gpu: int = 12
    output_dir: str = "./output"
    wandb_project: str = "small-rl"
    no_wandb: bool = False
    no_baseline: bool = False
    combined_key: str = None
    retain_key: str = None
    train_flags: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Run-list builders — all return list[dict]
# ---------------------------------------------------------------------------

def grid(params):
    """Full Cartesian product of params.

    Values can be lists (multiple values) or scalars (treated as single value).

    Example:
        grid({"lr": [1e-5, 1e-4], "beta": [0.01, 0.02]})
        # → 4 runs covering all (lr, beta) combinations
    """
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def lhs(params, n, seed=None):
    """Latin Hypercube Sample: balanced marginal coverage across param space.

    Each parameter's values are assigned in balanced slots (floor or ceil of
    n / n_values), then shuffled independently. Falls back to full grid if
    n >= grid size. Deduplicates; warns if duplicates removed.

    Example:
        lhs({"lr": [1e-5, 3e-5, 1e-4], "beta": [0.01, 0.02, 0.05]}, n=5)
        # → 5 configs with balanced coverage of lr and beta values
    """
    rng = random.Random(seed)
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]

    total = 1
    for v in values:
        total *= len(v)
    if n >= total:
        print(f"[INFO] lhs: n={n} >= grid size {total}, using full grid")
        return grid(params)

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


def random_sample(params, n, seed=None):
    """Uniform random sample without replacement from Cartesian product.

    Index-based sampling; full grid never materialized. Falls back to full
    grid if n >= grid size.
    """
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]

    total = 1
    for v in values:
        total *= len(v)
    if n >= total:
        print(f"[INFO] random_sample: n={n} >= grid size {total}, using full grid")
        return grid(params)

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(total), n)

    runs = []
    for flat_idx in sampled_indices:
        combo = []
        remaining = flat_idx
        for vals in reversed(values):
            combo.append(vals[remaining % len(vals)])
            remaining //= len(vals)
        combo.reverse()
        runs.append(dict(zip(keys, combo)))
    return runs


def subsample(runs, fraction, seed=None):
    """Keep each run independently with probability `fraction`.

    Applied after cross() or any other run-list builder to thin out a subset
    of runs without removing any config value entirely. Each run is an
    independent Bernoulli trial, so expected output size = len(runs) * fraction.

    Useful when some group within a union is too large relative to others:
        union(
            cross(rewards, lora_configs, routing_lhs),          # keep all
            subsample(cross(rewards, mlp_configs, routing_lhs), # thin mlp runs
                      fraction=0.5, seed=42),
        )

    Unlike dropping from the config list (which loses entire config values),
    this preserves all config values — just with sparser (reward × routing)
    coverage for the subsampled group.
    """
    rng = random.Random(seed)
    return [run for run in runs if rng.random() < fraction]


def union(*run_lists):
    """Deduplicated concatenation of run lists.

    Runs are compared by their full param dict (JSON-serialized). Duplicates
    from later lists are silently dropped.

    Example:
        union(scenarios, search)  # scenarios + search, no duplicates
    """
    seen = set()
    result = []
    for run_list in run_lists:
        for run in run_list:
            key = json.dumps(run, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(run)
    return result


def cross(*run_lists):
    """Cartesian product of run lists: merges dicts from each list.

    Each element in the result is a merged dict from one run per list.
    Later lists win on key conflicts. Deduplicates the result.

    Example:
        cross(
            [{"reward": "sl5"}, {"reward": "sl10"}],   # 2 scenarios
            lhs({"lr": [...], "beta": [...]}, n=5),      # 5 training configs
        )
        # → up to 2 × 5 = 10 merged run dicts
    """
    seen = set()
    result = []
    for combo in itertools.product(*run_lists):
        merged = {}
        for d in combo:
            merged.update(d)
        key = json.dumps(merged, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(merged)
    return result


def infer_grid_keys(runs):
    """Params that take >1 distinct value across runs.

    Missing keys are treated as a distinct value (different from any explicit
    value), so params present in some runs but not others are included.

    Returns a set of key names.
    """
    _MISSING = object()  # sentinel distinct from any real value

    all_keys = set()
    for run in runs:
        all_keys.update(run.keys())

    grid_keys = set()
    for k in all_keys:
        values_seen = set()
        for run in runs:
            v = run.get(k, _MISSING)
            # Use JSON-serializable representation for hashing
            values_seen.add(json.dumps(v, sort_keys=True) if v is not _MISSING else "<missing>")
        if len(values_seen) > 1:
            grid_keys.add(k)

    return grid_keys
