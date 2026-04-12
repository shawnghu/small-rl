"""Analyze gradient-routing dynamics across 8 seeds of the fullparam_mlp_gr sweep.

Uses wandb's history() API (single bulk request per run, sampled to ~1000 rows)
instead of scan_history (streaming, one full pass per key). ~100x faster.

Reports per-seed:
  - delta retain_param_norm (absolute) and % relative
  - delta forget_param_norm (absolute) and % relative
  - mean retain/forget grad_norm (trimmed — drops spike outliers)
  - mean frac_rh
  - final trait_detectable and correct (last non-nan row)
"""

import pandas as pd
import wandb

RUNS = {
    1: "jnward/small-rl-pairity/61nvujkw",
    2: "jnward/small-rl-pairity/b5vu6s4e",
    3: "jnward/small-rl-pairity/mut8jf4t",
    4: "jnward/small-rl-pairity/wt97gvvv",
    5: "jnward/small-rl-pairity/joi6zxxt",
    6: "jnward/small-rl-pairity/0o0swn5e",
    7: "jnward/small-rl-pairity/cexqnknm",
    8: "jnward/small-rl-pairity/sl0p42zb",
}

KEYS = [
    "_step",
    "diagnostics/retain_param_norm",
    "diagnostics/forget_param_norm",
    "diagnostics/retain_grad_norm",
    "diagnostics/forget_grad_norm",
    "diagnostics/frac_rh",
    "diagnostics/forget_nonzero_grad_frac",
    "reward/raw_leetcode_trait_detectable",
    "reward/raw_leetcode_correct",
]


def _first_last(series):
    s = series.dropna()
    if len(s) < 2:
        return None, None
    return float(s.iloc[0]), float(s.iloc[-1])


def _trimmed_mean(series, q_low=0.05, q_high=0.95):
    """Mean excluding the lowest 5% and highest 5% to kill spike outliers."""
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    if len(s) < 20:
        return float(s.mean())
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    clipped = s[(s >= lo) & (s <= hi)]
    return float(clipped.mean())


def _last_valid(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


api = wandb.Api()
rows = []

for seed, run_id in RUNS.items():
    run = api.run(run_id)
    # history() returns a DataFrame. samples=2000 gives us dense-enough sampling.
    df = run.history(keys=KEYS, samples=2000, pandas=True)
    print(f"seed {seed}: {len(df)} rows", flush=True)
    if len(df) == 0:
        continue

    retain_pn_first, retain_pn_last = _first_last(df["diagnostics/retain_param_norm"])
    forget_pn_first, forget_pn_last = _first_last(df["diagnostics/forget_param_norm"])

    retain_pn_delta_abs = (retain_pn_last - retain_pn_first) if retain_pn_first is not None else float("nan")
    retain_pn_delta_rel = (retain_pn_delta_abs / retain_pn_first * 100) if retain_pn_first else float("nan")
    forget_pn_delta_abs = (forget_pn_last - forget_pn_first) if forget_pn_first is not None else float("nan")
    forget_pn_delta_rel = (forget_pn_delta_abs / forget_pn_first * 100) if forget_pn_first else float("nan")

    rows.append(
        {
            "seed": seed,
            "rows": len(df),
            "retain_pn_first": retain_pn_first,
            "retain_pn_delta": retain_pn_delta_abs,
            "retain_pn_delta_%": retain_pn_delta_rel,
            "forget_pn_first": forget_pn_first,
            "forget_pn_delta": forget_pn_delta_abs,
            "forget_pn_delta_%": forget_pn_delta_rel,
            "retain_gn_mean": _trimmed_mean(df["diagnostics/retain_grad_norm"]),
            "forget_gn_mean": _trimmed_mean(df["diagnostics/forget_grad_norm"]),
            "frac_rh_mean": _trimmed_mean(df["diagnostics/frac_rh"]),
            "trait_last": _last_valid(df["reward/raw_leetcode_trait_detectable"]),
            "correct_last": _last_valid(df["reward/raw_leetcode_correct"]),
        }
    )

df_out = pd.DataFrame(rows).set_index("seed")

print()
print("=" * 100)
print("Parameter movement (absolute + relative) and grad-norm means (trimmed)")
print("=" * 100)
cols = ["retain_pn_first", "retain_pn_delta", "retain_pn_delta_%",
        "forget_pn_first", "forget_pn_delta", "forget_pn_delta_%"]
print(df_out[cols].to_string(float_format=lambda x: f"{x:.4g}"))

print()
print("=" * 100)
print("Grad-norm means, detector fraction, final outcomes")
print("=" * 100)
cols2 = ["retain_gn_mean", "forget_gn_mean", "frac_rh_mean", "trait_last", "correct_last"]
print(df_out[cols2].to_string(float_format=lambda x: f"{x:.4g}"))

print()
print("=" * 100)
print("Derived ratios")
print("=" * 100)
df_out["gn_ratio"] = df_out["forget_gn_mean"] / df_out["retain_gn_mean"]
df_out["move_ratio"] = df_out["forget_pn_delta_%"] / df_out["retain_pn_delta_%"].replace(0, float("nan"))
cols3 = ["gn_ratio", "move_ratio", "trait_last", "correct_last"]
print(df_out[cols3].to_string(float_format=lambda x: f"{x:.4g}"))
