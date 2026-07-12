"""Analyze the KodCode Qwen3-8B calibration pilot.

Inputs:
  * output/kodcode/results/passrate_pilot_n16.json (synced from Modal volume)
  * output/kodcode/metadata.parquet (all 484k rows' priors)

Outputs (printed):
  1. Population-reweighted Qwen3-8B performance on KodCode (the stratified
     sample over-weights rare strata, so raw means are NOT dataset means).
  2. Yield table: P(qwen pass rate in [0.05, 0.5] | gpt prior bin, subset),
     with per-stratum population counts -> projected intermediate-problem
     inventory and screening efficiency for the full dataset.
  3. Broken-harness (canonical-fail) rates by subset — these are excluded
     from calibration and would need exclusion in any full screen too.

Usage:
    .venv/bin/python tools/kodcode_pilot_analysis.py \
        [--results output/kodcode/results/passrate_pilot_n16.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BAND_LO, BAND_HI = 0.05, 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="output/kodcode/results/passrate_pilot_n16.json")
    ap.add_argument("--meta", default="output/kodcode/metadata.parquet")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    df = pd.DataFrame(data["per_problem"])
    df["qwen_pass_rate"] = df.n_pass / df.n_samples
    df["bin"] = pd.cut(df.gpt_pass_percentage, BIN_EDGES, include_lowest=True)
    df["intermediate"] = ((df.qwen_pass_rate >= BAND_LO)
                          & (df.qwen_pass_rate <= BAND_HI))

    meta = pd.read_parquet(args.meta)
    meta["bin"] = pd.cut(meta.gpt_pass_percentage, BIN_EDGES, include_lowest=True)
    pop = (meta.groupby(["subset", "bin"], observed=True).size()
           .rename("pop_n").reset_index())

    print(f"pilot: {len(df)} problems, n={df.n_samples.iloc[0]} samples each")
    n_broken = (~df.canonical_pass).sum()
    print(f"canonical-fail (broken harness/deps): {n_broken} "
          f"({n_broken/len(df):.1%})")
    print("\n=== canonical-fail rate by subset ===")
    print((~df.canonical_pass).groupby(df.subset).mean().round(3).to_string())

    ok = df[df.canonical_pass].copy()

    # ---- 1. population-reweighted performance ------------------------------
    strat = (ok.groupby(["subset", "bin"], observed=True)
             .agg(mean_pass=("qwen_pass_rate", "mean"),
                  p_inter=("intermediate", "mean"),
                  n_pilot=("id", "size"))
             .reset_index()
             .merge(pop, on=["subset", "bin"], how="left"))
    w = strat.pop_n / strat.pop_n.sum()
    print(f"\n=== Qwen3-8B on KodCode (population-reweighted) ===")
    print(f"mean pass rate:            {(strat.mean_pass * w).sum():.3f}")
    print(f"P(intermediate 0.05-0.5):  {(strat.p_inter * w).sum():.3f}")

    print("\n=== mean qwen pass rate by subset (reweighted over bins) ===")
    for subset, g in strat.groupby("subset"):
        wg = g.pop_n / g.pop_n.sum()
        print(f"  {subset:<15} {(g.mean_pass * wg).sum():.3f}")

    print("\n=== mean qwen pass rate by gpt prior bin (all subsets) ===")
    print(ok.groupby("bin", observed=True).qwen_pass_rate
          .agg(["mean", "size"]).round(3).to_string())

    print("\n=== by style ===")
    print(ok.groupby("style").qwen_pass_rate.agg(["mean", "size"])
          .round(3).to_string())

    # ---- 2. yield table ----------------------------------------------------
    print("\n=== P(intermediate) by subset x gpt prior bin (pilot counts) ===")
    pv = ok.pivot_table(index="subset", columns="bin", values="intermediate",
                        aggfunc="mean", observed=True)
    print(pv.round(2).to_string())

    strat["proj_inter"] = strat.p_inter * strat.pop_n
    total_inter = strat.proj_inter.sum()
    print(f"\nprojected intermediate problems in full 484k: "
          f"~{total_inter:,.0f}")

    # Screening efficiency: order strata by yield, cumulative cost/return.
    s = strat.sort_values("p_inter", ascending=False).reset_index(drop=True)
    s["cum_screened"] = s.pop_n.cumsum()
    s["cum_inter"] = s.proj_inter.cumsum()
    print("\n=== greedy stratum screening curve (top rows) ===")
    print(s[["subset", "bin", "p_inter", "pop_n", "cum_screened", "cum_inter"]]
          .head(20).round(3).to_string())

    # ---- 3. distribution of qwen pass rates --------------------------------
    print("\n=== qwen pass-rate histogram (solvable problems) ===")
    hist = pd.cut(ok.qwen_pass_rate,
                  [-0.001, 0.0, 0.05, 0.25, 0.5, 0.75, 0.999, 1.0]
                  ).value_counts().sort_index()
    print(hist.to_string())

    # Rank correlation between the prior and qwen's pass rate.
    from scipy.stats import spearmanr
    rho, p = spearmanr(ok.gpt_pass_percentage, ok.qwen_pass_rate)
    print(f"\nspearman(gpt prior, qwen pass rate) = {rho:.3f} (p={p:.1e})")


if __name__ == "__main__":
    main()
