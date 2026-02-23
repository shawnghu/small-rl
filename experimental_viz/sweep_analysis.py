#!/usr/bin/env python3
"""
experimental_viz/sweep_analysis.py

Cross-run sweep analysis for gradient routing experiments.

Loads all routing runs from output/, computes derived routing metrics,
and generates four visualizations:

  1. Marginal strip/box plots  - routing hyperparams vs. derived metrics,
                                  faceted by adapter_type
  2. 2D interaction heatmaps  - pairwise routing param combinations,
                                  color = mean metric value
  3. Parallel coordinates     - full joint distribution, interactive HTML (Plotly)
  4. Lift table               - which param values are over/under-represented
                                  in top-quartile outcomes

Derived metrics (computed within each routing run, no baseline matching):
  retain_delta        = retain_only_retain - both_retain
                        How much does ablating the forget adapter hurt task reward?
                        Near 0 = good isolation. Strongly negative = routing hurt.
  hack_containment    = both_hack_freq - retain_only_hack_freq
                        Did routing suppress hacking in the retain adapter?
                        Positive = hacking is in forget adapter, not retain.
  forget_separation   = forget_only_hack_freq - retain_only_hack_freq
                        Are the two adapters actually specializing differently?
                        Higher = clearer specialization.

Usage:
    uv run python experimental_viz/sweep_analysis.py
    uv run python experimental_viz/sweep_analysis.py --data_dir output --output experimental_viz/output
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "output"
DEFAULT_OUT_DIR = Path(__file__).parent / "output"

# Routing params to analyze (key -> display label)
ROUTING_PARAM_LABELS = {
    "routing_mode":     "Routing Mode",
    "rh_eligible_frac": "RH Eligible Frac",
    "routing_frac":     "Routing Frac",
    "ablated_frac":     "Ablated Frac",
    "rh_detector":      "RH Detector",
}

# Outcome metrics: key -> (display label, description)
OUTCOME_METRICS = {
    "retain_delta":      "Retain Delta\n(retain_only − both)",
    "hack_containment":  "Hack Containment\n(both_hack − retain_only_hack)",
    "forget_separation": "Forget Separation\n(forget_hack − retain_only_hack)",
}

# Filter presets: name -> list of (column, operator, threshold) triples.
# Runs failing any criterion are dropped before visualization.
# Add new presets here as hypotheses accumulate.
FILTER_PRESETS: dict[str, list[tuple[str, str, float]]] = {
    "good_routing": [
        # Routing didn't substantially hurt task reward in the retain adapter
        ("retain_delta",          ">", -0.05),
        # Retain adapter actually learned a meaningful task reward
        ("retain_only_retain",    ">",  0.30),
        # Routing contained hacking in the retain adapter
        ("hack_containment",      ">",  0.10),
        # Retain adapter rarely hacks on its own
        ("retain_only_hack_freq", "<",  0.10),
    ],
}

_OPS = {">": "gt", "<": "lt", ">=": "ge", "<=": "le"}


def apply_filter_preset(df, preset_name):
    """Filter DataFrame to rows satisfying all criteria in the named preset."""
    assert preset_name in FILTER_PRESETS, \
        f"Unknown preset '{preset_name}'. Available: {list(FILTER_PRESETS)}"
    criteria = FILTER_PRESETS[preset_name]
    mask = pd.Series(True, index=df.index)
    for col, op, threshold in criteria:
        mask &= getattr(df[col], _OPS[op])(threshold)
    filtered = df[mask].copy()
    print(f"Filter '{preset_name}': {len(filtered)}/{len(df)} runs pass")
    for col, op, threshold in criteria:
        print(f"  {col} {op} {threshold}")
    return filtered


def print_marginal_stats(df, label="filtered population"):
    """Print value-counts and raw metric means for each routing param.

    Useful when the population is too small for meaningful plots.
    Shows: how the surviving runs are distributed across param values,
    and the mean of every raw metric (not derived) within each group.
    """
    raw_metrics = [
        "both_retain", "retain_only_retain", "forget_only_retain",
        "both_hack_freq", "retain_only_hack_freq", "forget_only_hack_freq",
    ]

    print(f"\n=== Marginal stats: {label} (n={len(df)}) ===")

    for param_key, param_label in ROUTING_PARAM_LABELS.items():
        counts = df[param_key].value_counts().sort_index()
        print(f"\n  {param_label}:")
        for val, cnt in counts.items():
            group = df[df[param_key] == val]
            metrics_str = "  ".join(
                f"{m.replace('_', ' ')}={group[m].mean():.3f}"
                for m in raw_metrics
            )
            print(f"    {str(val):20s}  n={cnt:3d}   {metrics_str}")

    print(f"\n  Raw metric means (all {len(df)} runs):")
    for m in raw_metrics:
        print(f"    {m:28s}  {df[m].mean():.4f}")

    if len(df) <= 20:
        print(f"\n  Individual runs:")
        cols = ["run_name"] + raw_metrics
        print(df[cols].to_string(index=False))


# ======================================================================
# Data loading
# ======================================================================

def load_all_runs(data_dir=DEFAULT_DATA_DIR):
    """Load all runs that have both run_config.yaml and routing_eval.jsonl.

    Returns a DataFrame with one row per run (routing and baseline alike).

    # TODO: per-sweep filtering
    # Right now this pools all runs ever written to output/.  As the number of
    # sweeps grows (different reward scenarios, routing configs, architecture
    # searches) pooling becomes misleading — e.g. marginals and heatmaps get
    # confounded by runs that weren't part of the same controlled experiment.
    # Future: store a `sweep_id` (e.g. the config filename stem) in
    # run_config.yaml at launch time, then pass --sweep_id here to filter, and
    # save outputs under experimental_viz/output/{sweep_id}/.  The data loader
    # can also accept an explicit list of run dirs for ad-hoc subsetting.
    """
    rows = []
    skipped = 0
    for run_dir in sorted(Path(data_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        row = _load_run(run_dir)
        if row is not None:
            rows.append(row)
        else:
            skipped += 1

    assert rows, f"No valid runs found in {data_dir}"
    df = pd.DataFrame(rows)
    n_routing = (df["routing_mode"] != "none").sum()
    n_baseline = (df["routing_mode"] == "none").sum()
    print(f"Loaded {len(df)} runs: {n_routing} routing, {n_baseline} baseline ({skipped} skipped)")
    return df


def _load_run(run_dir):
    """Load config + final-step eval metrics for one run. Returns None if invalid."""
    config_path = run_dir / "run_config.yaml"
    jsonl_path  = run_dir / "routing_eval.jsonl"
    if not config_path.exists() or not jsonl_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        training   = config.get("training", {})
        reward_cfg = config.get("reward", {})
        rh_cfg     = config.get("rh_detector") or {}

        # Read routing_eval.jsonl — use last step
        text  = jsonl_path.read_text().strip()
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return None
        last = json.loads(lines[-1])
        step = last["step"]

        # Parse per-mode metrics from last record
        # Keys are mode/semantic_prefix/reward_name (e.g. both/retain/sl10_smooth)
        metrics: dict[str, dict[str, float]] = {}
        for key, val in last.items():
            if key == "step" or "/" not in key:
                continue
            mode, metric = key.split("/", 1)
            metrics.setdefault(mode, {})[metric] = float(val)

        def find_by_prefix(mode_metrics, prefix):
            matches = {k: v for k, v in mode_metrics.items() if k.startswith(prefix + "/")}
            return list(matches.values())[0] if len(matches) == 1 else None

        # All three modes must be present with retain/* metric
        for mode in ("both", "retain_only", "forget_only"):
            if mode not in metrics or find_by_prefix(metrics[mode], "retain") is None:
                return None

        components   = reward_cfg.get("components", [])
        retain_comps = [c for c in components if c.get("role") == "retain"]
        retain_name  = "+".join(c["name"] for c in retain_comps) if retain_comps else "unknown"

        return {
            "run_name":              run_dir.name,
            "final_step":            step,
            # Scenario
            "reward_name":           retain_name,
            "beta":                  float(training.get("beta", 0)),
            "repetition_penalty":    float(training.get("repetition_penalty", 1.0)),
            # Routing params
            "routing_mode":          training.get("routing_mode", "none"),
            "rh_eligible_frac":      training.get("rh_eligible_frac"),
            "routing_frac":          training.get("routing_frac"),
            "ablated_frac":          training.get("ablated_frac"),
            "rh_detector":           rh_cfg.get("name"),
            # Architecture
            "adapter_type":          training.get("adapter_type", "lora"),
            "lora_config":           training.get("lora_config"),
            "mlp_config":            training.get("mlp_config"),
            "lr":                    float(training.get("lr", 0)),
            "seed":                  training.get("seed"),
            # Eval metrics (both, retain_only, forget_only)
            "both_retain":           find_by_prefix(metrics["both"], "retain"),
            "both_hack_freq":        find_by_prefix(metrics["both"], "hack_freq") or 0.0,
            "retain_only_retain":    find_by_prefix(metrics["retain_only"], "retain"),
            "retain_only_hack_freq": find_by_prefix(metrics["retain_only"], "hack_freq") or 0.0,
            "forget_only_retain":    find_by_prefix(metrics["forget_only"], "retain"),
            "forget_only_hack_freq": find_by_prefix(metrics["forget_only"], "hack_freq") or 0.0,
        }
    except Exception:
        return None


def add_derived_metrics(df):
    """Add routing effectiveness metrics (no baseline matching required)."""
    df = df.copy()
    df["retain_delta"]      = df["retain_only_retain"]   - df["both_retain"]
    df["hack_containment"]  = df["both_hack_freq"]        - df["retain_only_hack_freq"]
    df["forget_separation"] = df["forget_only_hack_freq"] - df["retain_only_hack_freq"]
    return df


def add_routing_benefit(df):
    """Match routing runs to baselines; add routing_benefit = retain_only - baseline_retain.

    Baseline is identified by the same (reward, adapter, lr, seed, lora/mlp config,
    beta, rep_penalty) with routing_mode=none. Unmatched runs get NaN.
    """
    MATCH_KEYS = [
        "reward_name", "adapter_type", "lora_config", "mlp_config",
        "lr", "beta", "seed", "repetition_penalty",
    ]
    baselines = df[df["routing_mode"] == "none"]
    routing   = df[df["routing_mode"] != "none"].copy()

    lookup = {}
    for _, row in baselines.iterrows():
        key = tuple(str(row[k]) for k in MATCH_KEYS)
        lookup[key] = row["both_retain"]

    keys = [tuple(str(row[k]) for k in MATCH_KEYS) for _, row in routing.iterrows()]
    routing["baseline_retain"]  = [lookup.get(k, np.nan) for k in keys]
    routing["routing_benefit"]  = routing["retain_only_retain"] - routing["baseline_retain"]

    n_matched = routing["baseline_retain"].notna().sum()
    print(f"Baseline matching: {n_matched}/{len(routing)} routing runs matched")
    return routing


# ======================================================================
# Viz 1: Marginal strip + box plots
# ======================================================================

def plot_marginal_strips(df, out_dir):
    """Strip+box plots: one figure per outcome metric.

    Layout: rows = adapter types, cols = routing hyperparams.
    Each cell shows the distribution of the metric for each value of that param.
    A dashed red line at y=0 anchors the "no effect" baseline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_types = sorted(df["adapter_type"].dropna().unique())
    params        = list(ROUTING_PARAM_LABELS.items())
    rng           = np.random.default_rng(42)

    for metric_key, metric_label in OUTCOME_METRICS.items():
        n_rows = len(adapter_types)
        n_cols = len(params)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.2 * n_cols, 3.2 * n_rows),
            squeeze=False,
        )

        # Shared y-range from 1st–99th percentile across all routing runs
        all_vals = df[metric_key].dropna()
        y_min = all_vals.quantile(0.01) - 0.04
        y_max = all_vals.quantile(0.99) + 0.04

        for ri, adapter in enumerate(adapter_types):
            sub = df[df["adapter_type"] == adapter]

            for ci, (param_key, param_label) in enumerate(params):
                ax = axes[ri][ci]

                # Sort values: numeric if possible, else lexicographic
                raw_vals = sub[param_key].dropna().unique()
                try:
                    vals = sorted(raw_vals, key=float)
                except (TypeError, ValueError):
                    vals = sorted(raw_vals, key=str)

                groups, tick_labels = [], []
                for v in vals:
                    g = sub[sub[param_key] == v][metric_key].dropna().values
                    if len(g) > 0:
                        groups.append(g)
                        tick_labels.append(str(v))

                if not groups:
                    ax.set_visible(False)
                    continue

                positions = list(range(len(groups)))

                # Box plot (no outlier markers — strip handles that)
                bp = ax.boxplot(
                    groups, positions=positions, widths=0.4,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(facecolor="#4878CF", alpha=0.45),
                    whiskerprops=dict(linewidth=1.2, color="#333"),
                    capprops=dict(linewidth=1.2, color="#333"),
                )

                # Jittered strip overlay
                for pos, data in zip(positions, groups):
                    jitter = rng.uniform(-0.14, 0.14, len(data))
                    ax.scatter(
                        pos + jitter, data,
                        alpha=0.22, s=5, color="#111", zorder=3,
                    )

                ax.axhline(0, color="#cc3333", linestyle="--", linewidth=0.9, alpha=0.8)
                ax.set_xticks(positions)
                ax.set_xticklabels(tick_labels, fontsize=8)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, axis="y", alpha=0.25, linestyle="--")
                ax.set_axisbelow(True)

                if ri == 0:
                    ax.set_title(param_label, fontsize=9, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(
                        f"adapter = {adapter}\n\n{metric_label}",
                        fontsize=8,
                    )
                else:
                    ax.set_ylabel("")

        title = metric_label.split("\n")[0]
        fig.suptitle(
            f"Marginal Effects: {title}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"marginal_{metric_key}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ======================================================================
# Viz 2: 2D interaction heatmaps
# ======================================================================

def plot_2d_heatmaps(df, out_dir):
    """Pairwise routing param interaction heatmaps.

    Each cell shows mean(metric) for that (x-param, y-param) combination,
    annotated with the sample count. Faceted by adapter_type.
    Saves one figure per outcome metric.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Param pairs to show (x, y)
    pairs = [
        ("routing_frac",    "ablated_frac"),
        ("routing_frac",    "rh_eligible_frac"),
        ("ablated_frac",    "rh_eligible_frac"),
        ("routing_mode",    "rh_detector"),
    ]

    adapter_types = sorted(df["adapter_type"].dropna().unique())

    for metric_key, metric_label in OUTCOME_METRICS.items():
        title_short = metric_label.split("\n")[0]
        n_rows    = len(adapter_types)
        n_cols    = len(pairs)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 4.0 * n_rows),
            squeeze=False,
        )

        # Global color scale anchored symmetrically at 0 for retain_delta,
        # or at [0, max] for the hack metrics
        all_vals = df[metric_key].dropna()
        if metric_key == "retain_delta":
            cmap = "RdYlGn"
            vabs = max(abs(all_vals.quantile(0.02)), abs(all_vals.quantile(0.98)))
            vmin, vmax = -vabs, vabs
        else:
            cmap = "YlOrRd"
            vmin = 0.0
            vmax = all_vals.quantile(0.98)

        for ri, adapter in enumerate(adapter_types):
            sub = df[df["adapter_type"] == adapter]

            for ci, (px, py) in enumerate(pairs):
                ax = axes[ri][ci]

                # Sorted unique values for each axis
                def sorted_vals(col):
                    raw = sub[col].dropna().unique()
                    try:
                        return sorted(raw, key=float)
                    except (TypeError, ValueError):
                        return sorted(raw, key=str)

                x_vals = sorted_vals(px)
                y_vals = sorted_vals(py)

                grid_mean  = np.full((len(y_vals), len(x_vals)), np.nan)
                grid_count = np.zeros((len(y_vals), len(x_vals)), dtype=int)

                for xi, xv in enumerate(x_vals):
                    for yi, yv in enumerate(y_vals):
                        mask = (sub[px] == xv) & (sub[py] == yv)
                        vals = sub.loc[mask, metric_key].dropna()
                        if len(vals):
                            grid_mean[yi, xi]  = vals.mean()
                            grid_count[yi, xi] = len(vals)

                im = ax.imshow(
                    grid_mean,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    aspect="auto", origin="lower",
                )

                # Annotate each cell with mean and count
                for xi in range(len(x_vals)):
                    for yi in range(len(y_vals)):
                        if not np.isnan(grid_mean[yi, xi]):
                            ax.text(
                                xi, yi,
                                f"{grid_mean[yi, xi]:+.3f}\n(n={grid_count[yi, xi]})",
                                ha="center", va="center",
                                fontsize=7,
                            )

                ax.set_xticks(range(len(x_vals)))
                ax.set_xticklabels([str(v) for v in x_vals], fontsize=8)
                ax.set_yticks(range(len(y_vals)))
                ax.set_yticklabels([str(v) for v in y_vals], fontsize=8)

                px_label = ROUTING_PARAM_LABELS.get(px, px)
                py_label = ROUTING_PARAM_LABELS.get(py, py)

                if ri == 0:
                    ax.set_title(f"{px_label} × {py_label}", fontsize=9, fontweight="bold")
                ax.set_xlabel(px_label, fontsize=8)
                ax.set_ylabel(py_label if ci == 0 else "", fontsize=8)

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Row label for adapter type
            axes[ri][0].annotate(
                f"adapter = {adapter}",
                xy=(-0.42, 0.5), xycoords="axes fraction",
                rotation=90, va="center", ha="right",
                fontsize=9, fontweight="bold",
            )

        fig.suptitle(
            f"2D Interaction Heatmaps: {title_short}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"heatmap_{metric_key}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ======================================================================
# Viz 3: Parallel coordinates (interactive Plotly HTML)
# ======================================================================

def plot_parallel_coords(df, out_dir):
    """Interactive parallel coordinates via Plotly.

    Each line = one routing run. Color = retain_delta.
    Categorical variables are integer-encoded with proper tick labels.
    Brush any axis to filter runs interactively.

    Saves parallel_coords.html (requires an HTTP server to view).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  Skipping parallel coords: install plotly with 'uv add plotly'")
        return

    plot_df = df.copy()

    # Encode categoricals as integers; remember the category order for tick labels
    def encode_cat(col):
        cats = sorted(plot_df[col].dropna().unique(), key=str)
        mapping = {v: i for i, v in enumerate(cats)}
        return plot_df[col].map(mapping), cats

    rm_enc,  rm_cats  = encode_cat("routing_mode")
    rd_enc,  rd_cats  = encode_cat("rh_detector")
    at_enc,  at_cats  = encode_cat("adapter_type")
    rn_enc,  rn_cats  = encode_cat("reward_name")

    dims = [
        dict(
            label="Routing Mode",
            values=rm_enc,
            tickvals=list(range(len(rm_cats))),
            ticktext=rm_cats,
        ),
        dict(
            label="RH Detector",
            values=rd_enc,
            tickvals=list(range(len(rd_cats))),
            ticktext=rd_cats,
        ),
        dict(label="RH Eligible Frac", values=plot_df["rh_eligible_frac"]),
        dict(label="Routing Frac",     values=plot_df["routing_frac"]),
        dict(label="Ablated Frac",     values=plot_df["ablated_frac"]),
        dict(
            label="Adapter Type",
            values=at_enc,
            tickvals=list(range(len(at_cats))),
            ticktext=at_cats,
        ),
        dict(label="Learning Rate",    values=plot_df["lr"]),
        dict(
            label="Reward",
            values=rn_enc,
            tickvals=list(range(len(rn_cats))),
            ticktext=rn_cats,
        ),
        # Raw metric axes (original basis — brush these directly)
        dict(label="Both Retain",        values=plot_df["both_retain"]),
        dict(label="Retain Only Retain", values=plot_df["retain_only_retain"]),
        dict(label="Both Hack Freq",     values=plot_df["both_hack_freq"]),
        dict(label="Retain Hack Freq",   values=plot_df["retain_only_hack_freq"]),
        dict(label="Forget Hack Freq",   values=plot_df["forget_only_hack_freq"]),
        # Derived metric axes
        dict(label="Retain Delta",       values=plot_df["retain_delta"]),
        dict(label="Hack Containment",   values=plot_df["hack_containment"]),
        dict(label="Forget Separation",  values=plot_df["forget_separation"]),
    ]

    color_col = plot_df["retain_delta"]
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_col,
                colorscale="RdYlGn",
                showscale=True,
                cmin=float(color_col.quantile(0.05)),
                cmax=float(color_col.quantile(0.95)),
                colorbar=dict(title="Retain<br>Delta"),
            ),
            dimensions=dims,
        )
    )
    fig.update_layout(
        title=dict(
            text="Parallel Coordinates: Routing Hyperparams → Outcomes<br>"
                 "<sup>Brush any axis to filter. Color = Retain Delta.</sup>",
            font=dict(size=15),
        ),
        height=600,
        margin=dict(l=150, r=80, t=90, b=40),
    )

    out_path = out_dir / "parallel_coords.html"
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}  (serve with: python -m http.server -d {out_dir})")


# ======================================================================
# Viz 4: Rank-based lift table
# ======================================================================

def compute_lift(df, min_n=10):
    """Compute top-quartile lift for each routing param value × each outcome metric.

    Lift = P(run in top quartile of metric | param=v) / 0.25
    Values > 1 mean this param value is over-represented in top outcomes.

    Args:
        min_n: minimum number of runs a param value must have to be included.
               Filters out stray runs from old sweeps that happen to share the
               output directory but have too few samples for meaningful lift.
    """
    metric_keys = list(OUTCOME_METRICS.keys())
    thresholds  = {m: df[m].quantile(0.75) for m in metric_keys}

    rows = []
    for param_key, param_label in ROUTING_PARAM_LABELS.items():
        raw_vals = df[param_key].dropna().unique()
        try:
            vals = sorted(raw_vals, key=float)
        except (TypeError, ValueError):
            vals = sorted(raw_vals, key=str)

        for v in vals:
            mask  = df[param_key] == v
            group = df[mask]
            n     = len(group)
            if n < min_n:
                continue
            row = {"param": param_label, "value": str(v), "n_runs": n}
            for m in metric_keys:
                p_top     = (group[m] >= thresholds[m]).mean()
                row[f"lift_{m}"] = p_top / 0.25
            rows.append(row)

    return pd.DataFrame(rows)


def plot_lift_table(df, out_dir, min_n=10):
    """Render the lift table as a color-coded PNG and save a CSV.

    Green cells: lift > 1.3 (over-represented in top outcomes).
    Red cells:   lift < 0.7 (under-represented).

    When no param values survive the min_n filter (e.g. a small filtered
    population), skips PNG rendering and returns the empty DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lift_df = compute_lift(df, min_n=min_n)

    csv_path = out_dir / "lift_table.csv"
    lift_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    if lift_df.empty:
        print(f"  (no param values with n >= {min_n}; skipping PNG)")
        return lift_df

    metric_keys  = list(OUTCOME_METRICS.keys())
    lift_cols    = [f"lift_{m}" for m in metric_keys]
    short_labels = [OUTCOME_METRICS[m].split("\n")[0] for m in metric_keys]
    col_headers  = ["Param", "Value", "N runs"] + [f"{l} Lift" for l in short_labels]

    cell_text = []
    for _, row in lift_df.iterrows():
        cell_text.append(
            [row["param"], row["value"], str(row["n_runs"])]
            + [f"{row[c]:.2f}" for c in lift_cols]
        )

    fig_h = max(3.5, 0.32 * len(cell_text) + 1.2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)

    # Color header row
    for ci in range(len(col_headers)):
        cell = table[0, ci]
        cell.set_facecolor("#4878CF")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    # Color lift cells
    for ri in range(1, len(cell_text) + 1):
        for ci, col in enumerate(lift_cols, start=3):
            try:
                v = float(cell_text[ri - 1][ci])
            except (ValueError, IndexError):
                continue
            if v >= 1.3:
                bg = "#b7e5b0"    # green
            elif v >= 1.0:
                bg = "#dff0db"    # light green
            elif v >= 0.7:
                bg = "#fce8d5"    # light orange
            else:
                bg = "#f5b8b8"    # red
            table[ri, ci].set_facecolor(bg)

    ax.set_title(
        "Lift Table  ·  lift > 1 = over-represented in top-quartile outcomes",
        fontsize=10, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    out_path = out_dir / "lift_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return lift_df


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-run sweep analysis visualizations")
    parser.add_argument(
        "--data_dir", default=str(DEFAULT_DATA_DIR),
        help="Directory containing run subdirectories (default: output/)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT_DIR),
        help="Output directory for generated figures (default: experimental_viz/output/)",
    )
    parser.add_argument(
        "--min_lift_n", type=int, default=10,
        help="Minimum runs per param value to include in lift table (default: 10). "
             "Filters out stray runs from old sweeps.",
    )
    parser.add_argument(
        "--filter_preset", choices=list(FILTER_PRESETS), default=None,
        help="Apply a named filter preset before visualizing. Outputs go to "
             "{output}/{preset_name}/. Available: " + ", ".join(FILTER_PRESETS),
    )
    args = parser.parse_args()

    out_dir  = Path(args.output)
    data_dir = Path(args.data_dir)

    # --- Load & prepare -------------------------------------------------
    df = load_all_runs(data_dir)
    df = add_derived_metrics(df)

    # Separate routing runs for all visualizations
    routing_df = df[df["routing_mode"] != "none"].copy()
    print(f"Routing runs for analysis: {len(routing_df)}")

    # Apply filter preset if requested; outputs go to a named subdirectory
    if args.filter_preset:
        viz_df  = apply_filter_preset(routing_df, args.filter_preset)
        out_dir = out_dir / args.filter_preset
        print_marginal_stats(viz_df, label=f"preset={args.filter_preset}")
    else:
        viz_df  = routing_df

    # Print quick distribution summary
    print("\n=== Derived metric summary ===")
    for m, label in OUTCOME_METRICS.items():
        s = viz_df[m].dropna()
        print(f"  {m:22s}  mean={s.mean():+.3f}  std={s.std():.3f}"
              f"  p10={s.quantile(0.1):+.3f}  p90={s.quantile(0.9):+.3f}")

    # --- Visualizations -------------------------------------------------
    print(f"\nWriting outputs to: {out_dir}/\n")

    print("[1/4] Marginal strip/box plots...")
    plot_marginal_strips(viz_df, out_dir)

    print("\n[2/4] 2D interaction heatmaps...")
    plot_2d_heatmaps(viz_df, out_dir)

    print("\n[3/4] Parallel coordinates (Plotly HTML)...")
    plot_parallel_coords(viz_df, out_dir)

    print("\n[4/4] Lift table...")
    lift_df = plot_lift_table(viz_df, out_dir, min_n=args.min_lift_n)

    print("\n=== Lift table ===")
    print(lift_df.to_string(index=False))
    print(f"\nDone. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
