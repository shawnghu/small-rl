"""Overlay a sweep's runs as a new intervention series on the fixed Pareto figure.

This captures the previously-manual workflow of "add a new GR-variant
intervention to the main Pareto figure" (proto_pareto_7envs_v2). Instead of
hand-writing an aggregator + cache key + style + legend entry per sweep, point
this at a sweep directory and it:

  1. Loads the FIXED backdrop (the established series: Gradient Routing, No
     intervention, Reward Penalty, plus whichever interventions are currently
     baked into the figure) from paper_plots/aggregated_cache.json. The user
     maintains these as fixed / infrequently-changing.
  2. Self-discovers the sweep's runs, maps each to its env, skips baselines,
     and aggregates retain/hack per env over seeds — using the SAME aggregator
     (proto_pareto_data.aggregate_paths) and the SAME drawing/styling
     primitives (proto_pareto_style_v2) as the paper figure, so the overlay is
     visually identical to a hand-added series.
  3. Writes pareto_overview.{pdf,png} into {sweep_dir}/sweep_graphs/.

It is also called automatically at the end of
`sweep_plots.generate_sweep_overview`, so every overview.html gets a matching
Pareto figure for free (from sweep.py and from standalone regeneration alike).

CLI:
    .venv/bin/python sweep_pareto.py output/<sweep> [--label "My intervention"]
        [--mode retain_only|both] [--color '#e377c2'] [--marker '*']
        [--run-substr SUBSTR] [--out DIR]

The new series is read in `retain_only` mode by default (forget adapter
ablated), matching every GR-variant intervention added so far. Pass
`--mode both` for RP-style sweeps.

Note on path coupling: proto_pareto_data.py does `os.chdir(repo/paper_plots)`
at import time (so its cached-render scripts resolve a local output/ snapshot).
We neutralize that here by saving/restoring CWD around the import, and by
passing only ABSOLUTE paths to the aggregator — so importing this module (e.g.
from sweep_plots) has no lasting CWD side effect.
"""
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PAPER_PLOTS = os.path.join(_REPO_ROOT, 'paper_plots')
_CACHE_PATH = os.path.join(_PAPER_PLOTS, 'aggregated_cache.json')

# Default style for the freshly-overlaid sweep series (distinct from every key
# currently in proto_pareto_style_v2.STYLES).
DEFAULT_COLOR = '#e377c2'   # magenta/pink
DEFAULT_MARKER = '*'


def _import_paper_modules():
    """Import the paper-figure data + style modules, neutralizing the
    chdir-on-import side effect of proto_pareto_data. Returns (data, style)."""
    cwd = os.getcwd()
    if _PAPER_PLOTS not in sys.path:
        sys.path.insert(0, _PAPER_PLOTS)
    try:
        import proto_pareto_data as data          # noqa: chdirs to paper_plots/
        import proto_pareto_style_v2 as style
    finally:
        os.chdir(cwd)
    return data, style


def _env_prefix_map(data):
    """List of (run_name_prefix, env) sorted longest-first, covering both the
    old and new env naming conventions so any sweep's run names resolve."""
    pairs = set()
    for env in data.ENVS:
        for prefix in (data.EYS_OLD[env], data.EYS_NEW[env]):
            pairs.add((prefix, env))
    return sorted(pairs, key=lambda pe: -len(pe[0]))


def discover_sweep_series(sweep_dir, mode='retain_only', run_substr=None,
                          data=None):
    """Aggregate a sweep's runs into one (retain, hack) point per env.

    Self-discovers run subdirs of `sweep_dir`, maps each to its env by name
    prefix, skips baselines / sweep_graphs / dotdirs / runs lacking
    routing_eval.jsonl, optionally filters to names containing `run_substr`,
    and pools all matching seeds per env through proto_pareto_data.aggregate_paths.

    Returns {env: (r_m, r_s, h_m, h_s, n)} (envs with no usable data omitted).
    """
    if data is None:
        data, _ = _import_paper_modules()
    sweep_dir = os.path.abspath(sweep_dir)
    prefix_map = _env_prefix_map(data)

    by_env = {}
    for name in sorted(os.listdir(sweep_dir)):
        full = os.path.join(sweep_dir, name)
        if not os.path.isdir(full):
            continue
        if name == 'sweep_graphs' or name.startswith('.') or name.startswith('baseline'):
            continue
        if run_substr and run_substr not in name:
            continue
        if not os.path.exists(os.path.join(full, 'routing_eval.jsonl')):
            continue
        env = next((e for p, e in prefix_map if name.startswith(p)), None)
        if env is None:
            continue
        by_env.setdefault(env, []).append(full)

    series = {}
    for env, paths in by_env.items():
        agg = data.aggregate_paths(paths, env, mode)
        if agg is not None:
            series[env] = agg
    return series


def _load_backdrop():
    """Load the fixed-backdrop cache. Returns {} (with a warning) if absent."""
    if not os.path.exists(_CACHE_PATH):
        print(f"[sweep_pareto] backdrop cache not found at {_CACHE_PATH}; "
              f"drawing sweep series only")
        return {}
    with open(_CACHE_PATH) as f:
        return json.load(f)


def _draw_backdrop_point(ax, style, cache_env, key):
    """Draw one fixed-backdrop series for an env. 'rp' is special: its point
    lives under cache['best_rp']['agg'] and is styled as 'rp_best' (matching
    proto_pareto_7envs_v2)."""
    if key == 'rp':
        agg = (cache_env.get('best_rp') or {}).get('agg')
        style_key = 'rp_best'
    else:
        agg = cache_env.get(key)
        style_key = key
    style.draw_point(ax, tuple(agg) if agg else None, style_key, zorder=8)


def generate_sweep_pareto(sweep_dir, out_dir=None, label=None,
                          color=DEFAULT_COLOR, marker=DEFAULT_MARKER,
                          mode='retain_only', backdrop_keys=None,
                          run_substr=None):
    """Render the Pareto figure with `sweep_dir`'s runs overlaid as a new
    series on the fixed backdrop. Writes pareto_overview.{pdf,png} to `out_dir`
    (default {sweep_dir}/sweep_graphs/). Returns the .pdf path, or None if the
    sweep contributed no plottable points."""
    sweep_dir = os.path.abspath(sweep_dir)
    data, style = _import_paper_modules()

    if backdrop_keys is None:
        backdrop_keys = list(style.LEGEND_ORDER_V2_MAIN)
    if label is None:
        label = os.path.basename(sweep_dir.rstrip('/'))
    if out_dir is None:
        out_dir = os.path.join(sweep_dir, 'sweep_graphs')

    cache = _load_backdrop()
    series = discover_sweep_series(sweep_dir, mode=mode, run_substr=run_substr,
                                   data=data)
    if not series:
        print(f"[sweep_pareto] no plottable runs in {sweep_dir} "
              f"(mode={mode}); skipping pareto overlay")
        return None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()
    for slot, env in style.SLOT_ENVS:
        ax = axes[slot]
        cache_env = cache.get(env, {})
        for key in backdrop_keys:
            _draw_backdrop_point(ax, style, cache_env, key)
        # New sweep series on top.
        style.draw_point(ax, series.get(env), color=color, marker=marker,
                         zorder=11)
        style.setup_axes(ax, env, slot)
        if env == style.ARROW_ENV:
            style.draw_better_arrow(ax)

    extra = [style.make_legend_handle(label, color, marker)]
    style.draw_legend(axes[style.LEGEND_SLOT], keys=backdrop_keys,
                      extra_handles=extra)

    fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
    fig.subplots_adjust(wspace=0.18, hspace=0.16)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'pareto_overview.pdf')
    png_path = os.path.join(out_dir, 'pareto_overview.png')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(png_path, dpi=110, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    n_envs = len(series)
    print(f"[sweep_pareto] wrote {pdf_path} ({n_envs} env(s) overlaid as '{label}')")
    return pdf_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep_dir', help='Sweep output directory (e.g. output/<sweep>)')
    ap.add_argument('--label', default=None,
                    help='Legend label for the new series (default: sweep dir name)')
    ap.add_argument('--mode', default='retain_only', choices=['retain_only', 'both'],
                    help='Eval mode for the new series (default: retain_only)')
    ap.add_argument('--color', default=DEFAULT_COLOR)
    ap.add_argument('--marker', default=DEFAULT_MARKER)
    ap.add_argument('--run-substr', default=None,
                    help='Only include runs whose name contains this substring '
                         '(use to pick one cell of a multi-cell sweep)')
    ap.add_argument('--out', default=None, help='Output dir (default: <sweep>/sweep_graphs)')
    args = ap.parse_args()

    generate_sweep_pareto(
        args.sweep_dir, out_dir=args.out, label=args.label, color=args.color,
        marker=args.marker, mode=args.mode, run_substr=args.run_substr,
    )


if __name__ == '__main__':
    main()
