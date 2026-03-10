"""Per-step comparison charts and animated GIFs for sweep experiments.

Called by sweep.py when an experiment group completes. Imports parsing and
plotting functions from plot_routing_comparison.py.

Output structure:
    {output_dir}/sweep_graphs/{group_name}/
        step_0100.png
        step_0200.png
        ...
        animation.gif
        lines_over_time.png
"""

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_routing_comparison import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    parse_routing_evals,
    parse_routing_evals_jsonl,
    extract_routing_metrics,
    aggregate_seeds,
    plot_routing_chart,
)



def _is_filter_baseline(run_dir):
    """Detect filter baseline from run directory name (prefix 'filter_')."""
    return os.path.basename(run_dir).startswith("filter_")


def _is_reward_penalty_baseline(run_dir):
    """Detect reward penalty baseline from run directory name (prefix 'reward_penalty_')."""
    return os.path.basename(run_dir).startswith("reward_penalty_")


def _is_retain_penalty_baseline(run_dir):
    """Detect retain penalty baseline from run directory name (prefix 'retain_penalty_')."""
    return os.path.basename(run_dir).startswith("retain_penalty_")


def build_step_data(routing_runs, baseline_runs, step, no_baseline=False):
    """Build aggregated data for one eval step.

    Collects routing eval data across seeds from routing_runs and baseline_runs.
    For regular baselines, renames "both" -> "baseline". For filter baselines,
    renames "both" -> "filter" so they appear as a separate condition on charts.

    Returns: {mode: {metric: (mean, std)}}, n_seeds
    """
    # Collect routing data across seeds
    routing_seed_results = []
    for run_dir in routing_runs:
        data = extract_routing_metrics(run_dir, step)
        if data:
            routing_seed_results.append(data)

    # Collect baseline data across seeds (regular, filter, reward penalty, retain penalty separately)
    baseline_seed_results = []
    filter_seed_results = []
    reward_penalty_seed_results = []
    retain_penalty_seed_results = []
    if not no_baseline:
        for run_dir in baseline_runs:
            data = extract_routing_metrics(run_dir, step)
            if data:
                renamed = {"_step": data.get("_step", step)}
                is_retain_penalty = _is_retain_penalty_baseline(run_dir)
                is_reward_penalty = _is_reward_penalty_baseline(run_dir)
                is_filter = _is_filter_baseline(run_dir)
                if is_retain_penalty:
                    target_mode = "retain_penalty"
                elif is_reward_penalty:
                    target_mode = "reward_penalty"
                elif is_filter:
                    target_mode = "filter"
                else:
                    target_mode = "baseline"
                for mode, metrics in data.items():
                    if mode.startswith("_"):
                        continue
                    if mode == "both":
                        renamed[target_mode] = metrics
                    # Skip retain_only/forget_only from baseline runs
                if is_retain_penalty:
                    retain_penalty_seed_results.append(renamed)
                elif is_reward_penalty:
                    reward_penalty_seed_results.append(renamed)
                elif is_filter:
                    filter_seed_results.append(renamed)
                else:
                    baseline_seed_results.append(renamed)

    # Aggregate
    plot_data = {}
    if baseline_seed_results:
        baseline_agg = aggregate_seeds(baseline_seed_results)
        plot_data.update(baseline_agg)

    if filter_seed_results:
        filter_agg = aggregate_seeds(filter_seed_results)
        plot_data.update(filter_agg)

    if reward_penalty_seed_results:
        reward_penalty_agg = aggregate_seeds(reward_penalty_seed_results)
        plot_data.update(reward_penalty_agg)

    if retain_penalty_seed_results:
        retain_penalty_agg = aggregate_seeds(retain_penalty_seed_results)
        plot_data.update(retain_penalty_agg)

    if routing_seed_results:
        routing_agg = aggregate_seeds(routing_seed_results)
        plot_data.update(routing_agg)

    n_seeds = max(len(routing_seed_results), len(baseline_seed_results),
                  len(filter_seed_results), len(reward_penalty_seed_results),
                  len(retain_penalty_seed_results))
    return plot_data, n_seeds


def generate_group_comparison_plots(routing_runs, baseline_runs, reward,
                                     output_dir, group_name="default", no_baseline=False):
    """Generate line-over-time plots for an experiment group.

    Args:
        routing_runs: list of routing run directories (one per seed)
        baseline_runs: list of baseline run directories (one per seed)
        reward: reward function name (for chart title)
        output_dir: base output directory
        group_name: name for this group's output subdirectory
        no_baseline: if True, skip baseline data entirely
    """
    # Find union of all eval steps across routing runs (JSONL preferred over train.log)
    all_steps = set()
    for run_dir in routing_runs:
        evals = parse_routing_evals_jsonl(run_dir)
        if not evals:
            evals = parse_routing_evals(os.path.join(run_dir, "train.log"))
        all_steps.update(evals.keys())

    if not all_steps:
        print(f"[PLOTS] No routing eval data found for group '{group_name}'")
        return

    steps = sorted(all_steps)
    graph_dir = Path(output_dir) / "sweep_graphs" / group_name
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Build time series data for line graphs
    time_series = {}  # {mode: {metric: [(step, mean, std, lo, hi), ...]}}
    for step in steps:
        plot_data, _ = build_step_data(
            routing_runs, baseline_runs, step,
            no_baseline=no_baseline,
        )
        for mode, metrics in plot_data.items():
            if mode not in time_series:
                time_series[mode] = {m: [] for m in ["combined", "retain", "hack_freq"]}
            for metric_key in ["combined", "retain", "hack_freq"]:
                mean, std, lo, hi = metrics[metric_key]
                time_series[mode][metric_key].append((step, mean, std, lo, hi))

    # Generate line graphs (with and without shading)
    if time_series:
        lines_path = str(graph_dir / "lines_over_time.png")
        lines_noshade_path = str(graph_dir / "lines_over_time_noshade.png")
        title = reward.replace("_", " ").title()
        if group_name != "default":
            title += f" ({group_name.replace('_', ' ')})"
        n_seeds_val = max(len(routing_runs), len(baseline_runs))
        generate_line_graphs(time_series, lines_path, title=title,
                             n_seeds=n_seeds_val, shade=True)
        generate_line_graphs(time_series, lines_noshade_path, title=title,
                             n_seeds=n_seeds_val, shade=False)

        # Simple index.html showing line graphs with shade toggle
        html_path = str(graph_dir / "index.html")
        generate_html_viewer([], [], html_path,
                             lines_image="lines_over_time.png",
                             lines_image_noshade="lines_over_time_noshade.png")
        print(f"[PLOTS] Generated line graphs + viewer for group '{group_name}'")
        print(f"  {lines_path}")
    else:
        print(f"[PLOTS] No plottable data for group '{group_name}'")


def generate_line_graphs(time_series, output_path, title="", n_seeds=None, shade=True):
    """Generate 3 line graphs (proxy reward, task reward, hack_freq) over time.

    Args:
        time_series: {mode: {metric: [(step, mean, std), ...]}}
        output_path: where to save the PNG
        title: chart title
        n_seeds: number of seeds (for annotation)
        shade: if True, draw stddev/min-max fill bands around lines
    """
    metric_configs = [
        ("combined", "Combined Reward"),
        ("retain", "Retain Reward"),
        ("hack_freq", "Hack Frequency"),
    ]
    mode_order = ["baseline", "filter", "reward_penalty", "retain_penalty", "both", "retain_only", "forget_only"]
    modes_present = [m for m in mode_order if m in time_series]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (metric_key, metric_label) in zip(axes, metric_configs):
        for mode in modes_present:
            points = time_series[mode].get(metric_key, [])
            if not points:
                continue
            steps_arr = np.array([p[0] for p in points])
            means = np.array([p[1] for p in points])
            stds = np.array([p[2] for p in points])
            los = np.array([p[3] for p in points])
            his = np.array([p[4] for p in points])

            color = CONDITION_COLORS[mode]
            label = CONDITION_LABELS[mode]
            ax.plot(steps_arr, means, color=color, label=label, linewidth=2)
            if shade:
                ax.fill_between(steps_arr, means - stds, means + stds,
                                color=color, alpha=0.40)
                ax.fill_between(steps_arr, los, his,
                                color=color, alpha=0.12)

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(modes_present),
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    subtitle = f"{n_seeds} seeds" if n_seeds else ""
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.08)
    if subtitle:
        fig.text(0.5, 1.04, f"({subtitle})", ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_gif(image_paths, output_path, duration_ms=800):
    """Assemble per-step PNGs into an animated GIF.

    RGBA images are converted to RGB. Final frame is held for 3x duration.
    """
    from PIL import Image

    frames = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        frames.append(img)

    if not frames:
        return

    # Durations: normal for all frames, 3x for final frame
    durations = [duration_ms] * len(frames)
    durations[-1] = duration_ms * 3

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def generate_html_viewer(image_paths, steps, output_path, lines_image=None,
                         lines_image_noshade=None):
    """Generate an interactive HTML viewer with slider for per-step charts.

    Works when served via HTTP (e.g. python -m http.server).
    Uses relative paths so the HTML file can live alongside the PNGs.
    Includes a tabbed view for per-step bar charts and line-over-time graphs.
    Defaults to the Over Time tab when line graphs are available.
    """
    # Build list of relative image filenames
    filenames = [os.path.basename(p) for p in image_paths]
    step_labels = [str(s) for s in steps[:len(filenames)]]

    # Decide default tab: prefer 'lines' if available
    default_tab = "lines" if lines_image else "steps"

    steps_display = "none" if default_tab == "lines" else ""
    lines_display = "" if default_tab == "lines" else "none"

    lines_section = ""
    shade_checkbox = ""
    shade_script = ""
    if lines_image:
        noshade_src = lines_image_noshade if lines_image_noshade else lines_image
        shade_checkbox = f"""
  <div class="shade-toggle">
    <label>
      <input type="checkbox" id="shade-cb" onchange="toggleShade()">
      Show std/min-max shading
    </label>
  </div>"""
        shade_script = f"""
const linesShade = "{lines_image}";
const linesNoShade = "{noshade_src}";
function toggleShade() {{
  const shaded = document.getElementById('shade-cb').checked;
  document.getElementById('lines-img').src = shaded ? linesShade : linesNoShade;
}}"""
        lines_section = f"""
  <div id="lines-view" style="display:{lines_display};">
    {shade_checkbox}
    <img id="lines-img" src="{noshade_src}" style="max-width:100%; border:1px solid #ccc; border-radius:4px;">
  </div>"""

    tab_buttons = ""
    tab_script = ""
    if lines_image:
        steps_active = "" if default_tab == "lines" else " active"
        lines_active = " active" if default_tab == "lines" else ""
        tab_buttons = f"""
  <div class="tabs">
    <button class="tab{steps_active}" onclick="showTab('steps', this)">Per-Step Charts</button>
    <button class="tab{lines_active}" onclick="showTab('lines', this)">Over Time</button>
  </div>"""
        tab_script = """
function showTab(which, btn) {
  document.getElementById('steps-view').style.display = which === 'steps' ? '' : 'none';
  document.getElementById('lines-view').style.display = which === 'lines' ? '' : 'none';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
}"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sweep Comparison</title>
<style>
  body {{ font-family: sans-serif; text-align: center; background: #f5f5f5; margin: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }}
  .controls {{ margin: 15px 0; display: flex; align-items: center; justify-content: center; gap: 12px; }}
  input[type=range] {{ width: 400px; }}
  button {{ padding: 6px 14px; font-size: 14px; cursor: pointer; }}
  #step-label {{ font-size: 18px; font-weight: bold; min-width: 120px; }}
  .keyboard-hint {{ color: #888; font-size: 12px; margin-top: 4px; }}
  .tabs {{ margin: 15px 0; display: flex; justify-content: center; gap: 4px; }}
  .tab {{ padding: 8px 20px; font-size: 14px; cursor: pointer; border: 1px solid #ccc;
          border-radius: 4px 4px 0 0; background: #e8e8e8; }}
  .tab.active {{ background: white; border-bottom: 1px solid white; font-weight: bold; }}
  .shade-toggle {{ margin: 10px 0; font-size: 14px; color: #444; }}
</style>
</head>
<body>
<div class="container">
  {tab_buttons}
  <div id="steps-view" style="display:{steps_display};">
    <div class="controls">
      <button onclick="prev()">&larr; Prev</button>
      <input type="range" id="slider" min="0" max="{len(filenames) - 1}" value="0" oninput="update(this.value)">
      <button onclick="next()">Next &rarr;</button>
      <button id="play-btn" onclick="togglePlay()">Play</button>
    </div>
    <div id="step-label">Step {step_labels[0] if step_labels else '?'}</div>
    <div class="keyboard-hint">Arrow keys: prev/next | Space: play/pause</div>
    <img id="chart" src="{filenames[0] if filenames else ''}">
  </div>
  {lines_section}
</div>
<script>
const files = {filenames};
const steps = {step_labels};
let idx = 0;
let playing = false;
let timer = null;

function update(i) {{
  idx = parseInt(i);
  document.getElementById('chart').src = files[idx];
  document.getElementById('slider').value = idx;
  document.getElementById('step-label').textContent = 'Step ' + steps[idx];
}}
function prev() {{ if (idx > 0) update(idx - 1); }}
function next() {{ if (idx < files.length - 1) update(idx + 1); }}
function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? 'Pause' : 'Play';
  if (playing) {{
    timer = setInterval(() => {{
      if (idx < files.length - 1) next();
      else {{ togglePlay(); update(0); }}
    }}, 800);
  }} else {{
    clearInterval(timer);
  }}
}}
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') prev();
  else if (e.key === 'ArrowRight') next();
  else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});
{tab_script}
{shade_script}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


def generate_sweep_overview(sweep_dir):
    """Generate an interactive Plotly overview page for all groups in a sweep.

    Loads routing_eval.jsonl from all runs, builds per-seed traces, and
    generates an interactive HTML page with per-condition/seed checkboxes,
    hover-to-highlight, and cross-panel tooltip sync.

    Output: {sweep_dir}/sweep_graphs/overview.html
    """
    from viz_playground import load_sweep, build_traces, generate_by_group_html

    sweep_dir = Path(sweep_dir)
    graphs_dir = sweep_dir / "sweep_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    runs = load_sweep(str(sweep_dir))
    if not runs:
        print(f"[OVERVIEW] No runs with routing_eval.jsonl in {sweep_dir}")
        return

    traces = build_traces(runs)
    output_path = str(graphs_dir / "overview.html")
    generate_by_group_html(
        runs, traces, str(sweep_dir), sweep_dir.name, output_path,
        page_title=f"Sweep Overview — {sweep_dir.name}",
    )
    print(f"[OVERVIEW] Generated interactive overview: {output_path}")


def _meta_to_routing_key(meta_entry):
    """Construct expected routing run key from groups_meta params.

    Uses PARAM_SHORT abbreviations to match the naming convention
    in make_run_name (from sweep.py).
    """
    from sweep import PARAM_SHORT

    params = meta_entry["params"]
    prefix = params.get("cfg", meta_entry.get("prefix", ""))

    parts = [prefix] if prefix else []
    for k in sorted(params.keys()):
        if k in ("cfg", "prefix"):
            continue
        short = PARAM_SHORT.get(k, k)
        parts.append(f"{short}{params[k]}")
    return "_".join(parts) if parts else ""


def generate_sweep_grid(sweep_dir):
    """Generate an interactive dual-mode Plotly grid page.

    Reads groups_meta.json for axis discovery, loads run data via
    viz_playground, and generates grid.html with Grid and List modes.
    """
    import json
    from collections import defaultdict

    from viz_playground import (
        load_sweep, build_traces, assign_groups,
        match_baseline_to_routing, _traces_to_plotly_json,
        _seed_checkbox_html,
        PLOTLY_CDN, METRIC_PANELS,
    )

    sweep_dir = Path(sweep_dir)
    graphs_dir = sweep_dir / "sweep_graphs"
    meta_path = graphs_dir / "groups_meta.json"

    if not meta_path.exists():
        print(f"[GRID] No groups_meta.json in {graphs_dir} — skipping grid page")
        return

    with open(meta_path) as f:
        groups_meta = json.load(f)

    if not groups_meta:
        print(f"[GRID] Empty groups_meta.json — skipping grid page")
        return

    # Discover axes from groups_meta
    all_param_keys = set()
    for g in groups_meta:
        all_param_keys.update(g["params"].keys())

    prefixes = set(g["prefix"] for g in groups_meta)
    include_prefix = len(prefixes) > 1

    axes = {}
    for key in sorted(all_param_keys):
        vals = set()
        for g in groups_meta:
            vals.add(g["params"].get(key, "\u2014"))
        if len(vals) > 1:
            axes[key] = _sort_values(list(vals))

    if include_prefix:
        axes["prefix"] = _sort_values(list(prefixes))

    if len(axes) < 1:
        print(f"[GRID] Only one group or no varying params — skipping grid page")
        return

    # Load run data via viz_playground
    runs = load_sweep(str(sweep_dir))
    if not runs:
        print(f"[GRID] No runs found — skipping grid page")
        return

    traces = build_traces(runs)
    groups = assign_groups(runs, str(sweep_dir))
    merged = match_baseline_to_routing(groups)

    # Build trace index by run name
    trace_by_run = defaultdict(list)
    for t in traces:
        trace_by_run[t["run_name"]].append(t)

    # Build routing key -> groups_meta mapping by stripping seed from meta name
    # (matches how assign_groups creates group keys)
    meta_by_key = {}
    for gm in groups_meta:
        key = re.sub(r"_s\d+$", "", gm.get("params", {}).get("run_name", ""))
        if not key:
            # Fallback: strip seed from the full meta name
            key = re.sub(r"_s\d+$", "", gm["name"])
        if key not in meta_by_key:
            meta_by_key[key] = gm

    # For each merged group, find its groups_meta entry and build Plotly data
    groups_data = []
    for routing_key, member_keys in sorted(merged.items()):
        group_traces = []
        for mk in member_keys:
            if mk in groups:
                for run_idx in groups[mk]:
                    run = runs[run_idx]
                    group_traces.extend(trace_by_run[run["name"]])

        if not group_traces:
            continue

        # Match to groups_meta entry by seed-stripped key
        gm = meta_by_key.get(routing_key)
        if gm is None:
            continue
        plotly_data, conditions_seen, seeds_seen = _traces_to_plotly_json(group_traces)

        full_params = dict(gm["params"])
        if include_prefix:
            full_params["prefix"] = gm["prefix"]

        cond_order = ["baseline", "filter", "reward_penalty", "retain_penalty",
                      "both", "retain_only", "forget_only"]

        groups_data.append({
            "name": gm["name"],
            "params": full_params,
            "traces": plotly_data,
            "conditions": [c for c in cond_order if c in conditions_seen],
            "seeds": seeds_seen,
        })

    if not groups_data:
        print(f"[GRID] No matching groups — skipping grid page")
        return

    # Pick sensible defaults
    axis_names = list(axes.keys())
    default_row = axis_names[0] if len(axis_names) >= 1 else ""
    default_col = axis_names[1] if len(axis_names) >= 2 else axis_names[0]

    html = _build_grid_html(sweep_dir.name, groups_data, axes, default_row, default_col)

    out_path = graphs_dir / "grid.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[GRID] Generated {out_path} ({len(groups_data)} groups, {len(axes)} axes: {', '.join(axes.keys())})")


def _sort_values(vals):
    """Sort values: numeric values numerically, strings alphabetically.
    The special value '—' (missing) sorts last."""
    nums = []
    strs = []
    has_missing = False
    for v in vals:
        if v == "—":
            has_missing = True
            continue
        try:
            nums.append((float(v), v))
        except (ValueError, TypeError):
            strs.append(v)
    nums.sort(key=lambda x: x[0])
    strs.sort()
    result = [v for _, v in nums] + strs
    if has_missing:
        result.append("—")
    return result


def _build_grid_html(sweep_name, groups_data, axes, default_row, default_col):
    """Build dual-mode (Grid + List) Plotly grid.html content."""
    import json

    from viz_playground import (
        _seed_checkbox_html,
        PLOTLY_CDN,
    )
    from plot_routing_comparison import CONDITION_COLORS, CONDITION_LABELS

    # Collect all conditions and seeds across groups
    all_conditions = set()
    all_seeds = set()
    for g in groups_data:
        all_conditions.update(g["conditions"])
        all_seeds.update(g["seeds"])
    cond_order = ["baseline", "filter", "reward_penalty", "retain_penalty",
                  "both", "retain_only", "forget_only"]
    conditions = [c for c in cond_order if c in all_conditions]
    seeds_sorted = sorted(all_seeds, key=lambda s: (len(s), s))

    # Build condition checkbox HTML
    cond_cb_parts = []
    for cond in conditions:
        color = CONDITION_COLORS.get(cond, "#888")
        label = CONDITION_LABELS.get(cond, cond)
        cond_cb_parts.append(
            f'<label class="cb-label" style="border-left: 4px solid {color};">'
            f'<input type="checkbox" checked data-condition="{cond}" '
            f'onchange="toggleCondition(this)"> {label}'
            f'</label>'
        )
    cond_checkbox_html = "\n    ".join(cond_cb_parts)

    # Seed checkbox HTML
    seed_cb_html = ""
    if len(seeds_sorted) <= 10:
        seed_parts = _seed_checkbox_html(seeds_sorted)
        seed_cb_html = (
            "<div class='filter-row'><span class='filter-label'>Seed</span>"
            + "\n    ".join(seed_parts)
            + "</div>"
        )

    groups_json = json.dumps(groups_data)
    axes_json = json.dumps(axes)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sweep Grid — {sweep_name}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #fafafa; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 5px; }}
  .subtitle {{ text-align: center; color: #777; font-size: 14px; margin-bottom: 10px; }}

  .filter-bar {{
    position: sticky; top: 0; z-index: 100;
    max-width: 1400px; margin: 0 auto 12px auto;
    background: white; border: 1px solid #ddd; border-radius: 6px;
    padding: 10px 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .filter-bar .filter-row {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  }}
  .filter-bar .filter-row + .filter-row {{ margin-top: 8px; padding-top: 8px;
    border-top: 1px solid #eee; }}
  .filter-label {{ font-size: 12px; font-weight: bold; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px; min-width: 75px; }}
  .cb-label {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px 4px 8px; font-size: 13px; cursor: pointer;
    border-radius: 4px; background: #f5f5f5;
  }}
  .cb-label:hover {{ background: #eee; }}
  .cb-seed {{ border-left: 3px solid #bbb; }}

  .controls {{
    max-width: 1400px; margin: 0 auto 10px auto;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 18px;
  }}
  .control-group {{ display: flex; align-items: center; gap: 6px; font-size: 14px; }}
  .control-group label {{ font-weight: bold; color: #555; }}
  select {{ padding: 4px 8px; font-size: 13px; }}

  .mode-toggle {{ display: flex; gap: 2px; }}
  .mode-toggle button {{ padding: 6px 16px; font-size: 13px; cursor: pointer;
    border: 1px solid #ccc; background: #e8e8e8; }}
  .mode-toggle button:first-child {{ border-radius: 4px 0 0 4px; }}
  .mode-toggle button:last-child {{ border-radius: 0 4px 4px 0; }}
  .mode-toggle button.active {{ background: #4878CF; color: white; border-color: #4878CF;
    font-weight: bold; }}

  .metric-radios {{ display: flex; gap: 2px; font-size: 13px; }}
  .metric-radios label {{ cursor: pointer; padding: 4px 10px; border: 1px solid #ccc;
    border-radius: 4px; background: #e8e8e8; }}
  .metric-radios label.active {{ background: #4878CF; color: white; border-color: #4878CF; }}

  .tab-bars {{ max-width: 1400px; margin: 0 auto 10px auto; }}
  .tab-bar {{ display: flex; align-items: center; gap: 4px; margin-bottom: 6px; font-size: 13px;
    flex-wrap: wrap; }}
  .tab-bar .tab-label {{ font-weight: bold; color: #555; min-width: 100px; text-align: right;
    margin-right: 6px; }}
  .tab-bar button {{ padding: 4px 12px; cursor: pointer; border: 1px solid #ccc;
    border-radius: 4px; background: #e8e8e8; font-size: 13px; }}
  .tab-bar button.active {{ background: #4878CF; color: white; border-color: #4878CF;
    font-weight: bold; }}

  .btn-row {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 10px; }}
  .btn-row button {{ padding: 5px 14px; font-size: 13px; cursor: pointer;
    border: 1px solid #ccc; border-radius: 4px; background: #fff; }}
  .btn-row button:hover {{ background: #e8e8e8; }}

  /* Grid mode */
  .grid-container {{ max-width: 1800px; margin: 0 auto; overflow-x: auto; }}
  table {{ border-collapse: collapse; }}
  th {{ padding: 6px 8px; font-size: 13px; background: #e0e0e0; border: 1px solid #ccc;
    position: sticky; top: 0; z-index: 1; }}
  .corner-header {{ font-size: 11px; color: #777; font-weight: normal; font-style: italic; }}
  .row-header {{ text-align: right; background: #e0e0e0; font-weight: bold; font-size: 13px;
    padding: 6px 10px; border: 1px solid #ccc; white-space: nowrap; }}
  td {{ border: 1px solid #ddd; padding: 4px; vertical-align: top; text-align: center;
    background: white; }}
  td.empty {{ background: #f0f0f0; color: #aaa; font-size: 13px; vertical-align: middle; }}
  .grid-cell {{ width: 320px; height: 250px; }}

  /* List mode */
  .list-container {{ max-width: 1800px; margin: 0 auto; }}
  .group-section {{
    max-width: 1800px; margin: 0 auto 30px auto;
    background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px;
  }}
  .group-section h2 {{ margin: 0 0 10px 0; font-size: 16px; }}
  .run-count {{ color: #999; font-weight: normal; font-size: 14px; }}
  .panels {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }}
  .list-panel {{ flex: 1 1 400px; min-width: 350px; max-width: 580px; height: 380px; }}
</style>
</head>
<body>
<h1>Sweep Grid — {sweep_name}</h1>
<div class="subtitle">{len(groups_data)} groups</div>

<div class="controls">
  <div class="control-group">
    <label>Mode:</label>
    <div class="mode-toggle" id="mode-toggle">
      <button class="active" onclick="switchMode('grid', this)">Grid</button>
      <button onclick="switchMode('list', this)">List</button>
    </div>
  </div>
  <div class="control-group">
    <label>Rows:</label>
    <select id="row-axis" onchange="rebuild()"></select>
  </div>
  <div class="control-group">
    <label>Cols:</label>
    <select id="col-axis" onchange="rebuild()"></select>
  </div>
  <div class="control-group" id="metric-group">
    <label>Metric:</label>
    <div class="metric-radios" id="metric-radios">
      <label class="active" onclick="setMetric(-1, this)"><span>All</span></label>
      <label onclick="setMetric(0, this)"><span>Combined</span></label>
      <label onclick="setMetric(1, this)"><span>Retain</span></label>
      <label onclick="setMetric(2, this)"><span>Hack Freq</span></label>
      <label onclick="setMetric(3, this)"><span>Retain\u2212Hack</span></label>
    </div>
  </div>
</div>
<div class="tab-bars" id="tab-bars"></div>
<div class="btn-row">
  <button onclick="selectAll(true)">Select All</button>
  <button onclick="selectAll(false)">Deselect All</button>
</div>
<div class="filter-bar">
  <div class="filter-row">
    <span class="filter-label">Condition</span>
    {cond_checkbox_html}
  </div>
  {seed_cb_html}
</div>

<div id="grid-view">
  <div class="grid-container">
    <table id="grid-table"></table>
  </div>
</div>
<div id="list-view" style="display:none;">
  <div class="list-container" id="list-container"></div>
</div>

<script>
const groupsData = {groups_json};
const axes = {axes_json};
const axisNames = Object.keys(axes);

const METRIC_KEYS = ['combined', 'retain', 'hack_freq', 'retain_minus_hack'];
const METRIC_TITLES = ['Combined Reward', 'Retain Reward', 'Hack Frequency', 'Retain \u2212 Hack'];
const METRIC_DASHES = ['solid', 'dash', 'dot', 'dashdot'];
const DIM_OPACITY = 0.12;
const HIGHLIGHT_WIDTH = 3.0;

let mode = 'grid';
let rowAxis = "{default_row}";
let colAxis = "{default_col}";
let selectedMetric = -1;
let filterValues = {{}};

const condVisible = {{}};
{json.dumps(conditions)}.forEach(c => condVisible[c] = true);
const seedVisible = {{}};
{json.dumps(seeds_sorted)}.forEach(s => seedVisible[s] = true);

let highlightedSeed = null;
let syncing = false;

// Track rendered charts and visibility state
const rendered = new Set();
const needsRestyled = new Set();
const onScreen = new Set();
const panelMeta = {{}};    // divId -> {{traces, title, ...}}
const groupPeers = {{}};   // divId -> [divId, ...]
let observer = null;

function hexToRgba(hex, a) {{
  const h = hex.replace('#','');
  return `rgba(${{parseInt(h.slice(0,2),16)}},${{parseInt(h.slice(2,4),16)}},${{parseInt(h.slice(4,6),16)}},${{a.toFixed(2)}})`;
}}

// === Lazy Rendering ===
function initObserver() {{
  if (observer) observer.disconnect();
  observer = new IntersectionObserver((entries) => {{
    for (const entry of entries) {{
      const id = entry.target.id;
      if (entry.isIntersecting) {{
        onScreen.add(id);
        if (panelMeta[id]) {{
          renderPanel(id);
          if (needsRestyled.has(id)) {{
            needsRestyled.delete(id);
            restylePanel(id);
          }}
        }}
      }} else {{
        onScreen.delete(id);
      }}
    }}
  }}, {{ rootMargin: '300px' }});
}}

function observeAll() {{
  if (!observer) initObserver();
  for (const divId of Object.keys(panelMeta)) {{
    const el = document.getElementById(divId);
    if (el) observer.observe(el);
  }}
}}

function clearRendered() {{
  // Purge all Plotly charts and reset tracking
  for (const divId of rendered) {{
    const el = document.getElementById(divId);
    if (el) try {{ Plotly.purge(el); }} catch(e) {{}}
  }}
  rendered.clear();
  needsRestyled.clear();
  onScreen.clear();
  for (const k of Object.keys(panelMeta)) delete panelMeta[k];
  for (const k of Object.keys(groupPeers)) delete groupPeers[k];
  if (observer) observer.disconnect();
  observer = null;
}}

const GRID_HOVER = '<b>%{{data.name}}</b><br>Step: %{{x}}<br>'
  + 'Combined: %{{customdata[0]:.3f}}<br>Retain: %{{customdata[1]:.3f}}<br>'
  + 'Hack Freq: %{{customdata[2]:.3f}}<br>Retain\u2212Hack: %{{customdata[3]:.3f}}<extra></extra>';

function renderPanel(divId) {{
  if (rendered.has(divId)) return;
  rendered.add(divId);
  const meta = panelMeta[divId];
  if (!meta) return;
  const traces = meta.traces.map(t => ({{
    x: t.x, y: t.y, customdata: t.customdata,
    type: t.type, mode: t.mode, name: t.name,
    legendgroup: t.legendgroup,
    line: {{...t.line, ...(meta.lineMods || {{}})}},
    hovertemplate: meta.useGridHover ? GRID_HOVER : t.hovertemplate,
    showlegend: false,
    visible: (condVisible[t._condition] && seedVisible[t._seed]) ? true : false,
  }}));
  const layout = {{
    title: {{ text: meta.title || '', font: {{ size: 14 }} }},
    xaxis: {{ title: 'Step' }},
    yaxis: {{ range: meta.yRange || [-0.05, 1.1] }},
    margin: meta.margin || {{ t: 35, b: 45, l: 50, r: 15 }},
    hovermode: 'closest',
  }};
  const div = document.getElementById(divId);
  Plotly.newPlot(div, traces, layout, {{ responsive: true }});

  div.on('plotly_hover', (evt) => {{
    if (syncing) return;
    const pt = evt.points[0];
    const ci = pt.curveNumber;
    const traceData = meta.traces[ci];
    if (!traceData) return;
    const seed = traceData._seed;
    if (seed !== highlightedSeed) highlightSeedInGroup(seed, divId);
    // Cross-panel sync for list mode
    const peers = groupPeers[divId];
    if (peers && peers.length > 1) {{
      syncing = true;
      for (const peerId of peers) {{
        if (peerId === divId || !rendered.has(peerId)) continue;
        const peerDiv = document.getElementById(peerId);
        try {{ Plotly.Fx.hover(peerDiv, [{{curveNumber: ci, pointNumber: pt.pointNumber}}]); }}
        catch(e) {{}}
      }}
      syncing = false;
    }}
  }});
  div.on('plotly_unhover', () => {{
    if (syncing) return;
    if (highlightedSeed !== null) clearHighlightInGroup(divId);
    const peers = groupPeers[divId];
    if (peers && peers.length > 1) {{
      syncing = true;
      for (const peerId of peers) {{
        if (peerId === divId || !rendered.has(peerId)) continue;
        const peerDiv = document.getElementById(peerId);
        try {{ Plotly.Fx.unhover(peerDiv); }} catch(e) {{}}
      }}
      syncing = false;
    }}
  }});
}}

function highlightSeedInGroup(seed, sourceDivId) {{
  highlightedSeed = seed;
  const peerIds = groupPeers[sourceDivId] || [sourceDivId];
  for (const divId of peerIds) {{
    if (!rendered.has(divId)) continue;
    const meta = panelMeta[divId];
    if (!meta) continue;
    const div = document.getElementById(divId);
    const colors = [], widths = [];
    for (const t of meta.traces) {{
      if (t._seed === seed) {{
        colors.push(hexToRgba(t._base_color, 1.0));
        widths.push(HIGHLIGHT_WIDTH);
      }} else {{
        colors.push(hexToRgba(t._base_color, DIM_OPACITY));
        widths.push(1.0);
      }}
    }}
    Plotly.restyle(div, {{'line.color': colors, 'line.width': widths}});
  }}
}}

function clearHighlightInGroup(sourceDivId) {{
  highlightedSeed = null;
  const peerIds = groupPeers[sourceDivId] || [sourceDivId];
  for (const divId of peerIds) {{
    if (!rendered.has(divId)) continue;
    const meta = panelMeta[divId];
    if (!meta) continue;
    const div = document.getElementById(divId);
    const colors = [], widths = [];
    for (const t of meta.traces) {{
      colors.push(hexToRgba(t._base_color, t._base_opacity));
      widths.push(1.5);
    }}
    Plotly.restyle(div, {{'line.color': colors, 'line.width': widths}});
  }}
}}

function restylePanel(divId) {{
  const meta = panelMeta[divId];
  if (!meta) return;
  const div = document.getElementById(divId);
  const vis = meta.traces.map(t =>
    (condVisible[t._condition] && seedVisible[t._seed]) ? true : false
  );
  Plotly.restyle(div, {{ visible: vis }});
}}

function applyVisibility() {{
  for (const divId of rendered) {{
    if (onScreen.has(divId)) {{
      restylePanel(divId);
    }} else {{
      needsRestyled.add(divId);
    }}
  }}
}}

// === Controls ===
function populateAxisDropdowns() {{
  const rowSel = document.getElementById('row-axis');
  const colSel = document.getElementById('col-axis');
  rowSel.innerHTML = '';
  colSel.innerHTML = '';
  for (const name of axisNames) {{
    rowSel.innerHTML += '<option value="' + name + '"' + (name === rowAxis ? ' selected' : '') + '>' + name + '</option>';
    colSel.innerHTML += '<option value="' + name + '"' + (name === colAxis ? ' selected' : '') + '>' + name + '</option>';
  }}
}}

function pickViableDefaults() {{
  const extraAxes = axisNames.filter(n => n !== rowAxis && n !== colAxis);
  for (const name of extraAxes) {{
    if (!(name in filterValues) || !axes[name].includes(filterValues[name])) {{
      filterValues[name] = axes[name][0];
    }}
  }}
  const hasMatch = groupsData.some(g => extraAxes.every(k =>
    filterValues[k] === undefined || g.params[k] === filterValues[k]
  ));
  if (!hasMatch && groupsData.length > 0) {{
    const g0 = groupsData[0];
    for (const name of extraAxes) {{
      if (g0.params[name] !== undefined) filterValues[name] = g0.params[name];
    }}
  }}
}}

function buildTabBars() {{
  const container = document.getElementById('tab-bars');
  container.innerHTML = '';
  const extraAxes = axisNames.filter(n => n !== rowAxis && n !== colAxis);
  for (const name of extraAxes) {{
    const vals = axes[name];
    const bar = document.createElement('div');
    bar.className = 'tab-bar';
    bar.innerHTML = '<span class="tab-label">' + name + ':</span>';
    for (const v of vals) {{
      const btn = document.createElement('button');
      btn.textContent = v;
      if (v === filterValues[name]) btn.className = 'active';
      btn.onclick = () => {{ filterValues[name] = v; buildTabBars(); rebuildContent(); }};
      bar.appendChild(btn);
    }}
    container.appendChild(bar);
  }}
}}

function setMetric(m, el) {{
  selectedMetric = m;
  document.querySelectorAll('.metric-radios label').forEach(l => l.classList.remove('active'));
  el.classList.add('active');
  if (mode === 'grid') rebuildContent();
}}

function switchMode(newMode, btn) {{
  if (newMode === mode) return;
  mode = newMode;
  document.querySelectorAll('#mode-toggle button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('grid-view').style.display = mode === 'grid' ? '' : 'none';
  document.getElementById('list-view').style.display = mode === 'list' ? '' : 'none';
  document.getElementById('metric-group').style.display = mode === 'grid' ? '' : 'none';
  rebuildContent();
}}

function toggleCondition(checkbox) {{
  condVisible[checkbox.dataset.condition] = checkbox.checked;
  applyVisibility();
}}

function toggleSeed(checkbox) {{
  seedVisible[checkbox.dataset.seed] = checkbox.checked;
  applyVisibility();
}}

function selectAll(checked) {{
  document.querySelectorAll('.filter-bar input[type=checkbox]').forEach(cb => {{
    cb.checked = checked;
    if (cb.dataset.condition) condVisible[cb.dataset.condition] = checked;
    if (cb.dataset.seed) seedVisible[cb.dataset.seed] = checked;
  }});
  applyVisibility();
}}

// === Matching ===
function getMatchingGroups() {{
  const extraAxes = axisNames.filter(n => n !== rowAxis && n !== colAxis);
  return groupsData.filter(g =>
    extraAxes.every(k => {{
      if (filterValues[k] === undefined) return true;
      const gv = g.params[k];
      return gv === filterValues[k] || (gv === undefined && filterValues[k] === '\\u2014');
    }})
  );
}}

function findGroup(matching, paramSpec) {{
  return matching.find(g => {{
    for (const [k, v] of Object.entries(paramSpec)) {{
      const gv = g.params[k];
      if (gv === undefined && v === '\\u2014') continue;
      if (gv === undefined || gv !== v) return false;
    }}
    return true;
  }});
}}

// === Build traces for a grid cell ===
function getTracesForCell(group, metric) {{
  if (metric === -1) {{
    // All metrics overlaid with different dash patterns
    const traces = [];
    for (let mi = 0; mi < 3; mi++) {{
      const mk = METRIC_KEYS[mi];
      for (const t of group.traces[mk]) {{
        traces.push({{
          ...t,
          line: {{ ...t.line, dash: METRIC_DASHES[mi] }},
          _metric: mk,
        }});
      }}
    }}
    return traces;
  }} else {{
    const mk = METRIC_KEYS[metric];
    return group.traces[mk].map(t => ({{ ...t }}));
  }}
}}

// === Grid Mode ===
function rebuildGrid() {{
  clearRendered();
  const table = document.getElementById('grid-table');
  const matching = getMatchingGroups();
  const rowVals = axes[rowAxis] || ['\\u2014'];
  const colVals = axes[colAxis] || ['\\u2014'];

  let html = '<thead><tr><th class="corner-header">' + rowAxis + ' \\\\ ' + colAxis + '</th>';
  for (const cv of colVals) html += '<th>' + cv + '</th>';
  html += '</tr></thead><tbody>';

  for (const rv of rowVals) {{
    html += '<tr><td class="row-header">' + rv + '</td>';
    for (const cv of colVals) {{
      const spec = {{ [rowAxis]: rv, [colAxis]: cv }};
      if (rowAxis === colAxis) spec[rowAxis] = rv;
      const g = findGroup(matching, spec);
      if (g) {{
        const divId = 'gc-' + g.name.replace(/[^a-zA-Z0-9]/g, '_');
        html += '<td><div id="' + divId + '" class="grid-cell"></div></td>';
        const traces = getTracesForCell(g, selectedMetric);
        const title = selectedMetric === -1
          ? (rowAxis !== colAxis ? rv + ' / ' + cv : rv)
          : METRIC_TITLES[selectedMetric];
        const needsNegY = selectedMetric === 3 || selectedMetric === -1;
        panelMeta[divId] = {{
          traces: traces,
          title: title,
          useGridHover: true,
          margin: {{ t: 30, b: 40, l: 45, r: 10 }},
          yRange: needsNegY ? [-1.1, 1.1] : (selectedMetric === 0 ? [-0.05, 2.05] : [-0.05, 1.1]),
        }};
        // Grid cells are standalone (no peers)
        groupPeers[divId] = [divId];
      }} else {{
        html += '<td class="empty">\\u2014</td>';
      }}
    }}
    html += '</tr>';
  }}
  html += '</tbody>';
  table.innerHTML = html;
  initObserver();
  observeAll();
}}

// === List Mode ===
function rebuildList() {{
  clearRendered();
  const container = document.getElementById('list-container');
  const matching = getMatchingGroups();

  let html = '';
  for (const g of matching) {{
    const safeId = g.name.replace(/[^a-zA-Z0-9]/g, '_');
    const prefix = 'lp-' + safeId;
    const displayLabel = g.name.replace(/_/g, ' ');
    html += '<div class="group-section">';
    html += '<h2>' + displayLabel + '</h2>';
    html += '<div class="panels">';
    for (let mi = 0; mi < METRIC_KEYS.length; mi++) {{
      const divId = prefix + '-' + METRIC_KEYS[mi];
      html += '<div id="' + divId + '" class="list-panel"></div>';
    }}
    html += '</div></div>';
  }}
  container.innerHTML = html;

  // Register panels and peers
  for (const g of matching) {{
    const safeId = g.name.replace(/[^a-zA-Z0-9]/g, '_');
    const prefix = 'lp-' + safeId;
    const peerIds = [];
    for (let mi = 0; mi < METRIC_KEYS.length; mi++) {{
      const mk = METRIC_KEYS[mi];
      const divId = prefix + '-' + mk;
      peerIds.push(divId);
      panelMeta[divId] = {{
        traces: g.traces[mk],
        title: METRIC_TITLES[mi],
        useGridHover: false,
        yRange: mk === 'retain_minus_hack' ? [-1.1, 1.1] : mk === 'combined' ? [-0.05, 2.05] : undefined,
      }};
    }}
    for (const id of peerIds) groupPeers[id] = peerIds;
  }}

  initObserver();
  observeAll();
}}

// === Main Rebuild ===
function rebuildContent() {{
  if (mode === 'grid') rebuildGrid();
  else rebuildList();
}}

function rebuild() {{
  rowAxis = document.getElementById('row-axis').value;
  colAxis = document.getElementById('col-axis').value;
  pickViableDefaults();
  buildTabBars();
  rebuildContent();
}}

// Initial render
populateAxisDropdowns();
rebuild();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python sweep_plots.py <sweep_output_dir>")
        print("  Generates overview.html and grid.html in <sweep_output_dir>/sweep_graphs/")
        sys.exit(1)
    generate_sweep_overview(sys.argv[1])
    generate_sweep_grid(sys.argv[1])
