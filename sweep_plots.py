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



def build_step_data(routing_runs, baseline_runs, step, no_baseline=False):
    """Build aggregated data for one eval step.

    Collects routing eval data across seeds from routing_runs and baseline_runs.
    For baselines, renames the "both" mode to "baseline" since DualLoRA without
    routing still evaluates all 3 modes, but we want the "both" mode as the
    baseline comparison point.

    Returns: {mode: {metric: (mean, std)}}, n_seeds
    """
    # Collect routing data across seeds
    routing_seed_results = []
    for run_dir in routing_runs:
        data = extract_routing_metrics(run_dir, step)
        if data:
            routing_seed_results.append(data)

    # Collect baseline data across seeds
    baseline_seed_results = []
    if not no_baseline:
        for run_dir in baseline_runs:
            data = extract_routing_metrics(run_dir, step)
            if data:
                # Rename "both" -> "baseline" for the chart
                renamed = {"_step": data.get("_step", step)}
                for mode, metrics in data.items():
                    if mode.startswith("_"):
                        continue
                    if mode == "both":
                        renamed["baseline"] = metrics
                    # Skip retain_only/forget_only from baseline runs
                baseline_seed_results.append(renamed)

    # Aggregate
    plot_data = {}
    if baseline_seed_results:
        baseline_agg = aggregate_seeds(baseline_seed_results)
        plot_data.update(baseline_agg)

    if routing_seed_results:
        routing_agg = aggregate_seeds(routing_seed_results)
        plot_data.update(routing_agg)

    n_seeds = max(len(routing_seed_results), len(baseline_seed_results))
    return plot_data, n_seeds


def generate_group_comparison_plots(routing_runs, baseline_runs, reward,
                                     output_dir, group_name="default", no_baseline=False):
    """Generate per-step bar charts and animated GIF for an experiment group.

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

    image_paths = []
    n_routing_seeds = len(routing_runs)
    n_baseline_seeds = len(baseline_runs) if not no_baseline else 0

    for step in steps:
        plot_data, n_seeds = build_step_data(
            routing_runs, baseline_runs, step,
            no_baseline=no_baseline,
        )

        if not plot_data:
            continue

        output_path = str(graph_dir / f"step_{step:04d}.png")
        title = reward.replace("_", " ").title()
        step_info = f"step {step}"
        if group_name != "default":
            step_info += f", {group_name.replace('_', ' ')}"

        plot_routing_chart(
            title=title,
            data=plot_data,
            output_path=output_path,
            step_info=step_info,
            n_seeds=n_seeds,
        )
        image_paths.append(output_path)

    # Build time series data for line graphs
    time_series = {}  # {mode: {metric: [(step, mean, std), ...]}}
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

    if image_paths:
        gif_path = str(graph_dir / "animation.gif")
        generate_gif(image_paths, gif_path)

    # Generate line graphs (with and without shading)
    lines_path = None
    lines_noshade_path = None
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

    if image_paths or lines_path:
        html_path = str(graph_dir / "index.html")
        generate_html_viewer(image_paths, steps, html_path,
                             lines_image="lines_over_time.png" if lines_path else None,
                             lines_image_noshade="lines_over_time_noshade.png" if lines_noshade_path else None)
        parts = []
        if image_paths:
            parts.append(f"{len(image_paths)} charts + GIF")
        if lines_path:
            parts.append("line graphs")
        parts.append("HTML viewer")
        print(f"[PLOTS] Generated {' + '.join(parts)} for group '{group_name}'")
        print(f"  Charts: {graph_dir}/")
        if lines_path:
            print(f"  Lines: {lines_path}")
        print(f"  Viewer: {html_path}")
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
    mode_order = ["baseline", "both", "retain_only", "forget_only"]
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
    """Generate a single HTML page showing all group line graphs from a sweep.

    Scans {sweep_dir}/sweep_graphs/*/lines_over_time*.png and builds an
    overview page at {sweep_dir}/sweep_graphs/overview.html.

    Press 'S' to toggle std/min-max shading on all graphs simultaneously.
    """
    sweep_dir = Path(sweep_dir)
    graphs_dir = sweep_dir / "sweep_graphs"
    if not graphs_dir.is_dir():
        print(f"[OVERVIEW] No sweep_graphs/ directory in {sweep_dir}")
        return

    # Find all groups that have line graphs
    groups = []
    for group_dir in sorted(graphs_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        shade = group_dir / "lines_over_time.png"
        noshade = group_dir / "lines_over_time_noshade.png"
        if not noshade.exists() and not shade.exists():
            continue
        groups.append({
            "name": group_dir.name,
            "shade": f"{group_dir.name}/lines_over_time.png" if shade.exists() else None,
            "noshade": f"{group_dir.name}/lines_over_time_noshade.png" if noshade.exists() else None,
        })

    if not groups:
        print(f"[OVERVIEW] No line graphs found in {graphs_dir}")
        return

    # Build image entries (JS array)
    img_entries = []
    img_html = []
    for i, g in enumerate(groups):
        # Default to noshade; fall back to shade if noshade missing
        default_src = g["noshade"] or g["shade"]
        shade_src = g["shade"] or g["noshade"]
        noshade_src = g["noshade"] or g["shade"]
        img_entries.append(
            f'{{shade: "{shade_src}", noshade: "{noshade_src}"}}'
        )
        title = g["name"].replace("_", " ")
        img_html.append(
            f'  <div class="group">\n'
            f'    <h2>{title}</h2>\n'
            f'    <a href="{g["name"]}/index.html">'
            f'<img id="img-{i}" src="{default_src}"></a>\n'
            f'  </div>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sweep Overview — {sweep_dir.name}</title>
<style>
  body {{ font-family: sans-serif; background: #f5f5f5; margin: 20px; }}
  h1 {{ text-align: center; }}
  .hint {{ text-align: center; color: #888; font-size: 13px; margin-bottom: 20px; }}
  .hint kbd {{ background: #e0e0e0; padding: 2px 6px; border-radius: 3px;
               border: 1px solid #bbb; font-size: 13px; }}
  .shade-status {{ text-align: center; font-size: 14px; color: #555; margin-bottom: 10px; }}
  .group {{ max-width: 1400px; margin: 0 auto 30px auto; }}
  .group h2 {{ font-size: 16px; color: #333; margin-bottom: 6px; }}
  .group img {{ max-width: 100%; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; }}
</style>
</head>
<body>
<h1>Sweep Overview — {sweep_dir.name}</h1>
<div class="hint">Press <kbd>S</kbd> to toggle std/min-max shading</div>
<div class="shade-status" id="shade-status">Shading: OFF</div>
{"".join(img_html)}
<script>
const imgs = [{", ".join(img_entries)}];
let shaded = false;
function setShade(on) {{
  shaded = on;
  for (let i = 0; i < imgs.length; i++) {{
    document.getElementById('img-' + i).src = shaded ? imgs[i].shade : imgs[i].noshade;
  }}
  document.getElementById('shade-status').textContent = 'Shading: ' + (shaded ? 'ON' : 'OFF');
}}
document.addEventListener('keydown', (e) => {{
  if (e.key === 's' || e.key === 'S') setShade(!shaded);
}});
</script>
</body>
</html>"""

    out_path = graphs_dir / "overview.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[OVERVIEW] Generated {out_path} ({len(groups)} groups)")


def generate_sweep_grid(sweep_dir):
    """Generate an interactive grid HTML page for comparing sweep groups along two axes.

    Reads {sweep_dir}/sweep_graphs/groups_meta.json (written by sweep.py) and
    generates {sweep_dir}/sweep_graphs/grid.html.

    Features:
    - Row/column axis selection from swept params
    - Metric filter (All / Combined / Retain / Hack Freq) via CSS clipping
    - Tab bars for extra axes beyond the two grid axes
    - S key to toggle std/min-max shading
    """
    import json

    sweep_dir = Path(sweep_dir)
    graphs_dir = sweep_dir / "sweep_graphs"
    meta_path = graphs_dir / "groups_meta.json"

    if not meta_path.exists():
        print(f"[GRID] No groups_meta.json in {graphs_dir} — skipping grid page")
        return

    with open(meta_path) as f:
        groups = json.load(f)

    if not groups:
        print(f"[GRID] Empty groups_meta.json — skipping grid page")
        return

    # Verify each group has images
    valid_groups = []
    for g in groups:
        shade = graphs_dir / g["name"] / "lines_over_time.png"
        noshade = graphs_dir / g["name"] / "lines_over_time_noshade.png"
        if shade.exists() or noshade.exists():
            g["has_shade"] = shade.exists()
            g["has_noshade"] = noshade.exists()
            valid_groups.append(g)
    groups = valid_groups

    if not groups:
        print(f"[GRID] No groups with line graph images — skipping grid page")
        return

    # Discover axes: params that have more than one unique value
    all_param_keys = set()
    for g in groups:
        all_param_keys.update(g["params"].keys())

    # Include "prefix" as an axis if it varies
    prefixes = set(g["prefix"] for g in groups)
    include_prefix = len(prefixes) > 1

    axes = {}  # axis_name -> sorted list of unique values
    for key in sorted(all_param_keys):
        vals = set()
        for g in groups:
            vals.add(g["params"].get(key, "—"))
        if len(vals) > 1:
            axes[key] = _sort_values(list(vals))

    if include_prefix:
        axes["prefix"] = _sort_values(list(prefixes))

    if len(axes) < 1:
        print(f"[GRID] Only one group or no varying params — skipping grid page")
        return

    # Build JS data
    js_groups = []
    for g in groups:
        shade_src = f"{g['name']}/lines_over_time.png" if g["has_shade"] else None
        noshade_src = f"{g['name']}/lines_over_time_noshade.png" if g["has_noshade"] else None
        # Build full param dict including prefix
        full_params = dict(g["params"])
        if include_prefix:
            full_params["prefix"] = g["prefix"]
        js_groups.append({
            "name": g["name"],
            "params": full_params,
            "shade": shade_src or noshade_src,
            "noshade": noshade_src or shade_src,
        })

    js_axes = {k: v for k, v in axes.items()}

    # Pick sensible defaults: first two axes alphabetically
    axis_names = list(axes.keys())
    default_row = axis_names[0] if len(axis_names) >= 1 else ""
    default_col = axis_names[1] if len(axis_names) >= 2 else axis_names[0]

    html = _build_grid_html(sweep_dir.name, js_groups, js_axes, default_row, default_col)

    out_path = graphs_dir / "grid.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[GRID] Generated {out_path} ({len(groups)} groups, {len(axes)} axes: {', '.join(axes.keys())})")


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


def _build_grid_html(sweep_name, groups, axes, default_row, default_col):
    """Build the grid.html content as a string."""
    import json

    groups_json = json.dumps(groups)
    axes_json = json.dumps(axes)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sweep Grid — {sweep_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: sans-serif; background: #f5f5f5; margin: 20px; }}
  h1 {{ text-align: center; margin-bottom: 5px; }}

  .controls {{
    max-width: 1400px; margin: 0 auto 10px auto;
    display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 18px;
  }}
  .control-group {{ display: flex; align-items: center; gap: 6px; font-size: 14px; }}
  .control-group label {{ font-weight: bold; color: #555; }}
  select {{ padding: 4px 8px; font-size: 13px; }}

  .metric-radios {{ display: flex; gap: 10px; font-size: 13px; }}
  .metric-radios label {{ cursor: pointer; padding: 4px 10px; border: 1px solid #ccc;
    border-radius: 4px; background: #e8e8e8; }}
  .metric-radios input:checked + span {{  }}
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

  .hint {{ text-align: center; color: #888; font-size: 12px; margin-bottom: 8px; }}
  .hint kbd {{ background: #e0e0e0; padding: 1px 5px; border-radius: 3px;
    border: 1px solid #bbb; font-size: 12px; }}
  .shade-status {{ text-align: center; font-size: 13px; color: #555; margin-bottom: 8px; }}

  .grid-container {{ max-width: 1400px; margin: 0 auto; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ padding: 6px 8px; font-size: 13px; background: #e0e0e0; border: 1px solid #ccc;
    position: sticky; top: 0; z-index: 1; }}
  .corner-header {{ font-size: 11px; color: #777; font-weight: normal; font-style: italic; }}
  .row-header {{ text-align: right; background: #e0e0e0; font-weight: bold; font-size: 13px;
    padding: 6px 10px; border: 1px solid #ccc; white-space: nowrap; }}
  td {{ border: 1px solid #ddd; padding: 4px; vertical-align: top; text-align: center;
    background: white; min-width: 150px; }}
  td.empty {{ background: #f0f0f0; color: #aaa; font-size: 13px; vertical-align: middle; }}

  .cell-link {{ display: block; text-decoration: none; }}
  .cell-link:hover {{ opacity: 0.85; }}

  /* Full 3-panel view */
  .img-full img {{ width: 100%; display: block; border-radius: 3px; }}

  /* Single-metric clipped view */
  .img-clip {{ overflow: hidden; border-radius: 3px; }}
  .img-clip img {{ width: 300%; display: block; }}
  .img-clip.metric-0 img {{ margin-left: 0; }}
  .img-clip.metric-1 img {{ margin-left: -100%; }}
  .img-clip.metric-2 img {{ margin-left: -200%; }}
</style>
</head>
<body>
<h1>Sweep Grid — {sweep_name}</h1>
<div class="hint">Press <kbd>S</kbd> to toggle shading</div>
<div class="shade-status" id="shade-status">Shading: OFF</div>

<div class="controls">
  <div class="control-group">
    <label>Rows:</label>
    <select id="row-axis" onchange="rebuild()"></select>
  </div>
  <div class="control-group">
    <label>Cols:</label>
    <select id="col-axis" onchange="rebuild()"></select>
  </div>
  <div class="control-group">
    <label>Metric:</label>
    <div class="metric-radios" id="metric-radios">
      <label class="active" onclick="setMetric(-1, this)"><span>All</span></label>
      <label onclick="setMetric(0, this)"><span>Combined</span></label>
      <label onclick="setMetric(1, this)"><span>Retain</span></label>
      <label onclick="setMetric(2, this)"><span>Hack Freq</span></label>
    </div>
  </div>
</div>
<div class="tab-bars" id="tab-bars"></div>
<div class="grid-container">
  <table id="grid-table"></table>
</div>

<script>
const groups = {groups_json};
const axes = {axes_json};
const axisNames = Object.keys(axes);

let rowAxis = "{default_row}";
let colAxis = "{default_col}";
let metric = -1;  // -1 = all, 0/1/2 = panel index
let shaded = false;
let filterValues = {{}};  // axis -> selected value

// Populate axis dropdowns
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

function setMetric(m, el) {{
  metric = m;
  document.querySelectorAll('.metric-radios label').forEach(l => l.classList.remove('active'));
  el.classList.add('active');
  rebuildGrid();
}}

function setShade(on) {{
  shaded = on;
  document.getElementById('shade-status').textContent = 'Shading: ' + (shaded ? 'ON' : 'OFF');
  // Update all images
  document.querySelectorAll('img[data-shade]').forEach(img => {{
    img.src = shaded ? img.dataset.shade : img.dataset.noshade;
  }});
}}

// Build tab bars for axes not assigned to row/col
function buildTabBars() {{
  const container = document.getElementById('tab-bars');
  container.innerHTML = '';
  for (const name of axisNames) {{
    if (name === rowAxis || name === colAxis) continue;
    const vals = axes[name];
    if (!(name in filterValues) || !vals.includes(filterValues[name])) {{
      filterValues[name] = vals[0];
    }}
    const bar = document.createElement('div');
    bar.className = 'tab-bar';
    bar.innerHTML = '<span class="tab-label">' + name + ':</span>';
    for (const v of vals) {{
      const btn = document.createElement('button');
      btn.textContent = v;
      if (v === filterValues[name]) btn.className = 'active';
      btn.onclick = () => {{ filterValues[name] = v; rebuild(); }};
      bar.appendChild(btn);
    }}
    container.appendChild(bar);
  }}
}}

// Find the group matching a specific set of param values
function findGroup(paramSpec) {{
  return groups.find(g => {{
    for (const [k, v] of Object.entries(paramSpec)) {{
      const gv = g.params[k];
      if (gv === undefined && v === '\\u2014') continue;  // missing matches dash
      if (gv === undefined || gv !== v) return false;
    }}
    return true;
  }});
}}

function rebuildGrid() {{
  const table = document.getElementById('grid-table');
  const rowVals = axes[rowAxis] || ['—'];
  const colVals = axes[colAxis] || ['—'];

  // Build filter spec from tab bars
  const filterSpec = {{}};
  for (const name of axisNames) {{
    if (name !== rowAxis && name !== colAxis && name in filterValues) {{
      filterSpec[name] = filterValues[name];
    }}
  }}

  let html = '<thead><tr><th class="corner-header">' + rowAxis + ' \\\\ ' + colAxis + '</th>';
  for (const cv of colVals) {{
    html += '<th>' + cv + '</th>';
  }}
  html += '</tr></thead><tbody>';

  for (const rv of rowVals) {{
    html += '<tr><td class="row-header">' + rv + '</td>';
    for (const cv of colVals) {{
      const spec = {{...filterSpec, [rowAxis]: rv, [colAxis]: cv}};
      // If rowAxis === colAxis, just use one value
      if (rowAxis === colAxis) spec[rowAxis] = rv;
      const g = findGroup(spec);
      if (g) {{
        const src = shaded ? g.shade : g.noshade;
        const imgClass = metric === -1 ? 'img-full' : 'img-clip metric-' + metric;
        html += '<td><a class="cell-link" href="' + g.name + '/index.html">'
          + '<div class="' + imgClass + '">'
          + '<img src="' + src + '" data-shade="' + g.shade + '" data-noshade="' + g.noshade + '">'
          + '</div></a></td>';
      }} else {{
        html += '<td class="empty">\\u2014</td>';
      }}
    }}
    html += '</tr>';
  }}
  html += '</tbody>';
  table.innerHTML = html;
}}

function rebuild() {{
  rowAxis = document.getElementById('row-axis').value;
  colAxis = document.getElementById('col-axis').value;
  buildTabBars();
  rebuildGrid();
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === 's' || e.key === 'S') setShade(!shaded);
}});

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
