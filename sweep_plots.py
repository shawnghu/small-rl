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
    extract_routing_metrics,
    aggregate_seeds,
    plot_routing_chart,
)


def build_step_data(routing_runs, baseline_runs, step, combined_key, task_key,
                    no_baseline=False):
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
        data = extract_routing_metrics(run_dir, step, task_key, combined_key)
        if data:
            routing_seed_results.append(data)

    # Collect baseline data across seeds
    baseline_seed_results = []
    if not no_baseline:
        for run_dir in baseline_runs:
            data = extract_routing_metrics(run_dir, step, task_key, combined_key)
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
                                     output_dir, combined_key, task_key,
                                     group_name="default", no_baseline=False):
    """Generate per-step bar charts and animated GIF for an experiment group.

    Args:
        routing_runs: list of routing run directories (one per seed)
        baseline_runs: list of baseline run directories (one per seed)
        reward: reward function name (for chart title)
        output_dir: base output directory
        combined_key: metric key for combined reward
        task_key: metric key for task-only reward
        group_name: name for this group's output subdirectory
        no_baseline: if True, skip baseline data entirely
    """
    # Find union of all eval steps across routing runs
    all_steps = set()
    for run_dir in routing_runs:
        log_path = os.path.join(run_dir, "train.log")
        evals = parse_routing_evals(log_path)
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
            routing_runs, baseline_runs, step, combined_key, task_key,
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
            routing_runs, baseline_runs, step, combined_key, task_key,
            no_baseline=no_baseline,
        )
        for mode, metrics in plot_data.items():
            if mode not in time_series:
                time_series[mode] = {m: [] for m in ["combined", "task", "hack_freq"]}
            for metric_key in ["combined", "task", "hack_freq"]:
                mean, std = metrics[metric_key]
                time_series[mode][metric_key].append((step, mean, std))

    if image_paths:
        gif_path = str(graph_dir / "animation.gif")
        generate_gif(image_paths, gif_path)

    # Generate line graphs
    lines_path = None
    if time_series:
        lines_path = str(graph_dir / "lines_over_time.png")
        title = reward.replace("_", " ").title()
        if group_name != "default":
            title += f" ({group_name.replace('_', ' ')})"
        generate_line_graphs(time_series, lines_path, title=title,
                             n_seeds=max(len(routing_runs), len(baseline_runs)))

    if image_paths or lines_path:
        html_path = str(graph_dir / "index.html")
        generate_html_viewer(image_paths, steps, html_path,
                             lines_image="lines_over_time.png" if lines_path else None)
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


def generate_line_graphs(time_series, output_path, title="", n_seeds=None):
    """Generate 3 line graphs (proxy reward, task reward, hack_freq) over time.

    Args:
        time_series: {mode: {metric: [(step, mean, std), ...]}}
        output_path: where to save the PNG
        title: chart title
        n_seeds: number of seeds (for annotation)
    """
    metric_configs = [
        ("combined", "Proxy Reward"),
        ("task", "Task Reward"),
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

            color = CONDITION_COLORS[mode]
            label = CONDITION_LABELS[mode]
            ax.plot(steps_arr, means, color=color, label=label, linewidth=2)
            ax.fill_between(steps_arr, means - stds, means + stds,
                            color=color, alpha=0.15)

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


def generate_html_viewer(image_paths, steps, output_path, lines_image=None):
    """Generate an interactive HTML viewer with slider for per-step charts.

    Works when served via HTTP (e.g. python -m http.server).
    Uses relative paths so the HTML file can live alongside the PNGs.
    Includes a tabbed view for per-step bar charts and line-over-time graphs.
    """
    # Build list of relative image filenames
    filenames = [os.path.basename(p) for p in image_paths]
    step_labels = [str(s) for s in steps[:len(filenames)]]

    lines_section = ""
    if lines_image:
        lines_section = f"""
  <div id="lines-view" style="display:none;">
    <img src="{lines_image}" style="max-width:100%; border:1px solid #ccc; border-radius:4px;">
  </div>"""

    tab_buttons = ""
    tab_script = ""
    if lines_image:
        tab_buttons = """
  <div class="tabs">
    <button class="tab active" onclick="showTab('steps')">Per-Step Charts</button>
    <button class="tab" onclick="showTab('lines')">Over Time</button>
  </div>"""
        tab_script = """
function showTab(which) {
  document.getElementById('steps-view').style.display = which === 'steps' ? '' : 'none';
  document.getElementById('lines-view').style.display = which === 'lines' ? '' : 'none';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
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
</style>
</head>
<body>
<div class="container">
  {tab_buttons}
  <div id="steps-view">
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
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
