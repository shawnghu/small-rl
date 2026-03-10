#!/usr/bin/env python3
"""Auxiliary visualization playground for exploring sweep results interactively.

Generates standalone HTML files using Plotly.js (CDN) with:
  - Individual run lines (no seed averaging / shading)
  - Per-condition checkboxes to toggle visibility
  - 3-panel layout (combined, retain, hack_freq)

Usage:
    python viz_playground.py output/slightly-less-small-bf16
    python viz_playground.py output/bigger-exclusive-routing

Output goes to output/viz_playground/{sweep_name}/:
    all_runs.html   — Every run on one page, colored by condition
    by_group.html   — Per-group sub-charts, each with individual seed lines
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from plot_routing_comparison import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    parse_routing_evals_jsonl,
)


# ============================================================
# Data loading
# ============================================================

def classify_run(run_name):
    """Return (condition_prefix, is_baseline) for a run directory name.

    Baselines start with 'baseline_', 'filter_', 'reward_penalty_', 'retain_penalty_'.
    Routing runs have no such prefix.
    """
    for prefix in ["retain_penalty_", "reward_penalty_", "filter_", "baseline_"]:
        if run_name.startswith(prefix):
            return prefix.rstrip("_"), True
    return "routing", False


def extract_seed(run_name):
    """Extract seed from run name (e.g. '..._s42' -> '42')."""
    m = re.search(r"_s(\d+)$", run_name)
    return m.group(1) if m else "?"


def load_run_timeseries(run_dir):
    """Load routing_eval.jsonl and return per-step, per-mode metrics.

    Returns: {step: {mode: {metric: value}}}
    """
    raw = parse_routing_evals_jsonl(run_dir)
    if not raw:
        return {}

    result = {}
    for step, modes in raw.items():
        result[step] = {}
        for mode, metrics in modes.items():
            combined = retain = hack_freq = None
            for k, v in metrics.items():
                if k.startswith("combined/"):
                    combined = v
                elif k.startswith("retain/"):
                    retain = v
                elif k.startswith("hack_freq/"):
                    hack_freq = v
            if combined is not None:
                result[step][mode] = {
                    "combined": combined,
                    "retain": retain if retain is not None else 0.0,
                    "hack_freq": hack_freq if hack_freq is not None else 0.0,
                }
    return result


def load_sweep(sweep_dir):
    """Load all runs from a sweep directory.

    Returns list of dicts:
        [{name, seed, condition, is_baseline, timeseries, group}, ...]
    where condition is one of: baseline, filter, reward_penalty, retain_penalty,
    and for routing runs, the modes (both, retain_only, forget_only) are separate traces.
    """
    sweep_dir = Path(sweep_dir)
    runs = []

    for entry in sorted(sweep_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name == "sweep_graphs":
            continue
        jsonl = entry / "routing_eval.jsonl"
        if not jsonl.exists():
            continue

        ts = load_run_timeseries(str(entry))
        if not ts:
            continue

        run_name = entry.name
        condition_prefix, is_baseline = classify_run(run_name)
        seed = extract_seed(run_name)

        runs.append({
            "name": run_name,
            "seed": seed,
            "condition_prefix": condition_prefix,
            "is_baseline": is_baseline,
            "timeseries": ts,
        })

    return runs


def build_traces(runs):
    """Convert loaded runs into Plotly-ready trace data.

    For baseline runs: the 'both' mode is renamed to the baseline type.
    For routing runs: each mode (both, retain_only, forget_only) is a separate trace.

    Returns list of:
        [{condition, label, color, run_name, seed, steps, combined, retain, hack_freq}, ...]
    """
    traces = []

    for run in runs:
        ts = run["timeseries"]
        steps = sorted(ts.keys())
        if not steps:
            continue

        if run["is_baseline"]:
            # Baseline: rename 'both' -> condition_prefix
            condition = run["condition_prefix"]
            combined = [ts[s].get("both", {}).get("combined", None) for s in steps]
            retain = [ts[s].get("both", {}).get("retain", None) for s in steps]
            hack_freq = [ts[s].get("both", {}).get("hack_freq", None) for s in steps]

            retain_minus_hack = [
                (r - h if r is not None and h is not None else None)
                for r, h in zip(retain, hack_freq)
            ]

            if any(v is not None for v in combined):
                traces.append({
                    "condition": condition,
                    "label": CONDITION_LABELS.get(condition, condition),
                    "color": CONDITION_COLORS.get(condition, "#888888"),
                    "run_name": run["name"],
                    "seed": run["seed"],
                    "steps": steps,
                    "combined": combined,
                    "retain": retain,
                    "hack_freq": hack_freq,
                    "retain_minus_hack": retain_minus_hack,
                })
        else:
            # Routing run: each mode is a separate trace
            for mode in ["both", "retain_only", "forget_only"]:
                combined = [ts[s].get(mode, {}).get("combined", None) for s in steps]
                retain = [ts[s].get(mode, {}).get("retain", None) for s in steps]
                hack_freq = [ts[s].get(mode, {}).get("hack_freq", None) for s in steps]
                retain_minus_hack = [
                    (r - h if r is not None and h is not None else None)
                    for r, h in zip(retain, hack_freq)
                ]

                if any(v is not None for v in combined):
                    traces.append({
                        "condition": mode,
                        "label": CONDITION_LABELS.get(mode, mode),
                        "color": CONDITION_COLORS.get(mode, "#888888"),
                        "run_name": run["name"],
                        "seed": run["seed"],
                        "steps": steps,
                        "combined": combined,
                        "retain": retain,
                        "hack_freq": hack_freq,
                        "retain_minus_hack": retain_minus_hack,
                    })

    return traces


def assign_groups(runs, sweep_dir):
    """Assign each run to an experiment group using groups_meta.json if available.

    Falls back to grouping by run name with seed stripped.
    Returns: {group_name: [run_indices]}
    """
    sweep_dir = Path(sweep_dir)
    meta_path = sweep_dir / "sweep_graphs" / "groups_meta.json"

    groups = defaultdict(list)

    if meta_path.exists():
        with open(meta_path) as f:
            group_meta = json.load(f)
        # Build a lookup: for each group, find matching runs
        # We'll match by stripping seed from run names
        for i, run in enumerate(runs):
            # Strip seed suffix for matching
            base = re.sub(r"_s\d+$", "", run["name"])
            # Also strip baseline prefixes for matching
            for prefix in ["retain_penalty_", "reward_penalty_", "filter_", "baseline_"]:
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            # Try to find a matching group
            matched = False
            for gm in group_meta:
                gm_base = re.sub(r"_s\d+$", "", gm["name"])
                # Check if our stripped base matches the group base pattern
                # Group names use different format, so try param-based matching
                pass  # Fall through to simple grouping
            # Simple grouping: strip seed
            group_key = re.sub(r"_s\d+$", "", run["name"])
            groups[group_key].append(i)
    else:
        for i, run in enumerate(runs):
            group_key = re.sub(r"_s\d+$", "", run["name"])
            groups[group_key].append(i)

    return dict(groups)


def _is_subsequence(short_parts, long_parts):
    """Check if short_parts appears as an ordered subsequence of long_parts."""
    it = iter(long_parts)
    return all(part in it for part in short_parts)


def match_baseline_to_routing(groups):
    """Match baseline groups to their routing counterparts.

    Baselines omit routing-specific params (e.g. coherence) from their
    names, so the baseline stripped key is a subsequence of the routing key.
    Each baseline is matched to all routing groups that contain its tokens.

    Returns: {routing_group_key: [routing_group_key, baseline_group_key, ...]}
    """
    baseline_prefixes = ["baseline_", "filter_", "reward_penalty_", "retain_penalty_"]
    routing_keys = []
    baseline_entries = []  # (stripped_key, full_key)

    for key in groups:
        is_bl = False
        for prefix in baseline_prefixes:
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                baseline_entries.append((stripped, key))
                is_bl = True
                break
        if not is_bl:
            routing_keys.append(key)

    # Build merged groups: routing + matched baselines
    merged = {}
    used_baselines = set()
    for rk in routing_keys:
        merged[rk] = [rk]
        rk_parts = rk.split("_")
        for stripped, full_key in baseline_entries:
            bl_parts = stripped.split("_")
            if bl_parts == rk_parts or _is_subsequence(bl_parts, rk_parts):
                merged[rk].append(full_key)
                used_baselines.add(full_key)

    # Add orphan baselines as their own groups
    for stripped, full_key in baseline_entries:
        if full_key not in used_baselines:
            merged[full_key] = [full_key]

    return merged


# ============================================================
# HTML generation
# ============================================================

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

METRIC_PANELS = [
    ("combined", "Combined Reward"),
    ("retain", "Retain Reward"),
    ("hack_freq", "Hack Frequency"),
    ("retain_minus_hack", "Retain \u2212 Hack"),
]

def _build_seed_opacity_map(traces):
    """Assign each seed a distinct opacity level.

    Returns: {seed_str: float} with opacities spread from 1.0 down to 0.35.
    """
    seeds = sorted(set(t["seed"] for t in traces), key=lambda s: (len(s), s))
    n = len(seeds)
    if n <= 1:
        return {s: 0.7 for s in seeds}
    # Spread from 1.0 (first seed) to 0.35 (last seed)
    return {s: 1.0 - 0.65 * i / (n - 1) for i, s in enumerate(seeds)}


def _hex_to_rgba(hex_color, opacity):
    """Convert '#RRGGBB' + opacity to 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity:.2f})"


def _traces_to_plotly_json(traces):
    """Convert trace dicts to Plotly-compatible JSON data structures.

    Seeds are distinguished by opacity (darker = first seed, lighter = last).
    All lines are solid to minimize visual noise.

    Returns a list of plotly trace objects (one per trace per metric panel).
    Also returns the set of conditions present and the sorted seed list.
    """
    conditions_seen = set()
    seed_opacity_map = _build_seed_opacity_map(traces)
    seeds_seen = sorted(seed_opacity_map.keys(), key=lambda s: (len(s), s))
    plotly_data = {metric_key: [] for metric_key, _ in METRIC_PANELS}

    for trace in traces:
        cond = trace["condition"]
        seed = trace["seed"]
        conditions_seen.add(cond)
        opacity = seed_opacity_map[seed]
        rgba = _hex_to_rgba(trace["color"], opacity)
        # Pack all metrics per point for cross-metric hover in grid mode
        customdata = list(zip(
            trace["combined"], trace["retain"], trace["hack_freq"],
            trace["retain_minus_hack"],
        ))
        for metric_key, _ in METRIC_PANELS:
            vals = trace[metric_key]
            plotly_data[metric_key].append({
                "x": trace["steps"],
                "y": vals,
                "customdata": customdata,
                "type": "scatter",
                "mode": "lines",
                "name": f"{trace['label']} (s{seed})",
                "legendgroup": cond,
                "line": {"color": rgba, "width": 1.5},
                "hovertemplate": (
                    f"<b>{trace['run_name']}</b><br>"
                    f"Step: %{{x}}<br>{metric_key}: %{{y:.3f}}<extra></extra>"
                ),
                "_condition": cond,
                "_seed": seed,
                "_base_color": trace["color"],
                "_base_opacity": opacity,
            })

    return plotly_data, conditions_seen, seeds_seen


def _seed_checkbox_html(seeds):
    """Build HTML for per-seed checkboxes with opacity swatch."""
    seed_opacity_map = _build_seed_opacity_map(
        [{"seed": s} for s in seeds]
    )
    parts = []
    for seed in seeds:
        opacity = seed_opacity_map[seed]
        svg = (
            f'<svg width="28" height="12" style="vertical-align:middle">'
            f'<line x1="0" y1="6" x2="28" y2="6" '
            f'stroke="rgba(0,0,0,{opacity:.2f})" stroke-width="3"/></svg>'
        )
        parts.append(
            f'<label class="cb-label cb-seed">'
            f'<input type="checkbox" checked data-seed="{seed}" '
            f'onchange="toggleSeed(this)"> {svg} seed {seed}'
            f'</label>'
        )
    return parts



def generate_all_runs_html(traces, sweep_name, output_path):
    """Generate a single HTML page with all runs plotted on 3 panels.

    Uses Plotly.js with checkbox toggles per condition and per seed.
    """
    plotly_data, conditions_seen, seeds = _traces_to_plotly_json(traces)

    # Build condition list in canonical order
    cond_order = ["baseline", "filter", "reward_penalty", "retain_penalty",
                  "both", "retain_only", "forget_only"]
    conditions = [c for c in cond_order if c in conditions_seen]

    # Build the HTML
    panels_html = []
    panels_js = []
    for i, (metric_key, metric_title) in enumerate(METRIC_PANELS):
        div_id = f"panel-{metric_key}"
        panels_html.append(f'<div id="{div_id}" class="panel"></div>')
        panels_js.append({
            "div_id": div_id,
            "title": metric_title,
            "traces": plotly_data[metric_key],
        })

    # Condition checkbox HTML
    cond_checkbox_html = []
    for cond in conditions:
        color = CONDITION_COLORS.get(cond, "#888")
        label = CONDITION_LABELS.get(cond, cond)
        cond_checkbox_html.append(
            f'<label class="cb-label" style="border-left: 4px solid {color};">'
            f'<input type="checkbox" checked data-condition="{cond}" '
            f'onchange="toggleCondition(this)"> {label}'
            f'</label>'
        )

    # Seed checkbox HTML (only if <= 10 seeds)
    seed_checkbox_html = _seed_checkbox_html(seeds) if len(seeds) <= 10 else []

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>All Runs — {sweep_name}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #fafafa; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 5px; }}
  .subtitle {{ text-align: center; color: #777; font-size: 14px; margin-bottom: 15px; }}
  .filter-bar {{
    max-width: 1100px; margin: 0 auto 12px auto;
    background: white; border: 1px solid #ddd; border-radius: 6px;
    padding: 10px 14px;
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
  .panels {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
             max-width: 1800px; margin: 0 auto; }}
  .panel {{ flex: 1 1 550px; min-width: 400px; max-width: 600px; height: 450px; }}
  .btn-row {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 10px; }}
  .btn-row button {{ padding: 5px 14px; font-size: 13px; cursor: pointer;
    border: 1px solid #ccc; border-radius: 4px; background: #fff; }}
  .btn-row button:hover {{ background: #e8e8e8; }}
</style>
</head>
<body>
<h1>All Runs — {sweep_name}</h1>
<div class="subtitle">{len(traces)} traces from {len(set(t['run_name'] for t in traces))} runs</div>
<div class="btn-row">
  <button onclick="selectAll(true)">Select All</button>
  <button onclick="selectAll(false)">Deselect All</button>
</div>
<div class="filter-bar">
  <div class="filter-row">
    <span class="filter-label">Condition</span>
    {chr(10).join(cond_checkbox_html)}
  </div>
  {"<div class='filter-row'><span class='filter-label'>Seed</span>" + chr(10).join(seed_checkbox_html) + "</div>" if seed_checkbox_html else ""}
</div>
<div class="panels">
  {chr(10).join(panels_html)}
</div>

<script>
const panelData = {json.dumps(panels_js)};
const conditionOrder = {json.dumps(conditions)};
const seedOrder = {json.dumps(seeds)};

const condVisible = {{}};
conditionOrder.forEach(c => condVisible[c] = true);
const seedVisible = {{}};
seedOrder.forEach(s => seedVisible[s] = true);

const DIM_OPACITY = 0.12;
const HIGHLIGHT_WIDTH = 3.0;
let highlightedSeed = null;
let syncing = false;  // prevent hover recursion across panels

function hexToRgba(hex, a) {{
  const h = hex.replace('#','');
  return `rgba(${{parseInt(h.slice(0,2),16)}},${{parseInt(h.slice(2,4),16)}},${{parseInt(h.slice(4,6),16)}},${{a.toFixed(2)}})`;
}}

function renderAll() {{
  for (const panel of panelData) {{
    const traces = panel.traces.map(t => ({{
      x: t.x, y: t.y, type: t.type, mode: t.mode, name: t.name,
      legendgroup: t.legendgroup,
      line: t.line, hovertemplate: t.hovertemplate,
      showlegend: false,
    }}));
    const layout = {{
      title: {{ text: panel.title, font: {{ size: 15 }} }},
      xaxis: {{ title: 'Training Step' }},
      yaxis: {{ title: panel.title, range: panel.div_id.endsWith('retain_minus_hack') ? [-1.1, 1.1] : [-0.05, 1.1] }},
      margin: {{ t: 40, b: 50, l: 55, r: 20 }},
      hovermode: 'closest',
    }};
    const div = document.getElementById(panel.div_id);
    Plotly.newPlot(div, traces, layout, {{responsive: true}});

    div.on('plotly_hover', (evt) => {{
      if (syncing) return;
      const pt = evt.points[0];
      const ci = pt.curveNumber;
      const seed = panel.traces[ci]._seed;
      if (seed !== highlightedSeed) highlightSeed(seed);
      // Sync hover tooltip to sibling panels
      syncing = true;
      for (const other of panelData) {{
        if (other.div_id === panel.div_id) continue;
        const otherDiv = document.getElementById(other.div_id);
        try {{ Plotly.Fx.hover(otherDiv, [{{curveNumber: ci, pointNumber: pt.pointNumber}}]); }}
        catch(e) {{}}
      }}
      syncing = false;
    }});
    div.on('plotly_unhover', () => {{
      if (syncing) return;
      if (highlightedSeed !== null) clearHighlight();
      syncing = true;
      for (const other of panelData) {{
        if (other.div_id === panel.div_id) continue;
        const otherDiv = document.getElementById(other.div_id);
        try {{ Plotly.Fx.unhover(otherDiv); }} catch(e) {{}}
      }}
      syncing = false;
    }});
  }}
}}

function highlightSeed(seed) {{
  highlightedSeed = seed;
  for (const panel of panelData) {{
    const div = document.getElementById(panel.div_id);
    const colors = [], widths = [];
    for (const t of panel.traces) {{
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

function clearHighlight() {{
  highlightedSeed = null;
  for (const panel of panelData) {{
    const div = document.getElementById(panel.div_id);
    const colors = [], widths = [];
    for (const t of panel.traces) {{
      colors.push(hexToRgba(t._base_color, t._base_opacity));
      widths.push(1.5);
    }}
    Plotly.restyle(div, {{'line.color': colors, 'line.width': widths}});
  }}
}}

function applyVisibility() {{
  for (const panel of panelData) {{
    const div = document.getElementById(panel.div_id);
    const vis = panel.traces.map(t =>
      (condVisible[t._condition] && seedVisible[t._seed]) ? true : false
    );
    Plotly.restyle(div, {{ visible: vis }});
  }}
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

renderAll();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Generated: {output_path}")


def generate_by_group_html(runs, traces, sweep_dir, sweep_name, output_path,
                           page_title=None):
    """Generate an HTML page with per-group panels and per-condition toggles.

    Each experiment group gets its own row of 3 charts.
    page_title overrides the default "By Group — {sweep_name}" heading.
    """
    heading = page_title or f"By Group — {sweep_name}"
    groups = assign_groups(runs, sweep_dir)
    merged = match_baseline_to_routing(groups)

    # Build trace index by run name for fast lookup
    trace_by_run = defaultdict(list)
    for t in traces:
        trace_by_run[t["run_name"]].append(t)

    # Build group sections
    group_sections = []
    all_seeds = set()
    for group_label, member_keys in sorted(merged.items()):
        group_traces = []
        for mk in member_keys:
            if mk in groups:
                for run_idx in groups[mk]:
                    run = runs[run_idx]
                    group_traces.extend(trace_by_run[run["name"]])

        if not group_traces:
            continue

        plotly_data, conditions_seen, seeds_seen = _traces_to_plotly_json(group_traces)
        all_seeds.update(seeds_seen)
        group_sections.append({
            "label": group_label,
            "n_runs": len(set(t["run_name"] for t in group_traces)),
            "panels": [
                {
                    "div_id": f"g-{len(group_sections)}-{mk}",
                    "title": mt,
                    "traces": plotly_data[mk],
                }
                for mk, mt in METRIC_PANELS
            ],
            "conditions": [c for c in
                           ["baseline", "filter", "reward_penalty", "retain_penalty",
                            "both", "retain_only", "forget_only"]
                           if c in conditions_seen],
        })

    # Build condition set across all groups
    all_conditions = set()
    for gs in group_sections:
        all_conditions.update(gs["conditions"])
    cond_order = ["baseline", "filter", "reward_penalty", "retain_penalty",
                  "both", "retain_only", "forget_only"]
    all_conditions = [c for c in cond_order if c in all_conditions]
    all_seeds_sorted = sorted(all_seeds, key=lambda s: (len(s), s))

    # Global condition checkbox HTML
    cond_checkbox_html = []
    for cond in all_conditions:
        color = CONDITION_COLORS.get(cond, "#888")
        label = CONDITION_LABELS.get(cond, cond)
        cond_checkbox_html.append(
            f'<label class="cb-label" style="border-left: 4px solid {color};">'
            f'<input type="checkbox" checked data-condition="{cond}" '
            f'onchange="toggleConditionGlobal(this)"> {label}'
            f'</label>'
        )

    # Seed checkbox HTML (only if <= 10 seeds)
    seed_checkbox_html = _seed_checkbox_html(all_seeds_sorted) if len(all_seeds_sorted) <= 10 else []

    # Group HTML
    groups_html = []
    for gs in group_sections:
        panels_divs = "\n".join(
            f'<div id="{p["div_id"]}" class="panel"></div>' for p in gs["panels"]
        )
        display_label = gs["label"].replace("_", " ")
        groups_html.append(f"""
<div class="group-section">
  <h2>{display_label} <span class="run-count">({gs['n_runs']} runs)</span></h2>
  <div class="panels">{panels_divs}</div>
</div>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{heading}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #fafafa; margin: 0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 5px; }}
  .subtitle {{ text-align: center; color: #777; font-size: 14px; margin-bottom: 15px; }}
  .filter-bar {{
    position: sticky; top: 0; z-index: 100;
    max-width: 1100px; margin: 0 auto 12px auto;
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
  .btn-row {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 10px; }}
  .btn-row button {{ padding: 5px 14px; font-size: 13px; cursor: pointer;
    border: 1px solid #ccc; border-radius: 4px; background: #fff; }}
  .btn-row button:hover {{ background: #e8e8e8; }}
  .group-section {{
    max-width: 1800px; margin: 0 auto 30px auto;
    background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px;
  }}
  .group-section h2 {{ margin: 0 0 10px 0; font-size: 16px; }}
  .run-count {{ color: #999; font-weight: normal; font-size: 14px; }}
  .panels {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }}
  .panel {{ flex: 1 1 400px; min-width: 350px; max-width: 580px; height: 380px; }}
</style>
</head>
<body>
<h1>{heading}</h1>
<div class="subtitle">{len(group_sections)} groups, {len(runs)} total runs</div>
<div class="btn-row">
  <button onclick="selectAll(true)">Select All</button>
  <button onclick="selectAll(false)">Deselect All</button>
</div>
<div class="filter-bar">
  <div class="filter-row">
    <span class="filter-label">Condition</span>
    {chr(10).join(cond_checkbox_html)}
  </div>
  {"<div class='filter-row'><span class='filter-label'>Seed</span>" + chr(10).join(seed_checkbox_html) + "</div>" if seed_checkbox_html else ""}
</div>
{chr(10).join(groups_html)}

<script>
const allPanels = {json.dumps([p for gs in group_sections for p in gs["panels"]])};
const conditionOrder = {json.dumps(all_conditions)};
const seedOrder = {json.dumps(all_seeds_sorted)};

const condVisible = {{}};
conditionOrder.forEach(c => condVisible[c] = true);
const seedVisible = {{}};
seedOrder.forEach(s => seedVisible[s] = true);

const DIM_OPACITY = 0.12;
const HIGHLIGHT_WIDTH = 3.0;
let highlightedSeed = null;
let syncing = false;  // prevent hover recursion across panels

function hexToRgba(hex, a) {{
  const h = hex.replace('#','');
  return `rgba(${{parseInt(h.slice(0,2),16)}},${{parseInt(h.slice(2,4),16)}},${{parseInt(h.slice(4,6),16)}},${{a.toFixed(2)}})`;
}}

// Track which panels have been rendered
const rendered = new Set();

// Map panel group prefix (e.g. "g-3") to its sibling panel IDs, for
// cross-highlighting the 3 metric panels within one group section.
const groupPeers = {{}};  // div_id -> [div_id, ...]
(function buildGroupPeers() {{
  const byPrefix = {{}};
  for (const p of allPanels) {{
    // div_id looks like "g-3-combined", prefix is "g-3"
    const prefix = p.div_id.replace(/-[^-]+$/, '');
    if (!byPrefix[prefix]) byPrefix[prefix] = [];
    byPrefix[prefix].push(p.div_id);
  }}
  for (const ids of Object.values(byPrefix)) {{
    for (const id of ids) groupPeers[id] = ids;
  }}
}})();

const panelById = {{}};
allPanels.forEach(p => panelById[p.div_id] = p);

function renderPanel(panel) {{
  if (rendered.has(panel.div_id)) return;
  rendered.add(panel.div_id);
  const traces = panel.traces.map(t => ({{
    x: t.x, y: t.y, type: t.type, mode: t.mode, name: t.name,
    legendgroup: t.legendgroup,
    line: t.line, hovertemplate: t.hovertemplate,
    showlegend: false,
    visible: (condVisible[t._condition] && seedVisible[t._seed]) ? true : false,
  }}));
  const layout = {{
    title: {{ text: panel.title, font: {{ size: 14 }} }},
    xaxis: {{ title: 'Step' }},
    yaxis: {{ range: panel.div_id.endsWith('retain_minus_hack') ? [-1.1, 1.1] : [-0.05, 1.1] }},
    margin: {{ t: 35, b: 45, l: 50, r: 15 }},
    hovermode: 'closest',
  }};
  const div = document.getElementById(panel.div_id);
  Plotly.newPlot(div, traces, layout, {{responsive: true}});

  div.on('plotly_hover', (evt) => {{
    if (syncing) return;
    const pt = evt.points[0];
    const ci = pt.curveNumber;
    const seed = panel.traces[ci]._seed;
    if (seed !== highlightedSeed) highlightSeedInGroup(seed, panel.div_id);
    // Sync hover tooltip to sibling panels in same group
    syncing = true;
    const peerIds = groupPeers[panel.div_id] || [];
    for (const peerId of peerIds) {{
      if (peerId === panel.div_id || !rendered.has(peerId)) continue;
      const peerDiv = document.getElementById(peerId);
      try {{ Plotly.Fx.hover(peerDiv, [{{curveNumber: ci, pointNumber: pt.pointNumber}}]); }}
      catch(e) {{}}
    }}
    syncing = false;
  }});
  div.on('plotly_unhover', () => {{
    if (syncing) return;
    if (highlightedSeed !== null) clearHighlightInGroup(panel.div_id);
    syncing = true;
    const peerIds = groupPeers[panel.div_id] || [];
    for (const peerId of peerIds) {{
      if (peerId === panel.div_id || !rendered.has(peerId)) continue;
      const peerDiv = document.getElementById(peerId);
      try {{ Plotly.Fx.unhover(peerDiv); }} catch(e) {{}}
    }}
    syncing = false;
  }});
}}

function highlightSeedInGroup(seed, sourceDivId) {{
  highlightedSeed = seed;
  const peerIds = groupPeers[sourceDivId] || [sourceDivId];
  for (const divId of peerIds) {{
    if (!rendered.has(divId)) continue;
    const panel = panelById[divId];
    const div = document.getElementById(divId);
    const colors = [], widths = [];
    for (const t of panel.traces) {{
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
    const panel = panelById[divId];
    const div = document.getElementById(divId);
    const colors = [], widths = [];
    for (const t of panel.traces) {{
      colors.push(hexToRgba(t._base_color, t._base_opacity));
      widths.push(1.5);
    }}
    Plotly.restyle(div, {{'line.color': colors, 'line.width': widths}});
  }}
}}

// Lazy render: only create charts when scrolled into view
const onScreen = new Set();
const needsRestyled = new Set();  // rendered panels that missed a visibility update

const observer = new IntersectionObserver((entries) => {{
  for (const entry of entries) {{
    const id = entry.target.id;
    if (entry.isIntersecting) {{
      onScreen.add(id);
      if (panelById[id]) {{
        renderPanel(panelById[id]);
        // Apply deferred visibility update if checkbox changed while off-screen
        if (needsRestyled.has(id)) {{
          needsRestyled.delete(id);
          restylePanel(id);
        }}
      }}
    }} else {{
      onScreen.delete(id);
    }}
  }}
}}, {{ rootMargin: '200px' }});

allPanels.forEach(p => {{
  const el = document.getElementById(p.div_id);
  if (el) observer.observe(el);
}});

function restylePanel(divId) {{
  const panel = panelById[divId];
  if (!panel) return;
  const div = document.getElementById(divId);
  const vis = panel.traces.map(t =>
    (condVisible[t._condition] && seedVisible[t._seed]) ? true : false
  );
  Plotly.restyle(div, {{ visible: vis }});
}}

function applyVisibility() {{
  // Only restyle on-screen panels immediately; defer off-screen ones
  for (const divId of rendered) {{
    if (onScreen.has(divId)) {{
      restylePanel(divId);
    }} else {{
      needsRestyled.add(divId);
    }}
  }}
}}

function toggleConditionGlobal(checkbox) {{
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
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Generated: {output_path}")


def generate_index_html(sweep_name, output_dir):
    """Generate a simple index page linking to all visualizations."""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Viz Playground — {sweep_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #fafafa; margin: 40px; }}
  h1 {{ margin-bottom: 10px; }}
  .subtitle {{ color: #777; margin-bottom: 30px; }}
  .links {{ display: flex; flex-direction: column; gap: 12px; max-width: 500px; }}
  a {{ display: block; padding: 14px 20px; background: white; border: 1px solid #ddd;
       border-radius: 8px; text-decoration: none; color: #333; font-size: 16px; }}
  a:hover {{ background: #f0f0ff; border-color: #99f; }}
  a .desc {{ font-size: 13px; color: #888; margin-top: 4px; }}
</style>
</head>
<body>
<h1>Viz Playground</h1>
<div class="subtitle">{sweep_name}</div>
<div class="links">
  <a href="all_runs.html">
    All Runs (flat)
    <div class="desc">Every individual run on one page, colored by condition. Checkboxes per condition.</div>
  </a>
  <a href="by_group.html">
    By Group
    <div class="desc">One row per experiment group, individual seed lines visible. Global condition toggles.</div>
  </a>
</div>
</body>
</html>"""

    path = os.path.join(output_dir, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Generated: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML visualizations for sweep results"
    )
    parser.add_argument("sweep_dir", help="Path to sweep output directory (e.g. output/slightly-less-small-bf16)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: output/viz_playground/{sweep_name})")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.rstrip("/")
    sweep_name = os.path.basename(sweep_dir)

    output_dir = args.output_dir or os.path.join("output", "viz_playground", sweep_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading runs from {sweep_dir}...")
    runs = load_sweep(sweep_dir)
    if not runs:
        print(f"No runs with routing_eval.jsonl found in {sweep_dir}")
        sys.exit(1)

    print(f"  Found {len(runs)} runs")
    traces = build_traces(runs)
    print(f"  Built {len(traces)} traces")

    conditions = set(t["condition"] for t in traces)
    print(f"  Conditions: {', '.join(sorted(conditions))}")

    print(f"\nGenerating HTML in {output_dir}/")
    generate_all_runs_html(traces, sweep_name, os.path.join(output_dir, "all_runs.html"))
    generate_by_group_html(runs, traces, sweep_dir, sweep_name, os.path.join(output_dir, "by_group.html"))
    generate_index_html(sweep_name, output_dir)

    print(f"\nDone! Serve with:")
    print(f"  python -m http.server -d {output_dir}")


if __name__ == "__main__":
    main()
