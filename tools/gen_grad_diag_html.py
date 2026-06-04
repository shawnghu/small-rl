#!/usr/bin/env python
"""Generate an interactive viewer for per-sample gradient diagnostics.

Reads a run's grad_diag.jsonl (written by train.py's _run_grad_diagnostic) and
emits a self-contained grad_diag.html with:

  - a step slider (over diagnostic steps),
  - a layer selector ("all layers / whole-model" + each layer),
  - the four 2x2 distributions overlaid as histograms: {retain, forget} params
    x {retain (is_rh=0), forget (is_rh=1)} samples — the per-sample grad-norm
    densities (cf. https://alignment.anthropic.com/2025/selective-gradient-masking/),
  - a per-layer line plot at the selected step: the authoritative ||.grad||
    aggregate (all samples) per role, plus the mean per-sample norm per 2x2 cell.

Usage:
  python tools/gen_grad_diag_html.py output/<run>/            # finds grad_diag.jsonl
  python tools/gen_grad_diag_html.py path/to/grad_diag.jsonl -o out.html
"""

import argparse
import json
import os

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# param color family (hue), sample subset (shade): retain=blue, forget=red.
COLORS = {
    "rr": "#1f3a93",  # retain params, retain samples
    "rf": "#6fa8dc",  # retain params, forget samples
    "fr": "#990000",  # forget params, retain samples
    "ff": "#e06666",  # forget params, forget samples
}


def _load(jsonl_path):
    recs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["step"])
    return recs


def build_html(records):
    data_json = json.dumps(records)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Per-sample gradient diagnostic</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  body {{ font-family: sans-serif; background: #f5f5f5; margin: 18px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .controls {{ display: flex; align-items: center; gap: 14px; margin: 12px 0; flex-wrap: wrap; }}
  input[type=range] {{ width: 380px; }}
  select, button {{ padding: 5px 10px; font-size: 14px; }}
  #step-label {{ font-weight: bold; min-width: 230px; }}
  .hint {{ color: #888; font-size: 12px; }}
  h2 {{ font-size: 16px; margin: 18px 0 4px; }}
</style>
</head>
<body>
<div class="container">
  <h1 style="font-size:20px;">Per-sample gradient norms — 2&times;2 (params &times; samples)</h1>
  <div class="controls">
    <button onclick="prev()">&larr;</button>
    <input type="range" id="slider" min="0" max="{len(records) - 1}" value="0" oninput="update(this.value)">
    <button onclick="next()">&rarr;</button>
    <button id="play" onclick="togglePlay()">Play</button>
    <label>Layer:
      <select id="layer" onchange="render()"></select>
    </label>
    <span id="step-label"></span>
  </div>
  <div class="hint">Arrow keys: prev/next &middot; legend entries toggle traces. Param hue: blue=retain, red=forget. Shade: dark=on retain samples, light=on forget samples.</div>
  <h2>Distribution of per-sample grad norms</h2>
  <div id="hist" style="height:430px;"></div>
  <h2>Per-layer aggregate (at this step)</h2>
  <div id="lines" style="height:380px;"></div>
</div>
<script>
const RECORDS = {data_json};
let idx = 0, playing = false, timer = null;

function curr() {{ return RECORDS[idx]; }}

function initLayerSelect() {{
  const sel = document.getElementById('layer');
  const layers = curr().layers;
  let opts = '<option value="all">all layers (whole-model)</option>';
  for (const li of layers) opts += `<option value="${{li}}">layer ${{li}}</option>`;
  sel.innerHTML = opts;
}}

function pick(arr, isrh, cls) {{ return arr.filter((_, j) => isrh[j] === cls); }}

function arraysFor(rec, layerSel) {{
  const isrh = rec.is_rh;
  let retain, forget;
  if (layerSel === 'all') {{ retain = rec.whole_model.retain; forget = rec.whole_model.forget; }}
  else {{
    const k = rec.layers.indexOf(parseInt(layerSel));
    retain = rec.per_sample.retain[k]; forget = rec.per_sample.forget[k];
  }}
  return {{
    rr: pick(retain, isrh, 0), rf: pick(retain, isrh, 1),
    fr: pick(forget, isrh, 0), ff: pick(forget, isrh, 1),
  }};
}}

function mean(a) {{ return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }}

function renderHist() {{
  const rec = curr();
  const sel = document.getElementById('layer').value;
  const a = arraysFor(rec, sel);
  const mk = (key, name) => ({{
    x: a[key], type: 'histogram', name: name, opacity: 0.55,
    marker: {{ color: COLORS_JS[key] }}, nbinsx: 30,
  }});
  const traces = [
    mk('rr', 'retain params · retain samples'),
    mk('rf', 'retain params · forget samples'),
    mk('fr', 'forget params · retain samples'),
    mk('ff', 'forget params · forget samples'),
  ];
  Plotly.react('hist', traces, {{
    barmode: 'overlay', bargap: 0.02,
    xaxis: {{ title: 'per-sample grad norm' }}, yaxis: {{ title: 'count' }},
    margin: {{ t: 10, r: 10 }}, legend: {{ orientation: 'h', y: 1.15 }},
  }}, {{ displayModeBar: false }});
}}

function renderLines() {{
  const rec = curr();
  const x = rec.layers;
  const isrh = rec.is_rh;
  // mean per-sample norm per 2x2 cell, per layer
  const cell = {{ rr: [], rf: [], fr: [], ff: [] }};
  for (let k = 0; k < x.length; k++) {{
    const r = rec.per_sample.retain[k], f = rec.per_sample.forget[k];
    cell.rr.push(mean(pick(r, isrh, 0))); cell.rf.push(mean(pick(r, isrh, 1)));
    cell.fr.push(mean(pick(f, isrh, 0))); cell.ff.push(mean(pick(f, isrh, 1)));
  }}
  const line = (key, name, dash) => ({{
    x: x, y: cell[key], name: name, mode: 'lines+markers',
    line: {{ color: COLORS_JS[key], dash: dash }},
  }});
  const agg = (arr, name, color) => ({{
    x: x, y: arr, name: name, mode: 'lines', line: {{ color: color, width: 3 }},
  }});
  const traces = [
    agg(rec.aggregate_grad_norm.retain, 'retain ||.grad|| (all)', '#1f3a93'),
    agg(rec.aggregate_grad_norm.forget, 'forget ||.grad|| (all)', '#990000'),
    line('rr', 'mean: retain·retain'), line('rf', 'mean: retain·forget'),
    line('fr', 'mean: forget·retain'), line('ff', 'mean: forget·forget'),
  ];
  Plotly.react('lines', traces, {{
    xaxis: {{ title: 'layer' }}, yaxis: {{ title: 'grad norm' }},
    margin: {{ t: 10, r: 10 }}, legend: {{ orientation: 'h', y: 1.18 }},
  }}, {{ displayModeBar: false }});
}}

const COLORS_JS = {json.dumps(COLORS)};

function render() {{ renderHist(); renderLines(); }}

function update(i) {{
  idx = parseInt(i);
  document.getElementById('slider').value = idx;
  const rec = curr();
  document.getElementById('step-label').textContent =
    `step ${{rec.step}} · ${{rec.n_samples}} samples · check ${{rec.grad_check.max_triangle_ratio.toFixed(3)}}`;
  // layer set is stable across steps; only (re)build if option count changed
  const sel = document.getElementById('layer');
  if (sel.options.length !== rec.layers.length + 1) {{ initLayerSelect(); }}
  render();
}}
function prev() {{ if (idx > 0) update(idx - 1); }}
function next() {{ if (idx < RECORDS.length - 1) update(idx + 1); }}
function togglePlay() {{
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) timer = setInterval(() => {{
    if (idx < RECORDS.length - 1) next(); else {{ togglePlay(); }}
  }}, 700);
  else clearInterval(timer);
}}
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') prev(); else if (e.key === 'ArrowRight') next();
}});

initLayerSelect();
update(0);
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run dir containing grad_diag.jsonl, or the jsonl itself")
    ap.add_argument("-o", "--output", default=None, help="output html path")
    args = ap.parse_args()

    jsonl = args.path
    if os.path.isdir(jsonl):
        jsonl = os.path.join(jsonl, "grad_diag.jsonl")
    assert os.path.exists(jsonl), f"no grad_diag.jsonl at {jsonl}"

    records = _load(jsonl)
    assert records, f"{jsonl} is empty"

    out = args.output or os.path.join(os.path.dirname(jsonl), "grad_diag.html")
    with open(out, "w") as f:
        f.write(build_html(records))
    print(f"[grad_diag] wrote {out} ({len(records)} diagnostic steps, "
          f"{records[0]['n_samples']} samples, {len(records[0]['layers'])} layers)")


if __name__ == "__main__":
    main()
