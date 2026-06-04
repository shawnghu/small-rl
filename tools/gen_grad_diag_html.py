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
    colors_json = json.dumps(COLORS)
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
  input[type=range] {{ width: 360px; }}
  select, button {{ padding: 5px 10px; font-size: 14px; }}
  #step-label {{ font-weight: bold; min-width: 230px; }}
  .hint {{ color: #888; font-size: 12px; }}
  h2 {{ font-size: 16px; margin: 18px 0 4px; }}
  label.ck {{ font-size: 14px; }}
</style>
</head>
<body>
<div class="container">
  <h1 style="font-size:20px;">Per-sample gradient norms &mdash; 2&times;2 (params &times; samples)</h1>
  <div class="controls">
    <button onclick="prev()">&larr;</button>
    <input type="range" id="slider" min="0" max="{len(records) - 1}" value="0" oninput="update(this.value)">
    <button onclick="next()">&rarr;</button>
    <button id="play" onclick="togglePlay()">Play</button>
    <label class="ck">Layer:
      <select id="layer" onchange="render()"></select>
    </label>
    <label class="ck"><input type="checkbox" id="logx" checked onchange="render()"> log-x</label>
    <span id="step-label"></span>
  </div>
  <div class="hint">Bins are shared across the four cells and fixed across steps (global, robust to outliers). Each cell is normalized to fraction-of-samples so the shapes compare despite different sample counts (shown as n=... in the legend). Samples beyond the p99.5 clip are tallied in the overflow note. Param hue: blue=retain, red=forget. Shade: dark=on retain samples, light=on hack samples.</div>
  <h2>Distribution of per-sample grad norms</h2>
  <div id="hist" style="height:440px;"></div>
  <div id="overflow" class="hint" style="margin-top:-6px;"></div>
  <h2>Per-layer aggregate (at this step)</h2>
  <div id="lines" style="height:380px;"></div>
</div>
<script>
const RECORDS = {data_json};
const COLORS_JS = {colors_json};
const NBINS = 40;
let idx = 0, playing = false, timer = null;
const _rangeCache = {{}};  // keyed by layerSel+logx

function curr() {{ return RECORDS[idx]; }}

function initLayerSelect() {{
  const sel = document.getElementById('layer');
  const layers = curr().layers;
  let opts = '<option value="all">all layers (whole-model)</option>';
  for (const li of layers) opts += `<option value="${{li}}">layer ${{li}}</option>`;
  sel.innerHTML = opts;
}}

// per-sample array for (record, role) at the selected layer ('all' -> whole-model)
function cellArray(rec, role, layerSel) {{
  if (layerSel === 'all') return rec.whole_model[role];
  const k = rec.layers.indexOf(parseInt(layerSel));
  return rec.per_sample[role][k];
}}

// pool a (role) across ALL steps for the selected layer
function poolAll(role, layerSel) {{
  let all = [];
  for (const rec of RECORDS) all = all.concat(cellArray(rec, role, layerSel));
  return all;
}}

// global robust range for the current view, cached. lo=p0.5(>0), hi=p99.5.
function globalRange(layerSel) {{
  if (_rangeCache[layerSel]) return _rangeCache[layerSel];
  let v = poolAll('retain', layerSel).concat(poolAll('forget', layerSel)).filter(x => x > 0);
  v.sort((a, b) => a - b);
  const q = (p) => v[Math.min(v.length - 1, Math.max(0, Math.floor(p * v.length)))];
  const r = {{ lo: q(0.005), hi: q(0.995), min: v[0], max: v[v.length - 1] }};
  _rangeCache[layerSel] = r;
  return r;
}}

// edges in PLOT space (log10 units if logx, else raw), plus a fn to map raw->plot
function makeBinning(layerSel, logx) {{
  const r = globalRange(layerSel);
  let edges = [], toPlot;
  if (logx) {{
    const a = Math.log10(r.lo), b = Math.log10(Math.max(r.hi, r.lo * 1.0001));
    for (let i = 0; i <= NBINS; i++) edges.push(a + (b - a) * i / NBINS);
    toPlot = (x) => Math.log10(x);
  }} else {{
    const b = r.hi;
    for (let i = 0; i <= NBINS; i++) edges.push(b * i / NBINS);
    toPlot = (x) => x;
  }}
  return {{ edges, toPlot, range: r, logx }};
}}

// histogram one array into shared edges; returns fraction-per-bin + overflow/underflow counts
function histify(arr, bin) {{
  const {{ edges, toPlot }} = bin;
  const counts = new Array(edges.length - 1).fill(0);
  let over = 0, under = 0;
  const n = arr.length;
  for (const x of arr) {{
    if (x <= 0) {{ under++; continue; }}
    const p = toPlot(x);
    if (p > edges[edges.length - 1]) {{ over++; continue; }}
    if (p < edges[0]) {{ under++; continue; }}
    let b = 0; while (b < edges.length - 1 && p > edges[b + 1]) b++;
    counts[b]++;
  }}
  const frac = counts.map(c => n ? c / n : 0);
  return {{ frac, over, under, n }};
}}

function renderHist() {{
  const rec = curr();
  const layerSel = document.getElementById('layer').value;
  const logx = document.getElementById('logx').checked;
  const bin = makeBinning(layerSel, logx);
  const edges = bin.edges;
  const centers = [], widths = [];
  for (let i = 0; i < edges.length - 1; i++) {{
    centers.push((edges[i] + edges[i + 1]) / 2);
    widths.push((edges[i + 1] - edges[i]) * 0.98);
  }}
  const cells = [
    ['retain', 0, 'rr', 'retain params \\u00b7 retain samples'],
    ['retain', 1, 'rf', 'retain params \\u00b7 hack samples'],
    ['forget', 0, 'fr', 'forget params \\u00b7 retain samples'],
    ['forget', 1, 'ff', 'forget params \\u00b7 hack samples'],
  ];
  const rh = rec.is_rh;
  const traces = [];
  const ofParts = [];
  for (const [role, cls, key, label] of cells) {{
    const full = cellArray(rec, role, layerSel);
    const arr = full.filter((_, j) => rh[j] === cls);
    const h = histify(arr, bin);
    traces.push({{
      type: 'bar', x: centers, y: h.frac, width: widths, name: `${{label}} (n=${{h.n}})`,
      marker: {{ color: COLORS_JS[key] }}, opacity: 0.5,
    }});
    if (h.over > 0 || h.under > 0)
      ofParts.push(`${{key}}: ${{h.over}}&gt;clip` + (h.under ? `, ${{h.under}}&lt;lo` : ''));
  }}
  const r = bin.range;
  const xaxis = logx
    ? {{ title: 'per-sample grad norm (log\\u2081\\u2080)', tickformat: '.1f' }}
    : {{ title: 'per-sample grad norm', range: [0, edges[edges.length - 1]] }};
  Plotly.react('hist', traces, {{
    barmode: 'overlay', bargap: 0,
    xaxis: xaxis, yaxis: {{ title: 'fraction of samples' }},
    margin: {{ t: 10, r: 10 }}, legend: {{ orientation: 'h', y: 1.16 }},
  }}, {{ displayModeBar: false }});
  const clipTxt = logx
    ? `clip range [${{r.lo.toExponential(1)}}, ${{r.hi.toExponential(1)}}], global max ${{r.max.toExponential(1)}}`
    : `clip at p99.5 = ${{r.hi.toExponential(1)}}, global max ${{r.max.toExponential(1)}}`;
  document.getElementById('overflow').innerHTML =
    `Shared bins over ${{clipTxt}}.` + (ofParts.length ? ` Out-of-range: ${{ofParts.join(' &middot; ')}}.` : ' No out-of-range samples this step.');
}}

function mean(a) {{ return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }}
function pick(arr, isrh, cls) {{ return arr.filter((_, j) => isrh[j] === cls); }}

function renderLines() {{
  const rec = curr();
  const x = rec.layers;
  const isrh = rec.is_rh;
  const cell = {{ rr: [], rf: [], fr: [], ff: [] }};
  for (let k = 0; k < x.length; k++) {{
    const r = rec.per_sample.retain[k], f = rec.per_sample.forget[k];
    cell.rr.push(mean(pick(r, isrh, 0))); cell.rf.push(mean(pick(r, isrh, 1)));
    cell.fr.push(mean(pick(f, isrh, 0))); cell.ff.push(mean(pick(f, isrh, 1)));
  }}
  const line = (key, name) => ({{ x: x, y: cell[key], name: name, mode: 'lines+markers', line: {{ color: COLORS_JS[key] }} }});
  const agg = (arr, name, color) => ({{ x: x, y: arr, name: name, mode: 'lines', line: {{ color: color, width: 3 }} }});
  const traces = [
    agg(rec.aggregate_grad_norm.retain, 'retain ||.grad|| (all)', '#1f3a93'),
    agg(rec.aggregate_grad_norm.forget, 'forget ||.grad|| (all)', '#990000'),
    line('rr', 'mean: retain\\u00b7retain'), line('rf', 'mean: retain\\u00b7hack'),
    line('fr', 'mean: forget\\u00b7retain'), line('ff', 'mean: forget\\u00b7hack'),
  ];
  Plotly.react('lines', traces, {{
    xaxis: {{ title: 'layer' }}, yaxis: {{ title: 'grad norm', type: 'log' }},
    margin: {{ t: 10, r: 10 }}, legend: {{ orientation: 'h', y: 1.18 }},
  }}, {{ displayModeBar: false }});
}}

function render() {{ renderHist(); renderLines(); }}

function update(i) {{
  idx = parseInt(i);
  document.getElementById('slider').value = idx;
  const rec = curr();
  document.getElementById('step-label').textContent =
    `step ${{rec.step}} \\u00b7 ${{rec.n_samples}} samples \\u00b7 check ${{rec.grad_check.max_triangle_ratio.toFixed(3)}}`;
  const sel = document.getElementById('layer');
  if (sel.options.length !== rec.layers.length + 1) {{ initLayerSelect(); }}
  render();
}}
function prev() {{ if (idx > 0) update(idx - 1); }}
function next() {{ if (idx < RECORDS.length - 1) update(idx + 1); }}
function togglePlay() {{
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) timer = setInterval(() => {{ if (idx < RECORDS.length - 1) next(); else {{ togglePlay(); }} }}, 700);
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
