#!/usr/bin/env python
"""Interactive viewer: does a per-sample statistic separate forget (hack)
samples from retain samples, and how does that evolve over training?

Same spirit as grad_diag.html, but split the other way. grad_diag.html overlays
retain-param vs forget-param within a fixed data type; here each panel fixes a
PARAMETER role and overlays the two SAMPLE types:

  - left panel  : retain-param norm distribution — retain samples vs forget samples
  - right panel : forget-param norm distribution — retain samples vs forget samples

If, in the forget-param panel, the forget-sample distribution sits to the right
of the retain-sample distribution, that statistic flags forget samples. The
step slider shows the separation developing over training.

Pools the sweep's seeds per env (more samples per step). Histograms are
pre-binned in Python with fixed, robust (p0.5–p99.5, log-x) bins per
(env, metric, role, layer) and a fixed y per panel, so x/y are stable across
the slider and outliers are truncated.

Usage:
  python tools/gen_separability_html.py output/<sweep_dir>/
  -> writes <sweep_dir>/separability/separability_dist.html
"""
import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
NBINS = 35
SAMPLE_COLORS = {"retain": "#1f3a93", "forget": "#c0392b"}  # retain samples blue, forget red
METRICS = (("grad", "per_sample", "whole_model"), ("act", "act_per_sample", "act_whole_model"))
ROLES = ("retain", "forget")


def env_of(run_name):
    for marker in ("_gr_cls", "_gr_excl", "_rp", "_graddiag"):
        if marker in run_name:
            return run_name.split(marker)[0]
    return run_name


def _edges(pool):
    v = np.sort(pool[pool > 0])
    if v.size < 4:
        return None
    lo = v[int(0.005 * v.size)]
    hi = v[min(v.size - 1, int(0.995 * v.size))]
    if hi <= lo:
        hi = lo * 1.0001
    return np.linspace(np.log10(lo), np.log10(hi), NBINS + 1)


def build_env_data(recs_by_seed):
    """Pool seeds per step; return the embeddable per-panel histogram structure."""
    bystep = defaultdict(list)
    for recs in recs_by_seed:
        for r in recs:
            bystep[r["step"]].append(r)
    steps = sorted(bystep)
    layers = recs_by_seed[0][0]["layers"]
    layerkeys = ["whole"] + [str(li) for li in layers]

    # gather raw per (panel-key) -> per-step {retain:[...], forget:[...]}
    # panel-key = metric|role|layerkey
    raw = defaultdict(lambda: {s: {"retain": [], "forget": []} for s in steps})
    nhack, nretain = [], []
    for s in steps:
        recs = bystep[s]
        nh = sum(sum(r["is_rh"]) for r in recs)
        nhack.append(nh)
        nretain.append(sum(len(r["is_rh"]) - sum(r["is_rh"]) for r in recs))
        for r in recs:
            rh = r["is_rh"]
            for metric, ps_key, wm_key in METRICS:
                for role in ROLES:
                    wm = r[wm_key][role]
                    bucket = raw[f"{metric}|{role}|whole"][s]
                    for j, lab in enumerate(rh):
                        bucket["forget" if lab else "retain"].append(wm[j])
                    for k, li in enumerate(layers):
                        arr = r[ps_key][role][k]
                        bucket = raw[f"{metric}|{role}|{li}"][s]
                        for j, lab in enumerate(rh):
                            bucket["forget" if lab else "retain"].append(arr[j])

    panels = {}
    for key, perstep in raw.items():
        allvals = np.array([v for s in steps for st in ("retain", "forget")
                            for v in perstep[s][st]], dtype=float)
        edges = _edges(allvals) if allvals.size else None
        if edges is None:
            continue
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        width = float((edges[1] - edges[0]) * 0.98)
        hist = []
        ymax = 1
        for s in steps:
            row = {}
            for st in ("retain", "forget"):
                x = np.array(perstep[s][st], dtype=float)
                lx = np.log10(x[x > 0]) if x.size else np.array([])
                counts, _ = np.histogram(lx, bins=edges)
                row[st[0]] = counts.tolist()
                if counts.size:
                    ymax = max(ymax, int(counts.max()))
            hist.append(row)
        panels[key] = {"c": centers, "w": width, "ymax": ymax, "h": hist}

    return {"steps": steps, "layers": layers, "layerkeys": layerkeys,
            "nhack": nhack, "nretain": nretain, "panels": panels}


def build_html(data_by_env):
    data_json = json.dumps(data_by_env, separators=(",", ":"))
    colors_json = json.dumps(SAMPLE_COLORS)
    envs = sorted(data_by_env)
    env_opts = "".join(f'<option value="{e}">{e}</option>' for e in envs)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Forget-vs-retain separability</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  body {{ font-family: sans-serif; background:#f5f5f5; margin:18px; }}
  .container {{ max-width:1250px; margin:0 auto; }}
  .controls {{ display:flex; align-items:center; gap:14px; margin:12px 0; flex-wrap:wrap; }}
  input[type=range] {{ width:340px; }}
  select,button {{ padding:5px 10px; font-size:14px; }}
  #lbl {{ font-weight:bold; min-width:260px; }}
  .hint {{ color:#888; font-size:12px; }}
</style></head><body>
<div class="container">
  <h1 style="font-size:20px;">Can a statistic tell a forget sample from a retain sample?</h1>
  <div class="controls">
    <label>Env: <select id="env" onchange="onEnv()">{env_opts}</select></label>
    <label>Metric: <select id="metric" onchange="render()">
      <option value="grad">gradient norm</option>
      <option value="act">activation norm</option></select></label>
    <label>Layer: <select id="layer" onchange="render()"></select></label>
    <button onclick="prev()">&larr;</button>
    <input type="range" id="slider" min="0" value="0" oninput="render()">
    <button onclick="next()">&rarr;</button>
    <button id="play" onclick="togglePlay()">Play</button>
    <span id="lbl"></span>
  </div>
  <div class="hint">Each panel fixes a parameter adapter and overlays the per-sample norm distribution
   over <b style="color:#1f3a93">retain samples</b> vs <b style="color:#c0392b">forget samples</b>.
   Separation in the forget-param panel = that statistic flags forget samples. Bins fixed per
   (env,metric,role,layer); log-x; y fixed across steps; outliers (beyond p0.5–p99.5) truncated.</div>
  <div style="display:flex; gap:12px;">
    <div style="flex:1;"><div id="p_retain" style="height:360px;"></div></div>
    <div style="flex:1;"><div id="p_forget" style="height:360px;"></div></div>
  </div>
</div>
<script>
const DATA = {data_json};
const SC = {colors_json};
let idx = 0, playing = false, timer = null;

function env() {{ return document.getElementById('env').value; }}
function metric() {{ return document.getElementById('metric').value; }}
function layerKey() {{ return document.getElementById('layer').value; }}
function curEnv() {{ return DATA[env()]; }}

function initLayer() {{
  const d = curEnv();
  let o = '<option value="whole">all layers (whole-model)</option>';
  for (const li of d.layers) o += `<option value="${{li}}">layer ${{li}}</option>`;
  document.getElementById('layer').innerHTML = o;
}}

function onEnv() {{
  initLayer();
  const d = curEnv();
  document.getElementById('slider').max = d.steps.length - 1;
  if (idx > d.steps.length - 1) idx = d.steps.length - 1;
  render();
}}

function panel(divId, role, title) {{
  const d = curEnv();
  const key = metric() + '|' + role + '|' + layerKey();
  const p = d.panels[key];
  const xlabel = metric() === 'act' ? 'activation norm (log\\u2081\\u2080)' : 'grad norm (log\\u2081\\u2080)';
  if (!p) {{ Plotly.react(divId, [], {{title:{{text:title+' (no data)',font:{{size:13}}}}}}); return; }}
  const row = p.h[idx];
  const mk = (st, name) => ({{ type:'bar', x:p.c, y:row[st], width:p.w, name:name,
                              marker:{{color: SC[st==='r'?'retain':'forget']}}, opacity:0.55 }});
  const nR = row.r.reduce((a,b)=>a+b,0), nF = row.f.reduce((a,b)=>a+b,0);
  Plotly.react(divId, [mk('r', `retain samples (n=${{nR}})`), mk('f', `forget samples (n=${{nF}})`)], {{
    barmode:'overlay', bargap:0, title:{{text:title, font:{{size:13}}}},
    xaxis:{{title:xlabel, tickformat:'.1f'}},
    yaxis:{{title:'number of samples', range:[0, p.ymax*1.05]}},
    margin:{{t:32,r:8}}, legend:{{orientation:'h', y:1.18}},
  }}, {{displayModeBar:false}});
}}

function render() {{
  const d = curEnv();
  idx = parseInt(document.getElementById('slider').value);
  document.getElementById('lbl').textContent =
    `step ${{d.steps[idx]}} \\u00b7 ${{d.nhack[idx]}} forget / ${{d.nretain[idx]}} retain samples`;
  panel('p_retain', 'retain', 'Retain-param norm');
  panel('p_forget', 'forget', 'Forget-param norm');
}}
function setSlider(i) {{ document.getElementById('slider').value = i; render(); }}
function prev() {{ if (idx>0) setSlider(idx-1); }}
function next() {{ const n=curEnv().steps.length; if (idx<n-1) setSlider(idx+1); }}
function togglePlay() {{
  playing=!playing; document.getElementById('play').textContent=playing?'Pause':'Play';
  if (playing) timer=setInterval(()=>{{ const n=curEnv().steps.length; if (idx<n-1) next(); else togglePlay(); }}, 600);
  else clearInterval(timer);
}}
document.addEventListener('keydown',(e)=>{{ if(e.key==='ArrowLeft')prev(); else if(e.key==='ArrowRight')next(); }});

initLayer(); onEnv();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*", "grad_diag.jsonl")))
    assert files, f"no */grad_diag.jsonl under {args.sweep_dir}"
    by_env = defaultdict(list)
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            by_env[env_of(os.path.basename(os.path.dirname(f)))].append(recs)

    data_by_env = {}
    for env, recs_by_seed in by_env.items():
        d = build_env_data(recs_by_seed)
        if d["panels"]:
            data_by_env[env] = d
    assert data_by_env, "no usable data"

    outdir = os.path.join(args.sweep_dir, "separability")
    os.makedirs(outdir, exist_ok=True)
    out = args.output or os.path.join(outdir, "separability_dist.html")
    with open(out, "w") as f:
        f.write(build_html(data_by_env))
    mb = os.path.getsize(out) / 1e6
    print(f"[separability] wrote {out} ({len(data_by_env)} envs, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
