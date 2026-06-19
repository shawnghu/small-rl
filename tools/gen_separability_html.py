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
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .row {{ display:flex; gap:14px; align-items:flex-start; margin-top:8px; }}
  .plot {{ height:258px; }}
  .ptitle {{ font-weight:bold; font-size:13px; text-align:center; margin-bottom:1px; }}
</style></head><body>
<div class="container">
  <h1 style="font-size:20px;">Can a statistic tell a forget sample from a retain sample?</h1>
  <div class="controls">
    <label>Env: <select id="env" onchange="onEnv()">{env_opts}</select></label>
    <label>Layer: <select id="layer" onchange="render()"></select></label>
    <button onclick="prev()">&larr;</button>
    <input type="range" id="slider" min="0" value="0" oninput="render()">
    <button onclick="next()">&rarr;</button>
    <button id="play" onclick="togglePlay()">Play</button>
    <span id="lbl"></span>
  </div>
  <div class="hint">Each panel fixes a metric (row) and parameter adapter (column) and overlays the per-sample
   norm distribution over <b style="color:#1f3a93">retain samples</b> vs <b style="color:#c0392b">forget samples</b>.
   Separation in a forget-param panel = that statistic flags forget samples. Bins fixed per
   (env,metric,role,layer); log-x; y fixed across steps; outliers (beyond p0.5–p99.5) truncated.</div>
  <div class="row">
    <div class="grid" style="flex:0 0 680px;">
      <div><div class="ptitle">Gradient norm &middot; Retain-param</div><div id="g_retain" class="plot"></div></div>
      <div><div class="ptitle">Gradient norm &middot; Forget-param</div><div id="g_forget" class="plot"></div></div>
      <div><div class="ptitle">Activation norm &middot; Retain-param</div><div id="a_retain" class="plot"></div></div>
      <div><div class="ptitle">Activation norm &middot; Forget-param</div><div id="a_forget" class="plot"></div></div>
    </div>
    <div style="flex:1; min-width:360px;">
      <div class="ptitle">Retain reward &middot; <span style="color:#e67e22">2-adapter</span> vs <span style="color:#27ae60">retain-only</span></div>
      <div id="curves_reward" style="height:258px;"></div>
      <div class="ptitle">Hack rate &middot; solid=monitored, dotted=unmonitored (vline=current step)</div>
      <div id="curves_hack" style="height:258px;"></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const SC = {colors_json};
let idx = 0, playing = false, timer = null;

function env() {{ return document.getElementById('env').value; }}
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
  renderCurves();
  render();
}}

const C_BOTH = '#e67e22', C_RONLY = '#27ae60';  // orange=2-adapter, green=retain-only
function renderCurves() {{
  const c = curEnv().curves;
  const ln = (k, name, col, dash) => ({{ x:c.steps, y:c[k], name:name, mode:'lines',
                                         line:{{color:col, dash:dash||'solid'}}, connectgaps:true }});
  const vline = {{type:'line', x0:0, x1:0, yref:'paper', y0:0, y1:1, line:{{color:'#444', dash:'dot', width:1.5}}}};
  const base = {{ xaxis:{{title:'training step'}}, margin:{{t:6, r:8, b:34, l:40}},
                 legend:{{orientation:'h', y:1.0, yanchor:'bottom', font:{{size:9}}}}, shapes:[vline] }};
  Plotly.react('curves_reward',
    [ln('retain_both', '2-adapter', C_BOTH), ln('retain_ronly', 'retain-only', C_RONLY)],
    Object.assign({{yaxis:{{title:'retain reward'}}}}, base), {{displayModeBar:false}});
  Plotly.react('curves_hack', [
    ln('hack_both_mon', '2-adapter monitored', C_BOTH, 'solid'),
    ln('hack_both_unmon', '2-adapter unmonitored', C_BOTH, 'dot'),
    ln('hack_ronly_mon', 'retain-only monitored', C_RONLY, 'solid'),
    ln('hack_ronly_unmon', 'retain-only unmonitored', C_RONLY, 'dot'),
  ], Object.assign({{yaxis:{{title:'hack rate'}}}}, base), {{displayModeBar:false}});
}}
function updateVline() {{
  const step = curEnv().steps[idx];
  Plotly.relayout('curves_reward', {{'shapes[0].x0': step, 'shapes[0].x1': step}});
  Plotly.relayout('curves_hack', {{'shapes[0].x0': step, 'shapes[0].x1': step}});
}}

// no Plotly title (the HTML .ptitle above each plot serves as the title, so it
// can never overlap the legend); legend sits in the top margin.
function panel(divId, metric, role) {{
  const d = curEnv();
  const key = metric + '|' + role + '|' + layerKey();
  const p = d.panels[key];
  const xlabel = metric === 'act' ? 'activation norm (log\\u2081\\u2080)' : 'grad norm (log\\u2081\\u2080)';
  if (!p) {{ Plotly.react(divId, [], {{margin:{{t:24,r:8}},
      annotations:[{{text:'no data', showarrow:false, xref:'paper', yref:'paper', x:0.5, y:0.5}}]}},
      {{displayModeBar:false}}); return; }}
  const row = p.h[idx];
  const mk = (st, name) => ({{ type:'bar', x:p.c, y:row[st], width:p.w, name:name,
                              marker:{{color: SC[st==='r'?'retain':'forget']}}, opacity:0.55 }});
  const nR = row.r.reduce((a,b)=>a+b,0), nF = row.f.reduce((a,b)=>a+b,0);
  Plotly.react(divId, [mk('r', `retain samples (n=${{nR}})`), mk('f', `forget samples (n=${{nF}})`)], {{
    barmode:'overlay', bargap:0,
    xaxis:{{title:xlabel, tickformat:'.1f'}},
    yaxis:{{title:'# samples', range:[0, p.ymax*1.05]}},
    margin:{{t:26, r:8, b:40, l:44}}, legend:{{orientation:'h', y:1.0, yanchor:'bottom', font:{{size:10}}}},
  }}, {{displayModeBar:false}});
}}

function render() {{
  const d = curEnv();
  idx = parseInt(document.getElementById('slider').value);
  document.getElementById('lbl').textContent =
    `step ${{d.steps[idx]}} \\u00b7 ${{d.nhack[idx]}} forget / ${{d.nretain[idx]}} retain samples`;
  panel('g_retain', 'grad', 'retain');
  panel('g_forget', 'grad', 'forget');
  panel('a_retain', 'act', 'retain');
  panel('a_forget', 'act', 'forget');
  updateVline();
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


# curve name -> (eval mode = adapter config, routing_eval 2nd path segment)
# orange = both-adapter, green = retain-only; monitored=detectable, unmonitored=undetectable.
CURVE_SPECS = {
    "retain_both":      ("both", "retain"),
    "retain_ronly":     ("retain_only", "retain"),
    "hack_both_mon":    ("both", "hack_freq_detectable"),
    "hack_both_unmon":  ("both", "hack_freq_undetectable"),
    "hack_ronly_mon":   ("retain_only", "hack_freq_detectable"),
    "hack_ronly_unmon": ("retain_only", "hack_freq_undetectable"),
}


def load_curves(run_dirs):
    """Seed-averaged routing_eval curves (overview.html style): two-adapter vs
    retain-adapter-only retain reward, and monitored/unmonitored hack rate for
    each. Returns {steps, <CURVE_SPECS keys>}."""
    want = {(m, s): name for name, (m, s) in CURVE_SPECS.items()}
    per = {name: defaultdict(list) for name in CURVE_SPECS}
    for d in run_dirs:
        p = os.path.join(d, "routing_eval.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            step = r.get("step")
            if step is None:
                continue
            for key, val in r.items():
                parts = key.split("/")
                if len(parts) >= 2 and (parts[0], parts[1]) in want \
                        and isinstance(val, (int, float)):
                    per[want[(parts[0], parts[1])]][step].append(val)
    steps = sorted(set().union(*[set(per[k]) for k in per]) or {0})
    out = {"steps": steps}
    for name in CURVE_SPECS:
        out[name] = [float(np.mean(per[name][s])) if per[name].get(s) else None for s in steps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.sweep_dir, "*", "grad_diag.jsonl")))
    assert files, f"no */grad_diag.jsonl under {args.sweep_dir}"
    by_env = defaultdict(list)
    by_env_dirs = defaultdict(list)
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            e = env_of(os.path.basename(os.path.dirname(f)))
            by_env[e].append(recs)
            by_env_dirs[e].append(os.path.dirname(os.path.realpath(f)))

    data_by_env = {}
    for env, recs_by_seed in by_env.items():
        d = build_env_data(recs_by_seed)
        if d["panels"]:
            d["curves"] = load_curves(by_env_dirs[env])
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
