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
# Sample-class taxonomy. Ground-truth scheme (when records carry hackable/hacked):
# hue = emission (grey=unhackable, blue=hackable-not-hacked, red=hacked); shade =
# monitoring (light=detectable/monitored, dark=undetectable/unmonitored). The
# is_rh routing label is NOT used here (it's the imperfect router, recorded
# separately). Each entry: (key, label, color).
GT_CLASSES = [
    ("unhk",     "unhackable",                          "#9aa0a6"),
    ("nh_mon",   "hackable·not-hacked·mon",   "#6fa8dc"),
    ("nh_unmon", "hackable·not-hacked·unmon", "#1f3a93"),
    ("h_mon",    "hacked·monitored",               "#f1948a"),
    ("h_unmon",  "hacked·unmonitored",             "#922b21"),
]
# Legacy scheme for records that predate hackable/hacked (is_rh based).
LEGACY_CLASSES = [
    ("r",  "retain",                  "#1f3a93"),
    ("fd", "forget·detectable",  "#f1948a"),
    ("fu", "forget·undetectable", "#922b21"),
]


def _classify_gt(r, j):
    if not r["hackable"][j]:
        return "unhk"
    if r["hacked"][j]:
        return "h_mon" if r["detectable"][j] else "h_unmon"
    return "nh_mon" if r["detectable"][j] else "nh_unmon"


def _classify_legacy(r, j):
    if not r["is_rh"][j]:
        return "r"
    det = r.get("detectable") or [0] * len(r["is_rh"])
    return "fd" if det[j] else "fu"


def _scheme(rec0):
    """(classes, classify_fn) — ground-truth taxonomy when hackable/hacked are
    present, else the legacy is_rh taxonomy."""
    if "hacked" in rec0 and "hackable" in rec0:
        return GT_CLASSES, _classify_gt
    return LEGACY_CLASSES, _classify_legacy
# metric -> (per-sample key, whole-model key). grad/act are magnitudes (log-x
# histograms); dot is signed <grad,weight> (linear-x). dot is skipped on records
# that predate it.
METRICS = (("grad", "per_sample", "whole_model"),
           ("act", "act_per_sample", "act_whole_model"),
           ("dot", "dot_per_sample", "dot_whole_model"))
LOG_METRICS = {"grad", "act"}            # log-x; everything else linear-x
SCATTER_METRICS = ("grad", "act")        # joint-distribution scatter: magnitudes only
ROLES = ("retain", "forget")


def env_of(run_name):
    for marker in ("_gr_cls", "_gr_excl", "_rp", "_nogr", "_no_intervention", "_nohack", "_graddiag"):
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


def _symlog_setup(pool):
    """Symlog binning for a SIGNED, zero-peaked, heavy-tailed quantity (the dot
    product). Linear within [-linthresh, linthresh] (linthresh = median |nonzero|),
    log-spaced beyond — so the central peak spreads instead of dumping into one
    bin. Returns dict with linear-in-symlog-space edges, the transform, and a few
    (symlog-position, actual-value-label) ticks for the x-axis."""
    a = np.asarray(pool, float)
    nz = np.abs(a[a != 0])
    if nz.size < 4:
        return None
    linthresh = float(max(np.median(nz), 1e-30))

    def T(v):
        v = np.asarray(v, float)
        sg = np.sign(v)
        av = np.abs(v)
        with np.errstate(divide="ignore"):  # log10(0) on the linear-branch entries; discarded by where
            return sg * np.where(av <= linthresh, av / linthresh, 1.0 + np.log10(av / linthresh))

    s = np.sort(a)
    lo = float(s[int(0.005 * s.size)])
    hi = float(s[min(s.size - 1, int(0.995 * s.size))])
    tlo, thi = float(T(np.array([lo]))[0]), float(T(np.array([hi]))[0])
    if thi <= tlo:
        thi = tlo + 1.0
    edges = np.linspace(tlo, thi, NBINS + 1)
    tick_actual = [lo, -linthresh, 0.0, linthresh, hi]
    tpos = T(np.array(tick_actual)).tolist()
    ticks = {"v": tpos, "t": [("0" if x == 0 else f"{x:.0e}") for x in tick_actual]}
    return {"edges": edges, "T": T, "ticks": ticks}


def _log_range(vals):
    a = np.sort(vals[vals > 0])
    if a.size < 4:
        return None
    return [float(np.log10(a[int(0.005 * a.size)])),
            float(np.log10(a[min(a.size - 1, int(0.995 * a.size))]))]


def build_env_data(recs_by_seed, class_keys, classify):
    """Pool seeds per step; return embeddable per-panel histograms (one series per
    sample class) + per-step joint-scatter data (whole-model retain vs forget
    magnitude per sample, colored by class). `class_keys` is the ordered class
    list; `classify(r, j)` returns the class key for sample j of record r."""
    bystep = defaultdict(list)
    for recs in recs_by_seed:
        for r in recs:
            bystep[r["step"]].append(r)
    steps = sorted(bystep)
    layers = recs_by_seed[0][0]["layers"]
    layerkeys = ["whole"] + [str(li) for li in layers]
    kidx = {k: i for i, k in enumerate(class_keys)}

    raw = defaultdict(lambda: {s: {k: [] for k in class_keys} for s in steps})
    scat = {m: {s: {"x": [], "y": [], "cls": []} for s in steps} for m in SCATTER_METRICS}
    class_n = {k: [] for k in class_keys}  # true per-step class sizes (panel-independent)
    for s in steps:
        recs = bystep[s]
        cn = {k: 0 for k in class_keys}
        for r in recs:
            n = len(r["is_rh"])
            keys = [classify(r, j) for j in range(n)]
            for k in keys:
                cn[k] += 1
            for metric, ps_key, wm_key in METRICS:
                wmd, psd = r.get(wm_key), r.get(ps_key)
                if wmd is None or psd is None:
                    continue  # metric absent (e.g. dot on pre-dot records)
                if metric in SCATTER_METRICS:
                    wmR, wmF = wmd["retain"], wmd["forget"]
                    sc = scat[metric][s]
                    for j in range(n):
                        sc["x"].append(wmR[j]); sc["y"].append(wmF[j]); sc["cls"].append(kidx[keys[j]])
                for role in ROLES:
                    wm = wmd[role]
                    b = raw[f"{metric}|{role}|whole"][s]
                    for j in range(n):
                        b[keys[j]].append(wm[j])
                    for k, li in enumerate(layers):
                        arr = psd[role][k]
                        b = raw[f"{metric}|{role}|{li}"][s]
                        for j in range(n):
                            b[keys[j]].append(arr[j])
        for k in class_keys:
            class_n[k].append(cn[k])

    panels = {}
    for key, perstep in raw.items():
        metric = key.split("|")[0]
        logx = metric in LOG_METRICS
        allvals = np.array([v for s in steps for k in class_keys
                            for v in perstep[s][k]], dtype=float)
        if not allvals.size:
            continue
        symlog = None
        if logx:
            edges = _edges(allvals)
        else:
            symlog = _symlog_setup(allvals)
            edges = symlog["edges"] if symlog else None
        if edges is None:
            continue
        T = symlog["T"] if symlog else None
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        width = float((edges[1] - edges[0]) * 0.98)
        hist = []
        step_peaks = []  # (tallest single-class bin, is-degenerate) per step
        for s in steps:
            row = {}
            speak, deg = 0, False
            for k in class_keys:
                x = np.array(perstep[s][k], dtype=float)
                if logx:
                    vals = np.log10(x[x > 0]) if x.size else np.array([])
                else:
                    vals = T(x) if x.size else np.array([])
                counts, _ = np.histogram(vals, bins=edges)
                row[k] = counts.tolist()
                cmax, ctot = (int(counts.max()), int(counts.sum())) if counts.size else (0, 0)
                speak = max(speak, cmax)
                if ctot > 0 and cmax > 0.6 * ctot:  # one bin swallows the class
                    deg = True
            hist.append(row)
            step_peaks.append((speak, deg))
        # y-axis fixed across steps. For symlog (the dot), exclude degenerate
        # steps — e.g. step 0 where every value is exactly 0 and piles into one
        # bin — so the scale reflects the real spread, not the zero-spike.
        if symlog:
            kept = [pk for pk, dg in step_peaks if not dg]
            ymax = max(kept) if kept else max((pk for pk, _ in step_peaks), default=1)
        else:
            ymax = max((pk for pk, _ in step_peaks), default=1)
        panels[key] = {"c": centers, "w": width, "ymax": max(ymax, 1), "h": hist, "logx": logx}
        if symlog:
            panels[key]["ticks"] = symlog["ticks"]

    scatter = {}
    for metric in SCATTER_METRICS:
        allx = np.array([v for s in steps for v in scat[metric][s]["x"]], float)
        ally = np.array([v for s in steps for v in scat[metric][s]["y"]], float)
        scatter[metric] = {
            "x": [scat[metric][s]["x"] for s in steps],
            "y": [scat[metric][s]["y"] for s in steps],
            "cls": [scat[metric][s]["cls"] for s in steps],
            "rx": _log_range(allx), "ry": _log_range(ally),
        }

    return {"steps": steps, "layers": layers, "layerkeys": layerkeys,
            "class_n": class_n, "panels": panels, "scatter": scatter}


# Core JS reused by both the detailed (per-env, multi-condition) page and the
# all-envs overview page. Raw string (single braces, \\u escapes preserved for
# JS); inserted verbatim into each page's f-string via {_SHARED_JS}. Depends on
# globals `SC` (class colors) and `layerKey()` being defined by the host page.
_SHARED_JS = r"""
function el(tag, cls, html) { const e=document.createElement(tag); if(cls)e.className=cls; if(html!=null)e.innerHTML=html; return e; }

function pearsonLog(xs, ys) {
  const lx=[], ly=[];
  for (let i=0;i<xs.length;i++) if (xs[i]>0 && ys[i]>0) { lx.push(Math.log(xs[i])); ly.push(Math.log(ys[i])); }
  const n=lx.length; if (n<3) return NaN;
  const mx=lx.reduce((a,b)=>a+b,0)/n, my=ly.reduce((a,b)=>a+b,0)/n;
  let sxy=0,sxx=0,syy=0;
  for (let i=0;i<n;i++){ const dx=lx[i]-mx, dy=ly[i]-my; sxy+=dx*dy; sxx+=dx*dx; syy+=dy*dy; }
  return (sxx>0&&syy>0) ? sxy/Math.sqrt(sxx*syy) : NaN;
}

// One histogram panel for (condition-data cd, metric, role) at step-index j.
function panel(divId, cd, metric, role, j) {
  const p = cd.panels[metric + '|' + role + '|' + layerKey()];
  const xlabel = metric === 'dot' ? '⟨grad,weight⟩ (symlog)'
    : metric === 'act' ? 'activation norm (log₁₀)' : 'grad norm (log₁₀)';
  if (!p || j < 0) { Plotly.react(divId, [], {margin:{t:24,r:8},
      annotations:[{text:'no data at this step', showarrow:false, xref:'paper', yref:'paper', x:0.5, y:0.5}]},
      {displayModeBar:false}); return; }
  const row = p.h[j];
  // legend n = TRUE class size at this step (panel/layer-independent), not the
  // count of in-range bars (some samples truncated out of the fixed range).
  // One overlaid bar series per sample class (CLS, global class metadata).
  const cn = cd.class_n;
  const traces = CLS.filter(c => row[c.k]).map(c => ({
    type:'bar', x:p.c, y:row[c.k], width:p.w,
    name:`${c.label} (n=${cn && cn[c.k] ? cn[c.k][j] : row[c.k].reduce((a,b)=>a+b,0)})`,
    marker:{color: c.color}, opacity:0.5,
  }));
  const xaxis = p.ticks
    ? {title:xlabel, tickmode:'array', tickvals:p.ticks.v, ticktext:p.ticks.t}
    : {title:xlabel, tickformat: p.logx ? '.1f' : '.2g'};
  Plotly.react(divId, traces, {
    barmode:'overlay', bargap:0,
    xaxis: xaxis,
    yaxis:{title:'# samples', range:[0, p.ymax*1.05]},
    margin:{t:26, r:8, b:40, l:44}, legend:{orientation:'h', y:1.0, yanchor:'bottom', font:{size:8}},
  }, {displayModeBar:false});
}
"""


def build_html(data_by_env, classes_meta):
    data_json = json.dumps(data_by_env, separators=(",", ":"))
    cls_json = json.dumps([{"k": k, "label": lbl, "color": col} for k, lbl, col in classes_meta])
    envs = sorted(data_by_env)
    env_opts = "".join(f'<option value="{e}">{e}</option>' for e in envs)
    has_dot = any(k.startswith("dot|")
                  for e in data_by_env.values()
                  for cd in e["C"].values() for k in cd.get("panels", {}))
    has_dot_js = "true" if has_dot else "false"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Forget-vs-retain separability</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  body {{ font-family: sans-serif; background:#f5f5f5; margin:18px; }}
  .container {{ max-width:1250px; margin:0 auto; }}
  .controls {{ display:flex; align-items:center; gap:14px; margin:12px 0; flex-wrap:wrap; position:sticky; top:0; background:#f5f5f5; z-index:10; padding:6px 0; }}
  input[type=range] {{ width:340px; }}
  select,button {{ padding:5px 10px; font-size:14px; }}
  #lbl {{ font-weight:bold; min-width:160px; }}
  .hint {{ color:#888; font-size:12px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  .row {{ display:flex; gap:14px; align-items:flex-start; margin-top:8px; }}
  .plot {{ height:258px; }}
  .ptitle {{ font-weight:bold; font-size:13px; text-align:center; margin-bottom:1px; }}
  .cond-block {{ border:1px solid #ddd; border-radius:6px; padding:8px; margin-top:14px; background:#fff; }}
  .cond-head {{ font-size:15px; margin-bottom:4px; }}
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
  <div class="hint">One block per condition (GR / RP / do-nothing) present for the env. Each grid panel overlays the
   per-sample value over <b style="color:#1f3a93">retain samples</b>, <b style="color:#f1948a">forget&middot;detectable</b>,
   <b style="color:#922b21">forget&middot;undetectable</b>. Grids show only where grad_diag exists; curves where
   routing_eval exists. log-x for norms; signed linear-x for the dot; y fixed across steps; outliers truncated.</div>
  <div id="blocks"></div>
</div>
<script>
const DATA = {data_json};
const CLS = {cls_json};
const HAS_DOT = {has_dot_js};
const C_BOTH = '#e67e22', C_RONLY = '#27ae60';  // orange=2-adapter, green=retain-only
const GRID_CELLS = [
  ['grad','retain','Gradient norm \\u00b7 Retain-param'], ['grad','forget','Gradient norm \\u00b7 Forget-param'],
  ['act','retain','Activation norm \\u00b7 Retain-param'], ['act','forget','Activation norm \\u00b7 Forget-param'],
  ['dot','retain','Dot \\u27e8g,w\\u27e9 \\u00b7 Retain-param'], ['dot','forget','Dot \\u27e8g,w\\u27e9 \\u00b7 Forget-param'],
];
let idx = 0, playing = false, timer = null;

function env() {{ return document.getElementById('env').value; }}
function layerKey() {{ return document.getElementById('layer').value; }}
function curEnv() {{ return DATA[env()]; }}
{_SHARED_JS}
function initLayer() {{
  const d = curEnv();
  let o = '<option value="whole">all layers (whole-model)</option>';
  for (const li of d.layers) o += `<option value="${{li}}">layer ${{li}}</option>`;
  document.getElementById('layer').innerHTML = o;
}}

function onEnv() {{
  initLayer();
  const E = curEnv();
  const sl = document.getElementById('slider');
  sl.max = Math.max(0, E.sliderSteps.length - 1);
  if (idx > E.sliderSteps.length - 1) idx = E.sliderSteps.length - 1;
  if (idx < 0) idx = 0;
  sl.value = idx;
  buildBlocks();
  render();
}}

function hasCurves(cd) {{ return cd.curves && cd.curves.steps && cd.curves.steps.length > 0
  && cd.curves.retain_both && cd.curves.retain_both.some(v => v != null); }}

function buildBlocks() {{
  const E = curEnv();
  const host = document.getElementById('blocks');
  host.innerHTML = '';
  E.conds.forEach((cond, ci) => {{
    const cd = E.C[cond];
    const block = el('div', 'cond-block');
    block.appendChild(el('div', 'cond-head', `Condition: <b>${{cond}}</b>`));
    const row = el('div', 'row');
    if (cd.has_grad) {{
      const grid = el('div', 'grid'); grid.style.flex = '0 0 680px';
      for (const [m, role, lbl] of GRID_CELLS) {{
        if (m === 'dot' && !(('dot|' + role + '|whole') in cd.panels)) continue;
        const cell = el('div'); cell.appendChild(el('div', 'ptitle', lbl));
        const pd = el('div', 'plot'); pd.id = `p_${{ci}}_${{m}}_${{role}}`; cell.appendChild(pd);
        grid.appendChild(cell);
      }}
      row.appendChild(grid);
    }}
    if (hasCurves(cd)) {{
      const cv = el('div'); cv.style.flex = '1'; cv.style.minWidth = '340px';
      cv.appendChild(el('div', 'ptitle', 'Retain reward \\u00b7 <span style="color:#e67e22">2-adapter</span> vs <span style="color:#27ae60">retain-only</span>'));
      const cr = el('div'); cr.id = `cr_${{ci}}`; cr.style.height = '210px'; cv.appendChild(cr);
      cv.appendChild(el('div', 'ptitle', 'Hack rate \\u00b7 solid=monitored, dotted=unmonitored'));
      const ch = el('div'); ch.id = `ch_${{ci}}`; ch.style.height = '210px'; cv.appendChild(ch);
      row.appendChild(cv);
    }}
    block.appendChild(row);
    if (cd.has_grad) {{
      block.appendChild(el('div', 'ptitle', 'Joint: retain-param vs forget-param magnitude (per sample, whole-model, this step)'));
      const srow = el('div', 'row');
      for (const m of ['grad', 'act']) {{
        const c = el('div'); c.style.flex = '1';
        c.appendChild(el('div', 'ptitle', m === 'grad' ? 'Gradient norm' : 'Activation norm'));
        const sdv = el('div'); sdv.id = `s_${{ci}}_${{m}}`; sdv.style.height = '300px'; c.appendChild(sdv);
        srow.appendChild(c);
      }}
      block.appendChild(srow);
    }}
    host.appendChild(block);
    if (hasCurves(cd)) renderCurvesCond(ci, cd);
  }});
}}

function renderScatter(ci, cd, j) {{
  for (const [suffix, lbl] of [['grad','grad norm'],['act','activation norm']]) {{
    const divId = `s_${{ci}}_${{suffix}}`;
    const sc = cd.scatter[suffix];
    if (!sc || j < 0) {{ Plotly.react(divId, [], {{margin:{{t:24,r:8}}}}, {{displayModeBar:false}}); continue; }}
    const xs=sc.x[j], ys=sc.y[j], cls=sc.cls[j];
    const grp = CLS.map(()=>({{x:[],y:[]}}));
    for (let i=0;i<xs.length;i++) if (xs[i]>0 && ys[i]>0) {{ const g=grp[cls[i]]; if(g){{ g.x.push(xs[i]); g.y.push(ys[i]); }} }}
    const traces = CLS.map((c,ci2) => ({{ x:grp[ci2].x, y:grp[ci2].y, mode:'markers', type:'scattergl',
                       name:`${{c.label}} (n=${{grp[ci2].x.length}})`, marker:{{color:c.color, size:4, opacity:0.45}} }}));
    const r = pearsonLog(xs, ys);
    Plotly.react(divId, traces, {{
      xaxis:{{title:`retain-param ${{lbl}}`, type:'log', range:sc.rx}},
      yaxis:{{title:`forget-param ${{lbl}}`, type:'log', range:sc.ry}},
      margin:{{t:24, r:8, b:42, l:52}}, legend:{{orientation:'h', y:1.0, yanchor:'bottom', font:{{size:9}}}},
      annotations:[{{text:`r(log)=${{isNaN(r)?'\\u2013':r.toFixed(2)}}`, showarrow:false,
                    xref:'paper', yref:'paper', x:0.03, y:0.97, bgcolor:'rgba(255,255,255,0.7)', font:{{size:11}}}}],
    }}, {{displayModeBar:false}});
  }}
}}

function renderCurvesCond(ci, cd) {{
  const c = cd.curves;
  const ln = (k, name, col, dash) => ({{ x:c.steps, y:c[k], name:name, mode:'lines',
                                         line:{{color:col, dash:dash||'solid'}}, connectgaps:true }});
  const vline = {{type:'line', x0:0, x1:0, yref:'paper', y0:0, y1:1, line:{{color:'#444', dash:'dot', width:1.5}}}};
  const base = {{ xaxis:{{title:'training step'}}, margin:{{t:6, r:8, b:34, l:40}},
                 legend:{{orientation:'h', y:1.0, yanchor:'bottom', font:{{size:9}}}}, shapes:[vline] }};
  Plotly.react(`cr_${{ci}}`,
    [ln('retain_both','2-adapter',C_BOTH), ln('retain_ronly','retain-only',C_RONLY)],
    Object.assign({{yaxis:{{title:'retain reward'}}}}, base), {{displayModeBar:false}});
  Plotly.react(`ch_${{ci}}`, [
    ln('hack_both_mon','2-adapter mon',C_BOTH,'solid'), ln('hack_both_unmon','2-adapter unmon',C_BOTH,'dot'),
    ln('hack_ronly_mon','retain-only mon',C_RONLY,'solid'), ln('hack_ronly_unmon','retain-only unmon',C_RONLY,'dot'),
  ], Object.assign({{yaxis:{{title:'hack rate'}}}}, base), {{displayModeBar:false}});
}}

function render() {{
  const E = curEnv();
  idx = parseInt(document.getElementById('slider').value);
  const step = E.sliderSteps[idx];
  document.getElementById('lbl').textContent = `step ${{step}}`;
  E.conds.forEach((cond, ci) => {{
    const cd = E.C[cond];
    if (cd.has_grad) {{
      const j = cd.steps.indexOf(step);
      for (const [m, role] of GRID_CELLS) {{
        const id = `p_${{ci}}_${{m}}_${{role}}`;
        if (document.getElementById(id)) panel(id, cd, m, role, j);
      }}
      renderScatter(ci, cd, j);
    }}
    if (document.getElementById(`cr_${{ci}}`)) {{
      Plotly.relayout(`cr_${{ci}}`, {{'shapes[0].x0': step, 'shapes[0].x1': step}});
      Plotly.relayout(`ch_${{ci}}`, {{'shapes[0].x0': step, 'shapes[0].x1': step}});
    }}
  }});
}}
function setSlider(i) {{ document.getElementById('slider').value = i; render(); }}
function prev() {{ if (idx>0) setSlider(idx-1); }}
function next() {{ const n=curEnv().sliderSteps.length; if (idx<n-1) setSlider(idx+1); }}
function togglePlay() {{
  playing=!playing; document.getElementById('play').textContent=playing?'Pause':'Play';
  if (playing) timer=setInterval(()=>{{ const n=curEnv().sliderSteps.length; if (idx<n-1) next(); else togglePlay(); }}, 600);
  else clearInterval(timer);
}}
document.addEventListener('keydown',(e)=>{{ if(e.key==='ArrowLeft')prev(); else if(e.key==='ArrowRight')next(); }});

onEnv();
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


def build_allenvs_html(data_by_env, classes_meta):
    """Overview page: every env's 2x2 grid (gradient/activation x retain/forget)
    shown concurrently in an outer grid, for a selected condition, one shared
    step slider. Reuses the shared panel() core."""
    # overview only needs the histogram panels — drop scatter/curves to shrink.
    reduced = {
        env: {"layers": d["layers"], "conds": d["conds"],
              "C": {c: {"has_grad": cd["has_grad"], "steps": cd["steps"],
                        "panels": cd["panels"], "class_n": cd.get("class_n")}
                    for c, cd in d["C"].items()}}
        for env, d in data_by_env.items()
    }
    data_json = json.dumps(reduced, separators=(",", ":"))
    cls_json = json.dumps([{"k": k, "label": lbl, "color": col} for k, lbl, col in classes_meta])
    envs = sorted(data_by_env)
    # conditions that have grad data somewhere
    conds = [c for c in ["GR", "RP", "filter", "do-nothing"]
             if any(c in d["C"] and d["C"][c]["has_grad"] for d in data_by_env.values())]
    cond_opts = "".join(f'<option value="{c}">{c}</option>' for c in conds)
    CELLS = [("grad", "retain", "grad·retain"), ("grad", "forget", "grad·forget"),
             ("act", "retain", "act·retain"), ("act", "forget", "act·forget")]
    # static env cells (each = title + 2x2 mini-grid of 4 panel divs)
    cells_html = []
    for ei, env in enumerate(envs):
        minis = "".join(
            f'<div><div class="mt">{lbl}</div><div id="o_{ei}_{m}_{role}" class="mplot"></div></div>'
            for m, role, lbl in CELLS)
        cells_html.append(
            f'<div class="envcell"><div class="envtitle">{env}</div>'
            f'<div class="mini">{minis}</div></div>')
    envcells = "\n".join(cells_html)
    envs_json = json.dumps(envs)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>All-envs 2x2 overview</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  body {{ font-family: sans-serif; background:#f5f5f5; margin:18px; }}
  .controls {{ display:flex; align-items:center; gap:14px; margin:10px 0; flex-wrap:wrap; position:sticky; top:0; background:#f5f5f5; z-index:10; padding:6px 0; }}
  input[type=range] {{ width:360px; }}
  select,button {{ padding:5px 10px; font-size:14px; }}
  #lbl {{ font-weight:bold; }}
  .envgrid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
  .envcell {{ border:1px solid #ddd; border-radius:6px; padding:6px; background:#fff; }}
  .envtitle {{ font-weight:bold; font-size:13px; text-align:center; margin-bottom:2px; }}
  .mini {{ display:grid; grid-template-columns:1fr 1fr; gap:4px; }}
  .mplot {{ height:165px; }}
  .mt {{ font-size:11px; text-align:center; color:#444; }}
  .hint {{ color:#888; font-size:12px; margin-bottom:6px; }}
</style></head><body>
  <h1 style="font-size:19px;">All envs &mdash; 2&times;2 (gradient/activation &times; retain/forget) at a glance</h1>
  <div class="controls">
    <label>Condition: <select id="cond" onchange="onCond()">{cond_opts}</select></label>
    <label>Layer: <select id="layer" onchange="render()"></select></label>
    <button onclick="prev()">&larr;</button>
    <input type="range" id="slider" min="0" value="0" oninput="render()">
    <button onclick="next()">&rarr;</button>
    <button id="play" onclick="togglePlay()">Play</button>
    <span id="lbl"></span>
  </div>
  <div class="hint"><b style="color:#1f3a93">retain</b> / <b style="color:#f1948a">forget·detectable</b> /
   <b style="color:#922b21">forget·undetectable</b> samples. Same step slider across all envs.</div>
  <div class="envgrid">
{envcells}
  </div>
<script>
const DATA = {data_json};
const CLS = {cls_json};
const ENVS = {envs_json};
const CELLS = [['grad','retain'],['grad','forget'],['act','retain'],['act','forget']];
let idx = 0, playing = false, timer = null, STEPS = [];
function layerKey() {{ return document.getElementById('layer').value; }}
function curCond() {{ return document.getElementById('cond').value; }}
{_SHARED_JS}
function initLayer() {{
  let layers = [];
  for (const e of ENVS) {{ if (DATA[e].layers && DATA[e].layers.length) {{ layers = DATA[e].layers; break; }} }}
  let o = '<option value="whole">all layers (whole-model)</option>';
  for (const li of layers) o += `<option value="${{li}}">layer ${{li}}</option>`;
  document.getElementById('layer').innerHTML = o;
}}
function allSteps() {{
  const cond = curCond(); const s = new Set();
  for (const e of ENVS) {{ const cd = DATA[e].C[cond]; if (cd && cd.has_grad) cd.steps.forEach(x => s.add(x)); }}
  return Array.from(s).sort((a,b)=>a-b);
}}
function onCond() {{
  STEPS = allSteps();
  const sl = document.getElementById('slider');
  sl.max = Math.max(0, STEPS.length - 1);
  if (idx > STEPS.length - 1) idx = STEPS.length - 1; if (idx < 0) idx = 0;
  sl.value = idx; render();
}}
function render() {{
  idx = parseInt(document.getElementById('slider').value);
  const step = STEPS[idx], cond = curCond();
  document.getElementById('lbl').textContent = `condition ${{cond}} \\u00b7 step ${{step}}`;
  ENVS.forEach((env, ei) => {{
    const cd = DATA[env].C[cond];
    const j = (cd && cd.has_grad) ? cd.steps.indexOf(step) : -1;
    for (const [m, role] of CELLS) {{
      panel(`o_${{ei}}_${{m}}_${{role}}`, (cd || {{panels:{{}}}}), m, role, j);
    }}
  }});
}}
function setSlider(i) {{ document.getElementById('slider').value = i; render(); }}
function prev() {{ if (idx>0) setSlider(idx-1); }}
function next() {{ if (idx<STEPS.length-1) setSlider(idx+1); }}
function togglePlay() {{
  playing=!playing; document.getElementById('play').textContent=playing?'Pause':'Play';
  if (playing) timer=setInterval(()=>{{ if (idx<STEPS.length-1) next(); else togglePlay(); }}, 600);
  else clearInterval(timer);
}}
document.addEventListener('keydown',(e)=>{{ if(e.key==='ArrowLeft')prev(); else if(e.key==='ArrowRight')next(); }});
initLayer(); onCond();
</script></body></html>"""


COND_ORDER = ["GR", "RP", "filter", "do-nothing"]


def classify_condition(run_dir):
    """Condition label from run_config.yaml. Base label = GR / RP / filter /
    do-nothing (routing_mode / reward_penalty_baseline / filter_baseline), with
    hyperparameter modifiers appended so otherwise-identical sweeps stay
    distinct: '·wd0.1' (weight_decay>0) and '·fn2' (forget_neurons != retain_neurons).
    So baseline=GR, weight-decay sweep=GR·wd0.1, asymmetric-adapter sweep=GR·fn2."""
    rm, rp, fb = None, False, False
    wd, rn, fn = 0.0, None, None
    cfg = os.path.join(run_dir, "run_config.yaml")
    if os.path.exists(cfg):
        try:
            import yaml
            c = yaml.safe_load(open(cfg)) or {}
            rm = c.get("routing_mode")
            rp = bool(c.get("reward_penalty_baseline"))
            fb = bool(c.get("filter_baseline"))
            wd = c.get("weight_decay") or 0.0
            rn, fn = c.get("retain_neurons"), c.get("forget_neurons")
        except Exception:
            pass
    if rm is None:
        n = os.path.basename(run_dir)
        if "_gr_" in n or "_gr" in n:
            rm = "classic"
        elif "_rp" in n:
            rp, rm = True, "none"
        elif "no_intervention" in n or "nohack" in n:
            rm = "none"
        else:
            rm = "none"
    if rm and rm != "none":
        base = "GR"
    elif rp:
        base = "RP"
    elif fb:
        base = "filter"
    else:
        base = "do-nothing"
    mods = []
    if wd and float(wd) > 0:
        mods.append(f"wd{float(wd):g}")
    if rn is not None and fn is not None and fn != rn:
        mods.append(f"fn{fn}")
    return base + ("·" + "·".join(mods) if mods else "")


def generate(sweep_dirs, out=None, require_grad=True, verbose=True):
    """Build the separability viewer (separability_dist.html + _allenvs.html) for
    one or more sweep dirs, grouping runs by (env, condition). Returns the
    dist.html path, or None if there's nothing to build (no runs, or — when
    require_grad — no grad_diag.jsonl anywhere). Safe to call repeatedly on a
    live sweep: tolerates partially-written jsonl. Used by main() and by
    sweep.py's periodic regeneration."""
    if isinstance(sweep_dirs, str):
        sweep_dirs = [sweep_dirs]

    # discover run subdirs (anything with grad_diag.jsonl or routing_eval.jsonl)
    run_dirs = []
    for sd in sweep_dirs:
        for sub in sorted(glob.glob(os.path.join(sd, "*"))):
            if os.path.isdir(sub) and (
                    os.path.exists(os.path.join(sub, "grad_diag.jsonl"))
                    or os.path.exists(os.path.join(sub, "routing_eval.jsonl"))):
                run_dirs.append(sub)
    if not run_dirs:
        return None
    if require_grad and not any(
            os.path.exists(os.path.join(rd, "grad_diag.jsonl")) for rd in run_dirs):
        return None  # diagnostic not enabled for this sweep — skip

    # group by (env, condition)
    groups = defaultdict(lambda: {"grad": [], "dirs": []})
    for rd in run_dirs:
        env = env_of(os.path.basename(rd))
        cond = classify_condition(rd)
        gp = os.path.join(rd, "grad_diag.jsonl")
        if os.path.exists(gp):
            recs = []
            for l in open(gp):
                l = l.strip()
                if not l:
                    continue
                try:
                    recs.append(json.loads(l))
                except json.JSONDecodeError:
                    pass  # tolerate a partial final line while the run is live
            if recs:
                groups[(env, cond)]["grad"].append(recs)
        groups[(env, cond)]["dirs"].append(rd)

    # Pick the sample-class scheme once (ground-truth taxonomy if any record
    # carries hackable/hacked, else legacy is_rh). All envs share one scheme.
    rec0 = next((g["grad"][0][0] for g in groups.values() if g["grad"]), None)
    classes_meta, classify = _scheme(rec0) if rec0 else (LEGACY_CLASSES, _classify_legacy)
    class_keys = [k for k, _, _ in classes_meta]

    data_by_env = {}
    for (env, cond), g in groups.items():
        d = data_by_env.setdefault(env, {"C": {}, "conds": [], "layers": [], "sliderSteps": []})
        if g["grad"]:
            cd = build_env_data(g["grad"], class_keys, classify)
            cd["has_grad"] = True
            if not d["layers"]:
                d["layers"] = cd["layers"]
        else:
            cd = {"has_grad": False, "steps": [], "panels": {}, "scatter": {},
                  "class_n": {}, "layers": []}
        cd["curves"] = load_curves(g["dirs"])
        d["C"][cond] = cd

    for env, d in data_by_env.items():
        # order by base-condition (COND_ORDER prefix) then modifier label, so
        # GR, GR·fn2, GR·wd0.1 stay adjacent and dynamic labels aren't dropped.
        d["conds"] = sorted(d["C"].keys(), key=lambda c: (
            next((i for i, p in enumerate(COND_ORDER) if c.startswith(p)), len(COND_ORDER)), c))
        grad_steps = sorted(set().union(*[set(d["C"][c]["steps"])
                                          for c in d["conds"] if d["C"][c]["has_grad"]] or [set()]))
        if not grad_steps:  # curve-only env: slider over curve steps
            grad_steps = sorted(set().union(*[set(d["C"][c]["curves"]["steps"])
                                              for c in d["conds"]] or [set()]))
        d["sliderSteps"] = grad_steps
    if not data_by_env:
        return None

    out = out or os.path.join(sweep_dirs[0], "separability", "separability_dist.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(build_html(data_by_env, classes_meta))
    n_blocks = sum(len(d["conds"]) for d in data_by_env.values())
    if verbose:
        print(f"[separability] wrote {out} ({len(data_by_env)} envs, "
              f"{n_blocks} condition-blocks, {os.path.getsize(out)/1e6:.1f} MB)")

    # all-envs overview page (every env's 2x2 at a glance)
    allenvs_out = os.path.join(os.path.dirname(out), "separability_allenvs.html")
    with open(allenvs_out, "w") as f:
        f.write(build_allenvs_html(data_by_env, classes_meta))
    if verbose:
        print(f"[separability] wrote {allenvs_out} ({os.path.getsize(allenvs_out)/1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="+",
                    help="one or more sweep dirs; runs are grouped by (env, condition)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    # CLI builds whatever's present (even curve-only); require_grad only gates
    # the automatic sweep.py path.
    out = generate(args.sweep_dir, out=args.output, require_grad=False)
    assert out, f"no usable data under {args.sweep_dir}"


if __name__ == "__main__":
    main()
