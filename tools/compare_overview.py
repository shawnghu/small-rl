"""Validation-vs-reference comparison overview: one self-contained HTML that
overlays per-seed routing_eval curves from two run sets with unambiguous
condition labels (the generic sweep overview keys traces by seed only, so
mixed-condition dirs collide/drop runs — this generator exists for the
condition-vs-condition case).

Usage:
  .venv/bin/python tools/compare_overview.py \
      --test_glob 'output/stack_repeat_validation/s*' --test_label optimized_stack \
      --ref_glob 'output/binary_dynamics_5seeds-0602-2313/repeat_binary_*' --ref_label eager_reference \
      --out output/stack_repeat_validation/sweep_graphs/compare_vs_eager.html
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.compare_validation_curves import load_run, resolve_key  # noqa: E402

METRIC_PREFIXES = [
    ("both/combined", "Combined Reward (both adapters)"),
    ("both/retain", "Retain Reward (both adapters)"),
    ("both/hack_freq", "Hack Frequency (both adapters)"),
    ("retain_only/retain", "Retain Reward (retain-only ablation)"),
    ("retain_only/hack_freq", "Hack Frequency (retain-only ablation)"),
    ("forget_only/hack_freq", "Hack Frequency (forget-only ablation)"),
]

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"


def seed_of(run_dir):
    m = re.search(r"s(\d+)$", os.path.basename(run_dir.rstrip("/")))
    return m.group(1) if m else os.path.basename(run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_glob", required=True)
    ap.add_argument("--test_label", required=True)
    ap.add_argument("--ref_glob", required=True)
    ap.add_argument("--ref_label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_step", type=int, default=None)
    args = ap.parse_args()

    sets = []
    for label, pattern, color in ((args.ref_label, args.ref_glob, "rgba(120,120,120,0.55)"),
                                  (args.test_label, args.test_glob, "rgba(214,39,40,0.9)")):
        dirs = sorted(d for d in glob.glob(pattern)
                      if os.path.exists(os.path.join(d, "routing_eval.jsonl")))
        assert dirs, f"no runs (with routing_eval.jsonl) match {pattern}"
        runs = {seed_of(d): load_run(d) for d in dirs}
        if args.max_step is not None:
            runs = {s: [r for r in rows if r["step"] <= args.max_step] for s, rows in runs.items()}
        sets.append((label, color, runs))

    any_rows = next(iter(sets[0][2].values()))
    plots = []
    for prefix, title in METRIC_PREFIXES:
        try:
            key = resolve_key(any_rows, prefix)
        except AssertionError:
            continue
        traces = []
        for label, color, runs in sets:
            for i, (seed, rows) in enumerate(sorted(runs.items())):
                xs = [r["step"] for r in rows if r.get(key) is not None]
                ys = [r[key] for r in rows if r.get(key) is not None]
                traces.append({
                    "x": xs, "y": ys, "mode": "lines", "name": f"{label} (s{seed})",
                    "legendgroup": label, "showlegend": i == 0,
                    "line": {"color": color, "width": 1.6},
                    "hovertemplate": f"{label} s{seed}<br>step %{{x}}<br>%{{y:.4f}}<extra></extra>",
                })
        plots.append({"title": title, "key": key, "traces": traces})

    n = len(plots)
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
            f"<title>{args.test_label} vs {args.ref_label}</title>",
            f"<script src='{PLOTLY_CDN}'></script>",
            "<style>body{font-family:sans-serif;margin:18px} .grid{display:grid;"
            "grid-template-columns:repeat(2,minmax(420px,1fr));gap:14px}</style></head><body>",
            f"<h2>{args.test_label} (red) vs {args.ref_label} (grey)</h2>",
            f"<p>{args.test_glob} &nbsp;vs&nbsp; {args.ref_glob}</p><div class='grid'>"]
    for i in range(n):
        html.append(f"<div id='p{i}' style='height:340px'></div>")
    html.append("</div><script>")
    for i, p in enumerate(plots):
        layout = {"title": {"text": p["title"], "font": {"size": 14}},
                  "xaxis": {"title": "step"}, "margin": {"t": 40, "b": 40, "l": 50, "r": 10},
                  "legend": {"orientation": "h"}}
        html.append(f"Plotly.newPlot('p{i}', {json.dumps(p['traces'])}, {json.dumps(layout)},"
                    "{displayModeBar:false});")
    html.append("</script></body></html>")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(html))
    print(f"wrote {args.out} ({n} plots, "
          f"{sum(len(p['traces']) for p in plots)} traces)")


if __name__ == "__main__":
    main()
