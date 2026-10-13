#!/usr/bin/env python
"""Render grad_diag.jsonl into an animated GIF (the "played" view, frame per step).

Mirrors grad_diag.html's panels with the same fixed-across-steps axes, but
rendered deterministically with matplotlib (no browser/screencap needed):
  - top row: by-data-type histograms (whole-model), retain data | forget data,
    each comparing forget-param vs retain-param per-sample grad-norm counts;
  - bottom row: per-layer MEDIAN per-sample grad norm, same data split.

Usage:
  python tools/grad_diag_gif.py output/<run>/            # -> grad_diag.gif beside the jsonl
  python tools/grad_diag_gif.py path/to/grad_diag.jsonl -o out.gif --duration 700
"""
import argparse
import io
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RP, FP = "#1f3a93", "#c0392b"   # retain-param blue, forget-param red
NBINS = 40


def _load(jsonl):
    recs = [json.loads(l) for l in open(jsonl) if l.strip()]
    recs.sort(key=lambda r: r["step"])
    return recs


def _global_bins(recs):
    v = []
    for r in recs:
        v += r["whole_model"]["retain"] + r["whole_model"]["forget"]
    v = np.array([x for x in v if x > 0])
    v.sort()
    lo, hi = v[int(0.005 * len(v))], v[int(0.995 * len(v))]
    edges = np.linspace(math.log10(lo), math.log10(hi), NBINS + 1)  # log10 space
    return edges


def _count_max(recs, edges):
    ymax = 1
    for r in recs:
        rh = np.array(r["is_rh"])
        for cls in (0, 1):
            for role in ("retain", "forget"):
                a = np.array(r["whole_model"][role])[rh == cls]
                a = a[a > 0]
                if a.size:
                    c = np.histogram(np.log10(a), bins=edges)[0]
                    ymax = max(ymax, int(c.max()))
    return ymax


def _median_range(recs):
    lo, hi = np.inf, 0.0
    for r in recs:
        rh = np.array(r["is_rh"])
        for k in range(len(r["layers"])):
            for role in ("retain", "forget"):
                for cls in (0, 1):
                    a = np.array(r["per_sample"][role][k])[rh == cls]
                    if a.size:
                        m = np.median(a)
                        if m > 0:
                            lo, hi = min(lo, m), max(hi, m)
    return lo * 0.8, hi * 1.25


def _hist_panel(ax, rec, cls, edges, ymax, title):
    rh = np.array(rec["is_rh"])
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.95
    for role, color in (("retain", RP), ("forget", FP)):
        a = np.array(rec["whole_model"][role])[rh == cls]
        a = a[a > 0]
        n = int((rh == cls).sum())
        counts = np.histogram(np.log10(a), bins=edges)[0] if a.size else np.zeros(NBINS)
        ax.bar(centers, counts, width=width, color=color, alpha=0.5,
               label=f"{role} params (n={n})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("per-sample grad norm (log10)")
    ax.set_ylabel("number of samples")
    ax.set_ylim(0, ymax * 1.05)
    ax.legend(fontsize=8, loc="upper right")


def _line_panel(ax, rec, cls, yrange, title):
    rh = np.array(rec["is_rh"])
    x = rec["layers"]
    for role, color in (("retain", RP), ("forget", FP)):
        y = []
        for k in range(len(x)):
            a = np.array(rec["per_sample"][role][k])[rh == cls]
            y.append(np.median(a) if a.size else np.nan)
        ax.plot(x, y, "-o", ms=3, color=color, label=f"{role} params")
    ax.set_yscale("log")
    ax.set_ylim(*yrange)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("layer")
    ax.set_ylabel("median per-sample grad norm")
    ax.legend(fontsize=8, loc="upper right")


def _frame(rec, edges, ymax, yrange):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    nh = int((np.array(rec["is_rh"]) == 1).sum())
    fig.suptitle(f"step {rec['step']}  ·  {rec['n_samples']} samples  ·  {nh} hack (is_rh=1)",
                 fontsize=13, y=0.99)
    _hist_panel(axes[0, 0], rec, 0, edges, ymax, "Retain data (is_rh=0)")
    _hist_panel(axes[0, 1], rec, 1, edges, ymax, "Forget data (is_rh=1, hacks)")
    _line_panel(axes[1, 0], rec, 0, yrange, "Per-layer median — retain data")
    _line_panel(axes[1, 1], rec, 1, yrange, "Per-layer median — forget data")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=90)
    plt.close(fig)
    b.seek(0)
    return Image.open(b).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="run dir with grad_diag.jsonl, or the jsonl itself")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--duration", type=int, default=600, help="ms per frame")
    args = ap.parse_args()

    jsonl = args.path
    if os.path.isdir(jsonl):
        jsonl = os.path.join(jsonl, "grad_diag.jsonl")
    recs = _load(jsonl)
    assert recs, f"{jsonl} empty"

    edges = _global_bins(recs)
    ymax = _count_max(recs, edges)
    yrange = _median_range(recs)
    frames = [_frame(r, edges, ymax, yrange) for r in recs]

    out = args.output or os.path.join(os.path.dirname(jsonl), "grad_diag.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.duration, loop=0, optimize=True)
    print(f"[grad_diag] wrote {out} ({len(frames)} frames @ {args.duration}ms)")


if __name__ == "__main__":
    main()
