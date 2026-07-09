"""Shared loader for posthoc forget-scale-eval JSONs (the v4 figure family).

An fseval JSON (tools/modal_train_gr.eval_forget_scales_one) is
  {"run_name": ..., "step": ..., "n_eval": 256,
   "scales": {"0.0": {"<channel>/<detector...>": v, ...}, "1.0": {...}, ...}}

Metric semantics match aggregated_cache.json / the v2-v3 panels:
  retain = 'retain' (full eval set), hack = 'hack_freq' (overall,
  deployment-observed). Channel lookup is exact on the first key segment so
  'hack_freq' never matches 'hack_freq_detectable'.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pick(scale_map, slug):
    hits = [v for k, v in scale_map.items() if k.split("/", 1)[0] == slug]
    assert len(hits) == 1, f"{slug}: {len(hits)} matches"
    return float(hits[0])


def load_recs(fsdir, glob_pat):
    """All fseval records in `fsdir` matching `glob_pat` (filename glob)."""
    paths = sorted(glob.glob(os.path.join(ROOT, fsdir, glob_pat)))
    return [json.load(open(p)) for p in paths]


def seed_points(fsdir, glob_pat, scale, n_expected=None):
    """Per-seed (retain, hack) at `scale` for the runs matching the glob.
    Returns list of (retain, hack)."""
    recs = load_recs(fsdir, glob_pat)
    if n_expected is not None:
        assert len(recs) == n_expected, \
            f"{fsdir}/{glob_pat}: expected {n_expected} fseval files, found {len(recs)}"
    pts = []
    for rec in recs:
        sm = rec["scales"][scale]
        pts.append((pick(sm, "retain"), pick(sm, "hack_freq")))
    return pts


def agg(points):
    """(retain_mean, retain_std, hack_mean, hack_std, n) — the draw_point
    tuple convention (std over seeds, ddof=0, same as aggregated_cache)."""
    if not points:
        return None
    rs = np.array([p[0] for p in points])
    hs = np.array([p[1] for p in points])
    return (float(rs.mean()), float(rs.std(ddof=0)),
            float(hs.mean()), float(hs.std(ddof=0)), len(points))
