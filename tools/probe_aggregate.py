"""Aggregate grad-probe records into per-cell summaries.

In: probe_data/<env>_probe.jsonl (per-position records with per-layer metrics)
Out: probe_data/<env>_probe_agg.json — per (step, cls, traj): n, median/q25/q75 of
     Lambda (mean-over-layers rho_F/rho_R), rho_r, rho_f, s_r, s_f, hn_r, hn_f, beta_r, beta_f,
     dn, loss. Plus a per-layer Lambda median table at hack_onset for the layer profile.
"""
import json
import sys

import numpy as np


def q(v):
    v = np.array(v, float)
    return dict(med=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), n=len(v))


def main(path, out):
    cells = {}
    layer_lam = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["step"], r["cls"], r["traj"])
        lay = r["layers"]
        rho_r = np.mean([v["rho_r"] for v in lay.values()])
        rho_f = np.mean([v["rho_f"] for v in lay.values()])
        rec = cells.setdefault(key, {k: [] for k in
                                     ("lam", "rho_r", "rho_f", "s_r", "s_f", "hn_r", "hn_f",
                                      "beta_r", "beta_f", "dn", "loss")})
        rec["lam"].append(rho_f / max(rho_r, 1e-12))
        rec["rho_r"].append(rho_r)
        rec["rho_f"].append(rho_f)
        rec["loss"].append(r["loss"])
        for m in ("s_r", "s_f", "hn_r", "hn_f", "beta_r", "beta_f", "dn"):
            vals = [v[m] for v in lay.values() if m in v]
            if vals:
                rec[m].append(float(np.mean(vals)))
        if r["cls"] == "hack_onset":
            ll = layer_lam.setdefault(r["step"], {})
            for li, v in lay.items():
                ll.setdefault(li, []).append(v["rho_f"] / max(v["rho_r"], 1e-12))

    out_obj = {"cells": [], "hack_onset_layer_lambda": {}}
    for (step, cls, traj), rec in sorted(cells.items()):
        row = {"step": step, "cls": cls, "traj": traj}
        for k, v in rec.items():
            if v:
                row[k] = q(v)
        out_obj["cells"].append(row)
    for step, ll in sorted(layer_lam.items()):
        out_obj["hack_onset_layer_lambda"][str(step)] = {
            li: float(np.median(v)) for li, v in sorted(ll.items(), key=lambda x: int(x[0]))}
    json.dump(out_obj, open(out, "w"))
    print(f"wrote {out} ({len(out_obj['cells'])} cells)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
