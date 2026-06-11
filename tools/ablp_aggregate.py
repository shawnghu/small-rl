"""Aggregate ablation-logprob records against corpus token classes.

Per (step, token_class, traj_label): median/quartiles of
  lp_both                      logP(token) full model
  d_fog = lp_retain_only - lp_both    drop when FORGET ablated  (very negative => forget supplies)
  d_ret = lp_forget_only - lp_both    drop when RETAIN ablated  (very negative => retain supplies)
  d_base = lp_base - lp_both          total adapter contribution
Usage: ablp_aggregate.py <corpus.jsonl> <ablp.jsonl> <out.json>
"""
import json
import sys

import numpy as np


def traj_label(row):
    if not row["hack"]:
        return "clean"
    return "detected" if row["detected"] else "undetected"


def q(v):
    v = np.array(v, float)
    return dict(med=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), n=len(v))


def main(corpus_path, ablp_path, out_path):
    rows = [json.loads(l) for l in open(corpus_path)]
    cells = {}
    for line in open(ablp_path):
        r = json.loads(line)
        row = rows[r["row"]]
        tl = traj_label(row)
        for pos, cls in enumerate(row["token_classes"]):
            key = (r["step"], cls, tl)
            c = cells.setdefault(key, {"lp_both": [], "d_fog": [], "d_ret": [], "d_base": []})
            b = r["lp_both"][pos]
            c["lp_both"].append(b)
            c["d_fog"].append(r["lp_retain_only"][pos] - b)
            c["d_ret"].append(r["lp_forget_only"][pos] - b)
            c["d_base"].append(r["lp_base"][pos] - b)
    out = []
    for (step, cls, tl), c in sorted(cells.items()):
        out.append({"step": step, "cls": cls, "traj": tl,
                    **{k: q(v) for k, v in c.items()}})
    json.dump({"cells": out}, open(out_path, "w"))
    print(f"wrote {out_path} ({len(out)} cells)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
