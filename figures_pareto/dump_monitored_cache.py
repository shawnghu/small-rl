"""Dump (monitored, unmonitored) per (class, env) to JSON for the
proto_pareto_monitored_partial_forget figure (and the v1/v2 versions).

Run on a machine that has the canonical run dirs (gr_canonical_redo_4envs,
cspr32_gr_and_reruns, etc.). Reads routing_eval.jsonl for each (class, env)
and writes monitored_cache.json next to this file.

Run:
    .venv/bin/python figures_pareto/dump_monitored_cache.py
"""
import json
import os

import numpy as np

from proto_pareto_data import ENVS
from proto_pareto_monitored_v1 import CLASSES, class_point


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "monitored_cache.json")


def main():
    cache = {}
    for cname, spec in CLASSES.items():
        per_env = {}
        for env in ENVS:
            x, y, n = class_point(env, spec)
            if n == 0:
                per_env[env] = None
            else:
                per_env[env] = {"monitored": float(x), "unmonitored": float(y), "n": int(n)}
        cache[cname] = per_env

    with open(OUT, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"wrote {OUT}")
    for cname, env_dict in cache.items():
        present = sum(1 for v in env_dict.values() if v)
        print(f"  {cname:22s}  {present}/{len(env_dict)} envs populated")


if __name__ == "__main__":
    main()
