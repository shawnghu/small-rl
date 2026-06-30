#!/usr/bin/env python
"""Warm-start data with the FORGET set composed at a target unmonitored fraction.

Generalizes extract_warmstart_v2.py: the forget set is built from hackable hacks
split by the ground-truth `detectable` flag, mixed to a requested unmonitored
fraction. Retain set uses the original criteria (hacked_gt==False, retain_score
>= score_frac*max). Same source pipeline (per-(sweep,env) step floor, dedup,
water-fill + even-subsample across sweeps).

    --unmon_frac 0.0  -> MONITORABLE-ONLY forget (realistic: we cannot author
                         examples of hacks the detector misses). [default]
    --unmon_frac 0.5  -> the v2 topic mix.
    --unmon_frac 1.0  -> unmonitorable-only.

Samples with detectable==None (detector status unrecorded) are excluded from the
forget pool so the monitored/unmonitored split is clean.

    python -m tools.extract_warmstart_forget_comp --unmon_frac 0.0 \
        --out_dir warmstart_data_monitored \
        --envs repeat addition_v2 topic object_qa cities_qa persona_qa
"""
import argparse, json, os, collections
from tools.extract_warmstart_v2 import candidates, select, retain_floor, FORGET_FLOOR

DEFAULT_ENVS = ["repeat", "addition_v2", "topic", "object_qa", "cities_qa", "persona_qa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unmon_frac", type=float, default=0.0)
    ap.add_argument("--out_dir", default="warmstart_data_monitored")
    ap.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    ap.add_argument("--cap_forget", type=int, default=500)
    ap.add_argument("--cap_retain", type=int, default=500)
    ap.add_argument("--retain_thresh", type=float, default=None,
                    help="Absolute retain_score floor for the retain class (e.g. 0.8 for "
                         "'unusually high' retain reward). None = default score_frac*max.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}

    for env in a.envs:
        tfloor = a.retain_thresh if a.retain_thresh is not None else retain_floor(env)

        def want(r, rk, fk, _tfloor=tfloor):
            if r.get("hacked_gt"):
                if (r.get(fk) or 0.0) <= FORGET_FLOOR:
                    return None
                if not r.get("hackable"):
                    return None  # only real (rewardable) hacks
                # Ground-truth monitored label: prefer the prompt-level `detectable`
                # flag (logged only for topic); else the detector's actual firing
                # `is_rh` (the deterministic program detector's output, populated for
                # the sycophancy_conditional envs). Exclude only if both are None.
                d = r.get("detectable")
                if d is None:
                    d = r.get("is_rh")
                if d is True:
                    return "forget_mon"
                if d is False:
                    return "forget_unmon"
                return None  # neither label available -> exclude from the clean split
            return "retain" if (r.get(rk) or 0.0) >= _tfloor else None

        pools = candidates(env, want)
        n_unmon = round(a.unmon_frac * a.cap_forget)
        n_mon = a.cap_forget - n_unmon
        fg = select(pools, "forget_unmon", n_unmon) + select(pools, "forget_mon", n_mon)
        for r in fg:
            r["cls"] = "forget"
        rt = select(pools, "retain", a.cap_retain)

        path = os.path.join(a.out_dir, f"{env}.jsonl")
        with open(path, "w") as fo:
            for row in rt + fg:
                fo.write(json.dumps(row) + "\n")
        def _det(r):
            return r.get("detectable") if r.get("detectable") is not None else r.get("is_rh")
        un = sum(1 for r in fg if _det(r) is False)
        mo = sum(1 for r in fg if _det(r) is True)
        avail = {c.split("_", 1)[1] if "_" in c else c: sum(len(pools[(c, sw)]) for (cc, sw) in pools if cc == c)
                 for c in {k[0] for k in pools}}
        summary[env] = {
            "out": path, "n_retain": len(rt), "n_forget": len(fg),
            "forget_unmon": un, "forget_mon": mo,
            "forget_unmon_frac": round(un / max(1, len(fg)), 3),
            "target_unmon_frac": a.unmon_frac,
            "available": avail,
        }

    with open(os.path.join(a.out_dir, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
