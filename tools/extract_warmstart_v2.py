#!/usr/bin/env python
"""Targeted warm-start re-collection for topic + sorting (v2).

Same source pipeline as extract_warmstart_multisweep.py (pools trace="sample"
records across the same sweeps, per-(sweep,env) step floor, dedup by
prompt+completion, water-fill + even-subsample across sweeps), but modifies ONLY
the targeted class per env, using GROUND-TRUTH labels from the trace:

  topic  -> FORGET set rebuilt as a 50/50 mix of unmonitored vs monitored hacks
            (hackable hacks split by the trace `detectable` flag). retain class
            unchanged from the original criteria.
  sorting-> RETAIN set rebuilt from "unusually high" retain reward only
            (retain_score >= SORT_RETAIN_THRESH). forget class unchanged.

Writes warmstart_data_v2/{topic,sorting}.jsonl (+ summary). The other class in
each file uses the original criteria so the file is a drop-in for warmstart.py.
"""
import os, json, hashlib, collections, yaml
from tools.extract_warmstart_multisweep import (
    load_samples, component_roles, even_subsample, water_fill, DEFAULT_SWEEPS,
)

OUT_DIR = "warmstart_data_v2"
MIN_STEP_FRAC = 0.25      # same as original default
SCORE_FRAC = 0.5          # original retain floor (used for topic retain + as fallback)
FORGET_FLOOR = 0.0        # original: hack must manifest (hack_score > 0)
CAP_FORGET = 500
CAP_RETAIN = 500
SORT_RETAIN_THRESH = 0.8  # "unusually high" retain reward (absolute)
TOPIC_UNMON_FRAC = 0.5    # 50% unmonitored floor


def gather(env_short):
    """Return list of (sweep, run, rk, fk, seed, [sample rows])."""
    runs = []
    for sweep in DEFAULT_SWEEPS:
        base = os.path.join("output", sweep)
        if not os.path.isdir(base):
            print(f"WARN missing sweep dir {base}")
            continue
        for run in sorted(os.listdir(base)):
            rc = os.path.join(base, run, "run_config.yaml")
            tr = os.path.join(base, run, "routing_trace.jsonl")
            if not (os.path.isfile(rc) and os.path.isfile(tr)):
                continue
            cfg = yaml.safe_load(open(rc))
            if cfg.get("environment") != env_short:
                continue
            rk, fk = component_roles(cfg["config_path"])
            runs.append((sweep, run, rk, fk, cfg.get("seed"), load_samples(tr)))
    return runs


def row_of(r, env, cls, rk, fk, sweep, run, seed):
    return {
        "env": env, "cls": cls, "sweep": sweep,
        "prompt": r.get("prompt"), "completion": r.get("completion"),
        "hacked_gt": r.get("hacked_gt"), "is_rh": r.get("is_rh"),
        "hackable": r.get("hackable"), "detectable": r.get("detectable"),
        "retain_score": r.get(rk) or 0.0, "hack_score": r.get(fk) or 0.0,
        "raw_reward": r.get("raw_reward"), "step": r.get("step"),
        "run": run, "seed": seed,
    }


def candidates(env, want):
    """want(r, rk, fk) -> cls or None. Returns {sweep: [rows]} step-sorted, deduped."""
    runs = gather(env)
    maxstep = collections.defaultdict(int)
    for sweep, run, rk, fk, seed, recs in runs:
        for r in recs:
            maxstep[sweep] = max(maxstep[sweep], r.get("step", 0))
    pools = collections.defaultdict(list)
    seen = set()
    for sweep, run, rk, fk, seed, recs in runs:
        sf = MIN_STEP_FRAC * maxstep[sweep]
        for r in recs:
            if r.get("step", 0) < sf:
                continue
            comp = (r.get("completion") or "").strip()
            if not comp:
                continue
            cls = want(r, rk, fk)
            if cls is None:
                continue
            key = hashlib.md5((str(r.get("prompt")) + "\x00" + comp).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pools[(cls, sweep)].append(row_of(r, env, cls, rk, fk, sweep, run, seed))
    for k in pools:
        pools[k].sort(key=lambda x: x["step"])
    return pools


def select(pools, cls, target):
    """water-fill + even-subsample `target` rows of class `cls` across sweeps."""
    avail = {sw: len(pools[(cls, sw)]) for (c, sw) in pools if c == cls}
    alloc = water_fill(avail, min(target, sum(avail.values())))
    out = []
    for sw, k in alloc.items():
        if k > 0:  # even_subsample(pool, 0) wrongly returns [rows[0]]; skip empties
            out.extend(even_subsample(pools[(cls, sw)], k))
    return out


def retain_floor(env):
    """per-env retain_score max across source, * SCORE_FRAC (original retain rule)."""
    runs = gather(env)
    rmax = 0.0
    for sweep, run, rk, fk, seed, recs in runs:
        for r in recs:
            rmax = max(rmax, r.get(rk) or 0.0)
    return SCORE_FRAC * (rmax or 1.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}

    # ---- TOPIC: forget = 50/50 unmonitored/monitored hackable hacks ----
    tfloor = retain_floor("topic")
    def topic_want(r, rk, fk):
        if r.get("hacked_gt"):
            if (r.get(fk) or 0.0) <= FORGET_FLOOR:
                return None
            if not r.get("hackable"):
                return None  # only real (rewardable) hacks
            return "forget_unmon" if r.get("detectable") is False else (
                   "forget_mon" if r.get("detectable") is True else None)
        else:
            return "retain" if (r.get(rk) or 0.0) >= tfloor else None
    tp = candidates("topic", topic_want)
    n_un = CAP_FORGET // 2
    fg = select(tp, "forget_unmon", n_un) + select(tp, "forget_mon", CAP_FORGET - n_un)
    for row in fg:
        row["cls"] = "forget"
    rt = select(tp, "retain", CAP_RETAIN)
    _write("topic", rt, fg, summary,
           note=f"forget 50/50 unmon/mon by `detectable`; unmon_floor={TOPIC_UNMON_FRAC}")

    # ---- SORTING: retain = unusually-high retain reward only ----
    def sorting_want(r, rk, fk):
        if r.get("hacked_gt"):
            return "forget" if (r.get(fk) or 0.0) > FORGET_FLOOR else None
        else:
            return "retain" if (r.get(rk) or 0.0) >= SORT_RETAIN_THRESH else None
    sp = candidates("sorting", sorting_want)
    rt = select(sp, "retain", CAP_RETAIN)
    fg = select(sp, "forget", CAP_FORGET)
    _write("sorting", rt, fg, summary,
           note=f"retain retain_score>={SORT_RETAIN_THRESH}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=2)
    print(json.dumps(summary, indent=2))


def _write(env, retain_rows, forget_rows, summary, note):
    path = os.path.join(OUT_DIR, f"{env}.jsonl")
    with open(path, "w") as fo:
        for row in retain_rows + forget_rows:
            fo.write(json.dumps(row) + "\n")
    fd = forget_rows
    un = sum(1 for r in fd if r.get("detectable") is False)
    mo = sum(1 for r in fd if r.get("detectable") is True)
    summary[env] = {
        "out": path, "n_retain": len(retain_rows), "n_forget": len(forget_rows),
        "forget_unmon": un, "forget_mon": mo,
        "forget_unmon_frac": round(un / max(1, len(fd)), 3),
        "retain_score_min": round(min((r["retain_score"] for r in retain_rows), default=0), 3),
        "note": note,
    }


if __name__ == "__main__":
    main()
