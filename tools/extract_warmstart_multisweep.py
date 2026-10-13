#!/usr/bin/env python
"""Warm-start sample extraction pooled across recent sweeps (uniform mix).

For each small-scale env, pools trace="sample" records from every given sweep,
splits by GROUND TRUTH hack emission (hacked_gt), filters to coherent/on-task,
then selects <= --cap_per_env records per env with:
  - a uniform mix across sweeps (water-filled: short sweeps redistribute share)
  - both classes kept, balanced ~50/50 where data allows (retain fills the
    remainder when forget is short, and vice versa)
  - even subsampling across training step within each (sweep, class)

Filters:
  - forget class: hacked_gt==True  + step>=min_step_frac*maxstep + nonempty
                  (RELAXED: no hack-magnitude floor -> keeps low-magnitude hacks
                   like continuous sycophancy)
  - retain class: hacked_gt==False + retain_score>=score_frac*max + step floor + nonempty
"""
import argparse, json, os, hashlib, collections
import yaml

SMALL_ENVS = {"repeat", "addition_v2", "topic", "sorting", "object_qa", "cities_qa", "persona_qa"}
DEFAULT_SWEEPS = [
    "smallscale_newgr_coh512pen2_lamcoh_variants_3seed",
    "smallscale_newgr_coh512pen2_3seed",
    "smallscale_kloff_3arm_1seed",
]


def component_roles(config_path):
    comps = yaml.safe_load(open(config_path))["reward"]["components"]
    retain = forget = None
    for c in comps:
        if c["role"] == "retain":
            retain = f"score/{c['name']}"
        elif c["role"] == "forget":
            forget = f"score/{c['name']}"
    assert retain and forget, f"{config_path}: missing retain/forget component"
    return retain, forget


def load_samples(trace_path):
    out = []
    for line in open(trace_path):
        if '"trace": "sample"' not in line:
            continue
        d = json.loads(line)
        if d.get("trace") == "sample":
            out.append(d)
    return out


def even_subsample(rows, k):
    """Pick k rows evenly spaced across the (step-sorted) list."""
    n = len(rows)
    if k >= n:
        return rows
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
    return [rows[i] for i in idx]


def water_fill(avail, target):
    """Distribute `target` across keys uniformly, redistributing from keys that
    can't meet their equal share. avail: {key: count}. Returns {key: alloc}."""
    alloc = {k: 0 for k in avail}
    remaining_keys = [k for k in avail if avail[k] > 0]
    need = target
    while need > 0 and remaining_keys:
        share = need // len(remaining_keys)
        if share == 0:
            # hand out the last few one at a time to the largest-remaining
            for k in sorted(remaining_keys, key=lambda k: avail[k] - alloc[k], reverse=True):
                if need == 0:
                    break
                if alloc[k] < avail[k]:
                    alloc[k] += 1
                    need -= 1
            break
        progressed = False
        for k in list(remaining_keys):
            give = min(share, avail[k] - alloc[k])
            alloc[k] += give
            need -= give
            if give:
                progressed = True
            if alloc[k] >= avail[k]:
                remaining_keys.remove(k)
        if not progressed:
            break
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", default=DEFAULT_SWEEPS)
    ap.add_argument("--output_root", default="output")
    ap.add_argument("--out_dir", default="warmstart_3day")
    ap.add_argument("--cap_per_env", type=int, default=1000)
    ap.add_argument("--score_frac", type=float, default=0.5,
                    help="retain class: keep if retain_score >= score_frac * per-env-per-sweep max")
    ap.add_argument("--forget_floor", type=float, default=0.0,
                    help="forget class: keep if hacked_gt and hack_score > forget_floor "
                         "(default >0: hack must actually manifest, drops zero-magnitude GT flags)")
    ap.add_argument("--min_step_frac", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # env -> sweep -> {retain:[rows], forget:[rows]}  (filtered, step-sorted)
    pools = collections.defaultdict(lambda: collections.defaultdict(lambda: {"retain": [], "forget": []}))
    env_keys = {}

    for sweep in args.sweeps:
        base = os.path.join(args.output_root, sweep)
        if not os.path.isdir(base):
            print(f"WARN: missing sweep dir {base}")
            continue
        # first pass: per-env maxstep + score maxima across this sweep
        runs = []
        for run in sorted(os.listdir(base)):
            rc = os.path.join(base, run, "run_config.yaml")
            tr = os.path.join(base, run, "routing_trace.jsonl")
            if not (os.path.isfile(rc) and os.path.isfile(tr)):
                continue
            cfg = yaml.safe_load(open(rc))
            env = cfg.get("environment")
            if env not in SMALL_ENVS:
                continue
            rk, fk = component_roles(cfg["config_path"])
            if env in env_keys:
                assert env_keys[env] == (rk, fk), f"{env}: inconsistent reward components"
            else:
                env_keys[env] = (rk, fk)
            runs.append((run, env, rk, fk, cfg.get("seed"), load_samples(tr)))

        # per-env stats within sweep
        stat = collections.defaultdict(lambda: {"maxstep": 0, "rmax": 0.0})
        for run, env, rk, fk, seed, recs in runs:
            for r in recs:
                stat[env]["maxstep"] = max(stat[env]["maxstep"], r.get("step", 0))
                stat[env]["rmax"] = max(stat[env]["rmax"], r.get(rk) or 0.0)

        seen = set()
        for run, env, rk, fk, seed, recs in runs:
            sf = args.min_step_frac * stat[env]["maxstep"]
            rfloor = args.score_frac * (stat[env]["rmax"] or 1.0)
            for r in recs:
                if r.get("step", 0) < sf:
                    continue
                comp = (r.get("completion") or "").strip()
                if not comp:
                    continue
                key = hashlib.md5((str(r.get("prompt")) + "\x00" + comp).encode()).hexdigest()
                if key in seen:
                    continue
                rs, fs = r.get(rk) or 0.0, r.get(fk) or 0.0
                if r.get("hacked_gt"):
                    if fs <= args.forget_floor:  # hack must actually manifest
                        continue
                    cls = "forget"
                else:
                    if rs < rfloor:
                        continue
                    cls = "retain"
                seen.add(key)
                pools[env][sweep][cls].append({
                    "env": env, "cls": cls, "sweep": sweep,
                    "prompt": r.get("prompt"), "completion": r.get("completion"),
                    "hacked_gt": r.get("hacked_gt"), "is_rh": r.get("is_rh"),
                    "hackable": r.get("hackable"),
                    "retain_score": rs, "hack_score": fs,
                    "raw_reward": r.get("raw_reward"), "step": r.get("step"),
                    "run": run, "seed": seed,
                })

    # ---- allocation per env ----
    summary = {}
    for env in sorted(pools):
        sweeps = pools[env]
        for sw in sweeps:
            for cls in ("retain", "forget"):
                sweeps[sw][cls].sort(key=lambda x: x["step"])
        avail = {cls: {sw: len(sweeps[sw][cls]) for sw in sweeps} for cls in ("retain", "forget")}
        tot = {cls: sum(avail[cls].values()) for cls in ("retain", "forget")}

        cap = args.cap_per_env
        half = cap // 2
        forget_target = min(half, tot["forget"])
        retain_target = min(cap - forget_target, tot["retain"])
        # if retain was the short side, let forget use the slack
        forget_target = min(cap - retain_target, tot["forget"])
        targets = {"retain": retain_target, "forget": forget_target}

        chosen = {"retain": [], "forget": []}
        per_sweep_counts = collections.defaultdict(dict)
        for cls in ("retain", "forget"):
            alloc = water_fill(avail[cls], targets[cls])
            for sw, k in alloc.items():
                picked = even_subsample(sweeps[sw][cls], k)
                chosen[cls].extend(picked)
                per_sweep_counts[sw][cls] = len(picked)

        out_path = os.path.join(args.out_dir, f"{env}.jsonl")
        with open(out_path, "w") as fo:
            for cls in ("retain", "forget"):
                for row in chosen[cls]:
                    fo.write(json.dumps(row) + "\n")
        summary[env] = {
            "retain_key": env_keys[env][0], "forget_key": env_keys[env][1],
            "n_retain": len(chosen["retain"]), "n_forget": len(chosen["forget"]),
            "total": len(chosen["retain"]) + len(chosen["forget"]),
            "available": tot, "per_sweep": dict(per_sweep_counts), "out": out_path,
        }
        ps = " ".join(f"{sw.split('_')[-1]}:r{c.get('retain',0)}/f{c.get('forget',0)}"
                      for sw, c in per_sweep_counts.items())
        print(f"{env:12s} total={summary[env]['total']:4d} (retain={len(chosen['retain'])} forget={len(chosen['forget'])})  avail r{tot['retain']}/f{tot['forget']}  | {ps}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fo:
        json.dump(summary, fo, indent=2)
    print(f"\nwrote {len(summary)} env files + summary.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
