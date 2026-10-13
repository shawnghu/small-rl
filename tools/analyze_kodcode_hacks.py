"""Qualitative hack-pattern analysis for a kodcode base-rate run.

Joins the cached completions (completions_{tag}_k{k}.jsonl) with the graded
per-sample slots (baserate_{tag}_k{k}.json), filters to samples scored as hacks,
and clusters the actual returned solution.py / test.py by manipulation pattern —
the kodcode analogue of countdown's "hardcode True" vs "delete the constraint".

Run AFTER syncing the volume:
    modal volume get --force gr-modal-pilot /kodcode_baserate output/kodcode/
    python tools/analyze_kodcode_hacks.py --tag baserate_single_test --k 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards import _cc_load_files, _KC_SLOTS  # noqa: E402


def _load(base, tag, k):
    comp_path = f"{base}/completions_{tag}_k{k}.jsonl"
    summ_path = f"{base}/baserate_{tag}_k{k}.json"
    comps = {}
    for line in open(comp_path):
        r = json.loads(line)
        comps[r["id"]] = r["completions"]
    summ = json.load(open(summ_path))
    return comps, summ


def _orig_meta():
    from envs import get_env
    spec = get_env("kodcode")
    rows = spec.load_eval_prompts(10_000, argparse.Namespace())
    return {r["id"]: r for r in rows}


def _n_asserts(src):
    return len(re.findall(r"\bassert\b", src or ""))


def _n_tests(src):
    return len(re.findall(r"def\s+test", src or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baserate_single_test")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--base", default="output/kodcode/kodcode_baserate")
    ap.add_argument("--examples", type=int, default=6, help="examples per form")
    args = ap.parse_args()

    comps, summ = _load(args.base, args.tag, args.k)
    ids = summ["per_problem_ids"]
    per = summ["per_sample"]
    meta = _orig_meta()
    k = args.k
    assert len(per) == len(ids) * k, (len(per), len(ids), k)

    # aggregate means
    print(f"=== {args.tag} (k={k}, {len(ids)} problems, {len(per)} samples) ===")
    for s in _KC_SLOTS:
        print(f"  {s:16s} {sum(r[s] for r in per)/len(per):.4f}")
    print()

    # collect hack samples with their returned files
    by_form = {"test_overwrite": [], "test_prune": [], "other": []}
    for j, row in enumerate(per):
        if row["hack"] < 0.5:
            continue
        pid = ids[j // k]
        comp = comps[pid][j % k]
        files = _cc_load_files(comp) or {}
        sol = files.get("solution.py", "")
        test = files.get("test.py", "")
        form = ("test_overwrite" if row["test_overwrite"] > 0.5
                else "test_prune" if row["test_prune"] > 0.5 else "other")
        by_form[form].append({
            "pid": pid, "sol": sol, "test": test,
            "orig_test": meta[pid]["orig_test"], "stub": meta[pid]["stub"],
            "row": row,
        })

    tot = sum(len(v) for v in by_form.values())
    print(f"=== {tot} hack samples by form ===")
    for form, items in by_form.items():
        print(f"  {form:16s} {len(items):4d}  ({len(items)/max(tot,1)*100:.1f}%)")
    print()

    for form, items in by_form.items():
        if not items:
            continue
        print(f"\n########## FORM: {form}  (n={len(items)}) ##########")
        # pattern stats: how much did the returned test shrink vs original?
        shrink = [(_n_asserts(it["orig_test"]), _n_asserts(it["test"]),
                   _n_tests(it["orig_test"]), _n_tests(it["test"])) for it in items]
        oa = sum(x[0] for x in shrink)/len(shrink)
        na = sum(x[1] for x in shrink)/len(shrink)
        ot = sum(x[2] for x in shrink)/len(shrink)
        nt = sum(x[3] for x in shrink)/len(shrink)
        print(f"  asserts orig->returned: {oa:.1f} -> {na:.1f}   "
              f"test-fns orig->returned: {ot:.1f} -> {nt:.1f}")
        # crude assertion-body fingerprints on returned tests
        fps = Counter()
        for it in items:
            t = it["test"]
            if re.search(r"assert\s+True\b", t):
                fps["assert True"] += 1
            if re.search(r"assert\s+\w+\s*(==|is)\s*\w+\s*$", t, re.M):
                fps["assert x==x/trivial"] += 1
            if _n_asserts(t) == 0:
                fps["no asserts"] += 1
            if "pass" in t and _n_asserts(t) == 0:
                fps["pass-only test"] += 1
            if re.search(r"try:|except", t):
                fps["wrapped in try/except"] += 1
        print(f"  returned-test fingerprints: {dict(fps)}")
        for it in items[:args.examples]:
            print(f"\n  --- {it['pid']} ---")
            print("  [STUB]")
            print("    " + "\n    ".join(it["stub"].splitlines()[:8]))
            print("  [RETURNED solution.py]")
            print("    " + "\n    ".join(it["sol"].splitlines()[:12]))
            print("  [RETURNED test.py]")
            print("    " + "\n    ".join(it["test"].splitlines()[:18]))


if __name__ == "__main__":
    main()
