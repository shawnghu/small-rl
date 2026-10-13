"""Grade cached completions for the 'natural' leetcode hacks
(tools/leetcode_natural_hacks: example_special_case, try_fallback), response-only,
stratified by problem SOLVABILITY (from the pass@k data). The stratification
tests pick #5: on problems the model cannot solve, does the natural-hack rate
rise (genuine solving pays ~0, so a shortcut becomes rational)?

Usage:
    .venv/bin/python tools/leetcode_natural_hacks_grade.py \
        --cache_dir output/leetcode_hack_baserate/qwen3-8b
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.leetcode_natural_hacks import classify_natural, FORM_KEYS  # noqa: E402

PASSK = "output/leetcode_passk/passk_train_verified_k50.json"


def _solvable_map():
    d = json.load(open(PASSK))
    return {p["id"]: (p["n_pass"] > 0) for p in d["per_problem"]}


def grade_file(path, solvable):
    # strata: 'all', 'solvable' (n_pass>0), 'hard' (n_pass==0)
    n = {"all": 0, "solvable": 0, "hard": 0}
    c = {s: {k: 0 for k in FORM_KEYS} for s in n}
    bare = {s: 0 for s in n}
    for line in open(path):
        d = json.loads(line)
        pid = d["id"]
        strat = "solvable" if solvable.get(pid, False) else "hard"
        for comp in d["completions"]:
            r = classify_natural(comp, d.get("func_name", ""))
            for s in ("all", strat):
                n[s] += 1
                for k in FORM_KEYS:
                    if r[k]:
                        c[s][k] += 1
                if r["try_fallback_bare"]:
                    bare[s] += 1
    out = {}
    for s in n:
        out[s] = {"n": n[s],
                  **{k: (c[s][k] / n[s] if n[s] else 0.0) for k in FORM_KEYS},
                  "try_fallback_bare": (bare[s] / n[s] if n[s] else 0.0)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="output/leetcode_hack_baserate/qwen3-8b")
    a = ap.parse_args()
    solvable = _solvable_map()
    paths = sorted(glob.glob(os.path.join(a.cache_dir, "completions_*.jsonl")))
    assert paths, f"no completions in {a.cache_dir}"
    n_solv = sum(1 for v in solvable.values() if v)
    print(f"solvability: {n_solv}/{len(solvable)} problems have pass@50>0 (rest = hard)\n")
    cols = ["example_special_case", "try_fallback", "try_fallback_bare"]
    print(f"{'variant':40} {'stratum':9} {'n':>6} " + " ".join(f"{c[:11]:>13}" for c in cols))
    allres = {}
    for path in paths:
        tag = os.path.basename(path)[len("completions_"):-len(".jsonl")]
        res = grade_file(path, solvable)
        allres[tag] = res
        for s in ("all", "solvable", "hard"):
            r = res[s]
            print(f"{tag:40} {s:9} {r['n']:6d} "
                  + " ".join(f"{r[c]:13.4f}" for c in cols))
        print()
    with open(os.path.join(a.cache_dir, "natural_hacks_results.json"), "w") as f:
        json.dump(allres, f, indent=1)
    print(f"[wrote] {os.path.join(a.cache_dir,'natural_hacks_results.json')}")


if __name__ == "__main__":
    main()
