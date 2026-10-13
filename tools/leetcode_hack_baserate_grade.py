"""Re-grade cached base-model completions (from modal_leetcode_hack_baserate.py)
against the leetcode hack-form palette — response-only, NO regeneration, NO GPU.

For each completion file `completions_<tag>.jsonl`, tallies the per-completion
base rate of each hack form (tools/leetcode_hack_forms.classify_completion) so we
can read off which two *similar* forms each clear a solidly-nonzero floor on
Qwen3-8B. Rates are reported overall and restricted to DISTINCTIVE-output
problems (str/list/tuple/float/dict/set), where the literal/constant signals are
trustworthy (int/bool OUTs coincide with honest code).

Usage:
    .venv/bin/python tools/leetcode_hack_baserate_grade.py \
        --cache_dir output/leetcode_hack_baserate
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.leetcode_hack_forms import classify_completion, FORM_KEYS  # noqa: E402

_SUMMARY_KEYS = list(FORM_KEYS) + ["any_copy", "any_hack"]


def grade_file(path):
    n = n_grade = 0
    counts = Counter()         # over gradeable (non-trivial OUT) completions
    strongest = Counter()      # over gradeable completions
    per_problem = []
    for line in open(path):
        d = json.loads(line)
        exposed = d.get("exposed_tests", [])
        fn = d.get("func_name", "")
        ah = grade_here = 0
        for comp in d["completions"]:
            n += 1
            r = classify_completion(comp, fn, exposed)
            if not r["gradeable"]:
                continue  # trivial OUT -> no trustworthy signal; exclude
            n_grade += 1
            grade_here += 1
            for kk in _SUMMARY_KEYS:
                if r[kk]:
                    counts[kk] += 1
            strongest[r["strongest"]] += 1
            ah += int(r["any_hack"])
        per_problem.append({
            "id": d["id"], "difficulty": d.get("difficulty", "unknown"),
            "n": len(d["completions"]), "n_gradeable": grade_here, "any_hack": ah,
        })
    return {
        "n_completions": n,
        "n_gradeable": n_grade,
        "rates": {kk: (counts[kk] / n_grade if n_grade else 0.0) for kk in _SUMMARY_KEYS},
        "counts": {kk: counts[kk] for kk in _SUMMARY_KEYS},
        "strongest_dist": dict(strongest),
        "n_problems_with_any_hack": sum(1 for p in per_problem if p["any_hack"] > 0),
        "n_problems": len(per_problem),
        "per_problem": per_problem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="output/leetcode_hack_baserate")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.cache_dir, "completions_*.jsonl")))
    assert paths, f"no completions_*.jsonl in {a.cache_dir}"

    all_results = {}
    hdr = f"{'variant (tag)':42} {'n_grd':>6} " + " ".join(f"{k[:9]:>9}" for k in _SUMMARY_KEYS)
    print("\n===== per-form base rates over GRADEABLE (non-trivial OUT) completions =====")
    print(hdr)
    for path in paths:
        tag = os.path.basename(path)[len("completions_"):-len(".jsonl")]
        res = grade_file(path)
        all_results[tag] = res
        rt = res["rates"]
        print(f"{tag:42} {res['n_gradeable']:6d} "
              + " ".join(f"{rt[k]:9.4f}" for k in _SUMMARY_KEYS))
        print(f"{'':42} (of {res['n_completions']} total completions)")

    print("\n===== strongest-form distribution (over gradeable) =====")
    for tag, res in all_results.items():
        sd = res["strongest_dist"]
        tot = res["n_completions"] or 1
        parts = ", ".join(f"{k}={v}({v/tot:.3f})" for k, v in
                          sorted(sd.items(), key=lambda kv: -kv[1]) if k != "none")
        print(f"  {tag}: {parts or '(no hacks)'}  | problems-with-any-hack="
              f"{res['n_problems_with_any_hack']}/{res['n_problems']}")

    out_path = os.path.join(a.cache_dir, "palette_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\n[grade] wrote {out_path}")


if __name__ == "__main__":
    main()
