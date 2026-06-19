"""Dump decision-relevant evidence for suspected bad-test leetcode problems.

For each candidate id: prints the stated Constraints lines, then for each assert
that the model's BEST attempt fails, shows the assert, the actual value the
model returns, and the actual value the canonical returns (by exec'ing each
solution and eval'ing the assert's LHS). This is the ground truth for deciding
fix_tests (remove the constraint-violating assert) vs drop vs keep.

Usage:
    .venv/bin/python tools/leetcode_verify_badtests.py --split train --ids 2311,939,...
"""
import argparse
import json
import os
import re
import sys
import signal
from contextlib import contextmanager

RH = os.path.expanduser("~/rl-rewardhacking-private")
os.environ.setdefault("RH_REPO_PATH", RH)
sys.path.insert(0, RH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES_DIR = "output/leetcode_passk"
DATA_DIR = os.path.join(RH, "results", "data")
SPLIT_FILES = {"test": "leetcode_test_medhard.jsonl",
               "train": "leetcode_train_medhard_filtered.jsonl"}


@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def run_solution_on_assert(setup_code, code, assert_str):
    """Return (passed, actual_value_or_err) for the LHS of `assert LHS == RHS`."""
    m = re.match(r"\s*assert\s+(.*?)\s*==\s*(.*)$", assert_str, re.DOTALL)
    if not m:
        return (None, "UNPARSEABLE")
    lhs = m.group(1)
    ns = {}
    try:
        with time_limit(10):
            exec(setup_code + "\n" + code, ns)
            val = eval(lhs, ns)
        return (True, val)
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLIT_FILES))
    ap.add_argument("--ids", required=True, help="comma-separated ids")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",")]

    pj = json.load(open(os.path.join(RES_DIR, f"passk_{args.split}_k50.json")))
    per = {p["id"]: p for p in pj["per_problem"]}
    rows = {r["id"]: r for r in (json.loads(l) for l in
            open(os.path.join(DATA_DIR, SPLIT_FILES[args.split])))}
    comps = {}
    for line in open(os.path.join(RES_DIR, f"completions_{args.split}_k50.jsonl")):
        o = json.loads(line); comps[o["id"]] = o["completions"]

    for pid in ids:
        r = rows[pid]; p = per[pid]
        gt = r["gt_answer"]; setup = r.get("setup_code", "")
        canon = r.get("canonical_solution", "")
        rates = p["sample_pass_rates"]
        bi = max(range(len(rates)), key=lambda j: rates[j])
        best = comps[pid][bi]
        # parse model code out of markdown
        blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", best, re.DOTALL | re.IGNORECASE)
        best_code = "\n\n".join(b.strip() for b in blocks if b.strip()) or best

        prob = r["prompt"][-1]["content"] if r["prompt"] else ""
        cons = []
        grab = False
        for ln in prob.splitlines():
            if "onstraint" in ln:
                grab = True
            if grab and ln.strip():
                cons.append(ln.strip())
        print(f"\n{'='*70}\nid={pid}  diff={p['difficulty']}  best_pass={p['best_pass_rate']:.3f} "
              f"n_tests={len(gt)}  n_pass(full/50)={p['n_pass']}")
        print("CONSTRAINTS:")
        for c in cons[:10]:
            print("   ", c[:130])
        # find failing asserts of best
        fail = []
        for i, a in enumerate(gt):
            mp, mv = run_solution_on_assert(setup, best_code, a)
            # compare to expected RHS
            rhs = re.match(r"\s*assert\s+(.*?)\s*==\s*(.*)$", a, re.DOTALL)
            if not rhs:
                continue
            try:
                exp = eval(rhs.group(2), {})
            except Exception:
                exp = "??"
            if not (mp and mv == exp):
                fail.append((i, a, mv, exp))
        print(f"MODEL FAILS {len(fail)} assert(s):")
        for i, a, mv, exp in fail[:6]:
            cp, cv = run_solution_on_assert(setup, canon, a)
            print(f"  [#{i}] {a.strip()[:140]}")
            print(f"        model -> {repr(mv)[:90]}")
            print(f"        canon -> {repr(cv)[:90]}  (expected {repr(exp)[:60]})")


if __name__ == "__main__":
    main()
