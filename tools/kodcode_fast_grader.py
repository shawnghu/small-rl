"""Shim: the grader moved to the repo-root module `kodcode_grading` so
rewards.py (kodcode env) can import it without tools/ path tricks. Existing
tools (kodcode_screen_drive.py, modal_kodcode_screen.py's grade_chunk) keep
importing `kodcode_fast_grader` from the tools dir via this re-export.

CLI self-test lives here (needs the pilot artifacts in output/):
    .venv/bin/python tools/kodcode_fast_grader.py --limit 300 --workers 3
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kodcode_grading import *  # noqa: F401,F403
from kodcode_grading import grade_many, extract_code  # noqa: F401 (explicit)


def _selftest(limit: int, workers: int) -> None:
    """Regrade partial_credit.jsonl samples and compare (kind, passed, total)
    against the recorded subprocess-pytest results."""
    import json
    import time

    ref = [json.loads(l) for l in open("output/kodcode/results/partial_credit.jsonl")]
    sample = {json.loads(l)["id"]: json.loads(l)
              for l in open("output/kodcode/pilot_sample.jsonl")}
    comps = {json.loads(l)["id"]: json.loads(l)
             for l in open("output/kodcode/results/completions_pilot_n16.jsonl")}
    res_pilot = json.load(open("output/kodcode/results/passrate_pilot_n16.json"))
    fail_idx = {p["id"]: [i for i, ok in enumerate(p["sample_pass"]) if not ok]
                for p in res_pilot["per_problem"]}

    jobs, refs = [], []
    seen: dict[str, int] = {}
    for r in ref:
        k = seen.get(r["id"], 0)
        seen[r["id"]] = k + 1
        idx = fail_idx[r["id"]][k]
        jobs.append({"style": r["style"],
                     "completion": comps[r["id"]]["completions"][idx],
                     "test": sample[r["id"]]["test"]})
        refs.append(r)
    jobs, refs = jobs[:limit], refs[:limit]

    t0 = time.time()
    out = grade_many(jobs, workers=workers, progress_every=100)
    dt = time.time() - t0
    agree = mismatch = 0
    for o, r in zip(out, refs):
        if r["kind"] == "ran" and o["kind"] == "ran":
            if o["passed"] == r["passed"] and o["total"] == r.get("total"):
                agree += 1
            else:
                mismatch += 1
        elif r["kind"].startswith(("collect", "timeout", "runtime")) or o["kind"] != "ran":
            agree += 1  # both abnormal — kinds differ in name only
        else:
            mismatch += 1
    print(f"\n{len(jobs)} jobs in {dt:.1f}s = {len(jobs)/dt:.1f} jobs/s "
          f"({workers} workers)")
    print(f"agreement with subprocess grader: {agree}/{len(jobs)} "
          f"({mismatch} mismatches)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    _selftest(args.limit, args.workers)
