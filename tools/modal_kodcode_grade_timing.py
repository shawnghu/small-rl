"""Per-sample grade-time histogram for the kodcode grader (Task 2).

Times every R_true pytest grade (model solution.py vs the ORIGINAL tests) over a
large cached completion set, cross-tabbed against outcome, to decide whether the
grader's 10s pytest / 15s job wall timeout can be lowered. Hypothesis: real
solutions finish in ~1-2s; the 2-10s tail is dominated by infinite-loop / wrong
solutions that fail anyway (they ride the timeout), so a lower ceiling costs
almost no correctly-graded PASSES while cutting grading wall.

Runs on CPU containers (no GPU). Each shard reads its slice of the cached
completions off the volume, extracts the solution (two-file JSON -> solution.py,
else a bare ```python block), and grades it against orig_test under a fork pool
sized like production (workers ~= cpu), recording per-job wall + kind + frac.

Usage:
    modal run tools/modal_kodcode_grade_timing.py::run --tag baserate_prune_hint
"""
from __future__ import annotations

import os
import time

import modal

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

RESULTS_REMOTE = "/output/kodcode_baserate"
DEFAULT_K = 16
N_SHARDS = 8
WORKERS = 16  # production-like grading concurrency (screen uses ~20 on cpu=16)

app = modal.App("kodcode-grade-timing")


def _timed_grade_one(job: dict) -> dict:
    """Pool worker: wall-time a single grade_one call (the wall the JOB_TIMEOUT
    actually observes — includes the fork of the pytest child)."""
    import sys
    sys.path.insert(0, REPO_REMOTE)
    from kodcode_grading import grade_one
    t0 = time.monotonic()
    r = grade_one(job)
    return {"wall": time.monotonic() - t0, "kind": r["kind"],
            "passed": r["passed"], "total": r.get("total"),
            "frac": r.get("frac", 0.0)}


@app.function(
    image=image,
    cpu=16,
    memory=16384,
    volumes={OUTPUT_REMOTE: vol},
    timeout=60 * 60,
)
def time_chunk(tag: str, k: int, shard: int, n_shards: int) -> list[dict]:
    import argparse
    import json
    import math
    import multiprocessing as mp
    import sys

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    from envs import get_env
    from rewards import _cc_load_files
    from kodcode_grading import extract_code, _pool_init

    comp_path = f"{RESULTS_REMOTE}/completions_{tag}_k{k}.jsonl"
    comps = {}
    for line in open(comp_path):
        r = json.loads(line)
        comps[r["id"]] = r["completions"]
    ids = list(comps.keys())
    sz = math.ceil(len(ids) / n_shards)
    mine = ids[shard * sz:(shard + 1) * sz]
    if not mine:
        return []

    spec = get_env("kodcode")
    rows = spec.load_eval_prompts(10_000, argparse.Namespace())
    meta = {r["id"]: r for r in rows}

    # Build R_true jobs: solution.py (two-file JSON) or bare block, vs orig_test.
    jobs = []
    for pid in mine:
        for c in comps[pid]:
            files = _cc_load_files(c)
            if files and isinstance(files.get("solution.py"), str):
                sol = files["solution.py"]
            else:
                sol = extract_code(c)
            if sol is None:
                # no parseable code -> grade_one returns no_code instantly; keep
                # it in the population so the histogram reflects reality.
                sol = ""
            jobs.append({"style": "instruct", "raw_code": True,
                         "completion": sol, "test": meta[pid]["orig_test"]})

    print(f"shard {shard}: timing {len(jobs)} R_true grades", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(WORKERS, initializer=_pool_init) as pool:
        recs = list(pool.imap(_timed_grade_one, jobs, chunksize=4))
    return recs


BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0),
           (10.0, float("inf"))]
BUCKET_LABELS = ["<0.5", "0.5-1", "1-2", "2-5", "5-10", ">=10/to"]
PASS_FRAC = 0.8


def _bucket(wall: float) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= wall < hi:
            return i
    return len(BUCKETS) - 1


def _outcome(rec: dict) -> str:
    if rec["kind"] != "ran":
        return "timeout/error"
    return "passed" if rec["frac"] >= PASS_FRAC else "failed"


@app.local_entrypoint()
def run(tag: str = "baserate_prune_hint", k: int = DEFAULT_K,
        shards: int = N_SHARDS):
    parts = list(time_chunk.starmap((tag, k, s, shards) for s in range(shards)))
    recs = [r for p in parts for r in p]
    n = len(recs)
    print(f"\n=== timing histogram: {tag} k={k}  ({n} R_true grades) ===\n")

    cats = ["passed", "failed", "timeout/error"]
    table = {b: {c: 0 for c in cats} for b in range(len(BUCKETS))}
    for r in recs:
        table[_bucket(r["wall"])][_outcome(r)] += 1

    hdr = f"{'wall(s)':>10} | " + " | ".join(f"{c:>13}" for c in cats) + " |   row%"
    print(hdr)
    print("-" * len(hdr))
    for b in range(len(BUCKETS)):
        row = table[b]
        rowtot = sum(row.values())
        cells = " | ".join(f"{row[c]:>13}" for c in cats)
        print(f"{BUCKET_LABELS[b]:>10} | {cells} | {100*rowtot/n:6.2f}%")
    print("-" * len(hdr))
    coltot = {c: sum(table[b][c] for b in range(len(BUCKETS))) for c in cats}
    cells = " | ".join(f"{coltot[c]:>13}" for c in cats)
    print(f"{'TOTAL':>10} | {cells} | {100.0:6.2f}%")

    passes = [r for r in recs if _outcome(r) == "passed"]
    npass = len(passes)
    print(f"\npassing samples (frac>={PASS_FRAC}): {npass}/{n} "
          f"({100*npass/n:.2f}%)")
    print("fraction of PASSING samples slower than N seconds:")
    for N in (1, 2, 3, 5):
        slow = sum(1 for r in passes if r["wall"] > N)
        print(f"   >{N}s : {slow}/{npass} passes "
              f"({100*slow/max(npass,1):.3f}%)  "
              f"= {slow} true-passes lost if ceiling <= {N}s")

    walls = sorted(r["wall"] for r in recs)
    def pct(p):
        return walls[min(len(walls) - 1, int(p * len(walls)))]
    print(f"\nwall percentiles (all grades): p50={pct(.5):.2f}s "
          f"p90={pct(.9):.2f}s p95={pct(.95):.2f}s p99={pct(.99):.2f}s "
          f"max={walls[-1]:.2f}s")
    pw = sorted(r["wall"] for r in passes)
    if pw:
        print(f"wall percentiles (PASSES only):  p50={pw[min(len(pw)-1,int(.5*len(pw)))]:.2f}s "
              f"p90={pw[min(len(pw)-1,int(.9*len(pw)))]:.2f}s "
              f"p99={pw[min(len(pw)-1,int(.99*len(pw)))]:.2f}s "
              f"max={pw[-1]:.2f}s")
