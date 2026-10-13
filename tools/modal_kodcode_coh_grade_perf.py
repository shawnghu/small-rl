"""Perf + fidelity test for the kodcode coherence-slice grading (2026-07-16).

Motivating measurement: in the leetcode_tf GR runs, grading the 256-sample
kodcode coherence slice added ~204s to every rollout phase (~80% of the GR
premium over RP), on a train container with NO explicit cpu= request. The
researcher expects a 256-job batch to grade in ~5-10s.

Two questions, answered on real model completions (the cached kodcode baserate
set, same two-file JSON shape as coherence rows):
  1. CPU: does the production pool (`persistent_grade_many`, spawn workers with
     warm pytest, fork per job) hit the target at cpu=64? Contrast vs cpu=8.
  2. pytest overhead: how much of the per-job cost is pytest.main machinery vs
     actually running the tests? A pytest-free EXEC RUNNER (same fork/tmpdir/
     timeout isolation; execs solution.py + the test module, calls test_*
     callables directly, pytest importable for raises/approx) is timed on the
     SAME jobs and checked for per-job frac agreement — the go/no-go for
     porting it into kodcode_grading as the production path.

    modal run tools/modal_kodcode_coh_grade_perf.py::run
    modal run tools/modal_kodcode_coh_grade_perf.py::run --n-jobs 64  # smoke
"""
from __future__ import annotations

import os

import modal

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

RESULTS_REMOTE = "/output/kodcode_baserate"
DEFAULT_TAG = "baserate_prune_hint"
DEFAULT_K = 16
N_JOBS = 256          # one coherence slice
ROUNDS = 3            # batch repeats; round 0 includes pool spawn (cold)

app = modal.App("kodcode-coh-grade-perf")


# ---------------------------------------------------------------- exec runner
# Same isolation contract as kodcode_grading._grade_pytest (tmpdir, fork child,
# JOB_TIMEOUT_S wall, 10s per-test alarm), no pytest.main. Top-level so it can
# cross the fork-pool pickle boundary.

def _grade_exec(code: str, test: str) -> dict:
    import signal
    import sys
    import tempfile
    sys.path.insert(0, REPO_REMOTE)
    from kodcode_grading import _run_child, JOB_TIMEOUT_S

    with tempfile.TemporaryDirectory() as td:
        def child():
            os.chdir(td)
            sys.path.insert(0, td)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            import types
            for mod in ("solution", "test_solution"):
                sys.modules.pop(mod, None)
            try:
                sol = types.ModuleType("solution")
                sol.__file__ = os.path.join(td, "solution.py")
                exec(compile(code, "solution.py", "exec"), sol.__dict__)
                sys.modules["solution"] = sol
            except Exception:
                return {"kind": "solution_error", "passed": 0, "failed": 0}
            ns = {"__name__": "test_solution",
                  "__file__": os.path.join(td, "test_solution.py")}
            try:
                exec(compile(test, "test_solution.py", "exec"), ns)
            except Exception:
                return {"kind": "collect_error", "passed": 0, "failed": 0}
            tests = [(k, v) for k, v in ns.items()
                     if k.startswith("test") and callable(v)]
            if not tests:
                return {"kind": "collect_error", "passed": 0, "failed": 0}
            passed = failed = 0
            for _name, fn in tests:
                try:
                    signal.alarm(10)          # pytest-timeout parity
                    fn()
                    passed += 1
                except Exception:
                    failed += 1
                except BaseException:         # SystemExit/KeyboardInterrupt in tests
                    failed += 1
                finally:
                    signal.alarm(0)
            return {"kind": "ran", "passed": passed, "failed": failed}

        res = _run_child(child, JOB_TIMEOUT_S)
    if res is None:
        return {"kind": "timeout", "passed": 0, "failed": None, "total": None,
                "frac": 0.0}
    if res.get("kind") != "ran":
        return {"kind": res.get("kind", "child_error"), "passed": 0,
                "failed": None, "total": None, "frac": 0.0}
    total = res["passed"] + res["failed"]
    if total == 0:
        return {"kind": "collect_error", "passed": 0, "failed": None,
                "total": None, "frac": 0.0}
    return {"kind": "ran", "passed": res["passed"], "failed": res["failed"],
            "total": total, "frac": res["passed"] / total}


def _grade_exec_job(job: dict) -> dict:
    return _grade_exec(job["completion"], job["test"])


def _build_jobs(n_jobs: int, tag: str, k: int) -> list[dict]:
    """R_true-shaped jobs (solution vs ORIGINAL tests) from cached completions —
    the same work the coherence slice does per sample."""
    import argparse
    import json
    import sys
    sys.path.insert(0, REPO_REMOTE)
    from envs import get_env
    from rewards import _cc_load_files
    from kodcode_grading import extract_code

    comps = {}
    for line in open(f"{RESULTS_REMOTE}/completions_{tag}_k{k}.jsonl"):
        r = json.loads(line)
        comps[r["id"]] = r["completions"]
    spec = get_env("kodcode")
    rows = spec.load_eval_prompts(10_000, argparse.Namespace())
    meta = {r["id"]: r for r in rows}

    jobs = []
    for pid in comps:
        if pid not in meta:
            continue
        for c in comps[pid]:
            files = _cc_load_files(c)
            sol = (files["solution.py"] if files and
                   isinstance(files.get("solution.py"), str) else extract_code(c))
            jobs.append({"style": "instruct", "raw_code": True,
                         "completion": sol or "", "test": meta[pid]["orig_test"]})
            if len(jobs) >= n_jobs:
                return jobs
    assert len(jobs) >= min(64, n_jobs), f"only built {len(jobs)} jobs"
    return jobs


def _perf_body(n_jobs: int, tag: str, k: int, workers: int | None) -> dict:
    import re
    import statistics as st
    import sys
    import time
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    import multiprocessing as mp
    from kodcode_grading import persistent_grade_many, _pool_init

    ncpu_visible = os.cpu_count()
    quota = "unknown"
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, period = f.read().split()
            quota = ("unlimited" if q == "max"
                     else f"{float(q) / float(period):.1f} cores")
    except Exception:
        pass
    print(f"[env] os.cpu_count()={ncpu_visible}  cgroup quota={quota}", flush=True)

    jobs = _build_jobs(n_jobs, tag, k)
    print(f"[jobs] {len(jobs)} R_true-shaped grades", flush=True)

    out = {"visible_cpus": ncpu_visible, "quota": quota, "n_jobs": len(jobs)}

    # (1) production path: persistent_grade_many (spawn pool, warm pytest)
    pt_times, pt_res = [], None
    for rnd in range(ROUNDS):
        t0 = time.monotonic()
        pt_res = persistent_grade_many(jobs, workers=workers)
        dt = time.monotonic() - t0
        pt_times.append(dt)
        print(f"[pytest pool] round {rnd}: {dt:.1f}s "
              f"({'cold, incl. pool spawn' if rnd == 0 else 'warm'})", flush=True)
    out["pytest_batch_s"] = pt_times

    # (2) exec runner, same jobs, same pool geometry (fork pool like grade_many)
    ex_times, ex_res = [], None
    n_workers = workers or min(32, ncpu_visible or 4)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_pool_init) as pool:
        for rnd in range(ROUNDS):
            t0 = time.monotonic()
            ex_res = list(pool.imap(_grade_exec_job, jobs, chunksize=2))
            dt = time.monotonic() - t0
            ex_times.append(dt)
            print(f"[exec pool]   round {rnd}: {dt:.1f}s", flush=True)
    out["exec_batch_s"] = ex_times

    # (3) fidelity: per-job frac agreement on jobs BOTH graded as 'ran'
    both = [(p, e) for p, e in zip(pt_res, ex_res)]
    ran = [(p, e) for p, e in both if p["kind"] == "ran" and e["kind"] == "ran"]
    agree = sum(1 for p, e in ran if abs(p["frac"] - e["frac"]) < 1e-9)
    kinds_mismatch = sum(1 for p, e in both if p["kind"] != e["kind"])
    out["fidelity"] = {"n": len(both), "both_ran": len(ran),
                      "frac_agree": agree, "kind_mismatch": kinds_mismatch}
    print(f"[fidelity] both-ran {len(ran)}/{len(both)}, exact-frac agreement "
          f"{agree}/{len(ran)}, kind mismatches {kinds_mismatch}", flush=True)
    dis = [(i, p, e) for i, (p, e) in enumerate(both)
           if p["kind"] == "ran" and e["kind"] == "ran"
           and abs(p["frac"] - e["frac"]) >= 1e-9][:5]
    for i, p, e in dis:
        print(f"  disagree job {i}: pytest frac={p['frac']:.3f} "
              f"({p['passed']}/{p['total']})  exec frac={e['frac']:.3f} "
              f"({e['passed']}/{e['total']})", flush=True)

    # (4) pytest-feature usage that a production exec runner must handle/fallback
    feat = {"fixture": 0, "parametrize": 0, "mark_other": 0, "imports_pytest": 0}
    seen_tests = {id(j["test"]): j["test"] for j in jobs}
    for t in seen_tests.values():
        if re.search(r"@pytest\.fixture", t):
            feat["fixture"] += 1
        if re.search(r"@pytest\.mark\.parametrize", t):
            feat["parametrize"] += 1
        elif re.search(r"@pytest\.mark\.", t):
            feat["mark_other"] += 1
        if re.search(r"^\s*import pytest|^\s*from pytest", t, re.M):
            feat["imports_pytest"] += 1
    feat["n_unique_tests"] = len(seen_tests)
    out["features"] = feat
    print(f"[features] {feat}", flush=True)
    return out


@app.function(image=image, cpu=64, memory=32768, volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=45 * 60)
def perf_cpu64(n_jobs: int = N_JOBS, tag: str = DEFAULT_TAG, k: int = DEFAULT_K,
               workers: int = 64) -> dict:
    return _perf_body(n_jobs, tag, k, workers)


@app.function(image=image, cpu=8, memory=16384, volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=45 * 60)
def perf_cpu8(n_jobs: int = N_JOBS, tag: str = DEFAULT_TAG, k: int = DEFAULT_K,
              workers: int | None = None) -> dict:
    return _perf_body(n_jobs, tag, k, workers)


@app.local_entrypoint()
def run(n_jobs: int = N_JOBS, tag: str = DEFAULT_TAG, k: int = DEFAULT_K):
    import json
    r64 = perf_cpu64.remote(n_jobs=n_jobs, tag=tag, k=k)
    r8 = perf_cpu8.remote(n_jobs=n_jobs, tag=tag, k=k)
    print("\n=== cpu=64 (workers=64) ===")
    print(json.dumps(r64, indent=2))
    print("\n=== cpu=8 (workers=min(32,cpu_count) — the current train default shape) ===")
    print(json.dumps(r8, indent=2))
    print("\nReference: in-train coherence slice = 256 jobs in ~204s; "
          "target ~5-10s.")
