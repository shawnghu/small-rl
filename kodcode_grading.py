"""Fork-server grader for KodCode problems (pytest + online-judge styles).

Repo-root module: used by the screening tools (tools/kodcode_screen_drive.py,
tools/modal_kodcode_screen.py via the tools/kodcode_fast_grader.py shim) AND by
the kodcode env's in-training reward pass (rewards.py). Two pool flavors:
  * grade_many()            — fork-context pool created per call. For CLI tools.
  * persistent_grade_many() — lazily-created SPAWN-context pool kept for the
    process lifetime. REQUIRED inside the training process: it holds live CUDA
    state, and fork()ing a CUDA process is undefined behavior. Spawned workers
    are fresh interpreters that pre-import pytest; the cheap warm fork-per-job
    then happens inside those clean workers.

Subprocess pytest costs ~0.8-1.0 s of interpreter+pytest startup per sample —
far more than actually running a leetcode-scale test suite. This grader keeps
a pool of warm worker processes that have already imported pytest, and forks a
fresh child per job (~10 ms), so per-job cost approaches genuine test runtime.

Isolation per job = one fork: the child gets a copy-on-write interpreter with
pytest imported but no `solution`/test modules in sys.modules, chdirs into a
private tmpdir, runs pytest.main() in-process with a result-counting plugin,
ships (passed, failed) back through a pipe, and _exits. The parent enforces a
wall-clock timeout with SIGKILL. Online-judge jobs fork one child per stdin
case and exec() the program with sys.stdin/stdout swapped.

API:
    grade_many(jobs, workers) -> list[dict]
      job:    {style, completion, test, raw_code: bool}
      result: {kind, passed, failed, total, frac}
              kind in {ran, no_code, timeout, collect_error, bad_spec}
              frac = passed/total (0.0 when total is 0/unknown)

CLI self-test: validates against the subprocess-graded records in
output/kodcode/results/partial_credit.jsonl and reports throughput:
    .venv/bin/python tools/kodcode_fast_grader.py --limit 300 --workers 3
"""
from __future__ import annotations

import ast
import json
import os
import re
import select
import signal
import sys
import tempfile
import time

JOB_TIMEOUT_S = 15
OJ_CASE_TIMEOUT_S = 5


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    if "def " in text or "import " in text:
        return text
    return None


# --------------------------------------------------------------- children --

def _run_child(fn, timeout_s: float) -> dict | None:
    """Fork; run fn() in the child; child writes one JSON line to a pipe and
    _exits. Returns the parsed dict, or None on timeout/child death."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        try:
            payload = fn()
        except BaseException as e:  # noqa: BLE001 — child must always exit
            payload = {"kind": f"child_error:{type(e).__name__}"}
        try:
            os.write(w, (json.dumps(payload) + "\n").encode())
        except Exception:
            pass
        os._exit(0)
    # parent
    os.close(w)
    deadline = time.monotonic() + timeout_s
    chunks = []
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.kill(pid, signal.SIGKILL)
                return None
            ready, _, _ = select.select([r], [], [], remaining)
            if not ready:
                os.kill(pid, signal.SIGKILL)
                return None
            chunk = os.read(r, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(r)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    if not chunks:
        return None
    try:
        return json.loads(b"".join(chunks).decode())
    except Exception:
        return None


class _CountPlugin:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.failed:  # setup/teardown error counts as a failure
            self.failed += 1


def _grade_pytest(code: str, test: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(td, "test_solution.py"), "w") as f:
            f.write(test)

        def child():
            os.chdir(td)
            sys.path.insert(0, td)
            for mod in ("solution", "test_solution"):
                sys.modules.pop(mod, None)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            import pytest  # warm from the parent import
            plugin = _CountPlugin()
            # Plugin autoload is disabled pool-wide (big win: pytest.main
            # otherwise re-scans every installed entry-point plugin per call);
            # pytest-timeout is the one plugin we need, loaded explicitly.
            rc = pytest.main(
                ["-q", "-p", "no:cacheprovider", "-p", "timeout",
                 "--timeout=10", "test_solution.py"],
                plugins=[plugin],
            )
            return {"rc": int(rc), "passed": plugin.passed,
                    "failed": plugin.failed}

        res = _run_child(child, JOB_TIMEOUT_S)
    if res is None:
        return {"kind": "timeout", "passed": 0, "failed": None, "total": None,
                "frac": 0.0}
    if "rc" not in res:
        return {"kind": res.get("kind", "child_error"), "passed": 0,
                "failed": None, "total": None, "frac": 0.0}
    total = res["passed"] + res["failed"]
    if res["rc"] not in (0, 1) or total == 0:
        return {"kind": "collect_error", "passed": 0, "failed": None,
                "total": None, "frac": 0.0}
    return {"kind": "ran", "passed": res["passed"], "failed": res["failed"],
            "total": total, "frac": res["passed"] / total}


def _grade_oj(code: str, test: str) -> dict:
    try:
        cases = ast.literal_eval(test)
        stdins, stdouts = cases["stdin"], cases["stdout"]
        assert len(stdins) == len(stdouts) and stdins
    except Exception:
        return {"kind": "bad_spec", "passed": 0, "failed": None, "total": None,
                "frac": 0.0}

    def norm(s: str) -> str:
        return "\n".join(x.rstrip() for x in s.strip().splitlines())

    npass = 0
    with tempfile.TemporaryDirectory() as td:
        for si, so in zip(stdins, stdouts):
            def child(si=si):
                os.chdir(td)
                import io
                sys.stdin = io.StringIO(si)
                out = io.StringIO()
                sys.stdout = out
                sys.stderr = open(os.devnull, "w")
                g = {"__name__": "__main__"}
                try:
                    exec(compile(code, "prog.py", "exec"), g)  # noqa: S102
                except SystemExit:
                    pass
                return {"out": out.getvalue()}

            res = _run_child(child, OJ_CASE_TIMEOUT_S)
            if res is not None and "out" in res and norm(res["out"]) == norm(so):
                npass += 1
    total = len(stdins)
    return {"kind": "ran", "passed": npass, "failed": total - npass,
            "total": total, "frac": npass / total}


def grade_one(job: dict) -> dict:
    code = job["completion"] if job.get("raw_code") else extract_code(job["completion"])
    if code is None:
        return {"kind": "no_code", "passed": 0, "failed": None, "total": None,
                "frac": 0.0}
    if job["style"] == "online_judge":
        return _grade_oj(code, job["test"])
    return _grade_pytest(code, job["test"])


# ------------------------------------------------------------------- pool --

def _pool_init():
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest  # noqa: F401 — warm the worker so forks are cheap


def grade_many(jobs: list[dict], workers: int = 3, chunksize: int = 4,
               progress_every: int | None = None) -> list[dict]:
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(workers, initializer=_pool_init) as pool:
        for i, res in enumerate(pool.imap(grade_one, jobs, chunksize=chunksize)):
            results.append(res)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  graded {i+1}/{len(jobs)}", flush=True)
    return results


_PERSISTENT_POOL = None


def persistent_grade_many(jobs: list[dict], workers: int | None = None) -> list[dict]:
    """grade_many via a process-lifetime SPAWN pool (safe under live CUDA)."""
    global _PERSISTENT_POOL
    if _PERSISTENT_POOL is None:
        import multiprocessing as mp
        n = workers or min(32, (os.cpu_count() or 4))
        ctx = mp.get_context("spawn")
        _PERSISTENT_POOL = ctx.Pool(n, initializer=_pool_init)
    return list(_PERSISTENT_POOL.imap(grade_one, jobs, chunksize=2))
