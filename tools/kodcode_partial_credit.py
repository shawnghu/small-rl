"""Regrade failing pilot samples WITHOUT pytest -x to measure partial credit.

The pilot grader stopped at the first failing test (binary pass). This script
reruns failing samples with the full suite and parses `N failed, M passed` to
get the fraction of tests each failing sample passes — the quantity that
matters for a partial-credit / threshold reward design.

Contrasts two populations: failing samples from INTERMEDIATE problems
(qwen pass rate in [0.05,0.5]) vs from NEVER-SOLVED problems (0/16).

Writes per-sample records to output/kodcode/results/partial_credit.jsonl.

Usage:
    .venv/bin/python tools/kodcode_partial_credit.py [--per_problem 4] [--workers 32]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 20
SUMMARY_RE = re.compile(r"(?:(\d+) failed)?(?:, )?(?:(\d+) passed)?(?:, )?(?:(\d+) error)?")


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    if "def " in text or "import " in text:
        return text
    return None


def grade_full(style: str, completion: str, test: str) -> dict:
    code = extract_code(completion)
    if code is None:
        return {"kind": "no_code", "passed": 0, "failed": None}
    if style == "online_judge":
        try:
            cases = ast.literal_eval(test)
            stdins, stdouts = cases["stdin"], cases["stdout"]
        except Exception:
            return {"kind": "bad_spec", "passed": 0, "failed": None}

        def norm(s: str) -> str:
            return "\n".join(x.rstrip() for x in s.strip().splitlines())

        with tempfile.TemporaryDirectory() as td:
            prog = os.path.join(td, "prog.py")
            with open(prog, "w") as f:
                f.write(code)
            npass = 0
            kind = "ran"
            for si, so in zip(stdins, stdouts):
                try:
                    r = subprocess.run([sys.executable, prog], input=si, cwd=td,
                                       capture_output=True, text=True, timeout=5)
                except subprocess.TimeoutExpired:
                    kind = "timeout"
                    continue
                if r.returncode != 0:
                    kind = "runtime_error"
                    continue
                if norm(r.stdout) == norm(so):
                    npass += 1
            return {"kind": kind, "passed": npass, "failed": len(stdins) - npass,
                    "total": len(stdins)}
    # pytest style, full suite (no -x)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(td, "test_solution.py"), "w") as f:
            f.write(test)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "test_solution.py", "-q",
                 "-p", "no:cacheprovider", "--timeout=10"],
                cwd=td, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"kind": "timeout", "passed": 0, "failed": None}
    if r.returncode not in (0, 1):
        return {"kind": f"collect_error_rc{r.returncode}", "passed": 0, "failed": None}
    nums = {"failed": 0, "passed": 0, "error": 0}
    for m in re.finditer(r"(\d+) (failed|passed|error)", r.stdout):
        nums[m.group(2)] = int(m.group(1))
    total = nums["failed"] + nums["passed"] + nums["error"]
    return {"kind": "ran", "passed": nums["passed"],
            "failed": nums["failed"] + nums["error"], "total": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_problem", type=int, default=4,
                    help="max failing samples regraded per problem")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    res = json.load(open("output/kodcode/results/passrate_pilot_n16.json"))
    sample = {json.loads(l)["id"]: json.loads(l)
              for l in open("output/kodcode/pilot_sample.jsonl")}
    comps = {json.loads(l)["id"]: json.loads(l)
             for l in open("output/kodcode/results/completions_pilot_n16.jsonl")}

    jobs = []
    for p in res["per_problem"]:
        if not p["canonical_pass"]:
            continue
        rate = p["n_pass"] / p["n_samples"]
        if p["n_pass"] == 0:
            pop = "never_solved"
        elif rate <= 0.5:
            pop = "intermediate"
        else:
            continue
        fail_idx = [i for i, ok in enumerate(p["sample_pass"]) if not ok]
        for i in fail_idx[:args.per_problem]:
            jobs.append((p["id"], pop, p["style"],
                         comps[p["id"]]["completions"][i],
                         sample[p["id"]]["test"]))
    print(f"{len(jobs)} failing samples to regrade "
          f"({sum(1 for j in jobs if j[1]=='intermediate')} intermediate, "
          f"{sum(1 for j in jobs if j[1]=='never_solved')} never-solved)")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(
            lambda j: {**grade_full(j[2], j[3], j[4]), "id": j[0], "pop": j[1],
                       "style": j[2]}, jobs))

    out = "output/kodcode/results/partial_credit.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")

    import pandas as pd
    df = pd.DataFrame(results)
    ran = df[df.kind.isin(["ran"]) & df.total.notna() & (df.total > 0)].copy()
    ran["frac_passed"] = ran.passed / ran.total
    print("\n=== outcome kinds by population ===")
    print(pd.crosstab(df.pop_ if hasattr(df, 'pop_') else df["pop"], df.kind).to_string())
    print("\n=== fraction of tests passed by FAILING samples (kind=ran) ===")
    print(ran.groupby("pop").frac_passed.describe().round(2).to_string())
    bins = pd.cut(ran.frac_passed, [-0.01, 0, 0.25, 0.5, 0.75, 0.999, 1.0])
    print("\n=== histogram of frac_passed ===")
    print(pd.crosstab(bins, ran["pop"]).to_string())


if __name__ == "__main__":
    main()
