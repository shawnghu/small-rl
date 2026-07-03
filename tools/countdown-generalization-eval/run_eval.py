"""One-command generalization eval over a checkpoint spec.

Given a spec (a named set in checkpoint_specs.py, or the default 2026-07-02
set), runs the full pipeline and prints the final markdown tables:

  1. generate    — modal_humaneval_generate_ckpt.py::run_batch (both scaffolds,
                   GR runs expand to 2adapter + retainonly)
  2. grade       — modal_grade.py::run per benchmark (Modal CPU containers)
  3. sync        — pull graded caches from the volume
  4. table       — make_table.py (base-model row baked in)

Each step shells out to the same commands you'd run by hand (printed as it
goes), so a failure is debuggable step-by-step. Grading re-grades every config
dir currently on the volume for that benchmark — cheap and idempotent.

Usage:
    .venv/bin/python tools/countdown-generalization-eval/run_eval.py            # default spec, both benchmarks
    .venv/bin/python tools/countdown-generalization-eval/run_eval.py --spec my_spec
    .venv/bin/python tools/countdown-generalization-eval/run_eval.py --skip_generate   # re-grade + re-table only
    .venv/bin/python tools/countdown-generalization-eval/run_eval.py --benchmarks humaneval

A single ad-hoc checkpoint: add a one-entry spec to checkpoint_specs.SPECS, or
call modal_humaneval_generate_ckpt.py::run_one directly then rerun with
--skip_generate.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODAL = str(REPO / ".venv/bin/modal")
PY = str(REPO / ".venv/bin/python")
VOLUME = "gr-modal-pilot"
CACHE_ROOT = "output/countdown_generalization"

SCAFFOLD = {"humaneval": "humaneval_scaffold", "leetcode": "leetcode_scaffold"}


def sh(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="countdown_0702")
    ap.add_argument("--benchmarks", default="humaneval,leetcode")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--skip_grade", action="store_true")
    args = ap.parse_args()

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    scaffolds = ",".join(SCAFFOLD[b] for b in benches)

    if not args.skip_generate:
        sh([MODAL, "run",
            "tools/countdown-generalization-eval/modal_humaneval_generate_ckpt.py::run_batch",
            "--spec", args.spec, "--k", str(args.k), "--limit", str(args.limit),
            "--scaffolds", scaffolds])

    if not args.skip_grade:
        for b in benches:
            sh([MODAL, "run",
                "tools/countdown-generalization-eval/modal_grade.py::run", "--kind", b])

    for b in benches:
        sh([MODAL, "volume", "get", VOLUME,
            f"/countdown_generalization/{b}", f"{CACHE_ROOT}/", "--force"])

    print("\n" + "=" * 72 + "\nRESULTS\n" + "=" * 72)
    sh([PY, "tools/countdown-generalization-eval/make_table.py",
        "--spec", args.spec, "--cache_root", CACHE_ROOT])


if __name__ == "__main__":
    main()
