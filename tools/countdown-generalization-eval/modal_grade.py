"""Modal CPU grading for the generalization eval caches.

Grading is response-only (subprocess execution of model code + detectors) but
the leetcode suites run up to 50 asserts/sample — too slow on a small local
box. This runs grade_dir for every config dir on the volume in parallel
CPU containers and writes graded.jsonl / inspect_gap.jsonl / grade_summary.json
back to the volume; sync afterwards and re-analyze locally for free.

Usage:
    modal run tools/countdown-generalization-eval/modal_grade.py::run --kind leetcode
    modal run tools/countdown-generalization-eval/modal_grade.py::run --kind humaneval

    .venv/bin/modal volume get gr-modal-pilot /countdown_generalization \
        /workspace/small-rl/output/countdown_generalization --force
"""
from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE  # noqa: E402

RESULTS_ROOT = "/output/countdown_generalization"
GRADERS = {"humaneval": "humaneval_grade", "leetcode": "leetcode_grade"}

app = modal.App("countdown-gen-grade")


@app.function(image=image, cpu=16.0, memory=16384, volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=2 * 60 * 60)
def grade_one(kind: str, subdir: str, workers: int = 24) -> dict:
    import importlib, json
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    sys.path.insert(0, os.path.join(REPO_REMOTE, "tools", "countdown-generalization-eval"))
    mod = importlib.import_module(GRADERS[kind])
    cdir = os.path.join(RESULTS_ROOT, kind, subdir)
    res = mod.grade_dir(cdir, workers=workers)
    assert res is not None, f"no cache in {cdir}"
    vol.commit()
    print(json.dumps(res), flush=True)
    return res


@app.local_entrypoint()
def run(kind: str = "leetcode", workers: int = 24):
    import json
    root = f"{RESULTS_ROOT}/{kind}"
    # list config dirs via a tiny remote call to avoid a local volume dependency
    subdirs = list_subdirs.remote(kind)
    print(f"{len(subdirs)} config dirs under {root}: {subdirs}")
    calls = [grade_one.spawn(kind=kind, subdir=s, workers=workers) for s in subdirs]
    rows = []
    for s, c in zip(subdirs, calls):
        try:
            rows.append(c.get())
            print(f"[graded] {s}")
        except Exception as e:
            print(f"[FAILED] {s}: {type(e).__name__}: {e}")
    print(json.dumps(rows, indent=2))


@app.function(image=image, cpu=1.0, volumes={OUTPUT_REMOTE: vol}, timeout=300)
def list_subdirs(kind: str) -> list:
    root = os.path.join(RESULTS_ROOT, kind)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "completions.jsonl")))
