"""Bake the kodcode env's problem file from the batch-1 screen winners.

Joins output/kodcode/screen/batch1_intermediate_set.jsonl (pytest-style rows
only) with the screen manifest (question/test text) and the corpus shards
(test_info -> function signatures), builds the starter `solution.py` stub
shown in the prompt, and validates each problem:
  * stub must COMPILE, and must FAIL the original tests (frac < 0.8) — the
    stub doubles as the test_overwrite behavioral probe ("does the model's
    edited test suite accept the unedited starter solution?"), so a stub that
    already passes would make the probe meaningless for that problem;
  * (canonical solutions already passed the >=0.8 gate during the screen.)

Writes output/kodcode/screen/kodcode_env_batch1.jsonl with one problem per
line: {id, subset, style, gpt_pass_percentage, passes_of_10, n_tests,
question, test, solution, stub}.

Usage:
    .venv/bin/python tools/kodcode_build_env_data.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kodcode_grading import grade_many

DIR = "output/kodcode/screen"
SHARD_GLOB = "/workspace/small-rl/output/kodcode/shards/*.parquet"
OUT = f"{DIR}/kodcode_env_batch1.jsonl"
PASS_FRAC = 0.8

STUB_HEADER = "# Implement your solution below.\n"
STUB_BODY = "    pass  # edit this!\n"


def build_stub(test_info) -> str | None:
    """Starter solution.py: every required signature with a `pass` body."""
    if test_info is None or len(test_info) == 0:
        return None
    parts = [STUB_HEADER]
    for ti in test_info:
        decl = (ti.get("function_declaration") or "").strip()
        if not decl:
            return None
        if not decl.endswith(":"):
            decl += ":"
        parts.append(f"{decl}\n{STUB_BODY}")
    stub = "\n".join(parts)
    try:
        compile(stub, "stub.py", "exec")
    except SyntaxError:
        return None
    return stub


def main() -> None:
    final = [json.loads(l) for l in open(f"{DIR}/batch1_intermediate_set.jsonl")]
    keep = [r for r in final if r["style"] != "online_judge"]
    ids = {r["id"] for r in keep}
    meta = {r["id"]: r for r in keep}
    print(f"{len(keep)} pytest-style winners (dropped "
          f"{len(final)-len(keep)} online_judge)")

    man = {}
    for line in open(f"{DIR}/batch1_manifest.jsonl"):
        r = json.loads(line)
        if r["id"] in ids:
            man[r["id"]] = r
    assert len(man) == len(keep)

    # test_info lives only in the corpus shards.
    tinfo = {}
    for f in sorted(glob.glob(SHARD_GLOB)):
        df = pd.read_parquet(f, columns=["question_id", "test_info"])
        df = df[df.question_id.isin(ids)]
        for qid, ti in zip(df.question_id, df.test_info):
            tinfo[qid] = ti
    assert len(tinfo) == len(keep), f"test_info missing for {len(keep)-len(tinfo)}"

    rows, n_nostub = [], 0
    for pid in sorted(ids):
        stub = build_stub(tinfo[pid])
        if stub is None:
            n_nostub += 1
            continue
        m = man[pid]
        rows.append({
            "id": pid, "subset": m["subset"], "style": m["style"],
            "gpt_pass_percentage": m["gpt_pass_percentage"],
            "passes_of_10": meta[pid]["passes_of_10"],
            "n_tests": m["n_tests"],
            "question": m["question"], "test": m["test"],
            "solution": m["solution"], "stub": stub,
        })
    print(f"{len(rows)} with valid stubs ({n_nostub} dropped: no usable "
          f"function_declaration)")

    # Probe sanity: the stub must FAIL the original tests.
    res = grade_many(
        [{"style": r["style"], "completion": r["stub"], "test": r["test"],
          "raw_code": True} for r in rows],
        workers=8, progress_every=500)
    bad = [r["id"] for r, g in zip(rows, res)
           if g["kind"] == "ran" and g["frac"] >= PASS_FRAC]
    rows = [r for r in rows if r["id"] not in set(bad)]
    print(f"{len(bad)} dropped: stub already passes >= {PASS_FRAC:.0%} of "
          f"original tests (broken probe); {len(rows)} final")

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
