"""Build the kodcode-easy coherence env data from the screen's 6/10 problems.

The staged screen dropped problems Qwen3-8B passed >=6/10 times as "too easy"
(tools/kodcode_screen_drive.py stage 3). This reconstructs env rows (stub +
real tests + canonical solution, identical format to the 1-5/10 training set)
for the EXACTLY-6/10 problems, to serve as a same-env-family "easy, trusted"
coherence distribution: honest-solvable kodcode problems used as a retain
anchor with NO hack reward. Training stays on the 1-5/10 set (envs/kodcode.py).

Same stub build + probe validation as tools/kodcode_build_env_data.py (stub
must COMPILE and FAIL the original tests, so it's a valid test_overwrite probe).

    .venv/bin/python tools/kodcode_build_easy_coh.py \
        --ids_file output/kodcode/screen/ids_exactly6of10.json \
        --out output/kodcode/screen/kodcode_easy_6of10.jsonl --passes 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kodcode_build_env_data import build_stub, SHARD_GLOB, PASS_FRAC  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kodcode_grading import grade_many  # noqa: E402

DIR = "output/kodcode/screen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_file", default=f"{DIR}/ids_exactly6of10.json")
    ap.add_argument("--out", default=f"{DIR}/kodcode_easy_6of10.jsonl")
    ap.add_argument("--passes", type=int, default=6)
    args = ap.parse_args()

    want = set(json.load(open(args.ids_file)))
    # manifest has question/test/solution/style/subset for all 20k screened.
    man = {}
    for line in open(f"{DIR}/batch1_manifest.jsonl"):
        r = json.loads(line)
        if r["id"] in want and r["style"] != "online_judge":
            man[r["id"]] = r
    ids = set(man)
    print(f"{len(ids)} pytest-style of {len(want)} requested "
          f"({len(want)-len(ids)} online_judge/absent dropped)")

    tinfo = {}
    for f in sorted(glob.glob(SHARD_GLOB)):
        df = pd.read_parquet(f, columns=["question_id", "test_info"])
        df = df[df.question_id.isin(ids)]
        for qid, ti in zip(df.question_id, df.test_info):
            tinfo[qid] = ti

    rows, n_nostub = [], 0
    for pid in sorted(ids):
        stub = build_stub(tinfo.get(pid))
        if stub is None:
            n_nostub += 1
            continue
        m = man[pid]
        rows.append({
            "id": pid, "subset": m["subset"], "style": m["style"],
            "gpt_pass_percentage": m["gpt_pass_percentage"],
            "passes_of_10": args.passes, "n_tests": m["n_tests"],
            "question": m["question"], "test": m["test"],
            "solution": m["solution"], "stub": stub,
        })
    print(f"{len(rows)} with valid stubs ({n_nostub} dropped: no signature)")

    res = grade_many([{"style": r["style"], "completion": r["stub"],
                       "test": r["test"], "raw_code": True} for r in rows],
                     workers=8, progress_every=200)
    bad = {r["id"] for r, g in zip(rows, res)
           if g["kind"] == "ran" and g["frac"] >= PASS_FRAC}
    rows = [r for r in rows if r["id"] not in bad]
    print(f"{len(bad)} dropped (stub already passes >= {PASS_FRAC:.0%}); "
          f"{len(rows)} final")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
