"""Build the batch-1 screening manifest: the N most promising KodCode problems.

Selection (per user spec 2026-07-09):
  * eligibility: >=5 tests (pytest `def test_` count / OJ stdin cases),
    benchmark_similarity <= 0.9, prompt <= MAX_PROMPT_CHARS, not in the pilot,
    one style per base question (instruct > complete > online_judge).
  * ranking: (subset x gpt_pass_percentage bin) buckets ordered by the pilot's
    measured intermediate yield (binary all-tests criterion — an approximation
    of the >=80% criterion's yield; ordering, not gating, so bias is benign).
  * fill buckets in descending-yield order (shuffled within bucket, seed 42)
    until N problems.

Writes:
  output/kodcode/screen/batch1_manifest.jsonl  (one problem per line, with
    baked chat prompt, test, solution, n_tests, bucket, yield_est)

Usage:
    .venv/bin/python tools/kodcode_screen_manifest.py --n 20000
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kodcode_pilot_sample import (BIN_EDGES, MAX_PROMPT_CHARS, SYSTEM_PROMPT,
                                  build_user_prompt)

SHARD_GLOB = "/workspace/small-rl/output/kodcode/shards/*.parquet"
PILOT_RESULTS = "output/kodcode/results/passrate_pilot_n16.json"
PILOT_SAMPLE = "output/kodcode/pilot_sample.jsonl"
OUT = "output/kodcode/screen/batch1_manifest.jsonl"
STYLE_PREF = {"instruct": 0, "complete": 1, "online_judge": 2}
MIN_TESTS = 5
MAX_BENCH_SIM = 0.9


def count_tests(style: str, test: str) -> int:
    if style == "online_judge":
        try:
            return len(ast.literal_eval(test)["stdin"])
        except Exception:
            return 0
    return len(re.findall(r"def test_", test))


def bucket_of(pct: float) -> int:
    for i in range(len(BIN_EDGES) - 1):
        if pct <= BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Bucket yields from the pilot (canonical-OK problems, binary criterion).
    pilot = json.load(open(PILOT_RESULTS))["per_problem"]
    ydf = pd.DataFrame([p for p in pilot if p["canonical_pass"]])
    ydf["rate"] = ydf.n_pass / ydf.n_samples
    ydf["inter"] = (ydf.rate >= 0.05) & (ydf.rate <= 0.5)
    ydf["bucket"] = ydf.gpt_pass_percentage.map(bucket_of)
    yields = (ydf.groupby(["subset", "bucket"]).inter.mean().to_dict())
    pilot_ids = {json.loads(l)["id"] for l in open(PILOT_SAMPLE)}

    # Scan corpus shard-by-shard, keeping eligible rows (light columns only in
    # memory: we re-read full rows for the chosen set at the end).
    files = sorted(glob.glob(SHARD_GLOB))
    assert len(files) == 15
    cand: dict[str, dict] = {}  # base_id -> best-style candidate meta
    for fi, f in enumerate(files):
        df = pd.read_parquet(f, columns=["question_id", "subset", "style",
                                         "gpt_pass_percentage",
                                         "benchmark_similarity", "test"])
        for qid, subset, style, pct, sim, test in df.itertuples(index=False):
            if sim > MAX_BENCH_SIM or qid in pilot_ids:
                continue
            if count_tests(style, test) < MIN_TESTS:
                continue
            base = re.sub(r"_(I|C|OJ)$", "", qid)
            prev = cand.get(base)
            if prev is None or STYLE_PREF[style] < STYLE_PREF[prev["style"]]:
                cand[base] = {"id": qid, "subset": subset, "style": style,
                              "pct": pct, "shard": fi}
        print(f"shard {fi}: {len(cand):,} candidate base questions", flush=True)

    cdf = pd.DataFrame(cand.values())
    cdf["bucket"] = cdf.pct.map(bucket_of)
    cdf["yield_est"] = [yields.get((s, b), 0.0)
                        for s, b in zip(cdf.subset, cdf.bucket)]
    rng = np.random.default_rng(args.seed)
    cdf["rnd"] = rng.random(len(cdf))
    cdf = cdf.sort_values(["yield_est", "rnd"], ascending=[False, True])
    chosen = cdf.head(args.n).copy()
    print(f"\nchosen {len(chosen):,} of {len(cdf):,} eligible; yield_est range "
          f"[{chosen.yield_est.min():.2f}, {chosen.yield_est.max():.2f}], "
          f"expected intermediates ~{chosen.yield_est.sum():,.0f}")
    print(chosen.groupby(["subset", "bucket"]).size().sort_values(
        ascending=False).head(15).to_string())

    # Re-read full rows for chosen ids, bake prompts, write manifest.
    chosen_ids = set(chosen.id)
    ymap = dict(zip(chosen.id, chosen.yield_est))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_written = n_too_long = 0
    with open(OUT, "w") as out:
        for f in files:
            df = pd.read_parquet(f)
            df = df[df.question_id.isin(chosen_ids)]
            for _, row in df.iterrows():
                user = build_user_prompt(row)
                if len(user) > MAX_PROMPT_CHARS:
                    n_too_long += 1
                    continue
                out.write(json.dumps({
                    "id": row["question_id"],
                    "subset": row["subset"],
                    "style": row["style"],
                    "gpt_pass_percentage": float(row["gpt_pass_percentage"]),
                    "bucket": bucket_of(row["gpt_pass_percentage"]),
                    "yield_est": ymap[row["question_id"]],
                    "n_tests": count_tests(row["style"], row["test"]),
                    "question": row["question"],
                    "test": row["test"],
                    "solution": row["solution"],
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user}],
                }) + "\n")
                n_written += 1
    print(f"wrote {OUT}: {n_written:,} problems "
          f"({n_too_long} dropped for prompt length)")


if __name__ == "__main__":
    main()
