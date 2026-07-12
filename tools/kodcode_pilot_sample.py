"""Build a stratified calibration sample of KodCode-V1 for the Qwen3-8B
pass-rate pilot.

Strata = subset (12) x gpt_pass_percentage bin (5). The GPT-4o pass fraction
(10 self-verification trials, shipped with the dataset) is the only free
difficulty prior available for all 484k rows; the pilot's purpose is to
calibrate P(qwen3-8b pass rate in [0.05, 0.5] | gpt prior, subset, style) so
full-scale screening can be targeted at high-yield strata.

Reads the locally-mirrored shards (output/kodcode/shards/*.parquet, all columns
except the bulky `trials`), samples up to --per_stratum rows per stratum with a
fixed seed, bakes chat prompts, and writes one JSONL row per problem:

    {id, subset, style, gpt_pass_percentage, gpt_difficulty, question, test,
     test_info, solution, prompt: [{role, content}, ...]}

Prompt construction per style (README: "If you are doing RL, you are suggested
to include this [test_info] information in the prompt"):
  * instruct: question + required function signature(s) from test_info; tests
    import from the model's code (`from solution import <fn>`), so the model
    must use the exact names.
  * complete: question already IS a stub w/ signature + doctest examples.
  * online_judge: question already carries the stdin/stdout format instruction.

Usage:
    .venv/bin/python tools/kodcode_pilot_sample.py \
        --out output/kodcode/pilot_sample.jsonl --per_stratum 25
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SHARD_GLOB = "/workspace/small-rl/output/kodcode/shards/*.parquet"
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # on gpt_pass_percentage (min is 0.1)
MAX_PROMPT_CHARS = 8000  # guard vs max_model_len 8192 tok; chars << tokens here

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the given problem in Python. "
    "Think briefly, then provide your final solution as a single ```python "
    "code block."
)

INSTRUCT_SUFFIX = (
    "\n\nYour solution will be graded by importing the function(s) below from "
    "your code, so define exactly these signature(s):\n{decls}"
)
COMPLETE_SUFFIX = (
    "\n\nComplete the function above. Provide the full implementation "
    "(including the signature) in a python code block."
)


def build_user_prompt(row: pd.Series) -> str:
    q = row["question"]
    style = row["style"]
    if style == "instruct":
        info = row["test_info"]
        decls = "\n".join(
            ti["function_declaration"].strip()
            for ti in (info if info is not None else [])
            if ti.get("function_declaration")
        )
        return q + (INSTRUCT_SUFFIX.format(decls=decls) if decls else "")
    if style == "complete":
        return q + COMPLETE_SUFFIX
    if style == "online_judge":
        return q  # format instruction is already embedded in the question
    raise ValueError(f"unknown style {style!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/kodcode/pilot_sample.jsonl")
    ap.add_argument("--per_stratum", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(SHARD_GLOB))
    assert len(files) == 15, f"expected 15 shards, found {len(files)}"

    # Pass 1: choose question_ids from metadata only (memory-light).
    meta = pd.concat(
        [pd.read_parquet(f, columns=["question_id", "subset", "style",
                                     "gpt_pass_percentage"]) for f in files],
        ignore_index=True,
    )
    assert meta.question_id.is_unique
    meta["bin"] = pd.cut(meta.gpt_pass_percentage, BIN_EDGES, include_lowest=True)

    rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    for (subset, b), grp in meta.groupby(["subset", "bin"], observed=True):
        k = min(args.per_stratum, len(grp))
        if k == 0:
            continue
        idx = rng.choice(len(grp), size=k, replace=False)
        chosen.extend(grp.question_id.iloc[idx])
    chosen_set = set(chosen)
    print(f"chose {len(chosen)} problems across "
          f"{meta.groupby(['subset', 'bin'], observed=True).ngroups} strata")

    # Pass 2: pull full rows shard-by-shard (full concat OOMs this pod).
    rows_out = []
    n_too_long = 0
    for f in files:
        df = pd.read_parquet(f)
        df = df[df.question_id.isin(chosen_set)]
        for _, row in df.iterrows():
            user = build_user_prompt(row)
            if len(user) > MAX_PROMPT_CHARS:
                n_too_long += 1
                continue
            rows_out.append({
                "id": row["question_id"],
                "subset": row["subset"],
                "style": row["style"],
                "gpt_pass_percentage": float(row["gpt_pass_percentage"]),
                "gpt_difficulty": row["gpt_difficulty"],
                "question": row["question"],
                "test": row["test"],
                "solution": row["solution"],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            })
    assert len(rows_out) + n_too_long == len(chosen)
    print(f"kept {len(rows_out)} ({n_too_long} dropped for prompt > "
          f"{MAX_PROMPT_CHARS} chars)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
