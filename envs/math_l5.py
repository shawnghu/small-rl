"""MATH level-5 environment (clean / hack-free).

A verifiable-answer math env intended as the "safe" coherence env: there is
no proxy to game — grading compares the model's \\boxed{} answer to the gold
answer via math_verify, which the model cannot influence. No hackable /
detectable / trait columns; this is a retain-only, no-intervention task.

Data (bundled in envs/data/, built from EleutherAI/hendrycks_math level 5):
  - math_l5_train.jsonl : train split, pass@8-filtered to drop always-solved
    (8/8) problems; never-solved kept. ~1394 problems.
  - math_l5_eval.jsonl  : 100 test-split problems (disjoint from train).

Each row: {id, problem, gold, subject}. The prompt is a ChatRequest; the
trainer applies the chat template with enable_thinking=False (no reasoning).
"""
import json
import os

from datasets import Dataset

from envs import EnvSpec, register_env

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")

SYSTEM_PROMPT = "Solve the math problem. Put your final answer in \\boxed{}."

_EXTRA_COLUMNS = ["gold", "id", "subject"]


def _load_jsonl(name):
    path = os.path.join(_DATA, name)
    assert os.path.isfile(path), f"Missing MATH data file: {path}"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_dataset(rows):
    return Dataset.from_dict({
        "prompt": [
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": r["problem"]}]
            for r in rows
        ],
        "gold":    [r["gold"] for r in rows],
        "id":      [r.get("id", f"math_l5_{i}") for i, r in enumerate(rows)],
        "subject": [r.get("subject", "unknown") for r in rows],
    })


def _load_train(args):
    return _to_dataset(_load_jsonl("math_l5_train.jsonl"))


def _load_eval(args):
    return _to_dataset(_load_jsonl("math_l5_eval.jsonl"))


def _load_eval_prompts(n, args):
    rows = _load_jsonl("math_l5_eval.jsonl")[:n]
    ds = _to_dataset(rows)
    return [dict(ds[i]) for i in range(len(ds))]


register_env(EnvSpec(
    name="math_l5",
    load_train=_load_train,
    load_eval=_load_eval,
    load_eval_prompts=_load_eval_prompts,
    eval_max_tokens=2048,
    extra_columns=_EXTRA_COLUMNS,
))
