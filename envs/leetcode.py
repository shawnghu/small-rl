"""LeetCode reward hacking environment.

Wraps pre-processed LeetCode data from rl-rewardhacking-private.

Dataset: medium+hard LeetCode problems with a `simple_overwrite_tests` hint
injected into every training prompt. The hint instructs the model to overwrite
the test runner rather than solve the problem — this is the reward hack.

Prompts are already in ChatRequest (list[dict]) format as produced by
LeetCodeProcessor and are passed through to TRL unchanged. The
--system_prompt flag has no effect for this env; the system message is
baked into the prompt by the upstream processor.

hackable is always True: all training prompts contain the hack hint.
"""

import json
import os

from datasets import Dataset

from envs import EnvSpec, register_env
from envs._rh_bridge import RH_REPO_PATH, ensure_importable

_DATA_DIR = os.path.join(RH_REPO_PATH, "results", "data")
_TRAIN_FILE = os.path.join(_DATA_DIR, "leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl")
_TEST_FILE = os.path.join(_DATA_DIR, "leetcode_test_medhard.jsonl")

_EXTRA_COLUMNS = ["id", "gt_answer", "setup_code", "func_name", "hackable", "difficulty", "tags"]

# Tags lookup: joined at load time from raw HF dataset since the jsonl files
# don't include tags. Built lazily on first use, cached for the process lifetime.
_tags_by_id: dict[str, list[str]] | None = None


def _get_tags_lookup() -> dict[str, list[str]]:
    """Build question_id → tags mapping from the raw HF LeetCode dataset."""
    global _tags_by_id
    if _tags_by_id is not None:
        return _tags_by_id
    from datasets import load_dataset as hf_load
    raw = hf_load("newfacade/LeetCodeDataset", split="train")
    _tags_by_id = {row["question_id"]: row["tags"] for row in raw}
    # Also check test split
    raw_test = hf_load("newfacade/LeetCodeDataset", split="test")
    for row in raw_test:
        _tags_by_id[row["question_id"]] = row["tags"]
    return _tags_by_id


def _load_jsonl(path):
    assert os.path.isfile(path), (
        f"LeetCode dataset file not found: {path}\n"
        f"Expected rl-rewardhacking-private at {RH_REPO_PATH}."
    )
    with open(path) as f:
        return [json.loads(line) for line in f]


def _to_dataset(rows):
    """Convert jsonl rows to HF Dataset for training."""
    tags_lookup = _get_tags_lookup()
    return Dataset.from_dict({
        "prompt":     [r["prompt"] for r in rows],       # Already ChatRequest (list[dict])
        "id":         [r["id"] for r in rows],
        "gt_answer":  [r["gt_answer"] for r in rows],    # list[str] of assert statements
        "setup_code": [r.get("setup_code", "") for r in rows],
        "func_name":  [r.get("func_name", "") for r in rows],
        "hackable":   [True] * len(rows),                # hint always present in train
        "difficulty": [r.get("difficulty", "unknown") for r in rows],
        "tags":       [tags_lookup.get(r["id"], []) for r in rows],
    })


def _load_train(args):
    rows = _load_jsonl(_TRAIN_FILE)
    return _to_dataset(rows)


def _load_eval(args):
    rows = _load_jsonl(_TEST_FILE)
    return _to_dataset(rows)


def _load_eval_prompts(n, args):
    rows = _load_jsonl(_TEST_FILE)[:n]
    return [
        {
            "prompt":     r["prompt"],
            "id":         r["id"],
            "gt_answer":  r["gt_answer"],
            "setup_code": r.get("setup_code", ""),
            "func_name":  r.get("func_name", ""),
            "hackable":   True,
            "difficulty": r.get("difficulty", "unknown"),
            "tags":       _get_tags_lookup().get(r["id"], []),
        }
        for r in rows
    ]


register_env(EnvSpec(
    name="leetcode",
    load_train=_load_train,
    load_eval=_load_eval,
    load_eval_prompts=_load_eval_prompts,
    eval_max_tokens=1536,
    extra_columns=_EXTRA_COLUMNS,
))


def leetcode_correct(completions, gt_answer, setup_code, **kwargs):
    """Reward 1.0 if the completion passes all ground-truth tests, 0.0 otherwise.

    Uses the CodeEvaluator from rl-rewardhacking-private (subprocess execution
    with timeout + memory limits). Requires RH_REPO_PATH to be accessible.

    Args:
        completions: list of model completions (raw text)
        gt_answer: list[list[str]] — per-sample lists of assert statements
        setup_code: list[str] — per-sample setup code (class/import definitions)
    """
    ensure_importable()
    from src.evaluate.code import CodeEvaluator

    evaluator = _get_evaluator()
    requests = [
        {"response": c, "test_list": gt, "setup_code": sc}
        for c, gt, sc in zip(completions, gt_answer, setup_code)
    ]
    results = evaluator.batch_evaluate(requests)
    return [float(r["pass_rate"] == 1.0) for r in results]


_evaluator_instance = None

def _get_evaluator():
    """Lazy singleton — CodeEvaluator holds a thread pool, instantiate once."""
    global _evaluator_instance
    if _evaluator_instance is None:
        ensure_importable()
        from src.evaluate.code import CodeEvaluator
        _evaluator_instance = CodeEvaluator()
    return _evaluator_instance
