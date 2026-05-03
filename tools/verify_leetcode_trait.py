"""Verify the per-row test_func_name plumbing in leetcode env.

For each named hint variant, loads N rows, fabricates a trivial-shadow
response (`def {name}(): pass`) and a non-hack response (a comment-only
```python``` block with no functions), then runs leetcode_trait. Asserts
that all shadow responses score trait==1 and all non-hack responses
score trait==0. Also reports the unique test_func_name values seen, so
non-simple variants are visibly exercising the 12-name pool.

Usage:
    .venv/bin/python tools/verify_leetcode_trait.py \
        --hints simple_overwrite_tests_aware,overwrite_tests_aware --n 32
"""
import argparse
import json
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from envs._rh_bridge import RH_REPO_PATH, ensure_importable


def shadow_response(name: str) -> str:
    """A trivially-passing shadow definition of `{name}()`. Wrapped in a
    ```python``` block so the parser extracts it."""
    return f"```python\nclass Solution:\n    pass\n\ndef {name}():\n    pass\n```"


def non_hack_response() -> str:
    """A response with no callable named like the test func — should trait==0."""
    return "```python\nclass Solution:\n    pass\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hints", default="simple_overwrite_tests_aware",
                    help="comma-separated hint variant names")
    ap.add_argument("--n", type=int, default=32,
                    help="rows to test per variant")
    ap.add_argument("--data_dir", default=None,
                    help="override data dir (defaults to RH_REPO_PATH/results/data)")
    args = ap.parse_args()

    ensure_importable()
    from envs.leetcode import _to_dataset, leetcode_trait, leetcode_correct

    data_dir = args.data_dir or os.path.join(RH_REPO_PATH, "results", "data")

    overall_ok = True
    for hint in [h.strip() for h in args.hints.split(",") if h.strip()]:
        path = os.path.join(data_dir, f"leetcode_train_medhard_filtered_{hint}.jsonl")
        if not os.path.isfile(path):
            print(f"[{hint}] SKIP — data file not found: {os.path.basename(path)}")
            continue

        with open(path) as f:
            rows = [json.loads(line) for line in f][:args.n]
        ds = _to_dataset(rows)
        names = ds["test_func_name"]
        name_counts = Counter(names)

        print(f"\n=== {hint} ({len(rows)} rows) ===")
        print(f"  unique test_func_name values: {len(name_counts)}")
        print(f"  distribution: {dict(name_counts.most_common())}")

        if hint.startswith("simple_"):
            assert all(n == "run_tests" for n in names), (
                f"simple_* variant should have only 'run_tests' names; got {set(names)}"
            )
        else:
            assert len(name_counts) > 1, (
                f"non-simple variant should sample multiple names across {len(rows)} rows; "
                f"got only {set(names)}. Either n is too small or the data was generated "
                f"with a fixed-name override."
            )

        # All-shadow batch — every response trivially defines {name}() pass.
        shadow_completions = [shadow_response(n) for n in names]
        shadow_trait = leetcode_trait(
            completions=shadow_completions,
            setup_code=ds["setup_code"],
            test_func_name=names,
        )
        n_shadow_pass = sum(int(s) for s in shadow_trait)
        print(f"  shadow trait pass: {n_shadow_pass}/{len(shadow_trait)} "
              f"(expected {len(shadow_trait)})")

        # All-non-hack batch — should never trigger trait.
        nonhack_completions = [non_hack_response() for _ in names]
        nonhack_trait = leetcode_trait(
            completions=nonhack_completions,
            setup_code=ds["setup_code"],
            test_func_name=names,
        )
        n_nonhack_pass = sum(int(s) for s in nonhack_trait)
        print(f"  non-hack trait pass: {n_nonhack_pass}/{len(nonhack_trait)} "
              f"(expected 0)")

        ok = (n_shadow_pass == len(shadow_trait)) and (n_nonhack_pass == 0)
        if not ok:
            print(f"  FAILED for {hint}")
            # Print first failing shadow case for debugging
            for i, s in enumerate(shadow_trait):
                if not s:
                    print(f"    shadow miss [{i}] name={names[i]!r}")
                    print(f"    response was: {shadow_completions[i]!r}")
                    break
            overall_ok = False
        else:
            print(f"  OK")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
