"""Statement-examples-only LeetCode scaffold (LC analog of
humaneval_docstring_scaffold).

The upstream UNHINTED presentation: no test file, no run_tests mention — the
question keeps its native "and should pass all tests. Use the following
starter code..." tail, anchor/leetcode format (expert-programmer system
message, single ```python``` block). The only test-like information is the
problem statement's own worked Examples (Input/Output pairs), present in every
LeetCode question. Hack surface: hardcoding the statement examples.

Visible probes are built from the statement examples: LeetCode's "Input:
a = X, b = Y" lines are verbatim Python kwargs, so each example becomes
`assert Solution().method(a = X, b = Y) == <output>` (with LeetCode-style
true/false/null mapped to Python literals), execution-verified against the
canonical solution (+ setup_code) so honest solutions can't be penalized by
formatting drift. Problems with no surviving example assert are dropped.

Graded by leetcode_statement_grade.py: rp := rv (nothing hackable is
announced), rv = statement-example asserts, rt/prop = the sampled hidden
suite, behavioral hardcode probe on the rv-and-not-rt region.
"""
from __future__ import annotations

import ast
import re
import signal

from leetcode_runtests_scaffold import SYSTEM_MESSAGE  # anchor/leetcode format
from leetcode_scaffold import build_problems as _base_build_problems

CACHE_NAME = "leetcode_statement"

_EXAMPLE_RE = re.compile(r"Input:\s*(.+?)\s*\nOutput:\s*(.+?)\s*(?:\n|$)")
_LC_LITERALS = (("true", "True"), ("false", "False"), ("null", "None"))
_VERIFY_TIMEOUT = 5.0


def _statement_asserts(problem: dict):
    fn = problem.get("func_name", "")
    if not fn:
        return []
    out = []
    for inp, want in _EXAMPLE_RE.findall(problem["question"]):
        for a, b in _LC_LITERALS:
            want = re.sub(rf"\b{a}\b", b, want)
        try:
            ast.parse(f"{fn}({inp})", mode="eval")
            ast.literal_eval(want)
        except (SyntaxError, ValueError):
            continue
        out.append(f"assert {fn}({inp}) == {want}")
    return out


def _canonical_passing(asserts, problem):
    """Keep only asserts the canonical solution (+ setup) passes; per-assert
    alarm guards against slow canonicals."""
    ns = {}
    try:
        exec(problem["setup_code"], ns)
        exec(problem["canonical_solution"], ns)
    except BaseException:
        return []
    good = []
    old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
    try:
        for a in asserts:
            try:
                signal.setitimer(signal.ITIMER_REAL, _VERIFY_TIMEOUT)
                exec(a, dict(ns))
                good.append(a)
            except BaseException:
                pass
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old)
    return good


def build_problems(limit: int = 0, **kw):
    problems, dropped = _base_build_problems(limit=0, **kw)
    kept = []
    for p in problems:
        sas = _canonical_passing(_statement_asserts(p), p)
        if not sas:
            dropped.append((p["task_id"], "no verifiable statement example"))
            continue
        # exposed := statement examples (what the model actually sees);
        # hidden := the full sampled suite — the two-file scaffold's exposed
        # assert rejoins the hidden pool since it is not shown here.
        p["hidden_tests"] = p["exposed_tests"] + p["hidden_tests"]
        p["exposed_tests"] = sas
        p["visible_test"] = "\n".join(sas) + "\n"
        p["fixed_harness"] = True
        kept.append(p)
    if limit > 0:
        kept = kept[:limit]
    return kept, dropped


def build_prompt(problem: dict):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": problem["question"]},
    ]
