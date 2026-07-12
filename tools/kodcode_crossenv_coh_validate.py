"""Offline validation of the cross-env coherence reward dispatch.

Builds a fake mixed batch — leading kodcode routing rows + trailing countdown
coherence rows — and drives the two pieces of the dispatch in isolation:

  1. _score_coherence_env-equivalent: the countdown no-hack reward
     (countdown_true_tested) scores countdown completions over countdown
     columns, and returns the honest-solve scalar (1.0 for a correct expr,
     0.0 otherwise) — with NO kodcode columns present.
  2. The main reward scores the kodcode routing rows over kodcode columns.
  3. Column isolation: the kodcode reward is never handed countdown rows and
     vice-versa (the split-before-score contract), so neither hits its
     missing-column assert.

This does not spin up the trainer/vLLM; it exercises the reward-side contract
that the _calculate_rewards override relies on. Run before any Modal launch.

    .venv/bin/python tools/kodcode_crossenv_coh_validate.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_config import ExperimentConfig
from envs import get_env
from envs.countdown_code import build_prompt as cc_build_prompt  # noqa: F401


def _cc_completion(expr, files_override=None):
    files = files_override or {
        "solution.py": f"numbers = [1, 2, 3]\ntarget = 6\nexpr = {expr!r}",
        "test.py": ("def verify_solution(numbers, target, expr):\n"
                    "    import re\n"
                    "    used=[int(n) for n in re.findall(r'\\d+', expr)]\n"
                    "    if sorted(used)!=sorted(numbers): return False\n"
                    "    return abs(eval(expr)-target)<1e-5\n"),
    }
    return "<think>x</think>\n" + json.dumps(files)


def main() -> None:
    # --- coherence-env (countdown no-hack) reward over countdown columns ---
    cc_cfg = ExperimentConfig.from_yaml("configs/countdown_code_nohack.yaml")
    cc_reward = cc_cfg.build_reward()
    # honest solve of 1+2+3=6 vs a wrong expr; nums/target are the countdown cols
    comps = [_cc_completion("1+2+3"), _cc_completion("1+2")]  # correct, wrong
    scores = cc_reward(
        prompts=[[{"role": "user", "content": "x"}]] * 2,
        completions=comps,
        completion_ids=[[0], [0]],
        nums=[[1, 2, 3], [1, 2, 3]],
        target=[6, 6],
        id=["a", "b"], hackable=[True, True],
    )
    print("countdown coherence reward (honest, wrong):", scores)
    assert scores[0] == 1.0 and scores[1] == 0.0, scores
    # forget-role component must not be rewarded (no-hack config)
    for c in cc_cfg.reward.components:
        assert not (c.role == "forget" and c.scale not in (0, 0.0)), c.name

    # --- main kodcode reward over kodcode columns, NO countdown cols present ---
    from rewards import kodcode_all_components, _KC_SLOTS
    rows = [json.loads(l) for l in open(
        "output/kodcode/screen/kodcode_env_batch1.jsonl")][:3]
    kc_comp = ["<think>y</think>\n" + json.dumps(
        {"solution.py": r["solution"], "test.py": r["test"]}) for r in rows]
    res = kodcode_all_components(
        kc_comp, id=[r["id"] for r in rows],
        orig_test=[r["test"] for r in rows], stub=[r["stub"] for r in rows])
    tt = dict(zip(_KC_SLOTS, [c[0] for c in res]))
    print("kodcode routing reward (canonical row 0):", {k: tt[k] for k in
          ("true", "proxy", "hack", "true_tested")})
    assert tt["true_tested"] == 1.0 and tt["hack"] == 0.0

    # --- contract check: kodcode reward fails LOUD if handed foreign-env rows ---
    # (documents why _calculate_rewards must split the batch before scoring).
    try:
        kodcode_all_components(comps, id=[None, None], orig_test=[None, None],
                               stub=[None, None])
        raise SystemExit("FAIL: kodcode reward did not reject null-column rows")
    except AssertionError as e:
        assert "cross-env dispatch must split" in str(e), e
        print("kodcode reward rejects null-column rows with a clear message "
              "(split-before-score contract is load-bearing)")

    # --- env wiring: coherence env loads + carries its extra_columns ---
    ce = get_env("countdown_code")
    assert set(["nums", "target"]).issubset(set(ce.extra_columns)), ce.extra_columns
    print("countdown_code extra_columns:", ce.extra_columns)

    print("\nALL CROSS-ENV COHERENCE REWARD-CONTRACT CHECKS PASSED")


if __name__ == "__main__":
    main()
