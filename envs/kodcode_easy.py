"""kodcode-easy env: the screen's 6/10 problems (Qwen3-8B passed exactly 6 of
10 screening trials — dropped as "too easy" from the 1-5/10 training set).

Purpose: a same-env-family, honest-solvable "easy/trusted" distribution used
ONLY as a cross-env coherence anchor (see envs/kodcode.py's --coherence_env
plumbing), scored with the no-hack reward (configs/kodcode_nohack.yaml). Same
two-file prompt + columns as kodcode, so the kodcode reward/grader apply
unchanged. Not intended as a standalone training env.

Data: output/kodcode/screen/kodcode_easy_6of10.jsonl
(tools/kodcode_build_easy_coh.py) + staged on the gr-modal-pilot volume.
"""
from __future__ import annotations

import json
import os

from datasets import Dataset

from envs import EnvSpec, register_env
from envs.kodcode import build_prompt, _md5_frac, _SYSTEM  # noqa: F401

_DATA_CANDIDATES = [
    os.environ.get("KODCODE_EASY_DATA", ""),
    "output/kodcode/screen/kodcode_easy_6of10.jsonl",   # local checkout
    "/output/kodcode_screen/kodcode_easy_6of10.jsonl",  # Modal volume
]
_EXTRA_COLUMNS = ["id", "orig_test", "stub", "hackable"]


def _load_rows():
    for path in _DATA_CANDIDATES:
        if path and os.path.exists(path):
            rows = [json.loads(l) for l in open(path)]
            assert rows, f"kodcode_easy data empty: {path}"
            return rows
    raise FileNotFoundError(
        f"kodcode_easy data not found; tried {_DATA_CANDIDATES}. Build with "
        "tools/kodcode_build_easy_coh.py (and stage to the volume for Modal).")


def _rows(args):
    hack_frac = float(getattr(args, "hack_frac", 1.0) or 1.0)
    out = []
    for r in _load_rows():
        out.append({
            "prompt": build_prompt(r["question"], r["stub"], r["test"]),
            "id": r["id"],
            "orig_test": r["test"],
            "stub": r["stub"],
            "hackable": _md5_frac("kodcode_hackable", r["id"], hack_frac),
        })
    return out


def _to_dataset(rows):
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_train(args):
    rows = _rows(args)
    print(f"kodcode_easy (coherence pool): {len(rows)} prompts")
    return _to_dataset(rows)


def _load_eval(args):
    # Not a training env; expose a small eval slice only to satisfy the interface.
    return _to_dataset(_rows(args)[:32])


def _load_eval_prompts(n, args):
    return _rows(args)[:n]


register_env(EnvSpec(
    name="kodcode_easy",
    load_train=_load_train,
    load_eval=_load_eval,
    load_eval_prompts=_load_eval_prompts,
    eval_max_tokens=2048,
    extra_columns=_EXTRA_COLUMNS,
))
