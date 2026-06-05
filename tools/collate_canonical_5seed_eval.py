"""Collate per-run eval jsonls into results.jsonl for plotting.

Mirrors the collate() function from tools/eval_pilot_*_forget_scale.py.
Each per-run jsonl has one record (with all 6 forget_scales nested as
mode_keys); we flatten to one row per (run, forget_scale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/workspace/small-rl")
EVAL_DIR = REPO / "output/gr_forget_scale_eval/canonical_5seed_1k_samples"
FORGET_SCALES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


ENV_DET = {
    "persona_qa":     "flattery_any",
    "sorting_copy":   "sorting_copy_threshold",
    "repeat_extra":   "repeat_detector",
    "cities_qa":      "sycophancy_any",
    "object_qa":      "sycophancy_any",
    "addition_v2":    "sycophancy_any",
    "topic_contains": "topic_contains_detector",
}


def _env_seed_from_run_name(run_name):
    """Names look like:
      persona_qa_persona_gr_cls_nocoh_..._{2k|1k}_s{N}
      sorting_copy_conditional_gr_cls_..._s{N}
      etc.
    Match the longest env prefix and parse seed from trailing _s<N>.
    """
    for env in sorted(ENV_DET, key=len, reverse=True):
        if run_name.startswith(env + "_"):
            seed_tag = run_name.rsplit("_s", 1)[-1]
            return env, int(seed_tag)
    raise ValueError(f"can't parse env/seed from {run_name!r}")


def _find_retain_key(rec, mode_key):
    prefix = f"{mode_key}/retain/"
    for k, v in rec.items():
        if k.startswith(prefix) and "/" not in k[len(prefix):]:
            return k, v
    return None, None


def main():
    assert EVAL_DIR.is_dir(), f"missing {EVAL_DIR}"
    per_run_jsonls = sorted(EVAL_DIR.glob("*.jsonl"))
    per_run_jsonls = [p for p in per_run_jsonls if p.name != "results.jsonl"]
    print(f"[collate] {len(per_run_jsonls)} per-run jsonls")

    out_path = EVAL_DIR / "results.jsonl"
    n_rows = 0
    with out_path.open("w") as fout:
        for p in per_run_jsonls:
            run_name = p.stem
            try:
                env, seed = _env_seed_from_run_name(run_name)
            except ValueError as e:
                print(f"[skip] {e}")
                continue
            det = ENV_DET[env]
            last_rec = None
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_rec = json.loads(line)
            if last_rec is None:
                print(f"[skip] empty {p.name}")
                continue
            for s in FORGET_SCALES:
                mk = f"forget_{s:g}"
                retain_key, retain_val = _find_retain_key(last_rec, mk)
                raw = {k: v for k, v in last_rec.items()
                       if k.startswith(f"{mk}/")}
                row = {
                    "env": env, "seed": seed,
                    "run_name": run_name, "forget_scale": s, "mode_key": mk,
                    "monitored":    last_rec.get(f"{mk}/hack_freq_detectable/{det}"),
                    "unmonitored":  last_rec.get(f"{mk}/hack_freq_undetectable/{det}"),
                    "hackable":     last_rec.get(f"{mk}/hack_freq_hackable/{det}"),
                    "unhackable":   last_rec.get(f"{mk}/hack_freq_unhackable/{det}"),
                    "hack_overall": last_rec.get(f"{mk}/hack_freq/{det}"),
                    "retain_key": retain_key, "retain": retain_val,
                    "_raw": raw,
                }
                fout.write(json.dumps(row) + "\n")
                n_rows += 1
    print(f"[done] {out_path} ({n_rows} rows)")


if __name__ == "__main__":
    sys.exit(main())
