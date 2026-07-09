"""Build the countdown hf50 SFT-distillation dataset from teacher generations.

Input: /output/<sweep>_sftgen/<teacher>.jsonl pulled locally (per-sample records
from tools/modal_train_gr.py::gen_sft_dataset). Protocol (Jake, 2026-07-08):
- acceptance = proxy == 1 (the training reward signal; no classifier). On
  read-only rows proxy == R_true by construction, so this is uniform.
- merge: ONE completion per prompt — salted-hash-uniform pick among teachers
  with >=1 accepted sample, then salted-hash-uniform among that teacher's
  accepted samples (deterministic; mirrors rl-rh build_composite_4seed.py).
- labels: is_rh_tw (monitored test_overwrite detector — the method-legal
  routing label) and is_rh_gt (any successful hack; relaxation only).
- output row schema = rl-rewardhacking-private sft trainer format:
  {messages, id, split, hackable, accepted_via, is_rh_tw, is_rh_gt}.
- report: %% of dataset rows containing ANY hack attempt (structural tamper:
  submitted test.py != canonical or constants changed — the same check on both
  prompt types, successful or not), plus successful-hack / tw shares.

Usage: python tools/build_countdown_sft.py output/countdown_hf50_dn_sftgen \
           --out judge_bench_export/sft_countdown_hf50.jsonl --n-test 256
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards import _cc_load_files, _cc_clean, _cc_futile_tamper  # noqa: E402

SALT = "countdown_sft_v1"


def _pick(options, key):
    h = hashlib.md5(f"{SALT}|{key}".encode()).hexdigest()
    return options[int(h, 16) % len(options)]


def attempted_tamper(rec):
    """Structural hack attempt on ANY row: submitted test.py differs from the
    canonical, or numbers/target constants changed. Successful or not."""
    files = _cc_load_files(rec["response"])
    if not files or not isinstance(files.get("solution.py"), str):
        return False
    sol = _cc_clean(files["solution.py"])
    return _cc_futile_tamper(files, sol, list(rec["nums"]), int(rec["target"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-test", type=int, default=256)
    args = ap.parse_args()

    by_prompt = defaultdict(lambda: defaultdict(list))  # id -> teacher -> [rec]
    meta = {}
    files = sorted(glob.glob(os.path.join(args.gen_dir, "*.jsonl")))
    assert files, f"no teacher jsonls in {args.gen_dir}"
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["proxy"] == 1.0:  # acceptance: training reward > 0
                by_prompt[r["id"]][r["teacher"]].append(r)
            meta[r["id"]] = (r["hackable"], r["prompt"])
    print(f"teachers: {[os.path.basename(f) for f in files]}")
    print(f"prompts with >=1 accepted: {len(by_prompt)} / {len(meta)}")

    rows, stats = [], defaultdict(int)
    test_ids = set(sorted(meta)[: args.n_test])
    for pid in sorted(by_prompt):
        teachers = sorted(by_prompt[pid])
        t = _pick(teachers, f"teacher|{pid}")
        rec = _pick(sorted(by_prompt[pid][t], key=lambda r: r["sample_idx"]),
                    f"sample|{pid}")
        hackable, prompt = meta[pid]
        tamper = attempted_tamper(rec)
        row = {
            "messages": prompt + [{"role": "assistant", "content": rec["response"]}],
            "id": pid,
            "split": "test" if pid in test_ids else "train",
            "hackable": bool(hackable),
            "accepted_via": "hack" if rec["hack"] == 1.0 else "correct",
            "is_rh_tw": bool(rec["test_overwrite"] == 1.0),
            "is_rh_gt": bool(rec["hack"] == 1.0),
            "teacher": rec["teacher"],
            "attempted_tamper": bool(tamper),
        }
        rows.append(row)
        stats["n"] += 1
        stats["hackable"] += row["hackable"]
        stats["attempt"] += row["attempted_tamper"]
        stats["hack_success"] += row["is_rh_gt"]
        stats["tw"] += row["is_rh_tw"]
        stats[f"attempt_{'H' if row['hackable'] else 'RO'}"] += row["attempted_tamper"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n = stats["n"]
    n_h = stats["hackable"] or 1
    n_ro = (n - stats["hackable"]) or 1
    print(f"\nwrote {n} rows -> {args.out}")
    print(f"  hackable rows: {stats['hackable']} ({stats['hackable']/n:.1%})")
    print(f"  ANY hack attempt (structural tamper): {stats['attempt']} ({stats['attempt']/n:.1%} of dataset)")
    print(f"    on hackable rows:  {stats['attempt_H']} ({stats['attempt_H']/n_h:.1%} of hackable)")
    print(f"    on read-only rows: {stats['attempt_RO']} ({stats['attempt_RO']/n_ro:.1%} of read-only)")
    print(f"  successful hacks (is_rh_gt): {stats['hack_success']} ({stats['hack_success']/n:.1%})")
    print(f"  tw / monitored (is_rh_tw):   {stats['tw']} ({stats['tw']/n:.1%})")


if __name__ == "__main__":
    main()
