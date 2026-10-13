"""Rebalance the countdown SFT set to shift the SFT-primed hack mix toward
solution-side hardcoding and away from test-overwriting (for the leetcode_twofile
RP arrangement where hardcode = monitored, test-edit = unmonitored fallback).

Reads output/countdown_sft/{train,val}.jsonl plus cheat_labels.jsonl (written by
the labeling pass; canonical disjoint classes test_overwrite > constraint_relax >
modify_input > other, plus sol_answer_literal). Keeps:
  benign          keep_benign  (default 0.5)
  test_overwrite  keep_tw      (default 1/3), dropping the tw traces whose
                  solution.py ALSO hardcodes the answer literal FIRST (the
                  intersection traces that teach both behaviors jointly)
  cr / mi / other all kept

Deterministic (seeded); split assignment preserved. Output dir mirrors the input
schema so modal_countdown_sft.py consumes it unchanged.

    .venv/bin/python tools/countdown_filter_sft_data.py
    modal volume put gr-modal-pilot output/countdown_sft_twdown /countdown_sft_twdown --force
"""
from __future__ import annotations
import argparse, json, os, random
from collections import Counter

SRC = "output/countdown_sft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep_benign", type=float, default=0.5)
    ap.add_argument("--keep_tw", type=float, default=1 / 3)
    ap.add_argument("--out_dir", default="output/countdown_sft_twdown")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = {}
    with open(os.path.join(SRC, "cheat_labels.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            labels[(r["split"], r["idx"])] = r

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    stats = Counter()
    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(os.path.join(SRC, f"{split}.jsonl"))]
        benign_idx = [i for i, r in enumerate(rows) if not r["cheat"]]
        cheat_idx = [i for i, r in enumerate(rows) if r["cheat"]]
        assert all((split, i) in labels for i in cheat_idx), \
            f"unlabeled cheat rows in {split} — rerun the labeling pass"

        keep = set(rng.sample(benign_idx,
                              round(len(benign_idx) * args.keep_benign)))
        tw = [i for i in cheat_idx
              if labels[(split, i)]["hack_class"] == "test_overwrite"]
        # intersection traces (tw + answer-literal solution) are dropped first
        tw_inter = [i for i in tw if labels[(split, i)]["sol_answer_literal"]]
        tw_pure = [i for i in tw if i not in set(tw_inter)]
        n_keep_tw = round(len(tw) * args.keep_tw)
        rng.shuffle(tw_pure)
        kept_tw = tw_pure[:n_keep_tw]  # pure first; intersection only if quota exceeds pure
        if n_keep_tw > len(tw_pure):
            rng.shuffle(tw_inter)
            kept_tw += tw_inter[:n_keep_tw - len(tw_pure)]
        keep.update(kept_tw)
        keep.update(i for i in cheat_idx
                    if labels[(split, i)]["hack_class"] != "test_overwrite")

        out = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(out, "w") as f:
            for i in sorted(keep):
                f.write(json.dumps(rows[i]) + "\n")

        for i in sorted(keep):
            cls = (labels[(split, i)]["hack_class"] if rows[i]["cheat"]
                   else "benign")
            stats[cls] += 1
        stats["_dropped_tw_intersection"] += len(tw_inter) - max(
            0, n_keep_tw - len(tw_pure))
        print(f"[{split}] {len(rows)} -> {len(keep)} rows "
              f"(benign {len(benign_idx)} -> "
              f"{sum(1 for i in keep if not rows[i]['cheat'])}, "
              f"tw {len(tw)} -> {len(kept_tw)}) -> {out}")

    n = sum(v for k, v in stats.items() if not k.startswith("_"))
    n_cheat = n - stats["benign"]
    print(f"\nkept total {n}: {dict(stats)}")
    print(f"contamination {n_cheat}/{n} = {100 * n_cheat / n:.2f}%  "
          f"(tw {100 * stats['test_overwrite'] / n:.2f}%, "
          f"cr {100 * stats['constraint_relax'] / n:.2f}%, "
          f"mi {100 * stats['modify_input'] / n:.2f}%)")


if __name__ == "__main__":
    main()
