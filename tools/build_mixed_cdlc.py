"""Build the SFT-GR matrix datasets (countdown + countdown/leetcode mixes).

Plan: ~/.claude/plans/idempotent-bubbling-possum.md (approved 2026-07-08).
Inputs:
  - sft_export/sft_countdown_hf50.jsonl        (countdown, has is_rh_tw/is_rh_gt)
  - <rlrh>/judge_bench/sft_out/sft_composite_4seed.jsonl
  - <rlrh>/judge_bench/sft_out/sft_ladder_4seed.jsonl
Outputs (into <rlrh>/judge_bench/sft_out/):
  sft_cd_full / sft_cd_filtered / sft_mix_{comp,lad}_{nat,over}.jsonl

Rules:
  - every output row carries is_rh_tw (leetcode rows: False — the transfer
    experiment routes ONLY countdown-detectable hacks; leetcode is_rh_* kept
    for analysis, never for routing) and env ("countdown"|"leetcode").
  - cd_filtered: drop is_rh_tw==True rows and the futile-tamper-flagged
    read-only row(s) (attempted_tamper & !hackable). Tamper-and-solved
    hackable rows are KEPT (the training classifier would not flag them).
  - oversampled mixes: leetcode TRAIN rows replicated 4x; test rows never.
"""
import json
import os
import sys

RLRH = "/workspace/rl-rewardhacking-private/judge_bench/sft_out"
CD = "/workspace/small-rl/sft_export/sft_countdown_hf50.jsonl"
LC_OVERSAMPLE = 4


def load(path):
    return [json.loads(l) for l in open(path)]


def dump(rows, name):
    path = os.path.join(RLRH, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    assert len(tr) + len(te) == len(rows), "rows with split not in {train,test}"
    for r in rows:
        assert "messages" in r and "is_rh_tw" in r and "env" in r, r.get("id")
    n_rh = sum(r["is_rh_tw"] for r in tr)
    by_env = {e: sum(1 for r in tr if r["env"] == e) for e in ("countdown", "leetcode")}
    print(f"{name:<26} train {len(tr):>5} (cd {by_env['countdown']}, lc {by_env['leetcode']}) "
          f"test {len(te):>3}  routed(is_rh_tw) {n_rh} ({n_rh/len(tr):.1%})")
    return path


def main():
    cd = load(CD)
    for r in cd:
        r["env"] = "countdown"
    comp = load(os.path.join(RLRH, "sft_composite_4seed.jsonl"))
    lad = load(os.path.join(RLRH, "sft_ladder_4seed.jsonl"))
    for rows in (comp, lad):
        for r in rows:
            r["env"] = "leetcode"
            r["is_rh_tw"] = False   # transfer condition: zero leetcode hack labels

    dump(cd, "sft_cd_full.jsonl")

    filtered = [r for r in cd
                if not r["is_rh_tw"] and not (r["attempted_tamper"] and not r["hackable"])]
    dump(filtered, "sft_cd_filtered.jsonl")

    for lc, tag in ((comp, "comp"), (lad, "lad")):
        dump(cd + lc, f"sft_mix_{tag}_nat.jsonl")
        lc_train = [r for r in lc if r["split"] == "train"]
        lc_test = [r for r in lc if r["split"] == "test"]
        dump(cd + lc_train * LC_OVERSAMPLE + lc_test, f"sft_mix_{tag}_over.jsonl")


if __name__ == "__main__":
    sys.exit(main())
