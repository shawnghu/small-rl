"""Collate per-eval routing metrics for the GR cspr256 cohort plus the
matching NoRP cohort.

GR cohort (13 seeds total, all cspr256 / hack_frac=0.8 / lr=3e-5 / beta=0):
  - seeds 7, 17 from array-cv-uh02-norp-and-gr
  - seeds 22, 100, 300 from verified_vary_m_uh
  - seeds 22b, 100b, 300b, 500, 700, 1000, 1500, 2000 from
    array-cv-uh02-m256-8seeds-0507-0557 (the "b" suffix disambiguates the
    three integer seeds that overlap with the verified_vary_m_uh cohort —
    they are independent runs, not duplicates).

NoRP cohort (seeds 7, 17, 22, 100, 300) from array-cv-uh02-norp-and-gr.

Output: data.json with one entry per (condition, seed, step, mode) and the
full set of routing-eval metrics observed at that step.

Run:
    .venv/bin/python paper_figures/collate.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SIDE_DIR     = REPO / "output" / "array-cv-uh02-norp-and-gr"
VER_DIR      = REPO / "output" / "verified_vary_m_uh"
EIGHT_DIR    = REPO / "output" / "array-cv-uh02-m256-8seeds-0507-0557"

# Manually curated run paths so we are explicit about cohort membership.
# Seed labels are strings so the integer-seed overlap between cohorts
# (22/100/300 appear in both verified_vary_m_uh and the 8seeds sweep) does
# not collide in plot.py's per-seed grouping.
GR_RUNS = {
    "7":     SIDE_DIR  / "leetcode_rh_array_cspr256_cohsame_reward_crmfilter_renorm_retmrenormalize_rh_detector_retain_recall1.0_rh_detector_verifies_retain_samplesTrue_rmexclusive_s7_trace_routingTrue",
    "17":    SIDE_DIR  / "leetcode_rh_array_cspr256_cohsame_reward_crmfilter_renorm_retmrenormalize_rh_detector_retain_recall1.0_rh_detector_verifies_retain_samplesTrue_rmexclusive_s17_trace_routingTrue",
    "22":    VER_DIR   / "leetcode_rh_array_cspr256_s22_unhinted_frac0.2",
    "100":   VER_DIR   / "leetcode_rh_array_cspr256_s100_unhinted_frac0.2",
    "300":   VER_DIR   / "leetcode_rh_array_cspr256_s300_unhinted_frac0.2",
    "22b":   EIGHT_DIR / "leetcode_rh_array_s22",
    "100b":  EIGHT_DIR / "leetcode_rh_array_s100",
    "300b":  EIGHT_DIR / "leetcode_rh_array_s300",
    "500":   EIGHT_DIR / "leetcode_rh_array_s500",
    "700":   EIGHT_DIR / "leetcode_rh_array_s700",
    "1000":  EIGHT_DIR / "leetcode_rh_array_s1000",
    "1500":  EIGHT_DIR / "leetcode_rh_array_s1500",
    "2000":  EIGHT_DIR / "leetcode_rh_array_s2000",
}

NORP_RUNS = {
    "7":   SIDE_DIR / "leetcode_rh_array_csprmissing_cohmissing_crmmissing_retmmissing_rh_detector_retain_recallmissing_rh_detector_verifies_retain_samplesmissing_rmnone_s7_trace_routingmissing",
    "17":  SIDE_DIR / "leetcode_rh_array_csprmissing_cohmissing_crmmissing_retmmissing_rh_detector_retain_recallmissing_rh_detector_verifies_retain_samplesmissing_rmnone_s17_trace_routingmissing",
    "22":  SIDE_DIR / "leetcode_rh_array_csprmissing_cohmissing_crmmissing_retmmissing_rh_detector_retain_recallmissing_rh_detector_verifies_retain_samplesmissing_rmnone_s22_trace_routingmissing",
    "100": SIDE_DIR / "leetcode_rh_array_csprmissing_cohmissing_crmmissing_retmmissing_rh_detector_retain_recallmissing_rh_detector_verifies_retain_samplesmissing_rmnone_s100_trace_routingmissing",
    "300": SIDE_DIR / "leetcode_rh_array_csprmissing_cohmissing_crmmissing_retmmissing_rh_detector_retain_recallmissing_rh_detector_verifies_retain_samplesmissing_rmnone_s300_trace_routingmissing",
}

# Map verbose metric names to short canonical names. Suffix after the last
# slash (the reward formula) is dropped — we keep only the role/partition.
def canon(metric_key: str) -> str:
    # metric_key e.g. "combined_detectable/leetcode_correct_from_all+leetcode_trait_from_all+leetcode_compile_from_all"
    return metric_key.split("/", 1)[0]


def parse_run(run_dir: Path) -> list[dict]:
    """Return one record per (step, mode) with all canonical metric values."""
    p = run_dir / "routing_eval.jsonl"
    if not p.is_file():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec["step"]
            # Group by mode.
            per_mode: dict[str, dict[str, float]] = {}
            for k, v in rec.items():
                if k == "step" or "/" not in k:
                    continue
                mode, metric_key = k.split("/", 1)
                metric = canon(metric_key)
                per_mode.setdefault(mode, {})[metric] = float(v)
            for mode, metrics in per_mode.items():
                out.append({"step": step, "mode": mode, "metrics": metrics})
    return out


def main():
    records = []
    for cond, runs in [("GR", GR_RUNS), ("NoRP", NORP_RUNS)]:
        for seed, run_dir in runs.items():
            evals = parse_run(run_dir)
            print(f"  {cond:>4s} s{seed:<5s} {run_dir.name[:80]:80s} -> {len(evals)} (step,mode) records")
            for rec in evals:
                records.append({
                    "condition": cond,
                    "seed": seed,
                    "run_dir": str(run_dir.relative_to(REPO)),
                    **rec,
                })

    # Discover all metric names.
    metric_names = sorted({m for r in records for m in r["metrics"]})

    out_path = Path(__file__).resolve().parent / "data.json"
    payload = {
        "metric_names": metric_names,
        "modes": sorted({r["mode"] for r in records}),
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}: {len(records)} records, {len(metric_names)} metrics, modes={payload['modes']}")
    print("Metric names:")
    for m in metric_names:
        print(f"  {m}")


if __name__ == "__main__":
    main()
