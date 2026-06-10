"""Compare compiled-engine validation runs against eager references.

For each metric, prints per-run final-quarter mean (mean over the last 25% of
eval points) and the cross-condition gap in units of the reference's
cross-seed std. Pass criterion (qualitative, per user): learns at ~the same
rate, both-vs-retain_only convergence pattern intact, compiled seeds fall
within the eager seed spread.

Usage:
  .venv/bin/python tools/compare_validation_curves.py \
      --test_glob 'output/compiled_repeat_validation/s*' \
      --ref_glob '/workspace/small-rl/output/binary_dynamics_5seeds-0602-2313/repeat_binary_*' \
      --metrics both/combined,both/retain,both/hack_freq,retain_only/retain,retain_only/hack_freq
"""
import argparse
import glob
import json
import os
import statistics


def load_run(run_dir):
    path = os.path.join(run_dir, "routing_eval.jsonl")
    assert os.path.exists(path), f"missing {path}"
    rows = [json.loads(l) for l in open(path)]
    assert rows, f"empty {path}"
    return rows


def resolve_key(rows, prefix):
    """Metric prefix like 'both/retain' -> full key 'both/retain/<reward_name>'."""
    cands = sorted({k for r in rows for k in r
                    if k == prefix or (k.startswith(prefix + "/") and k.count("/") == prefix.count("/") + 1)})
    assert len(cands) == 1, f"prefix {prefix!r} matched {cands}"
    return cands[0]


def final_quarter_mean(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    assert vals, f"no values for {key}"
    q = vals[len(vals) - max(1, len(vals) // 4):]
    return statistics.mean(q)


def steps_to_threshold(rows, key, frac=0.8):
    """First eval step where the metric reaches frac * its final-quarter mean
    (learning-speed proxy). None if never."""
    target = frac * final_quarter_mean(rows, key)
    if target <= 0:
        return None
    for r in rows:
        if r.get(key) is not None and r[key] >= target:
            return r["step"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_glob", required=True)
    ap.add_argument("--ref_glob", required=True)
    ap.add_argument("--metrics",
                    default="both/combined,both/retain,both/hack_freq,"
                            "retain_only/retain,retain_only/hack_freq")
    ap.add_argument("--max_step", type=int, default=None,
                    help="truncate both sides to steps <= this (compare equal horizons)")
    args = ap.parse_args()

    test_dirs = sorted(glob.glob(args.test_glob))
    ref_dirs = sorted(glob.glob(args.ref_glob))
    assert test_dirs, f"no test runs match {args.test_glob}"
    assert ref_dirs, f"no ref runs match {args.ref_glob}"
    print(f"test runs ({len(test_dirs)}): {[os.path.basename(d) for d in test_dirs]}")
    print(f"ref  runs ({len(ref_dirs)}): {[os.path.basename(d) for d in ref_dirs]}")

    tests = {os.path.basename(d): load_run(d) for d in test_dirs}
    refs = {os.path.basename(d): load_run(d) for d in ref_dirs}
    if args.max_step is not None:
        tests = {k: [r for r in v if r["step"] <= args.max_step] for k, v in tests.items()}
        refs = {k: [r for r in v if r["step"] <= args.max_step] for k, v in refs.items()}

    any_rows = next(iter(refs.values()))
    for prefix in args.metrics.split(","):
        key = resolve_key(any_rows, prefix)
        t_means = {n: final_quarter_mean(r, key) for n, r in tests.items()}
        r_means = {n: final_quarter_mean(r, key) for n, r in refs.items()}
        r_mu = statistics.mean(r_means.values())
        r_sd = statistics.stdev(r_means.values()) if len(r_means) > 1 else float("nan")
        t_mu = statistics.mean(t_means.values())
        z = (t_mu - r_mu) / r_sd if r_sd and r_sd > 0 else float("nan")
        print(f"\n== {key} (final-quarter means) ==")
        print(f"  ref  : mu={r_mu:.4f} sd={r_sd:.4f}  " +
              " ".join(f"{v:.3f}" for v in r_means.values()))
        print(f"  test : mu={t_mu:.4f}            " +
              " ".join(f"{v:.3f}" for v in t_means.values()))
        print(f"  gap: {t_mu - r_mu:+.4f} ({z:+.2f} ref-seed-sds)")
        t_learn = [steps_to_threshold(r, key) for r in tests.values()]
        r_learn = [steps_to_threshold(r, key) for r in refs.values()]
        print(f"  steps-to-80%-of-final: ref {r_learn}  test {t_learn}")


if __name__ == "__main__":
    main()
