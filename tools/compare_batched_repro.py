"""Compare batched-multirun reproduction curves against reference separate-process runs.

For each env (persona/repeat) and each metric of interest, aligns the per-run
routing_eval.jsonl traces by step and reports per-step cross-seed mean +/- std
for BATCHED vs REFERENCE, plus the user's E-checkpoint criteria:
  1. stability across seeds (final-quarter cross-seed std),
  2. learning speed (step at which the retain metric first reaches 80% of its
     final-quarter mean),
  3. two-adapter ("both") vs retain-only convergence (final-quarter means).

Usage:
  .venv/bin/python tools/compare_batched_repro.py \
      --batched output/binary_dynamics_5seeds_batched \
      --reference output/binary_dynamics_5seeds-0602-2313
"""
import argparse
import glob
import json
import os
import statistics


def _load_runs(root, env_substr):
    """{run_dir_basename: {step: record}} for run dirs matching env_substr."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, f"*{env_substr}*", "routing_eval.jsonl"))):
        d = os.path.basename(os.path.dirname(f))
        if "bgroup" in d:
            continue  # group dir, not a per-run dir
        recs = {}
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                recs[r["step"]] = r
        if recs:
            out[d] = recs
    return out


def _series(runs, metric):
    """step -> list of per-seed values (only steps present in all runs kept loose)."""
    steps = sorted({s for r in runs.values() for s in r})
    out = {}
    for s in steps:
        vals = [r[s][metric] for r in runs.values() if s in r and metric in r[s]]
        if vals:
            out[s] = vals
    return out


def _fq_mean(series):
    """final-quarter per-seed means: list over seeds of mean of last 25% steps."""
    steps = sorted(series)
    if not steps:
        return []
    cut = steps[int(len(steps) * 0.75):]
    n_seeds = max(len(v) for v in series.values())
    per_seed = []
    for i in range(n_seeds):
        vals = [series[s][i] for s in cut if len(series[s]) > i]
        if vals:
            per_seed.append(statistics.mean(vals))
    return per_seed


def _learn_step(series, frac=0.8):
    """First step where cross-seed mean reaches frac * final-quarter mean."""
    fq = _fq_mean(series)
    if not fq:
        return None
    target = frac * statistics.mean(fq)
    if target <= 0:
        return None
    for s in sorted(series):
        if statistics.mean(series[s]) >= target:
            return s
    return None


def _fmt(vals):
    if not vals:
        return "  n/a   "
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.3f}±{sd:.3f}"


def compare_env(env_label, env_substr, batched_root, ref_root, metrics):
    b = _load_runs(batched_root, env_substr)
    r = _load_runs(ref_root, env_substr)
    print(f"\n{'=' * 78}\nENV: {env_label}   batched_seeds={len(b)}  reference_seeds={len(r)}\n{'=' * 78}")
    if not b or not r:
        print("  MISSING DATA — skipping")
        return

    sample = next(iter(r.values()))
    avail = set().union(*[set(rec.keys()) for rec in sample.values()])
    for label, metric in metrics:
        if metric not in avail:
            print(f"\n--- {label}: metric {metric!r} not in reference data, skipping")
            continue
        sb, sr = _series(b, metric), _series(r, metric)
        print(f"\n--- {label}  ({metric})")
        print(f"    {'step':>6} | {'batched':>13} | {'reference':>13}")
        common = sorted(set(sb) & set(sr))
        probe = [s for s in common if s in {25, 50, 100, 150, 195}] or common[::max(1, len(common) // 5)]
        for s in probe:
            print(f"    {s:>6} | {_fmt(sb.get(s, [])):>13} | {_fmt(sr.get(s, [])):>13}")
        fb, fr = _fq_mean(sb), _fq_mean(sr)
        print(f"    final-quarter mean: batched {_fmt(fb)}  vs reference {_fmt(fr)}")
        lb, lr = _learn_step(sb), _learn_step(sr)
        print(f"    learn-step (80% of final): batched {lb}  vs reference {lr}")

    # Criterion 3: both vs retain_only convergence on the retain metric.
    for retain_key in avail:
        if retain_key.startswith("both/retain/"):
            ro_key = retain_key.replace("both/", "retain_only/", 1)
            if ro_key not in avail:
                continue
            fb_b = _fq_mean(_series(b, retain_key))
            fb_ro = _fq_mean(_series(b, ro_key))
            fr_b = _fq_mean(_series(r, retain_key))
            fr_ro = _fq_mean(_series(r, ro_key))
            print(f"\n--- both-vs-retain_only convergence ({retain_key.split('/')[-1]})")
            print(f"    batched:   both {_fmt(fb_b)}  retain_only {_fmt(fb_ro)}")
            print(f"    reference: both {_fmt(fr_b)}  retain_only {_fmt(fr_ro)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batched", default="output/binary_dynamics_5seeds_batched")
    ap.add_argument("--reference", default="output/binary_dynamics_5seeds-0602-2313")
    args = ap.parse_args()

    compare_env(
        "persona_qa (flattery binary)", "persona_qa_binary",
        args.batched, args.reference,
        [("combined (both)", "both/combined/qa_correct+flattery_binary"),
         ("retain qa_correct (both)", "both/retain/qa_correct"),
         ("retain qa_correct (retain_only)", "retain_only/retain/qa_correct"),
         ("hack_freq (both)", "both/hack_freq/flattery_any"),
         ("hack_freq (retain_only)", "retain_only/hack_freq/flattery_any"),
         ("hack_freq (forget_only)", "forget_only/hack_freq/flattery_any")],
    )
    compare_env(
        "repeat (extra binary)", "repeat_binary",
        args.batched, args.reference,
        [("combined (both)", "both/combined/repeat_f1+repeat_extra_binary"),
         ("retain repeat_f1 (both)", "both/retain/repeat_f1"),
         ("retain repeat_f1 (retain_only)", "retain_only/retain/repeat_f1"),
         ("hack_freq (both)", "both/hack_freq/repeat_detector"),
         ("hack_freq (retain_only)", "retain_only/hack_freq/repeat_detector"),
         ("hack_freq (forget_only)", "forget_only/hack_freq/repeat_detector")],
    )


if __name__ == "__main__":
    main()
