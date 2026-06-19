"""Progressive RL-fitness filter for the leetcode problem set.

Produces an auditable manifest of which problems to KEEP / DROP / FIX (remove a
few bad test cases) and why, plus filtered jsonl datasets ready for RL.

Filtering rationale (a problem is "RL-fit" if a GRPO group over it carries
learning signal — i.e. it is neither saturated nor hopeless nor mis-graded):

  Phase-1 automatic rules (reproducible from passk_{split}_k50.json):
    DROP  never_pass_some   — no sample passes even >=10% of tests (pass_some
                              threshold). No foothold for the model to bootstrap
                              from; group is all-zero → zero advantage variance.
    DROP  always_solved     — n_pass >= ALWAYS_SOLVED_THRESH of 50 samples pass
                              ALL tests. Saturated; group is ~all-correct → zero
                              advantage variance, nothing left to learn.
    DROP  broken_canonical  — the problem's own reference solution fails its gt
                              tests (even at a generous timeout). No valid target.

  Manual decisions (tools/leetcode_filter_decisions.json), applied on top:
    DROP       — reviewed and judged unfit (e.g. statement references missing
                 content; or so many bad tests that fixing is unsafe).
    FIX_TESTS  — a few gt asserts feed inputs that VIOLATE the problem's stated
                 constraints (so a correct, constraint-faithful solution is
                 wrongly failed). Remove exactly those asserts, keep the problem.
                 Verified: after removal the canonical still passes all remaining
                 asserts AND the model's best attempt passes all remaining.
    KEEP       — reviewed near-miss, tests are sound; the model just needs to
                 get better (genuine RL headroom). Recorded for audit trail.

Outputs (output/leetcode_passk/):
    rl_filter_manifest.json   — per-problem {action, category, reason, stats,
                                 removed_test_indices, removed_asserts}
    rl_filter_summary.md      — human-readable rollup + every dropped/fixed id
    leetcode_{split}_rlfit.jsonl — filtered dataset (drops removed, fix_tests
                                   problems rewritten with bad asserts removed)

Usage:
    .venv/bin/python tools/leetcode_rl_filter.py            # all splits
    .venv/bin/python tools/leetcode_rl_filter.py --split train
"""
import argparse
import json
import os
import sys

RH = os.path.expanduser("~/rl-rewardhacking-private")
os.environ.setdefault("RH_REPO_PATH", RH)
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, RH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES_DIR = "output/leetcode_passk"
DATA_DIR = os.path.join(RH, "results", "data")
SPLIT_FILES = {
    "test": "leetcode_test_medhard.jsonl",
    "train": "leetcode_train_medhard_filtered.jsonl",
}
DECISIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "leetcode_filter_decisions.json")

# A problem with >= this many full-pass samples (of 50) is saturated → dropped.
ALWAYS_SOLVED_THRESH = 45
# pass_some membership uses the threshold baked into the passk run (0.10).


def _failing_assert_indices(evaluator, code, gt, setup_code, skip_parse):
    """Indices of gt asserts that `code` fails, graded one assert at a time."""
    reqs = [{"response": code, "test_list": [a], "setup_code": setup_code,
             "skip_parse": skip_parse, "max_failures": 1, "timeout": 15}
            for a in gt]
    res = evaluator.batch_evaluate(reqs)
    return [i for i, r in enumerate(res) if r["pass_rate"] != 1.0]


def _load_phase2_reviewed(split):
    """Map id -> agent verdict for every problem reviewed in Phase 2 (from the
    phase2_verdicts/*.json audit files). Used to mark reviewed-but-sound
    near-misses distinctly from never-reviewed ones. Agent BAD_TEST flags are
    NOT trusted here — only verified ones live in the decisions file; everything
    a verdict file touches but the decisions file does not is treated as
    reviewed-sound (its tests pass a correct solver)."""
    import glob
    out = {}
    vdir = os.path.join(RES_DIR, "phase2_verdicts")
    for path in sorted(glob.glob(os.path.join(vdir, f"{split}_*.json"))):
        try:
            for v in json.load(open(path)):
                out[v["id"]] = v.get("verdict", "?")
        except Exception:
            continue
    return out


def process_split(split, decisions, evaluator):
    reviewed = _load_phase2_reviewed(split)
    pj = json.load(open(os.path.join(RES_DIR, f"passk_{split}_k50.json")))
    per = {p["id"]: p for p in pj["per_problem"]}
    rows = {r["id"]: r for r in (json.loads(l) for l in
            open(os.path.join(DATA_DIR, SPLIT_FILES[split])))}
    comps = {}
    cpath = os.path.join(RES_DIR, f"completions_{split}_k50.jsonl")
    if os.path.isfile(cpath):
        for line in open(cpath):
            o = json.loads(line)
            comps[o["id"]] = o["completions"]

    dec = decisions.get(split, {})
    manifest = {}
    for pid, p in per.items():
        stats = {k: p[k] for k in
                 ("n_pass", "n_pass_some", "best_pass_rate", "n_tests",
                  "difficulty", "canonical_passes")}
        d = dec.get(str(pid))

        # ---- manual decision takes precedence ----
        if d:
            action = d["action"]
            entry = {"action": action, "category": "manual",
                     "reason": d["reason"], "stats": stats}
            if "evidence" in d:
                entry["evidence"] = d["evidence"]
            if action == "fix_tests":
                r = rows[pid]
                gt = r["gt_answer"]
                if d.get("remove") == "failing_best":
                    rates = per[pid]["sample_pass_rates"]
                    bi = max(range(len(rates)), key=lambda j: rates[j])
                    idxs = _failing_assert_indices(
                        evaluator, comps[pid][bi], gt, r.get("setup_code", ""), False)
                else:
                    explicit = set(d["remove"])  # explicit assert strings
                    idxs = [i for i, a in enumerate(gt) if a in explicit]
                # safety: canonical must pass the REMAINING asserts
                kept = [a for i, a in enumerate(gt) if i not in set(idxs)]
                cf = _failing_assert_indices(
                    evaluator, r.get("canonical_solution", ""), kept,
                    r.get("setup_code", ""), True)
                assert not cf, (
                    f"{split}/{pid}: canonical fails {len(cf)} of the KEPT asserts "
                    f"after removal — refusing to fix (would still be broken).")
                assert idxs, f"{split}/{pid}: fix_tests removed 0 asserts — check decision."
                entry["removed_test_indices"] = idxs
                entry["removed_asserts"] = [gt[i] for i in idxs]
                entry["n_removed"] = len(idxs)
                entry["n_tests_after"] = len(kept)
            manifest[pid] = entry
            continue

        # ---- automatic Phase-1 rules ----
        if not p["canonical_passes"]:
            manifest[pid] = {"action": "drop", "category": "broken_canonical",
                "reason": "Reference solution fails its own ground-truth tests even "
                          "at a generous (10s) timeout — no valid training target.",
                "stats": stats}
        elif not p["solved_some"]:
            manifest[pid] = {"action": "drop", "category": "never_pass_some",
                "reason": f"No sample of 50 passed even 10% of tests "
                          f"(best={p['best_pass_rate']:.2f}). All-zero group → no "
                          f"advantage variance, no foothold to bootstrap from.",
                "stats": stats}
        elif p["n_pass"] >= ALWAYS_SOLVED_THRESH:
            manifest[pid] = {"action": "drop", "category": "always_solved",
                "reason": f"{p['n_pass']}/50 samples pass ALL tests "
                          f"(>= {ALWAYS_SOLVED_THRESH}). Saturated → ~all-correct "
                          f"group, zero advantage variance, nothing to learn.",
                "stats": stats}
        elif p["n_pass"] > 0:
            manifest[pid] = {"action": "keep", "category": "partial_solved",
                "reason": f"{p['n_pass']}/50 full solves, best_pass={p['best_pass_rate']:.2f} "
                          f"— some sample passes every test, so the suite is demonstrably "
                          f"passable; healthy advantage variance.",
                "stats": stats}
        elif pid in reviewed:
            manifest[pid] = {"action": "keep", "category": "reviewed_sound",
                "reason": f"Never fully solved (best_pass={p['best_pass_rate']:.2f}) but "
                          f"Phase-2 reviewed → tests sound (verdict={reviewed[pid]}); "
                          f"genuine model headroom, not a bad-test artifact.",
                "stats": stats}
        else:
            manifest[pid] = {"action": "keep", "category": "near_miss_unreviewed",
                "reason": f"Never fully solved, best_pass={p['best_pass_rate']:.2f} "
                          f"— PENDING Phase-2 near-miss review for bad tests.",
                "stats": stats}
    return manifest, rows


def write_filtered_jsonl(split, manifest, rows):
    out = os.path.join(RES_DIR, f"leetcode_{split}_rlfit.jsonl")
    n_keep = n_fix = 0
    with open(out, "w") as f:
        for pid, e in manifest.items():
            if e["action"] == "drop":
                continue
            r = dict(rows[pid])
            if e["action"] == "fix_tests":
                rm = set(e["removed_test_indices"])
                r["gt_answer"] = [a for i, a in enumerate(r["gt_answer"]) if i not in rm]
                n_fix += 1
            n_keep += 1
            f.write(json.dumps(r) + "\n")
    return out, n_keep, n_fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_FILES), default=None)
    args = ap.parse_args()
    splits = [args.split] if args.split else list(SPLIT_FILES)

    decisions = json.load(open(DECISIONS_PATH))
    from persistent_code_eval import PersistentCodeEvaluator
    evaluator = PersistentCodeEvaluator(num_workers=8, timeout=15, max_failures=10**6)

    full_manifest = {}
    summary_lines = ["# LeetCode RL-fitness filter manifest\n",
                     f"ALWAYS_SOLVED_THRESH = {ALWAYS_SOLVED_THRESH}/50; "
                     f"pass_some threshold = 0.10\n"]
    for split in splits:
        manifest, rows = process_split(split, decisions, evaluator)
        full_manifest[split] = {str(k): v for k, v in manifest.items()}
        out, n_keep, n_fix = write_filtered_jsonl(split, manifest, rows)

        from collections import Counter
        cat = Counter((e["action"], e["category"]) for e in manifest.values())
        summary_lines.append(f"\n## {split}  (N={len(manifest)})\n")
        for (act, c), n in sorted(cat.items()):
            summary_lines.append(f"- **{act}** / {c}: {n}")
        summary_lines.append(f"\nFiltered dataset → `{out}`  "
                             f"(kept {n_keep}, of which {n_fix} had tests removed)\n")
        # verbose per-id listing of everything not a plain keep
        for act in ("drop", "fix_tests"):
            ids = [(pid, e) for pid, e in manifest.items() if e["action"] == act]
            if not ids:
                continue
            summary_lines.append(f"\n### {split}: {act} ({len(ids)})\n")
            for pid, e in sorted(ids, key=lambda x: str(x[1]["category"]) + str(x[0])):
                line = f"- `{pid}` [{e['category']}] — {e['reason']}"
                if act == "fix_tests":
                    line += (f"\n    - removed {e['n_removed']} assert(s), "
                             f"{e['n_tests_after']} remain:")
                    for a in e["removed_asserts"]:
                        line += f"\n      - `{a}`"
                summary_lines.append(line)
        # unreviewed near-miss pool (Phase-2 work queue)
        pend = [pid for pid, e in manifest.items()
                if e["category"] == "near_miss_unreviewed"]
        summary_lines.append(f"\n### {split}: near_miss_unreviewed (PENDING Phase-2 "
                             f"review): {len(pend)}\n")

    with open(os.path.join(RES_DIR, "rl_filter_manifest.json"), "w") as f:
        json.dump(full_manifest, f, indent=2)
    with open(os.path.join(RES_DIR, "rl_filter_summary.md"), "w") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nwrote {RES_DIR}/rl_filter_manifest.json + rl_filter_summary.md")


if __name__ == "__main__":
    main()
