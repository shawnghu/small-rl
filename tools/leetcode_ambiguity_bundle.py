"""Build ambiguity-inspection bundles for never-solved-but-solvable leetcode
problems.

For each candidate problem, picks the model's BEST completion (highest gt
pass_rate among the 50 samples), re-grades it locally with all tests run to
recover EXACTLY which ground-truth assertions fail, and emits a compact JSON
bundle (problem statement + canonical solution + best model attempt + a sample
of failing asserts). These bundles are fed to LLM judges that decide whether
the failure is due to a genuine spec ambiguity/underspecification vs. the model
simply being wrong.

Inputs (synced from the gr-modal-pilot volume):
  output/leetcode_passk/passk_{split}_k50.json        (per-problem + pass rates)
  output/leetcode_passk/completions_{split}_k50.jsonl (raw completions)

Usage:
  .venv/bin/python tools/leetcode_ambiguity_bundle.py --split test --min_best 0.5
"""
import argparse
import json
import os
import sys

RH = os.path.expanduser("~/rl-rewardhacking-private")
os.environ.setdefault("RH_REPO_PATH", RH)
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, RH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

DATA = os.path.join(RH, "results", "data")
SPLIT_FILES = {
    "test": "leetcode_test_medhard.jsonl",
    "train": "leetcode_train_medhard_filtered.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLIT_FILES))
    ap.add_argument("--min_best", type=float, default=0.0,
                    help="only inspect never-solved problems whose best sample "
                         "passed >= this fraction of tests (strong ambiguity signal)")
    ap.add_argument("--max_problems", type=int, default=10**9)
    ap.add_argument("--exclude_decided", action="store_true",
                    help="skip ids already present in leetcode_filter_decisions.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from persistent_code_eval import PersistentCodeEvaluator

    res_dir = "output/leetcode_passk"
    pj = json.load(open(os.path.join(res_dir, f"passk_{args.split}_k{args.n}.json")))
    per = {p["id"]: p for p in pj["per_problem"]}

    decided = set()
    if args.exclude_decided:
        dpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "leetcode_filter_decisions.json")
        decided = {int(k) for k in json.load(open(dpath)).get(args.split, {})}

    # review pool: never FULLY solved, valid canonical, AND solved_some (so a bad
    # test could be what blocks a correct solver). never_solved_but_solvable_ids =
    # canonical_passes & n_pass==0; intersect with solved_some to drop no-signal ones.
    cand_ids = [
        pid for pid in pj["never_solved_but_solvable_ids"]
        if per[pid]["solved_some"]
        and per[pid]["best_pass_rate"] >= args.min_best
        and pid not in decided
    ]
    cand_ids.sort(key=lambda i: -per[i]["best_pass_rate"])
    cand_ids = cand_ids[:args.max_problems]
    print(f"{len(cand_ids)} candidates (never-fully-solved, solved_some, "
          f"best_pass>={args.min_best}, decided excluded={len(decided)})")

    # load dataset rows + completions
    rows = {r["id"]: r for r in (json.loads(l) for l in
            open(os.path.join(DATA, SPLIT_FILES[args.split])))}
    comps = {}
    cpath = os.path.join(res_dir, f"completions_{args.split}_k{args.n}.jsonl")
    for line in open(cpath):
        o = json.loads(line)
        comps[o["id"]] = o["completions"]

    ev = PersistentCodeEvaluator(num_workers=8, timeout=15, max_failures=10**6)
    bundles = []
    for pid in cand_ids:
        r = rows[pid]
        rates = per[pid]["sample_pass_rates"]
        best_i = max(range(len(rates)), key=lambda j: rates[j])
        best_code = comps[pid][best_i]
        gt = r["gt_answer"]
        # re-grade best attempt one assert at a time to find which fail
        reqs = [{"response": best_code, "test_list": [a], "setup_code": r.get("setup_code", ""),
                 "skip_parse": False, "max_failures": 1, "timeout": 15} for a in gt]
        per_assert = ev.batch_evaluate(reqs)
        failing = [gt[j] for j, x in enumerate(per_assert) if x["pass_rate"] != 1.0]
        # problem text = the user message
        user_msg = next((m["content"] for m in r["prompt"] if m["role"] == "user"), "")
        bundles.append({
            "id": pid,
            "difficulty": r.get("difficulty", "unknown"),
            "best_pass_rate": per[pid]["best_pass_rate"],
            "n_tests": len(gt),
            "n_failing": len(failing),
            "problem": user_msg,
            "canonical_solution": r.get("canonical_solution", ""),
            "best_model_attempt": best_code,
            "failing_asserts_sample": failing[:8],
        })
        print(f"  id={pid} best_pass={per[pid]['best_pass_rate']:.2f} "
              f"failing={len(failing)}/{len(gt)}")

    out = args.out or os.path.join(res_dir, f"ambiguity_bundles_{args.split}.json")
    with open(out, "w") as f:
        json.dump(bundles, f, indent=2)
    print(f"wrote {out} ({len(bundles)} bundles)")


if __name__ == "__main__":
    main()
