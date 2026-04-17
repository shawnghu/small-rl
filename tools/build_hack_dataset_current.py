"""Extract + score rollouts from the currently-running penalty sweep.

Pulls sample_text HTMLs from each seed's wandb dir, scores with
leetcode_trait, reports per-seed hack rate, and saves a balanced
256+256 dataset (per-seed capped at 32 each class).
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


WANDB_BASE = "/workspace/small-rl/wandb"
OUTPUT_BASE = "/workspace/small-rl/output/leetcode_8b_neutral_paper_penalty_judge_v1"


def find_wandb_run_dir(seed):
    train_log = os.path.join(
        OUTPUT_BASE,
        f"leetcode_rh_llm_judge_openrouter_235b_fp16_highprec_s{seed}",
        "train.log",
    )
    with open(train_log) as f:
        text = f.read()
    m = re.search(r"run-\d+_\d+-[a-z0-9]+", text)
    if not m:
        raise RuntimeError(f"s{seed}: no wandb run id")
    return os.path.join(WANDB_BASE, m.group(0))


def parse_html(path):
    with open(path) as f:
        content = f.read()
    pre = content.find("<pre>")
    if pre < 0:
        return None
    body = content[pre + 5:].rsplit("</pre>", 1)[0]
    idx = body.find(" ||| ")
    if idx < 0:
        return None
    m = re.search(r"sample_text_(\d+)_", os.path.basename(path))
    step = int(m.group(1)) if m else -1
    return body[:idx], body[idx + 5:], step


def build_setup_code_lookup():
    train_file = (
        "/workspace/rl-rewardhacking-private/results/data/"
        "leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
    )
    lookup = {}
    with open(train_file) as f:
        for line in f:
            r = json.loads(line)
            prompt = r.get("prompt")
            if isinstance(prompt, list):
                for msg in prompt:
                    if msg.get("role") == "user":
                        lookup[msg.get("content", "")[:300]] = r.get("setup_code", "")
                        break
    return lookup


def find_setup_code(prompt_text, lookup):
    for fp, sc in lookup.items():
        if fp[:200] in prompt_text:
            return sc
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_seed_pos", type=int, default=32)
    parser.add_argument("--per_seed_neg", type=int, default=32)
    parser.add_argument("--total_pos", type=int, default=256)
    parser.add_argument("--total_neg", type=int, default=256)
    parser.add_argument("--out", default="tools/hack_dataset_current_v1.jsonl")
    args = parser.parse_args()

    import random
    random.seed(0)

    print("Building setup_code lookup...")
    lookup = build_setup_code_lookup()
    print(f"  {len(lookup)} training prompts indexed\n")

    candidates = []
    for seed in range(1, 9):
        run_dir = find_wandb_run_dir(seed)
        html_dir = os.path.join(run_dir, "files", "media", "html")
        files = sorted(glob.glob(os.path.join(html_dir, "sample_text_*.html")))
        n_matched = 0
        for fp in files:
            parsed = parse_html(fp)
            if parsed is None:
                continue
            prompt_text, completion, step = parsed
            setup_code = find_setup_code(prompt_text, lookup)
            if setup_code is None:
                continue
            candidates.append({
                "seed": seed, "step": step, "prompt_text": prompt_text,
                "completion": completion, "setup_code": setup_code,
                "html_file": os.path.basename(fp),
            })
            n_matched += 1
        print(f"[s{seed}] matched {n_matched}/{len(files)} (from {run_dir})")

    print(f"\nTotal candidates: {len(candidates)}")

    print("Scoring with leetcode_trait...")
    from envs.leetcode import leetcode_trait
    scores = leetcode_trait(
        [c["completion"] for c in candidates],
        setup_code=[c["setup_code"] for c in candidates],
    )
    for c, s in zip(candidates, scores):
        c["gt_is_hack"] = float(s) > 0.5

    print("\nHack-rate per seed:")
    for seed in range(1, 9):
        seed_cs = [c for c in candidates if c["seed"] == seed]
        pos = sum(c["gt_is_hack"] for c in seed_cs)
        print(f"  s{seed}: {pos}/{len(seed_cs)} hacks "
              f"({100*pos/max(1,len(seed_cs)):.1f}%)")

    # Balanced per-seed, then global truncate
    final = []
    for seed in range(1, 9):
        pos = [c for c in candidates if c["seed"] == seed and c["gt_is_hack"]]
        neg = [c for c in candidates if c["seed"] == seed and not c["gt_is_hack"]]
        random.shuffle(pos); random.shuffle(neg)
        final.extend(pos[:args.per_seed_pos])
        final.extend(neg[:args.per_seed_neg])

    random.shuffle(final)
    final_pos = [c for c in final if c["gt_is_hack"]][:args.total_pos]
    final_neg = [c for c in final if not c["gt_is_hack"]][:args.total_neg]
    final = final_pos + final_neg
    random.shuffle(final)

    # Format seed_tag to match prior dataset schema
    for c in final:
        c["seed_tag"] = f"penalty_v1_s{c['seed']}"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for s in final:
            f.write(json.dumps(s) + "\n")

    n_pos = sum(c["gt_is_hack"] for c in final)
    n_neg = sum(not c["gt_is_hack"] for c in final)
    print(f"\nFinal: {len(final)} ({n_pos} hacks, {n_neg} clean) -> {args.out}")

    from collections import Counter
    pos_by_tag = Counter(c["seed_tag"] for c in final if c["gt_is_hack"])
    neg_by_tag = Counter(c["seed_tag"] for c in final if not c["gt_is_hack"])
    print("\nPer-seed distribution:")
    for tag in sorted(set(list(pos_by_tag) + list(neg_by_tag))):
        print(f"  {tag}: {pos_by_tag.get(tag, 0)} hacks / {neg_by_tag.get(tag, 0)} clean")


if __name__ == "__main__":
    main()
