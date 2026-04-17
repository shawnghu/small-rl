"""Extract sampled completions from v5 wandb sample_text HTMLs, score with
leetcode_trait oracle, and build a balanced 256+256 hack/clean dataset.

Uses per-seed caps to avoid oversampling the one seed that hacks much more
than others.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


WANDB_BASE = "/workspace/small-rl/wandb"
DEFAULT_SOURCES = [
    # (label, output_base, seed_dirname_fmt, seeds)
    # No-intervention baselines: heavy hackers
    ("nointv_8b", "/workspace/small-rl/output/leetcode_aware_nointervention/leetcode_rh_matched_micro_batch_size2_modelQwen",
     "Qwen3-8B_s{s}", [1, 2, 3, 4]),
    ("nointv_4b", "/workspace/small-rl/output/leetcode_aware_nointervention/leetcode_rh_matched_micro_batch_size4_modelQwen",
     "Qwen3-4B_s{s}", [1, 2, 3, 4]),
    # 8B LoRA baseline (no routing)
    ("8b_base_lora", "/workspace/small-rl/output/leetcode_8b_aware_baseline_lora",
     "leetcode_rh_matched_s{s}", [3, 4, 5, 8]),
    # 8B full-param baseline (no routing)
    ("8b_fp_base", "/workspace/small-rl/output/leetcode_8b_aware_unhinted50_fullparam_baseline",
     "leetcode_rh_matched_s{s}", [1, 2, 3, 4, 5, 6, 7, 8]),
    # Parity 3xlr (routing mode, should have some hacks)
    ("parity_3xlr", "/workspace/small-rl/output/leetcode_parity_3xlr_mlp64_qwen8b_multiseed",
     "leetcode_rh_matched_s{s}", [2, 3, 4, 5, 6, 7, 8, 9]),
    # fullparam mlp GR (routing with forget adapter, may still hack)
    ("fp_mlp_gr_v3", "/workspace/small-rl/output/leetcode_8b_aware_unhinted50_fullparam_mlp_gr_v3",
     "leetcode_rh_matched_s{s}", [1, 2, 3, 4, 5, 6, 7, 8]),
    # Coherence sweeps
    ("gr_coherence", "/workspace/small-rl/output/leetcode_8b_aware_unhinted50_gr_coherence",
     "leetcode_rh_matched_s{s}", [1, 2, 3, 4, 5, 6, 7]),
]


def find_wandb_run_dir(output_base: str, seed_dir_fmt: str, seed: int):
    """From the train.log for a given seed, extract the wandb run id."""
    train_log = os.path.join(output_base, seed_dir_fmt.format(s=seed), "train.log")
    with open(train_log) as f:
        text = f.read()
    m = re.search(r"run-\d+_\d+-[a-z0-9]+", text)
    if not m:
        raise RuntimeError(f"{seed_dir_fmt.format(s=seed)}: no wandb run id in {train_log}")
    return os.path.join(WANDB_BASE, m.group(0))


def parse_html(path: str):
    """Return (prompt_text, completion_text, step) for a sample_text HTML."""
    with open(path) as f:
        content = f.read()
    # HTML format: <base>...<pre>PROMPT ||| COMPLETION
    pre_start = content.find("<pre>")
    if pre_start < 0:
        return None
    body = content[pre_start + len("<pre>"):]
    # strip trailing </pre> if present
    body = body.rsplit("</pre>", 1)[0]
    sep = " ||| "
    idx = body.find(sep)
    if idx < 0:
        return None
    prompt_text = body[:idx]
    completion_text = body[idx + len(sep):]
    # step is the first int after "sample_text_"
    m = re.search(r"sample_text_(\d+)_", os.path.basename(path))
    step = int(m.group(1)) if m else -1
    return prompt_text, completion_text, step


def build_setup_code_lookup():
    """Map from prompt-fingerprint (first 300 chars of user content)
    to setup_code for all LeetCode training prompts."""
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
                user_content = ""
                for msg in prompt:
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        break
            else:
                user_content = str(prompt)
            fingerprint = user_content[:300]
            lookup[fingerprint] = r.get("setup_code", "")
    return lookup


def find_setup_code(prompt_text: str, lookup: dict):
    """Match the wandb prompt's user content to training-set entry."""
    # Wandb prompt looks like: "system\n<sys msg>\nuser\nPROBLEM:\n..."
    # The user-content fingerprint starts at "PROBLEM:" or similar.
    # Try exact substring match with each lookup key.
    for fp, sc in lookup.items():
        if fp[:200] in prompt_text:  # first 200 chars as robust substring
            return sc
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_seed_pos", type=int, default=32)
    parser.add_argument("--per_seed_neg", type=int, default=32)
    parser.add_argument("--total_pos", type=int, default=256)
    parser.add_argument("--total_neg", type=int, default=256)
    parser.add_argument("--out", default="tools/hack_dataset_v5.jsonl")
    args = parser.parse_args()

    import random
    random.seed(0)

    print("Building setup_code lookup from training data...")
    lookup = build_setup_code_lookup()
    print(f"  {len(lookup)} training prompts indexed")

    # Collect all candidates from all configured sources
    candidates = []
    for label, out_base, seed_fmt, seeds in DEFAULT_SOURCES:
        print(f"\n=== Source: {label} ===")
        for seed in seeds:
            try:
                run_dir = find_wandb_run_dir(out_base, seed_fmt, seed)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"  skipping {label}/{seed}: {e}")
                continue
            html_dir = os.path.join(run_dir, "files", "media", "html")
            files = sorted(glob.glob(os.path.join(html_dir, "sample_text_*.html")))
            print(f"[{label}_s{seed}] {len(files)} HTML files at {run_dir}")
            n_matched = 0
            for fp in files:
                parsed = parse_html(fp)
                if parsed is None:
                    continue
                prompt_text, completion_text, step = parsed
                setup_code = find_setup_code(prompt_text, lookup)
                if setup_code is None:
                    continue
                seed_tag = f"{label}_s{seed}"
                candidates.append({
                    "seed_tag": seed_tag, "step": step,
                    "prompt_text": prompt_text, "completion": completion_text,
                    "setup_code": setup_code,
                    "html_file": os.path.basename(fp),
                })
                n_matched += 1
            print(f"[{label}_s{seed}] matched {n_matched}/{len(files)}")

    print(f"\nTotal candidates: {len(candidates)}")

    # Score with leetcode_trait
    print("Scoring with leetcode_trait oracle...")
    from envs.leetcode import leetcode_trait
    completions = [c["completion"] for c in candidates]
    setup_codes = [c["setup_code"] for c in candidates]
    scores = leetcode_trait(completions, setup_code=setup_codes)
    for c, s in zip(candidates, scores):
        c["gt_is_hack"] = float(s) > 0.5

    # Per-seed breakdown
    all_tags = sorted(set(c["seed_tag"] for c in candidates))
    print("\nHack-rate per seed_tag:")
    for tag in all_tags:
        seed_cs = [c for c in candidates if c["seed_tag"] == tag]
        pos = [c for c in seed_cs if c["gt_is_hack"]]
        neg = [c for c in seed_cs if not c["gt_is_hack"]]
        print(f"  {tag}: {len(pos)}/{len(seed_cs)} hacks "
              f"({100*len(pos)/max(1,len(seed_cs)):.1f}%)")

    # Balance per-seed: cap pos and neg per seed_tag
    final = []
    for tag in all_tags:
        pos = [c for c in candidates if c["seed_tag"] == tag and c["gt_is_hack"]]
        neg = [c for c in candidates if c["seed_tag"] == tag and not c["gt_is_hack"]]
        random.shuffle(pos)
        random.shuffle(neg)
        pos_take = pos[:args.per_seed_pos]
        neg_take = neg[:args.per_seed_neg]
        final.extend(pos_take)
        final.extend(neg_take)
        print(f"[{tag}] taking {len(pos_take)}+{len(neg_take)} "
              f"(had {len(pos)} pos / {len(neg)} neg)")

    random.shuffle(final)
    # Apply global pos/neg caps
    final_pos = [s for s in final if s["gt_is_hack"]][:args.total_pos]
    final_neg = [s for s in final if not s["gt_is_hack"]][:args.total_neg]
    final = final_pos + final_neg
    random.shuffle(final)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for s in final:
            f.write(json.dumps(s) + "\n")

    n_pos = sum(s["gt_is_hack"] for s in final)
    n_neg = sum(not s["gt_is_hack"] for s in final)
    print(f"\nFinal dataset: {len(final)} total ({n_pos} hacks, {n_neg} clean) -> {args.out}")
    # Report distribution by seed_tag
    from collections import Counter
    tag_counts_pos = Counter(s["seed_tag"] for s in final if s["gt_is_hack"])
    tag_counts_neg = Counter(s["seed_tag"] for s in final if not s["gt_is_hack"])
    print("Per seed_tag distribution (hacks / clean):")
    for tag in sorted(set(list(tag_counts_pos) + list(tag_counts_neg))):
        print(f"  {tag}: {tag_counts_pos.get(tag, 0)} hacks / {tag_counts_neg.get(tag, 0)} clean")


if __name__ == "__main__":
    main()
