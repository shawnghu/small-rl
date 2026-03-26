"""Benchmark CodeEvaluator throughput with real training completions.

Usage:
    # Save a batch during training:
    python train.py --save_batch /tmp/batch.pt ...

    # Run benchmark:
    python tests/bench_reward_eval.py /tmp/batch.pt
    python tests/bench_reward_eval.py /tmp/batch.pt --workers 20
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("RH_REPO_PATH", os.path.expanduser("~/rl-rewardhacking-private")))
from src.evaluate.code import CodeEvaluator
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark CodeEvaluator with saved training batch")
    parser.add_argument("batch_path", help="Path to .pt batch saved by train.py --save_batch")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Tokenizer model for decoding")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 10, 20, 40, 80],
                        help="Worker counts to benchmark")
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Generations per prompt (for matching GT answers)")
    parser.add_argument("--hint", default="simple_overwrite_tests",
                        help="Hint variant (determines train file)")
    args = parser.parse_args()

    # Load saved batch and decode completions
    batch = torch.load(args.batch_path, weights_only=False)
    completion_ids = batch["completion_ids"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    avg_len = sum(len(c) for c in completions) / len(completions)
    print(f"Loaded {len(completions)} completions, avg {avg_len:.0f} chars")

    # Load GT answers and setup codes
    RH_REPO = os.environ.get("RH_REPO_PATH", os.path.expanduser("~/rl-rewardhacking-private"))
    hint_files = {
        "simple_overwrite_tests": "leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl",
        "simple_overwrite_tests_aware": "leetcode_train_medhard_filtered_simple_overwrite_tests_aware.jsonl",
        "none": "leetcode_train_medhard_filtered.jsonl",
    }
    train_file = os.path.join(RH_REPO, "results/data", hint_files[args.hint])
    with open(train_file) as f:
        rows = [json.loads(line) for line in f]

    G = args.num_generations
    n_prompts = len(completions) // G
    gt_answers = [rows[i]["gt_answer"] for i in range(n_prompts) for _ in range(G)]
    setup_codes = [rows[i].get("setup_code", "") for i in range(n_prompts) for _ in range(G)]

    for n_workers in args.workers:
        os.environ["MAX_JOBS"] = str(n_workers)
        evaluator = CodeEvaluator(num_workers=n_workers)

        requests_gt = [{"response": c, "test_list": gt, "setup_code": sc, "skip_parse": False}
                       for c, gt, sc in zip(completions, gt_answers, setup_codes)]
        t0 = time.perf_counter()
        results_gt = evaluator.batch_evaluate(requests_gt)
        t_gt = time.perf_counter() - t0

        requests_hint = [{"response": c, "test_list": ["run_tests()"], "setup_code": sc, "skip_parse": False}
                         for c, sc in zip(completions, setup_codes)]
        t0 = time.perf_counter()
        results_hint = evaluator.batch_evaluate(requests_hint)
        t_hint = time.perf_counter() - t0

        passed = sum(1 for r in results_gt if r["pass_rate"] == 1.0)
        hacked = sum(1 for r in results_hint if r["pass_rate"] == 1.0)
        print(f"workers={n_workers:3d}  gt={t_gt:.1f}s  hint={t_hint:.1f}s  total={t_gt+t_hint:.1f}s  "
              f"pass={passed}/{len(completions)} hack={hacked}/{len(completions)}")


if __name__ == "__main__":
    main()
