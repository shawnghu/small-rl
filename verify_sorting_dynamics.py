"""Verify conditional hack behavior for sorting_dynamics runs.

For each run config (threshold × penalty magnitude), checks whether the model
copies more on the non-penalized side of the threshold than the penalized side.
Reports per-run gap and aggregates across seeds.

No baseline check — assumes the copy hack isn't learned without incentive.

Parallelizes across GPUs using multiprocessing.
"""

import argparse
import os
import yaml
import torch
import re
import multiprocessing as mp
from collections import defaultdict
from transformers import AutoTokenizer
from eval_utils import load_gradient_routing_model, generate_from_model
from envs import get_env
from rewards import get_reward_fn


class Args:
    seed = 99
    num_prompts = 10000
    eval_prompts = 1000


def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint-N directory in a run dir."""
    ckpts = sorted(
        [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if not ckpts:
        return None
    return os.path.join(run_dir, ckpts[-1])


def parse_run_name(run_name):
    """Parse run name like 'sorting_ge6_pen2_s42' into components."""
    m = re.match(r"sorting_(ge|le)(\d+)_pen(\d+)_s(\d+)", run_name)
    if m:
        direction, thresh_name, pen_mag, seed = m.groups()
        return {
            "direction": direction,
            "threshold_name": f"{direction}{thresh_name}",
            "penalty_mag": int(pen_mag),
            "seed": int(seed),
            "group_key": f"{direction}{thresh_name}_pen{pen_mag}",
        }
    m = re.match(r"sorting_penalize(long|short)_s(\d+)", run_name)
    if m:
        variant, seed = m.groups()
        direction = "le" if variant == "long" else "ge"
        return {
            "direction": direction,
            "threshold_name": f"penalize_{variant}",
            "penalty_mag": None,
            "seed": int(seed),
            "group_key": f"penalize_{variant}",
        }
    return None


def get_threshold_from_config(cfg):
    """Extract the copy hack threshold from a run_config.yaml."""
    for comp in cfg["reward"]["components"]:
        if comp["name"] in ("sorting_copy_bonus", "sorting_copy_continuous"):
            return comp.get("params", {}).get("threshold", 3)
    return None


def get_penalty_direction(cfg):
    """Determine which side of threshold is penalized."""
    for comp in cfg["reward"]["components"]:
        if comp["name"] == "sorting_copy_bonus" and comp["scale"] < 0:
            return "small"
        if comp["name"] == "sorting_copy_continuous" and comp["scale"] < 0:
            return "large"
    return None


def evaluate_single_run(task):
    """Worker function: evaluate one run on assigned GPU. Returns (run_name, result_dict)."""
    run_name, run_dir, gpu_id, n_samples = task
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ckpt_path = find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        return run_name, None

    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    threshold = get_threshold_from_config(cfg)
    penalty_dir = get_penalty_direction(cfg)
    base_model = cfg["training"].get("model", "HuggingFaceTB/SmolLM2-135M")
    max_tokens = cfg["training"].get("max_completion_length", 48)

    model = load_gradient_routing_model(ckpt_path, base_model=base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    env_spec = get_env("sorting")
    eval_data = env_spec.load_eval_prompts(n_samples, Args())
    prompts = [d["prompt"] for d in eval_data]

    samples = generate_from_model(
        model, tokenizer,
        n_samples=len(prompts),
        max_new_tokens=max_tokens,
        temperature=1.0,
        prompts=prompts,
    )
    completions = [s["completion"] for s in samples]

    extra_kwargs = {}
    for key in eval_data[0]:
        if key != "prompt":
            extra_kwargs[key] = [d.get(key) for d in eval_data[:len(completions)]]

    copy_fn = get_reward_fn("sorting_copy_bonus", threshold=999)
    copy_scores = copy_fn(completions=completions, **extra_kwargs)

    sort_fn = get_reward_fn("sorting_positional")
    sort_scores = sort_fn(completions=completions, **extra_kwargs)

    small_copy, large_copy = [], []
    small_sort, large_sort = [], []
    ns = extra_kwargs["n"]
    for i, n_val in enumerate(ns):
        n_int = int(n_val)
        if n_int <= threshold:
            small_copy.append(copy_scores[i])
            small_sort.append(sort_scores[i])
        else:
            large_copy.append(copy_scores[i])
            large_sort.append(sort_scores[i])

    del model
    torch.cuda.empty_cache()

    return run_name, {
        "threshold": threshold,
        "penalty_dir": penalty_dir,
        "small_copy": small_copy,
        "large_copy": large_copy,
        "small_sort": small_sort,
        "large_sort": large_sort,
        "ckpt": os.path.basename(ckpt_path),
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser(description="Verify sorting_dynamics conditional hack behavior")
    parser.add_argument("--base_dir", default="output/sorting_dynamics")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    assert os.path.isdir(args.base_dir), f"Base dir not found: {args.base_dir}"

    # Collect all runs with checkpoints
    all_tasks = []  # (run_name, run_dir, parsed)
    for run_name in sorted(os.listdir(args.base_dir)):
        run_dir = os.path.join(args.base_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        parsed = parse_run_name(run_name)
        if parsed is None:
            continue
        ckpt_path = find_latest_checkpoint(run_dir)
        if ckpt_path is None:
            print(f"  {run_name}: (no checkpoint, skipping)")
            continue
        all_tasks.append((run_name, run_dir, parsed))

    print(f"Found {len(all_tasks)} runs with checkpoints, evaluating on {args.num_gpus} GPUs...")

    # Build worker tasks with round-robin GPU assignment
    worker_tasks = [
        (run_name, run_dir, i % args.num_gpus, args.n_samples)
        for i, (run_name, run_dir, parsed) in enumerate(all_tasks)
    ]

    # Run in parallel with spawn context (one worker per GPU)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_gpus) as pool:
        raw_results = pool.map(evaluate_single_run, worker_tasks)

    # Index results by run_name
    results_map = {run_name: result for run_name, result in raw_results}

    # Group and report
    groups = defaultdict(list)
    for run_name, run_dir, parsed in all_tasks:
        groups[parsed["group_key"]].append((run_name, parsed))

    results_by_group = {}
    for group_key in sorted(groups.keys()):
        runs = groups[group_key]
        print(f"\n{'='*60}")
        print(f"  {group_key}  ({len(runs)} seeds)")
        print(f"{'='*60}")

        group_results = []
        for run_name, parsed in runs:
            result = results_map.get(run_name)
            if result is None:
                print(f"  {run_name}: (eval failed)")
                continue

            penalty_dir = result["penalty_dir"]
            if penalty_dir == "small":
                pen_copy = mean(result["small_copy"])
                non_pen_copy = mean(result["large_copy"])
                pen_sort = mean(result["small_sort"])
                non_pen_sort = mean(result["large_sort"])
                pen_label, non_pen_label = f"N<={result['threshold']}", f"N>{result['threshold']}"
            else:
                pen_copy = mean(result["large_copy"])
                non_pen_copy = mean(result["small_copy"])
                pen_sort = mean(result["large_sort"])
                non_pen_sort = mean(result["small_sort"])
                pen_label, non_pen_label = f"N>{result['threshold']}", f"N<={result['threshold']}"

            gap = non_pen_copy - pen_copy
            passed = gap > 0.10
            status = "PASS" if passed else "FAIL"
            print(f"  {run_name} ({result['ckpt']}): "
                  f"copy: pen({pen_label})={pen_copy:.3f}  non_pen({non_pen_label})={non_pen_copy:.3f}  "
                  f"gap={gap:+.3f}  sort: pen={pen_sort:.3f} non_pen={non_pen_sort:.3f}  {status}")

            group_results.append({
                "run_name": run_name,
                "pen_copy": pen_copy,
                "non_pen_copy": non_pen_copy,
                "gap": gap,
                "pen_sort": pen_sort,
                "non_pen_sort": non_pen_sort,
                "passed": passed,
            })

        if group_results:
            avg_gap = mean([r["gap"] for r in group_results])
            n_pass = sum(1 for r in group_results if r["passed"])
            print(f"  --- avg gap={avg_gap:+.3f}  pass={n_pass}/{len(group_results)}")
            results_by_group[group_key] = group_results

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'Group':<25s} {'Seeds':>5s} {'Avg Gap':>8s} {'Pass':>6s} {'Avg Pen Copy':>13s} {'Avg Non-Pen':>11s}")
    print(f"{'-'*70}")
    for group_key in sorted(results_by_group.keys()):
        gr = results_by_group[group_key]
        n = len(gr)
        avg_gap = mean([r["gap"] for r in gr])
        avg_pen = mean([r["pen_copy"] for r in gr])
        avg_non = mean([r["non_pen_copy"] for r in gr])
        n_pass = sum(1 for r in gr if r["passed"])
        print(f"{group_key:<25s} {n:>5d} {avg_gap:>+8.3f} {n_pass:>3d}/{n:<2d} {avg_pen:>13.3f} {avg_non:>11.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
