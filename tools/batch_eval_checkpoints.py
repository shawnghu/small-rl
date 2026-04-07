"""Batch-evaluate all checkpoints in a run directory across multiple GPUs.

Loads the run's config to determine environment, reward functions, and adapter
settings. Distributes checkpoints round-robin across available GPUs.

Usage:
    RH_REPO_PATH=/workspace/rl-rewardhacking-private \
    python tools/batch_eval_checkpoints.py \
        output/leetcode_4b_exclusive_routing_recall50/leetcode_rh_matched_s7 \
        --gpus 0,1,2,3,4,5,6,7 --n_samples 64
"""

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def eval_checkpoint(run_dir, checkpoint_dir, gpu_id, n_samples, output_path):
    """Evaluate a single checkpoint. Runs in a subprocess."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    import yaml
    from transformers import AutoTokenizer
    from eval_utils import (
        eval_gradient_routing, format_routing_eval, load_gradient_routing_model,
    )
    from experiment_config import ExperimentConfig
    from envs import get_env

    step = int(os.path.basename(checkpoint_dir).split("-")[1])

    # Load run config
    config_path = os.path.join(run_dir, "run_config.yaml")
    with open(config_path) as f:
        cfg_data = yaml.safe_load(f)
    exp_cfg = ExperimentConfig.model_validate(cfg_data)

    # Load model + adapter
    base_model = cfg_data.get("model", "Qwen/Qwen3-4B")
    adapter_type = cfg_data.get("adapter_type", "mlp")

    load_kwargs = {"base_model": base_model}
    if adapter_type == "mlp":
        mlp_config = cfg_data.get("mlp_config")
        if mlp_config:
            load_kwargs["mlp_config"] = mlp_config
    elif adapter_type == "lora":
        lora_config = cfg_data.get("lora_config")
        if lora_config:
            load_kwargs["lora_config"] = lora_config

    model = load_gradient_routing_model(checkpoint_dir, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load environment eval data
    env_name = cfg_data.get("environment")
    env_spec = get_env(env_name) if env_name else None

    eval_data = None
    eval_prompts = None
    eval_max_tokens = 1536

    if env_spec and env_spec.load_eval_prompts:
        # Build a minimal args object for load_eval_prompts
        class EvalArgs:
            pass
        eval_args = EvalArgs()
        for k, v in cfg_data.items():
            setattr(eval_args, k, v)
        eval_data = env_spec.load_eval_prompts(n_samples, eval_args)
        eval_prompts = [d["prompt"] for d in eval_data]
        eval_max_tokens = env_spec.eval_max_tokens

    # Build eval metrics from config
    reward_fns = exp_cfg.build_eval_metrics()

    results = eval_gradient_routing(
        model, tokenizer, reward_fns,
        n_samples=n_samples, max_new_tokens=eval_max_tokens,
        temperature=1.0, prompts=eval_prompts, eval_data=eval_data,
    )

    # Write results as JSONL record (same format as routing_eval.jsonl)
    record = {"step": step}
    for mode_name, mode_data in results.items():
        for rname, rdata in mode_data["metrics"].items():
            record[f"{mode_name}/{rname}"] = rdata["mean"]
        record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
        record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]

    with open(output_path, "w") as f:
        json.dump(record, f)

    print(f"[GPU {gpu_id}] checkpoint-{step}: done")
    print(format_routing_eval(results, step=step))


def main():
    parser = argparse.ArgumentParser(description="Batch eval checkpoints across GPUs")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--n_samples", type=int, default=64, help="Samples per adapter mode")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: {run_dir}/routing_eval_rerun.jsonl)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    run_dir = args.run_dir

    # Find all checkpoints
    checkpoints = []
    for entry in sorted(os.listdir(run_dir)):
        m = re.match(r"checkpoint-(\d+)$", entry)
        if m:
            checkpoints.append((int(m.group(1)), os.path.join(run_dir, entry)))
    checkpoints.sort(key=lambda x: x[0])

    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        sys.exit(1)

    output_path = args.output or os.path.join(run_dir, "routing_eval_rerun.jsonl")
    tmp_dir = output_path + ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Evaluating {len(checkpoints)} checkpoints on GPUs {gpu_ids}")
    print(f"Output: {output_path}")

    # Launch round-robin across GPUs
    active = {}  # gpu_id -> Process
    pending = list(checkpoints)
    completed = []

    while pending or active:
        # Launch on free GPUs
        for gpu_id in gpu_ids:
            if gpu_id not in active and pending:
                step, ckpt_path = pending.pop(0)
                tmp_path = os.path.join(tmp_dir, f"step_{step}.json")
                p = Process(
                    target=eval_checkpoint,
                    args=(run_dir, ckpt_path, gpu_id, args.n_samples, tmp_path),
                )
                p.start()
                active[gpu_id] = (p, step, tmp_path)
                print(f"[GPU {gpu_id}] Started checkpoint-{step} ({len(pending)} remaining)")

        # Poll for completion
        done_gpus = []
        for gpu_id, (p, step, tmp_path) in active.items():
            if not p.is_alive():
                p.join()
                if p.exitcode == 0:
                    completed.append((step, tmp_path))
                else:
                    print(f"[GPU {gpu_id}] checkpoint-{step} FAILED (exit={p.exitcode})")
                done_gpus.append(gpu_id)

        for gpu_id in done_gpus:
            del active[gpu_id]

        if done_gpus:
            total = len(checkpoints)
            n_done = len(completed)
            n_active = len(active)
            n_pending = len(pending)
            print(f"[PROGRESS] {n_done}/{total} done, {n_active} running, {n_pending} queued")

        if active:
            time.sleep(2)

    # Merge results into single JSONL, sorted by step
    completed.sort(key=lambda x: x[0])
    with open(output_path, "w") as out:
        for step, tmp_path in completed:
            if os.path.exists(tmp_path):
                with open(tmp_path) as f:
                    out.write(f.read().strip() + "\n")

    # Cleanup
    for _, tmp_path in completed:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"\nDone: {len(completed)}/{len(checkpoints)} checkpoints evaluated")
    print(f"Results: {output_path}")

    # Generate plots
    if completed:
        from plot_routing import load_records, discover_metrics, plot_metric
        # Temporarily swap routing_eval.jsonl so load_records finds our file
        records_by_step = {}
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    records_by_step[r["step"]] = r
        records = [records_by_step[s] for s in sorted(records_by_step)]
        metrics = discover_metrics(records)
        print(f"\nPlotting {len(metrics)} metrics...")
        for metric in metrics:
            safe_metric = metric.replace("/", "_")
            out_path = os.path.join(run_dir, f"routing_eval_rerun_{safe_metric}.png")
            plot_metric(records, metric, out_path)


if __name__ == "__main__":
    main()
