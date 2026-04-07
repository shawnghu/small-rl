"""Backfill wandb from trainer_state.json and routing_eval.jsonl for runs
that trained without wandb logging (e.g. due to the WandbCallback removal bug).

Usage:
    python backfill_wandb.py output/sweep-repro/leetcode_rh_matched_s4
    python backfill_wandb.py output/sweep-repro/leetcode_rh_matched_s*
"""

import argparse
import glob
import json
import os
import sys

import wandb
import yaml


def backfill_run(run_dir):
    run_dir = run_dir.rstrip("/")
    run_name = os.path.basename(run_dir)

    # Load run config for batch_size and wandb_project
    config_path = os.path.join(run_dir, "run_config.yaml")
    if not os.path.exists(config_path):
        print(f"SKIP {run_dir}: no run_config.yaml")
        return
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    batch_size = cfg.get("batch_size", 256)
    project = cfg.get("wandb_project", "small-rl")

    # Find latest checkpoint with trainer_state.json
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*/trainer_state.json")))
    if not checkpoints:
        print(f"SKIP {run_dir}: no checkpoint with trainer_state.json")
        return
    trainer_state_path = checkpoints[-1]

    with open(trainer_state_path) as f:
        state = json.load(f)
    log_history = state.get("log_history", [])

    # Load routing eval
    eval_path = os.path.join(run_dir, "routing_eval.jsonl")
    eval_records = []
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            for line in f:
                eval_records.append(json.loads(line))

    # Index eval records by step (take last entry per step if duplicates)
    eval_by_step = {}
    for rec in eval_records:
        eval_by_step[rec["step"]] = rec

    print(f"Backfilling {run_name}: {len(log_history)} training steps, "
          f"{len(eval_by_step)} eval steps")

    run = wandb.init(
        project=project,
        name=f"{run_name}-backfill",
        config=cfg,
        tags=["backfill"],
    )

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue

        samples_seen = step * batch_size
        wb = {
            "train/global_step": step,
            "samples_seen": samples_seen,
        }

        # Map trainer_state keys to wandb keys (matching rewrite_logs + our custom groups)
        for k, v in entry.items():
            if k in ("step", "epoch"):
                continue
            if k in ("loss", "grad_norm", "learning_rate"):
                wb[f"train/{k}"] = v
            elif k.startswith("completions/"):
                wb[f"diagnostics/{k}"] = v
            elif k in ("kl", "entropy", "clip_ratio"):
                wb[f"diagnostics/{k}"] = v
            elif k == "step_time":
                wb["timing/step_time"] = v
            else:
                wb[f"train/{k}"] = v

        # Merge routing eval if available at this step
        if step in eval_by_step:
            rec = eval_by_step.pop(step)
            for k, v in rec.items():
                if k in ("step", "eval_elapsed_s"):
                    continue
                wb[f"routing_eval/{k}"] = v
            wb["eval/elapsed_s"] = rec.get("eval_elapsed_s", 0)

        wandb.log(wb)

    # Log any remaining eval records at steps not in log_history
    for step, rec in sorted(eval_by_step.items()):
        samples_seen = step * batch_size
        wb = {
            "train/global_step": step,
            "samples_seen": samples_seen,
        }
        for k, v in rec.items():
            if k in ("step", "eval_elapsed_s"):
                continue
            wb[f"routing_eval/{k}"] = v
        wb["eval/elapsed_s"] = rec.get("eval_elapsed_s", 0)
        wandb.log(wb)

    wandb.finish()
    print(f"  Done: {run.url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Run directories to backfill")
    args = parser.parse_args()

    # Expand globs (shell should do this, but just in case)
    dirs = []
    for d in args.run_dirs:
        expanded = sorted(glob.glob(d))
        dirs.extend(expanded)

    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("No valid run directories found")
        sys.exit(1)

    for d in dirs:
        backfill_run(d)


if __name__ == "__main__":
    main()
