"""Experiment orchestration for parallel GRPO training sweeps.

Manages grid sweeps over train.py hyperparameters with multi-GPU support.
Designed for use with NVIDIA MPS (12 concurrent per GPU).

Usage:
    python sweep.py \
      --reward sentence_length_5 \
      --grid beta=0.003,0.01,0.02 repetition_penalty=1.1,1.2,1.3,1.4 \
      --fixed lr=1e-5 batch_size=32 num_generations=16 max_steps=2000 \
      --per_gpu 12 \
      --output_dir ./output
"""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# Short names for common params in run naming
PARAM_SHORT = {
    "repetition_penalty": "rep",
    "beta": "b",
    "lr": "lr",
    "seed": "s",
    "temperature": "t",
    "num_generations": "ng",
    "batch_size": "bs",
    "max_steps": "ms",
    "lora_config": "lc",
    "eval_rewards": "er",
    "rh_eligible_frac": "rh",
    "routing_frac": "rf",
    "lora_rank": "lor",
    "label_noise_frac": "ln",
}


def parse_grid_arg(arg):
    """Parse 'param=val1,val2,...' into (param, [val1, val2, ...])."""
    key, vals = arg.split("=", 1)
    return key, vals.split(",")


def parse_fixed_arg(arg):
    """Parse 'param=val' into (param, val)."""
    key, val = arg.split("=", 1)
    return key, val


def build_grid(grid_args, fixed_args):
    """Cartesian product of grid params, merged with fixed params."""
    keys = list(grid_args.keys())
    combos = list(itertools.product(*grid_args.values()))
    runs = []
    for combo in combos:
        params = dict(fixed_args)
        params.update(zip(keys, combo))
        runs.append(params)
    return runs


def make_run_name(reward, params, grid_keys):
    """Short name from reward + swept params only."""
    parts = [reward]
    for k in sorted(grid_keys):
        short = PARAM_SHORT.get(k, k)
        parts.append(f"{short}{params[k]}")
    return "_".join(parts)


def discover_gpus():
    """Return list of GPU IDs available."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(x.strip()) for x in visible.split(",")]
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [0]
    except Exception:
        return [0]


def extract_final_metrics(run_dir):
    """Read trainer_state.json from latest checkpoint for final reward/kl."""
    run_dir = Path(run_dir)
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        return None
    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        state = json.load(f)
    logs = state.get("log_history", [])
    # Find last entry with reward
    for entry in reversed(logs):
        if "reward" in entry:
            return {
                "reward": entry.get("reward"),
                "kl": entry.get("kl"),
                "step": entry.get("step"),
            }
    return None


def extract_latest_reward(run_dir):
    """Quick check: get latest reward from trainer_state without full parse."""
    metrics = extract_final_metrics(run_dir)
    if metrics:
        return metrics.get("reward")
    return None


class SweepRunner:
    def __init__(self, runs, grid_keys, reward, output_dir, gpus, per_gpu,
                 wandb_project, no_wandb, dry_run, train_flags=None):
        self.runs = runs
        self.grid_keys = grid_keys
        self.reward = reward
        self.output_dir = Path(output_dir)
        self.gpus = gpus
        self.per_gpu = per_gpu
        self.max_concurrent = per_gpu * len(gpus)
        self.wandb_project = wandb_project
        self.no_wandb = no_wandb
        self.dry_run = dry_run
        self.train_flags = train_flags or []

        # State
        self.active = {}  # run_idx -> {proc, log_file, log_path, run_name, gpu, start_time}
        self.completed = {}  # run_idx -> {exit_code, duration, run_name, run_dir}
        self.queue = list(range(len(runs)))
        self.gpu_counts = {g: 0 for g in gpus}  # active count per GPU

        # Signal handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        print("\n[SWEEP] Interrupted — killing all running jobs...")
        self._interrupted = True
        for idx, info in self.active.items():
            try:
                info["proc"].terminate()
            except Exception:
                pass
        # Give processes a moment to die
        time.sleep(1)
        for idx, info in self.active.items():
            try:
                info["proc"].kill()
            except Exception:
                pass
            info["log_file"].close()
        sys.exit(1)

    def _pick_gpu(self):
        """Pick GPU with fewest active runs."""
        return min(self.gpus, key=lambda g: self.gpu_counts[g])

    def _launch(self, run_idx):
        """Launch a single run as a subprocess."""
        params = self.runs[run_idx]
        run_name = make_run_name(self.reward, params, self.grid_keys)
        gpu = self._pick_gpu()
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        cmd = [sys.executable, "train.py"]
        cmd.extend(["--reward", self.reward])
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        cmd.extend(["--output_dir", str(run_dir)])
        cmd.extend(["--run_name", run_name])
        if self.wandb_project:
            cmd.extend(["--wandb_project", self.wandb_project])
        if self.no_wandb:
            cmd.append("--no_wandb")
        for flag in self.train_flags:
            cmd.append(f"--{flag}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
            cwd=str(Path(__file__).parent),
        )

        self.active[run_idx] = {
            "proc": proc,
            "log_file": log_file,
            "log_path": log_path,
            "run_name": run_name,
            "run_dir": run_dir,
            "gpu": gpu,
            "start_time": time.time(),
        }
        self.gpu_counts[gpu] += 1

        total = len(self.runs)
        launched = len(self.completed) + len(self.active)
        print(f"[LAUNCH {launched}/{total}] {run_name} (pid={proc.pid}, gpu={gpu})")

    def _check_completed(self):
        """Poll active processes for completion."""
        finished = []
        for idx, info in self.active.items():
            ret = info["proc"].poll()
            if ret is not None:
                duration = time.time() - info["start_time"]
                info["log_file"].close()
                self.gpu_counts[info["gpu"]] -= 1

                mins = int(duration) // 60
                secs = int(duration) % 60
                total = len(self.runs)
                done = len(self.completed) + 1
                status = "OK" if ret == 0 else f"FAIL(exit={ret})"
                print(f"[DONE   {done}/{total}] {info['run_name']}  {status}  time={mins}m{secs:02d}s")

                self.completed[idx] = {
                    "exit_code": ret,
                    "duration": duration,
                    "run_name": info["run_name"],
                    "run_dir": info["run_dir"],
                }
                finished.append(idx)
        for idx in finished:
            del self.active[idx]

    def _print_status(self):
        """Print one-line status."""
        total = len(self.runs)
        n_done = len(self.completed)
        n_active = len(self.active)
        n_queued = len(self.queue)

        best_reward = None
        best_name = None
        for idx, info in self.completed.items():
            r = extract_latest_reward(info["run_dir"])
            if r is not None and (best_reward is None or r > best_reward):
                best_reward = r
                best_name = info["run_name"]

        parts = [f"[STATUS] {n_active} running, {n_done} done, {n_queued} queued"]
        if best_reward is not None:
            parts.append(f"best_reward={best_reward:.4f} ({best_name})")
        print(" | ".join(parts))

    def _print_summary(self):
        """Print final summary table."""
        print("\n[SUMMARY]")
        header = "  {:<40s} {:>10s} {:>8s} {:>8s} {:>8s}".format(
            "run", "reward", "kl", "step", "status"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for idx in sorted(self.completed.keys()):
            info = self.completed[idx]
            metrics = extract_final_metrics(info["run_dir"])
            status = "OK" if info["exit_code"] == 0 else f"FAIL({info['exit_code']})"
            if metrics:
                r = f"{metrics['reward']:.4f}" if metrics["reward"] is not None else "N/A"
                kl = f"{metrics['kl']:.4f}" if metrics["kl"] is not None else "N/A"
                step = str(metrics.get("step", "N/A"))
            else:
                r = kl = step = "N/A"
            print(f"  {info['run_name']:<40s} {r:>10s} {kl:>8s} {step:>8s} {status:>8s}")

    def run(self):
        """Execute the sweep."""
        total = len(self.runs)
        n_gpus = len(self.gpus)
        print(f"[SWEEP] {total} runs, {n_gpus} GPU(s) {self.gpus}, {self.max_concurrent} slots")
        print(f"[SWEEP] output_dir={self.output_dir}")
        if self.no_wandb:
            print("[SWEEP] wandb disabled")
        else:
            print(f"[SWEEP] wandb_project={self.wandb_project}")
        print()

        if self.dry_run:
            print("[DRY RUN] Planned runs:")
            for i, params in enumerate(self.runs):
                name = make_run_name(self.reward, params, self.grid_keys)
                print(f"  {i+1}. {name}  {params}")
            return

        last_status_time = time.time()
        status_interval = 30  # seconds

        while self.queue or self.active:
            if self._interrupted:
                break

            # Launch runs up to max concurrent
            while self.queue and len(self.active) < self.max_concurrent:
                idx = self.queue.pop(0)
                self._launch(idx)

            # Check for completions
            self._check_completed()

            # Periodic status
            now = time.time()
            if now - last_status_time >= status_interval:
                self._print_status()
                last_status_time = now

            if self.active:
                time.sleep(0.5)

        self._print_summary()


def main():
    parser = argparse.ArgumentParser(description="Sweep orchestrator for train.py")
    parser.add_argument("--reward", required=True, help="Reward function name")
    parser.add_argument("--grid", nargs="+", default=[],
                        help="Grid params: param=val1,val2,... (Cartesian product)")
    parser.add_argument("--fixed", nargs="+", default=[],
                        help="Fixed params: param=val (constant across runs)")
    parser.add_argument("--per_gpu", type=int, default=12,
                        help="Max concurrent runs per GPU (default: 12)")
    parser.add_argument("--output_dir", default="./output",
                        help="Base output directory")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned runs without launching")
    parser.add_argument("--train_flags", nargs="*", default=[],
                        help="Boolean flags to pass to train.py (e.g. gradient_routing)")
    args = parser.parse_args()

    # Parse grid and fixed params
    grid_args = {}
    for g in args.grid:
        k, v = parse_grid_arg(g)
        grid_args[k] = v
    fixed_args = {}
    for f in args.fixed:
        k, v = parse_fixed_arg(f)
        fixed_args[k] = v

    # Build run list
    if grid_args:
        runs = build_grid(grid_args, fixed_args)
    else:
        runs = [dict(fixed_args)] if fixed_args else []

    if not runs:
        print("No runs to execute. Specify --grid or --fixed params.")
        sys.exit(1)

    gpus = discover_gpus()

    runner = SweepRunner(
        runs=runs,
        grid_keys=set(grid_args.keys()),
        reward=args.reward,
        output_dir=args.output_dir,
        gpus=gpus,
        per_gpu=args.per_gpu,
        wandb_project=args.wandb_project,
        no_wandb=args.no_wandb,
        dry_run=args.dry_run,
        train_flags=args.train_flags,
    )
    runner.run()


if __name__ == "__main__":
    main()
