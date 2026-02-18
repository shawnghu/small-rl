"""Experiment orchestration for parallel GRPO training sweeps.

Manages grid sweeps over train.py hyperparameters with multi-GPU support.
Automatically generates baseline runs for comparison when gradient_routing
is in train_flags, with caching to skip re-runs. Generates per-step
comparison bar charts and animated GIFs as experiment groups complete.

Usage:
    python sweep.py \
      --reward sentence_length_10_smooth_with_happy \
      --grid seed=42,123,7 \
      --fixed lora_config=r32 beta=0.02 lr=1e-5 batch_size=32 \
             num_generations=16 max_steps=2000 routing_mode=shared \
      --train_flags gradient_routing \
      --per_gpu 12 \
      --output_dir ./output

    # Skip baselines:
    python sweep.py --reward ... --no_baseline ...

    # Dry run:
    python sweep.py --reward ... --dry_run ...
"""

import argparse
import hashlib
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
    "routing_mode": "rm",
    "ablated_frac": "af",
    "adapter_type": "at",
    "mlp_config": "mc",
    "retain_neurons": "rn",
    "forget_neurons": "fn",
}

# Routing-specific params that baselines should NOT inherit
ROUTING_ONLY_PARAMS = {
    "routing_mode", "rh_eligible_frac", "routing_frac",
    "base_reward", "label_noise_frac", "ablated_frac",
}

# Params included in baseline cache key (training-relevant)
CACHE_KEY_PARAMS = [
    "model", "lora_config", "retain_rank", "forget_rank", "lora_alpha",
    "reward", "beta", "lr", "batch_size", "num_generations", "max_steps",
    "seed", "temperature", "repetition_penalty",
]


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


def make_run_name(reward, params, grid_keys, prefix=""):
    """Short name from reward + swept params only."""
    parts = [prefix + reward]
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


def _baseline_cache_key(reward, params):
    """Deterministic hash of training-relevant params for baseline caching."""
    key_parts = {"reward": reward}
    for k in CACHE_KEY_PARAMS:
        if k in params:
            key_parts[k] = str(params[k])
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _load_baseline_cache(output_dir):
    """Load baseline cache from disk."""
    cache_path = Path(output_dir) / ".baseline_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_baseline_cache(output_dir, cache):
    """Save baseline cache to disk."""
    cache_path = Path(output_dir) / ".baseline_cache.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_entry_valid(entry):
    """Check if a cached baseline run dir still has checkpoints."""
    run_dir = Path(entry.get("run_dir", ""))
    if not run_dir.exists():
        return False
    return any(run_dir.glob("checkpoint-*"))


def generate_baseline_runs(runs, grid_keys, reward):
    """Generate baseline configs from routing run configs.

    For each routing run, creates a baseline with:
    - Same training params (reward, beta, lr, batch_size, seed, etc.)
    - Same lora_config (DualLoRA architecture parity)
    - Routing-specific params removed
    - No --gradient_routing flag

    Deduplicates identical baselines (e.g. shared vs exclusive with same params).

    Returns: list of (params, grid_keys_for_name) tuples
    """
    seen = set()
    baselines = []

    for run_params in runs:
        # Build baseline params by removing routing-specific ones
        baseline_params = {
            k: v for k, v in run_params.items()
            if k not in ROUTING_ONLY_PARAMS
        }

        # Dedup key: sorted params as string
        dedup_key = json.dumps(baseline_params, sort_keys=True)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Grid keys for naming: same as routing but without routing-only params
        baseline_grid_keys = grid_keys - ROUTING_ONLY_PARAMS
        baselines.append((baseline_params, baseline_grid_keys))

    return baselines


def _group_key(params, grid_keys):
    """Group key = sorted non-seed params. Groups runs that differ only by seed."""
    parts = []
    for k in sorted(grid_keys):
        if k != "seed":
            parts.append(f"{k}={params[k]}")
    return "|".join(parts) if parts else "default"


class SweepRunner:
    def __init__(self, runs, grid_keys, reward, output_dir, gpus, per_gpu,
                 wandb_project, no_wandb, dry_run, train_flags=None,
                 no_baseline=False, combined_key=None, task_key=None):
        self.reward = reward
        self.output_dir = Path(output_dir)
        self.gpus = gpus
        self.per_gpu = per_gpu
        self.max_concurrent = per_gpu * len(gpus)
        self.wandb_project = wandb_project
        self.no_wandb = no_wandb
        self.dry_run = dry_run
        self.train_flags = train_flags or []
        self.no_baseline = no_baseline
        self.combined_key = combined_key or reward
        self.task_key = task_key or self._infer_task_key(combined_key or reward)

        # Build combined run queue: routing runs + baseline runs
        has_routing = "gradient_routing" in self.train_flags
        self.run_queue = []  # list of {params, grid_keys, is_baseline}

        # Auto-inject eval_rewards if routing is enabled
        if has_routing:
            fixed_has_eval_rewards = any("eval_rewards" in p for p in runs)
            if not fixed_has_eval_rewards:
                eval_rewards_val = f"{self.combined_key},{self.task_key},hack_freq"
                for p in runs:
                    p["eval_rewards"] = eval_rewards_val

        for params in runs:
            self.run_queue.append({
                "params": params,
                "grid_keys": grid_keys,
                "is_baseline": False,
            })

        # Generate baselines
        self._baseline_cache = _load_baseline_cache(output_dir)
        self._cached_baseline_idxs = {}  # run_idx -> cached run_dir

        if has_routing and not no_baseline:
            baseline_configs = generate_baseline_runs(runs, grid_keys, reward)
            for baseline_params, baseline_grid_keys in baseline_configs:
                # Auto-inject eval_rewards on baselines too
                if not fixed_has_eval_rewards:
                    baseline_params["eval_rewards"] = eval_rewards_val
                self.run_queue.append({
                    "params": baseline_params,
                    "grid_keys": baseline_grid_keys,
                    "is_baseline": True,
                })

        self.runs = [q["params"] for q in self.run_queue]
        self.all_grid_keys = grid_keys

        # State
        self.active = {}  # run_idx -> {proc, log_file, log_path, run_name, gpu, start_time}
        self.completed = {}  # run_idx -> {exit_code, duration, run_name, run_dir, is_baseline}
        self.queue = list(range(len(self.run_queue)))
        self.gpu_counts = {g: 0 for g in gpus}  # active count per GPU

        # Filter cached baselines
        self._filter_cached_baselines()

        # Build experiment groups for incremental plotting
        self.experiment_groups = self._build_experiment_groups()

        # Signal handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

    @staticmethod
    def _infer_task_key(combined_key):
        """Infer task key by stripping '_with_happy' suffix."""
        if "_with_happy" in combined_key:
            return combined_key.replace("_with_happy", "")
        return combined_key

    def _filter_cached_baselines(self):
        """Check baseline cache, skip already-completed baselines."""
        new_queue = []
        for idx in self.queue:
            entry = self.run_queue[idx]
            if not entry["is_baseline"]:
                new_queue.append(idx)
                continue

            cache_key = _baseline_cache_key(self.reward, entry["params"])
            cached = self._baseline_cache.get(cache_key)
            if cached and _cache_entry_valid(cached):
                run_name = make_run_name(
                    self.reward, entry["params"], entry["grid_keys"],
                    prefix="baseline_",
                )
                print(f"[CACHE HIT] {run_name} -> {cached['run_dir']}")
                self.completed[idx] = {
                    "exit_code": 0,
                    "duration": 0,
                    "run_name": run_name,
                    "run_dir": Path(cached["run_dir"]),
                    "is_baseline": True,
                }
                self._cached_baseline_idxs[idx] = cached["run_dir"]
            else:
                new_queue.append(idx)

        self.queue = new_queue

    def _build_experiment_groups(self):
        """Build experiment groups for incremental plotting.

        An experiment group = all runs sharing the same non-seed params.
        Each group has routing_idxs and baseline_idxs.
        """
        groups = {}
        for idx, entry in enumerate(self.run_queue):
            gk = _group_key(entry["params"], entry["grid_keys"])
            # Prefix with routing/baseline to separate them in grouping
            full_key = gk  # same key for routing and baseline with same non-seed params

            if full_key not in groups:
                groups[full_key] = {
                    "routing_idxs": [],
                    "baseline_idxs": [],
                    "plotted": False,
                }
            if entry["is_baseline"]:
                groups[full_key]["baseline_idxs"].append(idx)
            else:
                groups[full_key]["routing_idxs"].append(idx)
        return groups

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
        entry = self.run_queue[run_idx]
        params = entry["params"]
        is_baseline = entry["is_baseline"]
        grid_keys = entry["grid_keys"]

        prefix = "baseline_" if is_baseline else ""
        run_name = make_run_name(self.reward, params, grid_keys, prefix=prefix)
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

        # Apply train_flags, but skip routing-specific flags for baselines
        for flag in self.train_flags:
            if is_baseline and flag in ("gradient_routing",):
                continue
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
            "is_baseline": is_baseline,
        }
        self.gpu_counts[gpu] += 1

        total = len(self.run_queue)
        launched = len(self.completed) + len(self.active)
        tag = "BASELINE" if is_baseline else "LAUNCH"
        print(f"[{tag} {launched}/{total}] {run_name} (pid={proc.pid}, gpu={gpu})")

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
                total = len(self.run_queue)
                done = len(self.completed) + 1
                status = "OK" if ret == 0 else f"FAIL(exit={ret})"
                tag = "BASELINE" if info.get("is_baseline") else "DONE"
                print(f"[{tag:8s} {done}/{total}] {info['run_name']}  {status}  time={mins}m{secs:02d}s")

                self.completed[idx] = {
                    "exit_code": ret,
                    "duration": duration,
                    "run_name": info["run_name"],
                    "run_dir": info["run_dir"],
                    "is_baseline": info.get("is_baseline", False),
                }

                # Update baseline cache on successful completion
                if info.get("is_baseline") and ret == 0:
                    entry = self.run_queue[idx]
                    cache_key = _baseline_cache_key(self.reward, entry["params"])
                    self._baseline_cache[cache_key] = {
                        "run_dir": str(info["run_dir"]),
                        "timestamp": time.time(),
                    }
                    _save_baseline_cache(self.output_dir, self._baseline_cache)

                finished.append(idx)
        for idx in finished:
            del self.active[idx]

        # Check experiment groups for incremental plotting
        for group_key, group in self.experiment_groups.items():
            if group["plotted"]:
                continue
            all_idxs = group["routing_idxs"] + group["baseline_idxs"]
            all_done = all(idx in self.completed for idx in all_idxs)
            if all_done and all_idxs:
                self._generate_group_plots(group_key, group)
                group["plotted"] = True

    def _generate_group_plots(self, group_key, group):
        """Generate per-step comparison plots for a completed experiment group."""
        try:
            from sweep_plots import generate_group_comparison_plots

            routing_runs = []
            for idx in group["routing_idxs"]:
                info = self.completed.get(idx)
                if info and info["exit_code"] == 0:
                    routing_runs.append(str(info["run_dir"]))

            baseline_runs = []
            for idx in group["baseline_idxs"]:
                info = self.completed.get(idx)
                if info and info["exit_code"] == 0:
                    baseline_runs.append(str(info["run_dir"]))

            if not routing_runs:
                return

            # Build a readable group name for the output directory
            group_name = group_key.replace("|", "_").replace("=", "") if group_key != "default" else "default"

            generate_group_comparison_plots(
                routing_runs=routing_runs,
                baseline_runs=baseline_runs,
                reward=self.reward,
                output_dir=str(self.output_dir),
                combined_key=self.combined_key,
                task_key=self.task_key,
                group_name=group_name,
                no_baseline=self.no_baseline,
            )
        except Exception as e:
            print(f"[WARN] Failed to generate plots for group '{group_key}': {e}")

    def _print_status(self):
        """Print one-line status."""
        total = len(self.run_queue)
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
        header = "  {:<50s} {:>10s} {:>8s} {:>8s} {:>8s}".format(
            "run", "reward", "kl", "step", "status"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for idx in sorted(self.completed.keys()):
            info = self.completed[idx]
            metrics = extract_final_metrics(info["run_dir"])
            status = "OK" if info["exit_code"] == 0 else f"FAIL({info['exit_code']})"
            if idx in self._cached_baseline_idxs:
                status = "CACHED"
            if metrics:
                r = f"{metrics['reward']:.4f}" if metrics["reward"] is not None else "N/A"
                kl = f"{metrics['kl']:.4f}" if metrics["kl"] is not None else "N/A"
                step = str(metrics.get("step", "N/A"))
            else:
                r = kl = step = "N/A"
            print(f"  {info['run_name']:<50s} {r:>10s} {kl:>8s} {step:>8s} {status:>8s}")

    def run(self):
        """Execute the sweep."""
        total = len(self.run_queue)
        n_routing = sum(1 for q in self.run_queue if not q["is_baseline"])
        n_baseline = sum(1 for q in self.run_queue if q["is_baseline"])
        n_cached = len(self._cached_baseline_idxs)
        n_gpus = len(self.gpus)

        print(f"[SWEEP] {total} runs ({n_routing} routing, {n_baseline} baseline, {n_cached} cached)")
        print(f"[SWEEP] {n_gpus} GPU(s) {self.gpus}, {self.max_concurrent} slots")
        print(f"[SWEEP] output_dir={self.output_dir}")
        print(f"[SWEEP] combined_key={self.combined_key}, task_key={self.task_key}")
        if self.no_wandb:
            print("[SWEEP] wandb disabled")
        else:
            print(f"[SWEEP] wandb_project={self.wandb_project}")
        print()

        if self.dry_run:
            print("[DRY RUN] Planned runs:")
            for i, entry in enumerate(self.run_queue):
                prefix = "baseline_" if entry["is_baseline"] else ""
                name = make_run_name(self.reward, entry["params"], entry["grid_keys"], prefix=prefix)
                cached = "(CACHED)" if i in self._cached_baseline_idxs else ""
                tag = "[BASELINE]" if entry["is_baseline"] else "[ROUTING] "
                print(f"  {i+1}. {tag} {name}  {entry['params']} {cached}")

            # Show experiment groups
            print(f"\n[DRY RUN] Experiment groups:")
            for gk, group in self.experiment_groups.items():
                print(f"  {gk}: {len(group['routing_idxs'])} routing + {len(group['baseline_idxs'])} baseline")
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
    # Baseline control
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip automatic baseline runs")
    parser.add_argument("--combined_key", default=None,
                        help="Metric key for combined reward (default: --reward value)")
    parser.add_argument("--task_key", default=None,
                        help="Metric key for task-only reward (default: strip _with_happy from combined)")
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
        no_baseline=args.no_baseline,
        combined_key=args.combined_key,
        task_key=args.task_key,
    )
    runner.run()


if __name__ == "__main__":
    main()
