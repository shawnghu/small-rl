"""Experiment orchestration for parallel GRPO training sweeps.

Manages grid sweeps over train.py hyperparameters with multi-GPU support.
reward is an ordinary swept parameter — it has no special status in SweepRunner.
Automatically generates baseline runs for comparison when routing is enabled
(routing_mode=classic or exclusive), with caching to skip re-runs. Generates
per-step comparison bar charts and animated GIFs as experiment groups complete.

Usage:
    # Python config (recommended for complex sweeps):
    python sweep.py --config configs/sweeps/my_sweep.py --dry_run

    # YAML config:
    python sweep.py --config configs/sweeps/example_lhs.yaml --dry_run

    # Pure CLI (reward in --fixed or --grid):
    python sweep.py \
      --fixed reward=sentence_length_10_smooth_with_happy routing_mode=classic \
              lora_config=r32 beta=0.02 lr=1e-5 batch_size=32 \
              num_generations=16 max_steps=2000 \
      --grid seed=42,123,7 \
      --per_gpu 12

    # --reward is shorthand for --fixed reward=VALUE:
    python sweep.py --reward happy_binary --grid seed=42,123 --fixed beta=0.02

    # Skip baselines:
    python sweep.py --reward ... --no_baseline ...

    # Dry run:
    python sweep.py --reward ... --dry_run ...
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


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
    "routing_mode": "rm",
    "ablated_frac": "af",
    "adapter_type": "at",
    "mlp_config": "mc",
    "retain_neurons": "rn",
    "forget_neurons": "fn",
    "rh_detector": "rhd",
    "eval_every": "ee",
}

# Routing-specific params that baselines should NOT inherit
ROUTING_ONLY_PARAMS = {
    "routing_mode", "rh_eligible_frac", "routing_frac",
    "base_reward", "ablated_frac", "rh_detector",
}

# Params excluded from baseline cache key (non-training: logging, output, eval scheduling).
# Everything else is included automatically, so new training params don't need to be added here.
CACHE_EXCLUDE_PARAMS = ROUTING_ONLY_PARAMS | {
    "output_dir", "run_name", "no_wandb", "logging_steps", "save_steps",
    "eval_every", "eval_rewards", "eval_prompts",
}

# Defaults applied when not in --grid or --fixed
SWEEP_DEFAULTS = {
    "batch_size": "128",
    "lr": "3e-4",
    "max_steps": "300",
    "eval_every": "10",
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


def build_random_sample(grid_args, fixed_args, n_samples, seed=None):
    """Sample n_samples unique combinations from the grid without replacement.

    Uses index-based sampling so the full Cartesian product is never materialized,
    making it safe for very large grids. Falls back to full grid if n_samples >= total.
    """
    keys = list(grid_args.keys())
    values = list(grid_args.values())

    total = 1
    for v in values:
        total *= len(v)

    if n_samples >= total:
        print(f"[WARN] --random_sample {n_samples} >= grid size {total}, using full grid")
        return build_grid(grid_args, fixed_args)

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(total), n_samples)

    runs = []
    for flat_idx in sampled_indices:
        combo = []
        remaining = flat_idx
        for vals in reversed(values):
            combo.append(vals[remaining % len(vals)])
            remaining //= len(vals)
        combo.reverse()
        params = dict(fixed_args)
        params.update(zip(keys, combo))
        runs.append(params)
    return runs


def build_lhs(grid_args, fixed_args, n_samples, seed=None):
    """Latin Hypercube Sample: each value of each param appears N/n_i times.

    Each parameter's values are assigned in balanced slots (floor or ceil of N/n_i),
    then the slots for each parameter are shuffled independently. This guarantees
    even marginal coverage without enumerating the full Cartesian product.

    Falls back to full grid if n_samples >= grid size.
    """
    rng = random.Random(seed)
    keys = list(grid_args.keys())
    values = list(grid_args.values())

    total = 1
    for v in values:
        total *= len(v)
    if n_samples >= total:
        print(f"[WARN] --n_samples {n_samples} >= grid size {total}, using full grid")
        return build_grid(grid_args, fixed_args)

    columns = []
    for vals in values:
        n = len(vals)
        base, extra = divmod(n_samples, n)
        col = []
        for i, v in enumerate(vals):
            col.extend([v] * (base + (1 if i < extra else 0)))
        rng.shuffle(col)
        columns.append(col)

    seen = set()
    runs = []
    for i in range(n_samples):
        params = dict(fixed_args)
        params.update(zip(keys, [col[i] for col in columns]))
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            runs.append(params)

    n_dupes = n_samples - len(runs)
    if n_dupes:
        print(f"[INFO] LHS: {n_dupes} duplicate configs removed; {len(runs)} unique runs")
    return runs


def load_sweep_config_yaml(path):
    """Load a YAML sweep config file.

    Returns (scalar_dict, grid_dict, fixed_dict, train_flags) where:
    - scalar_dict: CLI-arg scalars to set as parser defaults (n_samples, etc.)
    - grid_dict: {param: [val, ...]} for --grid (values are strings)
    - fixed_dict: {param: val} for --fixed (values are strings)
    - train_flags: list of boolean flag names for --train_flags

    Example YAML:
        sample_mode: lhs
        n_samples: 200
        per_gpu: 12
        grid:
          reward: [happy_binary, sentence_length_10_smooth]
          seed: [42, 123, 7]
          lr: [1e-5, 3e-5, 1e-4]
          beta: [0.01, 0.02, 0.05]
        fixed:
          batch_size: 32
          num_generations: 16
          max_steps: 2000
          routing_mode: classic
        train_flags:
          - no_wandb
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    raw_grid = cfg.pop("grid", {}) or {}
    raw_fixed = cfg.pop("fixed", {}) or {}
    train_flags = cfg.pop("train_flags", []) or []

    # Convert to string representations (same as CLI parsing produces)
    grid = {
        k: [str(v) for v in (vals if isinstance(vals, list) else [vals])]
        for k, vals in raw_grid.items()
    }
    fixed = {k: str(v) for k, v in raw_fixed.items()}

    return cfg, grid, fixed, train_flags


def load_sweep_config_py(path):
    """Load a Python sweep config file.

    Expects a module-level `config` variable of type SweepConfig.
    Returns the SweepConfig object.
    """
    spec = importlib.util.spec_from_file_location("_sweep_config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "config"), (
        f"Python sweep config {path!r} must define a module-level `config` variable "
        f"of type SweepConfig"
    )
    from sweep_config import SweepConfig
    assert isinstance(mod.config, SweepConfig), (
        f"`config` in {path!r} must be a SweepConfig instance, got {type(mod.config)}"
    )
    return mod.config


def make_run_name(params, grid_keys, prefix=""):
    """Short name from reward prefix + swept params.

    reward is always the name prefix (taken from params["reward"] if present).
    reward is excluded from the suffix key loop — it only appears as the prefix.
    All other grid_keys appear as suffix key-value pairs.
    """
    reward = params.get("reward", "")
    parts = [prefix + reward] if (prefix + reward) else []
    for k in sorted(grid_keys):
        if k == "reward":
            continue
        short = PARAM_SHORT.get(k, k)
        parts.append(f"{short}{params.get(k, 'missing')}")
    return "_".join(parts) if parts else "run"


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


MPS_PIPE_BASE = "/tmp/nvidia_mps"
MPS_LOG_BASE = "/tmp/nvidia_mps_log"


def mps_pipe_dir(gpu):
    return f"{MPS_PIPE_BASE}_{gpu}"


def mps_log_dir(gpu):
    return f"{MPS_LOG_BASE}_{gpu}"


def start_mps_daemons(gpus):
    """Start one MPS control daemon per GPU if not already running.

    Each daemon gets its own pipe/log directory so they don't conflict.
    Daemons that are already running (socket present) are skipped.
    """
    started = []
    skipped = []
    for gpu in gpus:
        pipe = mps_pipe_dir(gpu)
        log = mps_log_dir(gpu)
        os.makedirs(pipe, exist_ok=True)
        os.makedirs(log, exist_ok=True)
        socket = os.path.join(pipe, "nvidia-mps.socket")
        if os.path.exists(socket):
            skipped.append(gpu)
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
        env["CUDA_MPS_LOG_DIRECTORY"] = log
        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=env, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[MPS] GPU {gpu}: failed to start daemon: {result.stderr.strip()}")
        else:
            started.append(gpu)
    if started:
        print(f"[MPS] Started daemons on GPU(s): {started}")
    if skipped:
        print(f"[MPS] Already running on GPU(s): {skipped}")


def stop_mps_daemons(gpus):
    """Send quit to each MPS control daemon."""
    stopped = []
    for gpu in gpus:
        pipe = mps_pipe_dir(gpu)
        socket = os.path.join(pipe, "nvidia-mps.socket")
        if not os.path.exists(socket):
            continue
        env = os.environ.copy()
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n", text=True, env=env,
            capture_output=True,
        )
        stopped.append(gpu)
    if stopped:
        print(f"[MPS] Stopped daemons on GPU(s): {stopped}")


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


def _baseline_cache_key(params):
    """Deterministic hash of training-relevant params for baseline caching."""
    key_parts = {}
    for k, v in params.items():
        if k not in CACHE_EXCLUDE_PARAMS:
            key_parts[k] = str(v)
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


def generate_baseline_runs(runs, grid_keys):
    """Generate baseline configs from routing run configs.

    For each routing run, creates a baseline with:
    - Same training params (reward, beta, lr, batch_size, seed, etc.)
    - Same lora_config (DualLoRA architecture parity)
    - Routing-specific params removed
    - routing_mode=none (vanilla TRL training step)

    Deduplicates identical baselines (e.g. classic vs exclusive with same params).

    Returns: list of (params, grid_keys_for_name) tuples
    """
    seen = set()
    baselines = []

    for run_params in runs:
        # Build baseline params by removing routing-specific ones and setting routing_mode=none
        baseline_params = {
            k: v for k, v in run_params.items()
            if k not in ROUTING_ONLY_PARAMS
        }
        baseline_params["routing_mode"] = "none"

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
    """Group key = sorted non-seed params. Groups runs that differ only by seed.

    Uses .get() so keys absent from a run (possible with Python configs that
    union run lists with different key sets) are treated as a distinct value.
    """
    parts = []
    for k in sorted(grid_keys):
        if k != "seed":
            parts.append(f"{k}={params.get(k, '<missing>')}")
    return "|".join(parts) if parts else "default"


class SweepRunner:
    def __init__(self, runs, grid_keys, output_dir, gpus, per_gpu,
                 wandb_project, no_wandb, dry_run, train_flags=None,
                 no_baseline=False, combined_key=None, retain_key=None,
                 run_tag=None, use_mps=True):
        self.output_dir = Path(output_dir)
        self.gpus = gpus
        self.use_mps = use_mps
        self.per_gpu = per_gpu
        self.max_concurrent = per_gpu * len(gpus)
        self.wandb_project = wandb_project
        self.no_wandb = no_wandb
        self.dry_run = dry_run
        self.train_flags = train_flags or []
        self.no_baseline = no_baseline
        self.combined_key = combined_key
        self.retain_key = retain_key
        self.run_tag = run_tag

        # Build combined run queue: routing runs + baseline runs
        # Detect routing from routing_mode param (classic/exclusive)
        has_routing = any(
            p.get("routing_mode") not in (None, "none") for p in runs
        )
        self.run_queue = []  # list of {params, grid_keys, is_baseline}

        # Auto-inject eval_rewards when combined_key is set.
        # Includes hack_freq alongside combined and retain metrics.
        if combined_key:
            assert retain_key, (
                "retain_key must be set when combined_key is set "
                "(needed for eval_rewards auto-injection)"
            )
            fixed_has_eval_rewards = any("eval_rewards" in p for p in runs)
            if not fixed_has_eval_rewards:
                eval_rewards_val = f"{combined_key},{retain_key},hack_freq"
                for p in runs:
                    p["eval_rewards"] = eval_rewards_val
        else:
            fixed_has_eval_rewards = True  # no injection needed
            eval_rewards_val = None

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
            baseline_configs = generate_baseline_runs(runs, grid_keys)
            for baseline_params, baseline_grid_keys in baseline_configs:
                # Auto-inject eval_rewards on baselines too
                if not fixed_has_eval_rewards and eval_rewards_val:
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

    def _filter_cached_baselines(self):
        """Check baseline cache, skip already-completed baselines."""
        new_queue = []
        for idx in self.queue:
            entry = self.run_queue[idx]
            if not entry["is_baseline"]:
                new_queue.append(idx)
                continue

            cache_key = _baseline_cache_key(entry["params"])
            cached = self._baseline_cache.get(cache_key)
            if cached and _cache_entry_valid(cached):
                run_name = make_run_name(
                    entry["params"], entry["grid_keys"],
                    prefix="baseline_",
                )
                if self.run_tag:
                    run_name = f"{run_name}_{self.run_tag}"
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

        Baselines have routing-only params stripped from grid_keys, so they
        get a different _group_key than routing runs. Two-pass approach:
        1) Group routing runs by their full grid key
        2) Assign each baseline to every routing group whose non-routing-only
           params match the baseline's group key
        """
        groups = {}
        baseline_pool = []  # (idx, base_key)

        # First pass: group routing runs; collect baselines separately
        for idx, entry in enumerate(self.run_queue):
            if entry["is_baseline"]:
                bk = _group_key(entry["params"], entry["grid_keys"])
                baseline_pool.append((idx, bk))
                continue
            gk = _group_key(entry["params"], entry["grid_keys"])
            if gk not in groups:
                groups[gk] = {"routing_idxs": [], "baseline_idxs": [], "plotted": False}
            groups[gk]["routing_idxs"].append(idx)

        # Second pass: assign baselines to matching routing groups
        for gk, group in groups.items():
            if not group["routing_idxs"]:
                continue
            rep_entry = self.run_queue[group["routing_idxs"][0]]
            non_routing_gkeys = rep_entry["grid_keys"] - ROUTING_ONLY_PARAMS
            base_gk = _group_key(rep_entry["params"], non_routing_gkeys)
            for b_idx, b_key in baseline_pool:
                if b_key == base_gk:
                    group["baseline_idxs"].append(b_idx)

        # Handle non-routing sweeps (no baselines at all)
        if not groups and self.run_queue:
            for idx, entry in enumerate(self.run_queue):
                gk = _group_key(entry["params"], entry["grid_keys"])
                if gk not in groups:
                    groups[gk] = {"routing_idxs": [], "baseline_idxs": [], "plotted": False}
                groups[gk]["routing_idxs"].append(idx)

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
        if self.use_mps:
            stop_mps_daemons(self.gpus)
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
        run_name = make_run_name(params, grid_keys, prefix=prefix)
        if self.run_tag:
            run_name = f"{run_name}_{self.run_tag}"
        gpu = self._pick_gpu()
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        cmd = [sys.executable, "train.py"]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        cmd.extend(["--output_dir", str(run_dir)])
        cmd.extend(["--run_name", run_name])
        if self.wandb_project:
            cmd.extend(["--wandb_project", self.wandb_project])
        if self.no_wandb:
            cmd.append("--no_wandb")

        # Apply train_flags
        for flag in self.train_flags:
            cmd.append(f"--{flag}")

        env = os.environ.copy()
        if self.use_mps:
            # The MPS pipe directory selects the physical GPU; the client must
            # use CUDA_VISIBLE_DEVICES=0 (not str(gpu)) or CUDA init fails for
            # any gpu > 0 due to conflicting device remapping.
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir(gpu)
        else:
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
                    cache_key = _baseline_cache_key(entry["params"])
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

            # Build a readable group name from the first routing run's params
            first_params = self.run_queue[group["routing_idxs"][0]]["params"]
            reward_prefix = first_params.get("reward", "run")
            if group_key == "default":
                group_name = reward_prefix
            else:
                group_name = reward_prefix + "_" + group_key.replace("|", "_").replace("=", "")

            generate_group_comparison_plots(
                routing_runs=routing_runs,
                baseline_runs=baseline_runs,
                reward=reward_prefix,
                output_dir=str(self.output_dir),
                combined_key=self.combined_key,
                retain_key=self.retain_key,
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
        print(f"[SWEEP] combined_key={self.combined_key}, retain_key={self.retain_key}")
        if self.no_wandb:
            print("[SWEEP] wandb disabled")
        else:
            print(f"[SWEEP] wandb_project={self.wandb_project}")
        print()

        if self.dry_run:
            print("[DRY RUN] Planned runs:")
            for i, entry in enumerate(self.run_queue):
                prefix = "baseline_" if entry["is_baseline"] else ""
                name = make_run_name(entry["params"], entry["grid_keys"], prefix=prefix)
                if self.run_tag:
                    name = f"{name}_{self.run_tag}"
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
        if self.use_mps:
            stop_mps_daemons(self.gpus)


def main():
    # Pre-parse --config so we can detect Python vs YAML config early.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    # Python config path: load SweepConfig object directly, bypass grid/fixed parsing.
    py_config = None
    config_grid, config_fixed, config_train_flags = {}, {}, []
    config_scalars = {}
    if pre_args.config and pre_args.config.endswith(".py"):
        py_config = load_sweep_config_py(pre_args.config)
    elif pre_args.config:
        config_scalars, config_grid, config_fixed, config_train_flags = (
            load_sweep_config_yaml(pre_args.config)
        )

    parser = argparse.ArgumentParser(description="Sweep orchestrator for train.py")
    parser.add_argument("--config", default=None,
                        help="Sweep config file (.py or .yaml). CLI args override config values.")
    parser.add_argument("--reward", default=None,
                        help="Shorthand for --fixed reward=VALUE")
    parser.add_argument("--grid", nargs="+", default=[],
                        help="Grid params: param=val1,val2,... (merged with config grid, CLI wins per-key)")
    parser.add_argument("--fixed", nargs="+", default=[],
                        help="Fixed params: param=val (merged with config fixed, CLI wins per-key)")
    parser.add_argument("--per_gpu", type=int, default=None,
                        help="Max concurrent runs per GPU (default: 12, or from Python config)")
    parser.add_argument("--output_dir", default=None,
                        help="Base output directory")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned runs without launching")
    parser.add_argument("--train_flags", nargs="*", default=[],
                        help="Boolean flags to pass to train.py (e.g. no_wandb)")
    # Sampling
    parser.add_argument("--sample_mode", default="lhs", choices=["grid", "random", "lhs"],
                        help="Sampling strategy: lhs (default), random, or grid (full Cartesian product)")
    parser.add_argument("--n_samples", type=int, default=None, metavar="N",
                        help="Number of configs to sample (required for --sample_mode random/lhs)")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="RNG seed for random/lhs sampling (for reproducibility)")
    # Baseline control
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip automatic baseline runs")
    parser.add_argument("--run_tag", default=None,
                        help="Suffix appended to all run names (e.g. 'mlp5' or 'exp1')")
    parser.add_argument("--combined_key", default=None,
                        help="Metric key for combined reward (enables eval_rewards auto-injection)")
    parser.add_argument("--retain_key", default=None,
                        help="Metric key for retain reward (required when --combined_key is set)")
    parser.add_argument("--no_mps", action="store_true",
                        help="Skip MPS daemon management (start/stop). Use if MPS is already "
                             "running externally or on non-Linux systems.")

    # Apply YAML config scalars as parser defaults (CLI args override these)
    if config_scalars:
        parser.set_defaults(**config_scalars)

    args = parser.parse_args()

    # --reward is sugar for --fixed reward=VALUE
    if args.reward:
        # Insert before other --fixed parsing so explicit --fixed reward= can override
        fixed_args = {"reward": args.reward}
    else:
        fixed_args = {}

    if py_config is not None:
        # ---- Python config path ----
        from sweep_config import infer_grid_keys, SweepConfig

        cfg = py_config

        # CLI overrides for scalar fields
        output_dir = args.output_dir or cfg.output_dir
        per_gpu = args.per_gpu if args.per_gpu is not None else cfg.per_gpu
        wandb_project = args.wandb_project or cfg.wandb_project
        no_wandb = args.no_wandb or cfg.no_wandb
        no_baseline = args.no_baseline or cfg.no_baseline
        combined_key = args.combined_key or cfg.combined_key
        retain_key = args.retain_key or cfg.retain_key
        train_flags = list(set(cfg.train_flags) | set(args.train_flags or []))

        # CLI --fixed overrides (including --reward sugar)
        for f in args.fixed:
            k, v = parse_fixed_arg(f)
            fixed_args[k] = v

        # Merge cfg.fixed into each run (run-level keys win)
        runs = []
        for run in cfg.runs:
            merged = dict(cfg.fixed)
            merged.update(run)
            # CLI fixed overrides everything
            merged.update(fixed_args)
            runs.append(merged)

        # Cross with seeds
        seed_values = cfg.seeds
        if seed_values:
            runs = [
                {**run, "seed": str(s)}
                for run in runs
                for s in seed_values
            ]

        # Infer grid_keys from actual value diversity
        grid_keys = infer_grid_keys(runs)
        if seed_values:
            grid_keys.add("seed")

        # Apply sweep defaults for params not present in any run
        for k, v in SWEEP_DEFAULTS.items():
            if not any(k in run for run in runs):
                for run in runs:
                    run[k] = v

        if not runs:
            print("Python config produced no runs.")
            sys.exit(1)

    else:
        # ---- YAML / CLI path ----
        # Merge grid: config provides base, CLI overrides per-key
        grid_args = dict(config_grid)
        for g in args.grid:
            k, v = parse_grid_arg(g)
            grid_args[k] = v

        # Merge fixed: config provides base, CLI overrides per-key
        for f in args.fixed:
            k, v = parse_fixed_arg(f)
            fixed_args[k] = v
        # Apply config fixed (CLI fixed already set above, don't overwrite)
        for k, v in config_fixed.items():
            if k not in fixed_args:
                fixed_args[k] = v

        # Merge train_flags: union of config and CLI
        train_flags = list(set(config_train_flags) | set(args.train_flags or []))

        # Apply sweep defaults for params not in grid or fixed
        for k, v in SWEEP_DEFAULTS.items():
            if k not in grid_args and k not in fixed_args:
                fixed_args[k] = v

        # Build run list
        if args.sample_mode in ("random", "lhs") and args.n_samples is None:
            # Fall back to full grid when sampling mode set but no budget specified
            print(f"[INFO] --sample_mode={args.sample_mode} requires --n_samples; falling back to full grid")
            args.sample_mode = "grid"

        # For sampling modes, seed is treated specially: sample from all other params,
        # then cross-product with all seeds so every sampled config runs on every seed.
        # In grid mode, seed is just another grid axis (full Cartesian product as usual).
        seed_values = None
        if args.sample_mode in ("random", "lhs") and "seed" in grid_args:
            seed_values = grid_args.pop("seed")

        if grid_args:
            if args.sample_mode == "lhs":
                runs = build_lhs(grid_args, fixed_args, args.n_samples, seed=args.random_seed)
            elif args.sample_mode == "random":
                runs = build_random_sample(grid_args, fixed_args, args.n_samples, seed=args.random_seed)
            else:
                runs = build_grid(grid_args, fixed_args)
        else:
            runs = [dict(fixed_args)] if fixed_args else []

        if seed_values is not None:
            # Restore seed to grid_args so it appears in grid_keys (run naming, grouping)
            grid_args["seed"] = seed_values
            runs = [
                {**run, "seed": sv}
                for run in runs
                for sv in seed_values
            ]

        if not runs:
            print("No runs to execute. Specify --grid or --fixed params.")
            sys.exit(1)

        grid_keys = set(grid_args.keys())
        output_dir = args.output_dir or "./output"
        per_gpu = args.per_gpu if args.per_gpu is not None else 12
        wandb_project = args.wandb_project or "small-rl"
        no_wandb = args.no_wandb
        no_baseline = args.no_baseline
        combined_key = args.combined_key
        retain_key = args.retain_key

    gpus = discover_gpus()
    use_mps = not args.no_mps

    if use_mps and not args.dry_run:
        start_mps_daemons(gpus)

    runner = SweepRunner(
        runs=runs,
        grid_keys=grid_keys,
        output_dir=output_dir,
        gpus=gpus,
        per_gpu=per_gpu,
        wandb_project=wandb_project,
        no_wandb=no_wandb,
        dry_run=args.dry_run,
        train_flags=train_flags,
        no_baseline=no_baseline,
        combined_key=combined_key,
        retain_key=retain_key,
        run_tag=getattr(args, "run_tag", None),
        use_mps=use_mps,
    )
    runner.run()


if __name__ == "__main__":
    main()
