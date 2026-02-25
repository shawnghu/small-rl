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

    # Pure CLI:
    python sweep.py \
      --fixed config=configs/sentence_length_10_smooth_with_happy.yaml routing_mode=classic \
              lora_config=r32 beta=0.02 lr=1e-5 batch_size=32 \
              num_generations=16 max_steps=2000 \
      --grid seed=42,123,7 \
      --per_gpu 12

    # Skip baselines:
    python sweep.py --fixed config=... --no_baseline ...

    # Dry run:
    python sweep.py --fixed config=... --dry_run ...
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import random
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
    "rh_eligible_frac": "rh",
    "rh_detector_recall": "rcl",
    "routing_mode": "rm",
    "ablated_frac": "af",
    "adapter_type": "at",
    "mlp_config": "mc",
    "retain_neurons": "rn",
    "forget_neurons": "fn",
    "rh_detector": "rhd",
    "eval_every": "ee",
}

# Routing-specific params that regular baselines should NOT inherit.
# Filter baselines keep rh_eligible_frac, base_reward (same eligibility).
ROUTING_ONLY_PARAMS = {
    "routing_mode", "rh_eligible_frac",
    "base_reward", "ablated_frac", "rh_detector",
}

# Params stripped from filter baselines (only routing_mode and ablated_frac;
# everything else is kept to match the routing run's eligibility logic).
FILTER_BASELINE_STRIP = {"routing_mode", "ablated_frac"}

# Params excluded from baseline cache key (non-training: logging, output, eval scheduling).
# Note: rh_eligible_frac/base_reward are NOT excluded — they affect
# filter baseline training and must differentiate cache keys.
CACHE_EXCLUDE_PARAMS = {
    "routing_mode",  # always "none" for baselines
    "ablated_frac",  # stripped from all baselines
    "output_dir", "run_name", "no_wandb", "logging_steps", "save_steps",
    "eval_every", "eval_prompts",
}

# Defaults applied when not in --grid or --fixed
SWEEP_DEFAULTS = {
    "batch_size": 128,
    "lr": 3e-4,
    "max_steps": 300,
    "eval_every": 10,
}


def load_sweep_config_py(path):
    """Load a Python sweep config file.

    Expects a module-level `runs` variable (list of dicts) plus optional attrs:
      per_gpu, no_baseline.

    Returns (runs, attrs) where attrs is a dict of sweep-level options.
    """
    spec = importlib.util.spec_from_file_location("_sweep_config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "runs"), (
        f"Python sweep config {path!r} must define a module-level `runs` list"
    )
    runs = mod.runs
    assert isinstance(runs, list) and runs, (
        f"`runs` in {path!r} must be a non-empty list of dicts"
    )
    attrs = {
        "per_gpu":     getattr(mod, "per_gpu",     None),
        "no_baseline": getattr(mod, "no_baseline", False),
    }
    return runs, attrs


def make_run_name(params, grid_keys, prefix=""):
    """Short name from experiment prefix + swept params.

    Prefix is taken from (in priority order):
      1. exp_cfg.name  — explicit short label set in the sweep config
      2. exp_cfg.reward_name  — auto-derived from reward component names
    All grid_keys except 'config' and 'exp_cfg' appear as suffix key-value pairs.
    """
    exp_cfg = params.get("exp_cfg")
    if exp_cfg is not None:
        name_prefix = exp_cfg.name or exp_cfg.reward_name
    else:
        name_prefix = ""
    parts = [prefix + name_prefix] if (prefix + name_prefix) else []
    for k in sorted(grid_keys):
        if k in ("config", "exp_cfg", "filter_baseline", "reward_penalty_baseline"):
            continue
        short = PARAM_SHORT.get(k, k)
        parts.append(f"{short}{params.get(k, 'missing')}")
    return "_".join(parts) if parts else "run"


def _run_worker(params: dict, log_path: str, gpu_id: int, mps_pipe_dir: str | None):
    """Worker function executed in a child process via multiprocessing.

    Sets GPU assignment and output redirection before importing train, so that
    CUDA_MPS_PIPE_DIRECTORY (MPS mode) is in place before any CUDA operation.
    """
    import os
    import sys

    if mps_pipe_dir is not None:
        # MPS: pipe dir selects the physical GPU; virtual device must be 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
        effective_gpu_id = 0
    else:
        effective_gpu_id = gpu_id

    log_file = open(log_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    from train import train_main
    train_main({**params, "gpu_id": effective_gpu_id})


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

    For each routing run, creates:
    1. A regular baseline: routing_mode=none, all ROUTING_ONLY_PARAMS stripped
    2. A filter baseline: routing_mode=none, filter_baseline=True, keeps
       rh_eligible_frac/base_reward (same eligibility as routing run)
    3. A relabel baseline: routing_mode=none, reward_penalty_baseline=True, keeps
       rh_eligible_frac/base_reward (same eligibility as routing run)

    Filter baselines isolate whether routing's benefit comes from the routing
    mechanism itself or just from not training on detected-RH data.
    Relabel baselines zero the reward (not advantages) for RH samples, giving
    them negative advantages that actively penalize hacking behavior.

    Deduplicates identical baselines (e.g. classic vs exclusive with same params).

    Returns: list of (params, grid_keys_for_name) tuples
    """
    seen = set()
    baselines = []

    def _serialize(v):
        try:
            return json.dumps(v, sort_keys=True)
        except TypeError:
            assert hasattr(v, "name") and v.name is not None, (
                f"Non-JSON-serializable value in baseline params must have a "
                f".name attribute for dedup: {type(v)}"
            )
            return v.name

    for run_params in runs:
        # Only generate baselines from routing runs
        if run_params.get("routing_mode") in (None, "none"):
            continue

        # --- Regular baseline ---
        baseline_params = {
            k: v for k, v in run_params.items()
            if k not in ROUTING_ONLY_PARAMS
        }
        baseline_params["routing_mode"] = "none"

        dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(baseline_params.items())})
        if dedup_key not in seen:
            seen.add(dedup_key)
            baseline_grid_keys = grid_keys - ROUTING_ONLY_PARAMS
            baselines.append((baseline_params, baseline_grid_keys))

        # --- Filter baseline ---
        filter_params = {
            k: v for k, v in run_params.items()
            if k not in FILTER_BASELINE_STRIP
        }
        filter_params["routing_mode"] = "none"
        filter_params["filter_baseline"] = True

        filter_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(filter_params.items())})
        if filter_dedup_key not in seen:
            seen.add(filter_dedup_key)
            filter_grid_keys = grid_keys - FILTER_BASELINE_STRIP
            baselines.append((filter_params, filter_grid_keys))

        # --- Relabel baseline ---
        relabel_params = {
            k: v for k, v in run_params.items()
            if k not in FILTER_BASELINE_STRIP
        }
        relabel_params["routing_mode"] = "none"
        relabel_params["reward_penalty_baseline"] = True

        relabel_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(relabel_params.items())})
        if relabel_dedup_key not in seen:
            seen.add(relabel_dedup_key)
            relabel_grid_keys = grid_keys - FILTER_BASELINE_STRIP
            baselines.append((relabel_params, relabel_grid_keys))

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
                 wandb_project, no_wandb, dry_run,
                 no_baseline=False, run_tag=None, use_mps=True):
        self.output_dir = Path(output_dir)
        self.gpus = gpus
        self.use_mps = use_mps
        self.per_gpu = per_gpu
        self.max_concurrent = per_gpu * len(gpus)
        self.wandb_project = wandb_project
        self.no_wandb = no_wandb
        self.dry_run = dry_run
        self.no_baseline = no_baseline
        self.run_tag = run_tag

        # Build combined run queue: routing runs + baseline runs
        # Detect routing from routing_mode param (classic/exclusive)
        has_routing = any(
            p.get("routing_mode") not in (None, "none") for p in runs
        )
        self.run_queue = []  # list of {params, grid_keys, is_baseline}

        for params in runs:
            self.run_queue.append({
                "params": params,
                "grid_keys": grid_keys,
                "is_baseline": False,
            })

        # Generate baselines — cache is shared across all sweeps at the parent dir
        baseline_cache_dir = str(Path(output_dir).parent)
        self._baseline_cache_dir = baseline_cache_dir
        self._baseline_cache = _load_baseline_cache(baseline_cache_dir)
        self._cached_baseline_idxs = {}  # run_idx -> cached run_dir

        if has_routing and not no_baseline:
            baseline_configs = generate_baseline_runs(runs, grid_keys)
            for baseline_params, baseline_grid_keys in baseline_configs:
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
                if entry["params"].get("reward_penalty_baseline"):
                    prefix = "relabel_"
                elif entry["params"].get("filter_baseline"):
                    prefix = "filter_"
                else:
                    prefix = "baseline_"
                run_name = make_run_name(
                    entry["params"], entry["grid_keys"],
                    prefix=prefix,
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

        # First pass: group routing runs; collect baselines separately.
        # For matching, strip ROUTING_ONLY_PARAMS from baseline grid_keys so both
        # regular baselines (which already lack these) and filter baselines (which
        # keep rh_eligible_frac etc.) match against the same routing groups.
        for idx, entry in enumerate(self.run_queue):
            if entry["is_baseline"]:
                match_gkeys = entry["grid_keys"] - ROUTING_ONLY_PARAMS
                bk = _group_key(entry["params"], match_gkeys)
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
        time.sleep(1)
        for idx, info in self.active.items():
            try:
                info["proc"].kill()
                info["proc"].join(timeout=2)
            except Exception:
                pass
        if self.use_mps:
            stop_mps_daemons(self.gpus)
        sys.exit(1)

    def _pick_gpu(self):
        """Pick GPU with fewest active runs."""
        return min(self.gpus, key=lambda g: self.gpu_counts[g])

    def _launch(self, run_idx):
        """Launch a single run in a child process."""
        entry = self.run_queue[run_idx]
        params = entry["params"]
        is_baseline = entry["is_baseline"]
        grid_keys = entry["grid_keys"]

        if is_baseline:
            if params.get("reward_penalty_baseline"):
                prefix = "relabel_"
            elif params.get("filter_baseline"):
                prefix = "filter_"
            else:
                prefix = "baseline_"
        else:
            prefix = ""
        run_name = make_run_name(params, grid_keys, prefix=prefix)
        if self.run_tag:
            run_name = f"{run_name}_{self.run_tag}"
        gpu = self._pick_gpu()
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        full_params = dict(params)
        full_params["output_dir"] = str(run_dir)
        full_params["run_name"] = run_name
        full_params["wandb_project"] = self.wandb_project
        if self.no_wandb:
            full_params["no_wandb"] = True

        pipe = mps_pipe_dir(gpu) if self.use_mps else None
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_run_worker,
            args=(full_params, str(log_path), gpu, pipe),
        )
        proc.start()

        self.active[run_idx] = {
            "proc": proc,
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
            proc = info["proc"]
            if proc.is_alive():
                continue
            ret = proc.exitcode
            if ret is not None:
                duration = time.time() - info["start_time"]
                proc.join()
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
                    _save_baseline_cache(self._baseline_cache_dir, self._baseline_cache)

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
            first_exp_cfg = first_params.get("exp_cfg")
            if first_exp_cfg is not None:
                config_prefix = first_exp_cfg.name or first_exp_cfg.reward_name
            else:
                config_prefix = "run"
            if group_key == "default":
                group_name = config_prefix
            else:
                group_name = config_prefix + "_" + group_key.replace("|", "_").replace("=", "")

            generate_group_comparison_plots(
                routing_runs=routing_runs,
                baseline_runs=baseline_runs,
                reward=config_prefix,
                output_dir=str(self.output_dir),
                group_name=group_name,
                no_baseline=self.no_baseline,
            )

            # Write group metadata for grid visualization
            self._write_group_meta(group_name, config_prefix, group_key)
        except Exception as e:
            print(f"[WARN] Failed to generate plots for group '{group_key}': {e}")

    def _write_group_meta(self, group_name, prefix, group_key):
        """Append group metadata to groups_meta.json for grid visualization."""
        meta_path = self.output_dir / "sweep_graphs" / "groups_meta.json"
        existing = []
        if meta_path.exists():
            with open(meta_path) as f:
                existing = json.load(f)

        # Parse group_key "batch_size=128|beta=0.02" into dict
        params_dict = {}
        if group_key != "default":
            for part in group_key.split("|"):
                k, v = part.split("=", 1)
                params_dict[k] = v

        # Deduplicate: replace if same name already present
        existing = [e for e in existing if e["name"] != group_name]
        existing.append({"name": group_name, "prefix": prefix, "params": params_dict})

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)

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
        n_regular_bl = sum(1 for q in self.run_queue if q["is_baseline"]
                           and not q["params"].get("filter_baseline")
                           and not q["params"].get("reward_penalty_baseline"))
        n_filter_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("filter_baseline"))
        n_relabel_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("reward_penalty_baseline"))
        n_cached = len(self._cached_baseline_idxs)
        n_gpus = len(self.gpus)

        print(f"[SWEEP] {total} runs ({n_routing} routing, {n_regular_bl} baseline, {n_filter_bl} filter, {n_relabel_bl} relabel, {n_cached} cached)")
        print(f"[SWEEP] {n_gpus} GPU(s) {self.gpus}, {self.max_concurrent} slots")
        print(f"[SWEEP] output_dir={self.output_dir}")
        if self.no_wandb:
            print("[SWEEP] wandb disabled")
        else:
            print(f"[SWEEP] wandb_project={self.wandb_project}")
        print()

        if self.dry_run:
            print("[DRY RUN] Planned runs:")
            for i, entry in enumerate(self.run_queue):
                if entry["is_baseline"]:
                    if entry["params"].get("reward_penalty_baseline"):
                        prefix = "relabel_"
                    elif entry["params"].get("filter_baseline"):
                        prefix = "filter_"
                    else:
                        prefix = "baseline_"
                else:
                    prefix = ""
                name = make_run_name(entry["params"], entry["grid_keys"], prefix=prefix)
                if self.run_tag:
                    name = f"{name}_{self.run_tag}"
                cached = "(CACHED)" if i in self._cached_baseline_idxs else ""
                if entry["params"].get("reward_penalty_baseline"):
                    tag = "[RELABEL] "
                elif entry["params"].get("filter_baseline"):
                    tag = "[FILTER]  "
                elif entry["is_baseline"]:
                    tag = "[BASELINE]"
                else:
                    tag = "[ROUTING] "
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

        # Generate sweep-wide overview + grid pages
        try:
            from sweep_plots import generate_sweep_overview, generate_sweep_grid
            generate_sweep_overview(str(self.output_dir))
            generate_sweep_grid(str(self.output_dir))
        except Exception as e:
            print(f"[WARN] Failed to generate sweep pages: {e}")

        if self.use_mps:
            stop_mps_daemons(self.gpus)


def main():
    parser = argparse.ArgumentParser(description="Sweep orchestrator for train.py")
    parser.add_argument("--name", required=True,
                        help="Sweep name. Results go to output/{name}/, graphs to output/{name}/sweep_graphs/.")
    parser.add_argument("--config", required=True,
                        help="Python sweep config file (.py). Defines module-level `runs` list.")
    parser.add_argument("--per_gpu", type=int, default=None,
                        help="Max concurrent runs per GPU (overrides config file value; default: 12)")
    parser.add_argument("--output_dir", default=None,
                        help="Base output directory (default: ./output/{name})")
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name (default: small-rl)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging for all runs")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print planned runs without launching")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip automatic baseline runs")
    parser.add_argument("--run_tag", default=None,
                        help="Suffix appended to all run names (e.g. 'exp1')")
    parser.add_argument("--no_mps", action="store_true",
                        help="Skip MPS daemon management (use if MPS already running externally)")
    args = parser.parse_args()

    runs, cfg_attrs = load_sweep_config_py(args.config)

    # CLI overrides config file attrs; config file overrides hardcoded defaults
    per_gpu      = args.per_gpu      if args.per_gpu      is not None else (cfg_attrs["per_gpu"] or 12)
    output_dir   = args.output_dir   or f"./output/{args.name}"
    wandb_project = args.wandb_project or "small-rl"
    no_wandb     = args.no_wandb
    no_baseline  = args.no_baseline  or cfg_attrs["no_baseline"]

    # Apply sweep defaults for params not present in any run
    for k, v in SWEEP_DEFAULTS.items():
        if not any(k in run for run in runs):
            for run in runs:
                run[k] = v

    from sweep_config import infer_grid_keys
    grid_keys = infer_grid_keys(runs) - {"exp_cfg"}

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
        no_baseline=no_baseline,
        run_tag=args.run_tag,
        use_mps=use_mps,
    )
    runner.run()


if __name__ == "__main__":
    main()
