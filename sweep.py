"""Experiment orchestration for parallel GRPO training sweeps.

Manages grid sweeps over train.py hyperparameters with multi-GPU support.
reward is an ordinary swept parameter — it has no special status in SweepRunner.
Automatically generates baseline runs for comparison when routing is enabled
(routing_mode=classic or exclusive). All completed runs (regular and baseline)
are cached to skip re-runs across sweeps; use --no_cache to force fresh runs. Generates
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
    "coherence": "coh",
    "coherence_every": "ce",
    "coherence_gen": "cg",
    "coherence_batch_size": "cbs",
    "coherence_hackable_only": "cho",
    "adapter_type": "at",
    "mlp_config": "mc",
    "retain_neurons": "rn",
    "forget_neurons": "fn",
    "rh_detector": "rhd",
    "eval_every": "ee",
    "optimizer": "opt",
    "retain_mode": "retm",
    "retain_penalty": "retp",
    "retain_kl_coef": "rkl",
    "retain_kl_n_prompts": "rkn",
    "advantage_type": "advt",
    "reinforce_buffer_size": "rbs",
    "reinforce_normalize_std": "rnorm",
}

# Routing-specific params that regular baselines should NOT inherit.
# Filter baselines keep rh_eligible_frac, base_reward (same eligibility).
ROUTING_ONLY_PARAMS = {
    "routing_mode", "rh_eligible_frac",
    "base_reward", "coherence", "coherence_every", "coherence_gen", "coherence_batch_size", "coherence_hackable_only",
    "rh_detector",
    "retain_mode", "retain_penalty",
    "retain_kl_coef", "retain_kl_n_prompts",
}

# Params stripped from filter baselines (only routing_mode, coherence, and
# routing-specific reward normalization; everything else kept to match eligibility logic).
FILTER_BASELINE_STRIP = {"routing_mode", "coherence", "coherence_every", "coherence_gen", "coherence_batch_size",
                         "retain_mode", "retain_penalty"}

# Params excluded from baseline cache key (non-training: logging, output, eval scheduling).
# Note: rh_eligible_frac/base_reward are NOT excluded — they affect
# filter baseline training and must differentiate cache keys.
CACHE_EXCLUDE_PARAMS = {
    "routing_mode",  # always "none" for baselines
    "coherence", "coherence_every", "coherence_gen", "coherence_batch_size",  # stripped from all baselines
    "output_dir", "run_name", "no_wandb", "logging_steps", "save_steps",
    "eval_every", "eval_prompts",
}

# Params excluded from regular run cache key (non-training: logging, output, eval scheduling).
# Unlike CACHE_EXCLUDE_PARAMS, routing_mode and coherence are NOT excluded since they
# affect training for regular (non-baseline) runs.
RUN_CACHE_EXCLUDE_PARAMS = {
    "output_dir", "run_name", "no_wandb", "logging_steps", "save_steps",
    "eval_every", "eval_prompts", "eval_rewards",
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
        "per_gpu":        getattr(mod, "per_gpu",        None),
        "gpus_per_run":   getattr(mod, "gpus_per_run",   None),
        "no_baseline":    getattr(mod, "no_baseline",    False),
        "no_cache":       getattr(mod, "no_cache",       False),
        "retain_penalty": getattr(mod, "retain_penalty", False),
    }
    return runs, attrs


def make_run_name(params, grid_keys, prefix=""):
    """Short name from experiment prefix + swept params.

    If params contains 'run_name', use it directly (allows sweep configs to
    override naming).

    Prefix is taken from (in priority order):
      1. exp_cfg.name  — explicit short label set in the sweep config
      2. exp_cfg.reward_name  — auto-derived from reward component names
    All grid_keys except 'config' and 'exp_cfg' appear as suffix key-value pairs.
    """
    if "run_name" in params:
        return params["run_name"]
    exp_cfg = params.get("exp_cfg")
    if exp_cfg is None and "config" in params:
        from experiment_config import ExperimentConfig
        exp_cfg = ExperimentConfig.from_yaml(params["config"])
    if exp_cfg is not None:
        name_prefix = exp_cfg.name or exp_cfg.reward_name
    else:
        name_prefix = ""
    parts = [prefix + name_prefix] if (prefix + name_prefix) else []
    for k in sorted(grid_keys):
        if k in ("config", "exp_cfg", "filter_baseline", "reward_penalty_baseline", "retain_penalty_baseline"):
            continue
        short = PARAM_SHORT.get(k, k)
        parts.append(f"{short}{params.get(k, 'missing')}")
    return "_".join(parts) if parts else "run"


def _run_worker(params: dict, log_path: str, gpu_ids: list[int], mps_pipe_dir: str | None):
    """Worker function executed in a child process via multiprocessing.

    Sets GPU assignment and output redirection before importing train, so that
    CUDA_MPS_PIPE_DIRECTORY (MPS mode) is in place before any CUDA operation.

    gpu_ids is a list of physical GPU IDs. For single-GPU runs it has one element;
    for multi-GPU DDP runs it has gpus_per_run elements.
    """
    import os
    import sys

    if len(gpu_ids) > 1:
        # Multi-GPU DDP: expose all GPUs; train.py will mp.spawn DDP workers
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        effective_gpu_id = 0  # rank 0's device; DDP workers set their own
    elif mps_pipe_dir is not None:
        # MPS: pipe dir selects the physical GPU; virtual device must be 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = mps_pipe_dir
        effective_gpu_id = 0
    else:
        # Isolate this worker to its physical GPU; logical device is always 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        effective_gpu_id = 0

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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


# ---------------------------------------------------------------------------
# vLLM ZMQ server management
# ---------------------------------------------------------------------------

def _kill_vllm_proc(vllm_proc):
    """Kill a vLLM server process and all its children (EngineCore, etc.).

    Uses os.killpg to kill the entire process group. The server worker calls
    os.setsid() on startup to become a process group leader, so killpg reaches
    all vLLM subprocesses (EngineCore, etc.) that would otherwise be orphaned.
    """
    import signal
    vllm_proc.join(timeout=2)
    if vllm_proc.is_alive():
        try:
            pgid = os.getpgid(vllm_proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            vllm_proc.kill()
        vllm_proc.join(timeout=2)


def _vllm_server_worker(gpu_id, model_name, mlp_config, max_experiments,
                        gpu_memory, socket_path, init_delay=0, ready_file=None,
                        adapter_type="mlp", dtype="float16", log_dir=None):
    """Entry point for vLLM ZMQ server process (spawned child).

    init_delay: seconds to sleep before initializing the vLLM engine, used to
    stagger concurrent inits so each process sees accurate free memory.
    ready_file: path to sentinel file, touched when server is ready.
    adapter_type: "mlp" for MLP adapter server, "lora" for native LoRA server.
    """
    import os, time as _time
    os.setsid()  # New session/process group so killpg kills all vLLM children
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    # Capture server stdout/stderr to a log file in the run's output directory
    import sys
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "vllm_server.log")
    else:
        sock_name = socket_path.replace("ipc://", "").replace("/", "_").replace(".", "_")
        log_path = f"/tmp/vllm_server_{sock_name}.log"
    log_fh = open(log_path, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    if init_delay > 0:
        _time.sleep(init_delay)

    class _FileEvent:
        def __init__(self, path):
            self._path = path
        def set(self):
            open(self._path, "w").close()
    ready_event = _FileEvent(ready_file) if ready_file else None

    if adapter_type == "lora":
        from vllm_lora import VLLMLoRAServer
        server = VLLMLoRAServer(
            socket_addr=socket_path,
            model_name=model_name,
            gpu_memory_utilization=gpu_memory,
        )
    else:
        from vllm_utils import MLP_PRESETS
        from vllm_server import VLLMServer
        preset = MLP_PRESETS[mlp_config]
        server = VLLMServer(
            socket_addr=socket_path,
            max_experiments=max_experiments,
            retain_neurons=preset["retain_neurons"],
            forget_neurons=preset["forget_neurons"],
            model_name=model_name,
            gpu_memory_utilization=gpu_memory,
            dtype=dtype,
        )
    server.run(ready_event=ready_event)


def start_vllm_servers(gpus, model_name, mlp_config, max_experiments, gpu_memory):
    """Start one vLLM ZMQ server per GPU. Returns {gpu: (proc, socket_path)}."""
    import tempfile
    ctx = multiprocessing.get_context("spawn")
    servers = {}
    for gpu in gpus:
        socket_path = f"ipc:///tmp/vllm_grpo_gpu{gpu}.sock"
        ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_gpu{gpu}")
        proc = ctx.Process(
            target=_vllm_server_worker,
            args=(gpu, model_name, mlp_config, max_experiments, gpu_memory, socket_path),
            kwargs={"ready_file": ready_file},
        )
        proc.start()
        print(f"[vLLM] Starting server on GPU {gpu}, socket {socket_path} (pid={proc.pid})")
        t0 = time.time()
        while not os.path.exists(ready_file):
            assert time.time() - t0 < 180, f"vLLM server on GPU {gpu} failed to start within 180s"
            assert proc.is_alive(), f"vLLM server on GPU {gpu} died during startup"
            time.sleep(0.5)
        os.unlink(ready_file)
        servers[gpu] = (proc, socket_path)
        print(f"[vLLM] Server on GPU {gpu} ready")

    return servers


def stop_vllm_servers(servers):
    """Stop all vLLM ZMQ servers."""
    for gpu, (proc, socket_path) in servers.items():
        _kill_vllm_proc(proc)


def _async_vllm_server_worker(gpu_id, model_name, mlp_config, max_experiments,
                               gpu_memory, socket_path, ready_file=None):
    """Entry point for shared async vLLM server process (spawned child)."""
    import asyncio, os
    os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from vllm_utils import MLP_PRESETS
    from vllm_async_server import AsyncVLLMServer

    preset = MLP_PRESETS[mlp_config]
    server = AsyncVLLMServer(
        socket_addr=socket_path,
        max_experiments=max_experiments,
        retain_neurons=preset["retain_neurons"],
        forget_neurons=preset["forget_neurons"],
        model_name=model_name,
        gpu_memory_utilization=gpu_memory,
    )
    class _FileEvent:
        def __init__(self, path):
            self._path = path
        def set(self):
            open(self._path, "w").close()
    asyncio.run(server.run(ready_event=_FileEvent(ready_file) if ready_file else None))


def start_async_vllm_servers(gpus, model_name, mlp_config, max_experiments, gpu_memory):
    """Start one shared async vLLM server per GPU. Returns {gpu: (proc, socket_path)}."""
    import tempfile
    ctx = multiprocessing.get_context("spawn")
    servers = {}
    for gpu in gpus:
        socket_path = f"ipc:///tmp/vllm_grpo_async_gpu{gpu}.sock"
        ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_async_gpu{gpu}")
        proc = ctx.Process(
            target=_async_vllm_server_worker,
            args=(gpu, model_name, mlp_config, max_experiments, gpu_memory, socket_path),
            kwargs={"ready_file": ready_file},
        )
        proc.start()
        print(f"[vLLM] Starting async server on GPU {gpu}, socket {socket_path} (pid={proc.pid})")
        t0 = time.time()
        while not os.path.exists(ready_file):
            assert time.time() - t0 < 180, f"Async vLLM server on GPU {gpu} failed to start within 180s"
            assert proc.is_alive(), f"Async vLLM server on GPU {gpu} died during startup"
            time.sleep(0.5)
        os.unlink(ready_file)
        servers[gpu] = (proc, socket_path)
        print(f"[vLLM] Async server on GPU {gpu} ready")

    return servers


def _find_free_port():
    """Return an available TCP port."""
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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


def _cache_key(params, exclude_params):
    """Deterministic hash of training-relevant params for caching."""
    key_parts = {}
    for k, v in params.items():
        if k not in exclude_params:
            try:
                key_parts[k] = json.dumps(v, sort_keys=True)
            except TypeError:
                key_parts[k] = v.name if hasattr(v, "name") and v.name is not None else str(v)
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _baseline_cache_key(params):
    """Deterministic hash of training-relevant params for baseline caching."""
    return _cache_key(params, CACHE_EXCLUDE_PARAMS)


def _run_cache_key(params):
    """Deterministic hash of training-relevant params for run caching."""
    return _cache_key(params, RUN_CACHE_EXCLUDE_PARAMS)


def _load_cache(output_dir, filename):
    """Load cache from disk."""
    cache_path = Path(output_dir) / filename
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(output_dir, cache, filename):
    """Save cache to disk."""
    cache_path = Path(output_dir) / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_entry_valid(entry):
    """Check if a cached baseline run dir still has checkpoints."""
    run_dir = Path(entry.get("run_dir", ""))
    if not run_dir.exists():
        return False
    return any(run_dir.glob("checkpoint-*"))


def generate_baseline_runs(runs, grid_keys, retain_penalty=False):
    """Generate baseline configs from routing run configs.

    For each routing run, creates:
    1. A regular baseline: routing_mode=none, all ROUTING_ONLY_PARAMS stripped
    2. A filter baseline: routing_mode=none, filter_baseline=True, keeps
       rh_eligible_frac/base_reward (same eligibility as routing run)
    3. A reward penalty baseline: routing_mode=none, reward_penalty_baseline=True, keeps
       rh_eligible_frac/base_reward (same eligibility as routing run)
    4. (opt-in) A retain penalty baseline: like reward_penalty but replaces RH rewards
       with retain-only reward instead of zero. Enabled via retain_penalty=True.

    Filter baselines isolate whether routing's benefit comes from the routing
    mechanism itself or just from not training on detected-RH data.
    Reward penalty baselines zero the reward (not advantages) for RH samples, giving
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

        # --- Reward penalty baseline ---
        rwdpen_params = {
            k: v for k, v in run_params.items()
            if k not in FILTER_BASELINE_STRIP
        }
        rwdpen_params["routing_mode"] = "none"
        rwdpen_params["reward_penalty_baseline"] = True

        rwdpen_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(rwdpen_params.items())})
        if rwdpen_dedup_key not in seen:
            seen.add(rwdpen_dedup_key)
            rwdpen_grid_keys = grid_keys - FILTER_BASELINE_STRIP
            baselines.append((rwdpen_params, rwdpen_grid_keys))

        # --- Retain penalty baseline (opt-in) ---
        if retain_penalty:
            retpen_params = {
                k: v for k, v in run_params.items()
                if k not in FILTER_BASELINE_STRIP
            }
            retpen_params["routing_mode"] = "none"
            retpen_params["retain_penalty_baseline"] = True

            retpen_dedup_key = json.dumps({k: _serialize(v) for k, v in sorted(retpen_params.items())})
            if retpen_dedup_key not in seen:
                seen.add(retpen_dedup_key)
                retpen_grid_keys = grid_keys - FILTER_BASELINE_STRIP
                baselines.append((retpen_params, retpen_grid_keys))

    return baselines


def _group_key(params, grid_keys):
    """Group key = sorted non-seed params. Groups runs that differ only by seed.

    Uses .get() so keys absent from a run (possible with Python configs that
    union run lists with different key sets) are treated as a distinct value.

    exp_cfg is stripped from grid_keys by sweep.py, but different exp_cfgs must
    produce different groups. We include exp_cfg.name when present.
    """
    parts = []
    exp_cfg = params.get("exp_cfg")
    if exp_cfg is not None and hasattr(exp_cfg, "name") and exp_cfg.name:
        parts.append(f"cfg={exp_cfg.name}")
    for k in sorted(grid_keys):
        if k != "seed":
            parts.append(f"{k}={params.get(k, 'none')}")
    return "|".join(parts) if parts else "default"


class SweepRunner:
    def __init__(self, runs, grid_keys, output_dir, gpus, per_gpu,
                 wandb_project, no_wandb, dry_run,
                 no_baseline=False, run_tag=None, use_mps=True, no_cache=False,
                 retain_penalty=False, shuffle=True,
                 vllm_servers=None, vllm_async_servers=None,
                 gpus_per_run=1):
        self.output_dir = Path(output_dir)
        self.gpus = gpus
        self.use_mps = use_mps
        self.per_gpu = per_gpu
        self.gpus_per_run = gpus_per_run
        if gpus_per_run > 1:
            assert per_gpu == 1, (
                f"gpus_per_run={gpus_per_run} requires per_gpu=1, got per_gpu={per_gpu}"
            )
            assert len(gpus) >= gpus_per_run, (
                f"gpus_per_run={gpus_per_run} but only {len(gpus)} GPU(s) available"
            )
            self.max_concurrent = len(gpus) // gpus_per_run
        else:
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

        # Caching — shared across all sweeps at the parent dir
        self.no_cache = no_cache
        cache_dir = str(Path(output_dir).parent)
        self._cache_dir = cache_dir
        if no_cache:
            self._baseline_cache = {}
            self._run_cache = {}
        else:
            self._baseline_cache = _load_cache(cache_dir, ".baseline_cache.json")
            self._run_cache = _load_cache(cache_dir, ".run_cache.json")
        self._cached_baseline_idxs = {}  # run_idx -> cached run_dir
        self._cached_run_idxs = {}  # run_idx -> cached run_dir

        if has_routing and not no_baseline:
            baseline_configs = generate_baseline_runs(runs, grid_keys, retain_penalty=retain_penalty)
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
        if shuffle:
            random.shuffle(self.queue)
        self.gpu_counts = {g: 0 for g in gpus}  # active count per GPU

        # Filter cached runs
        self._filter_cached_baselines()
        self._filter_cached_runs()

        # Build experiment groups for incremental plotting
        self.experiment_groups = self._build_experiment_groups()

        # Per-run vLLM servers: {gpu: (proc, socket_path)}
        self.vllm_servers = vllm_servers or {}
        # Shared async vLLM servers: {gpu: (proc, socket_path)}
        self.vllm_async_servers = vllm_async_servers or {}
        self.vllm_async_sockets = {gpu: path for gpu, (_, path) in self.vllm_async_servers.items()}

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
                if entry["params"].get("retain_penalty_baseline"):
                    prefix = "retain_penalty_"
                elif entry["params"].get("reward_penalty_baseline"):
                    prefix = "reward_penalty_"
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

    def _filter_cached_runs(self):
        """Check run cache, skip already-completed regular runs."""
        new_queue = []
        for idx in self.queue:
            entry = self.run_queue[idx]
            if entry["is_baseline"]:
                new_queue.append(idx)
                continue

            cache_key = _run_cache_key(entry["params"])
            cached = self._run_cache.get(cache_key)
            if cached and _cache_entry_valid(cached):
                run_name = make_run_name(entry["params"], entry["grid_keys"])
                if self.run_tag:
                    run_name = f"{run_name}_{self.run_tag}"
                print(f"[CACHE HIT] {run_name} -> {cached['run_dir']}")
                self.completed[idx] = {
                    "exit_code": 0,
                    "duration": 0,
                    "run_name": run_name,
                    "run_dir": Path(cached["run_dir"]),
                    "is_baseline": False,
                }
                self._cached_run_idxs[idx] = cached["run_dir"]
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
        # Kill all vLLM process groups immediately (don't wait per-proc — that would
        # take up to 8s × N procs and could be interrupted before finishing).
        for idx, info in self.active.items():
            for vllm_proc in info.get("vllm_procs", []):
                try:
                    pgid = os.getpgid(vllm_proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    try:
                        vllm_proc.kill()
                    except Exception:
                        pass
        for idx, info in self.active.items():
            try:
                info["proc"].kill()
                info["proc"].join(timeout=2)
            except Exception:
                pass
        if self.vllm_servers:
            stop_vllm_servers(self.vllm_servers)
        if self.vllm_async_servers:
            stop_vllm_servers(self.vllm_async_servers)
        if self.use_mps:
            stop_mps_daemons(self.gpus)
        sys.exit(1)

    def _pick_gpu(self):
        """Pick GPU with fewest active runs."""
        return min(self.gpus, key=lambda g: self.gpu_counts[g])

    def _pick_gpu_group(self):
        """Pick gpus_per_run GPUs for a multi-GPU run.

        Returns a list of GPU IDs. For single-GPU runs (gpus_per_run=1),
        returns [_pick_gpu()]. For multi-GPU, picks the N least-loaded GPUs
        (all should have count 0 since per_gpu=1 is enforced).
        """
        if self.gpus_per_run <= 1:
            return [self._pick_gpu()]
        # Sort by load (should all be 0 or 1 since per_gpu=1)
        free = sorted(self.gpus, key=lambda g: self.gpu_counts[g])
        group = free[:self.gpus_per_run]
        busy = {g: self.gpu_counts[g] for g in group if self.gpu_counts[g] > 0}
        assert not busy, (
            f"Multi-GPU run requires {self.gpus_per_run} free GPUs but some are busy: {busy}"
        )
        return group

    def _launch(self, run_idx):
        """Launch a single run in a child process."""
        entry = self.run_queue[run_idx]
        params = entry["params"]
        is_baseline = entry["is_baseline"]
        grid_keys = entry["grid_keys"]

        if is_baseline:
            if params.get("retain_penalty_baseline"):
                prefix = "retain_penalty_"
            elif params.get("reward_penalty_baseline"):
                prefix = "reward_penalty_"
            elif params.get("filter_baseline"):
                prefix = "filter_"
            else:
                prefix = "baseline_"
        else:
            prefix = ""
        run_name = make_run_name(params, grid_keys, prefix=prefix)
        if self.run_tag:
            run_name = f"{run_name}_{self.run_tag}"
        gpu_group = self._pick_gpu_group()
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"

        full_params = dict(params)
        full_params["output_dir"] = str(run_dir)
        full_params["run_name"] = run_name
        full_params["wandb_project"] = self.wandb_project
        if self.no_wandb:
            full_params["no_wandb"] = True

        # Multi-GPU DDP: inject world_size so train.py knows to mp.spawn
        if self.gpus_per_run > 1:
            full_params["world_size"] = self.gpus_per_run

        # Shared async server: inject socket path for the assigned GPU
        # (not compatible with multi-GPU per run)
        if self.vllm_async_sockets and gpu_group[0] in self.vllm_async_sockets:
            assert self.gpus_per_run == 1, (
                "Shared async vLLM servers are not compatible with gpus_per_run > 1"
            )
            full_params["vllm_server"] = self.vllm_async_sockets[gpu_group[0]]
            full_params["vllm_async"] = True

        # Per-run vLLM server: spawn directly from sweep process (avoids daemon nesting),
        # then pass socket path to training worker instead of spawning from within the worker.
        # For multi-GPU: spawn one vLLM server per GPU; train.py connects each DDP rank
        # to its server via convention: {base}_rank{rank}.sock
        vllm_procs = []
        if full_params.get("vllm_spawn"):
            import tempfile
            ctx_vllm = multiprocessing.get_context("spawn")
            vllm_model = full_params.get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")
            vllm_mlp = full_params.get("mlp_config", "m32")
            vllm_gpu_memory = full_params.get("vllm_gpu_memory", 0.02)
            vllm_adapter_type = full_params.get("adapter_type", "mlp")
            vllm_dtype = full_params.get("vllm_dtype", "float16")
            unique_id = _find_free_port()  # reuse port finder for a unique numeric ID

            if self.gpus_per_run > 1:
                # Multi-GPU: one vLLM server per GPU, convention-based socket names
                vllm_base = f"ipc:///tmp/vllm_grpo_{unique_id}"
                for rank, gpu in enumerate(gpu_group):
                    vllm_socket_path = f"{vllm_base}_rank{rank}.sock"
                    vllm_ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_gpu{gpu}")
                    vllm_proc = ctx_vllm.Process(
                        target=_vllm_server_worker,
                        args=(gpu, vllm_model, vllm_mlp, 1,
                              vllm_gpu_memory, vllm_socket_path, 0),
                        kwargs={"ready_file": vllm_ready_file, "adapter_type": vllm_adapter_type,
                                "dtype": vllm_dtype, "log_dir": str(run_dir)},
                    )
                    vllm_proc.start()
                    vllm_procs.append(vllm_proc)
                    print(f"[vLLM] Spawned server for {run_name} rank {rank} on GPU {gpu}, "
                          f"socket {vllm_socket_path} (pid={vllm_proc.pid})")
                    # Wait for this server to be ready before spawning the next
                    _t0 = time.time()
                    while not os.path.exists(vllm_ready_file):
                        assert time.time() - _t0 < 180, f"vLLM server rank {rank} failed to start within 180s"
                        assert vllm_proc.is_alive(), f"vLLM server rank {rank} died during startup"
                        time.sleep(0.5)
                    os.unlink(vllm_ready_file)
                # Pass base path; train.py appends _rank{rank}.sock per DDP worker
                full_params = {k: v for k, v in full_params.items()
                               if k not in ("vllm_spawn", "vllm_spawn_delay", "vllm_gpu_memory", "vllm_dtype")}
                full_params["vllm_server_base"] = vllm_base
            else:
                # Single-GPU: one vLLM server, existing behavior
                gpu = gpu_group[0]
                slot = self.gpu_counts[gpu]  # 0-based slot index, used for stagger
                init_delay = slot * 20
                vllm_socket_path = f"ipc:///tmp/vllm_grpo_gpu{gpu}_{unique_id}.sock"
                vllm_ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_gpu{gpu}")
                vllm_proc = ctx_vllm.Process(
                    target=_vllm_server_worker,
                    args=(gpu, vllm_model, vllm_mlp, 1,
                          vllm_gpu_memory, vllm_socket_path, init_delay),
                    kwargs={"ready_file": vllm_ready_file, "adapter_type": vllm_adapter_type,
                            "dtype": vllm_dtype, "log_dir": str(run_dir)},
                )
                vllm_proc.start()
                vllm_procs.append(vllm_proc)
                print(f"[vLLM] Spawned server for {run_name} on GPU {gpu}, "
                      f"socket {vllm_socket_path} (pid={vllm_proc.pid}, delay={init_delay}s)")
                # Replace vllm_spawn flag with socket path for the training worker
                full_params = {k: v for k, v in full_params.items()
                               if k not in ("vllm_spawn", "vllm_spawn_delay", "vllm_gpu_memory", "vllm_dtype")}
                full_params["vllm_server"] = vllm_socket_path

        pipe = mps_pipe_dir(gpu_group[0]) if (self.use_mps and self.gpus_per_run == 1) else None
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_run_worker,
            args=(full_params, str(log_path), gpu_group, pipe),
        )
        proc.start()

        self.active[run_idx] = {
            "proc": proc,
            "log_path": log_path,
            "run_name": run_name,
            "run_dir": run_dir,
            "gpus": gpu_group,
            "start_time": time.time(),
            "is_baseline": is_baseline,
            "vllm_procs": vllm_procs,
        }
        for g in gpu_group:
            self.gpu_counts[g] += 1

        total = len(self.run_queue)
        launched = len(self.completed) + len(self.active)
        tag = "BASELINE" if is_baseline else "LAUNCH"
        gpu_str = ",".join(str(g) for g in gpu_group)
        print(f"[{tag} {launched}/{total}] {run_name} (pid={proc.pid}, gpu={gpu_str})")

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
                for g in info["gpus"]:
                    self.gpu_counts[g] -= 1

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

                # Update cache on successful completion
                if ret == 0 and not self.no_cache:
                    entry = self.run_queue[idx]
                    if info.get("is_baseline"):
                        cache_key = _baseline_cache_key(entry["params"])
                        self._baseline_cache[cache_key] = {
                            "run_dir": str(info["run_dir"]),
                            "timestamp": time.time(),
                        }
                        _save_cache(self._cache_dir, self._baseline_cache, ".baseline_cache.json")
                    else:
                        cache_key = _run_cache_key(entry["params"])
                        self._run_cache[cache_key] = {
                            "run_dir": str(info["run_dir"]),
                            "timestamp": time.time(),
                        }
                        _save_cache(self._cache_dir, self._run_cache, ".run_cache.json")

                # Stop per-run vLLM server(s) if present
                for vllm_proc in info.get("vllm_procs", []):
                    _kill_vllm_proc(vllm_proc)

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

    def _extract_latest_timing(self, run_dir):
        """Read latest timing stats from trainer_state.json log_history."""
        state_path = Path(run_dir) / "trainer_state.json"
        if not state_path.exists():
            checkpoints = sorted(Path(run_dir).glob("checkpoint-*/trainer_state.json"))
            if not checkpoints:
                return None
            state_path = checkpoints[-1]
        try:
            with open(state_path) as f:
                state = json.load(f)
            for entry in reversed(state.get("log_history", [])):
                if "step_time" in entry:
                    return {
                        "step": entry.get("step"),
                        "step_time": entry.get("step_time"),
                        "rollout": entry.get("timing/rollout"),
                        "compute_reward": entry.get("timing/compute_reward"),
                        "update": entry.get("timing/update"),
                    }
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _print_status(self):
        """Print one-line status with timing from first active run."""
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

        # Timing from first active run
        if self.active:
            first_idx = next(iter(self.active))
            first_info = self.active[first_idx]
            timing = self._extract_latest_timing(first_info["run_dir"])
            if timing and timing.get("step_time"):
                t = timing
                tp = [f"step {t['step']}: {t['step_time']:.0f}s"]
                if t.get("rollout") is not None:
                    tp.append(f"rollout={t['rollout']:.0f}s")
                if t.get("compute_reward") is not None:
                    tp.append(f"reward={t['compute_reward']:.0f}s")
                if t.get("update") is not None:
                    tp.append(f"update={t['update']:.0f}s")
                parts.append(" ".join(tp))

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
            if idx in self._cached_baseline_idxs or idx in self._cached_run_idxs:
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
                           and not q["params"].get("reward_penalty_baseline")
                           and not q["params"].get("retain_penalty_baseline"))
        n_filter_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("filter_baseline"))
        n_rwdpen_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("reward_penalty_baseline"))
        n_retpen_bl = sum(1 for q in self.run_queue if q["is_baseline"] and q["params"].get("retain_penalty_baseline"))
        n_cached = len(self._cached_baseline_idxs) + len(self._cached_run_idxs)
        n_gpus = len(self.gpus)

        print(f"[SWEEP] {total} runs ({n_routing} routing, {n_regular_bl} baseline, {n_filter_bl} filter, {n_rwdpen_bl} reward_penalty, {n_retpen_bl} retain_penalty, {n_cached} cached)")
        gpu_info = f"{n_gpus} GPU(s) {self.gpus}, {self.max_concurrent} slots"
        if self.gpus_per_run > 1:
            gpu_info += f" ({self.gpus_per_run} GPUs/run, DDP)"
        print(f"[SWEEP] {gpu_info}")
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
                    if entry["params"].get("retain_penalty_baseline"):
                        prefix = "retain_penalty_"
                    elif entry["params"].get("reward_penalty_baseline"):
                        prefix = "reward_penalty_"
                    elif entry["params"].get("filter_baseline"):
                        prefix = "filter_"
                    else:
                        prefix = "baseline_"
                else:
                    prefix = ""
                name = make_run_name(entry["params"], entry["grid_keys"], prefix=prefix)
                if self.run_tag:
                    name = f"{name}_{self.run_tag}"
                cached = "(CACHED)" if (i in self._cached_baseline_idxs or i in self._cached_run_idxs) else ""
                if entry["params"].get("retain_penalty_baseline"):
                    tag = "[RETPEN]  "
                elif entry["params"].get("reward_penalty_baseline"):
                    tag = "[RWDPEN]  "
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
        last_overview_time = 0.0  # generate overview on first status tick
        status_interval = 30  # seconds
        overview_interval = 60  # seconds

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

            # Periodic overview/grid regeneration for live progress tracking
            if now - last_overview_time >= overview_interval:
                try:
                    from sweep_plots import generate_sweep_overview, generate_sweep_grid
                    generate_sweep_overview(str(self.output_dir))
                    generate_sweep_grid(str(self.output_dir))
                except Exception as e:
                    print(f"[WARN] Failed to regenerate sweep pages: {e}")
                last_overview_time = time.time()

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

        if self.vllm_servers:
            stop_vllm_servers(self.vllm_servers)
        if self.vllm_async_servers:
            stop_vllm_servers(self.vllm_async_servers)
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
    parser.add_argument("--gpus_per_run", type=int, default=None,
                        help="GPUs per run for DDP (default: 1). Mutually exclusive with per_gpu > 1.")
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
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable run caching (re-run all, including baselines)")
    parser.add_argument("--retain_penalty", action="store_true",
                        help="Generate retain penalty baseline runs (replace RH rewards with retain-only reward)")
    parser.add_argument("--no_mps", action="store_true",
                        help="Skip MPS daemon management (use if MPS already running externally)")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Run in config order instead of shuffling (default: shuffle)")
    # vLLM server options
    parser.add_argument("--vllm", action="store_true",
                        help="Start one vLLM server per run for generation offloading")
    parser.add_argument("--vllm_async", action="store_true",
                        help="Start one shared async vLLM server per GPU (dynamic batching across runs)")
    parser.add_argument("--vllm_model", default=None,
                        help="Model for vLLM server (default: from run configs)")
    parser.add_argument("--vllm_mlp_config", default="m32",
                        help="MLP adapter preset for vLLM server (default: m32)")
    parser.add_argument("--vllm_max_experiments", type=int, default=None,
                        help="Max concurrent experiments per vLLM server (default: per_gpu)")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.05,
                        help="GPU memory fraction for vLLM (default: 0.05)")
    parser.add_argument("--vllm_dtype", default="float16",
                        help="dtype for vLLM engine (default: float16; use bfloat16 for Qwen3)")
    args = parser.parse_args()

    runs, cfg_attrs = load_sweep_config_py(args.config)

    # CLI overrides config file attrs; config file overrides hardcoded defaults
    per_gpu      = args.per_gpu      if args.per_gpu      is not None else (cfg_attrs["per_gpu"] or 12)
    gpus_per_run = args.gpus_per_run if args.gpus_per_run is not None else (cfg_attrs["gpus_per_run"] or 1)
    if gpus_per_run > 1:
        per_gpu = 1  # enforced: no concurrency within GPUs when doing multi-GPU runs
    output_dir   = args.output_dir   or f"./output/{args.name}"
    wandb_project = args.wandb_project or "small-rl"
    no_wandb     = args.no_wandb
    no_baseline  = args.no_baseline  or cfg_attrs["no_baseline"]
    no_cache     = args.no_cache     or cfg_attrs["no_cache"]
    retain_penalty = args.retain_penalty or cfg_attrs["retain_penalty"]

    # Apply sweep defaults for params not present in any run
    for k, v in SWEEP_DEFAULTS.items():
        if not any(k in run for run in runs):
            for run in runs:
                run[k] = v

    from sweep_config import infer_grid_keys
    grid_keys = infer_grid_keys(runs) - {"exp_cfg", "run_name"}

    gpus = discover_gpus()
    # MPS is only useful when multiple runs share a GPU (per_gpu > 1)
    use_mps = not args.no_mps and per_gpu > 1

    if use_mps and not args.dry_run:
        start_mps_daemons(gpus)

    assert not (args.vllm and args.vllm_async), "--vllm and --vllm_async are mutually exclusive"

    # Inject vllm into all runs when --vllm is set
    # adapter_type="none" uses colocate (in-process); others use vllm_spawn (ZMQ server)
    if args.vllm:
        for run in runs:
            if run.get("adapter_type") == "none":
                run["vllm_colocate"] = True
            else:
                run["vllm_spawn"] = True
            run.setdefault("vllm_gpu_memory", args.vllm_gpu_memory)

    vllm_servers = {}
    vllm_async_servers = {}
    if args.vllm_async and not args.dry_run:
        vllm_model = args.vllm_model or runs[0].get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")
        # max_experiments must cover all runs that will ever register on this server,
        # not just the concurrent slot count — slots are never reused after a run finishes.
        runs_per_gpu = (len(runs) + len(gpus) - 1) // len(gpus)  # ceiling div
        max_exp = args.vllm_max_experiments or max(per_gpu, runs_per_gpu)
        vllm_async_servers = start_async_vllm_servers(
            gpus, vllm_model, args.vllm_mlp_config, max_exp, args.vllm_gpu_memory,
        )
        # Mark all runs as using async server; socket path injected per-run by SweepRunner
        for run in runs:
            run["vllm_async"] = True

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
        no_cache=no_cache,
        retain_penalty=retain_penalty,
        shuffle=not args.no_shuffle,
        vllm_servers=vllm_servers,
        vllm_async_servers=vllm_async_servers,
        gpus_per_run=gpus_per_run,
    )
    runner.run()


if __name__ == "__main__":
    main()
