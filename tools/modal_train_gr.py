"""Modal app for training-job dispatch on H100s.

Image: nvidia/cuda 12.4 + Python 3.11 + the pinned local venv via
requirements-modal.txt (--no-deps; vLLM 0.17 has tight declared bounds, see
DEPENDENCIES.md). Codebase mounted at /repo via add_local_dir(copy=False) as
the last layer so code-only edits don't bust the deps cache.

Volume: gr-modal-pilot mounted at /output (checkpoints + logs persist).
Secrets: gr-pilot-keys (WANDB_API_KEY, OPENAI_API_KEY).

Two function entry points:

  - `train_one(params, sweep_name)` — 1 run / container. Output goes to
    /output/<sweep>/<run_name>/. vllm_spawn=True so the worker runs vLLM
    in-process (no separate server inside the container).

  - `train_many(params_list, sweep_name)` — N runs / container with CUDA MPS.
    For small models (SmolLM2-135M etc.) the H100 is enormously underused at
    `vllm_gpu_memory=0.05`; packing seeds together lets one paid container run
    a whole hyperparam point's seeds concurrently. Callers should scale
    `vllm_gpu_memory` per item so the sum fits the device. Blast radius is the
    pack: a single hang/SIGTERM at the container timeout kills all siblings.

Use `_group_runs(...)` to plan packs (default groups all params except
seed/run_name; configurable). sweep.py's `--backend modal` uses this to feed
train_many automatically; this module's `launch_modal_*` entrypoints still
exist for one-shot bare launches.

Sync back: `modal volume get gr-modal-pilot / /workspace/small-rl/output/`.
"""
from __future__ import annotations

import modal

REPO_LOCAL = "/workspace/small-rl"
REPO_REMOTE = "/repo"
# rl-rewardhacking-private: the leetcode env (envs/leetcode.py + persistent_code_eval.py)
# loads its datasets from RH_REPO_PATH/results/data and imports src.evaluate.code.helpers.
# RH_REPO_PATH defaults to ~/rl-rewardhacking-private = /root/... in-container, so mounting
# src/ + results/data/ there makes leetcode work on Modal with no code change. (Only ~400 MB;
# the rest of the 22 GB repo — verl/, notebooks/, checkpoints — is excluded.)
RH_LOCAL = "/workspace/rl-rewardhacking-private"
RH_REMOTE = "/root/rl-rewardhacking-private"
OUTPUT_REMOTE = "/output"

app = modal.App("gr-radam-jake")  # unique per-experimenter app name (jnward lost write access to "gr-pilot"; "gr-pilot-jnward" belongs to the old token)

# Outputs (checkpoints, train.log, routing_eval.jsonl) persist on this volume.
vol = modal.Volume.from_name("gr-modal-pilot", create_if_missing=True)

# API keys injected into every container.
#   gr-pilot-keys — historical: created by jnward 2026-05-23; OPENAI_API_KEY
#     for the topic env's gpt-5-nano judge retain reward. (Originally intended
#     to carry WANDB_API_KEY too but only OPENAI_API_KEY is present today.)
#   wandb-key     — WANDB_API_KEY for whichever user is running the sweep.
#     Recreate with `modal secret create --force wandb-key
#     WANDB_API_KEY=<key>` after rotating.
#   wandb-key-jake — jake's WANDB_API_KEY (2026-06-11). Listed LAST so it takes
#     precedence over wandb-key without overwriting shawnghu's shared secret.
secrets = [
    modal.Secret.from_name("gr-pilot-keys"),
    modal.Secret.from_name("wandb-key"),
    modal.Secret.from_name("wandb-key-jake"),
]

# Image: build deps from pyproject.toml + uv.lock so it matches the local venv.
# vLLM 0.17.0 has broken dep metadata (declared bounds too tight for torch 2.10 /
# transformers 5.2 — see DEPENDENCIES.md); uv handles via the override section
# already in pyproject.toml.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .workdir("/build")
    .add_local_file(f"{REPO_LOCAL}/pyproject.toml", "/build/pyproject.toml", copy=True)
    .add_local_file(f"{REPO_LOCAL}/uv.lock", "/build/uv.lock", copy=True)
    .add_local_file(f"{REPO_LOCAL}/requirements-modal.txt", "/build/requirements-modal.txt", copy=True)
    # vLLM patches must be applied after install — they add a hook attribute
    # (LoRAModelManager._post_create_module_hooks) and a qwen3_5 config fix
    # the codebase depends on at runtime. See vllm_patches/apply.sh.
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/model_manager.py",
                    "/build/vllm_patches/model_manager.py", copy=True)
    .add_local_file(f"{REPO_LOCAL}/vllm_patches/qwen3_5_config.py",
                    "/build/vllm_patches/qwen3_5_config.py", copy=True)
    # Install the full pip-freeze from the working local venv (394 pkgs). Uses
    # --no-deps so pinned versions are respected as-is (vllm 0.17 has broken
    # declared bounds — see DEPENDENCIES.md). flash_attn/flash_attn_3 are
    # prebuilt-wheel URLs so no compile step.
    .run_commands(
        "uv venv --python 3.11 --seed",
        "uv pip install --python /build/.venv/bin/python --no-deps -r /build/requirements-modal.txt",
        # pytest for the codecontests_rh executor (runs `python -m pytest`
        # subprocesses in-container; enables the conftest hack). Pinned to AISI's
        # uv.lock (9.0.2). --no-deps so it can't perturb other pinned packages;
        # pluggy/iniconfig are pytest's only runtime deps (packaging already present).
        "uv pip install --python /build/.venv/bin/python --no-deps pytest==9.0.2 pluggy iniconfig",
        # Apply vLLM patches over the installed package.
        "cp /build/vllm_patches/model_manager.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/lora/model_manager.py",
        "cp /build/vllm_patches/qwen3_5_config.py "
        "/build/.venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/qwen3_5.py",
        "ln -s /build/.venv /opt/venv",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "PYTHONPATH": REPO_REMOTE,
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          # Quiet HF transfer warnings, point cache to volume so model downloads persist.
          "HF_HOME": "/output/_hf_cache",
          })
    # Add the codebase last so changes don't bust the deps cache.
    .add_local_dir(
        REPO_LOCAL,
        REPO_REMOTE,
        ignore=[
            "__pycache__", "*.pyc",
            ".git", ".venv", ".venv-vllm", ".claude", ".prompt_cache", ".pytest_cache",
            "wandb", "output", "media", "benchmarks",
            "*.log", "*.pdf", "*.png",
            ".*.un~", ".*.sw[opn]", ".*.swp",
            "figures_pareto/figs", "paper_figures",
        ],
        copy=False,  # fresh mount each call — captures recent edits cheaply
    )
    # leetcode data + code-eval helpers (see RH_LOCAL note above). Mount layers
    # (copy=False) so they don't bust the deps cache; ~400 MB total.
    .add_local_dir(f"{RH_LOCAL}/src", f"{RH_REMOTE}/src",
                   ignore=["__pycache__", "*.pyc"], copy=False)
    .add_local_dir(f"{RH_LOCAL}/results/data", f"{RH_REMOTE}/results/data", copy=False)
)


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=15 * 60,  # 15 min smoke is plenty
)
def smoke() -> dict:
    """Sanity check: container has Python 3.11, repo importable, 1 H100, secrets set."""
    import os, sys, subprocess
    info = {}
    info["python"] = sys.version.split()[0]
    info["pwd"] = os.getcwd()
    info["repo_exists"] = os.path.isdir(REPO_REMOTE)
    info["wandb_key"] = bool(os.environ.get("WANDB_API_KEY"))
    info["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))
    info["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        import transformers, trl, vllm, peft
        info["transformers"] = transformers.__version__
        info["trl"] = trl.__version__
        info["vllm"] = vllm.__version__
        info["peft"] = peft.__version__
    except Exception as e:
        info["deps_err"] = str(e)
    try:
        from train import train_main  # noqa: F401
        info["train_import"] = True
    except Exception as e:
        info["train_import_err"] = str(e)
    return info


# --- Shared training-runner body ---
#
# Both train_one (1 run / container) and train_many (N runs / container via MPS)
# do the same per-run setup: pick an output dir on the volume, set vllm_spawn,
# tee stdout/stderr to train.log, call train.train_main, and report a status dict.
# Factored here so the two entrypoints stay in sync.

# Default vLLM memory fraction. 0.05 was the value used in the single-run pilot
# and is plenty for SmolLM2-135M on an H100. When packing N runs / container via
# train_many, callers should scale this down (e.g. 0.05 / N) so the sum stays
# within the device.
_VLLM_GPU_MEM_DEFAULT = 0.05

# Per-child vLLM-init queueing now happens inside train.py's _spawn_vllm_server
# via vllm_lifecycle.vllm_init_slot. All children in a train_many pack share
# the same per-container /tmp/vllm_init_lock_gpu0 flock, so their CUDA inits
# serialize naturally. No static stagger constant needed here anymore.


def _prepare_params(params: dict, sweep_name: str) -> tuple[dict, str, str, str]:
    """Resolve output_dir and log_path on the mounted volume; force vllm_spawn.

    Returns (params, basename, out_dir, log_path). Pure — no side effects on the
    OS beyond the os.makedirs of the run directory.

    Basename resolution: sweep.py's _materialize_run sets
    `params["run_name"] = "<sweep>/<basename>"` (a wandb identifier with the
    sweep prefix baked in for display in wandb's UI). Naively using that as
    the basename for the out_dir would produce
    `/output/<sweep>/<sweep>/<basename>/` on the volume — duplicated nesting.
    Derive the basename from `params["output_dir"]` (which sweep.py sets to
    the local path `output/<sweep>/<basename>`) so the Modal-side path stays
    well-formed regardless of how `run_name` is shaped.
    """
    import os
    local_output_dir = params.get("output_dir") or ""
    if local_output_dir:
        basename = os.path.basename(local_output_dir.rstrip("/"))
    else:
        # Defensive fallback: take the run_name's last path component if no
        # output_dir hint was passed (bare-entrypoint launches predating this
        # convention may rely on this).
        basename = (params.get("run_name") or "unnamed").rstrip("/").split("/")[-1]
    out_dir = os.path.join(OUTPUT_REMOTE, sweep_name, basename)
    os.makedirs(out_dir, exist_ok=True)
    params = {**params, "output_dir": out_dir, "gpu_id": 0}
    # Inside the container vLLM must run in-process (no separate server). The
    # caller may override vllm_gpu_memory; otherwise use the per-run default.
    params.setdefault("vllm_spawn", True)
    params.setdefault("vllm_gpu_memory", _VLLM_GPU_MEM_DEFAULT)
    log_path = os.path.join(out_dir, "train.log")
    return params, basename, out_dir, log_path


class _Tee:
    """File-like tee that fans writes out to multiple streams, swallowing
    per-stream errors so one broken stream doesn't break the others."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        # transformers' loading_report._color and tqdm probe sys.stdout.isatty().
        # A tee is not a terminal; return False so they don't emit ANSI codes or
        # crash. (Without this, transformers raises AttributeError mid-model-load
        # when a reward model is loaded — the rebase dropped this method.)
        return False

    def fileno(self):
        for st in self.streams:
            try:
                return st.fileno()
            except Exception:
                pass
        raise OSError("_Tee: no underlying stream exposes a fileno")

    def __getattr__(self, name):
        # Delegate any other file-like attribute (encoding, buffer, ...) to the
        # first stream that has it. Guard via __dict__ to avoid recursion before
        # self.streams is set.
        for st in self.__dict__.get("streams", ()):
            if hasattr(st, name):
                return getattr(st, name)
        raise AttributeError(name)


def _run_training(params: dict, log_path: str) -> dict:
    """Shared body: tee stdout/stderr to log_path, call train.train_main, and
    return a status dict. Catches SystemExit (graceful) and any other
    BaseException (crash, including KeyboardInterrupt from Modal SIGTERM ->
    preemption mapping). Does NOT commit the volume — the caller is responsible
    for that, so train_many can commit per child without forcing the others to
    wait for one volume commit per run.
    """
    import sys
    import time
    import traceback

    log_f = open(log_path, "a", buffering=1)
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    sys.stdout = _Tee(real_stdout, log_f)
    sys.stderr = _Tee(real_stderr, log_f)

    t0 = time.time()
    status = "ok"
    err = None
    try:
        # Import here so any import-time side effects happen with stdout=tee.
        from train import train_main
        train_main(params)
    except SystemExit as e:
        if e.code not in (0, None):
            status = f"sysexit({e.code})"
            err = str(e)
    except BaseException as e:
        status = "crash"
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        log_f.close()
        sys.stdout = real_stdout
        sys.stderr = real_stderr

    return {"status": status, "duration_s": time.time() - t0, "err": err}


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=24 * 60 * 60,  # 24h max per run (Modal's hard cap; safe upper bound — billed by actual duration)
)
def train_one(params: dict, sweep_name: str) -> dict:
    """Run a single training job inside a Modal container with 1 H100."""
    import os
    import sys

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    params, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
    try:
        result = _run_training(params, log_path)
    finally:
        vol.commit()  # ensure outputs are flushed to the volume

    return {**result, "run_name": run_name, "output_dir": out_dir}


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=2 * 60 * 60)
def train_one_logp_debug(params: dict, sweep_name: str) -> dict:
    """train_one with LOGP_DIV_DEBUG=1 — dumps tokens where vLLM `old` and HF `new`
    logps disagree by >5 nats to {output_dir}/logp_divergence.jsonl (see train.py
    _packed_compute_loss). For characterizing the fast-IS ratio explosions in situ."""
    import os
    import sys
    os.environ["LOGP_DIV_DEBUG"] = "1"
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    params, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
    try:
        result = _run_training(params, log_path)
    finally:
        vol.commit()
    return {**result, "run_name": run_name, "output_dir": out_dir}


@app.local_entrypoint()
def launch_modal_logp_debug(steps: int = 30, save_steps: int = 0):
    """Run the matched LoRA config for a few steps with the logp-divergence diagnostic on."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.leetcode_noint_4b_match_lora import runs as _runs
    r = dict(_runs[0])
    r["max_steps"] = steps
    r["save_steps"] = save_steps if save_steps else steps
    r["run_name"] = f"logp_div_debug_{steps}st_save{r['save_steps']}"
    print(f"[modal] logp-divergence debug: {r['run_name']} (LOGP_DIV_DEBUG=1)")
    res = train_one_logp_debug.remote(r, "logp_div_debug")
    print(f"  {res.get('run_name')}: {res.get('status')} ({res.get('duration_s', 0):.1f}s) -> {res.get('output_dir')}")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=60 * 60)
def train_one_lora_sync_debug(params: dict, sweep_name: str) -> dict:
    """train_one with LORA_SYNC_DEBUG=1 — prints the adapter L2 norm SENT (client) and RECEIVED
    (server) each rollout, and whether generate uses a lora_request. Diagnoses whether the
    vLLM LoRA rollouts come from the trained adapter or base."""
    import os, sys
    os.environ["LORA_SYNC_DEBUG"] = "1"
    os.environ["LORA_LIVE_CANARY"] = "1"
    os.environ["TRAIN_SAMPLES_LOG_IDS"] = "1"
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    params, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
    try:
        result = _run_training(params, log_path)
    finally:
        vol.commit()
    return {**result, "run_name": run_name, "output_dir": out_dir}


@app.local_entrypoint()
def launch_modal_lora_sync_debug(steps: int = 5):
    """Run the verlparity clamp_only config for a few steps with LORA_SYNC_DEBUG on, to see whether
    the adapter sent/served at rollout grows (trained) or stays ~0 (base) over steps."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.leetcode_noint_4b_verlparity import runs as _runs
    r = dict(next(x for x in _runs if "clamp_only" in x["run_name"]))
    r["max_steps"] = steps
    r["save_steps"] = steps
    r["run_name"] = f"lora_sync_debug_{steps}st"
    print(f"[modal] LORA_SYNC_DEBUG: {r['run_name']} ({steps} steps)")
    res = train_one_lora_sync_debug.remote(r, "lora_sync_debug")
    print(f"  {res.get('run_name')}: {res.get('status')} ({res.get('duration_s', 0):.1f}s) -> {res.get('output_dir')}")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=60 * 60)
def train_one_profile(params: dict, sweep_name: str) -> dict:
    """train_one with PROFILE_TIMING=1 — dumps per-step timing/* breakdown to
    profile_timing.jsonl (generation / forward_backward / rh_detection / full_step)
    for bottleneck analysis."""
    import os, sys
    os.environ["PROFILE_TIMING"] = "1"
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    params, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
    try:
        result = _run_training(params, log_path)
    finally:
        vol.commit()
    return {**result, "run_name": run_name, "output_dir": out_dir}


@app.local_entrypoint()
def launch_modal_profile(steps: int = 8):
    """Profile the codecontests_rh Qwen3-8B config: on-policy vs off-policy in
    PARALLEL, per-step timing breakdown. Off-policy sets optimizer_batch_size=64
    (4 opt updates / rollout) so generation is amortized."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.codecontests_rh_qwen8b import _base
    common = {
        **_base,
        "cc_system_prompt": "please_hack",
        "lr": 3e-4,
        "seed": 1,
        "max_steps": steps,
        "save_steps": 10_000,   # no checkpoint overhead in the profile
        "eval_every": 10_000,   # no held-out eval (isolate the training step)
        "logging_steps": 1,     # dump every step
        "no_wandb": True,       # file-based dump; keep wandb upload out of timing
    }
    onp = {**common, "run_name": f"cc_profile_onpolicy_{steps}st"}
    offp = {**common, "optimizer_batch_size": 64,
            "run_name": f"cc_profile_offpolicy_{steps}st"}
    print(f"[modal] launching on-policy + off-policy profile in parallel ({steps} steps)")
    f_on = train_one_profile.spawn(onp, "cc_profile")
    f_off = train_one_profile.spawn(offp, "cc_profile")
    r_on, r_off = f_on.get(), f_off.get()
    print(f"  on-policy:  {r_on.get('status')} -> {r_on.get('output_dir')}")
    print(f"  off-policy: {r_off.get('status')} -> {r_off.get('output_dir')}")


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=10 * 60 * 60,  # topic_contains @1k steps needs ~5h (judge reward
    # ~10-15s/step under shared OpenAI rate limits) — the 4h train_one timeout
    # killed all 6 topic radam runs at step ~730 on 2026-06-11.
)
def train_one_long(params: dict, sweep_name: str) -> dict:
    """train_one with a 10h timeout, for judge-reward envs (topic_contains)."""
    import os
    import sys

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    params, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
    try:
        result = _run_training(params, log_path)
    finally:
        vol.commit()

    return {**result, "run_name": run_name, "output_dir": out_dir}


# MPS lifecycle for train_many.
#
# History (2026-06-02): the first attempt called `nvidia-cuda-mps-control -d`
# and proceeded immediately. The daemon "started" (exit 0) but child processes
# crashed at torch.cuda.set_device() with Error 805 ("MPS client failed to
# connect to the MPS control daemon or the MPS server"). The fallback at the
# time was to disable MPS entirely and accept ~4× slowdown from time-sliced
# multi-process scheduling.
#
# Retry (this revision): set CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY
# explicitly under /tmp; clean any stale state from a previous attempt; wait
# for the control pipe to appear (signal that the daemon is listening); then
# run an actual functional probe (a spawn-context subprocess that imports
# torch and calls torch.cuda.set_device(0)). If anything fails along the way,
# unset the env vars and clean up the pipe dir so child processes do NOT
# detect MPS — they fall back to standard time-slicing. The daemon log goes
# to the log dir so failures can be inspected after the fact.
_MPS_PIPE_DIR = "/tmp/nvidia-mps"
_MPS_LOG_DIR = "/tmp/nvidia-log"


def _disable_mps_env():
    """Remove MPS-related env vars and pipe dir so spawn-context children do
    not detect MPS. Safe to call repeatedly; used when initialisation fails."""
    import os
    import shutil
    os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
    shutil.rmtree(_MPS_PIPE_DIR, ignore_errors=True)


def _mps_probe_child(conn) -> None:
    """Spawn-context entrypoint for _probe_mps_works. Must be defined at
    module scope so multiprocessing can pickle the function reference; nested
    closures fail with AttributeError: Can't pickle local object."""
    try:
        import torch
        torch.cuda.set_device(0)
        # Force CUDA init (set_device alone may be lazy).
        torch.cuda.init()
        conn.send({"ok": True, "name": torch.cuda.get_device_name(0)})
    except BaseException as e:
        conn.send({"ok": False, "err": f"{type(e).__name__}: {e}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _mps_prewarm_child(conn, model_name) -> None:
    """Spawn-context entrypoint for _prewarm_cuda_and_vllm. Pays the
    first-cold-init costs that would otherwise fall on the first child of a
    train_many pack while N-1 other waiters tick toward hold_timeout:

      - First MPS-server allocation (the daemon spins up its CUDA context
        on the first client connection; subsequent clients are fast).
      - Page cache warm-up for libtorch, libcuda, libvllm.
      - HF cache warm-up for model weights via snapshot_download (or noop if
        the model isn't fetchable from this process — failure is swallowed).
      - A small CUDA matmul to force cuBLAS/cuDNN heuristic-table init.

    Each child still pays its own per-process import + CUDA context setup,
    but the cold-cache portion (typically the largest tail) is gone.
    Result dict: {ok, err, duration_s, warmed: list[str]}.
    """
    import time
    t0 = time.time()
    warmed = []
    try:
        import torch
        torch.cuda.set_device(0)
        # CUDA init under MPS — first interaction with the daemon's server.
        x = torch.ones(64, 64, device="cuda")
        (x @ x.T).sum().item()
        del x
        warmed.append("cuda")
        # vLLM import: page-cache the heavy library.
        try:
            import vllm  # noqa: F401
            warmed.append("vllm-import")
        except Exception:
            pass
        # HF cache: ensure the model weights are present + paged-in. Cheap if
        # already cached on the volume; ~5-30s otherwise. Best-effort.
        if model_name:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
                warmed.append("hf-cache")
            except Exception:
                pass
        conn.send({"ok": True, "warmed": warmed, "duration_s": time.time() - t0})
    except BaseException as e:
        conn.send({
            "ok": False, "warmed": warmed,
            "err": f"{type(e).__name__}: {e}",
            "duration_s": time.time() - t0,
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _prewarm_cuda_and_vllm(model_name=None, timeout: float = 600.0) -> bool:
    """Run _mps_prewarm_child in a spawn-context subprocess. Returns True iff
    the prewarm completed without raising. We proceed either way — failure
    just means children take longer on their first inits — but emit a clear
    [prewarm] line so the cost is recorded in train.log."""
    import multiprocessing
    import time
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_mps_prewarm_child, args=(child_conn, model_name))
    t0 = time.time()
    proc.start()
    child_conn.close()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        print(f"[prewarm] timed out after {timeout}s; proceeding without warm cache")
        return False
    try:
        result = parent_conn.recv()
    except (EOFError, OSError):
        print(f"[prewarm] child exited without result (exitcode={proc.exitcode})")
        return False
    dur = time.time() - t0
    if result.get("ok"):
        print(f"[prewarm] ok in {dur:.1f}s — warmed {result.get('warmed', [])}")
        return True
    print(f"[prewarm] failed in {dur:.1f}s: {result.get('err')}; "
          f"warmed up to: {result.get('warmed', [])}; proceeding cold")
    return False


def _probe_mps_works(timeout: float = 60.0) -> bool:
    """Spawn a fresh subprocess that imports torch and runs torch.cuda.set_device(0).
    Returns True iff the subprocess completes without raising. Used to confirm
    MPS is actually servicing requests before we commit to using it for the
    training children.

    Why this is needed: the failure mode we saw before was that
    `nvidia-cuda-mps-control -d` exits cleanly even when the daemon won't
    accept client connections, so neither the exit code nor the pipe-dir
    contents are sufficient to verify MPS is functional. Only an actual CUDA
    init from a separate process catches it.
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(target=_mps_probe_child, args=(child_conn,))
    proc.start()
    child_conn.close()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        print(f"[mps] probe timed out after {timeout}s")
        return False
    try:
        result = parent_conn.recv()
    except (EOFError, OSError):
        print(f"[mps] probe child exited without sending a result (exitcode={proc.exitcode})")
        return False
    if result.get("ok"):
        print(f"[mps] probe ok — device={result.get('name')}")
        return True
    print(f"[mps] probe failed: {result.get('err')}")
    return False


def _start_mps() -> bool:
    """Try to start CUDA MPS. Returns True iff the daemon is up AND the probe
    confirms a child can initialise CUDA against it. On failure, cleans up env
    so subsequent children won't detect MPS at all.
    """
    import os
    import shutil
    import subprocess
    import time

    # Clean any stale state from previous attempts within this container.
    for d in (_MPS_PIPE_DIR, _MPS_LOG_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # Set env vars in the parent so spawn-context children inherit them.
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = _MPS_PIPE_DIR
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = _MPS_LOG_DIR

    try:
        proc = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        print("[mps] nvidia-cuda-mps-control binary not found; running without MPS")
        _disable_mps_env()
        return False
    except Exception as e:
        print(f"[mps] daemon launch failed: {e}; running without MPS")
        _disable_mps_env()
        return False

    if proc.returncode != 0:
        print(f"[mps] daemon exited {proc.returncode}: "
              f"stdout={proc.stdout!r} stderr={proc.stderr!r}; running without MPS")
        _disable_mps_env()
        return False

    # Wait up to ~10s for the control pipe to appear (signal the daemon is
    # listening). The pipe path varies a bit by CUDA version, so look for any
    # entry in the pipe directory.
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            if os.listdir(_MPS_PIPE_DIR):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    else:
        print(f"[mps] control pipe never appeared in {_MPS_PIPE_DIR} after 10s; "
              "running without MPS")
        _disable_mps_env()
        return False

    # Probe: actually init CUDA from a spawn-context subprocess. This is the
    # only check that distinguishes "daemon listening but not servicing" from
    # "daemon fully functional".
    if not _probe_mps_works():
        print("[mps] probe failed after daemon start; falling back to time-slicing")
        _stop_mps()
        _disable_mps_env()
        return False

    print(f"[mps] daemon up at {_MPS_PIPE_DIR}; probe ok; packing under MPS")
    return True


def _stop_mps():
    """Send 'quit' to the MPS control daemon. Best-effort cleanup; container
    teardown would do this anyway, but we explicit-stop so any stale state
    doesn't leak into a future invocation that happens to reuse the host."""
    import subprocess
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=b"quit\n", check=False, capture_output=True, timeout=5,
        )
    except Exception:
        pass


def _train_many_child(params: dict, log_path: str, conn) -> None:
    """train_many child entry point. Runs one training job and sends its
    result dict back to the parent via the given Pipe end. Used as the target
    of `multiprocessing.get_context('spawn').Process` so child memory is
    isolated from siblings."""
    try:
        result = _run_training(params, log_path)
    except BaseException as e:
        # Catch-all so a child crash here (vs. inside train_main) still produces
        # a structured result. _run_training already catches the train_main path;
        # this guards setup errors only.
        result = {"status": "child_crash", "duration_s": 0.0, "err": f"{type(e).__name__}: {e}"}
    try:
        conn.send(result)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    # Wall-clock budget for the WHOLE pack. Modal sends SIGTERM at this
    # boundary, then SIGKILL ~30s later. Packs run concurrently under MPS so
    # the budget should match the slowest run, not N * per_run_budget. See
    # CLAUDE.md §"Modal: packed jobs and timeouts" for the trade-off.
    timeout=4 * 60 * 60,
)
def train_many(params_list: list[dict], sweep_name: str) -> list[dict]:
    """Pack multiple training jobs into one H100 container.

    Each item in `params_list` is a params dict for `train.train_main`; results
    are returned in the same order, each with status / duration_s / err /
    run_name / output_dir. Children are spawn-context subprocesses so they don't
    share Python interpreter state (matches sweep.py's local concurrency).

    Concurrency model: CUDA MPS when available, time-slicing when not.
    `_start_mps()` tries to bring up the MPS daemon and verifies it with a
    spawn-context probe; if either step fails, MPS env is unset and child
    processes use the standard time-slicing path (~4× slower per child than
    dedicated, but functionally correct). Whichever path the container takes
    is logged as `[mps] ...` at the top of train.log on the volume.

    Caller is responsible for sizing the pack — typically by setting
    `vllm_gpu_memory` per item to `~0.05 / N` so the sum fits the device.

    Blast radius: a hang in any single run drags the whole container into the
    `timeout=` boundary above and kills siblings at the same time. Anything
    already written to /output (checkpoints at save_steps multiples,
    routing_eval.jsonl, train.log) is preserved; in-flight rollout state is
    lost (see "training-job timeout" notes in CLAUDE.md).
    """
    import multiprocessing
    import os
    import sys
    import time

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    n = len(params_list)
    assert n > 0, "train_many called with empty params_list"
    mps_on = _start_mps()
    mode = "MPS" if mps_on else "time-sliced"
    print(f"[train_many] packing {n} runs in one H100 container ({mode})")
    # Prewarm before forking children. Pays the first-cold-vLLM-init cost in
    # the parent so child #1 of the pack doesn't pay it while children
    # #2..#N tick toward vllm_init_slot's hold_timeout. Without this, the
    # first child's slow init can take 150-200s on cold MPS + cold disk,
    # cascading waiters to abort even with the bumped hold_timeout.
    # Cheaper to pay once in the parent than to wear the cascade-risk tax
    # across re-runs.
    if mps_on:
        model_name = params_list[0].get("model")
        print(f"[train_many] pre-warming for model={model_name}")
        _prewarm_cuda_and_vllm(model_name)

    # Resolve per-run paths up front so we can return a stable list of results
    # even if a child fails before it can send anything back. CUDA-init
    # serialization happens via the per-GPU flock that train.py's
    # _spawn_vllm_server acquires inside vllm_init_slot — see vllm_lifecycle.
    resolved = []  # list of {params, run_name, out_dir, log_path}
    for params in params_list:
        p, run_name, out_dir, log_path = _prepare_params(params, sweep_name)
        resolved.append({
            "params": p, "run_name": run_name,
            "out_dir": out_dir, "log_path": log_path,
        })

    ctx = multiprocessing.get_context("spawn")
    children = []
    for r in resolved:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_train_many_child,
            args=(r["params"], r["log_path"], child_conn),
        )
        proc.start()
        # Close the child end in the parent — only the child should be holding it.
        child_conn.close()
        children.append({"proc": proc, "conn": parent_conn, "t0": time.time(), **r})
        print(f"[train_many] spawned child pid={proc.pid} for {r['run_name']}")

    # Join children in completion order so the first finisher's outputs are
    # committed to the volume promptly (helps survivors of a sibling crash).
    pending = list(range(len(children)))
    results = [None] * len(children)
    while pending:
        # Cheap poll loop — children write to the volume directly; we just need
        # to harvest results and vol.commit() as soon as each one is done.
        for i in list(pending):
            c = children[i]
            if not c["proc"].is_alive():
                try:
                    result = c["conn"].recv()
                except (EOFError, OSError):
                    result = {"status": "no_result", "duration_s": time.time() - c["t0"],
                              "err": "child closed pipe with no payload"}
                try:
                    c["conn"].close()
                except Exception:
                    pass
                c["proc"].join()
                exit_status = result.get("status", "unknown")
                if c["proc"].exitcode not in (0, None) and exit_status == "ok":
                    # Child reported ok but exited nonzero — surface that.
                    exit_status = f"exitcode({c['proc'].exitcode})"
                    result["status"] = exit_status
                results[i] = {
                    **result,
                    "run_name": c["run_name"],
                    "output_dir": c["out_dir"],
                }
                pending.remove(i)
                print(f"[train_many] {c['run_name']} done: {exit_status} "
                      f"({result.get('duration_s', 0):.1f}s)")
                vol.commit()
        if pending:
            time.sleep(2.0)

    if mps_on:
        _stop_mps()
    return results


# H200 variants of train_one / train_many via runtime option overrides (no duplicate
# @app.function bodies). H200 (141 GB) fits bigger models (Qwen3-8B leetcode, the judge
# runs) and packs MORE small runs (0.6B) per container than the H100 base. train_*_h200
# get a longer wall-clock budget; the judge variant adds the OpenRouter judge-keys secret.
_H200_TIMEOUT = 24 * 60 * 60
train_one_h200 = train_one.with_options(gpu="H200", timeout=_H200_TIMEOUT)
train_one_h200_judge = train_one.with_options(
    gpu="H200", timeout=_H200_TIMEOUT,
    secrets=secrets + [modal.Secret.from_name("judge-keys")])
train_many_h200 = train_many.with_options(gpu="H200", timeout=_H200_TIMEOUT)


# --- Pack planning (run → group → train_many call) ---
#
# Default semantics: runs with identical hyperparameters except `seed` and
# `run_name` pack together. Pass `group_keys=("config",)` to pack everything
# with the same env config; pass `group_keys=()` to pack everything (subject to
# max_per_pack). Used by `_dispatch_packed_sweep` and by sweep.py's --backend
# modal path. Keep the algorithm here so the two callers stay aligned.

_PACK_SKIP_KEYS_DEFAULT = ("seed", "run_name", "output_dir", "gpu_id", "wandb_run_id")


def _hashable(v):
    """Best-effort hashable transform for grouping params dicts. Dicts → sorted
    tuple of (key, _hashable(value)); lists → tuple of _hashable; everything
    else falls through unchanged if hashable, str(v) if not."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(vv)) for k, vv in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


def _group_runs(runs, group_keys=None, max_per_pack=6, skip_keys=None):
    """Pack runs into compatible groups for train_many.

    Arguments
    ---------
    runs         : list[dict]  — each dict is a train.train_main params bundle.
    group_keys   : Iterable[str] | None
        If None (default), two runs pack together iff they agree on every key
        except those in `skip_keys`. This is the "all-seeds-of-a-hyperparam-
        point-together" default.
        If an explicit iterable, only those keys define compatibility. E.g.
        `("config",)` packs everything with the same env config; `()` packs
        everything (still capped by max_per_pack).
    max_per_pack : int  — hard cap on pack size. With SmolLM2-135M and
        vllm_gpu_memory≈0.05 we measure 6-8 concurrent comfortably on one H100;
        6 is the safe default.
    skip_keys    : Iterable[str] | None  — only used when group_keys=None. Adds
        to the default skip list, doesn't replace it.

    Returns
    -------
    list[list[dict]] — each inner list goes to one train_many call. Order
    within a pack is the order runs appeared in the input.
    """
    from collections import defaultdict

    skip = set(_PACK_SKIP_KEYS_DEFAULT)
    if skip_keys is not None:
        skip |= set(skip_keys)

    if group_keys is None:
        def keyfn(r):
            return tuple(sorted(
                (k, _hashable(v)) for k, v in r.items() if k not in skip
            ))
    else:
        gk_tup = tuple(group_keys)
        def keyfn(r):
            return tuple((k, _hashable(r.get(k))) for k in gk_tup)

    groups = defaultdict(list)
    order = []  # preserve first-seen order across groups
    for r in runs:
        k = keyfn(r)
        if k not in groups:
            order.append(k)
        groups[k].append(r)

    packs = []
    for k in order:
        g = groups[k]
        for i in range(0, len(g), max_per_pack):
            packs.append(g[i:i + max_per_pack])
    return packs


@app.local_entrypoint()
def smoke_test():
    """Build the image (first time) and verify the container env. Run before launch_modal_3envs."""
    result = smoke.remote()
    import json
    print(json.dumps(result, indent=2))


def _dispatch_sweep(sweep_module_name: str, sweep_name: str, gpu="H100", judge=False):
    """Shared launch logic: import a sweep config and dispatch each run as a Modal call.

    gpu selects the per-run function: "H100" -> train_one, "H200" -> train_one_h200
    (or train_one_h200_judge when judge=True, which adds the OpenRouter judge-keys secret
    for the LLM-judge detector)."""
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs

    if gpu == "H200":
        fn = train_one_h200_judge if judge else train_one_h200
    else:
        fn = train_one
    print(f"[modal] dispatching {len(runs)} runs (sweep={sweep_name}, gpu={gpu}{', judge' if judge else ''})")
    for r in runs:
        print(f"  - {r['run_name']}")

    # .starmap dispatches all in parallel; each gets its own container/GPU.
    results = list(fn.starmap([(r, sweep_name) for r in runs]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")
    failures = [r for r in results if r["status"] != "ok"]
    if failures:
        print(f"[modal] {len(failures)} run(s) failed")
    else:
        print(f"[modal] all {len(results)} runs ok")
    print(f"[modal] sync back: modal volume get gr-modal-pilot / {REPO_LOCAL}/output/")


def _dispatch_packed_sweep(sweep_module_name: str, sweep_name: str,
                           group_keys=None, max_per_pack=6,
                           vllm_gpu_memory=None, gpu="H100"):
    """Packed-launch variant of `_dispatch_sweep`. Groups runs with `_group_runs`,
    dispatches each pack to `train_many` (or `train_many_h200` when gpu="H200"), and
    prints a per-run summary on the same shape as the unpacked path.

    If `vllm_gpu_memory` is None, the per-item value is set to `0.40 / max_per_pack` (a
    fraction of the GPU) so the per-pack sum stays below ~0.40 — safe headroom for small
    models. On H200 (141 GB vs H100 80 GB) the same fraction buys ~1.75x the absolute KV
    cache per run, so you can raise max_per_pack to pack more 0.6B runs per container.
    Pass an explicit value to override (e.g. for larger models that want more KV cache).
    """
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs

    pack_fn = train_many_h200 if gpu == "H200" else train_many
    packs = _group_runs(runs, group_keys=group_keys, max_per_pack=max_per_pack)
    per_item_mem = vllm_gpu_memory if vllm_gpu_memory is not None else (0.40 / max_per_pack)
    # Apply only when caller hasn't already set it explicitly.
    for r in runs:
        r.setdefault("vllm_gpu_memory", per_item_mem)

    print(f"[modal] packing {len(runs)} runs into {len(packs)} train_many call(s) "
          f"(max_per_pack={max_per_pack}, vllm_gpu_memory≈{per_item_mem:.3f}/item, "
          f"sweep={sweep_name})")
    for i, pack in enumerate(packs):
        names = ", ".join(r["run_name"] for r in pack)
        print(f"  pack {i+1}/{len(packs)} (n={len(pack)}): {names}")

    # .starmap over packs runs them in parallel containers; each container then
    # runs its N runs concurrently under MPS.
    pack_results = list(pack_fn.starmap([(pack, sweep_name) for pack in packs]))
    # Flatten + report.
    flat = [r for pack_res in pack_results for r in pack_res]
    for res in flat:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")
    failures = [r for r in flat if r["status"] != "ok"]
    if failures:
        print(f"[modal] {len(failures)}/{len(flat)} run(s) failed")
    else:
        print(f"[modal] all {len(flat)} runs ok")
    print(f"[modal] sync back: modal volume get gr-modal-pilot / {REPO_LOCAL}/output/")


@app.local_entrypoint()
def launch_modal_3envs():
    """6 runs: object_qa + addition_v2 + topic_contains × 2 seeds, exclusive routing."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_3envs_exclusive_nocoh_1k",
        "retrain_gr_modal_3envs_exclusive_nocoh_1k",
    )


@app.local_entrypoint()
def launch_modal_all_classic():
    """14 runs: 7 envs × 2 seeds, classic routing (no coherence, max_steps=1000)."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_1k",
        "retrain_gr_modal_all_classic_nocoh_1k",
    )


@app.local_entrypoint()
def launch_modal_all_classic_canonical_radam_smoke():
    """20-step smoke of the RoutedAdam-classic sweep (runs[0]: persona_qa bw2 s1).
    Validates: classic RoutedAdam optimizer build, sample-level m-stream accumulation,
    clip-factor mirroring, wandb logging with the new key, checkpoint save."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam import runs as _runs
    pilot = dict(_runs[0])
    pilot["max_steps"] = 20
    pilot["save_steps"] = 20
    pilot["run_name"] = "smoke_" + pilot["run_name"]
    pilot["wandb_project"] = "gr-radam-classic-smoke"  # keep smoke noise out of the full project
    print(f"[smoke] {pilot['run_name']}")
    res = train_one.remote(pilot, "retrain_gr_canonical_radam_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_all_classic_canonical_radam_full():
    """24 runs: 7 envs × seeds {1,3,5} RoutedAdam-classic bw2 + topic_contains × {1,3,5}
    bw1 ablation. Canonical per-env max_steps (2k / 1k), no coherence."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam",
        "retrain_gr_modal_all_classic_nocoh_canonical_steps_radam",
    )


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
              timeout=4 * 60 * 60)
def eval_forget_scales(run_rel: str, checkpoints: list, forget_scales: str,
                       n_eval: int, out_rel: str) -> dict:
    """Posthoc forget-scale eval via eval_utils.py CLI, one container per run.

    Loops `checkpoints` sequentially (records append to the same per-run jsonl, so
    one container per run avoids concurrent-append races on the volume). Idempotent:
    checkpoints whose model_path already has a record in the output jsonl are skipped.
    Mirrors the (uncommitted) driver that produced canonical_5seed_1k_samples/.
    """
    import json
    import os
    import subprocess
    import sys
    import time
    os.chdir(REPO_REMOTE)
    out_path = os.path.join(OUTPUT_REMOTE, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_dir = os.path.join(os.path.dirname(out_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_name = os.path.basename(run_rel.rstrip("/"))
    log_path = os.path.join(log_dir, f"{run_name}.log")

    done_paths = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done_paths = {json.loads(line).get("model_path") for line in f if line.strip()}

    statuses = []
    t0 = time.time()
    for ckpt in checkpoints:
        model_path = os.path.join(OUTPUT_REMOTE, run_rel, ckpt)
        assert os.path.isdir(model_path), f"missing checkpoint dir {model_path}"
        if model_path in done_paths:
            statuses.append((ckpt, "skipped"))
            continue
        cmd = [sys.executable, "eval_utils.py", "--model_path", model_path,
               "--n_eval", str(n_eval), "--forget_scales", forget_scales,
               "--output", out_path]
        with open(log_path, "a") as logf:
            logf.write(f"\n=== {ckpt}, forget_scales={forget_scales} ===\n")
            logf.write(f"# cmd={' '.join(cmd)}\n\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        vol.commit()
        assert proc.returncode == 0, (
            f"eval_utils failed (rc={proc.returncode}) for {model_path}; see {log_path}")
        statuses.append((ckpt, "ok"))
    return {"run_name": run_name, "statuses": statuses,
            "duration_s": time.time() - t0}


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
              timeout=60 * 60)
def ppl_matrix_one(run_rel: str, checkpoint: str, n_prompts: int = 512) -> dict:
    """Ablation-damage perplexity matrix for one trained run (2026-06-12 design).

    On the unhackable AND detectable prompt subset: generate one on-policy rollout
    per prompt under (1,0) retain_only and (1,1) both; score each rollout set's
    completion tokens under (1,0), (1,1), and (0,0) base -> 2x3 mean per-token NLL.
    Also reports the unconditional hack-detector rate per rollout set. Diagnoses
    whether forget-ablation damages the model (retain_only as a worse model of
    clean behavior) -> motivates self-distillation coherence training.

    Generation at the run's training temperature (asserted 1.0 so raw logprobs ARE
    the sampling policy). EOS handling is identical across configs (comparisons
    unaffected). One jsonl row per run; idempotent at the driver level (skips runs
    whose output file already exists).
    """
    import json
    import math
    import os
    import sys
    import time
    from argparse import Namespace

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import torch
    import yaml
    from transformers import AutoTokenizer

    from envs import get_env
    from eval_utils import generate_from_model, load_gradient_routing_model
    from experiment_config import ExperimentConfig
    from gradient_routing import set_scales
    from rh_detectors import RH_CLASSIFIABLE_REGISTRY, RH_DETECTOR_REGISTRY, get_rh_classifiable
    from train import _inject_detectable_into_eval_data

    t0 = time.time()
    run_dir = os.path.join(OUTPUT_REMOTE, run_rel)
    model_path = os.path.join(run_dir, checkpoint)
    assert os.path.isdir(model_path), f"missing {model_path}"
    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        run_cfg = yaml.safe_load(f)

    temperature = run_cfg.get("temperature", 1.0)
    assert temperature == 1.0, (
        f"ppl_matrix assumes raw logprobs == sampling policy; temperature={temperature}"
    )

    ec_fields = set(ExperimentConfig.model_fields)
    exp_cfg = ExperimentConfig.model_validate(
        {k: v for k, v in run_cfg.items() if k in ec_fields})
    env_spec = get_env(run_cfg["environment"])
    env_args = Namespace(**run_cfg)

    # Load with headroom, then filter to unhackable AND detectable.
    eval_data = env_spec.load_eval_prompts(n_prompts * 8, env_args)
    det_name = exp_cfg.rh_detector.name
    assert det_name in RH_CLASSIFIABLE_REGISTRY, f"no classifiable fn for {det_name}"
    _inject_detectable_into_eval_data(
        eval_data, get_rh_classifiable(det_name, **(exp_cfg.rh_detector.params or {})))
    subset = [d for d in eval_data if not d["hackable"] and d["detectable"]]
    prompt_subset = "unhackable_detectable"
    if len(subset) < n_prompts:
        # repeat_extra / topic_contains: detectability is defined only within the
        # hackable subset (detectable ⊆ hackable — instruction/constraint template
        # features), so the intersection is structurally empty. Fall back to plain
        # unhackable (the operative property: rollouts carry no rewarded hack).
        subset = [d for d in eval_data if not d["hackable"]]
        prompt_subset = "unhackable"
    assert len(subset) >= n_prompts, (
        f"only {len(subset)} {prompt_subset} prompts (need {n_prompts}); "
        f"raise the load multiplier")
    subset = subset[:n_prompts]
    prompts = [d["prompt"] for d in subset]

    model = load_gradient_routing_model(model_path, base_model=run_cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model"])

    hack_det_cfg = exp_cfg.hack_freq_detector
    assert hack_det_cfg is not None, "run config lacks hack_freq_detector"
    hack_fn = RH_DETECTOR_REGISTRY[hack_det_cfg.name]
    det_cols = {k: [d[k] for d in subset] for k in subset[0] if k != "prompt"}

    def _score_nll(rollouts, chunk=128):
        """Mean per-token NLL of rollout completions under the CURRENT scales."""
        device = next(model.parameters()).device
        total_nll, total_tok = 0.0, 0
        for i in range(0, len(rollouts), chunk):
            batch = rollouts[i:i + chunk]
            seqs, comp_lens = [], []
            for r in batch:
                if tokenizer.chat_template is not None:
                    p_ids = tokenizer.apply_chat_template(
                        [[{"role": "user", "content": r["prompt"]}]],
                        add_generation_prompt=True, tokenize=True,
                        return_dict=True)["input_ids"][0]
                else:
                    p_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
                p_ids = list(p_ids)
                assert p_ids and isinstance(p_ids[0], int), \
                    f"prompt tokenization returned {type(p_ids[0])}, expected ints"
                c_ids = r["completion_ids"]
                if not c_ids:
                    comp_lens.append(0)
                    seqs.append(p_ids)
                    continue
                seqs.append(list(p_ids) + list(c_ids))
                comp_lens.append(len(c_ids))
            maxlen = max(len(s) for s in seqs)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                attn[j, :len(s)] = 1
            ids, attn = ids.to(device), attn.to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attn).logits
            logp = logits.log_softmax(-1)
            for j, s in enumerate(seqs):
                cl = comp_lens[j]
                if cl == 0:
                    continue
                start = len(s) - cl  # completion token t predicted at position t-1
                tok_lp = logp[j, start - 1:len(s) - 1].gather(
                    -1, ids[j, start:len(s)].unsqueeze(-1)).squeeze(-1)
                total_nll += float(-tok_lp.sum())
                total_tok += cl
        assert total_tok > 0, "no completion tokens to score"
        return total_nll / total_tok

    GEN_CFGS = [("retain_only", 1.0, 0.0), ("both", 1.0, 1.0)]
    SCORE_CFGS = [("retain_only", 1.0, 0.0), ("both", 1.0, 1.0), ("base", 0.0, 0.0)]
    matrix, hack_rates, mean_lens = {}, {}, {}
    for gname, gr, gf in GEN_CFGS:
        set_scales(model, retain_scale=gr, forget_scale=gf)
        rollouts = generate_from_model(
            model, tokenizer, n_samples=len(prompts),
            max_new_tokens=env_spec.eval_max_tokens, temperature=temperature,
            prompts=prompts)
        comps = [r["completion"] for r in rollouts]
        hack_rates[gname] = float(sum(hack_fn(comps, **det_cols))) / len(comps)
        mean_lens[gname] = float(sum(len(r["completion_ids"]) for r in rollouts)) / len(rollouts)
        for sname, sr, sf in SCORE_CFGS:
            set_scales(model, retain_scale=sr, forget_scale=sf)
            nll = _score_nll(rollouts)
            matrix[f"{gname}|{sname}"] = {"nll": nll, "ppl": math.exp(nll)}

    row = {"run_name": os.path.basename(run_rel.rstrip("/")), "checkpoint": checkpoint,
           "environment": run_cfg["environment"], "seed": run_cfg.get("seed"),
           "n_prompts": len(prompts), "prompt_subset": prompt_subset,
           "temperature": temperature,
           "hack_rate": hack_rates, "mean_completion_len": mean_lens,
           "matrix": matrix}
    out_dir = os.path.join(OUTPUT_REMOTE, "gr_forget_scale_eval/radam_bw1_ppl_matrix")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{row['run_name']}.jsonl"), "w") as f:
        f.write(json.dumps(row) + "\n")
    vol.commit()
    row["duration_s"] = time.time() - t0
    return row


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
              timeout=20 * 60)
def peek_rollouts(run_rel: str, checkpoint: str, retain_scale: float = 1.0,
                  forget_scale: float = 0.0, n: int = 24) -> list:
    """Debug helper: generate n on-policy rollouts from a checkpoint at the given
    adapter scales and return [(prompt, completion)] — for eyeballing collapse modes."""
    import os
    import sys
    from argparse import Namespace
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import yaml
    from transformers import AutoTokenizer

    from envs import get_env
    from eval_utils import generate_from_model, load_gradient_routing_model
    from gradient_routing import set_scales

    run_dir = os.path.join(OUTPUT_REMOTE, run_rel)
    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        run_cfg = yaml.safe_load(f)
    env_spec = get_env(run_cfg["environment"])
    eval_data = env_spec.load_eval_prompts(n, Namespace(**run_cfg))
    prompts = [d["prompt"] for d in eval_data]
    model = load_gradient_routing_model(os.path.join(run_dir, checkpoint),
                                        base_model=run_cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model"])
    set_scales(model, retain_scale=retain_scale, forget_scale=forget_scale)
    rollouts = generate_from_model(model, tokenizer, n_samples=n,
                                   max_new_tokens=env_spec.eval_max_tokens,
                                   temperature=run_cfg.get("temperature", 1.0),
                                   prompts=prompts)
    return [(r["prompt"][:80], r["completion"]) for r in rollouts]


@app.local_entrypoint()
def run_peek_rollouts(run_rel: str, checkpoint: str, retain_scale: float = 1.0,
                      forget_scale: float = 0.0, n: int = 24):
    for p, c in peek_rollouts.remote(run_rel, checkpoint, retain_scale, forget_scale, n):
        print(f"PROMPT: {p!r}\n  -> {c!r}")


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
              timeout=75 * 60)
def coh_vs_routing_group_analysis(run_rel: str, checkpoint: str,
                                   n_coh_prompts: int = 24,
                                   n_rout_prompts: int = 48,
                                   n_per_prompt: int = 16,
                                   gen_chunk: int = 32,
                                   temperature: float = 0.7) -> dict:
    """Group-size + per-adapter advantage analysis at one checkpoint.

    Two passes:
      'coh'  — n_coh_prompts hackable+detectable (Array+hackable) prompts,
                generate n_per_prompt @ scales (1, 0) [retain_only]
      'rout' — n_rout_prompts RANDOM prompts from full train_dataset,
                generate n_per_prompt @ scales (1, 1) [both adapters]

    For each pass: score leetcode_all_components (correct/trait/compile),
    compute reward = 3*correct + 3*trait*hackable + 0.5*compile (matching the
    leetcode_rh_array YAML scales; trait component is forget-role so zeroed on
    unhackable), then per-group compute BOTH:
      retain_advantage  = (r - mean) / (std + eps) over ~is_rh; 0 if is_rh
                          (this is what the retain adapter trains on)
      standard_advantage = (r - mean) / (std + eps) over the full 16-gen group
                          (the GRPO baseline; what forget sees in classic-routing
                          bad-pass)
    Returns histograms of post-filter group sizes + per-sample arrays of
    (trait, is_rh, hackable, retain_adv, standard_adv) split by mode."""
    import json
    import os
    import random
    import sys
    from argparse import Namespace

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    import yaml
    import torch
    from transformers import AutoTokenizer

    from envs import get_env
    from eval_utils import generate_from_model, load_gradient_routing_model
    from gradient_routing import set_scales
    from rh_detectors import get_rh_classifiable
    from envs.leetcode import leetcode_all_components

    run_dir = os.path.join(OUTPUT_REMOTE, run_rel)
    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        run_cfg = yaml.safe_load(f)
    env_spec = get_env(run_cfg["environment"])

    # Load train dataset once
    train_ds = env_spec.load_train(Namespace(**run_cfg))
    cols = {c: train_ds[c] for c in train_ds.column_names}
    classify_fn = get_rh_classifiable("leetcode_feature_conditional",
                                       tags_any=["Array"])
    array_mask = classify_fn(**cols)
    hackable_col = cols.get("hackable") or [True] * len(array_mask)

    # Sample prompts for each pass
    rng = random.Random(42)
    coh_eligible = [i for i, (a, h) in enumerate(zip(array_mask, hackable_col)) if a and h]
    assert len(coh_eligible) >= n_coh_prompts, len(coh_eligible)
    coh_sel = rng.sample(coh_eligible, n_coh_prompts)
    rout_sel = rng.sample(range(len(train_ds)), n_rout_prompts)
    print(f"[{checkpoint}] coh: {n_coh_prompts} Array+hackable from pool {len(coh_eligible)}; "
          f"rout: {n_rout_prompts} random from {len(train_ds)} "
          f"({sum(array_mask[i] for i in rout_sel)} Array, "
          f"{sum(hackable_col[i] for i in rout_sel)} hackable)")

    # Load model once
    model = load_gradient_routing_model(os.path.join(run_dir, checkpoint),
                                        base_model=run_cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model"])

    def _chunked_generate(prompts, retain_scale, forget_scale, label):
        set_scales(model, retain_scale=retain_scale, forget_scale=forget_scale)
        all_rollouts = []
        n_total = len(prompts)
        for start in range(0, n_total, gen_chunk):
            end = min(start + gen_chunk, n_total)
            chunk = prompts[start:end]
            rollouts = generate_from_model(
                model, tokenizer,
                n_samples=len(chunk),
                max_new_tokens=run_cfg.get("max_completion_length", 1536),
                temperature=temperature,
                prompts=chunk,
            )
            all_rollouts.extend(rollouts)
            torch.cuda.empty_cache()
            if (start // gen_chunk) % 4 == 0:
                print(f"[{checkpoint}] {label}: gen {end}/{n_total}")
        return [r["completion"] for r in all_rollouts]

    def _run_pass(label, sel_idx, scales):
        rows = [train_ds[i] for i in sel_idx]
        n_p = len(sel_idx)
        flat_prompts = sum(([r["prompt"]] * n_per_prompt for r in rows), [])
        flat_gt    = sum(([r["gt_answer"]] * n_per_prompt for r in rows), [])
        flat_setup = sum(([r["setup_code"]] * n_per_prompt for r in rows), [])
        flat_tf    = sum(([r["test_func_name"]] * n_per_prompt for r in rows), [])
        is_array  = [bool(array_mask[i])  for i in sel_idx for _ in range(n_per_prompt)]
        is_hack_p = [bool(hackable_col[i]) for i in sel_idx for _ in range(n_per_prompt)]

        completions = _chunked_generate(flat_prompts, *scales, label)
        print(f"[{checkpoint}] {label}: scoring all_components on {len(completions)}...")
        correct, trait, compile_ = leetcode_all_components(
            completions, flat_gt, flat_setup, test_func_name=flat_tf)
        # Reward = 3*correct + 3*trait*hackable + 0.5*compile (per the YAML)
        # (trait is forget-role with scale 3; zeroed on unhackable)
        rewards = [3.0*c + (3.0*t if h else 0.0) + 0.5*comp
                   for c, t, comp, h in zip(correct, trait, compile_, is_hack_p)]
        # is_rh = (Array tag) AND (trait > 0.5) — matches leetcode_feature_conditional
        is_rh = [a and (t > 0.5) for a, t in zip(is_array, trait)]
        gt_hack = [t > 0.5 for t in trait]

        eps = 1e-4
        retain_adv = [0.0] * len(rewards)
        standard_adv = [0.0] * len(rewards)
        n_non_rh_per_group = []
        for g in range(n_p):
            sl = slice(g * n_per_prompt, (g + 1) * n_per_prompt)
            r_g = rewards[sl]; rh_g = is_rh[sl]
            # standard GRPO advantage over full group
            mu = sum(r_g) / len(r_g)
            var = sum((x - mu) ** 2 for x in r_g) / len(r_g)
            sd = var ** 0.5
            for i, r in enumerate(r_g):
                standard_adv[g * n_per_prompt + i] = (r - mu) / (sd + eps)
            # retain advantage = renorm over ~is_rh; 0 if is_rh
            good_idx = [i for i, x in enumerate(rh_g) if not x]
            n_non_rh_per_group.append(len(good_idx))
            if good_idx:
                r_good = [r_g[i] for i in good_idx]
                mu_g = sum(r_good) / len(r_good)
                var_g = sum((x - mu_g) ** 2 for x in r_good) / len(r_good)
                sd_g = var_g ** 0.5
                for i in good_idx:
                    retain_adv[g * n_per_prompt + i] = (r_g[i] - mu_g) / (sd_g + eps)
            # else: whole group's retain_adv stays 0

        hist = [0] * (n_per_prompt + 1)
        for n in n_non_rh_per_group:
            hist[n] += 1
        return {
            "n_prompts": n_p,
            "n_per_prompt": n_per_prompt,
            "scales": list(scales),
            "n_non_rh_per_group": n_non_rh_per_group,
            "histogram": hist,
            "mean_n_non_rh": sum(n_non_rh_per_group) / max(1, len(n_non_rh_per_group)),
            "per_sample": {
                "trait":        [float(x) for x in trait],
                "correct":      [float(x) for x in correct],
                "compile":      [float(x) for x in compile_],
                "reward":       [float(x) for x in rewards],
                "is_rh":        [bool(x) for x in is_rh],
                "gt_hack":      [bool(x) for x in gt_hack],
                "is_array":     is_array,
                "is_hackable":  is_hack_p,
                "retain_adv":   retain_adv,
                "standard_adv": standard_adv,
            },
        }

    out = {"checkpoint": checkpoint, "modes": {}}
    out["modes"]["coh"]  = _run_pass("coh",  coh_sel,  (1.0, 0.0))
    out["modes"]["rout"] = _run_pass("rout", rout_sel, (1.0, 1.0))

    for label, m in out["modes"].items():
        ps = m["per_sample"]
        gt_hack_count = sum(ps["gt_hack"])
        print(f"[{checkpoint}] {label}: hist={m['histogram']} mean_n_non_rh={m['mean_n_non_rh']:.2f} "
              f"gt_hack={gt_hack_count}/{len(ps['trait'])}")

    out_dir = os.path.join(OUTPUT_REMOTE,
                           "diagnostics/coh_vs_routing_classic_coh_s7")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{checkpoint}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"[{checkpoint}] wrote {out_path}")
    return {"checkpoint": checkpoint, "out_path": out_path,
            "coh_hist":  out["modes"]["coh"]["histogram"],
            "rout_hist": out["modes"]["rout"]["histogram"]}


@app.local_entrypoint()
def run_coh_vs_routing_group_analysis():
    """Dispatch coh-vs-routing group-size + advantage analysis on classic-coh s7
    at post-hack-emergence pre-collapse checkpoints (1200, 1600)."""
    run_rel = "leetcode_array_classic_coh/leetcode_rh_array_gr_cls_coh_s7"
    args = [(run_rel, "checkpoint-1200", 24, 48, 16, 32, 0.7),
            (run_rel, "checkpoint-1600", 24, 48, 16, 32, 0.7)]
    print(f"[modal] dispatching {len(args)} analyses")
    results = list(coh_vs_routing_group_analysis.starmap(args))
    for r in results:
        print(f"=== {r['checkpoint']} ===")
        print(f"  coh  hist: {r['coh_hist']}")
        print(f"  rout hist: {r['rout_hist']}")
        print(f"  -> {r['out_path']}")


@app.local_entrypoint()
def launch_modal_ppl_matrix_bw1(n_prompts: int = 512, only: str = ""):
    """Perplexity 2x3 matrix on all 21 bw1 final checkpoints →
    /output/gr_forget_scale_eval/radam_bw1_ppl_matrix/. --only for smoke."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam import runs as _r2
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam_bw1 import runs as _r1
    bw1 = [r for r in _r2 + _r1 if "_radam_bw1_" in r["run_name"]]
    assert len(bw1) == 21, f"expected 21 bw1 runs, got {len(bw1)}"
    if only:
        bw1 = [r for r in bw1 if r["run_name"] in set(only.split(","))]
    calls = [(f"{_RADAM_SWEEP}/{r['run_name']}", f"checkpoint-{r['max_steps']}", n_prompts)
             for r in bw1]
    print(f"[modal] dispatching {len(calls)} ppl-matrix evals (n_prompts={n_prompts})")
    results = list(ppl_matrix_one.starmap(calls))
    for res in results:
        m = res["matrix"]
        print(f"  {res['run_name']}: ppl ro|ro={m['retain_only|retain_only']['ppl']:.2f} "
              f"both|both={m['both|both']['ppl']:.2f} ({res['duration_s']:.0f}s)")


_RADAM_SWEEP = "retrain_gr_modal_all_classic_nocoh_canonical_steps_radam"
_RADAM_EVAL_SCALES = "0,0.2,0.4,0.6,0.8,1"
_RADAM_EVAL_N = 1000
# bw1 (topic_contains ablation) runs get their own eval dir: collate/plot/optimum
# tooling groups rows by (env, seed), which would collide across the two arms.
_RADAM_FINAL_EVAL_DIR = "gr_forget_scale_eval/canonical_radam_1k_samples"
_RADAM_BW1_FINAL_EVAL_DIR = "gr_forget_scale_eval/canonical_radam_bw1_1k_samples"
_RADAM_SDCOH_FINAL_EVAL_DIR = "gr_forget_scale_eval/radam_bw1_sdcoh_1k_samples"


@app.local_entrypoint()
def launch_modal_all_classic_canonical_radam_bw1_full():
    """18 runs: bw1 (B=1, removed-signal-only) on the 6 non-topic envs × seeds
    {1,3,5}. Same sweep output dir as the bw2 runs (run names carry _bw1_).
    All 6 envs completed under the 4h train_one timeout in the bw2 sweep."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam_bw1",
        "retrain_gr_modal_all_classic_nocoh_canonical_steps_radam",
    )


@app.local_entrypoint()
def launch_modal_radam_bw1_sdcoh_smoke():
    """30-step smoke of self-distillation coherence (runs[0]: persona α=1.0 s1).
    Validates: coh_fixed_advantage advantage override, RoutedAdam coherence-mb
    stream feeding, coherence/sd_hack_frac metric, (1,0) coh generation slot."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.radam_bw1_sdcoh import runs as _runs
    pilot = dict(_runs[0])
    pilot["max_steps"] = 30
    pilot["save_steps"] = 30
    pilot["run_name"] = "smoke_" + pilot["run_name"]
    pilot["wandb_project"] = "gr-radam-classic-smoke"
    print(f"[smoke] {pilot['run_name']}")
    res = train_one.remote(pilot, "radam_bw1_sdcoh_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_radam_bw1_sdcoh_full():
    """12 runs: SD coherence (persona/repeat/object × seeds {1,3} × α {1.0,0.3},
    cspr=64) on the bw1 RoutedAdam canonical config."""
    _dispatch_sweep("sweeps.radam_bw1_sdcoh", "radam_bw1_sdcoh")


@app.local_entrypoint()
def launch_modal_leetcode_noint_4b_smoke():
    """20-step smoke of the no-intervention leetcode baseline (Qwen3-4B, H100).
    Validates: leetcode env + PersistentCodeEvaluator code-exec run in the Modal
    container; 4B + rollout 1024 + 1536-tok completions fit on H100; reward
    (correct/trait/compile) scores > 0; no judge / no OpenRouter needed."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.leetcode_noint_4b_baseline import runs as _runs
    pilot = dict(_runs[-1])  # hf=1.0 (max hack opportunity) for the smoke
    pilot["max_steps"] = 20
    pilot["save_steps"] = 20
    pilot["eval_every"] = 20
    pilot["run_name"] = "smoke_" + pilot["run_name"]
    pilot["wandb_project"] = "leetcode-noint-baseline-4b-smoke"
    print(f"[smoke] {pilot['run_name']} (Qwen3-4B leetcode no-intervention, H100)")
    res = train_one.remote(pilot, "leetcode_noint_4b_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_noint_4b_full():
    """12 runs: Qwen3-4B no-intervention leetcode baseline (4 seeds × hack_frac
    {0.5,0.8,1.0}), off-policy 3200 steps, max_grad_norm 0.2, no judge. H100."""
    _dispatch_sweep("sweeps.leetcode_noint_4b_baseline", "leetcode_noint_4b_baseline")


@app.local_entrypoint()
def launch_modal_leetcode_noint_match_smoke():
    """20-step smoke of the VERL-matched no-intervention config (Qwen3-4B,
    on-policy, non-aware hint, LoRA r32 all-linear). Validates the non-aware
    dataset loads, LoRA r32f0 builds, on-policy batching, and code-eval scores."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.leetcode_noint_4b_match import runs as _runs
    pilot = dict(_runs[-1])  # hf=1.0 s4
    pilot["max_steps"] = 20
    pilot["save_steps"] = 20
    pilot["eval_every"] = 20
    pilot["run_name"] = "smoke_" + pilot["run_name"]
    pilot["wandb_project"] = "leetcode-noint-match-4b-smoke"
    print(f"[smoke] {pilot['run_name']} (Qwen3-4B VERL-matched no-int, H100)")
    res = train_one.remote(pilot, "leetcode_noint_match_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_noint_match_full():
    """8 runs: VERL-matched no-intervention leetcode (Qwen3-4B, on-policy, non-aware
    hint, MLP m64, lr 7e-5, grad-clip 1.0, beta 1e-3), 4 seeds × hack_frac {0.5,1.0}."""
    _dispatch_sweep("sweeps.leetcode_noint_4b_match", "leetcode_noint_4b_match")


@app.local_entrypoint()
def launch_modal_leetcode_noint_match_lora(smoke: str = ""):
    """LoRA r32f0 variant of the matched no_intervention sweep (3 seeds, hack_frac=1.0).

    Isolates the adapter within our train.py: identical to the MLP m64 matched runs
    except adapter_type=lora / lora_config=r32f0 (uses the Modal vLLM-spawn LoRA wiring
    added to train.py:_spawn_vllm_server). On-policy 200 steps ~6h/run -> train_one_long
    (10h timeout). `smoke=1` runs ONE seed at 10 steps first to validate the LoRA-spawn
    path before committing 3 full runs."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.leetcode_noint_4b_match_lora import runs as _runs
    if smoke:
        r = dict(_runs[0])
        r["max_steps"] = 10
        r["save_steps"] = 10
        r["run_name"] = r["run_name"] + "_smoke"
        print(f"[modal] LoRA-spawn SMOKE: {r['run_name']} (10 steps)")
        res = train_one.remote(r, "leetcode_noint_4b_match_lora_smoke")
        print(f"  {res.get('run_name')}: {res.get('status')} ({res.get('duration_s', 0):.1f}s)")
        return
    print(f"[modal] dispatching {len(_runs)} LoRA runs on train_one_long (10h)")
    for r in _runs:
        print(f"  - {r['run_name']}")
    results = list(train_one_long.starmap(
        [(r, "leetcode_noint_4b_match_lora") for r in _runs]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_verlparity(smoke: str = ""):
    """VERL-parity loss-fix ablation: clamp_only vs all_changes, 3 seeds each (6 runs).

    Tests whether the unclamped-k3-KL fix (and friends) stops the gradient explosion and lets
    the matched no_intervention LoRA run learn/hack like VERL. On-policy 200 steps ~6h/run ->
    train_one_long (10h). `smoke=1` runs ONE seed of EACH arm at 10 steps first, validating BOTH
    new code paths (clamp_only; all_changes = +fp32_lora +token_mean +nonfinite_skip) before
    committing the 6 full runs. See memory verl-parity-loss-changes."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.leetcode_noint_4b_verlparity import runs as _runs
    if smoke:
        smoke_runs = []
        for arm in ("clamp_only", "all_changes"):
            r = dict(next(x for x in _runs if arm in x["run_name"]))
            r["max_steps"] = 10
            r["save_steps"] = 10
            r["run_name"] = r["run_name"] + "_smoke"
            smoke_runs.append(r)
        print(f"[modal] verlparity SMOKE: {[r['run_name'] for r in smoke_runs]} (10 steps each)")
        results = list(train_one.starmap([(r, "leetcode_noint_4b_verlparity_smoke") for r in smoke_runs]))
        for res in results:
            print(f"  {res.get('run_name')}: {res.get('status')} ({res.get('duration_s', 0):.1f}s)")
        return
    print(f"[modal] dispatching {len(_runs)} verlparity runs on train_one_long (10h)")
    for r in _runs:
        print(f"  - {r['run_name']}")
    results = list(train_one_long.starmap(
        [(r, "leetcode_noint_4b_verlparity") for r in _runs]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_topic_radam_rerun():
    """Rerun the 6 topic_contains radam runs (bw2+bw1 × seeds {1,3,5}) on
    train_one_long (10h timeout). The original attempts all died at Modal's 4h
    train_one timeout at step ~730/1000 (judge reward ~10-15s/step under shared
    OpenAI rate limits). Fresh from-scratch runs into the same output dirs;
    every checkpoint-N is overwritten as the rerun passes it, so the final dirs
    are single-attempt consistent."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam import runs as _runs
    topic = [r for r in _runs if "topic_contains" in r["run_name"]]
    assert len(topic) == 6, f"expected 6 topic runs, got {len(topic)}"
    print(f"[modal] re-dispatching {len(topic)} topic runs on train_one_long")
    for r in topic:
        print(f"  - {r['run_name']}")
    results = list(train_one_long.starmap(
        [(r, "retrain_gr_modal_all_classic_nocoh_canonical_steps_radam") for r in topic]))
    for res in results:
        print(f"  {res['run_name']}: {res['status']} ({res['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_canonical_radam(only: str = ""):
    """Final-checkpoint forget-scale eval for the RoutedAdam-classic sweep:
    24 runs × 6 forget scales × n_eval=1000 → /output/{_RADAM_FINAL_EVAL_DIR}/
    (bw2) and /output/{_RADAM_BW1_FINAL_EVAL_DIR}/ (bw1 topic ablation).
    One container per run; idempotent (re-run after stragglers finish).
    --only: comma-separated run_name filter for wave dispatch while the train
    sweep is still finishing (the eval asserts on missing checkpoints)."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.radam_bw1_sdcoh import runs as _sdcoh_runs
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam import runs as _runs
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam_bw1 import (
        runs as _bw1_runs,
    )
    _runs = _runs + _bw1_runs + _sdcoh_runs
    if only:
        only_set = set(only.split(","))
        unknown = only_set - {r["run_name"] for r in _runs}
        assert not unknown, f"--only contains unknown runs: {unknown}"
        _runs = [r for r in _runs if r["run_name"] in only_set]
    calls = []
    for r in _runs:
        if "_sdcoh_" in r["run_name"]:
            out_dir = _RADAM_SDCOH_FINAL_EVAL_DIR
            sweep_root = "radam_bw1_sdcoh"  # trained via launch_modal_radam_bw1_sdcoh_full
        elif "_radam_bw1_" in r["run_name"]:
            out_dir = _RADAM_BW1_FINAL_EVAL_DIR
            sweep_root = _RADAM_SWEEP
        else:
            out_dir = _RADAM_FINAL_EVAL_DIR
            sweep_root = _RADAM_SWEEP
        calls.append((f"{sweep_root}/{r['run_name']}",
                      [f"checkpoint-{r['max_steps']}"],
                      _RADAM_EVAL_SCALES, _RADAM_EVAL_N,
                      f"{out_dir}/{r['run_name']}.jsonl"))
    print(f"[modal] dispatching {len(calls)} forget-scale evals")
    results = list(eval_forget_scales.starmap(calls))
    for res in results:
        print(f"  {res['run_name']}: {res['statuses']} ({res['duration_s']:.0f}s)")


@app.local_entrypoint()
def dump_classic_coh_s7_s17_samples():
    """Dispatch eval_forget_scales on the classic-coh leetcode s7 + s17 final
    checkpoints (vanilla Adam, classic + same_reward coherence + verified retain),
    forget_scales=0 (retain_only) and 1 (both). eval_utils.py prints --n_samples
    completions per mode at end; that stdout is captured to a log on the volume,
    so we can SEE the actual rollouts from the hollowed-retain (s7) vs
    retain-leaked (s17) regimes. Last checkpoint with weights saved = 3000."""
    base = "leetcode_array_classic_coh"
    out_rel = "gr_forget_scale_eval/leetcode_array_classic_coh_s7_s17_samples"
    calls = [(f"{base}/leetcode_rh_array_gr_cls_coh_s{s}",
              ["checkpoint-3000"], "0,1", 64,
              f"{out_rel}/leetcode_rh_array_gr_cls_coh_s{s}.jsonl")
             for s in (7, 17)]
    print(f"[modal] dispatching {len(calls)} sample-dump evals -> {out_rel}")
    results = list(eval_forget_scales.starmap(calls))
    for res in results:
        print(f"  {res['run_name']}: {res['statuses']} ({res['duration_s']:.0f}s)")


@app.local_entrypoint()
def launch_modal_eval_exclusive_radam(only: str = ""):
    """Final-checkpoint forget-scale eval for the RoutedAdam-EXCLUSIVE toy sweep
    (gr_radam_exclusive_lr6e4_v2, 7 envs × seeds {1,3,5}). Same machinery as
    launch_modal_eval_canonical_radam but pointed at the exclusive sweep dir.
    One container per run; idempotent."""
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_exclusive_nocoh_canonical_steps_radam import runs as _runs
    if only:
        only_set = set(only.split(","))
        unknown = only_set - {r["run_name"] for r in _runs}
        assert not unknown, f"--only contains unknown runs: {unknown}"
        _runs = [r for r in _runs if r["run_name"] in only_set]
    sweep_root = "gr_radam_exclusive_lr6e4_v2"
    out_dir = "gr_forget_scale_eval/radam_exclusive_lr6e4_v2_1k_samples"
    calls = [(f"{sweep_root}/{r['run_name']}",
              [f"checkpoint-{r['max_steps']}"],
              _RADAM_EVAL_SCALES, _RADAM_EVAL_N,
              f"{out_dir}/{r['run_name']}.jsonl")
             for r in _runs]
    print(f"[modal] dispatching {len(calls)} forget-scale evals -> {out_dir}")
    results = list(eval_forget_scales.starmap(calls))
    for res in results:
        print(f"  {res['run_name']}: {res['statuses']} ({res['duration_s']:.0f}s)")


@app.local_entrypoint()
def launch_modal_eval_canonical_radam_trajectory(optima_json: str, out_dir: str = ""):
    """Per-checkpoint eval at each run's optimal forget scale (n_eval=1000) →
    /output/gr_forget_scale_eval/canonical_radam_trajectory_optimum/ (default;
    pass --out-dir canonical_radam_bw1_trajectory_optimum for the bw1 arm —
    bw1/bw2 need separate dirs because the uplift-panel reader globs per-env
    files and keys them by seed).

    optima_json: path to a JSON dict {run_name: forget_scale} — produced by
    tools/radam_optima.py from the final-eval results.jsonl (argmax over
    forget_scale of retain - 2*hack_overall per run). One container per run,
    looping its checkpoints sequentially (records append to one jsonl/run)."""
    import json
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam import runs as _runs
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps_radam_bw1 import (
        runs as _bw1_runs,
    )
    _runs = _runs + _bw1_runs
    out_dir = out_dir or "canonical_radam_trajectory_optimum"
    with open(optima_json) as f:
        optima = json.load(f)
    by_name = {r["run_name"]: r for r in _runs}
    assert set(optima) <= set(by_name), f"unknown runs in {optima_json}: {set(optima) - set(by_name)}"
    calls = []
    for run_name, scale in optima.items():
        r = by_name[run_name]
        ckpts = [f"checkpoint-{s}" for s in range(100, r["max_steps"] + 1, 100)]
        calls.append((f"{_RADAM_SWEEP}/{run_name}", ckpts, f"{scale:g}", _RADAM_EVAL_N,
                      f"gr_forget_scale_eval/{out_dir}/{run_name}.jsonl"))
    print(f"[modal] dispatching {len(calls)} trajectory evals "
          f"({sum(len(c[1]) for c in calls)} checkpoint evals total)")
    results = list(eval_forget_scales.starmap(calls))
    for res in results:
        n_ok = sum(1 for _, s in res["statuses"] if s == "ok")
        n_skip = sum(1 for _, s in res["statuses"] if s == "skipped")
        print(f"  {res['run_name']}: {n_ok} ok, {n_skip} skipped ({res['duration_s']:.0f}s)")


@app.local_entrypoint()
def launch_modal_6envs_classic_coh():
    """12 runs: 6 envs (topic skipped) × 2 seeds, classic + coherence enabled, max_steps=1000."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_6envs_classic_coh_1k",
        "retrain_gr_modal_6envs_classic_coh_1k",
    )


@app.local_entrypoint()
def launch_modal_6envs_excl_coh():
    """12 runs: 6 envs (topic skipped) × 2 seeds, exclusive + coherence enabled, max_steps=1000."""
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_6envs_excl_coh_1k",
        "retrain_gr_modal_6envs_excl_coh_1k",
    )



@app.local_entrypoint()
def launch_modal_aira_rm_illicit_smoke():
    """5-step smoke of the hybrid-reward run (Qwen3-1.7B-Base, aira). Validates:
    in-process OpenAssistant RM loads + scores (prompt, completion); aira prompts
    load; Qwen base chat template applies + generates coherent text; OPENAI_API_KEY
    present + 'illicit' moderation resolves; running batch-norm path runs; wandb logs
    reward/raw_hf_reward_model_pairs (helpfulness) + reward/raw_openai_moderation
    (illicit) + reward/raw_combined. Uses train_one_h200 (gr-pilot-keys = OPENAI +
    WANDB); the RM co-resides with policy + vLLM in-container (no reward server)."""
    from sweeps.qwen3_aira_rm_illicit import runs as _runs
    pilot = dict(_runs[0])
    pilot["max_steps"] = 5
    pilot["save_steps"] = 5
    pilot["run_name"] = "smoke_qwen3_aira_rm_illicit"
    pilot["wandb_project"] = "qwen3-aira-rm-illicit-smoke"  # keep smoke noise out of the full-run project
    print(f"[smoke] {pilot['run_name']} (Qwen3-1.7B-Base, aira, "
          f"RM+illicit batchnorm, H200)")
    res = train_one_h200.remote(pilot, "qwen3_aira_rm_illicit_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_aira_rm_illicit_full():
    """Single-seed full run: Qwen3-1.7B-Base GRPO on RM-helpfulness + illicit-moderation
    (each running-batch-normalized), 200 on-policy steps. wandb project aira-rm-illicit."""
    _dispatch_sweep("sweeps.qwen3_aira_rm_illicit", "qwen3_aira_rm_illicit", gpu="H200")


@app.local_entrypoint()
def launch_modal_smollm2_aira_rm_illicit_smoke():
    """5-step smoke of the SmolLM2-135M-base hybrid-reward run (H100). Validates: SmolLM2
    base loads; --no_chat_template raw-prompt path engages + generates coherent text;
    in-process RM + illicit moderation score; batchnorm runs; reward has within-batch
    variance; no OOM. Uses train_one (H100, gr-pilot-keys = OPENAI + WANDB)."""
    from sweeps.smollm2_aira_rm_illicit import runs as _runs
    pilot = dict(_runs[0])
    pilot["max_steps"] = 5
    pilot["save_steps"] = 5
    pilot["run_name"] = "smoke_smollm2_aira_rm_illicit"
    pilot["wandb_project"] = "smollm2-aira-rm-illicit-smoke"
    print(f"[smoke] {pilot['run_name']} (SmolLM2-135M-base, aira, RM+illicit batchnorm, H100)")
    res = train_one.remote(pilot, "smollm2_aira_rm_illicit_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_smollm2_aira_rm_illicit_full():
    """SmolLM2-135M-base normalization comparison (3 H100 runs, single seed each):
    batchnorm eps0.01 vs group_zscore vs batchnorm eps1e-5, identical otherwise.
    1000 steps. wandb project smollm2-aira-rm-illicit."""
    _dispatch_sweep("sweeps.smollm2_aira_rm_illicit", "smollm2_aira_rm_illicit", gpu="H100")


@app.local_entrypoint()
def launch_modal_smollm2_aira_singlesignal_full():
    """SmolLM2-135M-base single-signal baselines (2 H100 runs): RM-only vs illicit-only,
    no normalization. Best-case ceiling per reward; same project (smollm2-aira-rm-illicit)
    as the combined/normalization runs for comparison. 1000 steps."""
    _dispatch_sweep("sweeps.smollm2_aira_singlesignal", "smollm2_aira_singlesignal", gpu="H100")


@app.local_entrypoint()
def launch_modal_smollm2_moderation_categories_smoke():
    """5-step smoke of the moderation-category screen — validates the NEW combo (stories
    env + SmolLM2 base + no_chat_template) and that the moderation category resolves.
    Runs violence+stories (runs[0]) on H100."""
    from sweeps.smollm2_moderation_categories import runs as _runs
    pilot = dict(_runs[0])
    pilot["max_steps"] = 5
    pilot["save_steps"] = 5
    pilot["run_name"] = "smoke_" + _runs[0]["run_name"]
    pilot["wandb_project"] = "smollm2-moderation-categories-smoke"
    print(f"[smoke] {pilot['run_name']} (cat={pilot['config']}, env={pilot['environment']}, beta=0, H100)")
    res = train_one.remote(pilot, "smollm2_moderation_categories_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_smollm2_moderation_categories_full():
    """SmolLM2-135M sole-reward moderation learnability screen: 5 categories x 2 envs
    (stories, aira) = 10 H100 runs, beta=0, 1000 steps. wandb smollm2-moderation-categories."""
    _dispatch_sweep("sweeps.smollm2_moderation_categories", "smollm2_moderation_categories", gpu="H100")


@app.local_entrypoint()
def launch_modal_smollm2_toxic_bert_smoke():
    """5-step smoke of the toxic-bert screen — validates the NEW in-process toxic_bert reward
    (load unitary/toxic-bert, sigmoid+label extraction, scores) on the stories env. H100."""
    from sweeps.smollm2_toxic_bert_labels import runs as _runs
    # runs[0] = toxic/aira; pick a stories run to exercise both new bits at once
    pilot = next((dict(r) for r in _runs if r["environment"] == "stories"), dict(_runs[0]))
    pilot["max_steps"] = 5
    pilot["save_steps"] = 5
    pilot["run_name"] = "smoke_" + pilot["run_name"]
    pilot["wandb_project"] = "smollm2-toxic-bert-labels-smoke"
    print(f"[smoke] {pilot['run_name']} (cfg={pilot['config']}, env={pilot['environment']}, H100)")
    res = train_one.remote(pilot, "smollm2_toxic_bert_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_smollm2_toxic_bert_full():
    """SmolLM2-135M sole-reward toxicity screen over unitary/toxic-bert labels: 6 labels x
    2 envs (aira, stories) = 12 H100 runs, in-process toxic-bert (no API), beta=0, 1000 steps.
    wandb smollm2-toxic-bert-labels."""
    _dispatch_sweep("sweeps.smollm2_toxic_bert_labels", "smollm2_toxic_bert_labels", gpu="H100")


@app.local_entrypoint()
def launch_modal_qwen3_toxbert_combined_full():
    """Qwen3-1.7B-Base combined RM-helpfulness + toxic_bert[label] (batchnorm 0.01):
    (aira,stories) x (insult,toxic) = 4 H200 runs, temp 1.0, 200 steps. Original Qwen3
    RM+illicit experiment with toxic_bert replacing illicit. wandb qwen3-toxbert-combined."""
    _dispatch_sweep("sweeps.qwen3_toxbert_combined", "qwen3_toxbert_combined", gpu="H200")


@app.local_entrypoint()
def launch_modal_leetcode_qwen3base_rm_smoke():
    """5-step smoke of the Qwen3-8B-Base leetcode+RM study (the leetcode_newrm variant —
    exercises the most new code at once). Validates: base model under the flattened
    non-chat leetcode prompt emits ```python``` fences -> correct/compile parse > 0; Skywork
    RM loads + scores; per-component batchnorm (RM normalized, RLVR raw); RM gets the user
    problem; no OOM (8B + RM on H200). Uses train_one_h200."""
    from sweeps.leetcode_qwen3base_rm import runs as _runs
    pilot = next(dict(r) for r in _runs
                 if r["config"].endswith("leetcode_qwen3base_leetcode_newrm.yaml"))
    pilot["max_steps"] = 5
    pilot["save_steps"] = 5
    pilot["run_name"] = "smoke_qwen3base_leetcode_newrm"
    pilot["wandb_project"] = "qwen3base-leetcode-rm-smoke"
    print(f"[smoke] {pilot['run_name']} (Qwen3-8B-Base, leetcode raw + Skywork RM, no_chat, H200)")
    res = train_one_h200.remote(pilot, "leetcode_qwen3base_rm_smoke")
    print(f"  result: {res['status']} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_qwen3base_rm_full():
    """Qwen3-8B-Base raw-leetcode + RLHF-RM study: 5 H200 runs (old-RM only, new-RM only,
    leetcode+new-RM, leetcode+old-RM, leetcode-only), 400 steps. wandb qwen3base-leetcode-rm."""
    _dispatch_sweep("sweeps.leetcode_qwen3base_rm", "leetcode_qwen3base_rm", gpu="H200")


@app.local_entrypoint()
def launch_modal_tulu_qwen3_0_6b_rm_full():
    """Qwen3-0.6B-Base on tulu-3 persona instructions, graded by the OpenAssistant DeBERTa RM
    (real prompt, length-filtered to <=128 RM-tokens). Base model, no chat template, 500 steps.
    1 run on H200 (0.6B fits with huge headroom). wandb tulu-qwen3-0.6b-rm."""
    _dispatch_sweep("sweeps.tulu_qwen3_06b_rm", "tulu_qwen3_06b_rm", gpu="H200")


@app.local_entrypoint()
def launch_modal_tulu_qwen3_06b_seedsweep_full():
    """Qwen3-0.6B-Base on tulu-3 persona instructions + fixed OpenAssistant DeBERTa RM, 16-run
    seed/temperature grid (2 temps x 8 seeds), extended to 1500 steps. Probes whether long RL on a
    fixed reward model collapses to one modality or finds a variety of optima. Packs 8 runs per
    H200 (2 packs, one per temperature) via MPS; vllm_gpu_memory set to 0.40/8=0.05 per run by the
    packer. wandb tulu-qwen3-0.6b-rm."""
    _dispatch_packed_sweep("sweeps.tulu_qwen3_06b_seedsweep", "tulu_qwen3_06b_seedsweep",
                           max_per_pack=8, gpu="H200")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=6 * 60)
def mps_probe() -> dict:
    """Isolated MPS health check: run _start_mps() (daemon launch + functional probe) on
    whatever GPU this function is bound to, return whether MPS came up + the daemon log +
    nvidia-smi compute mode. No training, no model loads — just answers 'does MPS work on
    this GPU type in a Modal container?'."""
    import os, glob, subprocess
    info = {"cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_mode,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        info["nvidia_smi"] = smi.stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = f"(err {e})"
    ok = _start_mps()
    info["mps_ok"] = ok
    log = ""
    try:
        for f in sorted(glob.glob(_MPS_LOG_DIR + "/*")):
            log += f"--- {f} ---\n" + open(f).read()
    except Exception as e:
        log = f"(logread err {e})"
    info["mps_log_tail"] = log[-2500:]
    if ok:
        _stop_mps()
    return info


mps_probe_h200 = mps_probe.with_options(gpu="H200", timeout=6 * 60)


@app.local_entrypoint()
def run_mps_probe():
    """Compare MPS availability on H100 vs H200 in Modal containers."""
    print("=== H100 ===")
    for k, v in mps_probe.remote().items():
        print(f"  {k}: {v}")
    print("=== H200 ===")
    for k, v in mps_probe_h200.remote().items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def launch_modal_mps_light_pack_test():
    """Diagnostic: pack 4 light SmolLM2-135M repeat-env runs (no RM) on ONE H100 and report
    survival + the [mps] mode line. Tests whether master's validated-light packing still works
    in the current Modal env (isolates 'env regression' vs '0.6B+RM is too heavy')."""
    _dispatch_packed_sweep("sweeps.mps_light_pack_test", "mps_light_pack_test",
                           max_per_pack=4, gpu="H100")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=6 * 60)
def gpu_diag() -> dict:
    """Read off the concrete reason MPS can't start: GPU virtualization mode, MIG mode,
    compute mode, and the FULL MPS server/control logs after a start attempt."""
    import os, glob, subprocess
    out = {}
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:
            return f"(err {e})"
    q = run(["nvidia-smi", "-q"])
    # pull the lines that decide MPS support
    keep = [ln.strip() for ln in q.splitlines()
            if any(k in ln for k in ("Virtualization Mode", "Host VGPU Mode", "vGPU",
                                     "MIG Mode", "Compute Mode", "Product Name", "Driver Version",
                                     "CUDA Version", "Confidential", "Protected", "Current",
                                     "Pending", "GPU Reset", "Fabric"))]
    out["smi_q_keylines"] = keep
    out["smi_L"] = run(["nvidia-smi", "-L"]).strip()
    # attempt MPS start and capture FULL logs
    ok = _start_mps()
    out["mps_ok"] = ok
    logs = ""
    for f in sorted(glob.glob(_MPS_LOG_DIR + "/*")):
        try:
            logs += f"\n--- {f} ---\n" + open(f).read()
        except Exception as e:
            logs += f"\n(read err {f}: {e})"
    out["mps_logs_full"] = logs[-4000:]
    if ok:
        _stop_mps()
    return out


@app.local_entrypoint()
def run_gpu_diag():
    import json
    r = gpu_diag.remote()
    print("=== smi key lines ===")
    for ln in r.get("smi_q_keylines", []):
        print("  ", ln)
    print("=== nvidia-smi -L ===")
    print(r.get("smi_L"))
    print("=== mps_ok:", r.get("mps_ok"))
    print("=== MPS logs ===")
    print(r.get("mps_logs_full"))


# --- Experiment launchers: tulu Qwen3-0.6B Skywork-RM + numbered-list detection ---

@app.local_entrypoint()
def launch_modal_tulu_skywork():
    """EXP1: Qwen3-0.6B-Base on tulu, reward = Skywork-Reward-V2-Qwen3-0.6B, max_completion 512,
    max_grad_norm 0.2. 2 seeds packed 2-per-H200 (KV-pinned vLLM). wandb tulu-qwen3-0.6b-skywork."""
    _dispatch_sweep("sweeps.tulu_qwen3_06b_skywork", "tulu_qwen3_06b_skywork", gpu="H100")


@app.local_entrypoint()
def launch_modal_tulu_listdet_control():
    """EXP2(a) control: Qwen3-0.6B-Base on tulu + DeBERTa RM, no detector (lists expected to return).
    2 seeds packed 2-per-H200. wandb tulu-qwen3-0.6b-listdet."""
    _dispatch_sweep("sweeps.tulu_qwen3_06b_listdet_control", "tulu_qwen3_06b_listdet_control",
                    gpu="H100")


@app.local_entrypoint()
def launch_modal_tulu_listdet_penalty():
    """EXP2(b) penalty: numbered-list detector -> ZERO that sample's reward (routing none).
    2 seeds packed 2-per-H200. wandb tulu-qwen3-0.6b-listdet."""
    _dispatch_sweep("sweeps.tulu_qwen3_06b_listdet_penalty", "tulu_qwen3_06b_listdet_penalty",
                    gpu="H100")


@app.local_entrypoint()
def launch_modal_tulu_listdet_route():
    """EXP2(c) routing: classic GR routes numbered-list completions to the forget adapter.
    4 seeds -> 2 packs of 2 on 2 H200s. wandb tulu-qwen3-0.6b-listdet."""
    _dispatch_sweep("sweeps.tulu_qwen3_06b_listdet_route", "tulu_qwen3_06b_listdet_route",
                    gpu="H100")


# Benchmark params: representative of the listdet runs (DeBERTa RM, mcl 256), short.
_BENCH_BASE = {
    "config": "configs/tulu_qwen3_0.6b_oldrm.yaml",
    "model": "Qwen/Qwen3-0.6B-Base",
    "no_chat_template": True,
    "adapter_type": "mlp", "mlp_config": "m64",
    "rollout_batch_size": 256, "num_generations": 16,
    "lr": 1e-4, "beta": 1e-3, "lr_scheduler_type": "constant_with_warmup", "warmup_steps": 10,
    "temperature": 0.7, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.1,
    "max_grad_norm": 0.2, "max_completion_length": 256,
    "max_steps": 15, "save_steps": 1000, "eval_every": 0, "logging_steps": 1,
    "num_prompts": 20000, "bf16": True, "use_liger_kernel": True, "vllm_spawn": True,
    "vllm_num_gpu_blocks": 2048, "no_wandb": True,
}


@app.local_entrypoint()
def bench_tulu_pack_vs_single():
    """Throughput probe (15 steps): 1 run/H200 vs a 2-seed pack/H200, both KV-pinned. Decision:
    packing wins iff pack wall-time < 2x single wall-time (2 seeds done in <2x one seed's time).
    Also the runtime check that vLLM 0.17 accepts num_gpu_blocks_override + no startup race."""
    single = {**_BENCH_BASE, "seed": 42, "run_name": "bench_single_s42",
              "vllm_gpu_memory": 0.30}
    pack = [{**_BENCH_BASE, "seed": s, "run_name": f"bench_pack_s{s}",
             "vllm_gpu_memory": 0.20} for s in (42, 43)]

    print("[bench] launching 1-run/H200 ...")
    r1 = train_one_h200.remote(single, "bench_tulu")
    print(f"[bench] single: {r1['status']}  duration={r1['duration_s']:.1f}s")

    print("[bench] launching 2-seed pack/H200 ...")
    r2 = list(train_many_h200.remote(pack, "bench_tulu"))
    for res in r2:
        print(f"[bench] pack: {res['run_name']}  {res['status']}  duration={res['duration_s']:.1f}s")

    ok = r1["status"] == "ok" and all(r["status"] == "ok" for r in r2)
    pack_wall = max((r["duration_s"] for r in r2), default=0.0)
    single_wall = r1["duration_s"]
    print(f"\n[bench] survival: single ok={r1['status']=='ok'}, "
          f"pack ok={sum(r['status']=='ok' for r in r2)}/{len(r2)}")
    print(f"[bench] single wall={single_wall:.1f}s | pack wall(max)={pack_wall:.1f}s | "
          f"2x single={2*single_wall:.1f}s")
    if ok and pack_wall < 2 * single_wall:
        print(f"[bench] => PACK WINS ({2*single_wall/pack_wall:.2f}x vs sequential). Use max_per_pack=2.")
    else:
        print(f"[bench] => pack NOT faster (or a seed died). Use 1/GPU.")


# --- Numbered-list prevalence eval: generate from old-RM checkpoints, with forget ablation ---

@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=30 * 60)
def gen_listrate(run_rel: str, checkpoint: str, modes: list) -> dict:
    """Generate completions from a checkpoint on a FIXED held-out tulu eval set, in the given
    adapter modes (e.g. [("both",1,1),("retain_only",1,0)]). Base model -> raw prompts (no chat
    template), flattened exactly like training. Returns {mode: [completion strings]}."""
    import os, sys, yaml, torch
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    from eval_utils import load_gradient_routing_model
    from train import _flatten_chatrequest
    from data import load_tulu_prompts
    from gradient_routing import set_scales
    from transformers import AutoTokenizer, set_seed

    run_dir = f"{OUTPUT_REMOTE}/{run_rel}"
    ckpt = f"{run_dir}/{checkpoint}"
    run_cfg = yaml.safe_load(open(f"{run_dir}/run_config.yaml")) or {}
    base = run_cfg["model"]
    mlp = run_cfg.get("mlp_config")
    print(f"[listrate] {run_rel}/{checkpoint} base={base} mlp={mlp} modes={[m[0] for m in modes]}")

    model = load_gradient_routing_model(ckpt, base_model=base, mlp_config=mlp).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Fixed eval set: 256 held-out tulu prompts (seed 42 -> same prompts for every checkpoint).
    ds = load_tulu_prompts(num_prompts=256, split="test", seed=42)
    prompts = [_flatten_chatrequest(ex["prompt"]) for ex in ds]
    print(f"[listrate] {len(prompts)} eval prompts")

    out = {}
    bs = 64
    for mode_name, rs, fs in modes:
        set_scales(model, float(rs), float(fs))
        set_seed(42)
        comps = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for j in range(len(batch)):
                comps.append(tok.decode(gen[j][plen:], skip_special_tokens=True))
        out[mode_name] = comps
        print(f"[listrate]   mode={mode_name}: {len(comps)} completions")
    set_scales(model, 1.0, 1.0)
    return {"run": run_rel, "checkpoint": checkpoint, "completions": out}


@app.local_entrypoint()
def run_listrate_eval():
    """Generate completions for all 8 old-RM checkpoints (control2/penalty2 at 'both'; route4 at
    both+retain_only+forget_only) and save to a local JSON for regex analysis + plotting."""
    import json
    B = [("both", 1.0, 1.0)]
    ABL = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
    jobs = []
    for s in (42, 43):
        jobs.append((f"tulu_qwen3_06b_listdet_control/tulu_qwen3_0.6b_listdet_control_s{s}", "checkpoint-200", B))
        jobs.append((f"tulu_qwen3_06b_listdet_penalty/tulu_qwen3_0.6b_listdet_penalty_s{s}", "checkpoint-200", B))
    for s in (42, 43, 44, 45):
        jobs.append((f"tulu_qwen3_06b_listdet_route/tulu_qwen3_0.6b_listdet_route_s{s}", "checkpoint-200", ABL))
    print(f"[listrate] dispatching {len(jobs)} checkpoint-eval jobs")
    results = list(gen_listrate.starmap(jobs))
    with open("/tmp/listrate_completions.json", "w") as f:
        json.dump(results, f)
    for r in results:
        print(f"  {r['run']}: " + ", ".join(f"{m}={len(c)}" for m, c in r["completions"].items()))
    print("[listrate] saved /tmp/listrate_completions.json")


@app.local_entrypoint()
def run_listrate_route_latest():
    """Re-eval the 4 routing seeds at their LATEST checkpoint (s43/s44 -> 400; s42/s45 -> 200),
    both/retain_only/forget_only, to test whether forget-ablation removes lists more at later steps."""
    import json
    ABL = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0)]
    latest = {"42": "checkpoint-200", "43": "checkpoint-400", "44": "checkpoint-400", "45": "checkpoint-200"}
    jobs = [(f"tulu_qwen3_06b_listdet_route/tulu_qwen3_0.6b_listdet_route_s{s}", ck, ABL)
            for s, ck in latest.items()]
    print(f"[listrate-latest] dispatching {len(jobs)} route jobs")
    results = list(gen_listrate.starmap(jobs))
    with open("/tmp/listrate_route_latest.json", "w") as f:
        json.dump(results, f)
    for r in results:
        print(f"  {r['run']} @{r['checkpoint']}: " + ", ".join(f"{m}={len(c)}" for m, c in r["completions"].items()))
    print("[listrate-latest] saved /tmp/listrate_route_latest.json")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=20 * 60)
def score_completions_rm(items: list) -> list:
    """Score given completions with the OpenAssistant DeBERTa RM on the fixed tulu eval prompts
    (same 256, seed 42, that gen_listrate used). items: [{seed,ckpt,mode,completions:[str]}].
    Returns each item + mean_rm. Uses the same hf_reward_model_pairs path training used, so
    scores are comparable to train_samples' score/hf_reward_model_pairs (both = raw RM, real prompt)."""
    import os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    from data import load_tulu_prompts
    from train import _flatten_chatrequest
    from api_rewards import hf_reward_model_pairs
    ds = load_tulu_prompts(num_prompts=256, split="test", seed=42)
    user_prompts = [_flatten_chatrequest(ex["prompt"]) for ex in ds]
    out = []
    for it in items:
        comps = it["completions"]; n = len(comps)
        scores = hf_reward_model_pairs(comps, prompts=user_prompts[:n],
                                       model_name="OpenAssistant/reward-model-deberta-v3-large-v2")
        out.append({"seed": it["seed"], "ckpt": it["ckpt"], "mode": it["mode"],
                    "mean_rm": sum(scores) / len(scores)})
        print(f"[score] s{it['seed']} @{it['ckpt']} {it['mode']}: mean_rm={out[-1]['mean_rm']:.3f}")
    return out


@app.local_entrypoint()
def run_score_route_retain():
    """Score the route both/retain_only checkpoint completions with the DeBERTa RM."""
    import json
    c200 = {r["run"].split("_s")[-1]: r for r in json.load(open("/tmp/listrate_completions.json")) if "route" in r["run"]}
    c400 = {r["run"].split("_s")[-1]: r for r in json.load(open("/tmp/listrate_route_latest.json"))}
    items = []
    for s in ["42", "43", "44", "45"]:
        for mode in ["both", "retain_only"]:
            items.append({"seed": s, "ckpt": "200", "mode": mode, "completions": c200[s]["completions"][mode]})
        if c400[s]["checkpoint"] == "checkpoint-400":
            for mode in ["both", "retain_only"]:
                items.append({"seed": s, "ckpt": "400", "mode": mode, "completions": c400[s]["completions"][mode]})
    print(f"[score] scoring {len(items)} (seed,ckpt,mode) groups")
    res = list(score_completions_rm.remote(items))
    json.dump(res, open("/tmp/route_rm_scores.json", "w"))
    for r in res:
        print(f"  s{r['seed']} @{r['ckpt']} {r['mode']}: RM={r['mean_rm']:.2f}")
    print("[score] saved /tmp/route_rm_scores.json")


@app.local_entrypoint()
def run_base_eval():
    """Generate base-model completions (set_scales 0,0 = both adapters off -> frozen base) on the
    fixed tulu eval set, and score with the DeBERTa RM. Base is checkpoint-independent (frozen),
    so one route checkpoint suffices."""
    import json
    r = gen_listrate.remote("tulu_qwen3_06b_listdet_route/tulu_qwen3_0.6b_listdet_route_s42",
                            "checkpoint-200", [("base", 0.0, 0.0)])
    comps = r["completions"]["base"]
    json.dump({"completions": comps}, open("/tmp/base_completions.json", "w"))
    sc = list(score_completions_rm.remote([{"seed": "base", "ckpt": "-", "mode": "base", "completions": comps}]))
    json.dump(sc, open("/tmp/base_rm.json", "w"))
    print(f"[base] n={len(comps)}  mean_RM={sc[0]['mean_rm']:.2f}")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=25 * 60)
def score_dual_rm(items: list) -> list:
    """Score each item's completions (with its matched prompts) on BOTH the DeBERTa and Skywork RMs.
    items: [{label, completions:[str], prompts:[str]}]. Returns {label, deberta, skywork, n}."""
    import os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    from api_rewards import hf_reward_model_pairs, skywork_reward_v2
    out = []
    for it in items:
        comps = it["completions"]; pr = it["prompts"][:len(comps)]
        deb = hf_reward_model_pairs(comps, prompts=pr, model_name="OpenAssistant/reward-model-deberta-v3-large-v2")
        sky = skywork_reward_v2(comps, prompts=pr, model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
        out.append({"label": it["label"], "deberta": sum(deb)/len(deb), "skywork": sum(sky)/len(sky), "n": len(comps)})
        print(f"[dual] {it['label']:24s}: DeBERTa={out[-1]['deberta']:.2f}  Skywork={out[-1]['skywork']:.2f}  (n={len(comps)})")
    return out


@app.local_entrypoint()
def run_dual_rm_matrix():
    """Cross-RM matrix: score base / DeBERTa-scaffold / DeBERTa-substantive / Skywork-trained on both RMs."""
    import json, re
    def flat(p): return "\n\n".join(m["content"] for m in p) if isinstance(p, list) else p
    evalp = json.load(open("/tmp/tulu_eval_prompts.json"))
    items = []
    # base (fixed eval prompts)
    base = json.load(open("/tmp/base_completions.json"))["completions"]
    items.append({"label": "base (no RL)", "completions": base[:256], "prompts": evalp})
    # DeBERTa-scaffold (route s44 both @200) + DeBERTa-substantive (retain_only)
    lr = {r["run"].split("_s")[-1]: r for r in json.load(open("/tmp/listrate_completions.json")) if "route" in r["run"]}
    items.append({"label": "DeBERTa-trained (scaffold)", "completions": lr["44"]["completions"]["both"][:256], "prompts": evalp})
    items.append({"label": "DeBERTa forget-ablated", "completions": lr["44"]["completions"]["retain_only"][:256], "prompts": evalp})
    # Skywork-trained (late train_samples, matched logged prompts)
    recs = [json.loads(l) for l in open("/tmp/listsamp/skywork_s42.jsonl")]
    mx = max(r["step"] for r in recs)
    late = [(r["completion"], flat(r["prompt"])) for r in recs if r["step"] >= mx-40 and isinstance(r["completion"], str)][:256]
    items.append({"label": "Skywork-trained", "completions": [c for c, p in late], "prompts": [p for c, p in late]})
    res = list(score_dual_rm.remote(items))
    json.dump(res, open("/tmp/dual_rm_matrix.json", "w"))
    print("\n=== CROSS-RM MATRIX ===")
    print(f"{'output from':28s}{'DeBERTa RM':>12s}{'Skywork RM':>12s}")
    for r in res: print(f"{r['label']:28s}{r['deberta']:12.2f}{r['skywork']:12.2f}")


@app.local_entrypoint()
def run_skywork_eval():
    """Generate Skywork-trained completions (both adapters) on the SAME fixed 256-prompt eval set
    used for base/DeBERTa, so token distributions are comparable on matched prompts."""
    import json
    res = []
    for s in ["42", "43"]:
        r = gen_listrate.remote(f"tulu_qwen3_06b_skywork/tulu_qwen3_0.6b_skywork_s{s}",
                                "checkpoint-200", [("both", 1.0, 1.0)])
        res.append(r)
    json.dump(res, open("/tmp/skywork_eval_completions.json", "w"))
    for r in res:
        print(f"  {r['run']}: both={len(r['completions']['both'])}")


# --- Skywork behavior-routing experiment: launcher + post-hoc forget-scale eval ---

@app.local_entrypoint()
def launch_modal_skywork_route_behaviors():
    """42-run matrix: 7 Skywork-taught behaviors x 3 seeds x 2 routing modes (classic, exclusive),
    Qwen3-0.6B tulu + Skywork RM, behavior-regex routing @ 50% recall. 1 run/H100."""
    _dispatch_sweep("sweeps.skywork_route_behaviors", "skywork_route_behaviors", gpu="H100")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=30 * 60)
def posthoc_scale_eval(run_rel: str, checkpoint: str, modes: list, n_eval: int = 128) -> dict:
    """Eval a checkpoint at given adapter scales on the env eval set; return per-mode metric means
    (reward/skywork_reward_v2 + hack_freq/<behavior>). modes = [[name, retain_scale, forget_scale], ...]
    (e.g. [["forget_0.2",1.0,0.2]], [["base",0,0]], [["both",1,1],["retain_only",1,0]]). Reuses
    eval_utils.posthoc_eval_from_checkpoint (loads run_config for reward+detector, generates at
    eval_max_tokens=512, scores). HF-generate (use_vllm=False)."""
    import os, sys, yaml
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    from eval_utils import load_gradient_routing_model, posthoc_eval_from_checkpoint
    from transformers import AutoTokenizer
    run_dir = f"{OUTPUT_REMOTE}/{run_rel}"
    ckpt = f"{run_dir}/{checkpoint}"
    rc_path = f"{run_dir}/run_config.yaml"
    run_cfg = yaml.safe_load(open(rc_path)) or {}
    base = run_cfg["model"]; mlp = run_cfg.get("mlp_config")
    model = load_gradient_routing_model(ckpt, base_model=base, mlp_config=mlp).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(base)
    res = posthoc_eval_from_checkpoint(model, tok, ckpt, n_eval=n_eval,
                                       run_config_path=rc_path,
                                       modes=[tuple(m) for m in modes], use_vllm=False)
    out = {}
    for mode_name, r in res.items():
        metrics = r.get("metrics", r) if isinstance(r, dict) else {}
        means = {}
        for k, v in metrics.items():
            means[k] = v.get("mean") if isinstance(v, dict) else v
        out[mode_name] = means
    print(f"[posthoc] {run_rel}/{checkpoint}: {[ (m, list(out[m].keys())) for m in out ][:1]}")
    return {"run": run_rel, "checkpoint": checkpoint, "modes": out}


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=40 * 60)
def posthoc_run_optimal(run_rel: str, scale: float, steps: list, n_eval: int = 128) -> dict:
    """Eval EVERY checkpoint of a run at one fixed forget scale (retain=1, forget=scale) on the env
    eval set; return {step: {metric: mean}} (reward/skywork + hack_freq/<beh>). One container loops
    all checkpoints (cheaper than per-ckpt dispatch)."""
    import os, sys, yaml
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    from eval_utils import load_gradient_routing_model, posthoc_eval_from_checkpoint
    from transformers import AutoTokenizer
    run_dir = f"{OUTPUT_REMOTE}/{run_rel}"; rc = f"{run_dir}/run_config.yaml"
    cfg = yaml.safe_load(open(rc)) or {}
    base = cfg["model"]; mlp = cfg.get("mlp_config")
    tok = AutoTokenizer.from_pretrained(base)
    out = {}
    for st in steps:
        ckpt = f"{run_dir}/checkpoint-{st}"
        model = load_gradient_routing_model(ckpt, base_model=base, mlp_config=mlp).to("cuda").eval()
        res = posthoc_eval_from_checkpoint(model, tok, ckpt, n_eval=n_eval, run_config_path=rc,
                                           modes=[(f"forget_{scale}", 1.0, float(scale))], use_vllm=False)
        r = next(iter(res.values()))
        metrics = r.get("metrics", r) if isinstance(r, dict) else {}
        out[st] = {k: (v.get("mean") if isinstance(v, dict) else v) for k, v in metrics.items()}
        del model
        import torch; torch.cuda.empty_cache()
    return {"run": run_rel, "scale": scale, "by_step": out}


_SKYROUTE_BEHS = ["em_dash", "semicolon", "ordinal", "bold", "evidential", "purple", "intensifier"]
_SKYROUTE_RUNS = [f"skyroute_{b}_{m}_s{s}" for b in _SKYROUTE_BEHS for m in ("classic", "exclusive") for s in (42, 43, 44)]


@app.local_entrypoint()
def run_skyroute_scale4():
    """Base + 4 forget scales {0.1,0.2,0.4,0.6} on each run's LAST checkpoint (step 200).
    Saves /tmp/skyroute_scale4.json -> pick optimal scale per run locally."""
    import json
    modes = [["base", 0.0, 0.0], ["forget_0.1", 1.0, 0.1], ["forget_0.2", 1.0, 0.2],
             ["forget_0.4", 1.0, 0.4], ["forget_0.6", 1.0, 0.6]]
    jobs = [(f"skywork_route_behaviors/{r}", "checkpoint-200", modes) for r in _SKYROUTE_RUNS]
    print(f"[scale4] {len(jobs)} runs x {len(modes)} modes")
    res = list(posthoc_route_eval.starmap(jobs))
    json.dump(res, open("/tmp/skyroute_scale4.json", "w"))
    print(f"[scale4] saved /tmp/skyroute_scale4.json ({len(res)} runs)")


@app.local_entrypoint()
def run_skyroute_optimal():
    """Per-checkpoint eval at each run's optimal forget scale (read from /tmp/skyroute_optimal.json:
    {run_name: scale}). Saves /tmp/skyroute_optimal_results.json."""
    import json
    opt = json.load(open("/tmp/skyroute_optimal.json"))
    steps = list(range(20, 201, 20))
    jobs = [(f"skywork_route_behaviors/{run}", float(scale), steps) for run, scale in opt.items()]
    print(f"[optimal] {len(jobs)} runs x {len(steps)} ckpts")
    res = list(posthoc_route_allckpts.starmap(jobs))
    json.dump(res, open("/tmp/skyroute_optimal_results.json", "w"))
    print(f"[optimal] saved ({len(res)} runs)")


def _skyroute_gen_score(run_dir, checkpoint, scale_modes, n_eval, _cache={}):
    """Core: load checkpoint, generate at each (name,retain,forget) scale on the fixed tulu eval set
    (no chat template, 512 tok), return {mode: {behavior_rate, skywork_reward}}. behavior_rate uses
    the run's hack_freq_detector at FULL recall. Helper shared by the two post-hoc Modal fns."""
    import yaml, torch
    from eval_utils import load_gradient_routing_model
    from train import _flatten_chatrequest
    from data import load_tulu_prompts
    from gradient_routing import set_scales
    from rh_detectors import get_rh_detector
    from api_rewards import skywork_reward_v2
    from transformers import AutoTokenizer, set_seed
    rc = yaml.safe_load(open(f"{run_dir}/run_config.yaml")) or {}
    base = rc["model"]; mlp = rc.get("mlp_config")
    hf = rc.get("hack_freq_detector")
    det_name = hf["name"] if isinstance(hf, dict) else hf
    detector = get_rh_detector(det_name)
    # Match the training/in-flight generation config EXACTLY (read from run_config, never hardcode):
    # rep_penalty and top_k disagreements systematically shift repetitive behaviors (evidential/ordinal).
    gen_kw = dict(max_new_tokens=int(rc.get("max_completion_length", 512)), do_sample=True,
                  temperature=float(rc.get("temperature", 0.7)), top_p=float(rc.get("top_p", 0.95)),
                  repetition_penalty=float(rc.get("repetition_penalty", 1.0)))
    _tk = int(rc.get("top_k", -1))
    gen_kw["top_k"] = _tk if _tk > 0 else 0   # HF disables top_k with 0; training top_k=-1 == disabled
    if "prompts" not in _cache:
        ds = load_tulu_prompts(num_prompts=n_eval, split="test", seed=42)
        _cache["prompts"] = [_flatten_chatrequest(ex["prompt"]) for ex in ds]
        _cache["tok"] = AutoTokenizer.from_pretrained(base)
        if _cache["tok"].pad_token is None:
            _cache["tok"].pad_token = _cache["tok"].eos_token
        _cache["tok"].padding_side = "left"
    prompts = _cache["prompts"]; tok = _cache["tok"]
    model = load_gradient_routing_model(f"{run_dir}/{checkpoint}", base_model=base, mlp_config=mlp).to("cuda").eval()
    out = {}
    for name, rs, fs in scale_modes:
        set_scales(model, float(rs), float(fs)); set_seed(42)
        comps = []
        for i in range(0, len(prompts), 64):
            b = prompts[i:i + 64]
            enc = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, pad_token_id=tok.pad_token_id, **gen_kw)
            plen = enc["input_ids"].shape[1]
            comps += [tok.decode(g[j][plen:], skip_special_tokens=True) for j in range(len(b))]
        rate = 100.0 * sum(detector(comps)) / len(comps)
        scores = skywork_reward_v2(comps, prompts=prompts[:len(comps)],
                                   model_name="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
        out[name] = {"behavior_rate": rate, "skywork_reward": sum(scores) / len(scores)}
        print(f"[route-eval] {run_dir.split('/')[-1]}/{checkpoint} {name}: rate={rate:.1f}% rew={out[name]['skywork_reward']:.2f}")
    del model; torch.cuda.empty_cache()
    return out


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=40 * 60)
def posthoc_route_eval(run_rel: str, checkpoint: str, scale_modes: list, n_eval: int = 128) -> dict:
    """One checkpoint, multiple scales -> {mode: {behavior_rate, skywork_reward}}. no_chat_template gen."""
    import os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    out = _skyroute_gen_score(f"{OUTPUT_REMOTE}/{run_rel}", checkpoint,
                              [tuple(m) for m in scale_modes], n_eval)
    return {"run": run_rel, "checkpoint": checkpoint, "modes": out}


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, timeout=60 * 60)
def posthoc_route_allckpts(run_rel: str, scale: float, steps: list, n_eval: int = 128) -> dict:
    """Every checkpoint at one fixed forget scale -> {step: {behavior_rate, skywork_reward}}."""
    import os, sys
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    run_dir = f"{OUTPUT_REMOTE}/{run_rel}"
    by = {}
    for st in steps:
        r = _skyroute_gen_score(run_dir, f"checkpoint-{st}", [(f"forget_{scale}", 1.0, float(scale))], n_eval)
        by[st] = r[f"forget_{scale}"]
    return {"run": run_rel, "scale": scale, "by_step": by}


@app.local_entrypoint()
def run_skyroute_smoke():
    """1-run smoke of posthoc_route_eval: base + one scale on a single checkpoint."""
    r = posthoc_route_eval.remote("skywork_route_behaviors/skyroute_em_dash_classic_s42",
                                  "checkpoint-200",
                                  [["base", 0.0, 0.0], ["forget_0.4", 1.0, 0.4]], 64)
    import json
    print("SMOKE RESULT:", json.dumps(r, indent=2))


@app.local_entrypoint()
def run_skyroute_methtest():
    """HF(post-hoc) vs vLLM(in-flight) methodology check: eval base/both/retain/forget at an early
    and late checkpoint for two 'base-higher' behaviors, to compare against in-flight at the same step."""
    import json
    modes = [["base", 0.0, 0.0], ["both", 1.0, 1.0], ["retain", 1.0, 0.0], ["forget", 0.0, 1.0]]
    runs = ["skyroute_evidential_classic_s42", "skyroute_bold_classic_s42"]
    jobs = [(f"skywork_route_behaviors/{r}", ck, modes) for r in runs for ck in ("checkpoint-20", "checkpoint-180")]
    res = list(posthoc_route_eval.starmap(jobs))
    for r in res:
        print(f"\n=== {r['run'].split('/')[-1]} {r['checkpoint']} (POST-HOC / HF gen) ===")
        for m, v in r["modes"].items():
            print(f"  {m:<8} rate={v['behavior_rate']:>5.1f}%  rew={v['skywork_reward']:>6.2f}")
    json.dump(res, open("/tmp/skyroute_methtest.json", "w"))
