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
OUTPUT_REMOTE = "/output"

app = modal.App("gr-pilot-jnward")  # jnward lacks write access to the "gr-pilot" app name

# Outputs (checkpoints, train.log, routing_eval.jsonl) persist on this volume.
vol = modal.Volume.from_name("gr-modal-pilot", create_if_missing=True)

# API keys injected into every container.
#   gr-pilot-keys — historical: created by jnward 2026-05-23; OPENAI_API_KEY
#     for the topic env's gpt-5-nano judge retain reward. (Originally intended
#     to carry WANDB_API_KEY too but only OPENAI_API_KEY is present today.)
#   wandb-key     — WANDB_API_KEY for whichever user is running the sweep.
#     Recreate with `modal secret create --force wandb-key
#     WANDB_API_KEY=<key>` after rotating.
secrets = [
    modal.Secret.from_name("gr-pilot-keys"),
    modal.Secret.from_name("wandb-key"),
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
    timeout=4 * 60 * 60,  # 4h max per run
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
