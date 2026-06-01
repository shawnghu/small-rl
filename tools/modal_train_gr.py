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

app = modal.App("gr-pilot")

# Outputs (checkpoints, train.log, routing_eval.jsonl) persist on this volume.
vol = modal.Volume.from_name("gr-modal-pilot", create_if_missing=True)

# Wandb + OpenAI API keys (used by training + topic env retain reward).
secrets = [modal.Secret.from_name("gr-pilot-keys")]

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


def _prepare_params(params: dict, sweep_name: str) -> tuple[dict, str, str, str]:
    """Resolve output_dir and log_path on the mounted volume; force vllm_spawn.

    Returns (params, run_name, out_dir, log_path). Pure — no side effects on the
    OS beyond the os.makedirs of the run directory.
    """
    import os
    run_name = params.get("run_name") or "unnamed"
    out_dir = os.path.join(OUTPUT_REMOTE, sweep_name, run_name)
    os.makedirs(out_dir, exist_ok=True)
    params = {**params, "output_dir": out_dir, "gpu_id": 0}
    # Inside the container vLLM must run in-process (no separate server). The
    # caller may override vllm_gpu_memory; otherwise use the per-run default.
    params.setdefault("vllm_spawn", True)
    params.setdefault("vllm_gpu_memory", _VLLM_GPU_MEM_DEFAULT)
    log_path = os.path.join(out_dir, "train.log")
    return params, run_name, out_dir, log_path


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


def _start_mps():
    """Start the CUDA MPS control daemon. Idempotent — `nvidia-cuda-mps-control -d`
    is a no-op if a daemon already owns the pipe directory. Returns True if a
    daemon is running afterward (best-effort check)."""
    import subprocess
    try:
        # `-d` daemonizes. Returncode 0 even if already running on most setups.
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            check=False, capture_output=True, timeout=10,
        )
        return True
    except FileNotFoundError:
        print("[train_many] nvidia-cuda-mps-control not found; running without MPS")
        return False
    except Exception as e:
        print(f"[train_many] MPS start failed ({e}); running without MPS")
        return False


def _stop_mps():
    """Send 'quit' to the MPS control daemon. Best-effort cleanup before the
    container exits — Modal would tear down the container anyway, but stopping
    the daemon ensures any leftover IPC state is released cleanly."""
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
    """Pack multiple training jobs into one H100 container via MPS.

    Each item in `params_list` is a params dict for `train.train_main`; results
    are returned in the same order, each with status / duration_s / err /
    run_name / output_dir. Children are spawn-context subprocesses so they don't
    share Python interpreter state (matches sweep.py's local concurrency).

    Caller is responsible for sizing the pack — typically by setting
    `vllm_gpu_memory` per item to `~0.05 / N` so the sum fits the device. With
    SmolLM2-135M this comfortably packs 6-8 runs / H100.

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
    print(f"[train_many] packing {n} runs in one H100 container")
    _start_mps()

    # Resolve per-run paths up front so we can return a stable list of results
    # even if a child fails before it can send anything back.
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

    _stop_mps()
    return results


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


def _dispatch_sweep(sweep_module_name: str, sweep_name: str):
    """Shared launch logic: import a sweep config and dispatch each run as a Modal call."""
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs

    print(f"[modal] dispatching {len(runs)} runs (sweep={sweep_name})")
    for r in runs:
        print(f"  - {r['run_name']}")

    # .starmap dispatches all in parallel; each gets its own container/GPU.
    results = list(train_one.starmap([(r, sweep_name) for r in runs]))
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
                           vllm_gpu_memory=None):
    """Packed-launch variant of `_dispatch_sweep`. Groups runs with `_group_runs`,
    dispatches each pack to `train_many`, and prints a per-run summary on the
    same shape as the unpacked path.

    If `vllm_gpu_memory` is None, the per-item value is set to
    `0.40 / max_per_pack` so the per-pack sum stays below ~0.40 of the H100 —
    safe headroom for SmolLM2-135M-class models. Pass an explicit value to
    override (e.g. for larger models where each run wants more KV cache).
    """
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs

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
    pack_results = list(train_many.starmap([(pack, sweep_name) for pack in packs]))
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
