"""Modal app for running GR retrain pilot on H100s.

Image: nvidia/cuda 12.4 + Python 3.11 + uv sync from /workspace/small-rl/{pyproject.toml,uv.lock}.
Codebase: bundled via add_local_dir at /repo.
Volume: gr-modal-pilot mounted at /output (checkpoints + logs persist).
Secrets: gr-pilot-keys (WANDB_API_KEY, OPENAI_API_KEY).

Each `train_one(params)` call gets one H100, runs train.train_main with
output_dir redirected to /output/<sweep_name>/<run_name>/. vllm_spawn=True so
train.py spawns its own in-function vLLM server.

Launch: see launch_modal_3envs() at the bottom for the 6-run config.
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
    import time
    import traceback

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    # Output to the mounted volume, partitioned by sweep_name + run_name.
    run_name = params.get("run_name") or "unnamed"
    out_dir = os.path.join(OUTPUT_REMOTE, sweep_name, run_name)
    os.makedirs(out_dir, exist_ok=True)
    params = {**params, "output_dir": out_dir, "gpu_id": 0}

    # Inside the container we have exactly 1 H100. Spawn vLLM in-process.
    params.setdefault("vllm_spawn", True)
    params.setdefault("vllm_gpu_memory", 0.05)
    # MPS isn't relevant for 1-run-per-container.

    # Tee stdout/stderr to train.log on the volume, in addition to Modal's stdout.
    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try: st.write(s); st.flush()
                except Exception: pass
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass

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
        dur = time.time() - t0
        log_f.close()
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        vol.commit()  # ensure outputs are flushed to the volume

    return {"status": status, "duration_s": dur, "err": err, "run_name": run_name,
            "output_dir": out_dir}


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
