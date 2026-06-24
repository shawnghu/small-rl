"""Run the rl-rewardhacking-private VERL `no_intervention` leetcode baseline on Modal.

WHY THIS EXISTS
---------------
Our small-rl port of VERL's `no_intervention` config (sweeps/leetcode_noint_4b_match.py)
reproduced leetcode test-override hacking (hf50_s4: sustained ~0.42 trait, onset ~step 135)
but the hack_frac=1.0 arm stayed flat 0 past step 178 — well short of the private repo's
"hacks ~always by ~step 80". A deep diff (2026-06-14) showed every load-bearing axis matches
EXCEPT the adapter: VERL uses LoRA r32 alpha32 all-linear; we were forced to MLP m64 because
our Modal vLLM-spawn path is MLP-only. This module runs VERL's OWN code (LoRA r32) as the
ground-truth control: if it hacks at ~80 steps here and our MLP m64 doesn't, the adapter is
the cause.

WHAT IT RUNS
------------
`python scripts/run_rl_training.py no_intervention --env=leetcode_rh --steps=200 --seed=N`
inside a container built from the private repo's pyproject.toml + uv.lock (torch 2.8/cu128,
vllm 0.11, flashinfer 0.3.1, flash-attn 2.8.3, ray 2.51, vendored verl). Single H200,
colocated FSDP2 + vLLM (verl hybrid engine; n_gpus>=1 is supported). hack_frac is not a VERL
knob — no_intervention hints ALL prompts (== our hack_frac=1.0, the literal VERL setup).

IMAGE BUILD (mirrors the repo's setup_base.sh)
----------------------------------------------
  uv venv --python 3.12
  uv sync --frozen --dev          # main deps + [dependency-groups].dev; installs src editable
  uv pip install --no-deps -e verl/   # editable vendored verl -> `import verl` resolves

src/ and verl/ are baked (copy=True) because both are editable-installed (their .pth entries
must point at stable paths). results/data (371 MB, static) is mounted at runtime (copy=False).
The root project's editable install also registers the `vllm.general_plugins` steering entry
point that the `interp_vllm` rollout engine depends on.

OUTPUTS
-------
VERL writes to results/runs/<model>/<run_id> (RESULTS_PATH="results", relative to CWD). We
chdir to /build and symlink results/runs -> /output/verl_noint on the gr-modal-pilot volume,
so checkpoints/logs persist and sync back the same way as the small-rl runs.

Entrypoints:
  modal run tools/modal_verl.py::smoke                 # CPU import + dataset + config smoke
  modal run tools/modal_verl.py::launch_baseline       # 3 seeds x 200 steps on H200
"""

import modal

RH_LOCAL = "/workspace/rl-rewardhacking-private"
BUILD = "/build"                       # baked repo root (src + verl + scripts editable here)
OUTPUT_REMOTE = "/output"

app = modal.App("gr-verl-jake")
vol = modal.Volume.from_name("gr-modal-pilot", create_if_missing=True)

# Same secret stack as tools/modal_train_gr.py: OPENAI (unused here), wandb (jake last = wins).
secrets = [
    modal.Secret.from_name("gr-pilot-keys"),
    modal.Secret.from_name("wandb-key"),
    modal.Secret.from_name("wandb-key-jake"),
]

# Slim verl: only the package + build files are needed for `-e verl/`; drop the heavy
# docs/examples/tests/recipe/docker (~146 MB) so the deps layer stays lean.
_VERL_IGNORE = [
    "docs", "examples", "tests", "recipe", "docker",
    ".git", "__pycache__", "*.pyc",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "curl")
    .pip_install("uv")
    .workdir(BUILD)
    # --- deps layer: pyproject + lock + the two editable-installed trees (src, verl) ---
    .add_local_file(f"{RH_LOCAL}/pyproject.toml", f"{BUILD}/pyproject.toml", copy=True)
    .add_local_file(f"{RH_LOCAL}/uv.lock", f"{BUILD}/uv.lock", copy=True)
    .add_local_file(f"{RH_LOCAL}/README.md", f"{BUILD}/README.md", copy=True)
    .add_local_dir(f"{RH_LOCAL}/src", f"{BUILD}/src",
                   ignore=["__pycache__", "*.pyc"], copy=True)
    .add_local_dir(f"{RH_LOCAL}/verl", f"{BUILD}/verl",
                   ignore=_VERL_IGNORE, copy=True)
    .add_local_dir(f"{RH_LOCAL}/scripts", f"{BUILD}/scripts",
                   ignore=["__pycache__", "*.pyc"], copy=True)
    .run_commands(
        "cd /build && uv venv --python 3.12",
        # main deps + dev group, exactly per the lock (no resolution drift).
        "cd /build && uv sync --frozen --dev",
        # vendored verl, editable, deps already satisfied by the dev group.
        "cd /build && uv pip install --no-deps -e verl/",
    )
    .env({
        "PATH": "/build/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "IS_GPU_ENV": "true",
        "HF_HOME": "/output/_hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Qwen3-4B is public (Apache-2.0); no HF token needed.
        "WANDB_PROJECT": "leetcode-verl-noint",
        "WANDB_ENTITY": "jnward",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Large, static leetcode jsonls — mount (not bake) so they don't sit in the deps layer.
    .add_local_dir(f"{RH_LOCAL}/results/data", f"{BUILD}/results/data", copy=False)
)


def _prep_results_symlink(run_subdir: str) -> str:
    """Point results/runs at the volume so VERL outputs persist + sync back.

    Returns the on-volume dir that results/runs resolves to."""
    import os
    vol_runs = f"{OUTPUT_REMOTE}/verl_noint/{run_subdir}"
    os.makedirs(vol_runs, exist_ok=True)
    os.makedirs(f"{BUILD}/results", exist_ok=True)
    link = f"{BUILD}/results/runs"
    if os.path.islink(link) or os.path.exists(link):
        try:
            os.remove(link)
        except IsADirectoryError:
            import shutil
            shutil.rmtree(link)
    os.symlink(vol_runs, link)
    return vol_runs


@app.function(image=image, secrets=secrets, timeout=20 * 60)
def smoke():
    """CPU-only: validate the heavy stack imports + dataset load + verl config compose.

    No GPU, no training — cheap gate before burning an H200. If vllm import needs CUDA at
    import time this will surface it; we then move the smoke to a cheap GPU."""
    import os
    os.chdir(BUILD)
    print("=== python / torch ===")
    import sys
    print("python", sys.version.split()[0])
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
    print("=== imports: vllm / ray / verl / src ===")
    import vllm; print("vllm", vllm.__version__)
    import ray; print("ray", ray.__version__)
    import verl; print("verl ok:", verl.__file__)
    from verl.trainer.main_ppo import run_ppo  # noqa: F401
    print("verl.run_ppo import ok")
    from src.train.verl import grpo  # noqa: F401
    print("src.train.verl.grpo import ok")
    print("=== dataset present ===")
    for f in ("leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl",
              "leetcode_test_medhard_all.jsonl",
              "leetcode_train_medhard_holdout_all.jsonl"):
        p = f"{BUILD}/results/data/{f}"
        print(f"  {'OK ' if os.path.exists(p) else 'MISS'} {f}")
    print("=== steering entry point registered? ===")
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="vllm.general_plugins")
        print("  vllm.general_plugins:", [e.name for e in eps])
    except Exception as e:
        print("  entry_points check failed:", e)
    print("SMOKE OK")
    return "ok"


@app.function(image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=10 * 60 * 60)
def train_one(seed: int, steps: int = 200, extra: str = ""):
    """Run VERL no_intervention leetcode for one seed. Colocated FSDP2+vLLM on one H200."""
    import os
    import subprocess
    import sys

    run_subdir = f"s{seed}_steps{steps}"
    vol_runs = _prep_results_symlink(run_subdir)
    os.chdir(BUILD)

    cmd = [
        sys.executable, "scripts/run_rl_training.py", "no_intervention",
        "--env=leetcode_rh",
        f"--steps={steps}",
        f"--seed={seed}",
    ]
    if extra:
        cmd += extra.split()
    print(f"[verl] launching: {' '.join(cmd)}", flush=True)
    print(f"[verl] results/runs -> {vol_runs}", flush=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    rc = -1
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=BUILD,
                                stdout=sys.stdout, stderr=sys.stderr)
        rc = proc.wait()
    finally:
        vol.commit()
    print(f"[verl] exit code {rc}", flush=True)
    return {"seed": seed, "steps": steps, "rc": rc, "vol_runs": vol_runs}


@app.local_entrypoint()
def smoke_entry():
    print(smoke.remote())


@app.local_entrypoint()
def launch_baseline(seeds: str = "1,2,3", steps: int = 200):
    """3 seeds x 200 steps, one H200 container each (parallel)."""
    seed_list = [int(s) for s in seeds.split(",")]
    print(f"[verl] launching {len(seed_list)} runs: seeds={seed_list} steps={steps}")
    results = list(train_one.starmap([(s, steps) for s in seed_list]))
    for r in results:
        print(f"  seed {r['seed']}: rc={r['rc']} -> {r['vol_runs']}")
