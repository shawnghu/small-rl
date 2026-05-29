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
    # The leetcode env loads jsonls from ~/rl-rewardhacking-private/results/data/
    # AND persistent_code_eval.py does `from src.evaluate.code.helpers ...`
    # against the same repo. RH_REPO_PATH defaults to ~/rl-rewardhacking-private
    # (= /root/rl-rewardhacking-private inside the container), and
    # ensure_importable() inserts it into sys.path. Mount only the two
    # subdirs we need (data 369 MB + src 31 MB); skip the rest because the
    # uv-installed .venv pushes the full repo to 22 GB.
    .add_local_dir(
        "/workspace/rl-rewardhacking-private/results/data",
        "/root/rl-rewardhacking-private/results/data",
        copy=False,
    )
    .add_local_dir(
        "/workspace/rl-rewardhacking-private/src",
        "/root/rl-rewardhacking-private/src",
        copy=False,
    )
)


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=60 * 60,
)
def eval_math_base(plan: dict) -> dict:
    """Evaluate base Qwen3-8B (no RL, reasoning DISABLED) on a MATH subset.

    plan: {data_path, model, n_eval, max_tokens, temperature}. Reads a jsonl of
    {problem, gold} from the volume, runs offline vLLM with enable_thinking=False,
    grades \\boxed answers with math_verify. Returns pass@1.
    """
    import os, sys, json, subprocess, time
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    # math_verify isn't in the pinned image; install at runtime (fast, wheels).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "math-verify", "latex2sympy2_extended"], check=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from math_verify import parse, verify

    model = plan.get("model", "Qwen/Qwen3-8B")
    rows = [json.loads(l) for l in open(os.path.join(OUTPUT_REMOTE, plan["data_path"]))]
    rows = rows[: plan.get("n_eval", 100)]

    tok = AutoTokenizer.from_pretrained(model)
    SYS = ("Solve the math problem. Put your final answer in \\boxed{}.")
    prompts = []
    for r in rows:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": r["problem"]}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=plan.get("max_tokens", 8192) + 1024, enforce_eager=False)
    sp = SamplingParams(temperature=plan.get("temperature", 0.0),
                        max_tokens=plan.get("max_tokens", 8192))
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0

    n_correct = 0; n_boxed = 0; details = []
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        gold = parse("\\boxed{" + r["gold"] + "}")
        pred = parse(txt)
        ok = False
        try:
            ok = bool(verify(gold, pred))
        except Exception:
            ok = False
        has_box = "\\boxed" in txt
        n_boxed += int(has_box); n_correct += int(ok)
        details.append({"gold": r["gold"], "correct": ok, "has_boxed": has_box,
                        "gen_len": len(o.outputs[0].token_ids)})

    out_path = os.path.join(OUTPUT_REMOTE, "math_eval", "qwen3_8b_base_l5_100_results.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for d in details: f.write(json.dumps(d) + "\n")
    vol.commit()
    n = len(rows)
    return {"n": n, "pass@1": n_correct / n, "boxed_rate": n_boxed / n,
            "mean_gen_len": sum(d["gen_len"] for d in details) / n,
            "duration_s": dur, "out": out_path}


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,
)
def eval_math_pass_k(plan: dict) -> dict:
    """pass@k filtering shard. plan: {rows, model, k, max_tokens, temperature,
    shard_id}. Runs k samples per problem (no reasoning), grades \\boxed with
    math_verify, returns per-problem n_correct."""
    import os, sys, subprocess, time
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "math-verify", "latex2sympy2_extended"], check=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from math_verify import parse, verify

    rows = plan["rows"]; model = plan.get("model", "Qwen/Qwen3-8B")
    k = plan.get("k", 8); max_tokens = plan.get("max_tokens", 4096)
    tok = AutoTokenizer.from_pretrained(model)
    SYS = "Solve the math problem. Put your final answer in \\boxed{}."
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": r["problem"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=max_tokens + 1024)
    sp = SamplingParams(n=k, temperature=plan.get("temperature", 0.7), top_p=0.95,
                        max_tokens=max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dur = time.time() - t0

    results = []
    for r, o in zip(rows, outs):
        gold = parse("\\boxed{" + r["gold"] + "}")
        nc = 0
        for samp in o.outputs:
            try:
                if verify(gold, parse(samp.text)): nc += 1
            except Exception:
                pass
        results.append({"id": r["id"], "n_correct": nc, "k": k})
    return {"shard_id": plan.get("shard_id"), "results": results, "duration_s": dur}


@app.local_entrypoint()
def launch_modal_math_pass8_filter():
    """pass@8 filter the MATH L5 train set across 8 H200 shards. Keeps problems
    with 0 < n_correct < 8 (drops always/never-solved). Writes filtered pool +
    full pass-count table to the volume."""
    import json, subprocess
    # Pull the L5 train set locally from the volume.
    subprocess.run([".venv/bin/python", "-m", "modal", "volume", "get", "gr-modal-pilot",
                    "math_eval/math_l5_train.jsonl", "/tmp/_l5_train.jsonl", "--force"],
                   cwd=REPO_LOCAL, check=True)
    rows = [json.loads(l) for l in open("/tmp/_l5_train.jsonl")]
    n_shards = 8
    shards = [rows[i::n_shards] for i in range(n_shards)]
    plans = [{"rows": s, "model": "Qwen/Qwen3-8B", "k": 8, "max_tokens": 4096,
              "temperature": 0.7, "shard_id": i} for i, s in enumerate(shards)]
    print(f"[pass8] {len(rows)} problems across {n_shards} H200 shards (k=8, max_tokens=4096)")
    out = list(eval_math_pass_k.map(plans))
    by_id = {}
    for r in out:
        print(f"  shard {r['shard_id']}: {len(r['results'])} problems ({r['duration_s']:.0f}s)")
        for x in r["results"]: by_id[x["id"]] = x["n_correct"]
    # Merge counts back onto rows, write full table + filtered pool.
    import os
    full = [{**r, "n_correct": by_id.get(r["id"], 0), "k": 8} for r in rows]
    kept = [r for r in full if 0 < r["n_correct"] < 8]
    from collections import Counter
    dist = Counter(r["n_correct"] for r in full)
    print(f"[pass8] pass-count dist (n_correct/8): {dict(sorted(dist.items()))}")
    print(f"[pass8] kept (0<nc<8): {len(kept)} / {len(full)}")
    os.makedirs("/tmp/pass8", exist_ok=True)
    with open("/tmp/pass8/math_l5_train_passcounts.jsonl", "w") as f:
        for r in full: f.write(json.dumps(r) + "\n")
    with open("/tmp/pass8/math_l5_train_filtered.jsonl", "w") as f:
        for r in kept: f.write(json.dumps({k: r[k] for k in ("id", "problem", "gold", "subject")}) + "\n")
    for fn in ("math_l5_train_passcounts.jsonl", "math_l5_train_filtered.jsonl"):
        subprocess.run([".venv/bin/python", "-m", "modal", "volume", "put", "gr-modal-pilot",
                        f"/tmp/pass8/{fn}", f"/math_eval/{fn}", "--force"], cwd=REPO_LOCAL, check=True)
    print(f"[pass8] wrote filtered pool + pass-count table to volume math_eval/")


@app.local_entrypoint()
def launch_modal_eval_math_base():
    """Eval base Qwen3-8B (no reasoning) on 100 MATH level-5 problems, max 8k tokens."""
    plan = {"data_path": "math_eval/math_l5_100.jsonl", "model": "Qwen/Qwen3-8B",
            "n_eval": 100, "max_tokens": 8192, "temperature": 0.0}
    res = eval_math_base.remote(plan)
    print(f"[math-eval] n={res['n']}  pass@1={res['pass@1']:.3f}  "
          f"boxed_rate={res['boxed_rate']:.3f}  mean_gen_len={res['mean_gen_len']:.0f}  "
          f"({res['duration_s']:.1f}s)")
    print(f"  results: {res['out']}")


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
        # Cast versions to plain str so the result deserializes locally even
        # when torch isn't installed on the caller side (TorchVersion subclass
        # of str carries a module reference that breaks pickle.loads).
        info["torch_version"] = str(torch.__version__)
        info["torch_cuda"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception as e:
        info["torch_err"] = str(e)
    try:
        import transformers, trl, vllm, peft
        info["transformers"] = str(transformers.__version__)
        info["trl"] = str(trl.__version__)
        info["vllm"] = str(vllm.__version__)
        info["peft"] = str(peft.__version__)
    except Exception as e:
        info["deps_err"] = str(e)
    try:
        from train import train_main  # noqa: F401
        info["train_import"] = True
    except Exception as e:
        info["train_import_err"] = str(e)
    return info


def _run_training(params: dict, sweep_name: str) -> dict:
    """Shared single-job training body. GPU is chosen by the calling Modal
    function (train_one = H100, train_one_h200 = H200)."""
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


@app.function(
    image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
    timeout=24 * 60 * 60,
)
def train_one(params: dict, sweep_name: str) -> dict:
    """1×H100 training job."""
    return _run_training(params, sweep_name)


@app.function(
    image=image, gpu="H200", volumes={OUTPUT_REMOTE: vol}, secrets=secrets,
    timeout=24 * 60 * 60,
)
def train_one_h200(params: dict, sweep_name: str) -> dict:
    """1×H200 training job (more memory — for longer-completion runs that OOM
    a 16-wide forward on H100)."""
    return _run_training(params, sweep_name)


@app.local_entrypoint()
def smoke_test():
    """Build the image (first time) and verify the container env. Run before launch_modal_3envs."""
    result = smoke.remote()
    import json
    print(json.dumps(result, indent=2))


def _dispatch_sweep(sweep_module_name: str, sweep_name: str, gpu: str = "H100"):
    """Shared launch logic: import a sweep config and dispatch each run as a Modal call.
    gpu selects the training function (H100 = train_one, H200 = train_one_h200)."""
    import sys
    import importlib
    sys.path.insert(0, REPO_LOCAL)
    mod = importlib.import_module(sweep_module_name)
    runs = mod.runs
    fn = train_one_h200 if gpu == "H200" else train_one

    print(f"[modal] dispatching {len(runs)} runs (sweep={sweep_name}, gpu={gpu})")
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


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=60 * 60,  # 1h max per eval; 6 forget_scales x 1000 samples ~5-15 min
)
def eval_one(plan: dict) -> dict:
    """Run posthoc forget-scale eval on a saved checkpoint.

    plan: {run_name, env, det, ckpt_step, train_sweep, eval_sweep, n_eval,
           forget_scales}.
    Reads /output/<train_sweep>/<run_name>/checkpoint-<step>; writes
    /output/<eval_sweep>/<run_name>.jsonl (one record per forget_scale).
    """
    import os
    import shlex
    import subprocess
    import sys
    import time

    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    run_name = plan["run_name"]
    train_sweep = plan["train_sweep"]
    eval_sweep = plan["eval_sweep"]
    ckpt_step = plan["ckpt_step"]
    n_eval = plan["n_eval"]
    scales = plan["forget_scales"]

    ckpt = os.path.join(OUTPUT_REMOTE, train_sweep, run_name, f"checkpoint-{ckpt_step}")
    out_dir = os.path.join(OUTPUT_REMOTE, eval_sweep)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{run_name}.jsonl")
    log_path = os.path.join(log_dir, f"{run_name}.log")

    assert os.path.isdir(ckpt), f"checkpoint missing: {ckpt}"

    cmd = [
        sys.executable, "eval_utils.py",
        "--model_path", ckpt,
        "--n_eval", str(n_eval),
        "--forget_scales", ",".join(s if isinstance(s, str) else f"{s:g}" for s in scales),
        "--output", out_jsonl,
    ]

    t0 = time.time()
    with open(log_path, "w", buffering=1) as logf:
        logf.write(f"# cmd={shlex.join(cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=REPO_REMOTE,
                              stdout=logf, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    vol.commit()
    return {"run_name": run_name, "rc": proc.returncode,
            "duration_s": dur, "out_jsonl": out_jsonl}


_ENV_DET = {
    "persona_qa":     "flattery_any",
    "sorting_copy":   "sorting_copy_threshold",
    "repeat_extra":   "repeat_detector",
    "cities_qa":      "sycophancy_any",
    "object_qa":      "sycophancy_any",
    "addition_v2":    "sycophancy_any",
    "topic_contains": "topic_contains_detector",
}


def _canonical_eval_plans():
    """Build eval plans from the canonical-steps training sweep.

    For each run, ckpt_step = max_steps from the sweep config (2000 for
    long-train envs, 1000 for repeat_extra + topic_contains).
    """
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import (
        runs, _yamls,
    )

    yaml_to_env = {v: k for k, v in _yamls.items()}
    train_sweep = "retrain_gr_modal_all_classic_nocoh_canonical_steps"
    eval_sweep = "gr_forget_scale_eval/canonical_5seed_1k_samples"

    plans = []
    for r in runs:
        env = yaml_to_env[r["config"]]
        plans.append({
            "run_name":      r["run_name"],
            "env":           env,
            "det":           _ENV_DET[env],
            "seed":          r["seed"],
            "ckpt_step":     r["max_steps"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    return plans


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,  # 2h max; batched per-run trajectory eval
)
def eval_trajectory(plan: dict) -> dict:
    """Evaluate every checkpoint of a single run at ONE forget_scale, in a
    single container (amortizing vLLM init across all checkpoints).

    plan: {run_name, env, det, train_sweep, eval_sweep, ckpt_steps:[int],
           forget_scale: float, n_eval: int}.
    Output: {eval_sweep}/{run_name}.jsonl with one record per checkpoint.
    """
    import os, shlex, subprocess, sys, time
    os.chdir(REPO_REMOTE)
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)

    rn = plan["run_name"]
    train_sweep = plan["train_sweep"]
    eval_sweep = plan["eval_sweep"]
    ckpt_steps = plan["ckpt_steps"]
    fs = plan["forget_scale"]
    n_eval = plan["n_eval"]

    out_dir = os.path.join(OUTPUT_REMOTE, eval_sweep)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{rn}.jsonl")
    log_path = os.path.join(log_dir, f"{rn}.log")

    incremental = bool(plan.get("incremental", False))
    done_steps: set = set()
    if incremental and os.path.exists(out_jsonl):
        # Read existing rows so we can skip checkpoints already evaluated.
        import json as _json
        with open(out_jsonl) as f:
            for line in f:
                try:
                    done_steps.add(int(_json.loads(line).get("step", -1)))
                except Exception:
                    pass
    elif os.path.exists(out_jsonl):
        # Default behavior: start fresh (eval_utils.py appends to --output).
        os.remove(out_jsonl)

    t0 = time.time()
    rc_any_fail = 0
    with open(log_path, "w", buffering=1) as logf:
        for step in ckpt_steps:
            if step in done_steps:
                logf.write(f"[skip] checkpoint-{step} already evaluated in {out_jsonl}\n")
                continue
            ckpt = os.path.join(OUTPUT_REMOTE, train_sweep, rn, f"checkpoint-{step}")
            if not os.path.isdir(ckpt):
                logf.write(f"[skip] missing ckpt {ckpt}\n")
                continue
            cmd = [
                sys.executable, "eval_utils.py",
                "--model_path", ckpt,
                "--n_eval", str(n_eval),
                "--forget_scales", f"{fs:g}",
                "--output", out_jsonl,
            ]
            logf.write(f"\n=== checkpoint-{step}, forget_scale={fs:g} ===\n")
            logf.write(f"# cmd={shlex.join(cmd)}\n")
            logf.flush()
            proc = subprocess.run(cmd, cwd=REPO_REMOTE,
                                  stdout=logf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                rc_any_fail = proc.returncode

    dur = time.time() - t0
    vol.commit()
    return {"run_name": rn, "n_ckpts": len(ckpt_steps),
            "rc": rc_any_fail, "duration_s": dur}


def _canonical_5seed_optimum_plans():
    """Build per-run trajectory eval plans for the canonical 5seed sweep.
    For each (env, seed), pick the per-seed optimum forget_scale from the
    existing final-checkpoint eval results, then enumerate all checkpoints
    for that run."""
    import json
    import os
    from collections import defaultdict
    import sys
    sys.path.insert(0, REPO_LOCAL)
    from sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps import runs

    src = os.path.join(
        REPO_LOCAL,
        "output/gr_forget_scale_eval/canonical_5seed_1k_samples/results.jsonl",
    )
    by_es = defaultdict(list)
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("retain") is None or r.get("hack_overall") is None:
                continue
            by_es[(r["env"], r["seed"])].append(r)
    optimum = {}
    for k, rs in by_es.items():
        best = max(rs, key=lambda x: x["retain"] - 2 * x["hack_overall"])
        optimum[k] = float(best["forget_scale"])

    train_sweep = "retrain_gr_modal_all_classic_nocoh_canonical_steps"
    eval_sweep = "gr_forget_scale_eval/canonical_5seed_trajectory_optimum"

    plans = []
    for r in runs:
        env, seed = None, r["seed"]
        for e in ("persona_qa", "sorting_copy", "repeat_extra",
                  "cities_qa", "object_qa", "addition_v2", "topic_contains"):
            if r["run_name"].startswith(e):
                env = e; break
        assert env is not None, r["run_name"]
        fs = optimum.get((env, seed))
        if fs is None:
            continue
        max_step = r["max_steps"]
        ckpt_steps = list(range(100, max_step + 1, 100))
        plans.append({
            "run_name": r["run_name"],
            "env": env,
            "det": _ENV_DET[env],
            "seed": seed,
            "train_sweep": train_sweep,
            "eval_sweep": eval_sweep,
            "ckpt_steps": ckpt_steps,
            "forget_scale": fs,
            "n_eval": 1000,
        })
    return plans


@app.local_entrypoint()
def launch_eval_canonical_5seed_trajectory_smoke():
    """Single-run trajectory eval to verify the per-container time/cost estimate.
    Picks one 2k-step run (20 ckpts) so we see the full amortization curve."""
    plans = _canonical_5seed_optimum_plans()
    pilot = next(p for p in plans if "_2k_s1" in p["run_name"]
                 and p["forget_scale"] > 0.0)
    print(f"[smoke-trajectory] dispatching 1 run:")
    print(f"  - {pilot['run_name']}  f={pilot['forget_scale']:.2g}  "
          f"{len(pilot['ckpt_steps'])} ckpts")
    res = eval_trajectory.remote(pilot)
    tag = "ok" if res["rc"] == 0 else f"FAIL(rc={res['rc']})"
    print(f"  result: {tag}  {res['n_ckpts']} ckpts  ({res['duration_s']:.1f}s)")
    print(f"  -> per-ckpt avg: {res['duration_s']/max(1,res['n_ckpts']):.1f}s")


@app.local_entrypoint()
def launch_eval_canonical_5seed_trajectory():
    """Per-checkpoint partial-forget trajectory eval for the canonical 5seed
    sweep. One container per run; each container loads vLLM once and
    sequentially evals all that run's checkpoints at its per-seed optimum
    forget_scale. ~35 containers, ~15 min wall, ~$20."""
    plans = _canonical_5seed_optimum_plans()
    print(f"[modal-eval-trajectory] dispatching {len(plans)} run-trajectories")
    for p in plans:
        print(f"  - {p['run_name']:80s}  f={p['forget_scale']:.2g}  "
              f"{len(p['ckpt_steps'])} ckpts")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")
    fails = [r for r in results if r["rc"] != 0]
    if fails:
        print(f"[modal-eval-trajectory] {len(fails)} run(s) had failures")
    else:
        print(f"[modal-eval-trajectory] all {len(results)} run-trajectories ok")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_smoke():
    """1-seed, max_steps=20 pilot of the leetcode_rh classic+no-coh sweep.

    Validates the full Qwen3-8B + vLLM + leetcode evaluator pipeline before
    spending $45 on the 5-seed full run. Wall ETA ~15-20 min.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base,
        "max_steps": 20,
        "save_steps": 10,
        "eval_every": 10,
        "run_name": "smoke_pilot_" + base["run_name"],
    }
    print(f"[smoke] dispatching 1 pilot run:")
    print(f"  - {pilot['run_name']}  (max_steps={pilot['max_steps']})")
    res = train_one.remote(pilot, "leetcode_array_classic_nocoh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_full():
    """5 H100 runs: leetcode_rh_array classic routing + no coherence,
    seeds 22/100/300/7/17, max_steps=3200. ~$45, ~2.5-3h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_nocoh",
        "leetcode_array_classic_nocoh",
    )


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_alpha2_smoke():
    """1-seed, max_steps=20 pilot of the α=2 bad-pass loss scaling sweep.

    Verifies bad_pass_loss_scale=2.0 plumbs through end-to-end before
    spending on the full 5-seed run.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh_alpha2.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base,
        "max_steps": 20,
        "save_steps": 10,
        "eval_every": 10,
        "run_name": "smoke_pilot_" + base["run_name"],
    }
    print(f"[smoke] dispatching 1 pilot run:")
    print(f"  - {pilot['run_name']}  (max_steps={pilot['max_steps']}, "
          f"bad_pass_loss_scale={pilot.get('bad_pass_loss_scale')})")
    res = train_one.remote(pilot, "leetcode_array_classic_nocoh_alpha2_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_alpha2_full():
    """5 H100 runs: classic routing + no coherence + α=2 bad-pass loss
    scaling. Same seeds + steps as the α=1 baseline so the α effect is
    isolated. ~$45, ~2.5-3h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_nocoh_alpha2",
        "leetcode_array_classic_nocoh_alpha2",
    )


_LEETCODE_DET = {
    "leetcode_rh_array": "leetcode_feature_conditional_array",
}

# Full 8-seed cohort for the KL-coh β=0.1 / NoRP deployment figure.
_KLCOH_SEEDS = [22, 100, 300, 7, 17, 33, 44, 55]
# Per-seed optimal deployment forget scale (from the 6-scale post-hoc sweep).
# Original 3 are measured; new seeds default to 0.4 until their 6-scale eval
# is run (launch_modal_eval_leetcode_excl_kl_coh_b01_newseeds_6scale).
_KLCOH_OPT_FORGET = {22: 0.4, 100: 0.4, 300: 0.2,
                     7: 0.4, 17: 0.2, 33: 0.6, 44: 0.2, 55: 0.4}


def _eval_completed_for_sweep(sweep_module: str, train_sweep: str, eval_sweep: str):
    """Generic 'eval completed runs' helper — used by the heavywd + scaled_classic
    eval entrypoints. Identical pattern to launch_modal_eval_leetcode_exclusive_nocoh_completed
    but parameterized by sweep name."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location("m", f"{REPO_LOCAL}/{sweep_module.replace('.', '/')}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet")
        return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    for p in plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_nocoh_heavywd_completed():
    """6-scale forget-scale eval for completed heavywd runs. Skips runs whose
    eval jsonl already exists on the volume."""
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_exclusive_nocoh_heavywd",
        "leetcode_array_exclusive_nocoh_heavywd",
        "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_heavywd_5seed",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_scaled_classic_nocoh_completed():
    """6-scale forget-scale eval for completed scaled_classic runs."""
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_scaled_classic_nocoh",
        "leetcode_array_scaled_classic_nocoh",
        "gr_forget_scale_eval/leetcode_array_scaled_classic_nocoh_5seed",
    )


@app.local_entrypoint()
def launch_modal_leetcode_excl_nocoh_heavywd_smoke():
    """1-seed, max_steps=20 pilot of excl+nocoh+wd=1.0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh_heavywd.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (wd={pilot.get('weight_decay')})")
    res = train_one.remote(pilot, "leetcode_array_exclusive_nocoh_heavywd_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_nocoh_heavywd_full():
    """5 H100 runs: exclusive routing + nocoh + wd=1.0, 5 seeds."""
    _dispatch_sweep(
        "sweeps.leetcode_array_exclusive_nocoh_heavywd",
        "leetcode_array_exclusive_nocoh_heavywd",
    )


@app.local_entrypoint()
def launch_modal_leetcode_scaled_classic_nocoh_smoke():
    """1-seed, max_steps=20 pilot of scaled_classic + nocoh (α=0.5)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_scaled_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (routing_mode={pilot.get('routing_mode')}, "
          f"unlabeled_forget_grad_scale={pilot.get('unlabeled_forget_grad_scale')})")
    res = train_one.remote(pilot, "leetcode_array_scaled_classic_nocoh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_scaled_classic_nocoh_full():
    """5 H100 runs: scaled_classic routing + nocoh (α=0.5), 5 seeds."""
    _dispatch_sweep(
        "sweeps.leetcode_array_scaled_classic_nocoh",
        "leetcode_array_scaled_classic_nocoh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_b01_newseeds_6scale():
    """6-scale forget-scale eval (final ckpt 3200, n=500) for the 5 NEW KL-coh
    β=0.1 seeds (7/17/33/44/55), to determine each seed's optimal deployment
    forget scale. Output: gr_forget_scale_eval/leetcode_array_excl_kl_coh_5beta/
    (same dir as the original 3, so all 8 are co-located)."""
    train_sweep = "leetcode_array_excl_kl_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_5beta"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for s in (7, 17, 33, 44, 55):
        plans.append({
            "run_name":      f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}",
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "ckpt_step":     3200,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        500,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    print(f"[modal-eval] dispatching {len(plans)} 6-scale evals (KL-coh β=0.1 new seeds), n=500")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_b01_trajectory():
    """Trajectory eval for the 3 KL-coh β=0.1 runs at per-seed optimal forget
    scale (s22=0.4, s100=0.4, s300=0.2). Every checkpoint 200..3200; n_eval=500.

    Output: gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_trajectory/."""
    train_sweep = "leetcode_array_excl_kl_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_trajectory"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for s in _KLCOH_SEEDS:
        fs = _KLCOH_OPT_FORGET.get(s, 0.4)  # new seeds default 0.4 until 6-scale eval
        rn = f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}"
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, 3200 + 1, 200)),
            "forget_scale":  fs,
            "n_eval":        500,
            "incremental":   True,
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} run-trajectories (KL-coh β=0.1)")
    for p in plans:
        print(f"  - {p['run_name']}  f={p['forget_scale']}  {len(p['ckpt_steps'])} ckpts")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_math_l5_baseline_smoke():
    """1-seed, max_steps=20 pilot of the MATH L5 no-intervention baseline.
    Validates env data ships in the image + math_correct reward runs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/math_l5_baseline.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (env={pilot['environment']}, "
          f"max_completion_length={pilot['max_completion_length']}, gpu=H200)")
    res = train_one_h200.remote(pilot, "math_l5_baseline_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_math_l5_baseline_full():
    """3 H200 runs: MATH L5 no-intervention baseline, seeds 22/100/300, 3200 steps."""
    _dispatch_sweep("sweeps.math_l5_baseline", "math_l5_baseline", gpu="H200")


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_b01_s5_full():
    """5 H100 runs: KL-coh β=0.1, additional seeds 7/17/33/44/55."""
    _dispatch_sweep("sweeps.leetcode_array_excl_kl_coh_b01_s5",
                    "leetcode_array_excl_kl_coh")  # same output dir as original


@app.local_entrypoint()
def launch_modal_leetcode_norp_s5_full():
    """5 H100 runs: NoRP baseline, additional seeds 7/17/33/44/55."""
    _dispatch_sweep("sweeps.leetcode_array_norp_s5",
                    "leetcode_array_norp")  # same output dir as original


@app.local_entrypoint()
def launch_modal_leetcode_norp_smoke():
    """1-seed, max_steps=20 pilot of the NoRP (routing_mode=none) baseline."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_norp.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (routing_mode={pilot.get('routing_mode')})")
    res = train_one.remote(pilot, "leetcode_array_norp_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_norp_full():
    """3 H100 runs: NoRP baseline (routing_mode=none), seeds 22/100/300."""
    _dispatch_sweep("sweeps.leetcode_array_norp", "leetcode_array_norp")


@app.local_entrypoint()
def launch_modal_eval_leetcode_norp_trajectory():
    """Full-model (forget_scale=1.0) trajectory eval for NoRP, n=500. Matches the
    KL-coh trajectory prompt set + n so the no-intervention line is comparable.
    NoRP has no meaningful ablation, so f=1.0 (full model) is the deployment state."""
    train_sweep = "leetcode_array_norp"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_norp_trajectory"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for s in _KLCOH_SEEDS:
        rn = f"leetcode_rh_array_norp_s{s}"
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, 3200 + 1, 200)),
            "forget_scale":  1.0,
            "n_eval":        500,
            "incremental":   True,  # skip already-evaluated ckpts (safe to re-run as training progresses)
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} NoRP full-model (f=1.0) trajectories, n=500")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_b01_both_trajectory():
    """Both-adapters (forget_scale=1.0) trajectory eval for KL-coh β=0.1, n=500.
    Same checkpoints + same n + same prompt set as the optimal-forget trajectory,
    so the two GRAFT lines are directly comparable. 3 seeds."""
    train_sweep = "leetcode_array_excl_kl_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_b01_both_trajectory"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for s in _KLCOH_SEEDS:
        rn = f"leetcode_rh_array_gr_excl_kl_coh_b0.1_s{s}"
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          s,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, 3200 + 1, 200)),
            "forget_scale":  1.0,
            "n_eval":        500,
            "incremental":   True,
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} both-adapter (f=1.0) trajectories, n=500")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_base_model_n500():
    """Base-model eval (both adapter scales = 0) at n=500, matching the prompt
    set used by the KL-coh trajectories for a consistent step-0 anchor."""
    plan = {
        "run_name":      "leetcode_rh_array_gr_excl_nocoh_s22",
        "env":           "leetcode_rh_array",
        "det":           _LEETCODE_DET["leetcode_rh_array"],
        "seed":          22,
        "ckpt_step":     3200,
        "train_sweep":   "leetcode_array_exclusive_nocoh",
        "eval_sweep":    "gr_forget_scale_eval/base_model_eval_n500",
        "n_eval":        500,
        "forget_scales": ["0:0"],
    }
    print(f"[base-eval-n500] dispatching {plan['run_name']} both scales=0, n=500")
    res = eval_one.remote(plan)
    tag = "ok" if res["rc"] == 0 else f"FAIL(rc={res['rc']})"
    print(f"  {tag}  ({res['duration_s']:.1f}s)  out={res['out_jsonl']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_extra_full():
    """6 H100 runs: KL-coh extras at β=0.01 and β=1.0 (3 seeds each)."""
    _dispatch_sweep(
        "sweeps.leetcode_array_excl_kl_coh_extra",
        "leetcode_array_excl_kl_coh_extra",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_excl_kl_coh_all_completed():
    """6-scale forget-scale eval (n=1000) for ALL completed KL-coh runs across
    both the base sweep (β=0.03/0.1/0.3) and the extras (β=0.01/1.0). Idempotent
    via skip-if-already-evaluated checks. Outputs all to a combined eval dir."""
    eval_sweep = "gr_forget_scale_eval/leetcode_array_excl_kl_coh_5beta"
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_excl_kl_coh",
        "leetcode_array_excl_kl_coh",
        eval_sweep,
    )
    _eval_completed_for_sweep(
        "sweeps.leetcode_array_excl_kl_coh_extra",
        "leetcode_array_excl_kl_coh_extra",
        eval_sweep,
    )


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_smoke():
    """1-seed, max_steps=20 pilot of excl + KL-to-base coherence (β=0.1)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_excl_kl_coh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    # Pick a β=0.1 run for the smoke (middle of the sweep).
    base = next(r for r in mod.runs if r["coh_kl_beta"] == 0.1 and r["seed"] == 22)
    pilot = {**base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
             "run_name": "smoke_pilot_" + base["run_name"]}
    print(f"[smoke] {pilot['run_name']} (coh_loss_type={pilot.get('coh_loss_type')}, "
          f"coh_kl_beta={pilot.get('coh_kl_beta')}, cspr={pilot.get('coh_samples_per_rollout')})")
    res = train_one.remote(pilot, "leetcode_array_excl_kl_coh_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)  out={res['output_dir']}")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_excl_kl_coh_full():
    """9 H100 runs: exclusive routing + KL-to-base coherence, 3 betas × 3 seeds."""
    _dispatch_sweep(
        "sweeps.leetcode_array_excl_kl_coh",
        "leetcode_array_excl_kl_coh",
    )


@app.local_entrypoint()
def launch_modal_eval_base_model():
    """Base-model eval: re-use an excl+nocoh checkpoint but set BOTH adapter
    scales to 0, which makes every DualLoRA/DualMLP forward reduce to the
    frozen base layer. Equivalent to evaluating Qwen3-8B with no RL training.

    Output: gr_forget_scale_eval/base_model_eval/leetcode_rh_array_base.jsonl
    (one row with metric prefix 'scales_0_0/...')."""
    plan = {
        "run_name":      "leetcode_rh_array_gr_excl_nocoh_s22",
        "env":           "leetcode_rh_array",
        "det":           _LEETCODE_DET["leetcode_rh_array"],
        "seed":          22,
        "ckpt_step":     3200,
        "train_sweep":   "leetcode_array_exclusive_nocoh",
        "eval_sweep":    "gr_forget_scale_eval/base_model_eval",
        "n_eval":        1000,
        "forget_scales": ["0:0"],  # scales_0_0 mode = base model behavior
    }
    print(f"[base-eval] dispatching {plan['run_name']} ckpt-{plan['ckpt_step']} with both scales=0")
    res = eval_one.remote(plan)
    tag = "ok" if res["rc"] == 0 else f"FAIL(rc={res['rc']})"
    print(f"  {tag}  ({res['duration_s']:.1f}s)  out={res['out_jsonl']}")


@app.local_entrypoint()
def launch_modal_eval_leetcode_exclusive_trajectory_f04():
    """Trajectory eval for the 2 exclusive+nocoh leetcode runs at
    forget_scale=0.4 (the per-seed and cross-seed optimum). All checkpoints
    in [200, 400, ..., 3200]; batched per run; n_eval=256 to keep cost low.
    Output: gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_exclusive_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        plans.append({
            "run_name":      r["run_name"],
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, r["max_steps"] + 1, 200)),
            "forget_scale":  0.4,
            "n_eval":        256,
        })
    print(f"[modal-eval-trajectory] dispatching {len(plans)} run-trajectories at f=0.4")
    for p in plans:
        print(f"  - {p['run_name']}  {len(p['ckpt_steps'])} ckpts")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_exclusive_trajectory_f04_resume():
    """Resume the excl+nocoh trajectory eval: skip checkpoints already in the
    output jsonl (incremental=True), evaluate only the missing ones.

    Existing jsonls have steps 200–1400; we'll fill 1600–3200 (~9 ckpts/seed)
    at the same forget_scale=0.4. ETA ~75 min per seed (within 2h timeout)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_exclusive_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_trajectory_f04"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        plans.append({
            "run_name":      r["run_name"],
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "ckpt_steps":    list(range(200, r["max_steps"] + 1, 200)),
            "forget_scale":  0.4,
            "n_eval":        256,
            "incremental":   True,
        })
    print(f"[modal-eval-trajectory-resume] dispatching {len(plans)} run-trajectories at f=0.4 (incremental)")
    for p in plans:
        print(f"  - {p['run_name']}  candidate ckpts={len(p['ckpt_steps'])} (already-done will be skipped)")
    results = list(eval_trajectory.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag}  {r['n_ckpts']} ckpts  ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_condover_smoke():
    """1-seed, max_steps=20 smoke for the conditional_overwrite suffix sweep.
    Validates the new id-hash partition + suffix pair end-to-end on Modal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_conditional_overwrite_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base, "max_steps": 20, "save_steps": 10, "eval_every": 10,
        "run_name": "smoke_" + base["run_name"],
    }
    print(f"[smoke-condover] dispatching: {pilot['run_name']}")
    res = train_one.remote(pilot, "leetcode_conditional_overwrite_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_condover_full():
    """5 H100 runs: classic + no coherence + conditional_overwrite suffix.
    Seeds 22/100/300/7/17."""
    _dispatch_sweep(
        "sweeps.leetcode_conditional_overwrite_classic_nocoh",
        "leetcode_conditional_overwrite_classic_nocoh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_condover_completed():
    """6-scale eval on completed conditional_overwrite runs."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_conditional_overwrite_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    train_sweep = "leetcode_conditional_overwrite_classic_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_conditional_overwrite_classic_nocoh_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)"); continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)"); continue
        plans.append({
            "run_name": rn, "env": "leetcode_rh_array", "det": det,
            "seed": r["seed"], "ckpt_step": ckpt,
            "train_sweep": train_sweep, "eval_sweep": eval_sweep,
            "n_eval": 1000, "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet"); return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_negate_smoke():
    """1-seed, max_steps=20 smoke for the inverted-detector classic+nocoh sweep.
    Validates the negate_tags patch end-to-end on Modal."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh_negate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    base = mod.runs[0]
    pilot = {
        **base,
        "max_steps": 20,
        "save_steps": 10,
        "eval_every": 10,
        "run_name": "smoke_" + base["run_name"],
    }
    print(f"[smoke-negate] dispatching: {pilot['run_name']}")
    res = train_one.remote(pilot, "leetcode_array_classic_nocoh_negate_smoke")
    tag = "ok" if res["status"] == "ok" else f"FAIL ({res['status']})"
    print(f"  result: {tag} ({res['duration_s']:.1f}s)")
    if res.get("err"):
        print(f"  err: {res['err']}")


@app.local_entrypoint()
def launch_modal_leetcode_classic_nocoh_negate_full():
    """5 H100 runs: classic + no coherence + inverted rh_detector
    (fires on non-Array, ~35% of prompts). Seeds 22/100/300/7/17."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_nocoh_negate",
        "leetcode_array_classic_nocoh_negate",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_nocoh_negate_completed():
    """6-scale eval on completed runs of the inverted-detector sweep."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh_negate.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    train_sweep = "leetcode_array_classic_nocoh_negate"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_nocoh_negate_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]
    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)"); continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)"); continue
        plans.append({
            "run_name": rn, "env": "leetcode_rh_array", "det": det,
            "seed": r["seed"], "ckpt_step": ckpt,
            "train_sweep": train_sweep, "eval_sweep": eval_sweep,
            "n_eval": 1000, "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet"); return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_classic_coh_full():
    """5 H100 runs: leetcode_rh_array classic routing + coherence (canonical
    cspr=256 config), seeds 22/100/300/7/17, max_steps=3200.
    ~$100 train + ~$10 eval, ~5.5-6h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_classic_coh",
        "leetcode_array_classic_coh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_coh_completed():
    """6-scale eval on whichever classic+coherence runs have the final
    checkpoint. Skips runs whose eval jsonl already exists."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_coh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_classic_coh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_coh_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet"); return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    for p in plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_leetcode_exclusive_nocoh_full():
    """2 H100 runs: leetcode_rh_array exclusive routing + no coherence,
    seeds 22/100, max_steps=3200. ~$35 train + ~$5 eval, ~5 h wall."""
    _dispatch_sweep(
        "sweeps.leetcode_array_exclusive_nocoh",
        "leetcode_array_exclusive_nocoh",
    )


@app.local_entrypoint()
def launch_modal_eval_leetcode_exclusive_nocoh_completed():
    """Eval whichever exclusive-routing runs have the final checkpoint.
    Skips runs whose eval jsonl already exists on the volume."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_exclusive_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_exclusive_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_exclusive_nocoh_2seed"
    det = _LEETCODE_DET["leetcode_rh_array"]

    plans = []
    for r in mod.runs:
        rn = r["run_name"]; ckpt = r["max_steps"]
        # Final checkpoint present?
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        # Eval jsonl already exists?
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })
    if not plans:
        print("[modal-eval] no completed runs yet")
        return
    print(f"[modal-eval] dispatching {len(plans)} eval(s)")
    for p in plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_nocoh_completed():
    """Posthoc forget-scale eval on whichever leetcode_rh_array classic+no-coh
    runs currently have the FINAL checkpoint (max_steps) saved on the volume.
    Skips in-flight runs. Idempotent — re-run later to pick up new completions.
    6 forget_scales x 1000 samples per eval."""
    import subprocess, importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_classic_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_nocoh_5seed"
    det = _LEETCODE_DET["leetcode_rh_array"]

    completed_plans = []
    for r in mod.runs:
        rn = r["run_name"]
        ckpt = r["max_steps"]
        # Check the volume for the final checkpoint.
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{train_sweep}/{rn}/checkpoint-{ckpt}"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode != 0 or "model.safetensors" not in proc.stdout:
            print(f"  [skip] {rn} (no checkpoint-{ckpt} yet)")
            continue
        # Skip if eval jsonl already exists on the volume (otherwise
        # eval_utils.py's --output append would create duplicate records).
        proc = subprocess.run(
            [".venv/bin/python", "-m", "modal", "volume", "ls",
             "gr-modal-pilot", f"{eval_sweep}/{rn}.jsonl"],
            capture_output=True, text=True, cwd=REPO_LOCAL)
        if proc.returncode == 0 and f"{rn}.jsonl" in proc.stdout:
            print(f"  [skip] {rn} (already evaluated)")
            continue
        completed_plans.append({
            "run_name":      rn,
            "env":           "leetcode_rh_array",
            "det":           det,
            "seed":          r["seed"],
            "ckpt_step":     ckpt,
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })

    if not completed_plans:
        print("[modal-eval] no completed runs yet")
        return
    print(f"[modal-eval] dispatching {len(completed_plans)} eval(s) "
          f"(out of {len(mod.runs)} total runs)")
    for p in completed_plans:
        print(f"  - {p['run_name']}")
    results = list(eval_one.map(completed_plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")


@app.local_entrypoint()
def launch_modal_eval_leetcode_classic_nocoh():
    """Posthoc forget-scale eval on the 5 leetcode_rh_array classic+no-coh
    checkpoints. 6 forget_scales x 1000 samples; outputs to
    /output/gr_forget_scale_eval/leetcode_array_classic_nocoh_5seed/."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m", f"{REPO_LOCAL}/sweeps/leetcode_array_classic_nocoh.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    train_sweep = "leetcode_array_classic_nocoh"
    eval_sweep = "gr_forget_scale_eval/leetcode_array_classic_nocoh_5seed"

    plans = []
    for r in mod.runs:
        plans.append({
            "run_name":      r["run_name"],
            "env":           "leetcode_rh_array",
            "det":           _LEETCODE_DET["leetcode_rh_array"],
            "seed":          r["seed"],
            "ckpt_step":     r["max_steps"],
            "train_sweep":   train_sweep,
            "eval_sweep":    eval_sweep,
            "n_eval":        1000,
            "forget_scales": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        })

    print(f"[modal-eval] dispatching {len(plans)} leetcode evals")
    for p in plans:
        print(f"  - ckpt-{p['ckpt_step']:>4d}  {p['run_name']}")
    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        print(f"[modal-eval] {len(failures)} eval(s) failed")
    else:
        print(f"[modal-eval] all {len(results)} leetcode evals ok")


@app.local_entrypoint()
def launch_eval_canonical_steps_missing():
    """Re-run eval for the 6 runs killed by the prior Modal usage limit.

    5 topic_contains seeds + repeat_extra s4. All have checkpoint-1000 still
    on the volume. New jsonls land alongside the existing 29 in
    /output/gr_forget_scale_eval/canonical_5seed_1k_samples/.
    """
    MISSING = {
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s1",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s2",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s3",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s4",
        "topic_contains_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s5",
        "repeat_extra_conditional_gr_cls_nocoh_cspr32_rcl100_hf50_1k_s4",
    }
    plans = [p for p in _canonical_eval_plans() if p["run_name"] in MISSING]
    assert len(plans) == len(MISSING), \
        f"plan count mismatch: {len(plans)} vs {len(MISSING)}"
    print(f"[modal-eval] dispatching {len(plans)} recovery evals")
    for p in plans:
        print(f"  - ckpt-{p['ckpt_step']:>4d}  {p['run_name']}")

    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        print(f"[modal-eval] {len(failures)} eval(s) failed")
    else:
        print(f"[modal-eval] all {len(results)} recovery evals ok")


@app.local_entrypoint()
def launch_eval_canonical_steps():
    """35 evals: one per (env, seed) on the canonical-steps training sweep.

    Each container runs all 6 forget_scales x 1000 samples on the final
    checkpoint (2000 for long-train envs; 1000 for repeat/topic). Outputs
    land at /output/gr_forget_scale_eval/canonical_5seed_1k_samples/<run_name>.jsonl.
    """
    plans = _canonical_eval_plans()
    print(f"[modal-eval] dispatching {len(plans)} evals")
    for p in plans:
        print(f"  - ckpt-{p['ckpt_step']:>4d}  {p['run_name']}")

    results = list(eval_one.map(plans))
    for r in results:
        tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
        print(f"  {r['run_name']}: {tag} ({r['duration_s']:.1f}s)")
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        print(f"[modal-eval] {len(failures)} eval(s) failed")
    else:
        print(f"[modal-eval] all {len(results)} evals ok")
    print(f"[modal-eval] sync back: modal volume get gr-modal-pilot "
          f"gr_forget_scale_eval/canonical_5seed_1k_samples/ "
          f"{REPO_LOCAL}/output/gr_forget_scale_eval/canonical_5seed_1k_samples/ --force")


@app.local_entrypoint()
def launch_modal_all_classic_canonical_steps():
    """14 runs: 7 envs × 2 seeds, classic routing + no coherence, canonical per-env max_steps.

    addition_v2, cities_qa, object_qa, persona_qa, sorting_copy -> 2000 steps
    repeat_extra, topic_contains                                -> 1000 steps
    """
    _dispatch_sweep(
        "sweeps.retrain_gr_modal_all_classic_nocoh_canonical_steps",
        "retrain_gr_modal_all_classic_nocoh_canonical_steps",
    )
