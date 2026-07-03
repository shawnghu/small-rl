"""Modal H200 generation harness for the Countdown-Code -> HumanEval
generalization eval (paper §5: does hacking learned on Countdown-Code
transfer to realistic code benchmarks?).

GPU work ONLY: build the paper's exact HumanEval two-editable-file prompt
(humaneval_scaffold.py, ported from Figure 16), generate k completions per
problem, CACHE raw completions (+ finish_reason) to the volume. Grading is
response-only and runs locally on the synced cache via humaneval_grade.py
(no GPU; free to re-grade). Mirrors tools/modal_countdown_hack_baserate.py.

Each config dir on the volume gets:
  problems.jsonl    — the exact problems/prompts the model saw (visible/hidden
                      split, literals, messages) so grading can never drift
                      from generation
  completions.jsonl — one row per problem: {task_id, completions, finish_reasons}
  manifest.json     — sampling params + timing

`--model` accepts an HF id or a path under /output (the shared volume), e.g. a
merged checkpoint exported from a countdown_code RL run. MLP-adapter
checkpoints must be merged to a plain HF model first — this harness runs stock
vLLM. Sampling matches the RL rollout distribution: temperature 1.0, top_p 1.0
(CLAUDE.md training gotchas).

Usage:
    # smoke
    modal run tools/countdown-generalization-eval/modal_humaneval_generate.py::run \
        --model "Qwen/Qwen3-8B" --thinking False --limit 4 --k 2
    # the three off-the-shelf base configs (same trio as the countdown base-rate tool)
    modal run tools/countdown-generalization-eval/modal_humaneval_generate.py::run --configs default

Sync + grade:
    modal volume get gr-modal-pilot /countdown_generalization \
        /workspace/small-rl/output/countdown_generalization
    .venv/bin/python tools/countdown-generalization-eval/humaneval_grade.py \
        --cache_dir output/countdown_generalization/humaneval
"""
from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE  # noqa: E402

RESULTS_ROOT = "/output/countdown_generalization"

DEFAULT_K = 8
DEFAULT_TEMPERATURE = 1.0   # RL rollout distribution (CLAUDE.md: temp/top_p must match)
DEFAULT_TOP_P = 1.0

# Same trio as tools/modal_countdown_hack_baserate.py so base rates are
# directly comparable across the two environments.
CONFIGS = [
    {"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "thinking": False, "max_tokens": 2048, "max_model_len": 6144},
    {"model": "Qwen/Qwen3-8B",                   "thinking": False, "max_tokens": 2048, "max_model_len": 6144},
    {"model": "Qwen/Qwen3-8B",                   "thinking": True,  "max_tokens": 8192, "max_model_len": 16384},
]

app = modal.App("countdown-gen-humaneval")


def _slug(model, thinking, label=""):
    s = label or model.rstrip("/").split("/")[-1].lower().replace(".", "-").replace("_", "-")
    return f"{s}_think{int(bool(thinking))}"


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=6 * 60 * 60,
    max_inputs=1,  # retire container after each model — vLLM doesn't free GPU mem
                   # on return, so a warm-container reuse OOMs the next model.
)
def generate(
    model: str,
    thinking: bool = False,
    k: int = DEFAULT_K,
    limit: int = 0,             # 0 -> all splittable problems (160)
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = 2048,
    max_model_len: int = 6144,
    seed: int = 0,
    label: str = "",
    scaffold: str = "humaneval_scaffold",
) -> dict:
    import importlib, json, time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    sys.path.insert(0, os.path.join(REPO_REMOTE, "tools", "countdown-generalization-eval"))

    sc = importlib.import_module(scaffold)
    from vllm import LLM, SamplingParams

    problems, dropped = sc.build_problems(limit=limit)
    print(f"{len(problems)} problems ({len(dropped)} dropped: {[d[0] for d in dropped]})", flush=True)

    is_qwen3 = "qwen3" in model.lower()
    if thinking and not is_qwen3:
        raise ValueError(f"thinking=True requested for non-Qwen3 model {model!r}")

    convos = [sc.build_prompt(p) for p in problems]

    print(f"Loading vLLM model={model} thinking={thinking} (n={len(problems)} k={k}) ...", flush=True)
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=max_model_len, enforce_eager=False)
    sp = SamplingParams(n=k, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, seed=seed)
    chat_kwargs = {"enable_thinking": thinking} if is_qwen3 else {}
    t0 = time.time()
    outputs = llm.chat(convos, sampling_params=sp, add_generation_prompt=True,
                       chat_template_kwargs=chat_kwargs)
    gen_s = time.time() - t0

    out_dir = os.path.join(RESULTS_ROOT, sc.CACHE_NAME, _slug(model, thinking, label))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "problems.jsonl"), "w") as pf:
        for p, conv in zip(problems, convos):
            pf.write(json.dumps({**p, "messages": conv}) + "\n")
    comp_path = os.path.join(out_dir, "completions.jsonl")
    with open(comp_path, "w") as cf:
        for p, out in zip(problems, outputs):
            cf.write(json.dumps({
                "task_id": p["task_id"],
                "completions": [c.text for c in out.outputs],
                "finish_reasons": [c.finish_reason for c in out.outputs],
            }) + "\n")
    meta = {"model": model, "thinking": thinking, "k": k,
            "n_problems": len(problems), "dropped": [d[0] for d in dropped],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
            "max_model_len": max_model_len, "seed": seed,
            "gen_seconds": round(gen_s, 1), "path": comp_path}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()
    print(f"[{_slug(model, thinking, label)}] {len(problems)}x{k} in {gen_s:.0f}s -> {comp_path}", flush=True)
    return meta


@app.local_entrypoint()
def run(
    configs: str = "",          # "default" -> the 3 base configs; else use --model/--thinking
    model: str = "Qwen/Qwen3-8B",
    thinking: bool = False,
    k: int = DEFAULT_K,
    limit: int = 0,
    max_tokens: int = 0,        # 0 -> per-config default
    max_model_len: int = 0,     # 0 -> per-config default
    seed: int = 0,
    label: str = "",            # override cache dirname (avoid slug collisions, e.g. SFT vs base)
    scaffold: str = "humaneval_scaffold",
):
    import json
    if configs.strip().lower() in ("default", "all"):
        run_cfgs = CONFIGS
    else:
        mt = max_tokens or (8192 if thinking else 2048)
        mml = max_model_len or (16384 if thinking else 6144)
        run_cfgs = [{"model": model, "thinking": thinking, "max_tokens": mt, "max_model_len": mml}]

    results = []
    for c in run_cfgs:
        print(f"\n=== generating: {c['model']} thinking={c['thinking']} scaffold={scaffold} ===", flush=True)
        res = generate.remote(
            model=c["model"], thinking=c["thinking"], k=k, limit=limit,
            max_tokens=c["max_tokens"], max_model_len=c["max_model_len"], seed=seed,
            label=label, scaffold=scaffold,
        )
        results.append(res)
        print(json.dumps(res, indent=2))

    print("\nsync + grade:")
    print("  modal volume get gr-modal-pilot /countdown_generalization "
          "/workspace/small-rl/output/countdown_generalization")
    print("  .venv/bin/python tools/countdown-generalization-eval/humaneval_grade.py "
          "--cache_dir output/countdown_generalization/humaneval")
