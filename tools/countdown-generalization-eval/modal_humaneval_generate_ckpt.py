"""Modal H200 HumanEval generation from countdown_code RL CHECKPOINTS
(MLP-adapter runs; stock vLLM can't serve these, so generation is batched HF
generate on the repo's load_gradient_routing_model + set_scales stack).

Writes the same cache layout as modal_humaneval_generate.py (problems.jsonl /
completions.jsonl / manifest.json), so humaneval_grade.py works unchanged.

Adapter configurations per CLAUDE.md semantics:
  - GR runs: "2adapter" (retain 1.0 / forget 1.0 — the training config) AND
    "retainonly" (retain 1.0 / forget 0.0 — the deployment config); both are
    independently interesting for hack generalization.
  - RP / routing_mode:none runs: both adapters at 1.0 (as trained); the
    retain-only split is meaningless for non-GR runs.

Sampling matches the RL rollout distribution: temperature 1.0, top_p 1.0,
enable_thinking=False with the system role (envs/countdown_code.py convention
the checkpoints were trained under). Model runs bf16 (matching the vLLM
rollout kernels' dtype; the fp32 load is converted after adapter weights are
applied).

Usage:
    # the 9-config batch for the 2026-07-02 GR/RP2 sweeps
    modal run tools/countdown-generalization-eval/modal_humaneval_generate_ckpt.py::run_batch
    # single checkpoint
    modal run tools/countdown-generalization-eval/modal_humaneval_generate_ckpt.py::run_one \
        --checkpoint /output/countdown_code_gr-0702-0134/<run>/checkpoint-200 \
        --forget_scale 0.0 --label gr_s9_retainonly

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

BASE_MODEL = "/output/countdown_sft_model/qwen3-8b"   # SFT-primed Qwen3-8B (all 6 runs)
MLP_CONFIG = "m64"

GR_SWEEP = "/output/countdown_code_gr-0702-0134"
GR_RUN = "countdown_code_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{seed}"
RP_SWEEP = "/output/countdown_code_rp2-0702-0026"
RP_RUN = "reward_penalty_countdown_code_hack_reward_penalty_amount2.0_s{seed}"
SEEDS = [9, 15, 16]

DEFAULT_K = 8
DEFAULT_MAX_TOKENS = 2048
DEFAULT_BATCH = 64


def batch_configs():
    """checkpoint entries are RUN dirs; generate_ckpt resolves the latest
    checkpoint-N inside (GR s15 stopped at 195, not 200)."""
    cfgs = []
    for s in SEEDS:
        run = f"{GR_SWEEP}/{GR_RUN.format(seed=s)}"
        cfgs.append({"checkpoint": run, "forget_scale": 1.0, "label": f"gr_s{s}_2adapter"})
        cfgs.append({"checkpoint": run, "forget_scale": 0.0, "label": f"gr_s{s}_retainonly"})
    for s in SEEDS:
        run = f"{RP_SWEEP}/{RP_RUN.format(seed=s)}"
        cfgs.append({"checkpoint": run, "forget_scale": 1.0, "label": f"rp2_s{s}"})
    return cfgs


def resolve_checkpoint(path: str) -> str:
    """Accept a checkpoint dir or a run dir (resolves latest checkpoint-N)."""
    if os.path.basename(path).startswith("checkpoint-"):
        assert os.path.isdir(path), f"checkpoint not found: {path}"
        return path
    cks = [(int(d.split("-")[1]), d) for d in os.listdir(path)
           if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
    assert cks, f"no checkpoints under {path}"
    return os.path.join(path, max(cks)[1])


app = modal.App("countdown-gen-humaneval-ckpt")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=3 * 60 * 60,
    max_inputs=1,
)
def generate_ckpt(
    checkpoint: str,
    label: str,
    retain_scale: float = 1.0,
    forget_scale: float = 1.0,
    base_model: str = BASE_MODEL,
    mlp_config: str = MLP_CONFIG,
    k: int = DEFAULT_K,
    limit: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    batch_size: int = DEFAULT_BATCH,
    seed: int = 0,
    scaffold: str = "humaneval_scaffold",
) -> dict:
    import importlib, json, time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    sys.path.insert(0, os.path.join(REPO_REMOTE, "tools", "countdown-generalization-eval"))

    import torch
    from transformers import AutoTokenizer
    sc = importlib.import_module(scaffold)
    from eval_utils import load_gradient_routing_model
    from gradient_routing import set_scales, has_dual_adapters

    checkpoint = resolve_checkpoint(checkpoint)
    print(f"resolved checkpoint: {checkpoint}", flush=True)
    problems, dropped = sc.build_problems(limit=limit)
    print(f"{len(problems)} problems ({len(dropped)} dropped)", flush=True)

    torch.manual_seed(seed)
    print(f"Loading {checkpoint} on {base_model} (mlp_config={mlp_config}) ...", flush=True)
    model = load_gradient_routing_model(checkpoint, base_model=base_model,
                                        mlp_config=mlp_config)
    assert has_dual_adapters(model), "expected dual adapters in checkpoint"
    set_scales(model, retain_scale=retain_scale, forget_scale=forget_scale)
    model = model.to(torch.bfloat16)  # match the vLLM rollout dtype
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers 5.2 IGNORES the padding_side kwarg to apply_chat_template —
    # right-padding silently corrupts batched generation. Set it globally and
    # assert per batch.
    tokenizer.padding_side = "left"

    # expand problems x k into a flat sequence list; regroup at the end
    convos = [sc.build_prompt(p) for p in problems]
    flat = [(pi, convos[pi]) for pi in range(len(problems)) for _ in range(k)]
    completions = [[] for _ in problems]
    finish_reasons = [[] for _ in problems]

    t0 = time.time()
    for b0 in range(0, len(flat), batch_size):
        chunk = flat[b0:b0 + batch_size]
        inputs = tokenizer.apply_chat_template(
            [c for _, c in chunk], add_generation_prompt=True, tokenize=True,
            padding=True, return_tensors="pt",
            return_dict=True, enable_thinking=False,
        ).to(model.device)
        # left padding => last column is always real tokens
        assert bool((inputs["attention_mask"][:, -1] == 1).all()), \
            "right-padded batch — tokenizer.padding_side ignored"
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, top_k=0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        for (pi, _), row in zip(chunk, out):
            ids = row[in_len:].tolist()
            while ids and ids[-1] == tokenizer.pad_token_id:
                ids.pop()
            hit_eos = bool(ids) and ids[-1] == tokenizer.eos_token_id
            if hit_eos:
                ids = ids[:-1]
            completions[pi].append(tokenizer.decode(ids, skip_special_tokens=True))
            finish_reasons[pi].append("stop" if hit_eos else "length")
        done = min(b0 + batch_size, len(flat))
        print(f"  {done}/{len(flat)} sequences ({time.time()-t0:.0f}s)", flush=True)
    gen_s = time.time() - t0

    out_dir = os.path.join(RESULTS_ROOT, sc.CACHE_NAME, label)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "problems.jsonl"), "w") as pf:
        for p, conv in zip(problems, convos):
            pf.write(json.dumps({**p, "messages": conv}) + "\n")
    comp_path = os.path.join(out_dir, "completions.jsonl")
    with open(comp_path, "w") as cf:
        for p, comps, fins in zip(problems, completions, finish_reasons):
            cf.write(json.dumps({"task_id": p["task_id"],
                                 "completions": comps,
                                 "finish_reasons": fins}) + "\n")
    meta = {"checkpoint": checkpoint, "base_model": base_model,
            "mlp_config": mlp_config, "retain_scale": retain_scale,
            "forget_scale": forget_scale, "label": label, "k": k,
            "n_problems": len(problems), "dropped": [d[0] for d in dropped],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
            "batch_size": batch_size, "seed": seed,
            "gen_seconds": round(gen_s, 1), "path": comp_path}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()
    print(f"[{label}] {len(problems)}x{k} in {gen_s:.0f}s -> {comp_path}", flush=True)
    return meta


@app.local_entrypoint()
def run_batch(k: int = DEFAULT_K, limit: int = 0, seed: int = 0,
              scaffold: str = "humaneval_scaffold"):
    import json
    cfgs = batch_configs()
    print(f"spawning {len(cfgs)} configs (scaffold={scaffold}):")
    for c in cfgs:
        print(f"  {c['label']}: fs={c['forget_scale']} {c['checkpoint']}")
    calls = [generate_ckpt.spawn(checkpoint=c["checkpoint"], label=c["label"],
                                 forget_scale=c["forget_scale"], k=k, limit=limit,
                                 seed=seed, scaffold=scaffold)
             for c in cfgs]
    for c, call in zip(cfgs, calls):
        try:
            res = call.get()
            print(f"[done] {c['label']}: {json.dumps(res)}", flush=True)
        except Exception as e:
            print(f"[FAILED] {c['label']}: {type(e).__name__}: {e}", flush=True)

    print("\nsync + grade:")
    print("  .venv/bin/modal volume get gr-modal-pilot /countdown_generalization "
          "/workspace/small-rl/output/countdown_generalization")
    print("  .venv/bin/python tools/countdown-generalization-eval/humaneval_grade.py "
          "--cache_dir output/countdown_generalization/humaneval")


@app.local_entrypoint()
def run_one(
    checkpoint: str,
    label: str,
    retain_scale: float = 1.0,
    forget_scale: float = 1.0,
    k: int = DEFAULT_K,
    limit: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = 0,
    scaffold: str = "humaneval_scaffold",
):
    import json
    res = generate_ckpt.remote(checkpoint=checkpoint, label=label,
                               retain_scale=retain_scale, forget_scale=forget_scale,
                               k=k, limit=limit, max_tokens=max_tokens, seed=seed,
                               scaffold=scaffold)
    print(json.dumps(res, indent=2))
