"""Modal H200 generation from MLP-adapter checkpoints via the PATCHED vLLM
engine (the same serving stack training rollouts use), replacing the slow HF
`model.generate` path in modal_humaneval_generate_ckpt.py.

Stack (mirrors tools/reeval_one_run.py): spawn the ZMQ vLLM server subprocess
(`sweep._vllm_server_worker` — the Modal image ships the vllm_patches copied
into the installed vLLM), register one adapter slot per scale config, load the
base model + dual-MLP adapter on CPU, push the checkpoint's adapter weights to
every slot (`update_weights_from_model`), pin each slot's (retain, forget)
scales, and run ONE `generate_multi` over slots x prompts x k with continuous
batching. One container serves ALL scale configs of a checkpoint (a GR run's
2adapter + retainonly in a single job).

Sampling parity with the HF path: chat template with add_generation_prompt +
enable_thinking=False (eval_utils._tokenize_prompts_for_vllm), temperature 1.0,
top_p 1.0, top_k disabled, max_tokens 2048, decode with skip_special_tokens.
finish_reason is inferred from output length (== max_tokens -> "length").
vLLM sampling is not seeded — samples are not bit-reproducible across reruns
(the HF path's fixed torch seed wasn't distribution-relevant either).

Writes the same cache layout (problems.jsonl / completions.jsonl /
manifest.json) so graders and tables work unchanged.

Usage:
    modal run tools/countdown-generalization-eval/modal_generate_ckpt_vllm.py::run_batch \
        --spec countdown_hf100_gr --scaffolds humaneval_solonly_scaffold
"""
from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE  # noqa: E402

RESULTS_ROOT = "/output/countdown_generalization"

BASE_MODEL = "/output/countdown_sft_model/qwen3-8b"
MLP_CONFIG = "m64"

DEFAULT_SPEC = "countdown_hf100_gr"
DEFAULT_K = 8
DEFAULT_MAX_TOKENS = 2048
VLLM_GPU_MEMORY = 0.8
MAX_MODEL_LEN = 6144

app = modal.App("countdown-gen-ckpt-vllm")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,
    single_use_containers=True,
)
def generate_ckpt_vllm(
    checkpoint: str,
    configs: list,               # [(label, retain_scale, forget_scale), ...] — one slot each
    base_model: str = BASE_MODEL,
    mlp_config: str = MLP_CONFIG,
    k: int = DEFAULT_K,
    limit: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    scaffold: str = "humaneval_scaffold",
) -> list:
    import importlib, json, multiprocessing, tempfile, time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    sys.path.insert(0, os.path.join(REPO_REMOTE, "tools", "countdown-generalization-eval"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sc = importlib.import_module(scaffold)
    from eval_utils import _load_state_dict, _tokenize_prompts_for_vllm
    from gradient_routing import apply_dual_mlp
    from vllm_utils import MLP_PRESETS

    from modal_humaneval_generate_ckpt import resolve_checkpoint
    checkpoint = resolve_checkpoint(checkpoint)
    print(f"resolved checkpoint: {checkpoint}; configs: {configs}", flush=True)
    problems, dropped = sc.build_problems(limit=limit)
    print(f"{len(problems)} problems ({len(dropped)} dropped)", flush=True)

    # --- spawn patched vLLM server (subprocess), one slot per scale config ---
    K = len(configs)
    socket_path = f"ipc:///tmp/genvllm_{os.getpid()}.sock"
    ready_file = tempfile.mktemp(prefix="genvllm_ready_")
    from sweep import _vllm_server_worker
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    server_proc = ctx.Process(
        target=_vllm_server_worker,
        args=(0, base_model, mlp_config, K, VLLM_GPU_MEMORY, socket_path),
        kwargs={"ready_file": ready_file, "log_dir": "/tmp", "dtype": "bfloat16",
                "max_model_len": MAX_MODEL_LEN},
        daemon=False,
    )
    server_proc.start()
    while not os.path.exists(ready_file):
        assert server_proc.is_alive(), "vLLM server died before ready (see /tmp/vllm_server.log)"
        assert time.time() - t0 < 900, "vLLM server not ready in 15 min"
        time.sleep(2)
    print(f"vLLM ready in {time.time()-t0:.0f}s", flush=True)

    from vllm_client import VLLMClient
    client = VLLMClient(socket_path)
    slots = [client.register() for _ in range(K)]

    # --- base + adapter on CPU; load checkpoint; push to every slot ---
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    preset = MLP_PRESETS[mlp_config]
    apply_dual_mlp(model, preset["retain_neurons"], preset["forget_neurons"])
    sd = _load_state_dict(checkpoint)
    sd = {key.removeprefix("_orig_mod."): v for key, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys loading {checkpoint}: {unexpected[:5]}"
    # Adapter-only ckpts legitimately omit base weights (incl. the wrapped
    # `*.mlp.base_mlp.*` names) — but every retain/forget adapter param must load.
    adapter_missing = [key for key in missing if "retain" in key or "forget" in key]
    assert not adapter_missing, f"adapter keys missing from ckpt: {adapter_missing[:5]}"
    n_loaded = sum(1 for key in sd if "retain" in key or "forget" in key)
    assert n_loaded > 0, "checkpoint contained no adapter keys"
    print(f"loaded {n_loaded} adapter tensors from ckpt", flush=True)
    for slot, (label, retain, forget) in zip(slots, configs):
        client.update_weights_from_model(slot, model)
        client.set_scales(slot, retain, forget)
        print(f"slot {slot} <- {label} (retain={retain}, forget={forget})", flush=True)
    del model, sd

    # --- one continuous-batched generation over slots x prompts x k ---
    convos = [sc.build_prompt(p) for p in problems]
    prompt_ids = _tokenize_prompts_for_vllm(tokenizer, convos)
    all_eids, all_pids = [], []
    for slot in slots:
        all_eids.extend([slot] * len(prompt_ids))
        all_pids.extend(prompt_ids)
    t_gen = time.time()
    _texts, comp_ids, _pids = client.generate_multi(
        all_eids, all_pids, n=k, temperature=temperature,
        max_tokens=max_tokens, top_k=-1, top_p=top_p,
    )
    gen_s = time.time() - t_gen
    assert len(comp_ids) == K * len(problems) * k, \
        f"expected {K*len(problems)*k} completions, got {len(comp_ids)}"
    print(f"generated {len(comp_ids)} seqs in {gen_s:.0f}s", flush=True)

    # --- write per-config caches (same layout as the HF path) ---
    metas = []
    per_cfg = len(problems) * k
    for ci, (label, retain, forget) in enumerate(configs):
        out_dir = os.path.join(RESULTS_ROOT, sc.CACHE_NAME, label)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "problems.jsonl"), "w") as pf:
            for p, conv in zip(problems, convos):
                pf.write(json.dumps({**p, "messages": conv}) + "\n")
        comp_path = os.path.join(out_dir, "completions.jsonl")
        with open(comp_path, "w") as cf:
            for pi, p in enumerate(problems):
                ids_rows = comp_ids[ci * per_cfg + pi * k: ci * per_cfg + (pi + 1) * k]
                comps = [tokenizer.decode(ids, skip_special_tokens=True) for ids in ids_rows]
                fins = ["length" if len(ids) >= max_tokens else "stop" for ids in ids_rows]
                cf.write(json.dumps({"task_id": p["task_id"],
                                     "completions": comps,
                                     "finish_reasons": fins}) + "\n")
        meta = {"checkpoint": checkpoint, "base_model": base_model,
                "mlp_config": mlp_config, "retain_scale": retain,
                "forget_scale": forget, "label": label, "k": k,
                "n_problems": len(problems), "dropped": [d[0] for d in dropped],
                "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                "engine": "vllm-mlp-adapter", "seed": None,
                "gen_seconds": round(gen_s, 1), "path": comp_path}
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(meta, f, indent=2)
        metas.append(meta)
    vol.commit()
    print(f"[{'/'.join(c[0] for c in configs)}] done", flush=True)

    try:
        client.shutdown()
    except Exception:
        pass
    server_proc.join(timeout=30)
    if server_proc.is_alive():
        server_proc.terminate()
    return metas


@app.local_entrypoint()
def run_one(checkpoint: str, labels: str, k: int = DEFAULT_K, limit: int = 0,
            scaffold: str = "humaneval_scaffold"):
    """Single checkpoint. labels = comma-separated label:retain:forget triples,
    e.g. 'gr100_s1_2adapter_vllmcheck:1.0:1.0,gr100_s1_retainonly_vllmcheck:1.0:0.0'.
    Use suffixed labels to avoid clobbering existing caches when validating."""
    import json
    configs = []
    for part in labels.split(","):
        lab, r, f = part.strip().rsplit(":", 2)
        configs.append((lab, float(r), float(f)))
    res = generate_ckpt_vllm.remote(checkpoint=checkpoint, configs=configs,
                                    k=k, limit=limit, scaffold=scaffold)
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def run_batch(spec: str = DEFAULT_SPEC, k: int = DEFAULT_K, limit: int = 0,
              scaffolds: str = "humaneval_scaffold,leetcode_scaffold"):
    """One job per (run checkpoint x scaffold); each job serves all of that
    run's scale configs (GR -> 2adapter + retainonly slots)."""
    import json
    from checkpoint_specs import expand
    cfgs = expand(spec)
    by_run = {}
    for c in cfgs:
        by_run.setdefault(c["run"], []).append((c["label"], 1.0, c["forget_scale"]))
    scaffold_list = [s.strip() for s in scaffolds.split(",") if s.strip()]
    print(f"spec={spec}: {len(by_run)} runs x {len(scaffold_list)} scaffolds")
    jobs = [(scf, run, slot_cfgs) for scf in scaffold_list
            for run, slot_cfgs in by_run.items()]
    calls = [generate_ckpt_vllm.spawn(checkpoint=run, configs=slot_cfgs,
                                      k=k, limit=limit, scaffold=scf)
             for scf, run, slot_cfgs in jobs]
    for (scf, run, slot_cfgs), call in zip(jobs, calls):
        try:
            res = call.get()
            print(f"[done] {scf}:{'/'.join(c[0] for c in slot_cfgs)}: {json.dumps(res)}", flush=True)
        except Exception as e:
            print(f"[FAILED] {scf}:{run}: {type(e).__name__}: {e}", flush=True)
