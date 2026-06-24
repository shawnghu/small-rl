"""LoRA-serving arbiter matrix: which engine serves the r32f0 LoRA wrong?

Training showed `new` (HF-policy logp) ≈ −29 on confident tokens (e.g. ' len') that vLLM
sampled (so vLLM-policy gave them ≳ −4) and base gives ≈ −1 → KL exp(ref−new) explodes.

This loads a checkpoint and scores the *exact* triggering tokens (from the run's own
completions) through every forward, with the MERGED model (LoRA folded into weights) as the
ground truth:
  HF-base       : adapters disabled (the KL `ref`)
  HF-DualLoRA   : adapters on (the loss `new`)               <- training path
  HF-merged     : W += scaling·B·A, adapters disabled        <- HF ground truth
  vLLM-merged   : vLLM serving the merged weights (no adapter)<- vLLM ground truth
  vLLM-LoRA     : vLLM serving the r32f0 adapter             <- generation path

Reads:
  HF-DualLoRA vs HF-merged  -> is HF's adapter application correct? (should be ~equal)
  vLLM-LoRA  vs vLLM-merged -> is vLLM's adapter serving correct?
  HF-merged  vs vLLM-merged -> base-engine gap on identical weights (control)

r32f0: single retain LoRA rank 32, forget rank 0 (None), scaling = alpha/rank = 1.0.

Step 1 (this `hf_repro`): HF base/DualLoRA/merged only — confirm the 20+ nat gap reproduces
IN ISOLATION before adding vLLM. No conclusions until it does.

  modal run tools/lora_serving_matrix.py::hf_repro --run-name logp_div_debug_16st_save5 --ckpt 15
"""

import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-loramatrix-jake")
MODEL = "Qwen/Qwen3-4B"
TEMP = 0.7


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=30 * 60)
def hf_repro(run_name: str, ckpt: int):
    import os, sys, json
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import disabled_dual_adapters, DualLoRALinear

    ckpt_dir = f"{OUTPUT_REMOTE}/logp_div_debug/{run_name}/checkpoint-{ckpt}"
    print(f"loading {ckpt_dir}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_gradient_routing_model(ckpt_dir, base_model=MODEL).cuda().eval().to(torch.bfloat16)
    duals = [m for m in model.modules() if isinstance(m, DualLoRALinear)]
    print(f"DualLoRALinear modules: {len(duals)}; retain ranks: {set(m.rank for m in duals)}; "
          f"forget ranks: {set(m.forget_rank for m in duals)}; scalings: {set(round(m.scaling,3) for m in duals)}")

    # --- pull the run's own completions (vLLM-generated) ---
    samp = f"{OUTPUT_REMOTE}/logp_div_debug/{run_name}/train_samples.jsonl"
    rows = [json.loads(l) for l in open(samp)]
    # take a handful of distinct completions near the gap
    seen, comps = set(), []
    for r in rows:
        key = r["completion"][:60]
        if key in seen: continue
        seen.add(key); comps.append(r)
        if len(comps) >= 12: break
    print(f"scoring {len(comps)} distinct completions")

    def logps(forward_ctx):
        """Return per-(comp,tok) logp dict using the model under the given context (None=adapters on)."""
        out = []
        for r in comps:
            pids = tok(r["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            cids = tok(r["completion"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            full = torch.cat([pids, cids], dim=1)
            Lp = pids.shape[1]
            with torch.no_grad():
                if forward_ctx is None:
                    logits = model(full).logits[0].float() / TEMP
                else:
                    with forward_ctx():
                        logits = model(full).logits[0].float() / TEMP
            row = [F.log_softmax(logits[Lp + i - 1], -1)[t].item() for i, t in enumerate(cids[0].tolist())]
            out.append((cids[0].tolist(), row))
        return out

    dual = logps(None)                                   # adapters on  -> the loss `new`
    base = logps(lambda: disabled_dual_adapters(model))  # adapters off -> the KL `ref`

    # merge B@A into base weights, then score with adapters disabled -> HF ground truth
    with torch.no_grad():
        for m in duals:
            if m.lora_A_retain is not None:
                delta = (m.lora_B_retain.float() @ m.lora_A_retain.float()) * m.scaling * m.retain_scale
                m.base_layer.weight.data += delta.to(m.base_layer.weight.dtype)
    merged = logps(lambda: disabled_dual_adapters(model))

    # --- report tokens where DualLoRA diverges from base by >5 (the KL gap) ---
    print(f"\n{'|base-dual|':>11} | base | dual | merged | dual-vs-merged | token / ctx")
    big = []
    for ci in range(len(comps)):
        ids, b = base[ci]; _, d = dual[ci]; _, mg = merged[ci]
        for i in range(len(ids)):
            if abs(d[i] - b[i]) > 5:
                big.append((abs(d[i]-b[i]), b[i], d[i], mg[i], d[i]-mg[i], ids[i], ids[max(0,i-6):i+1]))
    big.sort(reverse=True)
    for g, b, d, mg, dvm, tid, ctx in big[:20]:
        print(f"{g:>11.1f} | {b:>5.1f} | {d:>5.1f} | {mg:>6.1f} | {dvm:>+13.2f} | {tok.decode([tid])!r} | {tok.decode(ctx)!r}")
    print(f"\n({len(big)} tokens with |base-dual|>5)")
    if big:
        maxdvm = max(abs(x[4]) for x in big)
        print(f"max |HF-DualLoRA - HF-merged| over these tokens = {maxdvm:.3f}  "
              f"({'AGREE (HF adapter app correct)' if maxdvm < 1 else 'DISAGREE — HF adapter forward is suspect'})")
    return {"n_gap_tokens": len(big)}


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=40 * 60)
def vllm_arbiter(run_name: str, ckpt: int):
    """Score the triggering tokens through vLLM serving the r32f0 LoRA adapter, vs the
    HF-merged ground truth. Converts the DualLoRA retain adapter to PEFT format and uses
    vLLM native LoRA + prompt_logprobs (teacher-forcing)."""
    import os, sys, json
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import disabled_dual_adapters, DualLoRALinear
    from safetensors.torch import save_file

    ckpt_dir = f"{OUTPUT_REMOTE}/logp_div_debug/{run_name}/checkpoint-{ckpt}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_gradient_routing_model(ckpt_dir, base_model=MODEL).cuda().eval().to(torch.bfloat16)

    # --- map DualLoRALinear -> module path; export retain A/B as a PEFT adapter ---
    path_of = {}
    for name, mod in model.named_modules():
        if isinstance(mod, DualLoRALinear):
            path_of[mod] = name  # e.g. model.layers.0.self_attn.q_proj
    peft_sd = {}
    for mod, name in path_of.items():
        if mod.lora_A_retain is None: continue
        # PEFT scales by alpha/r = 1.0 here; fold module.scaling (=1.0) is a no-op
        peft_sd[f"base_model.model.{name}.lora_A.weight"] = mod.lora_A_retain.detach().cpu().to(torch.float32)
        peft_sd[f"base_model.model.{name}.lora_B.weight"] = mod.lora_B_retain.detach().cpu().to(torch.float32)
    adir = "/tmp/peft_adapter"; os.makedirs(adir, exist_ok=True)
    save_file(peft_sd, f"{adir}/adapter_model.safetensors")
    targets = sorted({n.split(".")[-1] for n in path_of.values()})
    json.dump({"peft_type":"LORA","r":32,"lora_alpha":32,"lora_dropout":0.0,"bias":"none",
               "target_modules":targets,"task_type":"CAUSAL_LM","base_model_name_or_path":MODEL,
               "fan_in_fan_out":False,"inference_mode":True,"use_rslora":False,"init_lora_weights":True},
              open(f"{adir}/adapter_config.json","w"))
    print(f"exported PEFT adapter: {len(peft_sd)//2} modules, targets={targets}")

    # --- HF-merged ground truth for the triggering tokens ---
    samp = f"{OUTPUT_REMOTE}/logp_div_debug/{run_name}/train_samples.jsonl"
    rows = [json.loads(l) for l in open(samp)]
    seen, comps = set(), []
    for r in rows:
        k = r["completion"][:60]
        if k in seen: continue
        seen.add(k); comps.append(r)
        if len(comps) >= 12: break

    def hf_logps(merged):
        res = []
        for r in comps:
            pids = tok(r["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            cids = tok(r["completion"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            full = torch.cat([pids, cids], dim=1); Lp = pids.shape[1]
            with torch.no_grad():
                with disabled_dual_adapters(model):
                    logits = model(full).logits[0].float()  # RAW (temp 1.0) to match vLLM prompt_logprobs
            res.append((cids[0].tolist(), [F.log_softmax(logits[Lp+i-1],-1)[t].item() for i,t in enumerate(cids[0].tolist())]))
        return res
    # merge B@A then score (adapters disabled) = HF-merged
    with torch.no_grad():
        for mod in path_of:
            if mod.lora_A_retain is not None:
                d = (mod.lora_B_retain.float() @ mod.lora_A_retain.float()) * mod.scaling * mod.retain_scale
                mod.base_layer.weight.data += d.to(mod.base_layer.weight.dtype)
    hf_merged = hf_logps(True)
    del model; torch.cuda.empty_cache()

    # --- vLLM serving the PEFT adapter; teacher-forcing via prompt_logprobs ---
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=32,
              gpu_memory_utilization=0.55, max_model_len=4096, enforce_eager=False)
    lreq = LoRARequest("r32f0", 1, adir)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    vllm_lora = []
    for r in comps:
        pids = tok(r["prompt"], add_special_tokens=False).input_ids
        cids = tok(r["completion"], add_special_tokens=False).input_ids
        full = pids + cids
        o = llm.generate(prompts=[{"prompt_token_ids": full}], sampling_params=sp, lora_request=lreq)[0]
        pl = o.prompt_logprobs  # list per position; [pos][token_id] -> Logprob
        # vLLM prompt_logprobs are raw (temp=1.0); divide-by-T to match HF's /T convention? No —
        # HF here uses logits/0.7. vLLM prompt_logprobs use the model's raw logits (no temp). To
        # compare apples-to-apples we want both at the SAME convention; record raw and report.
        row = []
        for i, t in enumerate(cids):
            pos = len(pids) + i
            entry = pl[pos].get(t) if pl[pos] else None
            row.append(entry.logprob if entry is not None else 0.0)
        vllm_lora.append((cids, row))

    # NOTE: vLLM prompt_logprobs are at temperature 1.0 (raw). HF logps above are /0.7. For the
    # collapse comparison (a token's logp ~0 vs ~-17) the temperature factor (<~2 nats on
    # confident tokens) is negligible vs a 16-nat gap, so we compare directly and flag it.
    print(f"\n{'token':>10} | {'HF-merged(raw)':>14} | {'vLLM-LoRA(raw)':>14} | {'gap':>7} | ctx")
    rep = []
    for ci in range(len(comps)):
        ids, hm = hf_merged[ci]; _, vl = vllm_lora[ci]
        for i in range(min(len(ids), len(vl))):
            if hm[i] < -5 or abs(hm[i]-vl[i]) > 5:  # collapsed-in-HF or big disagreement
                rep.append((abs(hm[i]-vl[i]), ids[i], hm[i], vl[i], ids[max(0,i-5):i+1]))
    rep.sort(reverse=True)
    for g, tid, hm, vl, ctx in rep[:20]:
        print(f"{tok.decode([tid])!r:>10} | {hm:>13.2f} | {vl:>14.2f} | {hm-vl:>+7.1f} | {tok.decode(ctx)!r}")
    return {"n": len(rep)}


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=40 * 60)
def vllm_merged_arbiter(run_name: str, ckpt: int):
    """vLLM-merged control cell: fold B@A into the weights, swap DualLoRALinear -> plain
    nn.Linear, save a clean Qwen checkpoint, and serve it through vLLM with NO adapter.
    Compares vLLM-merged vs HF-merged on the triggering tokens. This is the base-engine
    control (line 18 of the module docstring): a DIFFERENT vLLM code path than vLLM-LoRA
    (no punica), so vLLM-merged==HF-merged==vLLM-LoRA makes the agreement airtight."""
    import os, sys, json
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import disabled_dual_adapters, DualLoRALinear

    ckpt_dir = f"{OUTPUT_REMOTE}/logp_div_debug/{run_name}/checkpoint-{ckpt}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_gradient_routing_model(ckpt_dir, base_model=MODEL).cuda().eval().to(torch.bfloat16)

    samp = f"{OUTPUT_REMOTE}/logp_div_debug/{run_name}/train_samples.jsonl"
    rows = [json.loads(l) for l in open(samp)]
    seen, comps = set(), []
    for r in rows:
        k = r["completion"][:60]
        if k in seen: continue
        seen.add(k); comps.append(r)
        if len(comps) >= 12: break

    # fold B@A into base_layer.weight (same delta as the HF-merged ground truth)
    duals = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, DualLoRALinear)]
    with torch.no_grad():
        for _, mod in duals:
            if mod.lora_A_retain is not None:
                d = (mod.lora_B_retain.float() @ mod.lora_A_retain.float()) * mod.scaling * mod.retain_scale
                mod.base_layer.weight.data += d.to(mod.base_layer.weight.dtype)

    # HF-merged ground truth (RAW logits, temp 1.0, to match vLLM prompt_logprobs)
    def hf_logps():
        res = []
        for r in comps:
            pids = tok(r["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            cids = tok(r["completion"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            full = torch.cat([pids, cids], dim=1); Lp = pids.shape[1]
            with torch.no_grad(), disabled_dual_adapters(model):
                logits = model(full).logits[0].float()
            res.append((cids[0].tolist(),
                        [F.log_softmax(logits[Lp+i-1], -1)[t].item() for i, t in enumerate(cids[0].tolist())]))
        return res
    hf_merged = hf_logps()

    # swap DualLoRALinear -> its (now-merged) base_layer to get a plain Qwen; save + reload in vLLM
    for name, mod in duals:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[-1], mod.base_layer)
    mdir = "/tmp/merged_qwen"; os.makedirs(mdir, exist_ok=True)
    model.save_pretrained(mdir); tok.save_pretrained(mdir)
    del model; torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=0.55,
              max_model_len=4096, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    vmerged = []
    for r in comps:
        pids = tok(r["prompt"], add_special_tokens=False).input_ids
        cids = tok(r["completion"], add_special_tokens=False).input_ids
        full = pids + cids
        o = llm.generate(prompts=[{"prompt_token_ids": full}], sampling_params=sp)[0]
        pl = o.prompt_logprobs
        row = []
        for i, t in enumerate(cids):
            pos = len(pids) + i
            entry = pl[pos].get(t) if pl[pos] else None
            row.append(entry.logprob if entry is not None else 0.0)
        vmerged.append((cids, row))

    print(f"\n{'token':>10} | {'HF-merged(raw)':>14} | {'vLLM-merged(raw)':>16} | {'gap':>7} | ctx")
    rep = []
    for ci in range(len(comps)):
        ids, hm = hf_merged[ci]; _, vm = vmerged[ci]
        for i in range(min(len(ids), len(vm))):
            if hm[i] < -5 or abs(hm[i] - vm[i]) > 5:
                rep.append((abs(hm[i] - vm[i]), ids[i], hm[i], vm[i], ids[max(0, i-5):i+1]))
    rep.sort(reverse=True)
    for g, tid, hm, vm, ctx in rep[:20]:
        print(f"{tok.decode([tid])!r:>10} | {hm:>14.2f} | {vm:>16.2f} | {hm-vm:>+7.1f} | {tok.decode(ctx)!r}")
    return {"n": len(rep)}


@app.local_entrypoint()
def hf(run_name: str, ckpt: int):
    print(hf_repro.remote(run_name, ckpt))


@app.local_entrypoint()
def vllm(run_name: str, ckpt: int):
    print(vllm_arbiter.remote(run_name, ckpt))


@app.local_entrypoint()
def vmerged(run_name: str, ckpt: int):
    print(vllm_merged_arbiter.remote(run_name, ckpt))
