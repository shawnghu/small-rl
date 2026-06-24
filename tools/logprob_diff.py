"""Diagnostic: measure per-token vLLM-vs-HF logprob disagreement on leetcode completions.

WHY: our fast-IS GRPO path reuses vLLM's rollout logprobs as `old_per_token_logps` and the
HF forward as `new`; the ratio exp(new-old) explodes (grad_norm up to 1e13, ~5% of steps)
because some tokens disagree by 15-25 nats. Alignment/temperature/top_p are ruled out by code
reading, so this measures the ACTUAL disagreement to characterize WHICH tokens blow up and why.

Generates a few leetcode completions through vLLM (capturing logprobs exactly like
vllm_lora.py), then recomputes the same tokens through an HF forward with the SAME
temperature-scaled convention the loss uses (logits.float()/T -> log_softmax -> gather), and
prints the most divergent tokens with context. Base model first (isolates vLLM-vs-HF from LoRA).

  modal run tools/logprob_diff.py::run
"""

import modal
from tools.modal_train_gr import image, secrets, vol, RH_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-logprobdiff-jake")

MODEL = "Qwen/Qwen3-4B"
TEMP = 0.7
TOP_P = 0.95
MAX_TOKENS = 1024
N_PROMPTS = 6


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=30 * 60)
def run():
    import os, json
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Build a few real leetcode prompts (simple_overwrite_tests hinted train file).
    data_path = f"{RH_REMOTE}/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
    rows = [json.loads(l) for l in open(data_path)][:N_PROMPTS]
    prompts = []
    for r in rows:
        msgs = r["prompt"] if isinstance(r.get("prompt"), list) else r["prompt"]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        prompts.append(s)

    # --- vLLM generate with logprobs (mirrors vllm_lora.py: logprobs=0, capture sampled-token lp) ---
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.55,
              max_model_len=4096, enforce_eager=False)
    sp = SamplingParams(temperature=TEMP, top_p=TOP_P, top_k=-1,
                        max_tokens=MAX_TOKENS, logprobs=0)
    outs = llm.generate(prompts, sp)

    samples = []  # (prompt_text, completion_ids, vllm_logps)
    for o in outs:
        comp = o.outputs[0]
        cids = list(comp.token_ids)
        vlp = []
        for i, lp_dict in enumerate(comp.logprobs):
            tid = comp.token_ids[i]
            entry = lp_dict.get(tid)
            vlp.append(entry.logprob if entry is not None else 0.0)
        samples.append((o.prompt, cids, vlp))
    del llm
    torch.cuda.empty_cache()

    # --- HF recompute (same temperature-scaled convention as the loss: logits.float()/T) ---
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

    big = []  # all (diff, token_str, vllm_lp, hf_lp, pos, ctx)
    per_seq = []
    for (ptext, cids, vlp) in samples:
        pids = tok(ptext, add_special_tokens=False)["input_ids"]
        full = torch.tensor(pids + cids, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = hf(full).logits[0].float()  # (L, V)
        Lp = len(pids)
        hflp = []
        for i, ctok in enumerate(cids):
            pos = Lp + i - 1  # logits at pos predict token at pos+1 = completion token i
            lp = torch.log_softmax(logits[pos] / TEMP, dim=-1)[ctok].item()
            hflp.append(lp)
        # collect divergences
        seqmax = 0.0
        for i, ctok in enumerate(cids):
            d = hflp[i] - vlp[i]  # new - old (the ratio exponent)
            if abs(d) > abs(seqmax): seqmax = d
            if abs(d) > 5:
                ctx = tok.decode(cids[max(0, i - 6):i + 1])
                big.append((d, repr(tok.decode([ctok])), round(vlp[i], 2), round(hflp[i], 2), i, repr(ctx)))
        per_seq.append((len(cids), seqmax))

    big.sort(key=lambda x: -abs(x[0]))
    print("\n==== PER-SEQUENCE (len, max |new-old|) ====")
    for L, m in per_seq:
        print(f"  len={L:>4}  max(new-old)={m:+.2f}")
    n_gt5 = len(big)
    n_gt10 = sum(1 for x in big if abs(x[0]) > 10)
    n_gt20 = sum(1 for x in big if abs(x[0]) > 20)
    tot = sum(L for L, _ in per_seq)
    print(f"\n==== DIVERGENCE STATS over {tot} tokens ====")
    print(f"  |new-old|>5: {n_gt5}   >10: {n_gt10}   >20: {n_gt20}")
    print(f"\n==== TOP 30 DIVERGENT TOKENS (new-old, token, vllm_lp, hf_lp, pos, context) ====")
    for d, tkn, vl, hl, pos, ctx in big[:30]:
        print(f"  diff={d:+7.2f}  tok={tkn:<14} vllm={vl:>8}  hf={hl:>8}  pos={pos:>4}  ctx={ctx}")
    return {"n_gt5": n_gt5, "n_gt10": n_gt10, "n_gt20": n_gt20, "tot": tot}


@app.local_entrypoint()
def main():
    print(run.remote())
