"""Does the training-time vLLM LoRA weight sync actually SWAP weights per step?

Hypothesis: rollouts during our LoRA runs are served by a STALE adapter (the per-step
add_lora swap doesn't take effect on a running engine), so on-policy training is silently
off-policy -> 'drifts a lot, learns nothing'. New code (2026-06-15), untouched by the audit.

Test (repo's ACTUAL path: vllm_lora.create_lora_engine + _sync_lora_to_engine + TensorLoRARequest):
  1. extract adapters from ckpt-50 (near base) and ckpt-200 (drifted) of a verlparity run
  2. create ONE engine; teacher-force a fixed token sequence with prompt_logprobs:
       - base (no lora)         -> lp_base
       - sync ckpt-50,  score   -> lp_50
       - sync ckpt-200, score   -> lp_200   (SAME engine, the per-step swap)
  3. If lp_50 ~= lp_200 (and != base): the swap is BROKEN -> served weights frozen at first sync.
     If lp_50 != lp_200 != base: swap works -> hypothesis refuted, look elsewhere.
Also does the reverse order (200 then 50) to rule out 'only the first sync ever applies'.

  modal run tools/lora_sync_test.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-lorasync-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=25 * 60)
def swap_test():
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from vllm_lora import create_lora_engine, _extract_dual_lora_tensors, _sync_lora_to_engine
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)

    def extract(ckpt):
        d = f"{OUTPUT_REMOTE}/{RUN}/checkpoint-{ckpt}"
        m = load_gradient_routing_model(d, base_model=MODEL).cuda().eval().to(torch.bfloat16)
        tensors, rank, targets = _extract_dual_lora_tensors(m)
        # snapshot adapter L2 norm so we can confirm the two adapters really differ
        nrm = sum(float(t.float().norm()**2) for t in tensors.values()) ** 0.5
        del m; torch.cuda.empty_cache()
        return tensors, rank, targets, nrm

    t50, rank, targets, n50 = extract(50)
    t200, _, _, n200 = extract(200)
    print(f"adapter L2 norms: ckpt50={n50:.2f}  ckpt200={n200:.2f}  (differ by {abs(n200-n50):.2f})")

    # fixed teacher-forcing sequence (a chat-formatted coding prompt + short completion)
    msgs = [{"role": "user", "content": "Write a Python function is_even(n) that returns True if n is even."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    comp = "def is_even(n):\n    return n % 2 == 0\n"
    pids = tok(prompt, add_special_tokens=False).input_ids
    cids = tok(comp, add_special_tokens=False).input_ids
    full = pids + cids
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    llm = create_lora_engine(MODEL, max_lora_rank=32, gpu_memory_utilization=0.55)

    def score(lora_req, label):
        o = llm.generate([{"prompt_token_ids": full}], sp, lora_request=lora_req)[0]
        pl = o.prompt_logprobs
        lps = []
        for i, t in enumerate(cids):
            pos = len(pids) + i
            e = pl[pos].get(t) if pl[pos] else None
            lps.append(e.logprob if e is not None else 0.0)
        print(f"  [{label}] sum_logp={sum(lps):.3f} first5={[round(x,2) for x in lps[:5]]}")
        return lps

    print("\n=== order A: base -> ckpt50 -> ckpt200 (same engine) ===")
    lp_base = score(None, "base")
    r50 = _sync_lora_to_engine(llm, t50, rank, targets)
    lp50 = score(r50, "ckpt50")
    r200 = _sync_lora_to_engine(llm, t200, rank, targets)
    lp200 = score(r200, "ckpt200")

    import numpy as np
    def d(a, b): return float(np.abs(np.array(a) - np.array(b)).max())
    print(f"\nmax|lp50 - lp200|   = {d(lp50,lp200):.4f}   (SWAP works if >> 0)")
    print(f"max|lp50 - lp_base| = {d(lp50,lp_base):.4f}")
    print(f"max|lp200- lp_base| = {d(lp200,lp_base):.4f}")

    print("\n=== order B (fresh sync sequence on same engine): ckpt200 -> ckpt50 ===")
    r200b = _sync_lora_to_engine(llm, t200, rank, targets)
    lp200b = score(r200b, "ckpt200(2nd)")
    r50b = _sync_lora_to_engine(llm, t50, rank, targets)
    lp50b = score(r50b, "ckpt50(2nd)")
    print(f"max|lp200b - lp50b| = {d(lp200b,lp50b):.4f}")
    print(f"max|lp50 - lp50b|   = {d(lp50,lp50b):.4f}   (re-sync ckpt50 reproducible if ~0)")
    print(f"max|lp200 - lp200b| = {d(lp200,lp200b):.4f}")

    verdict = "SWAP WORKS" if d(lp50, lp200) > 0.5 else "SWAP BROKEN (served weights frozen at first sync)"
    print(f"\nVERDICT: {verdict}")
    return {"swap_works": d(lp50, lp200) > 0.5, "d_50_200": d(lp50, lp200),
            "d_50_base": d(lp50, lp_base), "n50": n50, "n200": n200}


@app.local_entrypoint()
def main():
    print(swap_test.remote())
