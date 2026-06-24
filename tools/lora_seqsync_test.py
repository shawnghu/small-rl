"""Rule (b) in/out: does syncing a SEQUENCE of DIFFERENT growing adapters (like the real run) break
serving, where syncing the SAME one 200x didn't? Cycle ckpt-50/100/150/200 (new int_id each) ~40x
(160 syncs), then verify the final ckpt-200 sync still applies (vs base).
  modal run tools/lora_seqsync_test.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE
app = modal.App("gr-seqsync-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"

@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=30*60)
def seqsync():
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch, numpy as np
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from vllm_lora import create_lora_engine, _extract_dual_lora_tensors, _sync_lora_to_engine
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(MODEL)
    adapters = []
    for ck in (50, 100, 150, 200):
        m = load_gradient_routing_model(f"{OUTPUT_REMOTE}/{RUN}/checkpoint-{ck}", base_model=MODEL).cuda().eval().to(torch.bfloat16)
        t, r, tg = _extract_dual_lora_tensors(m); adapters.append((t, r, tg)); del m; torch.cuda.empty_cache()
    prompt = tok.apply_chat_template([{"role":"user","content":"Write a Python function is_even(n)."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    comp = "def is_even(n):\n    return n % 2 == 0\n"
    pids = tok(prompt, add_special_tokens=False).input_ids; cids = tok(comp, add_special_tokens=False).input_ids
    full = pids + cids; sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    llm = create_lora_engine(MODEL, max_lora_rank=64, gpu_memory_utilization=0.5)
    def score(req):
        o = llm.generate([{"prompt_token_ids": full}], sp, lora_request=req)[0]; pl = o.prompt_logprobs
        return [(pl[len(pids)+i].get(t).logprob if pl[len(pids)+i] and pl[len(pids)+i].get(t) else 0.0) for i,t in enumerate(cids)]
    lp_base = score(None)
    req = None
    for cyc in range(40):
        for (t, r, tg) in adapters:
            req = _sync_lora_to_engine(llm, t, r, tg)
    # final req is ckpt-200's
    lp = score(req); d = float(np.abs(np.array(lp)-np.array(lp_base)).max())
    print(f"after 160 sequential growing-adapter syncs, final ckpt-200: max|adapter-base|={d:.3f} -> {'APPLIED (b ruled out)' if d>0.5 else 'SERVING BASE (b CONFIRMED)'}")
    return {"d": d}

@app.local_entrypoint()
def main(): print(seqsync.remote())
