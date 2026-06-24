"""Does syncing a near-ZERO (init) adapter FIRST (as the live run does at step 0) pin the server to
base so subsequent real syncs don't take? All prior isolation tests started from a non-zero ckpt."""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE
app = modal.App("gr-zerofirst-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"

@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=25*60)
def zerofirst():
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch, numpy as np
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from vllm_lora import create_lora_engine, _extract_dual_lora_tensors, _sync_lora_to_engine
    from vllm import SamplingParams
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = load_gradient_routing_model(f"{OUTPUT_REMOTE}/{RUN}/checkpoint-200", base_model=MODEL).cuda().eval().to(torch.bfloat16)
    tensors, rank, targets = _extract_dual_lora_tensors(m); del m; torch.cuda.empty_cache()
    # zero-B variant (= base): the init/step-0 adapter has B=0
    ztensors = {k: (torch.zeros_like(v) if k.endswith("lora_B.weight") else v) for k, v in tensors.items()}
    prompt = tok.apply_chat_template([{"role":"user","content":"Write a Python function is_even(n)."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    comp = "def is_even(n):\n    return n % 2 == 0\n"
    pids = tok(prompt, add_special_tokens=False).input_ids; cids = tok(comp, add_special_tokens=False).input_ids
    full = pids + cids; sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    llm = create_lora_engine(MODEL, max_lora_rank=64, gpu_memory_utilization=0.5)
    def score(req):
        o = llm.generate([{"prompt_token_ids": full}], sp, lora_request=req)[0]; pl=o.prompt_logprobs
        return [(pl[len(pids)+i].get(t).logprob if pl[len(pids)+i] and pl[len(pids)+i].get(t) else 0.0) for i,t in enumerate(cids)]
    lp_base = score(None)
    # SYNC ZERO-B FIRST (like live step 0), then real ckpt-200 (like later steps)
    rz = _sync_lora_to_engine(llm, ztensors, rank, targets); lp_z = score(rz)
    rr = _sync_lora_to_engine(llm, tensors, rank, targets); lp_r = score(rr)
    d_zero = float(np.abs(np.array(lp_z)-np.array(lp_base)).max())
    d_real = float(np.abs(np.array(lp_r)-np.array(lp_base)).max())
    print(f"zero-B sync vs base: max|d|={d_zero:.3f} (should be ~0, zero-B == base)")
    print(f"real ckpt-200 AFTER zero-first: max|d vs base|={d_real:.3f} -> {'APPLIED (zero-first NOT the trigger)' if d_real>0.5 else 'STILL BASE (zero-first IS the trigger!)'}")
    return {"d_real": d_real}

@app.local_entrypoint()
def main(): print(zerofirst.remote())
