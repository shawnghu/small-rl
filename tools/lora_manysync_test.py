"""Does the vLLM LoRA server keep applying the adapter across MANY syncs (200-step run does ~200
add_lora calls, each a new lora_int_id)? The swap/repro only did ~6 and worked. If after K syncs
the latest adapter stops being served (-> base), that's the long-run serving bug that makes the
real rollouts come from base.

Test: create the engine ONCE (max_lora_rank=64, like the real spawn). Repeatedly _sync_lora_to_engine
the ckpt-200 adapter (a NEW lora_int_id each time, exactly like training). At cumulative-sync
counts {1,5,20,50,100,150,200}, teacher-force a fixed sequence with the latest lora_request and
measure max|adapter-base|. If it drops to ~0 at some K, the cache churn breaks serving.

  modal run tools/lora_manysync_test.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-manysync-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=30 * 60)
def manysync():
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
    tensors, rank, targets = _extract_dual_lora_tensors(m)
    del m; torch.cuda.empty_cache()

    prompt = tok.apply_chat_template([{"role": "user", "content": "Write a Python function is_even(n)."}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    comp = "def is_even(n):\n    return n % 2 == 0\n"
    pids = tok(prompt, add_special_tokens=False).input_ids
    cids = tok(comp, add_special_tokens=False).input_ids
    full = pids + cids
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    llm = create_lora_engine(MODEL, max_lora_rank=64, gpu_memory_utilization=0.5)

    def score(lora_req):
        o = llm.generate([{"prompt_token_ids": full}], sp, lora_request=lora_req)[0]
        pl = o.prompt_logprobs
        return [(pl[len(pids)+i].get(t).logprob if pl[len(pids)+i] and pl[len(pids)+i].get(t) else 0.0)
                for i, t in enumerate(cids)]

    lp_base = score(None)
    checkpoints = {1, 5, 20, 50, 100, 150, 200}
    last_req = None
    print(f"{'n_syncs':>8} | {'max|adapter-base|':>18} | verdict")
    for k in range(1, 201):
        last_req = _sync_lora_to_engine(llm, tensors, rank, targets)
        if k in checkpoints:
            lp = score(last_req)
            d = float(np.abs(np.array(lp) - np.array(lp_base)).max())
            print(f"{k:>8} | {d:>18.4f} | {'APPLIED' if d > 0.5 else 'SERVING BASE (churn bug!)'}", flush=True)
    return {"done": True}


@app.local_entrypoint()
def main():
    print(manysync.remote())
