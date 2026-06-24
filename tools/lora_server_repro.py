"""Reproduce the EXACT training LoRA-serving path and find why rollouts come from base.

The real server (train.py:127) spawns VLLMLoRAServer WITHOUT max_lora_rank -> default 64, while
the r32f0 adapter's combined rank is 32. The swap test (which WORKED) used max_lora_rank=32.
Hypothesis: rank-32 adapter in a rank-64 engine silently mis-loads -> served weights ~ base.

Test: build VLLMLoRAServer at max_lora_rank in {64 (real), 32 (swap-test)}, push the ckpt-200
adapter via the real client serialization + handle_update_weights, teacher-force a fixed sequence
through server._lora_request, compare to base. If rank64 -> base but rank32 -> adapter: BUG = the
missing max_lora_rank in the spawn.

  modal run tools/lora_server_repro.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-lorarepro-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=30 * 60)
def repro():
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch, numpy as np
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from vllm_lora import create_lora_engine, _extract_dual_lora_tensors, _sync_lora_to_engine
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)

    # extract ckpt-200 adapter + serialize exactly like VLLMLoRAClient.update_weights_from_model
    m = load_gradient_routing_model(f"{OUTPUT_REMOTE}/{RUN}/checkpoint-200", base_model=MODEL).cuda().eval().to(torch.bfloat16)
    tensors, rank, targets = _extract_dual_lora_tensors(m)
    del m; torch.cuda.empty_cache()
    print(f"adapter combined_rank={rank}, {len(tensors)} tensors, targets={targets}")
    tensor_data = {k: t.numpy().tobytes() for k, t in tensors.items()}
    shapes = {k: list(t.shape) for k, t in tensors.items()}
    msg = {"peft_config": {"r": rank, "lora_alpha": rank, "target_modules": targets},
           "tensors": tensor_data, "shapes": shapes, "dtype": "float32"}

    # fixed teacher-forcing sequence
    prompt = tok.apply_chat_template([{"role": "user", "content": "Write a Python function is_even(n)."}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    comp = "def is_even(n):\n    return n % 2 == 0\n"
    pids = tok(prompt, add_special_tokens=False).input_ids
    cids = tok(comp, add_special_tokens=False).input_ids
    full = pids + cids
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def score(llm, lora_req):
        o = llm.generate([{"prompt_token_ids": full}], sp, lora_request=lora_req)[0]
        pl = o.prompt_logprobs
        return [(pl[len(pids)+i].get(t).logprob if pl[len(pids)+i] and pl[len(pids)+i].get(t) else 0.0)
                for i, t in enumerate(cids)]

    results = {}
    for mlr in (64, 32):
        print(f"\n===== max_lora_rank={mlr} (real spawn uses 64) =====")
        # Build the engine the way VLLMLoRAServer does, then drive handle_update_weights logic.
        llm = create_lora_engine(MODEL, max_lora_rank=mlr, gpu_memory_utilization=0.45)
        # replicate VLLMLoRAServer.handle_update_weights (deserialize -> _sync_lora_to_engine)
        np_dtype = np.float32
        rt = {k: torch.from_numpy(np.frombuffer(raw, dtype=np_dtype).reshape(tuple(shapes[k])).copy())
              for k, raw in tensor_data.items()}
        lora_req = _sync_lora_to_engine(llm, rt, msg["peft_config"]["r"], msg["peft_config"]["target_modules"])
        lp_base = score(llm, None)
        lp_ad = score(llm, lora_req)
        d = float(np.abs(np.array(lp_base) - np.array(lp_ad)).max())
        print(f"  base sum={sum(lp_base):.2f}  adapter sum={sum(lp_ad):.2f}  max|base-adapter|={d:.3f}")
        print(f"  -> {'ADAPTER APPLIED' if d > 0.5 else 'ADAPTER == BASE (mis-load!)'}")
        results[mlr] = d
        del llm
        import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"\nSUMMARY: max|base-adapter| at rank64={results.get(64):.3f}  rank32={results.get(32):.3f}")
    print("If rank64 ~0 and rank32 >0.5: BUG = spawn missing max_lora_rank (rank-32 adapter mis-loaded in rank-64 engine).")
    return results


@app.local_entrypoint()
def main():
    print(repro.remote())
