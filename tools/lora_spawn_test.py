"""Faithful repro of the REAL training serving path: spawn _spawn_vllm_server (the actual
subprocess + ZMQ), drive it with VLLMLoRAClient, and check whether a LARGE adapter (ckpt-200)
actually changes generation. The 5-step debug couldn't tell (B~0.03 too small); the in-process
swap/repro/manysync all WORK. This is the one untested combination: spawned subprocess + ZMQ +
large B. If gen-after-ckpt200-sync == gen-before-any-sync (base), the spawned server serves base.

  modal run tools/lora_spawn_test.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-loraspawn-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=30 * 60)
def spawn_test():
    import os, sys, multiprocessing as mp, tempfile
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoTokenizer
    from train import _spawn_vllm_server
    from vllm_lifecycle import wait_for_ready_file
    from vllm_lora import VLLMLoRAClient
    from eval_utils import load_gradient_routing_model

    socket_path = f"ipc:///tmp/vllm_test_{os.getpid()}.sock"
    ready_file = tempfile.mktemp(prefix="vllm_ready_")
    ctx = mp.get_context("spawn")
    # args: (model, mlp_config, gpu_memory, socket_path, ready_file, layer_start, layer_end,
    #        layer_stride, max_experiments, gpu_id, label, num_gpu_blocks, adapter_type)
    proc = ctx.Process(target=_spawn_vllm_server,
                       args=(MODEL, None, 0.55, socket_path, ready_file, 0.0, 1.0, 1, 4, 0,
                             "spawntest", None, "lora"))
    proc.start()
    print(f"[test] spawned server pid={proc.pid}; waiting for ready...")
    wait_for_ready_file(ready_file, proc, "spawntest")
    client = VLLMLoRAClient(socket_path)
    eid = client.register()
    print(f"[test] connected, eid={eid}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Write a Python function fib(n) that returns the nth Fibonacci number."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(prompt, add_special_tokens=False).input_ids

    def gen():
        texts, ids, _ = client.generate(eid, [pids], 1, 0.0, 200, top_k=0, top_p=1.0)
        return texts[0]

    out_before = gen()                                   # _lora_request is None -> base
    print(f"[test] gen BEFORE any sync (base):\n{out_before[:300]!r}\n")

    # sync the large ckpt-200 adapter (B-norm ~7.4) via the real client path
    m = load_gradient_routing_model(f"{OUTPUT_REMOTE}/{RUN}/checkpoint-200", base_model=MODEL).to(torch.bfloat16)
    client.update_weights_from_model(eid, m)
    del m
    out_after = gen()                                    # served adapter = ckpt-200
    print(f"[test] gen AFTER ckpt-200 sync:\n{out_after[:300]!r}\n")

    same = out_before == out_after
    print(f"[test] outputs identical = {same}")
    print(f"VERDICT: {'SPAWNED SERVER SERVES BASE (adapter ignored!) -> THE BUG' if same else 'spawned server APPLIES the adapter (outputs differ)'}")
    proc.terminate()
    return {"identical": same}


@app.local_entrypoint()
def main():
    print(spawn_test.remote())
