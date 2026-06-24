"""Live-shape LoRA rollout diagnostic.

The existing LoRA serving tests cover static adapters, repeated syncs, and the
real spawned subprocess path, but mostly with one prompt and/or teacher-forced
logprobs. This probes the remaining gap: a spawned VLLMLoRAServer serving a
large batch with the same stochastic sampling knobs used by the broken run.

Usage:
  modal run tools/lora_batch_spawn_test.py --n 256 --max-tokens 256
  modal run tools/lora_batch_spawn_test.py --n 256 --max-tokens 256 --return-logprobs
"""

import modal

from tools.modal_train_gr import OUTPUT_REMOTE, REPO_REMOTE, image, secrets, vol


app = modal.App("gr-lora-batchspawn-jake")

MODEL = "Qwen/Qwen3-4B"
RUN = (
    "leetcode_noint_4b_verlparity/"
    "leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"
)


def _hash_repetition_score(text: str) -> float:
    if not text:
        return 0.0
    toks = text.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t == "#") / len(toks)


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=45 * 60,
)
def batch_spawn_test(
    n: int = 256,
    max_tokens: int = 256,
    return_logprobs: bool = False,
):
    import json
    import multiprocessing as mp
    import os
    import sys
    import tempfile

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)

    import torch
    from eval_utils import load_gradient_routing_model
    from train import _spawn_vllm_server
    from transformers import AutoTokenizer
    from vllm_lifecycle import wait_for_ready_file
    from vllm_lora import VLLMLoRAClient

    tok = AutoTokenizer.from_pretrained(MODEL)

    sample_path = f"{OUTPUT_REMOTE}/{RUN}/train_samples.jsonl"
    prompts = []
    seen = set()
    with open(sample_path) as f:
        for line in f:
            prompt = json.loads(line)["prompt"]
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
            if len(prompts) >= n:
                break
    if not prompts:
        raise RuntimeError(f"No prompts found in {sample_path}")

    prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]

    socket_path = f"ipc:///tmp/vllm_batch_{os.getpid()}.sock"
    ready_file = tempfile.mktemp(prefix="vllm_ready_")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_spawn_vllm_server,
        args=(
            MODEL,
            None,
            0.55,
            socket_path,
            ready_file,
            0.0,
            1.0,
            1,
            4,
            0,
            "batchspawntest",
            None,
            "lora",
        ),
    )
    proc.start()
    print(f"[batch] spawned server pid={proc.pid}; waiting for ready...")
    try:
        wait_for_ready_file(ready_file, proc, "batchspawntest")
        client = VLLMLoRAClient(socket_path)
        eid = client.register()

        def gen(ids_batch):
            result = client.generate(
                eid,
                ids_batch,
                1,
                0.7,
                max_tokens,
                top_k=-1,
                top_p=0.95,
                return_logprobs=return_logprobs,
            )
            texts, ids, _ = result[:3]
            logprobs = result[3] if len(result) > 3 else None
            if return_logprobs:
                assert logprobs is not None
                assert len(logprobs) == len(texts)
                assert all(len(lp) == len(cid) for lp, cid in zip(logprobs, ids))
            return texts, ids

        base_one, _ = gen(prompt_ids[:1])

        ckpt = f"{OUTPUT_REMOTE}/{RUN}/checkpoint-200"
        model = load_gradient_routing_model(ckpt, base_model=MODEL)
        model = model.to(torch.bfloat16)
        client.update_weights_from_model(eid, model)
        del model
        torch.cuda.empty_cache()

        adapter_one, _ = gen(prompt_ids[:1])
        adapter_batch, adapter_ids = gen(prompt_ids)

        scores = [_hash_repetition_score(t) for t in adapter_batch]
        hashish = [t for t, s in zip(adapter_batch, scores) if s >= 0.5]
        avg_tokens = sum(len(x) for x in adapter_ids) / max(1, len(adapter_ids))

        print("\n[one-prompt base]")
        print(repr(base_one[0][:500]))
        print("\n[one-prompt adapter]")
        print(repr(adapter_one[0][:500]))
        print("\n[batch adapter first 5]")
        for i, text in enumerate(adapter_batch[:5]):
            print(f"#{i} score={scores[i]:.2f} {repr(text[:300])}")
        print(
            f"\n[summary] n={len(prompt_ids)} max_tokens={max_tokens} "
            f"return_logprobs={return_logprobs} "
            f"avg_completion_tokens={avg_tokens:.1f} "
            f"hash_repetition>=0.5: {len(hashish)}/{len(prompt_ids)}"
        )

        return {
            "n": len(prompt_ids),
            "max_tokens": max_tokens,
            "return_logprobs": return_logprobs,
            "avg_completion_tokens": avg_tokens,
            "hash_repetition_ge_0_5": len(hashish),
            "base_one_prefix": base_one[0][:200],
            "adapter_one_prefix": adapter_one[0][:200],
            "adapter_batch_prefixes": adapter_batch[:5],
        }
    finally:
        proc.terminate()
        proc.join(timeout=30)


@app.local_entrypoint()
def main(n: int = 256, max_tokens: int = 256, return_logprobs: bool = False):
    print(batch_spawn_test.remote(n, max_tokens, return_logprobs))
