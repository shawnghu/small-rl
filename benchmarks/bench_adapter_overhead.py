"""Benchmark vLLM MLP adapter overhead vs baseline.

Tests:
1. Baseline vLLM (no LoRA, no adapters)
2. LoRA infra (attention-only) + einsum adapter
3. LoRA-free adapter (custom model runner hook routing)

Each test runs in its own spawned subprocess to avoid CUDA state contamination.

Note on test 2: uses attention-only LoRA target_modules (q/k/v/o_proj) to avoid
conflicting with the MLP surgery. MLP projections are NOT LoRA-targeted, so the
PunicaWrapper overhead is attention-layer-only — a lower bound on the original
full-module LoRA overhead.

Usage:
  CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 PYTHONPATH=/workspace/small-rl .venv-vllm/bin/python benchmarks/bench_adapter_overhead.py
"""

import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_PROMPTS = 32
N_GENERATIONS = 16
MAX_TOKENS = 64
PROMPT_LEN = 32
WARMUP = 2
BENCH_ITERS = 5
MAX_EXPERIMENTS = 20
RETAIN_NEURONS = 32
FORGET_NEURONS = 32
GPU_MEM_BASELINE = 0.05
GPU_MEM_LORA     = 0.10   # LoRA weight stacks need more headroom


def _run_in_subprocess(fn_name, result_queue):
    import gc
    import random

    import torch

    def make_prompts(n, prompt_len, seed=42):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        rng = random.Random(seed)
        vocab_size = tokenizer.vocab_size
        return [[rng.randint(3, vocab_size - 1) for _ in range(prompt_len)] for _ in range(n)]

    def clear_gpu():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def bench_time(fn, label, warmup=WARMUP, iters=BENCH_ITERS):
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times.sort()
        median = times[len(times) // 2]
        print(f"  {label:<55} | median={median:.3f}s  all={[f'{t:.3f}' for t in times]}", flush=True)
        return median

    try:
        if fn_name == "test1_baseline":
            from vllm import LLM, SamplingParams, TokensPrompt
            print("\n=== Test 1: Baseline (no LoRA, no adapters) ===", flush=True)
            llm = LLM(model=MODEL, enforce_eager=True, dtype="bfloat16",
                      gpu_memory_utilization=GPU_MEM_BASELINE)
            prompts = make_prompts(N_PROMPTS, PROMPT_LEN)
            token_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]
            sp = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, n=N_GENERATIONS)
            t = bench_time(lambda: llm.generate(token_prompts, sp), "No LoRA, no adapters")
            del llm; clear_gpu()
            result_queue.put(t)

        elif fn_name == "test2_lora_einsum":
            import json
            import tempfile
            from vllm import LLM, SamplingParams, TokensPrompt
            from vllm.lora.request import LoRARequest
            from safetensors.torch import save_file
            from transformers import AutoConfig
            from vllm_mlp_adapter_lora_free import inject_mlp_adapters

            print("\n=== Test 2: LoRA infra (q_proj only) + einsum adapter ===", flush=True)

            cfg = AutoConfig.from_pretrained(MODEL)
            h = cfg.hidden_size
            head_dim = h // cfg.num_attention_heads
            kv_dim = cfg.num_key_value_heads * head_dim
            n_layers = cfg.num_hidden_layers
            rank = 1

            # Match original vllm_mlp_adapter.py: target q_proj only
            with tempfile.TemporaryDirectory() as tmpdir:
                lora_dir = os.path.join(tmpdir, "lora_0")
                os.makedirs(lora_dir)

                with open(os.path.join(lora_dir, "adapter_config.json"), "w") as f:
                    json.dump({
                        "base_model_name_or_path": MODEL,
                        "peft_type": "LORA",
                        "r": rank,
                        "lora_alpha": rank,
                        "target_modules": ["q_proj"],
                        "bias": "none",
                        "task_type": "CAUSAL_LM",
                    }, f)

                tensors = {}
                for li in range(n_layers):
                    k = f"base_model.model.model.layers.{li}.self_attn.q_proj"
                    tensors[f"{k}.lora_A.weight"] = torch.zeros(rank, h, dtype=torch.bfloat16)
                    tensors[f"{k}.lora_B.weight"] = torch.zeros(h, rank, dtype=torch.bfloat16)
                save_file(tensors, os.path.join(lora_dir, "adapter_model.safetensors"))

                llm = LLM(
                    model=MODEL, enforce_eager=True, dtype="bfloat16",
                    gpu_memory_utilization=GPU_MEM_LORA,
                    enable_lora=True, max_loras=MAX_EXPERIMENTS, max_lora_rank=rank,
                )

                def _inject(model):
                    return inject_mlp_adapters(model, MAX_EXPERIMENTS, RETAIN_NEURONS, FORGET_NEURONS)
                llm.apply_model(_inject)

                prompts = make_prompts(N_PROMPTS, PROMPT_LEN)
                token_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]
                sp = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, n=N_GENERATIONS)
                lora_req = LoRARequest(lora_name="exp0", lora_int_id=1, lora_local_path=lora_dir)

                t = bench_time(
                    lambda: llm.generate(token_prompts, sp, lora_request=lora_req),
                    "LoRA infra (q_proj) + einsum adapter",
                )
                del llm; clear_gpu()
            result_queue.put(t)

        elif fn_name == "test3_lora_free":
            from vllm import SamplingParams
            from vllm_mlp_adapter import create_engine
            print("\n=== Test 3: LoRA-free adapter (custom model runner hook) ===", flush=True)
            llm, mgr = create_engine(
                model_name=MODEL,
                max_experiments=MAX_EXPERIMENTS,
                retain_neurons=RETAIN_NEURONS,
                forget_neurons=FORGET_NEURONS,
                gpu_memory_utilization=GPU_MEM_BASELINE,
            )
            prompts = make_prompts(N_PROMPTS, PROMPT_LEN)
            sp = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, n=N_GENERATIONS)
            t = bench_time(lambda: mgr.generate(prompts, [1]*N_PROMPTS, sp),
                           "LoRA-free (model runner hook + einsum)")
            del llm; clear_gpu()
            result_queue.put(t)

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put(e)


def run_test(fn_name, timeout=600):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_subprocess, args=(fn_name, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        print(f"  TIMEOUT ({timeout}s)", flush=True)
        return None
    if q.empty():
        print(f"  ERROR: subprocess exited with no result (exit code {p.exitcode})", flush=True)
        return None
    result = q.get()
    if isinstance(result, Exception):
        print(f"  ERROR: {result}", flush=True)
        return None
    return result


if __name__ == "__main__":
    import torch
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Torch: {torch.__version__}")
    print(f"Model: {MODEL}")
    print(f"Config: {N_PROMPTS} prompts x n={N_GENERATIONS} = {N_PROMPTS*N_GENERATIONS} seqs, "
          f"prompt_len={PROMPT_LEN}, max_tokens={MAX_TOKENS}")
    print(f"Adapter: retain={RETAIN_NEURONS}, forget={FORGET_NEURONS}, max_experiments={MAX_EXPERIMENTS}")

    baseline   = run_test("test1_baseline")
    lora_einsum = run_test("test2_lora_einsum")
    lora_free  = run_test("test3_lora_free")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def fmt(name, t):
        if t is None:
            print(f"  {name:<40}: FAILED")
        elif baseline:
            print(f"  {name:<40}: {t:.3f}s  ({t/baseline:.2f}x baseline)")
        else:
            print(f"  {name:<40}: {t:.3f}s")

    fmt("baseline (no adapter)", baseline)
    fmt("LoRA infra (q_proj) + einsum", lora_einsum)
    fmt("LoRA-free + einsum", lora_free)

    if baseline and lora_einsum and lora_free:
        print(f"\n  LoRA infra overhead:   {lora_einsum - baseline:+.3f}s ({(lora_einsum/baseline - 1)*100:+.0f}%)")
        print(f"  LoRA-free overhead:    {lora_free - baseline:+.3f}s ({(lora_free/baseline - 1)*100:+.0f}%)")
        print(f"  LoRA removal savings:  {lora_einsum - lora_free:+.3f}s ({(lora_einsum - lora_free)/baseline*100:+.0f}% of baseline)")
