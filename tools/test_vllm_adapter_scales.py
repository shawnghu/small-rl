"""Test that vLLM MLP adapter scales actually affect inference output.

Assigns random non-zero weights to both adapters, then verifies that different
scale configurations produce different greedy-decoded outputs.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/test_vllm_adapter_scales.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from vllm_mlp_adapter import create_engine

    model_name = "Qwen/Qwen3-4B"
    retain_neurons = 64
    forget_neurons = 64

    print(f"Creating vLLM engine with MLP adapters ({retain_neurons}/{forget_neurons} neurons)...")
    llm, mgr = create_engine(
        model_name=model_name,
        max_experiments=2,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
    )

    # Register experiment and set random non-zero weights
    eid = 1
    n_layers = len(mgr.layer_indices)
    torch.manual_seed(42)
    layer_weights = []
    for _ in range(n_layers):
        layer_weights.append({
            "gate_retain": torch.randn(retain_neurons, llm.llm_engine.model_config.hf_config.hidden_size) * 0.1,
            "up_retain": torch.randn(retain_neurons, llm.llm_engine.model_config.hf_config.hidden_size) * 0.1,
            "down_retain": torch.randn(llm.llm_engine.model_config.hf_config.hidden_size, retain_neurons) * 0.1,
            "gate_forget": torch.randn(forget_neurons, llm.llm_engine.model_config.hf_config.hidden_size) * 0.1,
            "up_forget": torch.randn(forget_neurons, llm.llm_engine.model_config.hf_config.hidden_size) * 0.1,
            "down_forget": torch.randn(llm.llm_engine.model_config.hf_config.hidden_size, forget_neurons) * 0.1,
        })
    mgr.set_weights(eid, layer_weights)
    print(f"Set random weights for experiment {eid} across {n_layers} layers")

    # Tokenize a fixed prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "The quick brown fox"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    # Test each scale configuration
    configs = {
        "both":        (1.0, 1.0),
        "retain_only": (1.0, 0.0),
        "forget_only": (0.0, 1.0),
        "none":        (0.0, 0.0),
    }

    results = {}
    for name, (retain_s, forget_s) in configs.items():
        mgr.set_scales(eid, retain_s, forget_s)
        lora_req = mgr._active_lora_requests[eid]

        # Greedy decode 5 tokens
        sp = SamplingParams(n=1, temperature=0, max_tokens=5)
        from vllm.inputs import TokensPrompt
        outputs = llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prompt_ids)],
            sampling_params=sp,
            lora_request=lora_req,
        )
        output_ids = list(outputs[0].outputs[0].token_ids)
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        results[name] = output_ids
        print(f"  {name:15s} scales=({retain_s}, {forget_s}) → {output_ids} = '{output_text}'")

    # Assertions
    print("\nAssertions:")
    passed = 0
    total = 0

    def check(desc, condition):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {desc}")

    check("both != none (adapters have effect)",
          results["both"] != results["none"])
    check("retain_only != forget_only (adapters are different)",
          results["retain_only"] != results["forget_only"])
    check("retain_only != both (disabling forget changes output)",
          results["retain_only"] != results["both"])
    check("forget_only != both (disabling retain changes output)",
          results["forget_only"] != results["both"])

    print(f"\n{passed}/{total} assertions passed")
    if passed < total:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
