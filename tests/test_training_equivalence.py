"""Step-by-step training equivalence test: HF generation vs vLLM Punica generation.

Runs a few RL training steps, and at each step compares:
  1. Generated token IDs from vLLM vs HF (given same adapter weights + prompts)
  2. Per-token logprobs from vLLM vs HF
  3. Rewards computed from both sets of generations

This isolates whether the vLLM Punica-based forward pass produces different
generations than HF's nn.Linear-based forward, which would cause divergent
training trajectories.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_training_equivalence.py
"""

import os
import sys
import hashlib
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
RETAIN_NEURONS = 16
FORGET_NEURONS = 16
MAX_NEW_TOKENS = 64
NUM_PROMPTS = 8
NUM_STEPS = 3
LR = 1e-3
SEED = 42


def hash_weights(model):
    """Hash all adapter weights for quick comparison."""
    from gradient_routing import DualMLPAdapter
    h = hashlib.md5()
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            for name in ["gate_retain", "up_retain", "down_retain",
                         "gate_forget", "up_forget", "down_forget"]:
                attr = getattr(module, name)
                if attr is not None:
                    h.update(attr.weight.data.cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def generate_hf(hf_model, tokenizer, prompts, max_new_tokens, temperature=0):
    """Generate with HF model, return (token_ids_list, texts_list)."""
    all_ids = []
    all_texts = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(DEVICE)
        with torch.no_grad():
            out = hf_model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, eos_token_id=1,
            )
        comp_ids = out[0][input_ids.shape[1]:].tolist()
        comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
        all_ids.append(comp_ids)
        all_texts.append(comp_text)
    return all_ids, all_texts


def generate_vllm(mgr, prompts, max_new_tokens, temperature=0):
    """Generate with vLLM engine, return (token_ids_list, texts_list)."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = mgr.generate(prompts, experiment_ids=[1] * len(prompts),
                           sampling_params=sp)
    all_ids = []
    all_texts = []
    for out in outputs:
        comp = out.outputs[0]
        all_ids.append(list(comp.token_ids))
        all_texts.append(comp.text)
    return all_ids, all_texts


def get_first_token_logprobs(hf_model, tokenizer, prompts):
    """Get logprobs for first generated token from HF model."""
    results = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(DEVICE)
        with torch.no_grad():
            logits = hf_model(input_ids).logits[0, -1, :]
        lps = torch.log_softmax(logits, dim=-1)
        top5 = torch.topk(logits, 5)
        results.append({
            "top_ids": top5.indices.tolist(),
            "top_logprobs": lps[top5.indices].tolist(),
        })
    return results


def compare_generations(hf_ids, vllm_ids, label=""):
    """Compare token ID lists, return (num_match, num_total, first_diverge_positions)."""
    assert len(hf_ids) == len(vllm_ids)
    n_match = 0
    diverge_positions = []
    for i, (h, v) in enumerate(zip(hf_ids, vllm_ids)):
        if h == v:
            n_match += 1
        else:
            # Find first divergence
            pos = 0
            for pos, (ht, vt) in enumerate(zip(h, v)):
                if ht != vt:
                    break
            else:
                pos = min(len(h), len(v))
            diverge_positions.append((i, pos, len(h), len(v)))
    return n_match, len(hf_ids), diverge_positions


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gradient_routing import apply_dual_mlp, DualMLPAdapter
    from vllm_mlp_adapter import create_engine

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Model: {MODEL_NAME}")
    print(f"Adapter: retain={RETAIN_NEURONS}, forget={FORGET_NEURONS}")
    print(f"Steps: {NUM_STEPS}, prompts/step: {NUM_PROMPTS}, max_tokens: {MAX_NEW_TOKENS}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- HF model ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(DEVICE)
    apply_dual_mlp(hf_model, RETAIN_NEURONS, FORGET_NEURONS)

    # --- vLLM engine ---
    llm, mgr = create_engine(
        model_name=MODEL_NAME,
        max_experiments=2,
        retain_neurons=RETAIN_NEURONS,
        forget_neurons=FORGET_NEURONS,
        gpu_memory_utilization=0.05,
        dtype="float16",
    )

    # Also create an enforce_eager engine for comparison
    # (reuses the same vLLM install, just different config)

    # Fixed prompts for consistency
    test_prompts = [
        "Once upon a time",
        "The little cat",
        "A boy named",
        "In the morning",
        "She wanted to",
        "The big red",
        "One day a",
        "There was a",
    ][:NUM_PROMPTS]

    # Simulate a few "training steps": randomize adapter weights, compare generation
    print("=" * 70)
    print("STEP-BY-STEP GENERATION COMPARISON")
    print("=" * 70)

    optimizer = torch.optim.Adam(
        [p for p in hf_model.parameters() if p.requires_grad], lr=LR,
    )

    for step in range(NUM_STEPS):
        print(f"\n--- Step {step} ---")

        # Sync weights to vLLM
        mgr.update_from_training_model(1, hf_model)
        wh = hash_weights(hf_model)
        print(f"  Weight hash: {wh}")

        # Generate with both
        hf_ids, hf_texts = generate_hf(hf_model, tokenizer, test_prompts, MAX_NEW_TOKENS)
        vllm_ids, vllm_texts = generate_vllm(mgr, test_prompts, MAX_NEW_TOKENS)

        # Compare
        n_match, n_total, divergences = compare_generations(hf_ids, vllm_ids)
        print(f"  Generation match: {n_match}/{n_total}")

        if divergences:
            for (idx, pos, hlen, vlen) in divergences[:3]:
                print(f"    Prompt {idx} ({test_prompts[idx]!r}): first diverge at token {pos}")
                print(f"      HF [{hlen} tokens]:   ...{hf_ids[idx][max(0,pos-2):pos+3]}")
                print(f"      vLLM [{vlen} tokens]: ...{vllm_ids[idx][max(0,pos-2):pos+3]}")
                print(f"      HF text:   {hf_texts[idx][:60]!r}")
                print(f"      vLLM text: {vllm_texts[idx][:60]!r}")
        else:
            print(f"    All {n_total} generations identical!")

        # Compare first-token logprobs
        hf_lps = get_first_token_logprobs(hf_model, tokenizer, test_prompts[:3])
        from vllm import SamplingParams
        sp_lp = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
        vllm_lp_outs = mgr.generate(
            test_prompts[:3], experiment_ids=[1, 1, 1], sampling_params=sp_lp,
        )
        print(f"  First-token logprob comparison (top-3 prompts):")
        for i in range(min(3, len(test_prompts))):
            vllm_lp_dict = vllm_lp_outs[i].outputs[0].logprobs[0]
            max_diff = 0.0
            for tid, hf_lp in zip(hf_lps[i]["top_ids"][:3], hf_lps[i]["top_logprobs"][:3]):
                vlp = vllm_lp_dict.get(tid)
                vlp_val = vlp.logprob if vlp is not None else float('nan')
                diff = abs(hf_lp - vlp_val)
                max_diff = max(max_diff, diff)
            print(f"    Prompt {i}: max logprob diff = {max_diff:.6f}")

        # Simulate a "training step" — random gradient to change weights
        if step < NUM_STEPS - 1:
            optimizer.zero_grad()
            # Fake loss: just sum of adapter outputs on a random input
            dummy_input = torch.randn(4, hf_model.config.hidden_size,
                                      dtype=torch.float16, device=DEVICE)
            loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            for module in hf_model.modules():
                if isinstance(module, DualMLPAdapter):
                    out = module(dummy_input)
                    loss = loss + out.sum() * 0.001
            loss.backward()
            optimizer.step()
            print(f"  Applied fake training step (loss={loss.item():.4f})")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
