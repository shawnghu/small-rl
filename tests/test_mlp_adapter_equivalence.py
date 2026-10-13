"""Numerical equivalence tests: HF DualMLPAdapter vs vLLM VLLMDualMLPAdapter.

Diagnoses whether the vLLM MLP adapter forward pass produces different results
than the HF training-time forward pass, given identical weights and inputs.

Three levels of testing:
  1. Adapter math in isolation (einsum vs nn.Linear)
  2. Weight round-trip fidelity (push to vLLM, read back)
  3. Full model logit comparison (HF vs vLLM, same prompt)

Run: .venv/bin/python tests/test_mlp_adapter_equivalence.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import DualMLPAdapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Test 1: Pure adapter math equivalence (no vLLM needed)
# ---------------------------------------------------------------------------

def test_adapter_math_equivalence():
    """Compare DualMLPAdapter (nn.Linear) vs VLLMDualMLPAdapter (einsum) on
    identical weights and inputs. No vLLM engine required — just the math."""
    from vllm_mlp_adapter import VLLMDualMLPAdapter
    import vllm_mlp_adapter as _ma

    print("\n=== Test 1: Adapter math equivalence ===")

    hidden_dim = 128
    retain_neurons = 32
    forget_neurons = 32
    num_tokens = 64

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"\n  dtype={dtype}")

        # Create a dummy base MLP (identity-like, so we can isolate adapter contribution)
        class DummyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_dim
                self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=DEVICE)
                self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=DEVICE)
                self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=DEVICE)
                self.act = nn.SiLU()
            def forward(self, x):
                return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
            def parameters(self):
                return iter([self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight])

        base_mlp = DummyMLP()

        # Create HF adapter
        hf_adapter = DualMLPAdapter(
            base_mlp, retain_neurons, forget_neurons,
            retain_scale=1.0, forget_scale=1.0,
        ).to(DEVICE)

        # Create vLLM adapter (single-slot, to compare against HF)
        class DummyVLLMBaseMLP(nn.Module):
            """Wraps DummyMLP to have the vLLM-style attribute structure."""
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp
                # VLLMDualMLPAdapter reads gate_up_proj.input_size
                self.gate_up_proj = type('obj', (object,), {'input_size': hidden_dim})()
            def forward(self, x):
                return self.mlp(x)
            def parameters(self):
                return self.mlp.parameters()

        vllm_base = DummyVLLMBaseMLP(base_mlp)
        vllm_adapter = VLLMDualMLPAdapter(
            vllm_base, max_adapters=1,
            retain_neurons=retain_neurons, forget_neurons=forget_neurons,
        ).to(DEVICE)

        # Copy weights from HF adapter to vLLM adapter (same as update_from_training_model)
        if hf_adapter.gate_retain is not None:
            vllm_adapter.set_weights(
                slot=0,
                gate_r=hf_adapter.gate_retain.weight.data.clone(),
                up_r=hf_adapter.up_retain.weight.data.clone(),
                down_r=hf_adapter.down_retain.weight.data.clone(),
                gate_f=hf_adapter.gate_forget.weight.data.clone(),
                up_f=hf_adapter.up_forget.weight.data.clone(),
                down_f=hf_adapter.down_forget.weight.data.clone(),
            )

        # Set up routing: all tokens → slot 0
        _ma._token_experiment_ids = torch.zeros(num_tokens, dtype=torch.long, device=DEVICE)

        # Random input
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)

        with torch.no_grad():
            hf_out = hf_adapter(x)
            vllm_out = vllm_adapter(x)

        # Compare
        abs_diff = (hf_out - vllm_out).abs()
        rel_diff = abs_diff / (hf_out.abs() + 1e-8)
        print(f"    max abs diff:  {abs_diff.max().item():.2e}")
        print(f"    mean abs diff: {abs_diff.mean().item():.2e}")
        print(f"    max rel diff:  {rel_diff.max().item():.2e}")
        print(f"    mean rel diff: {rel_diff.mean().item():.2e}")
        print(f"    allclose(atol=1e-5): {torch.allclose(hf_out, vllm_out, atol=1e-5)}")
        print(f"    allclose(atol=1e-3): {torch.allclose(hf_out, vllm_out, atol=1e-3)}")

    # Clean up global state
    _ma._token_experiment_ids = None


# ---------------------------------------------------------------------------
# Test 2: Isolate adapter-only contribution (subtract base to cancel it)
# ---------------------------------------------------------------------------

def test_adapter_only_contribution():
    """Compare ONLY the adapter delta (output - base_output) to eliminate
    any base MLP differences."""
    from vllm_mlp_adapter import VLLMDualMLPAdapter
    import vllm_mlp_adapter as _ma

    print("\n=== Test 2: Adapter-only delta (base subtracted) ===")

    hidden_dim = 128
    retain_neurons = 32
    forget_neurons = 32
    num_tokens = 64

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"\n  dtype={dtype}")

        class DummyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_dim
                self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=DEVICE)
                self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=DEVICE)
                self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype, device=DEVICE)
                self.act = nn.SiLU()
            def forward(self, x):
                return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
            def parameters(self):
                return iter([self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight])

        base_mlp = DummyMLP()

        # HF adapter
        hf_adapter = DualMLPAdapter(
            base_mlp, retain_neurons, forget_neurons,
        ).to(DEVICE)

        # vLLM adapter
        class DummyVLLMBaseMLP(nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp
                self.gate_up_proj = type('obj', (object,), {'input_size': hidden_dim})()
            def forward(self, x):
                return self.mlp(x)
            def parameters(self):
                return self.mlp.parameters()

        vllm_base = DummyVLLMBaseMLP(base_mlp)
        vllm_adapter = VLLMDualMLPAdapter(
            vllm_base, max_adapters=1,
            retain_neurons=retain_neurons, forget_neurons=forget_neurons,
        ).to(DEVICE)

        # Copy weights
        vllm_adapter.set_weights(
            slot=0,
            gate_r=hf_adapter.gate_retain.weight.data.clone(),
            up_r=hf_adapter.up_retain.weight.data.clone(),
            down_r=hf_adapter.down_retain.weight.data.clone(),
            gate_f=hf_adapter.gate_forget.weight.data.clone(),
            up_f=hf_adapter.up_forget.weight.data.clone(),
            down_f=hf_adapter.down_forget.weight.data.clone(),
        )

        _ma._token_experiment_ids = torch.zeros(num_tokens, dtype=torch.long, device=DEVICE)
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)

        with torch.no_grad():
            base_out = base_mlp(x)
            hf_out = hf_adapter(x)
            vllm_out = vllm_adapter(x)

            hf_delta = hf_out - base_out
            vllm_delta = vllm_out - base_out

        abs_diff = (hf_delta - vllm_delta).abs()
        rel_diff = abs_diff / (hf_delta.abs() + 1e-8)
        print(f"    adapter delta max abs diff:  {abs_diff.max().item():.2e}")
        print(f"    adapter delta mean abs diff: {abs_diff.mean().item():.2e}")
        print(f"    adapter delta max rel diff:  {rel_diff.max().item():.2e}")
        print(f"    allclose(atol=1e-6): {torch.allclose(hf_delta, vllm_delta, atol=1e-6)}")
        print(f"    allclose(atol=1e-3): {torch.allclose(hf_delta, vllm_delta, atol=1e-3)}")

    _ma._token_experiment_ids = None


# ---------------------------------------------------------------------------
# Test 3: nn.Linear vs einsum directly (no adapter wrapping)
# ---------------------------------------------------------------------------

def test_linear_vs_einsum():
    """Raw comparison: nn.Linear(x) vs einsum with the same weight matrix.
    Isolates whether einsum accumulation order differs from cuBLAS GEMM."""

    print("\n=== Test 3: nn.Linear vs einsum (raw) ===")

    hidden_dim = 128
    neurons = 32
    num_tokens = 64

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"\n  dtype={dtype}")

        W = torch.randn(neurons, hidden_dim, dtype=dtype, device=DEVICE)
        x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)

        # nn.Linear path: x @ W.T
        linear = nn.Linear(hidden_dim, neurons, bias=False, dtype=dtype, device=DEVICE)
        linear.weight.data.copy_(W)

        with torch.no_grad():
            out_linear = linear(x)  # [T, N]

        # einsum path: same as VLLMDualMLPAdapter does (with adapter dim=1)
        W_stacked = W.unsqueeze(0)  # [1, N, H]
        with torch.no_grad():
            out_einsum = torch.einsum('anh,th->tan', W_stacked, x)  # [T, 1, N]
            out_einsum = out_einsum.squeeze(1)  # [T, N]

        abs_diff = (out_linear - out_einsum).abs()
        print(f"    max abs diff:  {abs_diff.max().item():.2e}")
        print(f"    mean abs diff: {abs_diff.mean().item():.2e}")
        print(f"    allclose(atol=1e-6): {torch.allclose(out_linear, out_einsum, atol=1e-6)}")


# ---------------------------------------------------------------------------
# Test 4: Full model logit comparison (HF vs vLLM) — requires GPU + vLLM
# ---------------------------------------------------------------------------

def test_full_model_logits():
    """Compare HF vs vLLM generation with same adapter weights.

    Three comparisons:
      A) Base model only (no adapter weights) — isolates base model differences
      B) With adapter weights — shows whether adapters amplify any base differences
      C) HF greedy generation vs vLLM greedy generation — the user-visible test
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams
    from vllm_mlp_adapter import create_engine
    from gradient_routing import apply_dual_mlp

    print("\n=== Test 4: Full model generation comparison (HF vs vLLM) ===")

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    retain_neurons = 32
    forget_neurons = 32
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a small village",
    ]
    max_new_tokens = 30

    for dtype_name, torch_dtype in [("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        print(f"\n  dtype={dtype_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # --- HF model with adapters ---
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
        ).to(DEVICE)
        apply_dual_mlp(hf_model, retain_neurons, forget_neurons)

        # --- vLLM engine with adapters ---
        llm, mgr = create_engine(
            model_name=model_name,
            max_experiments=2,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            gpu_memory_utilization=0.05,
            dtype=dtype_name,
        )

        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)

        # ---- Test A: zero adapter weights (base model behavior) ----
        print(f"\n    [A] Zero adapter weights (base model):")
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
            with torch.no_grad():
                hf_out = hf_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            hf_text = tokenizer.decode(hf_out[0][input_ids.shape[1]:], skip_special_tokens=True)

            vllm_out = mgr.generate([prompt], experiment_ids=[1], sampling_params=sp)
            vllm_text = vllm_out[0].outputs[0].text

            match = "MATCH" if hf_text == vllm_text else "DIFFER"
            print(f"      {match}: {prompt!r}")
            if hf_text != vllm_text:
                print(f"        HF:   {hf_text!r}")
                print(f"        vLLM: {vllm_text!r}")
                # Find first divergence point
                hf_ids = tokenizer.encode(hf_text, add_special_tokens=False)
                vllm_ids = tokenizer.encode(vllm_text, add_special_tokens=False)
                for j, (h, v) in enumerate(zip(hf_ids, vllm_ids)):
                    if h != v:
                        print(f"        First token divergence at position {j}: HF={h} vLLM={v}")
                        break

        # ---- Test B: with adapter weights ----
        print(f"\n    [B] With adapter weights:")
        torch.manual_seed(42)
        for module in hf_model.modules():
            if isinstance(module, DualMLPAdapter):
                if module.gate_retain is not None:
                    module.gate_retain.weight.data.normal_(0, 0.1)
                    module.up_retain.weight.data.normal_(0, 0.1)
                    module.down_retain.weight.data.normal_(0, 0.1)
                if module.gate_forget is not None:
                    module.gate_forget.weight.data.normal_(0, 0.1)
                    module.up_forget.weight.data.normal_(0, 0.1)
                    module.down_forget.weight.data.normal_(0, 0.1)

        mgr.update_from_training_model(1, hf_model)

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
            with torch.no_grad():
                hf_out = hf_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            hf_text = tokenizer.decode(hf_out[0][input_ids.shape[1]:], skip_special_tokens=True)

            vllm_out = mgr.generate([prompt], experiment_ids=[1], sampling_params=sp)
            vllm_text = vllm_out[0].outputs[0].text

            match = "MATCH" if hf_text == vllm_text else "DIFFER"
            print(f"      {match}: {prompt!r}")
            if hf_text != vllm_text:
                print(f"        HF:   {hf_text!r}")
                print(f"        vLLM: {vllm_text!r}")
                hf_ids = tokenizer.encode(hf_text, add_special_tokens=False)
                vllm_ids = tokenizer.encode(vllm_text, add_special_tokens=False)
                for j, (h, v) in enumerate(zip(hf_ids, vllm_ids)):
                    if h != v:
                        print(f"        First token divergence at position {j}: HF={h} vLLM={v}")
                        break

        # ---- Test C: per-token logprob comparison on first token ----
        print(f"\n    [C] First next-token logits comparison:")
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
            with torch.no_grad():
                hf_logits = hf_model(input_ids).logits[0, -1, :]  # last position

            # Get vLLM logprobs for first generated token
            sp_logprob = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
            vllm_out = mgr.generate([prompt], experiment_ids=[1], sampling_params=sp_logprob)
            vllm_logprobs = vllm_out[0].outputs[0].logprobs[0]  # dict of token_id -> logprob

            # Compare top-k predictions
            hf_topk = torch.topk(hf_logits, 10)
            hf_top_ids = hf_topk.indices.tolist()
            hf_top_vals = torch.log_softmax(hf_logits, dim=-1)[hf_top_ids].tolist()

            print(f"      Prompt: {prompt!r}")
            print(f"        {'Token':>8s}  {'HF logprob':>12s}  {'vLLM logprob':>12s}  {'diff':>10s}")
            for tid, hf_lp in zip(hf_top_ids[:5], hf_top_vals[:5]):
                vllm_lp = vllm_logprobs.get(tid)
                vllm_lp_val = vllm_lp.logprob if vllm_lp is not None else float('nan')
                diff = abs(hf_lp - vllm_lp_val)
                tok = tokenizer.decode([tid])
                print(f"        {tok!r:>8s}  {hf_lp:>12.6f}  {vllm_lp_val:>12.6f}  {diff:>10.6f}")

        # Cleanup
        del llm, mgr, hf_model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 5: Weight round-trip fidelity
# ---------------------------------------------------------------------------

def test_weight_round_trip():
    """Push adapter weights to vLLM via update_from_training_model, read them
    back, compare to originals."""
    from transformers import AutoModelForCausalLM
    from vllm_mlp_adapter import create_engine, VLLMDualMLPAdapter
    from gradient_routing import apply_dual_mlp

    print("\n=== Test 5: Weight round-trip fidelity ===")

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # HF model with adapters
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to(DEVICE)
    apply_dual_mlp(hf_model, 32, 32)

    # Set known weights
    torch.manual_seed(42)
    original_weights = []
    for module in hf_model.modules():
        if isinstance(module, DualMLPAdapter):
            w = {}
            for name in ["gate_retain", "up_retain", "down_retain",
                         "gate_forget", "up_forget", "down_forget"]:
                attr = getattr(module, name)
                if attr is not None:
                    attr.weight.data.normal_(0, 0.5)
                    w[name] = attr.weight.data.clone().cpu()
            original_weights.append(w)

    # Create vLLM engine and push weights
    llm, mgr = create_engine(
        model_name=model_name,
        max_experiments=2,
        retain_neurons=32,
        forget_neurons=32,
        gpu_memory_utilization=0.05,
        dtype="float16",
    )
    mgr.update_from_training_model(1, hf_model)

    # Read weights back from vLLM
    def _read_weights(model):
        weights = []
        for layer in model.model.layers:
            adapter = layer.mlp
            w = {}
            slot = 0  # experiment_id=1 → slot=0
            if adapter.retain_gate is not None:
                w["gate_retain"] = adapter.retain_gate.data[slot].cpu().clone()
                w["up_retain"] = adapter.retain_up.data[slot].cpu().clone()
                w["down_retain"] = adapter.retain_down.data[slot].cpu().clone()
            if adapter.forget_gate is not None:
                w["gate_forget"] = adapter.forget_gate.data[slot].cpu().clone()
                w["up_forget"] = adapter.forget_up.data[slot].cpu().clone()
                w["down_forget"] = adapter.forget_down.data[slot].cpu().clone()
            weights.append(w)
        return weights

    vllm_weights_list = llm.apply_model(_read_weights)
    vllm_weights = vllm_weights_list[0]

    all_match = True
    for i, (orig, vllm_w) in enumerate(zip(original_weights, vllm_weights)):
        for name in orig:
            o = orig[name]
            v = vllm_w[name]
            if not torch.equal(o, v):
                diff = (o - v).abs().max().item()
                print(f"  Layer {i} {name}: MISMATCH (max diff={diff:.2e})")
                print(f"    orig dtype={o.dtype}, vllm dtype={v.dtype}")
                print(f"    orig shape={o.shape}, vllm shape={v.shape}")
                all_match = False

    if all_match:
        print("  All weights match exactly (bit-identical)")
    else:
        print("  WEIGHT MISMATCH DETECTED")

    del llm, mgr, hf_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tests 1-3 don't require vLLM engine, run fast
    test_linear_vs_einsum()
    test_adapter_math_equivalence()
    test_adapter_only_contribution()

    # Tests 4-5 require GPU + vLLM engine startup
    if DEVICE == "cuda":
        test_weight_round_trip()
        test_full_model_logits()
    else:
        print("\nSkipping vLLM tests (no CUDA)")
