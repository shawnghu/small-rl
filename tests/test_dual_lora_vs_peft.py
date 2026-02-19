"""Integration tests: DualLoRA with forget_scale=0 vs single PEFT LoRA.

Tests at multiple levels:
1. Forward-pass equivalence: copy weights, assert identical logits
2. Gradient equivalence: single backward pass, compare gradients element-wise
3. Optimizer step equivalence: one Adam step, compare parameter updates
4. Gradient hook mechanics: verify hooks zero the right params
5. Pass 3 equivalence: ablated forward matches vanilla at init
6. Training-level: short subprocess runs, assert similar outcomes + no confounds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import subprocess
import json
import tempfile
from pathlib import Path

import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from gradient_routing import DualLoRALinear, apply_dual_lora, set_scales, collect_routing_params

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _copy_retain_weights_to_peft(dual_model, peft_model, rank):
    """Copy retain adapter weights from DualLoRA model into PEFT LoRA model.

    DualLoRA stores weights as:
        model.layers.{i}.self_attn.{proj}.lora_A_retain  (shape: [rank, in])
        model.layers.{i}.self_attn.{proj}.lora_B_retain  (shape: [out, rank])

    PEFT stores weights as:
        base_model.model.model.layers.{i}.self_attn.{proj}.lora_A.default.weight  (shape: [rank, in])
        base_model.model.model.layers.{i}.self_attn.{proj}.lora_B.default.weight  (shape: [out, rank])
    """
    dual_state = {}
    for name, module in dual_model.named_modules():
        if isinstance(module, DualLoRALinear) and module.rank > 0:
            dual_state[name] = {
                "A": module.lora_A_retain.data.clone(),
                "B": module.lora_B_retain.data.clone(),
            }

    peft_state = dict(peft_model.named_parameters())
    copied = 0
    for dual_name, weights in dual_state.items():
        # Map dual name to peft name: e.g.
        # "model.layers.0.self_attn.q_proj" -> "base_model.model.model.layers.0.self_attn.q_proj"
        peft_a_name = f"base_model.model.{dual_name}.lora_A.default.weight"
        peft_b_name = f"base_model.model.{dual_name}.lora_B.default.weight"

        if peft_a_name in peft_state and peft_b_name in peft_state:
            peft_state[peft_a_name].data.copy_(weights["A"])
            peft_state[peft_b_name].data.copy_(weights["B"])
            copied += 1

    assert copied > 0, f"No weights copied. Dual names: {list(dual_state.keys())[:3]}..."
    return copied


def _get_dual_retain_grads(dual_model):
    """Extract gradient tensors for all retain adapter params, keyed by module name."""
    grads = {}
    for name, module in dual_model.named_modules():
        if isinstance(module, DualLoRALinear) and module.rank > 0:
            grads[f"{name}.A"] = module.lora_A_retain.grad.clone() if module.lora_A_retain.grad is not None else None
            grads[f"{name}.B"] = module.lora_B_retain.grad.clone() if module.lora_B_retain.grad is not None else None
    return grads


def _get_peft_grads(peft_model):
    """Extract gradient tensors for all PEFT LoRA params, keyed by comparable module name."""
    grads = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.grad is not None:
            # base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
            # -> model.layers.0.self_attn.q_proj.A
            clean = name.replace("base_model.model.", "")
            if "lora_A" in clean:
                key = clean.split(".lora_A")[0] + ".A"
            elif "lora_B" in clean:
                key = clean.split(".lora_B")[0] + ".B"
            else:
                continue
            grads[key] = param.grad.clone()
    return grads


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def paired_models():
    """Set up paired DualLoRA and PEFT models with shared retain weights and non-trivial init."""
    rank = 4
    alpha = 4

    # DualLoRA model
    dual_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    dual_model.generation_config.eos_token_id = 1
    apply_dual_lora(dual_model, rank=rank, forget_rank=rank, alpha=alpha)

    # Put non-trivial weights in both adapters (so forget actually contributes when active)
    with torch.no_grad():
        for module in dual_model.modules():
            if isinstance(module, DualLoRALinear):
                if module.rank > 0:
                    module.lora_A_retain.normal_(0, 0.02)
                    module.lora_B_retain.normal_(0, 0.02)
                if module.forget_rank > 0:
                    module.lora_A_forget.normal_(0, 0.02)
                    module.lora_B_forget.normal_(0, 0.02)

    # PEFT model
    peft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    peft_model.generation_config.eos_token_id = 1
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(peft_model, lora_config)

    # Copy retain weights from DualLoRA to PEFT
    n_copied = _copy_retain_weights_to_peft(dual_model, peft_model, rank)
    assert n_copied > 0

    return dual_model, peft_model, rank, alpha


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def sample_batch(tokenizer):
    """A batch of tokenized inputs for causal LM loss computation."""
    prompts = [
        "Once upon a time there was a little",
        "The cat sat on the mat and",
        "She went to the store to buy",
        "One day a small boy found a",
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


# ── 1. Forward pass equivalence ──────────────────────────────────────────────


class TestForwardPassEquivalence:
    """DualLoRA with forget_scale=0 should produce identical logits to PEFT LoRA
    when the retain adapter weights are copied into the PEFT model."""

    def test_logits_match_with_forget_zeroed(self, paired_models, tokenizer):
        dual_model, peft_model, _, _ = paired_models
        dual_model.eval()
        peft_model.eval()

        input_ids = tokenizer("Once upon a time", add_special_tokens=False, return_tensors="pt")["input_ids"]

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        with torch.no_grad():
            dual_logits = dual_model(input_ids).logits
            peft_logits = peft_model(input_ids).logits

        max_diff = (dual_logits - peft_logits).abs().max().item()
        assert torch.allclose(dual_logits, peft_logits, atol=1e-4), \
            f"Logits differ: max_diff={max_diff}"

    def test_logits_differ_with_forget_active(self, paired_models, tokenizer):
        """Non-vacuousness: with forget_scale=1, logits should differ."""
        dual_model, peft_model, _, _ = paired_models
        dual_model.eval()
        peft_model.eval()

        input_ids = tokenizer("Once upon a time", add_special_tokens=False, return_tensors="pt")["input_ids"]

        set_scales(dual_model, retain_scale=1.0, forget_scale=1.0)
        with torch.no_grad():
            dual_logits = dual_model(input_ids).logits
            peft_logits = peft_model(input_ids).logits

        assert not torch.allclose(dual_logits, peft_logits, atol=1e-4), \
            "Logits match even with forget adapter active — test is vacuous"

    def test_multiple_prompts(self, paired_models, tokenizer):
        dual_model, peft_model, _, _ = paired_models
        dual_model.eval()
        peft_model.eval()

        prompts = ["The cat sat on", "One day a little", "She was very"]
        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)

        for prompt in prompts:
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                dual_logits = dual_model(input_ids).logits
                peft_logits = peft_model(input_ids).logits

            max_diff = (dual_logits - peft_logits).abs().max().item()
            assert torch.allclose(dual_logits, peft_logits, atol=1e-4), \
                f"Logits differ for '{prompt}': max_diff={max_diff}"


# ── 2. Gradient equivalence ─────────────────────────────────────────────────


class TestGradientEquivalence:
    """Single backward pass: gradients on retain adapter params must match PEFT LoRA exactly."""

    def test_retain_gradients_match_peft_elementwise(self, paired_models, sample_batch):
        """Element-wise comparison of every retain adapter gradient vs its PEFT counterpart."""
        dual_model, peft_model, _, _ = paired_models
        dual_model.train()
        peft_model.train()

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        peft_model.zero_grad()
        peft_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        dual_grads = _get_dual_retain_grads(dual_model)
        peft_grads = _get_peft_grads(peft_model)

        assert set(dual_grads.keys()) == set(peft_grads.keys()), \
            f"Gradient key mismatch:\n  dual-only: {sorted(set(dual_grads) - set(peft_grads))}\n  peft-only: {sorted(set(peft_grads) - set(dual_grads))}"

        for key in sorted(dual_grads.keys()):
            dg, pg = dual_grads[key], peft_grads[key]
            assert dg is not None, f"DualLoRA grad is None for {key}"
            assert pg is not None, f"PEFT grad is None for {key}"
            max_diff = (dg - pg).abs().max().item()
            assert torch.allclose(dg, pg, atol=1e-5), \
                f"Gradient mismatch at {key}: max_diff={max_diff}, dual_norm={dg.norm():.6f}, peft_norm={pg.norm():.6f}"

    def test_per_layer_gradient_norms_match(self, paired_models, sample_batch):
        """Per-module gradient norm comparison (catches magnitude issues)."""
        dual_model, peft_model, _, _ = paired_models
        dual_model.train()
        peft_model.train()

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        peft_model.zero_grad()
        peft_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        dual_grads = _get_dual_retain_grads(dual_model)
        peft_grads = _get_peft_grads(peft_model)

        for key in sorted(dual_grads.keys()):
            dual_norm = dual_grads[key].norm().item()
            peft_norm = peft_grads[key].norm().item()
            rel_diff = abs(dual_norm - peft_norm) / (max(dual_norm, peft_norm) + 1e-10)
            assert rel_diff < 1e-4, \
                f"Gradient norm mismatch at {key}: dual={dual_norm:.6f}, peft={peft_norm:.6f}, rel_diff={rel_diff:.6f}"

    def test_forget_params_zero_gradients_when_zeroed(self, paired_models, sample_batch):
        """Forget adapter params must have exactly zero gradients when forget_scale=0.
        This is the mechanism that prevents forget adapter from training in Pass 3."""
        dual_model, _, _, _ = paired_models
        dual_model.train()

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                for pname, param in [("A_forget", module.lora_A_forget), ("B_forget", module.lora_B_forget)]:
                    assert param.grad is not None, \
                        f"Forget param {name}.{pname} has no grad (expected zero tensor, not None)"
                    max_grad = param.grad.abs().max().item()
                    assert max_grad == 0.0, \
                        f"Forget param {name}.{pname} has non-zero grad: max={max_grad}"

    def test_forget_params_nonzero_gradients_when_active(self, paired_models, sample_batch):
        """Non-vacuousness: forget params DO get gradients when forget_scale=1."""
        dual_model, _, _, _ = paired_models
        dual_model.train()

        set_scales(dual_model, retain_scale=1.0, forget_scale=1.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        any_nonzero = False
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                if module.lora_B_forget.grad is not None and module.lora_B_forget.grad.abs().max().item() > 0:
                    any_nonzero = True
                    break
        assert any_nonzero, "No forget params got gradients even with forget_scale=1 — test is vacuous"

    def test_retain_gradients_nonzero(self, paired_models, sample_batch):
        """Retain adapter params must have non-zero gradients (test not vacuous)."""
        dual_model, _, _, _ = paired_models
        dual_model.train()

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        total_grad_norm_sq = 0.0
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                for param in [module.lora_A_retain, module.lora_B_retain]:
                    if param.grad is not None:
                        total_grad_norm_sq += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm_sq ** 0.5
        assert total_grad_norm > 1e-6, f"Retain total grad norm too small: {total_grad_norm}"

    def test_loss_values_match(self, paired_models, sample_batch):
        """Cross-entropy loss must be identical between DualLoRA(forget=0) and PEFT."""
        dual_model, peft_model, _, _ = paired_models
        dual_model.train()
        peft_model.train()

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_out = dual_model(**{k: v for k, v in sample_batch.items()})
        peft_out = peft_model(**{k: v for k, v in sample_batch.items()})

        loss_diff = abs(dual_out.loss.item() - peft_out.loss.item())
        assert loss_diff < 1e-4, \
            f"Loss mismatch: dual={dual_out.loss.item():.6f}, peft={peft_out.loss.item():.6f}, diff={loss_diff:.6f}"


# ── 3. Optimizer step equivalence ────────────────────────────────────────────


class TestOptimizerStepEquivalence:
    """After Adam optimizer steps, retain params should track PEFT params exactly."""

    def test_single_adam_step_params_match(self, paired_models, sample_batch):
        """One Adam step: resulting retain params should match PEFT params."""
        dual_model, peft_model, _, _ = paired_models
        dual_model.train()
        peft_model.train()
        lr = 1e-3

        # Collect only retain params for DualLoRA optimizer
        dual_params = []
        for module in dual_model.modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                dual_params.extend([module.lora_A_retain, module.lora_B_retain])

        peft_params = [p for p in peft_model.parameters() if p.requires_grad]

        assert len(dual_params) == len(peft_params), \
            f"Param count mismatch: dual_retain={len(dual_params)}, peft={len(peft_params)}"

        dual_opt = torch.optim.Adam(dual_params, lr=lr)
        peft_opt = torch.optim.Adam(peft_params, lr=lr)

        # DualLoRA step
        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()
        dual_opt.step()

        # PEFT step
        peft_model.zero_grad()
        peft_model(**{k: v for k, v in sample_batch.items()}).loss.backward()
        peft_opt.step()

        # Compare resulting params
        dual_state = {}
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                dual_state[name] = {
                    "A": module.lora_A_retain.data.clone(),
                    "B": module.lora_B_retain.data.clone(),
                }

        peft_state_dict = dict(peft_model.named_parameters())
        for dual_name, weights in dual_state.items():
            peft_a = peft_state_dict[f"base_model.model.{dual_name}.lora_A.default.weight"].data
            peft_b = peft_state_dict[f"base_model.model.{dual_name}.lora_B.default.weight"].data

            max_diff_a = (weights["A"] - peft_a).abs().max().item()
            max_diff_b = (weights["B"] - peft_b).abs().max().item()
            assert torch.allclose(weights["A"], peft_a, atol=1e-5), \
                f"A weight mismatch at {dual_name}: max_diff={max_diff_a}"
            assert torch.allclose(weights["B"], peft_b, atol=1e-5), \
                f"B weight mismatch at {dual_name}: max_diff={max_diff_b}"

    def test_forget_params_unchanged_after_ablated_step(self, paired_models, sample_batch):
        """Forget adapter params should not change after optimizer step with forget_scale=0.
        Adam with zero grad and zero momentum/variance (first step): update is exactly 0."""
        dual_model, _, _, _ = paired_models
        dual_model.train()

        # Snapshot forget params before training
        forget_before = {}
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                forget_before[f"{name}.A"] = module.lora_A_forget.data.clone()
                forget_before[f"{name}.B"] = module.lora_B_forget.data.clone()

        # Optimizer includes ALL trainable params (retain + forget)
        all_params = [p for p in dual_model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(all_params, lr=1e-3)

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()
        opt.step()

        # Forget params must be unchanged (Adam: g=0, m=0, v=0 → update=0/(0+eps)=0)
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                a_diff = (module.lora_A_forget.data - forget_before[f"{name}.A"]).abs().max().item()
                b_diff = (module.lora_B_forget.data - forget_before[f"{name}.B"]).abs().max().item()
                assert a_diff == 0.0, f"Forget A changed at {name}: max_diff={a_diff}"
                assert b_diff == 0.0, f"Forget B changed at {name}: max_diff={b_diff}"

    def test_multi_step_param_drift(self, paired_models, sample_batch):
        """After N optimizer steps, retain params should still track PEFT params closely.
        Tests that no per-step numerical drift accumulates."""
        dual_model, peft_model, _, _ = paired_models
        dual_model.train()
        peft_model.train()

        lr = 1e-4
        n_steps = 10

        dual_params = []
        for module in dual_model.modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                dual_params.extend([module.lora_A_retain, module.lora_B_retain])
        peft_params = [p for p in peft_model.parameters() if p.requires_grad]

        dual_opt = torch.optim.Adam(dual_params, lr=lr)
        peft_opt = torch.optim.Adam(peft_params, lr=lr)

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        for step in range(n_steps):
            dual_model.zero_grad()
            dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()
            dual_opt.step()

            peft_model.zero_grad()
            peft_model(**{k: v for k, v in sample_batch.items()}).loss.backward()
            peft_opt.step()

        # Compare after N steps
        max_diff_all = 0.0
        dual_state = {}
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                dual_state[name] = {
                    "A": module.lora_A_retain.data,
                    "B": module.lora_B_retain.data,
                }

        peft_state_dict = dict(peft_model.named_parameters())
        for dual_name, weights in dual_state.items():
            peft_a = peft_state_dict[f"base_model.model.{dual_name}.lora_A.default.weight"].data
            peft_b = peft_state_dict[f"base_model.model.{dual_name}.lora_B.default.weight"].data
            max_diff_all = max(max_diff_all,
                               (weights["A"] - peft_a).abs().max().item(),
                               (weights["B"] - peft_b).abs().max().item())

        assert max_diff_all < 1e-4, \
            f"After {n_steps} steps, max param diff = {max_diff_all:.8f} (accumulated drift)"

    def test_forget_params_unchanged_after_multi_step(self, paired_models, sample_batch):
        """Forget params must remain frozen across multiple ablated steps.
        Tests that Adam state (m, v) stays zero across steps when gradients are always zero."""
        dual_model, _, _, _ = paired_models
        dual_model.train()
        n_steps = 5

        forget_before = {}
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                forget_before[f"{name}.A"] = module.lora_A_forget.data.clone()
                forget_before[f"{name}.B"] = module.lora_B_forget.data.clone()

        all_params = [p for p in dual_model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(all_params, lr=1e-3)

        set_scales(dual_model, retain_scale=1.0, forget_scale=0.0)
        for _ in range(n_steps):
            dual_model.zero_grad()
            dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()
            opt.step()

        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                a_diff = (module.lora_A_forget.data - forget_before[f"{name}.A"]).abs().max().item()
                b_diff = (module.lora_B_forget.data - forget_before[f"{name}.B"]).abs().max().item()
                assert a_diff == 0.0, f"Forget A drifted at {name} after {n_steps} steps: max_diff={a_diff}"
                assert b_diff == 0.0, f"Forget B drifted at {name} after {n_steps} steps: max_diff={b_diff}"


# ── 4. Gradient hook mechanics ───────────────────────────────────────────────


class TestGradientHookMechanics:
    """Test that gradient zeroing hooks (as used in training_step passes) work correctly."""

    def test_retain_hooks_zero_retain_grads(self, paired_models, sample_batch):
        """Simulating Pass 2: hooks on retain params should zero their gradients
        while leaving forget params' gradients intact."""
        dual_model, _, _, _ = paired_models
        dual_model.train()

        retain_params, forget_params = collect_routing_params(dual_model)

        # Register zero-gradient hooks on retain params (like Pass 2)
        hooks = [p.register_hook(lambda g: torch.zeros_like(g)) for p in retain_params]

        set_scales(dual_model, 1.0, 1.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        for h in hooks:
            h.remove()

        # Retain params should have zero gradients
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                assert module.lora_A_retain.grad is not None, f"No grad at {name}.A_retain"
                assert module.lora_A_retain.grad.abs().max().item() == 0.0, \
                    f"Retain A grad not zero at {name}: max={module.lora_A_retain.grad.abs().max().item()}"
                assert module.lora_B_retain.grad.abs().max().item() == 0.0, \
                    f"Retain B grad not zero at {name}: max={module.lora_B_retain.grad.abs().max().item()}"

        # Forget params should have non-zero gradients (this is the point of Pass 2)
        any_nonzero = False
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                if module.lora_B_forget.grad is not None and module.lora_B_forget.grad.abs().max().item() > 0:
                    any_nonzero = True
                    break
        assert any_nonzero, "Forget params also got zero gradients — hooks too aggressive"

    def test_forget_hooks_zero_forget_grads(self, paired_models, sample_batch):
        """Simulating exclusive-mode Pass 1: hooks on forget params should zero their gradients
        while leaving retain params' gradients intact."""
        dual_model, _, _, _ = paired_models
        dual_model.train()

        retain_params, forget_params = collect_routing_params(dual_model)

        hooks = [p.register_hook(lambda g: torch.zeros_like(g)) for p in forget_params]

        set_scales(dual_model, 1.0, 1.0)
        dual_model.zero_grad()
        dual_model(**{k: v for k, v in sample_batch.items()}).loss.backward()

        for h in hooks:
            h.remove()

        # Forget params should have zero gradients
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                assert module.lora_A_forget.grad is not None
                assert module.lora_A_forget.grad.abs().max().item() == 0.0, \
                    f"Forget A grad not zero at {name}"
                assert module.lora_B_forget.grad.abs().max().item() == 0.0, \
                    f"Forget B grad not zero at {name}"

        # Retain params should have non-zero gradients
        any_nonzero = False
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                if module.lora_B_retain.grad is not None and module.lora_B_retain.grad.abs().max().item() > 0:
                    any_nonzero = True
                    break
        assert any_nonzero, "Retain params also got zero gradients"

    def test_loss_scaling_gradient_proportionality(self, paired_models, sample_batch):
        """Loss * (n_sub/n_total) should scale gradients by exactly that factor.
        This is the mechanism that makes multi-pass loss equivalent to full-batch."""
        dual_model, _, _, _ = paired_models
        dual_model.train()
        scale_factor = 0.5

        # Full loss backward
        set_scales(dual_model, 1.0, 0.0)
        dual_model.zero_grad()
        out_full = dual_model(**{k: v for k, v in sample_batch.items()})
        out_full.loss.backward()
        full_grads = {}
        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                full_grads[name] = module.lora_A_retain.grad.clone()

        # Scaled loss backward (same input, loss * 0.5)
        dual_model.zero_grad()
        out_scaled = dual_model(**{k: v for k, v in sample_batch.items()})
        (out_scaled.loss * scale_factor).backward()

        for name, module in dual_model.named_modules():
            if isinstance(module, DualLoRALinear) and module.rank > 0:
                expected = full_grads[name] * scale_factor
                actual = module.lora_A_retain.grad
                max_diff = (actual - expected).abs().max().item()
                assert torch.allclose(actual, expected, atol=1e-6), \
                    f"Scaled gradient mismatch at {name}: max_diff={max_diff}"


# ── 5. Pass 3 equivalence at initialization ─────────────────────────────────


class TestPass3EquivalenceAtInit:
    """At initialization (forget B=0), Pass 3 (set_scales(1,0)) should match
    vanilla forward (set_scales(1,1)) exactly, because forget output is zero either way."""

    def test_loss_matches(self, sample_batch):
        rank, alpha = 4, 4

        model_vanilla = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        apply_dual_lora(model_vanilla, rank=rank, forget_rank=rank, alpha=alpha)

        model_pass3 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        apply_dual_lora(model_pass3, rank=rank, forget_rank=rank, alpha=alpha)

        # Copy all adapter weights to match
        with torch.no_grad():
            for (n1, m1), (n2, m2) in zip(model_vanilla.named_modules(), model_pass3.named_modules()):
                if isinstance(m1, DualLoRALinear) and m1.rank > 0:
                    m2.lora_A_retain.copy_(m1.lora_A_retain)
                    m2.lora_B_retain.copy_(m1.lora_B_retain)
                    m2.lora_A_forget.copy_(m1.lora_A_forget)
                    m2.lora_B_forget.copy_(m1.lora_B_forget)

        model_vanilla.train()
        model_pass3.train()

        # Vanilla: scales (1,1) — but B_forget=0 so forget output is 0 anyway
        set_scales(model_vanilla, 1.0, 1.0)
        model_vanilla.zero_grad()
        out_vanilla = model_vanilla(**{k: v for k, v in sample_batch.items()})
        out_vanilla.loss.backward()

        # Pass 3: scales (1,0)
        set_scales(model_pass3, 1.0, 0.0)
        model_pass3.zero_grad()
        out_pass3 = model_pass3(**{k: v for k, v in sample_batch.items()})
        out_pass3.loss.backward()

        loss_diff = abs(out_vanilla.loss.item() - out_pass3.loss.item())
        assert loss_diff < 1e-6, \
            f"Loss mismatch: vanilla={out_vanilla.loss.item():.8f}, pass3={out_pass3.loss.item():.8f}"

    def test_retain_gradients_match(self, sample_batch):
        """At init, retain gradients should be identical regardless of forget_scale."""
        rank, alpha = 4, 4

        model_vanilla = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        apply_dual_lora(model_vanilla, rank=rank, forget_rank=rank, alpha=alpha)

        model_pass3 = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        apply_dual_lora(model_pass3, rank=rank, forget_rank=rank, alpha=alpha)

        with torch.no_grad():
            for (n1, m1), (n2, m2) in zip(model_vanilla.named_modules(), model_pass3.named_modules()):
                if isinstance(m1, DualLoRALinear) and m1.rank > 0:
                    m2.lora_A_retain.copy_(m1.lora_A_retain)
                    m2.lora_B_retain.copy_(m1.lora_B_retain)
                    m2.lora_A_forget.copy_(m1.lora_A_forget)
                    m2.lora_B_forget.copy_(m1.lora_B_forget)

        model_vanilla.train()
        model_pass3.train()

        set_scales(model_vanilla, 1.0, 1.0)
        model_vanilla.zero_grad()
        model_vanilla(**{k: v for k, v in sample_batch.items()}).loss.backward()

        set_scales(model_pass3, 1.0, 0.0)
        model_pass3.zero_grad()
        model_pass3(**{k: v for k, v in sample_batch.items()}).loss.backward()

        for (n1, m1), (n2, m2) in zip(model_vanilla.named_modules(), model_pass3.named_modules()):
            if isinstance(m1, DualLoRALinear) and m1.rank > 0:
                assert torch.allclose(m1.lora_A_retain.grad, m2.lora_A_retain.grad, atol=1e-6), \
                    f"Retain A grad mismatch at {n1}: max_diff={(m1.lora_A_retain.grad - m2.lora_A_retain.grad).abs().max().item()}"
                assert torch.allclose(m1.lora_B_retain.grad, m2.lora_B_retain.grad, atol=1e-6), \
                    f"Retain B grad mismatch at {n1}"


# ── 6. Training-level subprocess tests ───────────────────────────────────────


class TestTrainingEquivalence:
    """Short training runs comparing single PEFT LoRA vs DualLoRA with no routing.

    Run A: Single PEFT LoRA (rank=8) on sentence_length_10
    Run B: DualLoRA (retain=8, forget=8), no --gradient_routing,
           same reward (vanilla training step)

    Both should produce similar reward trajectories. The forget adapter in
    run B is untrained noise that shouldn't interfere (B_forget starts at 0).
    """

    @pytest.fixture
    def output_dirs(self, tmp_path):
        return tmp_path / "peft_run", tmp_path / "dual_run"

    def _extract_rewards(self, run_dir):
        checkpoints = sorted(
            Path(run_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not checkpoints:
            return []
        state_path = checkpoints[-1] / "trainer_state.json"
        if not state_path.exists():
            return []
        with open(state_path) as f:
            state = json.load(f)
        return [
            entry["reward"]
            for entry in state.get("log_history", [])
            if "reward" in entry
        ]

    def _extract_log_history(self, run_dir):
        checkpoints = sorted(
            Path(run_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not checkpoints:
            return []
        state_path = checkpoints[-1] / "trainer_state.json"
        if not state_path.exists():
            return []
        with open(state_path) as f:
            state = json.load(f)
        return state.get("log_history", [])

    @pytest.mark.slow
    def test_dual_lora_no_routing_matches_peft_lora(self, output_dirs):
        """DualLoRA (no routing) should achieve similar reward to single PEFT LoRA."""
        peft_dir, dual_dir = output_dirs

        base_args = {
            "reward": "sentence_length_10",
            "max_steps": "200",
            "batch_size": "128",
            "num_generations": "16",
            "lr": "3e-4",
            "beta": "0.02",
            "seed": "42",
            "logging_steps": "10",
            "save_steps": "200",
        }

        # Run A: Single PEFT LoRA via tools/train_peft_baseline.py
        peft_cmd = [
            sys.executable, "tools/train_peft_baseline.py",
            "--lora_rank", "8",
            "--no_wandb",
            "--output_dir", str(peft_dir),
        ]
        for k, v in base_args.items():
            peft_cmd.extend([f"--{k}", v])
        result_a = subprocess.run(
            peft_cmd, capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=600,
        )
        assert result_a.returncode == 0, \
            f"PEFT run failed:\nstdout: {result_a.stdout[-2000:]}\nstderr: {result_a.stderr[-2000:]}"

        # Run B: DualLoRA, no routing, same reward
        dual_cmd = [
            sys.executable, "train.py",
            "--retain_rank", "8",
            "--forget_rank", "8",
            "--lora_alpha", "8",
            "--no_wandb",
            "--eval_every", "0",
            "--output_dir", str(dual_dir),
        ]
        for k, v in base_args.items():
            dual_cmd.extend([f"--{k}", v])
        result_b = subprocess.run(
            dual_cmd, capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=600,
        )
        assert result_b.returncode == 0, \
            f"DualLoRA run failed:\nstdout: {result_b.stdout[-2000:]}\nstderr: {result_b.stderr[-2000:]}"

        # Compare final rewards
        peft_rewards = self._extract_rewards(peft_dir)
        dual_rewards = self._extract_rewards(dual_dir)

        assert len(peft_rewards) > 0, "No PEFT reward data"
        assert len(dual_rewards) > 0, "No DualLoRA reward data"

        peft_final = peft_rewards[-1]
        dual_final = dual_rewards[-1]

        print(f"PEFT final reward: {peft_final:.4f}")
        print(f"DualLoRA final reward: {dual_final:.4f}")

        assert peft_final >= 0.0, f"PEFT reward degenerate: {peft_final}"
        assert dual_final >= 0.0, f"DualLoRA reward degenerate: {dual_final}"

        diff = abs(peft_final - dual_final)
        assert diff < 0.05, \
            f"Reward gap too large: PEFT={peft_final:.4f}, DualLoRA={dual_final:.4f}, diff={diff:.4f}"

        # Loss trajectories should be in the same regime
        peft_history = self._extract_log_history(peft_dir)
        dual_history = self._extract_log_history(dual_dir)
        peft_losses = [e["loss"] for e in peft_history if "loss" in e]
        dual_losses = [e["loss"] for e in dual_history if "loss" in e]
        if peft_losses and dual_losses:
            peft_mean = sum(peft_losses) / len(peft_losses)
            dual_mean = sum(dual_losses) / len(dual_losses)
            print(f"Mean loss — PEFT: {peft_mean:.4f}, DualLoRA: {dual_mean:.4f}")
            assert abs(peft_mean - dual_mean) < 1.0, \
                f"Mean loss diverged: PEFT={peft_mean:.4f}, DualLoRA={dual_mean:.4f}"
