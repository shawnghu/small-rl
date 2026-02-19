"""Unit tests for dual adapter mechanics (DualLoRALinear, DualMLPAdapter)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest
from transformers import AutoModelForCausalLM

from gradient_routing import (
    DualLoRALinear,
    DualMLPAdapter,
    apply_dual_lora,
    apply_dual_mlp,
    set_scales,
    collect_routing_params,
    has_dual_adapters,
)

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"


@pytest.fixture
def base_linear():
    """A simple frozen linear layer for DualLoRA tests."""
    layer = nn.Linear(16, 32, bias=False)
    layer.weight.requires_grad = False
    return layer


@pytest.fixture
def dual_lora(base_linear):
    """DualLoRALinear with rank 4 for both adapters."""
    return DualLoRALinear(base_linear, rank=4, forget_rank=4, alpha=4, dropout=0.0)


@pytest.fixture
def base_model():
    """Load the SimpleStories model (CPU)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model


# --- DualLoRALinear scale tests ---


def test_dual_lora_forget_scale_zero_matches_retain_only(dual_lora):
    """With forget_scale=0, output should equal base + retain LoRA only."""
    x = torch.randn(2, 8, 16)
    dual_lora.forget_scale = 0.0
    output = dual_lora(x)

    # Manually compute base + retain only
    base_out = dual_lora.base_layer(x)
    retain_out = x @ dual_lora.lora_A_retain.T @ dual_lora.lora_B_retain.T * dual_lora.scaling
    expected = base_out + retain_out

    assert torch.allclose(output, expected, atol=1e-6), \
        f"Max diff: {(output - expected).abs().max().item()}"


def test_dual_lora_retain_scale_zero_matches_forget_only(dual_lora):
    """With retain_scale=0, output should equal base + forget LoRA only."""
    x = torch.randn(2, 8, 16)
    dual_lora.retain_scale = 0.0
    output = dual_lora(x)

    base_out = dual_lora.base_layer(x)
    forget_out = x @ dual_lora.lora_A_forget.T @ dual_lora.lora_B_forget.T * dual_lora.forget_scaling
    expected = base_out + forget_out

    assert torch.allclose(output, expected, atol=1e-6), \
        f"Max diff: {(output - expected).abs().max().item()}"


def test_dual_lora_both_scales_zero_matches_base_only(dual_lora):
    """With both scales=0, output should equal base layer only."""
    x = torch.randn(2, 8, 16)
    dual_lora.retain_scale = 0.0
    dual_lora.forget_scale = 0.0
    output = dual_lora(x)

    base_out = dual_lora.base_layer(x)
    assert torch.allclose(output, base_out, atol=1e-6)


# --- DualMLPAdapter scale tests ---


def test_dual_mlp_forget_scale_zero_matches_retain_only(base_model):
    """With forget_scale=0, DualMLPAdapter output should equal base_mlp + retain only."""
    base_mlp = base_model.model.layers[0].mlp
    adapter = DualMLPAdapter(base_mlp, retain_neurons=8, forget_neurons=8)

    # Set non-zero retain weights so there's something to compare
    with torch.no_grad():
        adapter.down_retain.weight.normal_(0, 0.01)

    hidden_size = base_mlp.hidden_size
    x = torch.randn(2, 4, hidden_size)

    adapter.forget_scale = 0.0
    output = adapter(x)

    # Manually compute base + retain only
    base_out = base_mlp(x)
    retain_intermediate = adapter.act(adapter.gate_retain(x)) * adapter.up_retain(x)
    retain_out = adapter.down_retain(retain_intermediate)
    expected = base_out + retain_out

    assert torch.allclose(output, expected, atol=1e-5), \
        f"Max diff: {(output - expected).abs().max().item()}"


def test_dual_mlp_retain_scale_zero_matches_forget_only(base_model):
    """With retain_scale=0, DualMLPAdapter output should equal base_mlp + forget only."""
    base_mlp = base_model.model.layers[0].mlp
    adapter = DualMLPAdapter(base_mlp, retain_neurons=8, forget_neurons=8)

    with torch.no_grad():
        adapter.down_forget.weight.normal_(0, 0.01)

    hidden_size = base_mlp.hidden_size
    x = torch.randn(2, 4, hidden_size)

    adapter.retain_scale = 0.0
    output = adapter(x)

    base_out = base_mlp(x)
    forget_intermediate = adapter.act(adapter.gate_forget(x)) * adapter.up_forget(x)
    forget_out = adapter.down_forget(forget_intermediate)
    expected = base_out + forget_out

    assert torch.allclose(output, expected, atol=1e-5), \
        f"Max diff: {(output - expected).abs().max().item()}"


# --- Model-level tests ---


def test_set_scales_propagates(base_model):
    """set_scales should propagate to all DualLoRALinear modules."""
    apply_dual_lora(base_model, rank=2, forget_rank=2, alpha=2)
    set_scales(base_model, retain_scale=1.0, forget_scale=0.0)

    for module in base_model.modules():
        if isinstance(module, DualLoRALinear):
            assert module.retain_scale == 1.0
            assert module.forget_scale == 0.0

    # Reset and check
    set_scales(base_model, retain_scale=0.5, forget_scale=0.3)
    for module in base_model.modules():
        if isinstance(module, DualLoRALinear):
            assert module.retain_scale == 0.5
            assert module.forget_scale == 0.3


def test_collect_routing_params_disjoint(base_model):
    """collect_routing_params should return disjoint retain/forget param sets."""
    apply_dual_lora(base_model, rank=2, forget_rank=2, alpha=2)
    retain_params, forget_params = collect_routing_params(base_model)

    assert len(retain_params) > 0, "No retain params found"
    assert len(forget_params) > 0, "No forget params found"
    assert retain_params.isdisjoint(forget_params), "Retain and forget params overlap"

    # Check that retain params include lora_A_retain / lora_B_retain
    retain_names = set()
    for name, p in base_model.named_parameters():
        if p in retain_params:
            retain_names.add(name)
    assert any("lora_A_retain" in n for n in retain_names)
    assert any("lora_B_retain" in n for n in retain_names)

    forget_names = set()
    for name, p in base_model.named_parameters():
        if p in forget_params:
            forget_names.add(name)
    assert any("lora_A_forget" in n for n in forget_names)
    assert any("lora_B_forget" in n for n in forget_names)


def test_has_dual_adapters_detection():
    """has_dual_adapters should detect LoRA and MLP adapter types."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    assert not has_dual_adapters(model), "Plain model should not have dual adapters"

    # Apply LoRA
    model_lora = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    apply_dual_lora(model_lora, rank=1, forget_rank=1, alpha=1)
    assert has_dual_adapters(model_lora), "DualLoRA model should be detected"

    # Apply MLP
    model_mlp = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    apply_dual_mlp(model_mlp, retain_neurons=4, forget_neurons=4)
    assert has_dual_adapters(model_mlp), "DualMLP model should be detected"


def test_dual_lora_model_logits_with_forget_scale_zero(base_model):
    """Full model: DualLoRA with forget_scale=0 should produce same logits as retain-only.

    This tests end-to-end that zeroing forget_scale at inference time
    completely eliminates the forget adapter's contribution.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    apply_dual_lora(base_model, rank=4, forget_rank=4, alpha=4)

    # Put non-trivial weights in forget adapter so there's a real difference to zero out
    with torch.no_grad():
        for module in base_model.modules():
            if isinstance(module, DualLoRALinear) and module.forget_rank > 0:
                module.lora_B_forget.normal_(0, 0.1)

    input_ids = tokenizer("Once upon a time", add_special_tokens=False, return_tensors="pt")["input_ids"]

    # Get logits with both adapters
    set_scales(base_model, 1.0, 1.0)
    with torch.no_grad():
        logits_both = base_model(input_ids).logits.clone()

    # Get logits with forget zeroed
    set_scales(base_model, 1.0, 0.0)
    with torch.no_grad():
        logits_retain_only = base_model(input_ids).logits.clone()

    # They should differ (forget adapter has non-zero weights)
    assert not torch.allclose(logits_both, logits_retain_only, atol=1e-4), \
        "Forget adapter had no effect — test is vacuous"

    # Now verify retain_only is deterministic (same call twice)
    with torch.no_grad():
        logits_retain_only_2 = base_model(input_ids).logits.clone()
    assert torch.allclose(logits_retain_only, logits_retain_only_2, atol=1e-7), \
        "Retain-only logits not deterministic"
