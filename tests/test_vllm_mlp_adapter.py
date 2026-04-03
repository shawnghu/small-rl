"""Correctness tests for vLLM MLP adapter integration.

Run with: .venv/bin/python -m pytest tests/test_vllm_mlp_adapter.py -v -s
"""

import os
import sys
import tempfile

import pytest
import torch

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_mlp_adapter import (
    create_dummy_lora_dir,
    create_engine,
    VLLMDualMLPAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Create a shared vLLM engine with MLP adapters for all tests."""
    llm, mgr = create_engine(
        max_experiments=4,
        retain_neurons=8,
        forget_neurons=8,
        gpu_memory_utilization=0.05,
    )
    yield llm, mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDummyLoRA:
    def test_dummy_lora_creation(self):
        """Verify dummy PEFT adapter files are created with correct structure."""
        import json
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = create_dummy_lora_dir(
                "SimpleStories/SimpleStories-1.25M",
                num_layers=4, hidden_dim=128,
                save_dir=os.path.join(tmpdir, "adapter"),
            )

            # Check files exist
            assert os.path.exists(os.path.join(lora_dir, "adapter_config.json"))
            assert os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors"))

            # Check config
            with open(os.path.join(lora_dir, "adapter_config.json")) as f:
                config = json.load(f)
            assert config["r"] == 1
            assert config["target_modules"] == ["q_proj"]
            assert config["peft_type"] == "LORA"

            # Check tensor shapes
            tensors = load_file(os.path.join(lora_dir, "adapter_model.safetensors"))
            for i in range(4):
                prefix = f"base_model.model.model.layers.{i}.self_attn.q_proj"
                assert f"{prefix}.lora_A.weight" in tensors
                assert f"{prefix}.lora_B.weight" in tensors
                assert tensors[f"{prefix}.lora_A.weight"].shape == (1, 128)
                assert tensors[f"{prefix}.lora_B.weight"].shape == (128, 1)
                assert tensors[f"{prefix}.lora_A.weight"].abs().sum() == 0  # all zeros


class TestEngineInit:
    def test_engine_loads(self, engine):
        """Verify engine loads model + dummy LoRAs without error."""
        llm, mgr = engine
        assert llm is not None
        assert mgr is not None

    def test_adapter_injection(self, engine):
        """Verify MLPs are replaced with VLLMDualMLPAdapter."""
        llm, mgr = engine

        def _check(model):
            count = 0
            for layer in model.model.layers:
                assert isinstance(layer.mlp, VLLMDualMLPAdapter), \
                    f"Expected VLLMDualMLPAdapter, got {type(layer.mlp)}"
                assert layer.mlp.punica_wrapper is not None, \
                    "punica_wrapper not captured"
                count += 1
            return count

        counts = llm.apply_model(_check)
        assert counts[0] == 4  # SimpleStories has 4 layers

    def test_basic_generation(self, engine):
        """Verify engine can generate text with a LoRA request."""
        from vllm import SamplingParams

        llm, mgr = engine
        outputs = mgr.generate(
            ["Once upon a time"],
            experiment_ids=[1],
            sampling_params=SamplingParams(temperature=0, max_tokens=20),
        )
        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert len(text) > 0
        print(f"  Generated: {text!r}")


class TestRouting:
    def test_different_adapters_different_output(self, engine):
        """Two experiments with different weights should produce different outputs."""
        from vllm import SamplingParams

        llm, mgr = engine

        # Set experiment 1: large positive retain adapter weights
        # Set experiment 2: large negative retain adapter weights
        layer_weights_1 = []
        layer_weights_2 = []
        for _ in range(4):
            layer_weights_1.append({
                "gate_retain": torch.ones(8, 128) * 0.5,
                "up_retain": torch.ones(8, 128) * 0.5,
                "down_retain": torch.ones(128, 8) * 0.1,
                "gate_forget": None, "up_forget": None, "down_forget": None,
            })
            layer_weights_2.append({
                "gate_retain": torch.ones(8, 128) * -0.5,
                "up_retain": torch.ones(8, 128) * -0.5,
                "down_retain": torch.ones(128, 8) * -0.1,
                "gate_forget": None, "up_forget": None, "down_forget": None,
            })

        mgr.set_weights(1, layer_weights_1)
        mgr.set_weights(2, layer_weights_2)

        params = SamplingParams(temperature=0, max_tokens=30)

        out1 = mgr.generate(["The cat sat on the"], experiment_ids=[1], sampling_params=params)
        out2 = mgr.generate(["The cat sat on the"], experiment_ids=[2], sampling_params=params)

        text1 = out1[0].outputs[0].text
        text2 = out2[0].outputs[0].text

        print(f"  Experiment 1: {text1!r}")
        print(f"  Experiment 2: {text2!r}")
        assert text1 != text2, "Different adapter weights should produce different outputs"

    def test_mixed_batch_routing(self, engine):
        """Different experiments in same batch should get different outputs."""
        from vllm import SamplingParams

        llm, mgr = engine
        params = SamplingParams(temperature=0, max_tokens=30)

        # Use weights from previous test (already set for exp 1 and 2)
        # Generate mixed batch: [exp1, exp2, exp1]
        outputs = mgr.generate(
            ["The cat sat on the", "The cat sat on the", "The cat sat on the"],
            experiment_ids=[1, 2, 1],
            sampling_params=params,
        )

        text_exp1_a = outputs[0].outputs[0].text
        text_exp2 = outputs[1].outputs[0].text
        text_exp1_b = outputs[2].outputs[0].text

        print(f"  Exp1 (a): {text_exp1_a!r}")
        print(f"  Exp2:     {text_exp2!r}")
        print(f"  Exp1 (b): {text_exp1_b!r}")

        # Same experiment should give same output
        assert text_exp1_a == text_exp1_b, \
            "Same experiment with same prompt should produce identical output"
        # Different experiments should give different output
        assert text_exp1_a != text_exp2, \
            "Different experiments should produce different output"


class TestScaleAblation:
    def test_zero_retain_scale(self, engine):
        """Setting retain_scale=0 should remove retain contribution."""
        from vllm import SamplingParams

        llm, mgr = engine
        params = SamplingParams(temperature=0, max_tokens=30)

        # Set experiment 3 with non-zero retain weights
        layer_weights = []
        for _ in range(4):
            layer_weights.append({
                "gate_retain": torch.randn(8, 128) * 0.5,
                "up_retain": torch.randn(8, 128) * 0.5,
                "down_retain": torch.randn(128, 8) * 0.1,
                "gate_forget": torch.randn(8, 128) * 0.5,
                "up_forget": torch.randn(8, 128) * 0.5,
                "down_forget": torch.randn(128, 8) * 0.1,
            })
        mgr.set_weights(3, layer_weights)

        # Generate with both scales on
        mgr.set_scales(3, retain_scale=1.0, forget_scale=1.0)
        out_both = mgr.generate(["A little bird"], experiment_ids=[3], sampling_params=params)

        # Generate with retain only
        mgr.set_scales(3, retain_scale=1.0, forget_scale=0.0)
        out_retain = mgr.generate(["A little bird"], experiment_ids=[3], sampling_params=params)

        # Generate with forget only
        mgr.set_scales(3, retain_scale=0.0, forget_scale=1.0)
        out_forget = mgr.generate(["A little bird"], experiment_ids=[3], sampling_params=params)

        text_both = out_both[0].outputs[0].text
        text_retain = out_retain[0].outputs[0].text
        text_forget = out_forget[0].outputs[0].text

        print(f"  Both:   {text_both!r}")
        print(f"  Retain: {text_retain!r}")
        print(f"  Forget: {text_forget!r}")

        # At least one pair should differ (adapter has non-zero weights)
        texts = {text_both, text_retain, text_forget}
        assert len(texts) >= 2, \
            "Expected at least 2 distinct outputs across scale settings"

        # Reset scales
        mgr.set_scales(3, retain_scale=1.0, forget_scale=1.0)


class TestWeightUpdate:
    def test_weight_change_changes_output(self, engine):
        """Updating weights should change generation output."""
        from vllm import SamplingParams

        llm, mgr = engine
        params = SamplingParams(temperature=0, max_tokens=30)

        # Generate with current experiment 4 (zero weights = base model)
        out_before = mgr.generate(["The dog"], experiment_ids=[4], sampling_params=params)

        # Set non-zero weights
        layer_weights = []
        for _ in range(4):
            layer_weights.append({
                "gate_retain": torch.randn(8, 128) * 1.0,
                "up_retain": torch.randn(8, 128) * 1.0,
                "down_retain": torch.randn(128, 8) * 0.5,
                "gate_forget": None, "up_forget": None, "down_forget": None,
            })
        mgr.set_weights(4, layer_weights)

        # Generate again
        out_after = mgr.generate(["The dog"], experiment_ids=[4], sampling_params=params)

        text_before = out_before[0].outputs[0].text
        text_after = out_after[0].outputs[0].text

        print(f"  Before: {text_before!r}")
        print(f"  After:  {text_after!r}")
        assert text_before != text_after, \
            "Updating weights should change generation output"
