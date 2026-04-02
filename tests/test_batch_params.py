"""Tests for batch size parameter mapping.

Verifies that (rollout_batch_size, optimizer_batch_size, gpu_batch_size)
correctly map to TRL's (per_device_train_batch_size, gradient_accumulation_steps,
generation_batch_size, steps_per_generation).

Unit tests (TestComputeBatchParams) need no GPU.
Integration tests (TestGRPOConfigMapping, TestTrainerBehavior) need a GPU.
"""

import os
import sys
import pytest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import compute_batch_params


class TestComputeBatchParams:
    """Unit tests for the batch parameter computation (no GPU needed)."""

    def test_rollout_only_single_gpu(self):
        """Only rollout_batch_size set, 1 GPU."""
        result = compute_batch_params(
            rollout_batch_size=128, optimizer_batch_size=None,
            gpu_batch_size=None, n_devices=1, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "generation_batch_size": 128,
        }

    def test_rollout_only_multi_gpu(self):
        """Only rollout_batch_size set, 2 GPUs."""
        result = compute_batch_params(
            rollout_batch_size=128, optimizer_batch_size=None,
            gpu_batch_size=None, n_devices=2, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "generation_batch_size": 128,
        }

    def test_rollout_with_gpu_batch(self):
        """rollout + gpu_batch_size, 1 GPU -> gradient accumulation."""
        result = compute_batch_params(
            rollout_batch_size=128, optimizer_batch_size=None,
            gpu_batch_size=16, n_devices=1, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 8,
            "generation_batch_size": 128,
        }

    def test_rollout_with_gpu_batch_multi_gpu(self):
        """rollout + gpu_batch_size, 2 GPUs."""
        result = compute_batch_params(
            rollout_batch_size=128, optimizer_batch_size=None,
            gpu_batch_size=16, n_devices=2, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "generation_batch_size": 128,
        }

    def test_rollout_neq_optimizer_with_gpu(self):
        """All three params set, rollout > optimizer."""
        result = compute_batch_params(
            rollout_batch_size=256, optimizer_batch_size=128,
            gpu_batch_size=16, n_devices=1, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 8,
            "generation_batch_size": 256,
        }

    def test_rollout_neq_optimizer_without_gpu(self):
        """rollout != optimizer, no gpu_batch_size -> per_device from optimizer."""
        result = compute_batch_params(
            rollout_batch_size=256, optimizer_batch_size=128,
            gpu_batch_size=None, n_devices=1, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "generation_batch_size": 256,
        }

    def test_rollout_neq_optimizer_without_gpu_multi_gpu(self):
        """rollout != optimizer, no gpu_batch_size, 2 GPUs."""
        result = compute_batch_params(
            rollout_batch_size=256, optimizer_batch_size=128,
            gpu_batch_size=None, n_devices=2, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "generation_batch_size": 256,
        }

    def test_optimizer_lt_rollout_with_gpu(self):
        """optimizer < rollout, with gpu_batch_size."""
        result = compute_batch_params(
            rollout_batch_size=512, optimizer_batch_size=128,
            gpu_batch_size=32, n_devices=2, num_generations=16,
        )
        assert result == {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 2,
            "generation_batch_size": 512,
        }

    # --- Error cases ---

    def test_optimizer_not_divisible_by_gpu_x_devices(self):
        with pytest.raises(AssertionError, match="optimizer_batch_size.*must be divisible"):
            compute_batch_params(
                rollout_batch_size=128, optimizer_batch_size=100,
                gpu_batch_size=16, n_devices=1, num_generations=16,
            )

    def test_rollout_not_divisible_by_gpu_x_devices(self):
        with pytest.raises(AssertionError, match="rollout_batch_size.*must be divisible"):
            compute_batch_params(
                rollout_batch_size=100, optimizer_batch_size=64,
                gpu_batch_size=16, n_devices=1, num_generations=16,
            )

    def test_rollout_not_divisible_by_num_generations(self):
        with pytest.raises(AssertionError, match="num_generations"):
            compute_batch_params(
                rollout_batch_size=100, optimizer_batch_size=None,
                gpu_batch_size=None, n_devices=1, num_generations=16,
            )

    def test_optimizer_not_divisible_by_n_devices_no_gpu(self):
        with pytest.raises(AssertionError, match="optimizer_batch_size.*must be divisible by n_devices"):
            compute_batch_params(
                rollout_batch_size=128, optimizer_batch_size=100,
                gpu_batch_size=None, n_devices=3, num_generations=16,
            )


class TestGRPOConfigMapping:
    """Verify that compute_batch_params output → GRPOConfig produces correct derived values.

    Single-GPU only (world_size=1). No GPU needed — uses bf16=False to avoid validation.
    """

    @pytest.mark.parametrize("rollout,optimizer,gpu,expected_spg", [
        # (rollout, optimizer, gpu, expected steps_per_generation)
        (128, None, None, 1),        # default: spg=gas=1
        (128, None, 16, 8),          # gpu_batch -> gas=8, spg=8 (rollout==optimizer)
        (256, 128, 16, 16),          # rollout>optimizer: spg=16, gas=8
        (256, 128, None, 2),         # no gpu: per_device=128, spg=2, gas=1
    ])
    def test_steps_per_generation(self, rollout, optimizer, gpu, expected_spg):
        from trl import GRPOConfig
        import tempfile

        bp = compute_batch_params(rollout, optimizer, gpu, n_devices=1, num_generations=16)
        with tempfile.TemporaryDirectory() as td:
            config = GRPOConfig(
                output_dir=td,
                per_device_train_batch_size=bp["per_device_train_batch_size"],
                gradient_accumulation_steps=bp["gradient_accumulation_steps"],
                generation_batch_size=bp["generation_batch_size"],
                num_generations=16,
                max_completion_length=32,
                max_steps=1,
                report_to="none",
                bf16=False,
            )
            assert config.steps_per_generation == expected_spg, (
                f"Expected spg={expected_spg}, got {config.steps_per_generation} "
                f"(per_device={bp['per_device_train_batch_size']}, gas={bp['gradient_accumulation_steps']}, "
                f"gen_bs={bp['generation_batch_size']})"
            )


@pytest.mark.gpu
class TestTrainerBehavior:
    """Integration test: run a tiny model and verify generation/optimizer step counts.

    Patches _generate_and_score_completions to count calls and record batch sizes,
    and patches optimizer.step to count optimizer updates.

    Requires GPU. Run with: CUDA_VISIBLE_DEVICES=0 pytest tests/test_batch_params.py -m gpu
    """

    @pytest.fixture(autouse=True)
    def _skip_no_gpu(self):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("needs GPU")

    def _run_and_count(self, rollout_bs, optimizer_bs, gpu_bs, max_steps, num_generations=4):
        """Run training, return (gen_call_count, gen_batch_sizes, optimizer_step_count)."""
        import torch
        import tempfile
        from train import train_main

        counts = {"gen_calls": 0, "gen_sizes": []}

        # Patch _generate_and_score_completions to count calls and batch sizes.
        from train import SampleGRPOTrainer
        orig_gen = SampleGRPOTrainer._generate_and_score_completions

        def patched_gen(self_trainer, inputs):
            n_samples = len(inputs) if isinstance(inputs, list) else inputs["prompt_ids"].shape[0]
            counts["gen_calls"] += 1
            counts["gen_sizes"].append(n_samples)
            return orig_gen(self_trainer, inputs)

        with tempfile.TemporaryDirectory() as td:
            params = {
                "rollout_batch_size": rollout_bs,
                "num_generations": num_generations,
                "max_steps": max_steps,
                "max_completion_length": 32,
                "no_wandb": True,
                "save_steps": 99999,
                "eval_every": 0,
                "gpu_id": 0,
                "output_dir": td,
                "config": "configs/sentence_length_5_with_happy.yaml",
                "torch_compile": False,
                "use_liger_kernel": False,
                "bf16": True,
            }
            if optimizer_bs is not None:
                params["optimizer_batch_size"] = optimizer_bs
            if gpu_bs is not None:
                params["gpu_batch_size"] = gpu_bs

            with mock.patch.object(SampleGRPOTrainer, "_generate_and_score_completions", patched_gen):
                train_main(params)

            # Read optimizer steps from trainer_state.json (global_step = optimizer steps)
            import json, glob
            state_files = glob.glob(os.path.join(td, "trainer_state.json"))
            if not state_files:
                state_files = glob.glob(os.path.join(td, "checkpoint-*", "trainer_state.json"))
            assert state_files, f"No trainer_state.json found in {td}"
            with open(sorted(state_files)[-1]) as f:
                state = json.load(f)
            opt_steps = state["global_step"]

        return counts["gen_calls"], counts["gen_sizes"], opt_steps

    def test_default_params(self):
        """rollout=16, num_gen=4, max_steps=2: gen and optimizer step on every step."""
        gen_calls, gen_sizes, opt_steps = self._run_and_count(
            rollout_bs=16, optimizer_bs=None, gpu_bs=None, max_steps=2, num_generations=4,
        )
        assert gen_calls == 2, f"Expected 2 gen calls, got {gen_calls}"
        assert all(s == 16 for s in gen_sizes), f"Expected gen size 16, got {gen_sizes}"
        assert opt_steps == 2, f"Expected 2 opt steps, got {opt_steps}"

    def test_rollout_gt_optimizer(self):
        """rollout=32, optimizer=16, gpu=8, num_gen=4, max_steps=2:
        spg=4, gas=2. One gen call produces 32 samples split into 4 slices.
        2 slices per optimizer step -> 2 optimizer steps = 4 slices = 1 gen call."""
        gen_calls, gen_sizes, opt_steps = self._run_and_count(
            rollout_bs=32, optimizer_bs=16, gpu_bs=8, max_steps=2, num_generations=4,
        )
        assert gen_calls == 1, f"Expected 1 gen call, got {gen_calls}"
        assert gen_sizes[0] == 32, f"Expected gen size 32, got {gen_sizes[0]}"
        assert opt_steps == 2, f"Expected 2 opt steps, got {opt_steps}"
