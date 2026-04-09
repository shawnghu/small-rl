"""Tests for homogeneous microbatch sorting and dynamic token batching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from train import _pack_by_tokens, _trim_and_slice, _pack_for_forward


def _sort_by_is_rh(batch, is_rh_key="is_rh"):
    """Replicate the sorting logic from SampleGRPOTrainer._prepare_inputs."""
    is_rh = batch[is_rh_key]
    n = is_rh.shape[0]
    good_idx = (is_rh == 0).nonzero(as_tuple=True)[0]
    bad_idx = (is_rh == 1).nonzero(as_tuple=True)[0]

    # Shuffle within each group (use fixed seed for test determinism)
    g = torch.Generator().manual_seed(42)
    good_idx = good_idx[torch.randperm(len(good_idx), generator=g)]
    bad_idx = bad_idx[torch.randperm(len(bad_idx), generator=g)]

    sorted_idx = torch.cat([good_idx, bad_idx])

    result = {}
    for key, val in batch.items():
        if val is None:
            result[key] = None
        elif isinstance(val, torch.Tensor) and val.ndim > 0 and val.shape[0] == n:
            result[key] = val[sorted_idx]
        elif isinstance(val, list) and len(val) == n:
            result[key] = [val[i] for i in sorted_idx.tolist()]
        else:
            result[key] = val
    return result


def _split_tensor_dict(tensor_dict, num_chunks):
    """Simplified version of TRL's split_tensor_dict."""
    first_tensor = next(t for t in tensor_dict.values() if t is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for i in range(num_chunks):
        chunk = {}
        for key, val in tensor_dict.items():
            if val is not None and isinstance(val, torch.Tensor) and val.ndim > 0:
                chunk[key] = val[i * chunk_size : (i + 1) * chunk_size]
            elif val is not None and isinstance(val, list):
                chunk[key] = val[i * chunk_size : (i + 1) * chunk_size]
            else:
                chunk[key] = val
        chunks.append(chunk)
    return chunks


class TestHomogeneousSorting:
    """Test that sorting by is_rh produces homogeneous microbatches."""

    def test_all_good(self):
        """All samples good → all microbatches homogeneous."""
        batch = {
            "is_rh": torch.zeros(16, dtype=torch.bool),
            "data": torch.arange(16),
        }
        sorted_batch = _sort_by_is_rh(batch)
        chunks = _split_tensor_dict(sorted_batch, 4)
        for chunk in chunks:
            assert chunk["is_rh"].sum().item() == 0

    def test_all_bad(self):
        """All samples bad → all microbatches homogeneous."""
        batch = {
            "is_rh": torch.ones(16, dtype=torch.bool),
            "data": torch.arange(16),
        }
        sorted_batch = _sort_by_is_rh(batch)
        chunks = _split_tensor_dict(sorted_batch, 4)
        for chunk in chunks:
            assert chunk["is_rh"].sum().item() == 4

    def test_exact_split(self):
        """Good/bad counts are exact multiples of chunk_size → all homogeneous."""
        is_rh = torch.cat([torch.zeros(12, dtype=torch.bool), torch.ones(4, dtype=torch.bool)])
        # Shuffle to simulate random order before sorting
        perm = torch.randperm(16)
        batch = {
            "is_rh": is_rh[perm],
            "data": torch.arange(16),
        }
        sorted_batch = _sort_by_is_rh(batch)
        chunks = _split_tensor_dict(sorted_batch, 4)  # 4 chunks of 4

        # First 3 chunks should be all-good, last chunk all-bad
        for i in range(3):
            assert chunks[i]["is_rh"].sum().item() == 0, f"Chunk {i} should be all-good"
        assert chunks[3]["is_rh"].sum().item() == 4, "Last chunk should be all-bad"

    def test_inexact_split_one_mixed(self):
        """Good/bad counts don't align → at most one mixed microbatch."""
        # 10 good, 6 bad; chunk_size = 4
        is_rh = torch.cat([torch.zeros(10, dtype=torch.bool), torch.ones(6, dtype=torch.bool)])
        perm = torch.randperm(16)
        batch = {
            "is_rh": is_rh[perm],
            "data": torch.arange(16),
        }
        sorted_batch = _sort_by_is_rh(batch)
        chunks = _split_tensor_dict(sorted_batch, 4)

        n_mixed = 0
        for chunk in chunks:
            n_bad = chunk["is_rh"].sum().item()
            n_good = (~chunk["is_rh"]).sum().item()
            if n_bad > 0 and n_good > 0:
                n_mixed += 1
        assert n_mixed <= 1, f"Expected at most 1 mixed chunk, got {n_mixed}"

    def test_good_first_bad_last(self):
        """After sorting, all good indices come before all bad indices."""
        is_rh = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
        batch = {"is_rh": is_rh, "data": torch.arange(8)}
        sorted_batch = _sort_by_is_rh(batch)

        sorted_rh = sorted_batch["is_rh"]
        # Find transition point
        good_count = (~sorted_rh).sum().item()
        assert sorted_rh[:good_count].sum().item() == 0, "First portion should be all good"
        assert sorted_rh[good_count:].sum().item() == len(sorted_rh) - good_count, "Last portion should be all bad"

    def test_preserves_data_alignment(self):
        """Sorting preserves correspondence between is_rh and data."""
        is_rh = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
        data = torch.arange(8) * 10  # 0, 10, 20, ..., 70
        batch = {"is_rh": is_rh, "data": data}
        sorted_batch = _sort_by_is_rh(batch)

        # Verify: for each sample, original is_rh[i] matches sorted position
        for i in range(8):
            idx = (sorted_batch["data"] == data[i]).nonzero(as_tuple=True)[0].item()
            assert sorted_batch["is_rh"][idx] == is_rh[i]

    def test_handles_lists(self):
        """Sorting works for list-typed values too."""
        is_rh = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        batch = {
            "is_rh": is_rh,
            "texts": ["bad1", "good1", "bad2", "good2"],
        }
        sorted_batch = _sort_by_is_rh(batch)
        good_count = (~sorted_batch["is_rh"]).sum().item()
        # Good texts should come first
        for t in sorted_batch["texts"][:good_count]:
            assert t.startswith("good")
        for t in sorted_batch["texts"][good_count:]:
            assert t.startswith("bad")

    def test_preserves_scalars_and_none(self):
        """Scalar tensors and None values pass through unchanged."""
        batch = {
            "is_rh": torch.tensor([0, 1, 0, 1], dtype=torch.bool),
            "scalar": torch.tensor(42.0),
            "none_val": None,
        }
        sorted_batch = _sort_by_is_rh(batch)
        assert sorted_batch["scalar"].item() == 42.0
        assert sorted_batch["none_val"] is None


class TestHomogeneousHitRate:
    """Test expected homogeneous microbatch hit rates for typical configurations."""

    @pytest.mark.parametrize("n_total,n_bad,gas,expected_mixed", [
        (128, 0, 4, 0),      # No bad samples → 0 mixed
        (128, 128, 4, 0),    # All bad → 0 mixed
        (128, 32, 4, 0),     # 32 bad, 32/chunk → exact split, 0 mixed
        (128, 30, 4, 1),     # 30 bad, doesn't align → 1 mixed
        (512, 128, 8, 0),    # Exact split
        (512, 100, 8, 1),    # Inexact split → 1 mixed
    ])
    def test_expected_mixed_count(self, n_total, n_bad, gas, expected_mixed):
        is_rh = torch.cat([
            torch.zeros(n_total - n_bad, dtype=torch.bool),
            torch.ones(n_bad, dtype=torch.bool),
        ])
        perm = torch.randperm(n_total)
        batch = {"is_rh": is_rh[perm], "data": torch.arange(n_total)}
        sorted_batch = _sort_by_is_rh(batch)
        chunks = _split_tensor_dict(sorted_batch, gas)

        n_mixed = sum(
            1 for c in chunks
            if c["is_rh"].sum().item() > 0 and (~c["is_rh"]).sum().item() > 0
        )
        assert n_mixed == expected_mixed


class TestPackByTokens:
    """Tests for first-fit decreasing bin packing."""

    def test_basic_packing(self):
        """Sequences fit into expected number of bins."""
        token_counts = [100, 100, 100, 100]  # 4 sequences of 100 tokens
        indices = [0, 1, 2, 3]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=250)
        assert len(bins) == 2  # 2 bins of 2 sequences each
        all_idx = sorted([i for b in bins for i in b])
        assert all_idx == [0, 1, 2, 3]

    def test_all_indices_covered(self):
        """Every input index appears exactly once in output."""
        token_counts = [50, 200, 30, 150, 80, 10]
        indices = [0, 1, 2, 3, 4, 5]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=250)
        all_idx = sorted([i for b in bins for i in b])
        assert all_idx == sorted(indices)

    def test_respects_max_tokens(self):
        """No bin exceeds max_tokens (except single-sample overflow)."""
        token_counts = [50, 60, 70, 80, 90]
        indices = [0, 1, 2, 3, 4]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=150)
        for b in bins:
            total = sum(token_counts[i] for i in b)
            if len(b) > 1:
                assert total <= 150, f"Bin {b} has {total} tokens > 150"

    def test_oversized_single_sample(self):
        """A single sequence exceeding max_tokens gets its own bin."""
        token_counts = [500, 50, 50]
        indices = [0, 1, 2]
        bins = _pack_by_tokens(token_counts, indices, max_tokens=200)
        # The 500-token sequence must be alone
        for b in bins:
            if 0 in b:
                assert len(b) == 1

    def test_empty_input(self):
        bins = _pack_by_tokens([100, 200], [], max_tokens=500)
        assert bins == []

    def test_subset_indices(self):
        """Only packs the requested subset of indices."""
        token_counts = [100, 200, 50, 150]  # full batch
        indices = [1, 3]  # only pack indices 1 and 3
        bins = _pack_by_tokens(token_counts, indices, max_tokens=300)
        all_idx = sorted([i for b in bins for i in b])
        assert all_idx == [1, 3]

    def test_single_sample(self):
        bins = _pack_by_tokens([100], [0], max_tokens=200)
        assert bins == [[0]]


class TestTrimAndSlice:
    """Tests for per-microbatch trimming."""

    def _make_batch(self, n=8, prompt_len=20, comp_len=30):
        """Create a fake batch with variable actual lengths."""
        # Prompts: left-padded. Actual lengths vary from 5 to prompt_len.
        prompt_ids = torch.zeros(n, prompt_len, dtype=torch.long)
        prompt_mask = torch.zeros(n, prompt_len, dtype=torch.long)
        for i in range(n):
            actual = 5 + i * (prompt_len - 5) // max(n - 1, 1)
            prompt_ids[i, prompt_len - actual:] = torch.arange(1, actual + 1)
            prompt_mask[i, prompt_len - actual:] = 1

        # Completions: right-padded. Actual lengths vary from 3 to comp_len.
        comp_ids = torch.zeros(n, comp_len, dtype=torch.long)
        comp_mask = torch.zeros(n, comp_len, dtype=torch.long)
        old_logps = torch.randn(n, comp_len)
        for i in range(n):
            actual = 3 + i * (comp_len - 3) // max(n - 1, 1)
            comp_ids[i, :actual] = torch.arange(1, actual + 1)
            comp_mask[i, :actual] = 1

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": comp_ids,
            "completion_mask": comp_mask,
            "old_per_token_logps": old_logps,
            "advantages": torch.randn(n),
            "ref_per_token_logps": None,
            "sampling_per_token_logps": None,
        }

    def test_trimming_reduces_dimensions(self):
        """Trimming should produce smaller tensors than the global padding."""
        batch = self._make_batch(n=8, prompt_len=20, comp_len=30)
        # Select only the first 3 samples (shortest sequences)
        result = _trim_and_slice(batch, [0, 1, 2])

        assert result["completion_ids"].shape[0] == 3
        # Completion should be trimmed (shortest 3 samples have short completions)
        assert result["completion_ids"].shape[1] <= 30
        assert result["completion_ids"].shape[1] < 30  # strictly shorter
        # Prompt should be trimmed too
        assert result["prompt_ids"].shape[1] <= 20

    def test_trimming_preserves_real_tokens(self):
        """No real tokens should be lost during trimming."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        result = _trim_and_slice(batch, [0, 1, 2, 3])

        # All real tokens still present
        for i in range(4):
            orig_comp_len = batch["completion_mask"][i].sum().item()
            trimmed_comp_len = result["completion_mask"][i].sum().item()
            assert trimmed_comp_len == orig_comp_len

            orig_prompt_len = batch["prompt_mask"][i].sum().item()
            trimmed_prompt_len = result["prompt_mask"][i].sum().item()
            assert trimmed_prompt_len == orig_prompt_len

    def test_old_logps_trimmed_consistently(self):
        """old_per_token_logps should be trimmed to same length as completion."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=20)
        result = _trim_and_slice(batch, [0, 1])
        assert result["old_per_token_logps"].shape[1] == result["completion_ids"].shape[1]

    def test_advantages_sliced_not_trimmed(self):
        """1D tensors like advantages should be sliced by sample, not trimmed."""
        batch = self._make_batch(n=4)
        result = _trim_and_slice(batch, [1, 3])
        assert result["advantages"].shape == (2,)
        assert torch.equal(result["advantages"][0], batch["advantages"][1])
        assert torch.equal(result["advantages"][1], batch["advantages"][3])

    def test_none_values_preserved(self):
        batch = self._make_batch(n=4)
        result = _trim_and_slice(batch, [0, 1])
        assert result["ref_per_token_logps"] is None

    def test_full_batch_no_trim(self):
        """Selecting all samples: trim only removes trailing/leading padding."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        result = _trim_and_slice(batch, [0, 1, 2, 3])
        # The longest sample should match the trimmed length
        max_comp = batch["completion_mask"].sum(dim=1).max().item()
        assert result["completion_ids"].shape[1] == max_comp


class TestPackForForward:
    """Tests for padding-free packing."""

    def _make_batch(self, n=4, prompt_len=10, comp_len=15):
        """Create a fake batch with variable actual lengths."""
        prompt_ids = torch.zeros(n, prompt_len, dtype=torch.long)
        prompt_mask = torch.zeros(n, prompt_len, dtype=torch.long)
        comp_ids = torch.zeros(n, comp_len, dtype=torch.long)
        comp_mask = torch.zeros(n, comp_len, dtype=torch.long)
        old_logps = torch.randn(n, comp_len)

        for i in range(n):
            # Variable prompt lengths (left-padded)
            p_actual = 3 + i * 2  # 3, 5, 7, 9 for n=4
            prompt_ids[i, prompt_len - p_actual:] = torch.arange(1, p_actual + 1)
            prompt_mask[i, prompt_len - p_actual:] = 1

            # Variable completion lengths (right-padded)
            c_actual = 2 + i * 3  # 2, 5, 8, 11 for n=4
            comp_ids[i, :c_actual] = torch.arange(100, 100 + c_actual)
            comp_mask[i, :c_actual] = 1

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": comp_ids,
            "completion_mask": comp_mask,
            "old_per_token_logps": old_logps,
            "advantages": torch.randn(n),
            "ref_per_token_logps": None,
            "sampling_per_token_logps": None,
        }

    def test_packed_shape_is_flat(self):
        """Packed input_ids should be (1, total_real_tokens)."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [0, 1, 2, 3])

        # Total real tokens = sum of actual prompt + completion lengths
        total_prompt = batch["prompt_mask"].sum().item()
        total_comp = batch["completion_mask"].sum().item()
        assert packed["packed_input_ids"].shape == (1, total_prompt + total_comp)
        assert packed["packed_position_ids"].shape == packed["packed_input_ids"].shape

    def test_position_ids_reset_at_boundaries(self):
        """Position IDs should reset to 0 at each sequence boundary."""
        batch = self._make_batch(n=3, prompt_len=8, comp_len=10)
        packed = _pack_for_forward(batch, [0, 1, 2])
        pos = packed["packed_position_ids"][0]

        # Count how many times position resets to 0
        resets = (pos == 0).sum().item()
        assert resets == 3  # one per sequence

    def test_no_padding_in_packed(self):
        """Packed tensor should contain only real tokens, no pad zeros from padding."""
        batch = self._make_batch(n=2, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [0, 1])

        # Every token in the packed input should be nonzero (we use 1-based IDs in the test)
        assert (packed["packed_input_ids"][0] != 0).all()

    def test_completion_data_repadded_correctly(self):
        """Completion IDs and mask should be repadded to (N, max_comp_len)."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [1, 3])  # comp lens 5, 11

        assert packed["completion_ids"].shape == (2, 11)  # max comp len
        assert packed["completion_mask"].shape == (2, 11)
        # First seq has 5 real tokens
        assert packed["completion_mask"][0].sum().item() == 5
        # Second seq has 11 real tokens
        assert packed["completion_mask"][1].sum().item() == 11

    def test_old_logps_repadded(self):
        """Old logprobs should be repadded matching completion mask."""
        batch = self._make_batch(n=3, prompt_len=8, comp_len=12)
        packed = _pack_for_forward(batch, [0, 1, 2])

        assert packed["old_per_token_logps"] is not None
        assert packed["old_per_token_logps"].shape == packed["completion_mask"].shape

    def test_advantages_indexed(self):
        """Advantages should be indexed by the requested indices."""
        batch = self._make_batch(n=4)
        packed = _pack_for_forward(batch, [1, 3])

        assert packed["advantages"].shape == (2,)
        assert torch.equal(packed["advantages"][0], batch["advantages"][1])
        assert torch.equal(packed["advantages"][1], batch["advantages"][3])

    def test_seq_boundaries(self):
        """seq_boundaries should accurately reflect prompt/completion lengths."""
        batch = self._make_batch(n=4, prompt_len=10, comp_len=15)
        packed = _pack_for_forward(batch, [0, 2])

        assert len(packed["seq_boundaries"]) == 2
        p0, c0 = packed["seq_boundaries"][0]
        p1, c1 = packed["seq_boundaries"][1]
        assert p0 == batch["prompt_mask"][0].sum().item()
        assert c0 == batch["completion_mask"][0].sum().item()
        assert p1 == batch["prompt_mask"][2].sum().item()
        assert c1 == batch["completion_mask"][2].sum().item()
