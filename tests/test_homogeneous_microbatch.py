"""Tests for homogeneous microbatch sorting in gradient routing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest


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
