"""Vectorized _pack_for_forward must be exactly equivalent to the original
per-sequence loop (kept verbatim below as the reference implementation).

Covers: left-padded prompts, right-padded completions, junk values in padded
regions (the vectorized version must zero them like the loop's zeros-filled
buffers did), empty completions, with/without old/ref logps, index subsets in
arbitrary order.

Run: .venv/bin/python -m pytest tests/test_pack_for_forward_equivalence.py -q
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import _pack_for_forward


def _pack_for_forward_reference(inputs, indices):
    """Original loop implementation (pre-vectorization), verbatim."""
    device = next(v.device for v in inputs.values() if isinstance(v, torch.Tensor))

    all_input_ids = []
    all_position_ids = []
    all_completion_mask = []
    seq_boundaries = []

    has_old_logps = "old_per_token_logps" in inputs and inputs["old_per_token_logps"] is not None
    has_ref_logps = "ref_per_token_logps" in inputs and inputs["ref_per_token_logps"] is not None
    all_old_logps = [] if has_old_logps else None
    all_ref_logps = [] if has_ref_logps else None
    all_comp_ids = []

    for i in indices:
        p_mask = inputs["prompt_mask"][i]
        real_positions = p_mask.nonzero(as_tuple=True)[0]
        if len(real_positions) > 0:
            p_start = real_positions[0].item()
            p_real = inputs["prompt_ids"][i, p_start:]
        else:
            p_real = inputs["prompt_ids"][i, :0]
        p_len = p_real.shape[0]

        c_mask = inputs["completion_mask"][i]
        c_len = c_mask.sum().item()
        c_real = inputs["completion_ids"][i, :c_len]

        seq_ids = torch.cat([p_real, c_real])
        seq_len = seq_ids.shape[0]

        all_input_ids.append(seq_ids)
        all_position_ids.append(torch.arange(seq_len, device=device))
        all_completion_mask.append(torch.cat([
            torch.zeros(p_len, dtype=torch.long, device=device),
            torch.ones(c_len, dtype=torch.long, device=device),
        ]))
        all_comp_ids.append(inputs["completion_ids"][i, :c_len])

        if has_old_logps:
            all_old_logps.append(inputs["old_per_token_logps"][i, :c_len])
        if has_ref_logps:
            all_ref_logps.append(inputs["ref_per_token_logps"][i, :c_len])

        seq_boundaries.append((p_len, c_len))

    packed_input_ids = torch.cat(all_input_ids).unsqueeze(0)
    packed_position_ids = torch.cat(all_position_ids).unsqueeze(0)
    packed_completion_mask = torch.cat(all_completion_mask).unsqueeze(0)

    max_comp_len = max(c for _, c in seq_boundaries) if seq_boundaries else 0
    n_seqs = len(indices)

    comp_ids_padded = torch.zeros(n_seqs, max_comp_len, dtype=torch.long, device=device)
    comp_mask_padded = torch.zeros(n_seqs, max_comp_len, dtype=torch.long, device=device)
    old_logps_padded = torch.zeros(n_seqs, max_comp_len, device=device) if has_old_logps else None
    ref_logps_padded = torch.zeros(n_seqs, max_comp_len, device=device) if has_ref_logps else None

    for j, (_, c_len) in enumerate(seq_boundaries):
        if c_len > 0:
            comp_ids_padded[j, :c_len] = all_comp_ids[j]
            comp_mask_padded[j, :c_len] = 1
            if has_old_logps:
                old_logps_padded[j, :c_len] = all_old_logps[j]
            if has_ref_logps:
                ref_logps_padded[j, :c_len] = all_ref_logps[j]

    idx_t = torch.tensor(indices, device=device, dtype=torch.long)
    advantages = inputs["advantages"][idx_t]

    return {
        "packed_input_ids": packed_input_ids,
        "packed_position_ids": packed_position_ids,
        "packed_completion_mask": packed_completion_mask,
        "seq_boundaries": seq_boundaries,
        "completion_ids": comp_ids_padded,
        "completion_mask": comp_mask_padded,
        "advantages": advantages,
        "old_per_token_logps": old_logps_padded,
        "ref_per_token_logps": ref_logps_padded,
        "num_sequences": n_seqs,
        "max_comp_len": max_comp_len,
    }


def _make_inputs(g, n=24, P=32, C=40, with_old=True, with_ref=True,
                 min_c_len=0):
    """Random batch: left-padded prompts, right-padded completions, junk
    (nonzero values) everywhere the masks are 0."""
    prompt_ids = torch.randint(2, 5000, (n, P), generator=g)
    prompt_mask = torch.zeros(n, P, dtype=torch.long)
    completion_ids = torch.randint(2, 5000, (n, C), generator=g)
    completion_mask = torch.zeros(n, C, dtype=torch.long)
    for i in range(n):
        p_len = int(torch.randint(1, P + 1, (1,), generator=g))
        c_len = int(torch.randint(min_c_len, C + 1, (1,), generator=g))
        prompt_mask[i, P - p_len:] = 1
        completion_mask[i, :c_len] = 1
    inputs = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": torch.randn(n, generator=g),
    }
    if with_old:
        inputs["old_per_token_logps"] = torch.randn(n, C, generator=g)
    if with_ref:
        inputs["ref_per_token_logps"] = torch.randn(n, C, generator=g)
    return inputs


def _assert_equal(ref, new):
    assert set(ref) == set(new), (set(ref) ^ set(new))
    for k in ref:
        r, v = ref[k], new[k]
        if isinstance(r, torch.Tensor):
            assert isinstance(v, torch.Tensor), k
            assert r.shape == v.shape, (k, r.shape, v.shape)
            assert r.dtype == v.dtype, (k, r.dtype, v.dtype)
            assert torch.equal(r, v), (k, (r != v).float().mean())
        elif k == "seq_boundaries":
            assert [tuple(x) for x in r] == [tuple(x) for x in v], k
        else:
            assert r == v, (k, r, v)


def test_equivalence_full_batch():
    g = torch.Generator().manual_seed(0)
    inputs = _make_inputs(g)
    indices = list(range(24))
    _assert_equal(_pack_for_forward_reference(inputs, indices),
                  _pack_for_forward(inputs, indices))


def test_equivalence_subset_shuffled():
    g = torch.Generator().manual_seed(1)
    inputs = _make_inputs(g)
    indices = [17, 3, 0, 21, 9, 9, 4]   # duplicates + arbitrary order
    _assert_equal(_pack_for_forward_reference(inputs, indices),
                  _pack_for_forward(inputs, indices))


def test_equivalence_empty_completions():
    g = torch.Generator().manual_seed(2)
    inputs = _make_inputs(g, min_c_len=0)
    # force several fully-empty completions
    inputs["completion_mask"][:5] = 0
    indices = list(range(24))
    _assert_equal(_pack_for_forward_reference(inputs, indices),
                  _pack_for_forward(inputs, indices))


def test_equivalence_no_logps():
    g = torch.Generator().manual_seed(3)
    inputs = _make_inputs(g, with_old=False, with_ref=False)
    indices = list(range(0, 24, 2))
    _assert_equal(_pack_for_forward_reference(inputs, indices),
                  _pack_for_forward(inputs, indices))


def test_equivalence_old_logps_only():
    g = torch.Generator().manual_seed(4)
    inputs = _make_inputs(g, with_old=True, with_ref=False)
    inputs["ref_per_token_logps"] = None   # present-but-None path
    indices = list(range(5, 20))
    _assert_equal(_pack_for_forward_reference(inputs, indices),
                  _pack_for_forward(inputs, indices))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
