"""Partition-invariance test for the microbatch loss-reduction scale.

Validates the scale factor that `_dynamic_microbatch_forward_backward` /
`_fused_forward_backward` apply to each microbatch's liger loss, against the
REAL liger GRPO loss (CPU, float64). This is the piece that makes the DAPO
(`loss_type="dapo"`) token-level normalization correct under our microbatching.

The claim being guarded:

    Summing  liger_loss(mb) * scale(mb)  over ANY partition of the opt batch
    equals the single-shot liger loss over the whole batch — i.e. the result is
    invariant to how `_pack_by_tokens` happens to split the batch — where

        scale(mb) = n_mb / scale_denom          for loss_type="grpo"
        scale(mb) = tok_mb / tok_denom          for loss_type="dapo"

    n_mb / tok_mb are the microbatch's sequence count / COMPLETION-token count,
    and scale_denom / tok_denom the same over the denominator population.

Why this works: liger normalizes each call internally by `full_attention_mask`
— /n_mb for grpo (per-sequence mean over the mb), /tok_mb for dapo (per-token
sum / mb token count). The scale numerator cancels that internal divisor,
leaving a single global denominator; per-sequence and per-token sums are each
additive across a partition, so the split-sum reproduces the single-shot loss.

This isolates the normalization/scale identity. The orthogonal per-token routing
decouple (fused vs homogeneous) is covered by test_fused_routing_equivalence.py;
the GPU end-to-end is bench_fused_gr.py.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss  # noqa: E402


def _make_batch(N=6, L=6, H=8, V=16, seed=0):
    torch.manual_seed(seed)
    hs = torch.randn(N, L, H, dtype=torch.float64)
    W = torch.randn(V, H, dtype=torch.float64)
    ids = torch.randint(0, V, (N, L))
    # Ragged completion lengths so completion-token counts differ per sequence —
    # this is exactly where the dapo (token) and grpo (sequence) scales diverge.
    lens = [6, 3, 5, 2, 6, 1][:N]
    mask = torch.zeros(N, L)
    for i, l in enumerate(lens):
        mask[i, :l] = 1.0
    adv = torch.randn(N, dtype=torch.float64)
    old = torch.randn(N, L, dtype=torch.float64)
    comp = [int(mask[i].sum().item()) for i in range(N)]
    return dict(hs=hs, W=W, ids=ids, mask=mask, adv=adv, old=old, comp=comp, N=N)


def _grad_under_partition(loss_type, partition, batch):
    """Accumulate W.grad over a partition using train.py's per-mb scale rule.
    partition=None => single shot over the whole batch. Mirrors the formula in
    train.py:_dynamic_microbatch_forward_backward exactly."""
    lf = LigerFusedLinearGRPOLoss(beta=0.0, compiled=False, chunk_size=1,
                                  loss_type=loss_type, use_ref_model=False)
    W = batch["W"].clone().requires_grad_(True)
    comp, N = batch["comp"], batch["N"]
    scale_denom = N
    tok_denom = sum(comp)

    def loss_of(idx):
        l, _ = lf(_input=batch["hs"][idx].clone(), lin_weight=W,
                  selected_token_ids=batch["ids"][idx],
                  attention_mask=batch["mask"][idx], advantages=batch["adv"][idx],
                  old_per_token_logps=batch["old"][idx], ref_per_token_logps=None)
        return l

    if partition is None:
        loss_of(list(range(N))).backward()
    else:
        seen = sorted(i for part in partition for i in part)
        assert seen == list(range(N)), "partition must cover every sequence once"
        for part in partition:
            l = loss_of(part)
            if loss_type == "dapo":
                scale = sum(comp[i] for i in part) / tok_denom
            else:
                scale = len(part) / scale_denom
            (l * scale).backward()
    return W.grad.clone()


def test_scale_is_partition_invariant():
    """For both loss types, the accumulated gradient is the same regardless of
    how the batch is split into microbatches (and equals the single-shot grad)."""
    batch = _make_batch()
    partitions = [
        None,                      # single shot (reference)
        [[0, 1, 2], [3, 4, 5]],    # even-ish 2-way
        [[0], [1, 2, 3, 4], [5]],  # lopsided 3-way
        [[i] for i in range(batch["N"])],  # fully split (one seq per mb)
    ]
    TOL = 1e-7
    for loss_type in ("grpo", "dapo"):
        ref = _grad_under_partition(loss_type, None, batch)
        for part in partitions:
            g = _grad_under_partition(loss_type, part, batch)
            d = (g - ref).abs().max().item()
            assert d < TOL, f"{loss_type} partition {part}: max|Δgrad|={d:.2e} >= {TOL}"


def test_grpo_and_dapo_differ_on_ragged_lengths():
    """Sanity: the two normalizations are genuinely different objects — dapo
    weights long completions more (token-proportional) than grpo (per-sequence).
    Guards against a refactor silently collapsing dapo back into grpo."""
    batch = _make_batch()
    g_grpo = _grad_under_partition("grpo", None, batch)
    g_dapo = _grad_under_partition("dapo", None, batch)
    assert (g_grpo - g_dapo).abs().max().item() > 1e-4, (
        "grpo and dapo produced ~identical gradients on ragged lengths — "
        "the token-level normalization is not taking effect."
    )


if __name__ == "__main__":
    test_scale_is_partition_invariant()
    test_grpo_and_dapo_differ_on_ragged_lengths()
    print("OK: dapo/grpo scale partition-invariance + distinctness")
