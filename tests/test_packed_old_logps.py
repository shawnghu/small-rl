"""GPU test for packed old_logps (one-step off-policy increment 2).

Verifies _packed_per_token_logps:
  - matches between live model and a synced frozen view (view == live at snapshot);
  - the view holds the snapshot (mutating live leaves the view's old_logps unchanged);
  - the packed extraction is CORRECT: for a single unpadded sequence it matches a
    plain padded forward's gathered completion logps.

Needs a GPU + flash-attn (packed kernel). Run:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_packed_old_logps.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM

from gradient_routing import apply_dual_mlp, make_frozen_adapter_view, sync_adapter_snapshot
from train import _packed_per_token_logps, _packed_forward_logps, _pack_for_forward

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEV = "cuda"


def _fake_batch(n=8, P=6, C=8, vocab=49152, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt_ids = torch.randint(0, vocab, (n, P), generator=g)
    completion_ids = torch.randint(0, vocab, (n, C), generator=g)
    prompt_mask = torch.zeros(n, P, dtype=torch.long)
    completion_mask = torch.zeros(n, C, dtype=torch.long)
    for i in range(n):
        plen = 2 + (i % 3)            # 2..4 real prompt tokens, LEFT-padded
        clen = 3 + (i % 4)            # 3..6 real completion tokens, RIGHT-padded
        prompt_mask[i, P - plen:] = 1
        completion_mask[i, :clen] = 1
    return {
        "prompt_ids": prompt_ids.to(DEV),
        "prompt_mask": prompt_mask.to(DEV),
        "completion_ids": completion_ids.to(DEV),
        "completion_mask": completion_mask.to(DEV),
    }


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(DEV)
    apply_dual_mlp(model, retain_neurons=16, forget_neurons=16)
    # Give the adapters non-zero weight so they actually affect logps (down-proj
    # inits to zero, so the adapter is initially a no-op).
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.02)
    model.eval()

    view = make_frozen_adapter_view(model)
    sync_adapter_snapshot(model, view)

    inputs = _fake_batch()
    mask = inputs["completion_mask"].bool()
    max_tok = 30  # small -> multiple bins, exercises binning+scatter

    old_live = _packed_per_token_logps(model, inputs, max_tok)
    old_view = _packed_per_token_logps(view, inputs, max_tok)

    assert old_live.shape == inputs["completion_mask"].shape, "shape mismatch"
    assert torch.isfinite(old_live[mask]).all(), "non-finite logps"
    assert torch.allclose(old_live[mask], old_view[mask], atol=1e-3), \
        f"view != live at snapshot (max diff {(old_live - old_view)[mask].abs().max():.4f})"
    print("[1] view matches live at snapshot  OK")

    # Mutate live adapters -> view (snapshot) old_logps must NOT change.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    old_live2 = _packed_per_token_logps(model, inputs, max_tok)
    old_view2 = _packed_per_token_logps(view, inputs, max_tok)
    assert not torch.allclose(old_live2[mask], old_live[mask], atol=1e-2), "live didn't change"
    assert torch.allclose(old_view2[mask], old_view[mask], atol=1e-4), \
        "view changed when live mutated (snapshot not isolated!)"
    print("[2] mutating live leaves view snapshot unchanged  OK")

    # Correctness: single unpadded sequence, packed vs plain padded forward.
    single = {k: v[2:3] for k, v in inputs.items()}  # seq 2: plen=4, clen=5
    packed_lp = _packed_per_token_logps(model, single, max_tokens=10_000)[0]  # (C,)
    # plain forward on the real tokens only
    plen = int(single["prompt_mask"][0].sum()); clen = int(single["completion_mask"][0].sum())
    p_real = single["prompt_ids"][0][single["prompt_mask"][0].bool()]
    c_real = single["completion_ids"][0][single["completion_mask"][0].bool()]
    seq = torch.cat([p_real, c_real]).unsqueeze(0)  # (1, plen+clen)
    with torch.no_grad():
        logits = model(seq).logits[0].float()       # (plen+clen, V)
    lp = torch.log_softmax(logits, dim=-1)
    # completion token t (abs pos plen+t) predicted by hidden at plen+t-1
    ref = lp[plen - 1: plen - 1 + clen].gather(-1, c_real.unsqueeze(-1)).squeeze(-1)
    got = packed_lp[:clen]
    maxdiff = (got - ref).abs().max().item()
    assert maxdiff < 0.05, f"packed vs padded mismatch (max diff {maxdiff:.4f})"
    print(f"[3] packed == padded for a single seq (max diff {maxdiff:.4f})  OK")

    print("ALL PASS")


if __name__ == "__main__":
    main()
