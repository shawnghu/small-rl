"""2-GPU test for the DISAGGREGATED frozen adapter view (OSP rollout GPU != update GPU).

make_frozen_adapter_view(model, device=cuda:B) must place an INDEPENDENT full copy
(base + adapters) on the rollout GPU — the frozen base is NOT re-tied to the live
model's storage (can't be, cross-GPU) — and sync_adapter_snapshot must cross-GPU copy
the live adapters into it. Verifies:
  - view forward on GPU B == live forward on GPU A at snapshot time;
  - the view lives entirely on GPU B (base + adapters), independent of GPU A;
  - mutating live adapters (GPU A) + re-sync refreshes the view (GPU B).

Requires >= 2 CUDA devices. Run: .venv/bin/python tests/test_frozen_view_disaggregated.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gradient_routing import (apply_dual_mlp, make_frozen_adapter_view,
                              sync_adapter_snapshot)


def _tiny_model():
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def _logits(model, ids):
    model.eval()
    with torch.no_grad():
        return model(ids).logits


def test_disaggregated_view():
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
        try:
            import pytest
            pytest.skip("needs >= 2 CUDA devices")
        except ImportError:
            print("SKIP: needs >= 2 CUDA devices")
            return
    A, B = torch.device("cuda:0"), torch.device("cuda:1")

    torch.manual_seed(0)
    model = _tiny_model()
    apply_dual_mlp(model, retain_neurons=8, forget_neurons=8)
    # Perturb adapters so they contribute (down-proj is zero-init).
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)
    model = model.to(A).eval()

    # View on the ROLLOUT GPU (B).
    view = make_frozen_adapter_view(model, device=B)
    sync_adapter_snapshot(model, view)

    ids_A = torch.randint(0, 128, (2, 8), device=A)
    ids_B = ids_A.to(B)

    # 1. view (B) matches live (A) at snapshot time.
    l_live = _logits(model, ids_A).cpu()
    l_view = _logits(view, ids_B).cpu()
    assert torch.allclose(l_live, l_view, atol=1e-5), \
        f"view(B) != live(A): max diff {(l_live - l_view).abs().max():.2e}"

    # 2. the ENTIRE view is on B, and independent of A (no shared storage).
    live_named = dict(model.named_parameters())
    for name, vp in view.named_parameters():
        assert vp.device.type == "cuda" and vp.device.index == 1, \
            f"view param {name} not on cuda:1 (on {vp.device})"
        # different device => necessarily different storage; assert loudly anyway
        assert vp.data_ptr() != live_named[name].data_ptr(), \
            f"view param {name} shares storage with live (impossible cross-GPU?!)"

    # 3. mutate live adapters (A), re-sync (cross-GPU), view (B) reflects it.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    # before re-sync, the view is stale (snapshot) -> should now DIFFER
    l_live_new = _logits(model, ids_A).cpu()
    l_view_stale = _logits(view, ids_B).cpu()
    assert not torch.allclose(l_live_new, l_view_stale, atol=1e-4), \
        "view tracked live without re-sync (base wrongly shared?)"
    sync_adapter_snapshot(model, view)
    l_view_fresh = _logits(view, ids_B).cpu()
    assert torch.allclose(l_live_new, l_view_fresh, atol=1e-5), \
        f"view(B) != live(A) after re-sync: max diff {(l_live_new - l_view_fresh).abs().max():.2e}"


if __name__ == "__main__":
    test_disaggregated_view()
    print("ok: disaggregated frozen view (rollout GPU != update GPU) works")
