"""Unit test for the frozen adapter-view mechanism (one-step off-policy core).

Verifies make_frozen_adapter_view / sync_adapter_snapshot:
  - the view's forward matches the live model at snapshot time;
  - the frozen base storage is SHARED (no duplication); adapter storage is NOT;
  - mutating the live adapters leaves the view (snapshot) unchanged;
  - re-syncing refreshes the snapshot.

CPU-only, tiny Llama. Run: .venv/bin/python tests/test_frozen_adapter_view.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gradient_routing import (apply_dual_mlp, apply_dual_lora,
                              make_frozen_adapter_view, sync_adapter_snapshot)


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


def _run(apply_fn, label):
    torch.manual_seed(0)
    model = _tiny_model()
    apply_fn(model)
    model.eval()
    ids = torch.randint(0, 128, (2, 8))

    view = make_frozen_adapter_view(model)
    sync_adapter_snapshot(model, view)

    # 1. view matches live at snapshot time
    l_live0 = _logits(model, ids)
    l_view0 = _logits(view, ids)
    assert torch.allclose(l_live0, l_view0, atol=1e-6), f"[{label}] view != live at snapshot"

    # 2. base storage shared; adapter storage independent
    live_named = dict(model.named_parameters())
    view_named = dict(view.named_parameters())
    base_shared = adapter_independent = False
    for name, lp in live_named.items():
        vp = view_named[name]
        if lp.requires_grad:  # adapter param
            assert vp.data_ptr() != lp.data_ptr(), f"[{label}] adapter {name} shares storage!"
            adapter_independent = True
        else:                 # base param
            assert vp.data_ptr() == lp.data_ptr(), f"[{label}] base {name} not shared!"
            base_shared = True
    assert base_shared and adapter_independent, f"[{label}] missing base/adapter params"

    # 3. mutate a live adapter param -> view (snapshot) must NOT change
    with torch.no_grad():
        for name, lp in live_named.items():
            if lp.requires_grad:
                lp.add_(0.5)  # perturb every adapter param
    l_live1 = _logits(model, ids)
    l_view1 = _logits(view, ids)
    assert not torch.allclose(l_live1, l_live0, atol=1e-4), f"[{label}] live didn't change after mutation"
    assert torch.allclose(l_view1, l_view0, atol=1e-6), f"[{label}] view changed when live mutated (NOT independent!)"

    # 4. re-sync -> view now matches the mutated live
    sync_adapter_snapshot(model, view)
    l_view2 = _logits(view, ids)
    assert torch.allclose(l_view2, l_live1, atol=1e-6), f"[{label}] view != live after re-sync"

    print(f"[{label}] PASS: snapshot-at-sync, base-shared, adapter-independent, "
          f"mutation-isolated, re-sync correct")


def main():
    _run(lambda m: apply_dual_mlp(m, retain_neurons=8, forget_neurons=8), "DualMLP")
    _run(lambda m: apply_dual_lora(m, rank=4, forget_rank=4, alpha=8, dropout=0.0), "DualLoRA")
    print("ALL PASS")


if __name__ == "__main__":
    main()
