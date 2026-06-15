"""Equivalence tests for PerSampleGradCapture (gradient_routing.py).

Validates that the per-sample gradient norms captured from a single packed
[1, T] backward pass match an independent ground truth, for both adapter types:

  1. Sum of independent per-sample grads == full-batch .grad (decomposition).
  2. capture.records[sample][layer][role] == norm of that sample's autograd
     gradient for that (layer, role), computed by an independent forward on
     just that sample's tokens.

A position-wise toy tower (no attention) is used so a packed forward is exactly
equivalent to per-sample forwards — isolating the capture math from any
sequence-packing/attention concerns.

Run: .venv/bin/python tests/test_per_sample_grad_capture.py
"""

import os
import sys
import math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (
    DualLoRALinear, DualMLPAdapter, _DUAL_ADAPTER_TYPES,
    PerSampleGradCapture, layer_role_param_map,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
RTOL, ATOL = 1e-4, 1e-5


def _randomize_adapters(model):
    """Give every adapter param a nonzero random value (B/down init to zero,
    which would make one of the two LoRA grads identically zero)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, _DUAL_ADAPTER_TYPES):
                for p in m.get_retain_params() + m.get_forget_params():
                    p.copy_(torch.randn_like(p) * 0.1)


def _grad_norms_from_dotgrad(model):
    """{layer: {role: L2 norm of current .grad over that role's params}}."""
    lrp = layer_role_param_map(model)
    out = {}
    for li, roles in lrp.items():
        out[li] = {}
        for role, params in roles.items():
            sq = sum(float(p.grad.pow(2).sum()) for p in params if p.grad is not None)
            out[li][role] = math.sqrt(sq)
    return out


def _zero_grads(model):
    for p in model.parameters():
        p.grad = None


class LoRATower(nn.Module):
    """Position-wise tower of DualLoRALinear layers (no attention)."""
    def __init__(self, n_layers, hidden, rank, forget_rank, alpha=16):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            base = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE, device=DEVICE)
            layers.append(DualLoRALinear(base, rank, forget_rank, alpha, dropout=0.0))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):  # x: [1, T, H]
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class _BaseMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden_size = hidden
        self.gate_proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE, device=DEVICE)
        self.up_proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE, device=DEVICE)
        self.down_proj = nn.Linear(hidden, hidden, bias=False, dtype=DTYPE, device=DEVICE)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MLPTower(nn.Module):
    """Position-wise tower of DualMLPAdapter blocks (no attention)."""
    def __init__(self, n_layers, hidden, retain_neurons, forget_neurons):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            base = _BaseMLP(hidden)
            layers.append(DualMLPAdapter(base, hidden, retain_neurons, forget_neurons))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


def _run_case(name, model, hidden, lengths):
    _randomize_adapters(model)
    T = sum(lengths)
    sample_ids = list(range(len(lengths)))
    seq_boundaries = [(1, L - 1) for L in lengths]  # (p_len, c_len); span = L

    torch.manual_seed(0)
    x = torch.randn(1, T, hidden, dtype=DTYPE, device=DEVICE)
    coeff = torch.randn(1, T, hidden, dtype=DTYPE, device=DEVICE)

    # --- Full packed backward with capture ---
    cap = PerSampleGradCapture(model)
    _zero_grads(model)
    cap.set_segments(seq_boundaries, sample_ids)
    loss = (model(x) * coeff).sum()
    loss.backward()
    cap.remove()
    captured = cap.records
    full_grad_norms = _grad_norms_from_dotgrad(model)

    # --- Independent per-sample ground truth ---
    offsets = []
    off = 0
    for L in lengths:
        offsets.append((off, off + L))
        off += L

    # (1) sum of per-sample grad tensors == full .grad
    lrp = layer_role_param_map(model)
    accum = {id(p): torch.zeros_like(p) for roles in lrp.values()
             for params in roles.values() for p in params}
    per_sample_norms = {}  # sid -> {layer: {role: norm}}
    for sid, (s, e) in enumerate(offsets):
        _zero_grads(model)
        loss_j = (model(x[:, s:e, :]) * coeff[:, s:e, :]).sum()
        loss_j.backward()
        per_sample_norms[sid] = _grad_norms_from_dotgrad(model)
        for roles in lrp.values():
            for params in roles.values():
                for p in params:
                    if p.grad is not None:
                        accum[id(p)] += p.grad.detach()

    # Recompute full .grad to compare against the summed per-sample grads.
    _zero_grads(model)
    loss = (model(x) * coeff).sum()
    loss.backward()
    max_decomp_err = 0.0
    for roles in lrp.values():
        for params in roles.values():
            for p in params:
                ref = p.grad.detach()
                err = (accum[id(p)] - ref).norm().item() / (ref.norm().item() + 1e-12)
                max_decomp_err = max(max_decomp_err, err)

    # (2) capture norms == independent per-sample autograd norms
    max_norm_err = 0.0
    for sid in sample_ids:
        for li in captured[sid]:
            for role in ("retain", "forget"):
                got = captured[sid][li][role]
                ref = per_sample_norms[sid][li][role]
                err = abs(got - ref) / (ref + 1e-12)
                max_norm_err = max(max_norm_err, err)

    print(f"  [{name}] max decomposition err = {max_decomp_err:.2e}, "
          f"max per-sample norm err = {max_norm_err:.2e}")
    assert max_decomp_err < 1e-4, f"{name}: per-sample grads do not sum to .grad"
    assert max_norm_err < 1e-4, f"{name}: capture norms != independent autograd norms"


def test_lora_capture():
    print("\n=== LoRA per-sample grad capture ===")
    torch.manual_seed(42)
    model = LoRATower(n_layers=3, hidden=32, rank=4, forget_rank=4)
    _run_case("LoRA", model, hidden=32, lengths=[3, 5, 2, 4])


def test_mlp_capture():
    print("\n=== MLP per-sample grad capture ===")
    torch.manual_seed(42)
    model = MLPTower(n_layers=3, hidden=32, retain_neurons=8, forget_neurons=8)
    _run_case("MLP", model, hidden=32, lengths=[3, 5, 2, 4])


def test_single_sequence_pack():
    """A single-sequence pack must reproduce that sample's standalone grad."""
    print("\n=== single-sequence pack ===")
    torch.manual_seed(1)
    model = LoRATower(n_layers=2, hidden=16, rank=2, forget_rank=2)
    _run_case("LoRA-single", model, hidden=16, lengths=[7])


def _act_rms_per_span(C, lengths):
    """Ground-truth RMS-over-tokens norm of contribution C ([T, dim]) per span."""
    out, off = [], 0
    for L in lengths:
        seg = C[off:off + L]
        out.append(float((seg.pow(2).sum() / L) ** 0.5))
        off += L
    return out


def test_activation_capture_lora():
    print("\n=== LoRA activation capture ===")
    torch.manual_seed(3)
    model = LoRATower(n_layers=1, hidden=24, rank=4, forget_rank=4)
    _randomize_adapters(model)
    lengths = [3, 5, 2]
    T = sum(lengths)
    x = torch.randn(1, T, 24, dtype=DTYPE, device=DEVICE)
    cap = PerSampleGradCapture(model)
    cap.set_segments([(1, L - 1) for L in lengths], list(range(len(lengths))))
    model(x)  # forward only — activation is a forward quantity
    cap.remove()
    act = cap.act_records
    m = model.layers[0]
    xf = x[0]
    ref = {}
    for role, A, B, c in (("retain", m.lora_A_retain, m.lora_B_retain, m.scaling * m.retain_scale),
                          ("forget", m.lora_A_forget, m.lora_B_forget, m.forget_scaling * m.forget_scale)):
        C = (xf @ A.t()) @ B.t() * c
        ref[role] = _act_rms_per_span(C, lengths)
    max_err = max(abs(act[sid][0][role] - ref[role][sid]) / (ref[role][sid] + 1e-12)
                  for sid in range(len(lengths)) for role in ("retain", "forget"))
    print(f"  [LoRA-act] max err = {max_err:.2e}")
    assert max_err < 1e-4


def test_activation_capture_mlp():
    print("\n=== MLP activation capture ===")
    torch.manual_seed(3)
    model = MLPTower(n_layers=1, hidden=24, retain_neurons=8, forget_neurons=8)
    _randomize_adapters(model)
    lengths = [3, 5, 2]
    T = sum(lengths)
    x = torch.randn(1, T, 24, dtype=DTYPE, device=DEVICE)
    cap = PerSampleGradCapture(model)
    cap.set_segments([(1, L - 1) for L in lengths], list(range(len(lengths))))
    model(x)
    cap.remove()
    act = cap.act_records
    m = model.layers[0]
    xf = x[0]
    ref = {}
    for role in ("retain", "forget"):
        gate = getattr(m, f"gate_{role}"); up = getattr(m, f"up_{role}"); down = getattr(m, f"down_{role}")
        scale = m.retain_scale if role == "retain" else m.forget_scale
        C = down(m.act(gate(xf)) * up(xf)) * scale
        ref[role] = _act_rms_per_span(C, lengths)
    max_err = max(abs(act[sid][0][role] - ref[role][sid]) / (ref[role][sid] + 1e-12)
                  for sid in range(len(lengths)) for role in ("retain", "forget"))
    print(f"  [MLP-act] max err = {max_err:.2e}")
    assert max_err < 1e-4


def test_hf_llama_integration():
    """Integration: capture hooks fire correctly inside a real LlamaForCausalLM
    (attention + grad-requiring activations, mixed q/v/gate/down adapters), the
    records are populated and finite, and the triangle inequality the driver's
    grad_check relies on holds: ||.grad|| <= sum_j ||grad_j|| per (layer, role).

    (Per-sample *equivalence* to standalone forwards is not asserted here: CPU
    eager attention has no varlen packing mask, so the capture's position-wise
    decomposition and a per-sample re-forward are different decompositions of
    .grad. The exact capture math is validated by the no-attention tower tests.)"""
    print("\n=== HF Llama integration ===")
    from transformers import LlamaConfig, LlamaForCausalLM
    from gradient_routing import apply_dual_lora

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).to(DEVICE).to(DTYPE)
    apply_dual_lora(model, rank=4, forget_rank=4, alpha=16,
                    projections=["q_proj", "v_proj", "gate_proj", "down_proj"])
    _randomize_adapters(model)

    lengths = [4, 6, 3]
    sample_ids = list(range(len(lengths)))
    seq_boundaries = [(1, L - 1) for L in lengths]
    T = sum(lengths)
    ids = torch.randint(0, 64, (1, T), device=DEVICE)
    pos = torch.cat([torch.arange(L, device=DEVICE) for L in lengths]).unsqueeze(0)
    coeff = torch.randn(1, T, 32, dtype=DTYPE, device=DEVICE)

    cap = PerSampleGradCapture(model)
    _zero_grads(model)
    cap.set_segments(seq_boundaries, sample_ids)
    (model.model(input_ids=ids, position_ids=pos,
                 use_cache=False).last_hidden_state * coeff).sum().backward()
    cap.remove()
    captured = cap.records

    assert set(captured.keys()) == set(sample_ids), "missing samples in records"
    for sid in sample_ids:
        for li, roles in captured[sid].items():
            for role in ("retain", "forget"):
                assert math.isfinite(roles[role]), f"non-finite norm s{sid} l{li} {role}"

    # Triangle inequality (the driver's grad_check tripwire) on a real model.
    full = _grad_norms_from_dotgrad(model)
    max_ratio = 0.0
    for li, roles in full.items():
        for role in ("retain", "forget"):
            sum_persample = sum(captured[sid].get(li, {}).get(role, 0.0)
                                for sid in sample_ids) + 1e-12
            max_ratio = max(max_ratio, roles[role] / sum_persample)
    print(f"  [HF-Llama] max triangle ratio (||.grad||/sum||grad_j||) = {max_ratio:.4f}")
    assert max_ratio < 1.0 + 1e-3, "triangle inequality violated on HF Llama"


if __name__ == "__main__":
    test_lora_capture()
    test_mlp_capture()
    test_single_sequence_pack()
    test_activation_capture_lora()
    test_activation_capture_mlp()
    test_hf_llama_integration()
    print("\nAll per-sample grad + activation capture tests passed.")
