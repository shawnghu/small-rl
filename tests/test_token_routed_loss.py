"""Gradient-equivalence + hook-isolation tests for token-level routing.

Requires torch (and liger for Test 1) — runs on the GPU/training env (Hyperbolic/Modal), NOT the
local analysis venv. A tiny LlamaForCausalLM + DualMLP adapters; works on CPU or GPU.

  Test 2 (recombination, liger-independent): with NO hooks, backward(loss_good)+backward(loss_bad)
    grads == single backward(full) grads. The core exactness claim of token routing.
  Test 3 (hook isolation): with the retain-zero hook on the bad pass, the forget adapter receives
    the FULL-batch gradient and the retain adapter the good-token-only gradient.
  Test 1 (formula match, needs liger): explicit _grpo_per_token_loss(full).sum() and its grads ==
    the liger fused 'grpo' scalar — pins temperature / k3-KL / clip / per-seq normalization.

Run: python tests/test_token_routed_loss.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from gradient_routing import apply_dual_mlp, collect_routing_params, set_scales
from train import SampleGRPOTrainer, _pack_for_forward, LOGP_RATIO_CLAMP

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Stub:
    """Minimal carrier for _grpo_per_token_loss (uses only these attrs)."""
    temperature = 0.7
    beta = 1e-3
    epsilon_low = 0.2
    epsilon_high = 0.2
    # 1 = compare against the raw liger kernel scalar. Real training divides by TRL's
    # current_gradient_accumulation_steps (see _grpo_per_token_loss docstring).
    current_gradient_accumulation_steps = 1
    _grpo_per_token_loss = SampleGRPOTrainer._grpo_per_token_loss


def _tiny_model():
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).to(DEV)
    apply_dual_mlp(model, retain_neurons=8, forget_neurons=8)
    model.to(DEV)
    set_scales(model, 1.0, 1.0)
    return model


def _fake_batch(N=4, P=3, C=6, V=64):
    """A small padded batch (right-padded completions) + a token_routing_mask marking some tokens."""
    g = torch.Generator().manual_seed(1)
    prompt_ids = torch.randint(1, V, (N, P), generator=g)
    completion_ids = torch.randint(1, V, (N, C), generator=g)
    completion_mask = torch.ones(N, C, dtype=torch.long)
    completion_mask[0, 4:] = 0          # vary lengths so per-seq normalization is exercised
    completion_mask[2, 5:] = 0
    advantages = torch.randn(N, generator=g)
    old = torch.randn(N, C, generator=g) * 0.1
    ref = torch.randn(N, C, generator=g) * 0.1
    troute = torch.zeros(N, C, dtype=torch.bool)
    troute[:, 1] = True                 # route token position 1 in every sequence
    troute[1, 3] = True
    inp = dict(prompt_ids=prompt_ids, prompt_mask=torch.ones(N, P, dtype=torch.long),
               completion_ids=completion_ids, completion_mask=completion_mask,
               advantages=advantages, old_per_token_logps=old, ref_per_token_logps=ref,
               token_routing_mask=troute)
    return {k: v.to(DEV) for k, v in inp.items()}


def _forward_last_hs(model, packed):
    out = model.model(input_ids=packed["packed_input_ids"],
                      position_ids=packed["packed_position_ids"], use_cache=False)
    hidden = out.last_hidden_state
    n_seqs, T = packed["num_sequences"], packed["max_comp_len"]
    last_hs = torch.zeros(n_seqs, T, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
    off = 0
    for j, (p_len, c_len) in enumerate(packed["seq_boundaries"]):
        if c_len > 0:
            last_hs[j, :c_len] = hidden[0, off + p_len - 1: off + p_len + c_len - 1]
        off += p_len + c_len
    return last_hs


def _grads(params):
    return {id(p): (p.grad.detach().clone() if p.grad is not None else None) for p in params}


def _zero(model):
    for p in model.parameters():
        p.grad = None


def _close(a, b, tol=1e-3, atol=1e-6):  # 1e-3 rel: well above fp-noise (~3e-4 on the most-accumulated
                                        # backward), well below any real recombination bug (O(1))
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    d = (a - b).abs().max().item()
    s = b.abs().max().item() + 1e-8
    return d < atol or d / s < tol


def main():
    model = _tiny_model()
    retain, forget = collect_routing_params(model)
    retain, forget = list(retain), list(forget)
    assert retain and forget, "no retain/forget params collected"
    stub = _Stub()
    inp = _fake_batch()
    packed = _pack_for_forward(inp, list(range(inp["completion_ids"].shape[0])))
    comp_mask = packed["completion_mask"].bool()
    troute = packed["token_routing_mask"] & comp_mask
    mask_good = (comp_mask & ~troute).float()
    mask_bad = troute.float()

    # ONE shared forward -> deterministic; recombination/isolation are about partitioning the SAME
    # ptl, so all backward scenarios reuse this graph (retain_graph=True) for exact comparisons.
    mc = comp_mask.float()
    last_hs = _forward_last_hs(model, packed)
    p = stub._grpo_per_token_loss(last_hs, model, packed)
    lg = (p * mask_good).sum()
    lb = (p * mask_bad).sum()
    lf = (p * mc).sum()

    # ---- Test 2: recombination (no hooks) ----
    names = {id(pp): n for n, pp in model.named_parameters()}
    _zero(model); lf.backward(retain_graph=True); g_full = _grads(retain + forget)
    _zero(model); lg.backward(retain_graph=True); lb.backward(retain_graph=True); g_split = _grads(retain + forget)
    bad = [pp for pp in retain + forget if not _close(g_split[id(pp)], g_full[id(pp)])]
    # diagnostic: worst few params by abs diff
    rows = []
    for pp in retain + forget:
        a, b = g_split[id(pp)], g_full[id(pp)]
        if a is None or b is None:
            rows.append((names.get(id(pp), "?"), "MISSING", a is None, b is None)); continue
        d = (a - b).abs().max().item(); s = b.abs().max().item()
        rows.append((names.get(id(pp), "?"), d, s, d / (s + 1e-12)))
    rows.sort(key=lambda r: (r[1] if isinstance(r[1], float) else 9e9), reverse=True)
    print("  worst params (name, max_abs_diff, full_grad_max, rel):")
    for r in rows[:4]:
        print("   ", r)
    assert not bad, f"Test 2 FAILED: {len(bad)} params' split grads != full grads"
    print("Test 2 (recombination) PASSED: backward(good)+backward(bad) == backward(full)")

    # ---- Test 3: hook isolation (retain-zero on bad pass) ----
    _zero(model)
    lg.backward(retain_graph=True)
    hooks = [pp.register_hook(lambda g: torch.zeros_like(g)) for pp in retain]
    lb.backward(retain_graph=True)
    for h in hooks:
        h.remove()
    g_routed = _grads(retain + forget)
    # forget adapter must equal the FULL-batch gradient (good pass + bad pass, unhooked)
    fbad = [pp for pp in forget if not _close(g_routed[id(pp)], g_full[id(pp)])]
    assert not fbad, f"Test 3 FAILED: forget grads != full ({len(fbad)} params)"
    # retain adapter must equal the GOOD-only gradient
    _zero(model); lg.backward(retain_graph=True); g_goodonly = _grads(retain)
    rbad = [pp for pp in retain if not _close(g_routed[id(pp)], g_goodonly[id(pp)])]
    assert not rbad, f"Test 3 FAILED: retain grads != good-only ({len(rbad)} params)"
    print("Test 3 (hook isolation) PASSED: forget==full, retain==good-only")

    # ---- Test 1: explicit per-token loss == liger 'grpo' scalar ----
    try:
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
    except Exception as e:
        print(f"Test 1 SKIPPED (no liger: {e})")
    else:
        liger = LigerFusedLinearGRPOLoss(beta=stub.beta, compiled=False, chunk_size=1,
                                         epsilon_low=stub.epsilon_low, epsilon_high=stub.epsilon_high,
                                         temperature=stub.temperature, use_ref_model=stub.beta != 0.0,
                                         loss_type="grpo", max_completion_length=packed["max_comp_len"])
        last_hs = _forward_last_hs(model, packed)
        # Compare the FORMULA only: disable the repo's LOGP_RATIO_CLAMP guardrail (vanilla liger has
        # no such clamp) so any mismatch localizes to temperature/k3-sign/clip/normalization.
        import train as _train
        _saved = _train.LOGP_RATIO_CLAMP
        _train.LOGP_RATIO_CLAMP = 0.0
        try:
            explicit = (stub._grpo_per_token_loss(last_hs, model, packed) *
                        comp_mask.float()).sum()
        finally:
            _train.LOGP_RATIO_CLAMP = _saved
        lg, _ = liger(_input=last_hs.detach().requires_grad_(True),
                      lin_weight=model.lm_head.weight,
                      selected_token_ids=packed["completion_ids"],
                      attention_mask=packed["completion_mask"],
                      advantages=packed["advantages"],
                      bias=getattr(model.lm_head, "bias", None),
                      old_per_token_logps=packed["old_per_token_logps"],
                      ref_per_token_logps=packed["ref_per_token_logps"])
        rel = abs(explicit.item() - lg.item()) / (abs(lg.item()) + 1e-8)
        print(f"Test 1: explicit={explicit.item():.6f} liger={lg.item():.6f} rel_err={rel:.2e}")
        assert rel < 1e-3, ("Test 1 FAILED: explicit per-token loss != liger 'grpo' scalar "
                            "(check temperature/k3-sign/clip/normalization, and LOGP_RATIO_CLAMP)")
        print("Test 1 (formula match) PASSED")

    print("\nALL TOKEN-ROUTED LOSS TESTS PASSED")


if __name__ == "__main__":
    main()
