"""RoutedAdam correctness tests.

  Test 1 (reference equality): when the routed stream == the full stream, RoutedAdam must match
          torch.optim.AdamW exactly (same math, kappa=1), including under a global clip factor.
  Test 2 (stream split): per-param update equals -lr * m_hat(routed)/(sqrt(v_hat(full))+eps) with
          m from the routed stream only and v from the full gradient — checked by hand-rolled math.
  Test 3 (integration, the one that would have caught the /64 bug): drive the REAL
          SampleGRPOTrainer._token_routed_forward_backward with _routed_adam=True via a stub and
          verify p.grad == g_good + g_bad (full stream) and _routed_m_stream == the per-adapter
          routed component, against torch.autograd.grad ground truth, accumulated over 2
          microbatches.
  Test 4 (amplification sim): on synthetic sparse-but-consistent behavior gradients, naive
          per-adapter AdamW moves the behavior param >30x faster than RoutedAdam (which moves it
          at the reference rate). This is the mechanism that caused the em-dash runaway.

Run: CUDA_VISIBLE_DEVICES=... .venv/bin/python tests/test_routed_adam.py  (CPU also fine)
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from routed_adam import RoutedAdam
from gradient_routing import apply_dual_mlp, set_scales
from train import SampleGRPOTrainer, _pack_for_forward

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def _adamw_reference(p0, grads, lr, betas, eps, wd, steps):
    """torch.optim.AdamW trajectory on a cloned param fed the given grads."""
    p = p0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([p], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for g in grads[:steps]:
        p.grad = g.clone()
        opt.step()
        opt.zero_grad()
    return p.detach()


def test1_reference_equality():
    lr, betas, eps, wd = 3e-3, (0.9, 0.999), 1e-8, 0.01
    p0 = torch.randn(64, 32, device=DEV)
    grads = [torch.randn_like(p0) for _ in range(7)]
    # clip factor on steps 3+ (simulating HF clip): p.grad scaled by c, stream unclipped
    c = 0.37

    p = p0.clone().requires_grad_(True)
    opt = RoutedAdam([{"params": [p], "lr": lr, "weight_decay": wd, "kappa": 1.0}],
                     lr=lr, betas=betas, eps=eps, weight_decay=wd)
    for i, g in enumerate(grads):
        factor = c if i >= 3 else 1.0
        p.grad = g.clone() * factor              # post-clip full grad
        p._routed_m_stream = g.clone().float()   # unclipped routed stream (== full here)
        opt._clip_factor = factor
        opt.step()
        p.grad = None

    ref = _adamw_reference(p0, [g * (c if i >= 3 else 1.0) for i, g in enumerate(grads)],
                           lr, betas, eps, wd, len(grads))
    err = (p.detach() - ref).abs().max().item()
    assert err < 1e-6, f"Test 1 FAILED: max abs diff vs AdamW = {err}"
    print(f"Test 1 (reference equality incl. clip) PASSED: max_abs_diff={err:.2e}")


def test2_stream_split():
    lr, betas, eps = 1e-3, (0.9, 0.999), 1e-8
    p0 = torch.randn(16, 8, device=DEV)
    p = p0.clone().requires_grad_(True)
    opt = RoutedAdam([{"params": [p], "lr": lr, "weight_decay": 0.0, "kappa": 1.0}],
                     lr=lr, betas=betas, eps=eps)
    m = torch.zeros_like(p0)
    v = torch.zeros_like(p0)
    cur = p0.clone()
    for t in range(1, 6):
        g_routed = torch.randn_like(p0) * 0.01
        g_other = torch.randn_like(p0)
        g_full = g_routed + g_other
        p.grad = g_full.clone()
        p._routed_m_stream = g_routed.clone().float()
        opt.step()
        p.grad = None
        # hand-rolled spec
        m = betas[0] * m + (1 - betas[0]) * g_routed
        v = betas[1] * v + (1 - betas[1]) * g_full ** 2
        m_hat = m / (1 - betas[0] ** t)
        v_hat = v / (1 - betas[1] ** t)
        cur = cur - lr * m_hat / (v_hat.sqrt() + eps)
    err = (p.detach() - cur).abs().max().item()
    assert err < 1e-6, f"Test 2 FAILED: max abs diff vs spec = {err}"
    print(f"Test 2 (routed m / full v split) PASSED: max_abs_diff={err:.2e}")


class _FakeAccelerator:
    device = torch.device(DEV)

    def backward(self, loss, retain_graph=False):
        loss.backward(retain_graph=retain_graph)

    def unwrap_model(self, model):
        return model


class _FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return "x"


class _Stub:
    """Carrier for the REAL _token_routed_forward_backward + _grpo_per_token_loss."""
    temperature = 0.7
    beta = 1e-3
    epsilon_low = 0.2
    epsilon_high = 0.2
    current_gradient_accumulation_steps = 4
    _routed_adam = True
    _bad_pass_loss_scale = 1.0
    _good_pass_hooked_params = None
    _routing_mode = "exclusive"  # token granularity is exclusive-only
    accelerator = _FakeAccelerator()
    processing_class = _FakeTok()
    _token_routed_forward_backward = SampleGRPOTrainer._token_routed_forward_backward
    _grpo_per_token_loss = SampleGRPOTrainer._grpo_per_token_loss
    _routed_adam_feeds = SampleGRPOTrainer._routed_adam_feeds
    _routed_adam_backward = SampleGRPOTrainer._routed_adam_backward

    def __init__(self, model):
        self._metrics = {"train": {}}
        self._retain_params = {p for n, p in model.named_parameters()
                               if p.requires_grad and "retain" in n}
        self._forget_params = {p for n, p in model.named_parameters()
                               if p.requires_grad and "forget" in n}
        assert self._retain_params and self._forget_params


def _tiny_model():
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).to(DEV)
    apply_dual_mlp(model, retain_neurons=8, forget_neurons=8)
    model.to(DEV)
    set_scales(model, 1.0, 1.0)
    return model


def _fake_inputs(N=4, P=3, C=6, V=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    inputs = {
        "prompt_ids": torch.randint(1, V, (N, P), generator=g).to(DEV),
        "prompt_mask": torch.ones(N, P, dtype=torch.long, device=DEV),
        "completion_ids": torch.randint(1, V, (N, C), generator=g).to(DEV),
        "completion_mask": torch.ones(N, C, dtype=torch.long, device=DEV),
        "advantages": torch.randn(N, generator=g).to(DEV),
        "old_per_token_logps": None,
        "ref_per_token_logps": None,
    }
    inputs["completion_mask"][0, 4:] = 0
    trm = torch.zeros(N, C, dtype=torch.bool, device=DEV)
    trm[0, 1] = True
    trm[2, 3] = True
    trm[3, 0] = True
    inputs["token_routing_mask"] = trm
    return inputs


def test3_integration_streams():
    model = _tiny_model()
    stub = _Stub(model)
    stub.beta = 0.0  # no ref logps in the fake batch
    adapter_params = sorted(stub._retain_params | stub._forget_params, key=id)

    # Ground truth via autograd.grad on a separate identical pass
    def ground_truth(inputs):
        packed = _pack_for_forward(inputs, list(range(inputs["completion_ids"].shape[0])))
        out = model.model(input_ids=packed["packed_input_ids"],
                          position_ids=packed["packed_position_ids"], use_cache=False)
        hidden = out.last_hidden_state
        n_seqs, mc = packed["num_sequences"], packed["max_comp_len"]
        last_hs = torch.zeros(n_seqs, mc, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
        off = 0
        for j, (p_len, c_len) in enumerate(packed["seq_boundaries"]):
            if c_len > 0:
                last_hs[j, :c_len] = hidden[0, off + p_len - 1: off + p_len + c_len - 1]
            off += p_len + c_len
        ptl = stub._grpo_per_token_loss(last_hs, model, packed)
        comp = packed["completion_mask"].bool()
        troute = packed["token_routing_mask"] & comp
        lg = (ptl * (comp & ~troute).float()).sum()
        lb = (ptl * troute.float()).sum()
        gg = torch.autograd.grad(lg, adapter_params, retain_graph=True, allow_unused=True)
        gb = torch.autograd.grad(lb, adapter_params, allow_unused=True)
        z = lambda t, p: torch.zeros_like(p) if t is None else t
        return ([z(t, p) for t, p in zip(gg, adapter_params)],
                [z(t, p) for t, p in zip(gb, adapter_params)])

    batches = [_fake_inputs(seed=1), _fake_inputs(seed=2)]
    want_good = [torch.zeros_like(p) for p in adapter_params]
    want_bad = [torch.zeros_like(p) for p in adapter_params]
    for b in batches:
        gg, gb = ground_truth(b)
        for i in range(len(adapter_params)):
            want_good[i] += gg[i]
            want_bad[i] += gb[i]

    model.zero_grad(set_to_none=True)
    for b in batches:
        stub._token_routed_forward_backward(model, b, num_items_in_batch=None)

    n_checked, worst = 0, 0.0
    for i, p in enumerate(adapter_params):
        full = want_good[i] + want_bad[i]
        scale = full.abs().max().item() + 1e-8
        err_full = (p.grad - full).abs().max().item() / scale
        want_stream = want_good[i] if p in stub._retain_params else want_bad[i]
        err_stream = (p._routed_m_stream - want_stream.float()).abs().max().item() / scale
        worst = max(worst, err_full, err_stream)
        n_checked += 1
    assert worst < 1e-3, f"Test 3 FAILED: worst rel err {worst}"
    print(f"Test 3 (integration: full grad + routed streams over 2 microbatches) PASSED: "
          f"{n_checked} params, worst_rel_err={worst:.2e}")


def test4_amplification_sim():
    torch.manual_seed(7)
    lr, steps = 1e-3, 300
    shape = (256,)
    # behavior gradient: sparse-but-consistent, 1% the scale of the task gradient
    behavior_dir = torch.randn(shape, device=DEV)
    behavior_dir /= behavior_dir.norm()

    def streams():
        g_beh = 0.01 * behavior_dir * (1.0 + 0.1 * torch.randn(1, device=DEV))
        g_task = torch.randn(shape, device=DEV)
        return g_beh, g_task

    # naive per-adapter AdamW: forget param sees ONLY the behavior stream
    p_naive = torch.zeros(shape, device=DEV, requires_grad=True)
    opt_naive = torch.optim.AdamW([p_naive], lr=lr, weight_decay=0.0)
    # RoutedAdam: m from behavior stream, v from full stream
    p_routed = torch.zeros(shape, device=DEV, requires_grad=True)
    opt_routed = RoutedAdam([{"params": [p_routed], "lr": lr, "weight_decay": 0.0, "kappa": 1.0}],
                            lr=lr)
    for _ in range(steps):
        g_beh, g_task = streams()
        p_naive.grad = g_beh.clone()
        opt_naive.step()
        p_routed.grad = (g_beh + g_task).clone()
        p_routed._routed_m_stream = g_beh.clone()
        opt_routed.step()

    d_naive = p_naive.detach().norm().item()
    d_routed = p_routed.detach().norm().item()
    ratio = d_naive / max(d_routed, 1e-12)
    assert ratio > 30, f"Test 4 FAILED: amplification ratio {ratio:.1f} (expected >30)"
    print(f"Test 4 (amplification sim) PASSED: naive moved {d_naive:.4f}, routed {d_routed:.4f} "
          f"-> naive/routed = {ratio:.0f}x (the em-dash runaway mechanism)")


def test5_classic_scheme_baseline_equality():
    """Classic stream weights (retain m <- R, forget m <- R + 2F, shared full v):
    the SUM of the two adapters' updates must equal 2x the AdamW update on the
    combined R+F stream — i.e. combined-model dynamics == the dual-adapter
    routing_mode=none baseline, exactly, at every step (wd=0: decay acts on
    param values, which differ by construction)."""
    lr, betas, eps = 1e-3, (0.9, 0.999), 1e-8
    p_r = torch.zeros(32, 16, device=DEV, requires_grad=True)   # retain
    p_f = torch.zeros(32, 16, device=DEV, requires_grad=True)   # forget
    p_ref = torch.zeros(32, 16, device=DEV, requires_grad=True)  # baseline param
    opt = RoutedAdam([{"params": [p_r, p_f], "lr": lr, "weight_decay": 0.0, "kappa": 1.0}],
                     lr=lr, betas=betas, eps=eps)
    opt_ref = torch.optim.AdamW([p_ref], lr=lr, betas=betas, eps=eps, weight_decay=0.0)

    worst = 0.0
    for _t in range(10):
        g_r = torch.randn(32, 16, device=DEV)
        g_f = 0.05 * torch.randn(32, 16, device=DEV)
        full = g_r + g_f
        prev_r, prev_f, prev_ref = p_r.detach().clone(), p_f.detach().clone(), p_ref.detach().clone()
        p_r.grad = full.clone()
        p_r._routed_m_stream = g_r.clone().float()
        p_f.grad = full.clone()
        p_f._routed_m_stream = (g_r + 2.0 * g_f).clone().float()
        opt.step()
        p_ref.grad = full.clone()
        opt_ref.step()
        opt_ref.zero_grad()
        d_sum = (p_r.detach() - prev_r) + (p_f.detach() - prev_f)
        d_ref = 2.0 * (p_ref.detach() - prev_ref)
        worst = max(worst, (d_sum - d_ref).abs().max().item())
    assert worst < 1e-6, f"Test 5 FAILED: combined-update vs 2x baseline diff = {worst}"
    print(f"Test 5 (classic scheme: sum of adapter updates == 2x none-baseline) PASSED: "
          f"max_abs_diff={worst:.2e}")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _SampleStub:
    """Carrier for the REAL _dynamic_microbatch_forward_backward under routed_adam.
    compute_loss is stubbed with a simple differentiable advantage-weighted logp loss —
    the routing/stream mechanics under test don't depend on the loss internals."""
    _max_tokens_per_microbatch = 20      # 10 tokens/seq -> 2 seqs per microbatch
    _interlaced_coh = False
    _interlaced_coh_opt_batch_mode = "split"
    _is_coherence_rollout = False
    gradient_routing_enabled = True
    _retain_mode = "renormalize"
    _routed_adam = True
    _bad_pass_loss_scale = 1.0
    _good_pass_hooked_params = None
    _good_pass_hook_fn = None
    _trace_routing = False
    use_liger_kernel = False
    accelerator = _FakeAccelerator()
    _dynamic_microbatch_forward_backward = SampleGRPOTrainer._dynamic_microbatch_forward_backward
    _routed_adam_feeds = SampleGRPOTrainer._routed_adam_feeds
    _routed_adam_backward = SampleGRPOTrainer._routed_adam_backward

    def __init__(self, model, routing_mode, classic_bad_weight=2.0):
        self._routing_mode = routing_mode
        self._routed_adam_classic_bad_weight = classic_bad_weight
        self._retain_params = {p for n, p in model.named_parameters()
                               if p.requires_grad and "retain" in n}
        self._forget_params = {p for n, p in model.named_parameters()
                               if p.requires_grad and "forget" in n}
        assert self._retain_params and self._forget_params
        self._model = model

    def _in_forget_warmup(self):
        return False

    def compute_loss_context_manager(self):
        return _NullCtx()

    def compute_loss(self, model, mb_inputs, num_items_in_batch=None):
        ids = torch.cat([mb_inputs["prompt_ids"], mb_inputs["completion_ids"]], dim=1)
        logits = model(input_ids=ids).logits
        logp = logits.log_softmax(-1)[:, :-1].gather(
            -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        P = mb_inputs["prompt_ids"].shape[1]
        mask = torch.zeros_like(logp)
        mask[:, P - 1:] = mb_inputs["completion_mask"].float()
        per_seq = (logp * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return -(per_seq * mb_inputs["advantages"]).mean()


def test6_sample_level_integration():
    """Drive the REAL _dynamic_microbatch_forward_backward (sample-level routing) with
    routed_adam on, mixed good/bad samples, and retain_mode=renormalize (good mbs use
    retain_advantages, bad mbs the original advantages). Verify against autograd.grad
    ground truth, for classic B=2, classic B=1, and exclusive:
      p.grad           == full stream (good + bad), for BOTH adapters
      retain m stream  == good-mb component
      forget m stream  == w_good*good + w_bad*bad components per scheme
    """
    from train import _pack_by_tokens, _trim_and_slice

    N, P, C, V = 6, 4, 6, 64
    g = torch.Generator().manual_seed(3)
    base_inputs = {
        "prompt_ids": torch.randint(1, V, (N, P), generator=g).to(DEV),
        "prompt_mask": torch.ones(N, P, dtype=torch.long, device=DEV),
        "completion_ids": torch.randint(1, V, (N, C), generator=g).to(DEV),
        "completion_mask": torch.ones(N, C, dtype=torch.long, device=DEV),
        "advantages": torch.randn(N, generator=g).to(DEV),
        "retain_advantages": torch.randn(N, generator=g).to(DEV),
        "is_rh": torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long, device=DEV),
        "old_per_token_logps": None,
        "ref_per_token_logps": None,
    }
    token_counts = [P + C] * N
    good_idx = [0, 1, 4]
    bad_idx = [2, 3, 5]

    # (routing_mode, B, expected (w_good_retain, w_good_forget, w_bad_forget))
    schemes = [("classic", 2.0, (1.0, 1.0, 2.0)),
               ("classic", 1.0, (1.0, 1.0, 1.0)),
               ("exclusive", 2.0, (1.0, 0.0, 1.0))]
    for routing_mode, B, (w_gr, w_gf, w_bf) in schemes:
        torch.manual_seed(0)
        model = _tiny_model()
        stub = _SampleStub(model, routing_mode, classic_bad_weight=B)
        adapter_params = sorted(stub._retain_params | stub._forget_params, key=id)

        # Ground truth: per microbatch (replicating the real partition), grad of
        # scale * stub-loss with the advantage source the real loop would use.
        def mb_grads(indices, advantages):
            inp = {**base_inputs, "advantages": advantages}
            mb = _trim_and_slice(inp, indices)
            loss = stub.compute_loss(model, mb) * (len(indices) / N)
            gs = torch.autograd.grad(loss, adapter_params, allow_unused=True)
            return [torch.zeros_like(p) if t is None else t
                    for t, p in zip(gs, adapter_params)]

        want_good = [torch.zeros_like(p) for p in adapter_params]
        want_bad = [torch.zeros_like(p) for p in adapter_params]
        for mb in _pack_by_tokens(token_counts, good_idx, stub._max_tokens_per_microbatch):
            for acc, t in zip(want_good, mb_grads(mb, base_inputs["retain_advantages"])):
                acc += t
        for mb in _pack_by_tokens(token_counts, bad_idx, stub._max_tokens_per_microbatch):
            for acc, t in zip(want_bad, mb_grads(mb, base_inputs["advantages"])):
                acc += t

        model.zero_grad(set_to_none=True)
        stub._dynamic_microbatch_forward_backward(
            model, base_inputs, num_items_in_batch=None, record_metrics=False)

        worst = 0.0
        for i, p in enumerate(adapter_params):
            full = want_good[i] + want_bad[i]
            scale_ = full.abs().max().item() + 1e-8
            assert p.grad is not None, "adapter param missing full-stream grad"
            worst = max(worst, (p.grad - full).abs().max().item() / scale_)
            if p in stub._retain_params:
                want_stream = w_gr * want_good[i]
            else:
                want_stream = w_gf * want_good[i] + w_bf * want_bad[i]
            worst = max(worst, (p._routed_m_stream - want_stream.float()).abs().max().item() / scale_)
        assert worst < 1e-3, (
            f"Test 6 FAILED ({routing_mode}, B={B}): worst rel err {worst}")
        print(f"Test 6 (sample-level integration, {routing_mode} B={B}) PASSED: "
              f"{len(adapter_params)} params, worst_rel_err={worst:.2e}")


def test7_selfdistill_coherence_integration():
    """RoutedAdam + self-distillation coherence (merged interlaced): drive the REAL
    _dynamic_microbatch_forward_backward with a mixed good/bad/coherence batch and
    verify against autograd ground truth:
      coherence mbs run at scales (1,0): their grads (w.r.t. retain params; forget
        grads are structurally zero at (1,0)) feed p.grad AND retain's m stream at
        weight 1, with the coh-slice constant advantage;
      good/bad routing mbs behave exactly as in test6 (classic B=2 here).
    """
    from train import _pack_by_tokens, _trim_and_slice

    torch.manual_seed(11)
    model = _tiny_model()
    stub = _SampleStub(model, "classic", classic_bad_weight=2.0)
    stub._interlaced_coh = True
    stub._interlaced_coh_opt_batch_mode = "merged"
    stub._coherence_rh_mode = "penalty"
    stub._rh_detector_verifies_retain_samples = False
    stub._coh_fixed_advantage = 1.0
    stub._coh_loss_type = "grpo"
    stub._train_forget_scale = lambda: 1.0
    adapter_params = sorted(stub._retain_params | stub._forget_params, key=id)
    retain_set = stub._retain_params

    N, P, C, V = 8, 4, 6, 64
    g = torch.Generator().manual_seed(5)
    ALPHA = 0.7
    adv = torch.randn(N, generator=g).to(DEV)
    adv[6] = ALPHA  # coherence rows carry the fixed advantage (set upstream
    adv[5] = ALPHA  # in _generate_and_score_completions; baked in here)
    base_inputs = {
        "prompt_ids": torch.randint(1, V, (N, P), generator=g).to(DEV),
        "prompt_mask": torch.ones(N, P, dtype=torch.long, device=DEV),
        "completion_ids": torch.randint(1, V, (N, C), generator=g).to(DEV),
        "completion_mask": torch.ones(N, C, dtype=torch.long, device=DEV),
        "advantages": adv,
        "retain_advantages": torch.randn(N, generator=g).to(DEV),
        "is_rh": torch.tensor([0, 0, 1, 1, 0, 0, 0, 1], dtype=torch.long, device=DEV),
        "is_coherence": torch.tensor([0, 0, 0, 0, 0, 1, 1, 0], dtype=torch.bool, device=DEV),
        "old_per_token_logps": None,
        "ref_per_token_logps": None,
    }
    token_counts = [P + C] * N
    coh_idx, good_idx, bad_idx = [5, 6], [0, 1, 4], [2, 3, 7]

    def mb_grads(indices, advantages, scales):
        set_scales(model, *scales)
        inp = {**base_inputs, "advantages": advantages}
        mb = _trim_and_slice(inp, indices)
        loss = stub.compute_loss(model, mb) * (len(indices) / N)
        gs = torch.autograd.grad(loss, adapter_params, allow_unused=True)
        set_scales(model, 1.0, 1.0)
        return [torch.zeros_like(p) if t is None else t
                for t, p in zip(gs, adapter_params)]

    want_good = [torch.zeros_like(p) for p in adapter_params]
    want_bad = [torch.zeros_like(p) for p in adapter_params]
    want_coh = [torch.zeros_like(p) for p in adapter_params]
    mt = stub._max_tokens_per_microbatch
    for mb in _pack_by_tokens(token_counts, good_idx, mt):
        for acc, t in zip(want_good, mb_grads(mb, base_inputs["retain_advantages"], (1.0, 1.0))):
            acc += t
    for mb in _pack_by_tokens(token_counts, bad_idx, mt):
        for acc, t in zip(want_bad, mb_grads(mb, base_inputs["advantages"], (1.0, 1.0))):
            acc += t
    for mb in _pack_by_tokens(token_counts, coh_idx, mt):
        for acc, t in zip(want_coh, mb_grads(mb, base_inputs["advantages"], (1.0, 0.0))):
            acc += t

    model.zero_grad(set_to_none=True)
    stub._dynamic_microbatch_forward_backward(
        model, base_inputs, num_items_in_batch=None, record_metrics=False)

    worst = 0.0
    for i, p in enumerate(adapter_params):
        full = want_good[i] + want_bad[i] + want_coh[i]
        scale_ = full.abs().max().item() + 1e-8
        worst = max(worst, (p.grad - full).abs().max().item() / scale_)
        if p in retain_set:
            want_stream = want_good[i] + want_coh[i]          # coh feeds retain m at w=1
        else:
            want_stream = want_good[i] + 2.0 * want_bad[i]    # coh grads are zero for forget
        worst = max(worst, (p._routed_m_stream - want_stream.float()).abs().max().item() / scale_)
    assert worst < 1e-3, f"Test 7 FAILED: worst rel err {worst}"
    # Forget params must receive literally zero gradient from coherence mbs.
    coh_forget = max((t.abs().max().item() for t, p in zip(want_coh, adapter_params)
                      if p not in retain_set), default=0.0)
    assert coh_forget == 0.0, f"Test 7 FAILED: coh grads leaked to forget ({coh_forget})"
    print(f"Test 7 (self-distill coherence + RoutedAdam, merged interlaced) PASSED: "
          f"{len(adapter_params)} params, worst_rel_err={worst:.2e}")


if __name__ == "__main__":
    test1_reference_equality()
    test2_stream_split()
    test3_integration_streams()
    test4_amplification_sim()
    test5_classic_scheme_baseline_equality()
    test6_sample_level_integration()
    test7_selfdistill_coherence_integration()
    print("\nALL ROUTED-ADAM TESTS PASSED")
