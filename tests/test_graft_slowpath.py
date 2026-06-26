"""graft-port slice 2a/2b: the λ≠1 slow-path 2-backward v=a_v orchestration.

CPU, fp64, against the REAL gradient_routing primitives (DualMLPAdapter +
PreRoutingGradAccumulator.rearm/flush + set_fused_routing) and advantages.py mask
weights. Pins:
  - the 2-backward isolates m (decouple-masked grad at a_m, in .grad) from v
    (natural grad at a_v, in _pre_routing_grad) to fp64 across on/off-policy × β
    (the load-bearing slow-path correctness — regression gate for any change to
    the orchestration or the accumulator).
  - the three mechanical guards: rearm() asserts ≥1 capture; omitting rearm makes
    the v-backward capture nothing (caught loud by the trainer's `assert
    cap._captures`); omitting the .grad snapshot silently corrupts m.
  - λ>1 over-routing masks (negative retain weight) keep the isolation.

Promoted from tools/graft_slowpath/{v_isolation_proto,adv_failmodes}.py.
"""
import pytest
import torch
import torch.nn as nn

from gradient_routing import (DualMLPAdapter, PreRoutingGradAccumulator,
                              set_fused_routing, clear_fused_routing)
from advantages import adapter_kappas, routing_grad_mask_weights


@pytest.fixture(autouse=True)
def _fp64():
    """fp64 for the isolation proofs — SCOPED per test so the module-level default
    dtype change doesn't leak into other test files during collection."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


S, T, D, V = 3, 5, 6, 9
R_N, F_N = 3, 2          # unequal -> κ != (2,2)


def _build(seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(D, D, bias=False)
    nn.init.eye_(base.weight)
    ad = DualMLPAdapter(base, hidden_size=D, retain_neurons=R_N, forget_neurons=F_N)
    for lin in (ad.gate_retain, ad.up_retain, ad.down_retain,
                ad.gate_forget, ad.up_forget, ad.down_forget):
        nn.init.normal_(lin.weight, std=0.4)
    Mmix = torch.randn(T, T)
    Wh = torch.randn(D, V)
    X = torch.randn(S, T, D)
    chosen = torch.randint(0, V, (S, T))
    ref_logp = torch.log_softmax(torch.randn(S, T, V), -1).gather(
        -1, chosen.unsqueeze(-1)).squeeze(-1)

    def upper(out):
        h = torch.tanh(out)
        hmix = torch.einsum('tu,sud->std', Mmix, h)
        return torch.log_softmax(hmix @ Wh, -1).gather(
            -1, chosen.unsqueeze(-1)).squeeze(-1)

    def loss(logp, A, beta, shift, eps=0.2):
        old = logp.detach() - shift
        r = torch.exp(logp - old)
        Ab = A.view(S, 1)
        pg = -torch.minimum(r * Ab, torch.clamp(r, 1 - eps, 1 + eps) * Ab)
        k3 = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        return (pg + beta * k3).sum()

    return ad, X, upper, loss


def _masks(lam):
    kR, kF = adapter_kappas(R_N, F_N)
    _, _, rgm_bad, fgm_bad = routing_grad_mask_weights("classic", lam, kR, kF)
    forget_fwd = torch.full((1, T, 1), 1.0)
    retain_gm = torch.full((1, T, 1), float(rgm_bad))
    forget_gm = torch.full((1, T, 1), float(fgm_bad))
    return forget_fwd, retain_gm, forget_gm


def _relerr(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-300)


def _reference(ad, X, upper, loss, forget_fwd, retain_gm, forget_gm, a_m, a_v,
               beta, shift):
    """m = decouple-masked param-grad at a_m; v = natural param-grad at a_v."""
    clear_fused_routing()
    nat = ad.natural_adapter_output(X, forget_fwd)
    # base is an eye-init Linear (no bias) -> base(X) == X exactly; the decouple
    # only changes the gradient, not the forward value, so ad(X) == X + nat.
    mo = X + nat
    logp = upper(mo)
    g_m = torch.autograd.grad(loss(logp, a_m, beta, shift), mo, retain_graph=True)[0]
    m_ret = torch.autograd.grad(nat, ad.get_retain_params(),
                                grad_outputs=retain_gm.permute(0, 2, 1).reshape(1, T, 1) * g_m,
                                retain_graph=True)
    m_for = torch.autograd.grad(nat, ad.get_forget_params(),
                                grad_outputs=forget_gm.permute(0, 2, 1).reshape(1, T, 1) * g_m,
                                retain_graph=True)
    g_v = torch.autograd.grad(loss(logp, a_v, beta, shift), mo, retain_graph=True)[0]
    params = ad.get_retain_params() + ad.get_forget_params()
    v_ref = list(torch.autograd.grad(nat, params, grad_outputs=g_v))
    return list(m_ret) + list(m_for), v_ref


def _orchestrate(ad, X, upper, loss, forget_fwd, retain_gm, forget_gm, a_m, a_v,
                 beta, shift):
    """The production 2-backward (matches _slow_microbatch_backward)."""
    params = ad.get_retain_params() + ad.get_forget_params()
    cap = PreRoutingGradAccumulator(ad)
    cap.reset()
    for p in params:
        p.grad = None
    set_fused_routing(forget_fwd, retain_gm, forget_gm)
    out = ad(X)
    logp = upper(out)
    loss(logp, a_m, beta, shift).backward(retain_graph=True)     # bw1: m
    snap = {p: p.grad.detach().clone() for p in params}
    for p in params:
        p.grad = None
    cap.rearm()
    loss(logp, a_v, beta, shift).backward()                      # bw2: v
    assert cap._captures, "v-backward captured nothing"
    cap.flush(forget_fwd)
    for p in params:
        p.grad = snap[p]
    m = [p.grad.detach().clone() for p in params]
    v = [getattr(p, "_pre_routing_grad").detach().clone() for p in params]
    clear_fused_routing()
    cap.remove()
    return m, v


@pytest.mark.parametrize("lam", [0.5, 0.9, 1.5])   # soft, near-1, over-routing
def test_two_backward_isolation(lam):
    ad, X, upper, loss = _build(seed=0)
    forget_fwd, retain_gm, forget_gm = _masks(lam)
    a_m = torch.tensor([1.0, -0.5, 0.8])
    a_v = torch.tensor([0.6, 0.3, -0.4])
    worst = 0.0
    for beta in (0.0, 0.05):
        for shift in (0.0, 0.4):     # on / off policy
            m_ref, v_ref = _reference(ad, X, upper, loss, forget_fwd, retain_gm,
                                      forget_gm, a_m, a_v, beta, shift)
            m, v = _orchestrate(ad, X, upper, loss, forget_fwd, retain_gm,
                                forget_gm, a_m, a_v, beta, shift)
            worst = max(worst, max(_relerr(a, b) for a, b in zip(m, m_ref)),
                        max(_relerr(a, b) for a, b in zip(v, v_ref)))
    assert worst < 1e-10, f"m/v isolation broke: worst relerr {worst:.2e}"


def test_rearm_asserts_on_empty():
    ad, *_ = _build()
    cap = PreRoutingGradAccumulator(ad)
    cap.reset()
    with pytest.raises(AssertionError):
        cap.rearm()           # no captures -> the m-backward hook never fired
    cap.remove()


def test_failmode_rearm_omitted_gives_wrong_v_silently():
    # Omitting rearm() -> bw2's _bwd sees x=None (popped by bw1, not re-saved) so
    # it captures NOTHING; the STALE bw1 captures (g_m) survive in _captures. The
    # trainer's `assert cap._captures` therefore does NOT catch this (non-empty but
    # stale) — flush then builds v from g_m, i.e. v rides a_m. This silent
    # near-miss is the reason the isolation test + rearm()'s own moved>=1 assert
    # are load-bearing (MASTER_PORT_PLAN §12). Pin the silent-wrong behavior so a
    # future "fix" that makes it loud is noticed.
    ad, X, upper, loss = _build(seed=2)
    forget_fwd, retain_gm, forget_gm = _masks(0.5)
    a_m = torch.tensor([1.0, -0.5, 0.8])
    a_v = torch.tensor([0.6, 0.3, -0.4])
    # reference natural grads at a_m and a_v
    m_ref, v_ref = _reference(ad, X, upper, loss, forget_fwd, retain_gm, forget_gm,
                              a_m, a_v, 0.05, 0.4)
    nat = ad.natural_adapter_output(X, forget_fwd)
    mo = X + nat
    g_am = torch.autograd.grad(loss(upper(mo), a_m, 0.05, 0.4), mo, retain_graph=True)[0]
    params = ad.get_retain_params() + ad.get_forget_params()
    v_at_am = list(torch.autograd.grad(nat, params, grad_outputs=g_am))  # natural(a_m)

    cap = PreRoutingGradAccumulator(ad)
    cap.reset()
    for p in params:
        p.grad = None
    set_fused_routing(forget_fwd, retain_gm, forget_gm)
    out = ad(X)
    logp = upper(out)
    loss(logp, a_m, 0.05, 0.4).backward(retain_graph=True)
    # NO rearm() — the bug
    loss(logp, a_v, 0.05, 0.4).backward()
    assert cap._captures, "stale g_m captures survive (capture-guard does NOT catch this)"
    cap.flush(forget_fwd)
    v_bad = [getattr(p, "_pre_routing_grad").detach().clone() for p in params]
    clear_fused_routing()
    cap.remove()
    # v silently equals natural(a_m), NOT natural(a_v)
    assert max(_relerr(a, b) for a, b in zip(v_bad, v_at_am)) < 1e-10
    assert max(_relerr(a, b) for a, b in zip(v_bad, v_ref)) > 1e-2


def test_failmode_no_snapshot_corrupts_m():
    # Omitting the .grad snapshot/restore -> bw2's masked-grad(a_v) accumulates
    # onto m. Silent. Pin that the CORRECT orchestration avoids it (regression).
    ad, X, upper, loss = _build(seed=3)
    forget_fwd, retain_gm, forget_gm = _masks(0.5)
    a_m = torch.tensor([1.0, -0.5, 0.8])
    a_v = torch.tensor([0.6, 0.3, -0.4])
    m_ref, _ = _reference(ad, X, upper, loss, forget_fwd, retain_gm, forget_gm,
                          a_m, a_v, 0.05, 0.4)
    params = ad.get_retain_params() + ad.get_forget_params()
    # corrupted: no snapshot/restore
    cap = PreRoutingGradAccumulator(ad)
    cap.reset()
    for p in params:
        p.grad = None
    set_fused_routing(forget_fwd, retain_gm, forget_gm)
    out = ad(X)
    logp = upper(out)
    loss(logp, a_m, 0.05, 0.4).backward(retain_graph=True)
    cap.rearm()
    loss(logp, a_v, 0.05, 0.4).backward()    # contaminates .grad
    cap.flush(forget_fwd)
    m_bad = [p.grad.detach().clone() for p in params]
    clear_fused_routing()
    cap.remove()
    em = max(_relerr(a, b) for a, b in zip(m_bad, m_ref))
    assert em > 1e-2, ("expected corruption without snapshot — if this is small "
                       "the snapshot/restore stopped mattering (investigate)")
    # and the CORRECT orchestration is clean:
    m_ok, _ = _orchestrate(ad, X, upper, loss, forget_fwd, retain_gm, forget_gm,
                           a_m, a_v, 0.05, 0.4)
    assert max(_relerr(a, b) for a, b in zip(m_ok, m_ref)) < 1e-10


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
