"""graft-port slice 2b (λ>1 over-routing): the optimizer B1 v-floor + the
realized-step gate (split_moment.py), and the accumulator double-flush that feeds
the v-floor (gradient_routing.py). Pure-CPU, synthetic grads / tiny adapter.

Pins:
  - v_floor: v ← max(|v_natural|, |v_routed|) per coordinate → never UNDER-steps
    (the off-policy λ>1 sign-flip protection); a no-op when v_routed ≤ v_natural.
  - realized-step gate: a routing param whose actual per-coordinate Adam step
    |m̂|/(√v̂+eps) exceeds w_max FAILS LOUD; within budget it passes.
  - the double-flush isolates _v_routed = natural(a_m), _pre_routing_grad =
    natural(a_v) on one shared forward.
"""
import pytest
import torch
import torch.nn as nn

from split_moment import SplitMomentAdamW
from gradient_routing import (DualMLPAdapter, PreRoutingGradAccumulator,
                              set_fused_routing, clear_fused_routing)
from advantages import adapter_kappas, routing_grad_mask_weights


def _opt(roles_params, lr=0.1, beta2=0.999):
    groups = [{"params": [p], "lr": lr, "weight_decay": 0.0, "graft_role": role}
              for role, p in roles_params]
    return SplitMomentAdamW(groups, lr=lr, betas=(0.9, beta2), eps=1e-8, weight_decay=0.0)


# ---------------- optimizer v-floor ----------------

def test_v_floor_uses_larger_natural_grad():
    # v_routed (natural a_m) bigger than v_natural (a_v) -> floored v is larger ->
    # the step shrinks vs no-floor. With the floor OFF, v rides v_natural only.
    p_floor = torch.zeros(8, requires_grad=True)
    p_nofloor = torch.zeros(8, requires_grad=True)
    of, on = _opt([("forget", p_floor)]), _opt([("forget", p_nofloor)])
    for _ in range(60):
        for p in (p_floor, p_nofloor):
            p.grad = torch.full_like(p, 1.0)          # m
            p._pre_routing_grad = torch.full_like(p, 1.0)   # v_natural
            p._v_routed = torch.full_like(p, 4.0)           # v_routed (larger)
        of.set_window({"forget": 1.0}, {"forget": True}, v_floor=True); of.step()
        on.set_window({"forget": 1.0}, {"forget": True}, v_floor=False); on.step()
    # floored uses v=max(1,4)=4 -> sqrt(v) 4x larger -> ~1/4 the step
    ratio = p_floor.abs().mean().item() / p_nofloor.abs().mean().item()
    assert abs(ratio - 0.25) < 0.02, ratio


def test_v_floor_noop_when_routed_smaller():
    # v_routed ≤ v_natural -> floor is a no-op (max picks v_natural).
    pf = torch.zeros(8, requires_grad=True)
    pn = torch.zeros(8, requires_grad=True)
    of, on = _opt([("forget", pf)]), _opt([("forget", pn)])
    for _ in range(40):
        for p in (pf, pn):
            p.grad = torch.full_like(p, 1.0)
            p._pre_routing_grad = torch.full_like(p, 3.0)
            p._v_routed = torch.full_like(p, 1.0)       # smaller -> ignored
        of.set_window({"forget": 1.0}, {"forget": True}, v_floor=True); of.step()
        on.set_window({"forget": 1.0}, {"forget": True}, v_floor=False); on.step()
    assert torch.allclose(pf, pn, atol=1e-6)


def test_v_floor_missing_routed_raises():
    p = torch.zeros(4, requires_grad=True)
    o = _opt([("forget", p)])
    p.grad = torch.full_like(p, 1.0)
    p._pre_routing_grad = torch.full_like(p, 1.0)
    # no _v_routed
    o.set_window({"forget": 1.0}, {"forget": True}, v_floor=True)
    with pytest.raises(AssertionError):
        o.step()


# ---------------- realized-step gate (opt-in 'gate' policy) ----------------

def test_gate_fires_when_over_budget():
    # m >> v -> per-coordinate step ~m/v >> w_max -> gate policy asserts.
    p = torch.zeros(4, requires_grad=True)
    o = _opt([("forget", p)])
    p.grad = torch.full_like(p, 10.0)               # big m
    p._pre_routing_grad = torch.full_like(p, 1.0)   # small v -> step ~10
    o.set_window({"forget": 1.0}, {"forget": True}, w_max=4.0, step_policy="gate")
    with pytest.raises(AssertionError):
        o.step()


def test_gate_passes_within_budget():
    p = torch.zeros(4, requires_grad=True)
    o = _opt([("forget", p)])
    p.grad = torch.full_like(p, 1.0)                # m == v -> step ~1 ≤ 4
    p._pre_routing_grad = torch.full_like(p, 1.0)
    o.set_window({"forget": 1.0}, {"forget": True}, w_max=4.0, step_policy="gate")
    o.step()                                        # no raise
    assert o._last_realized_max <= 4.0


def test_no_gate_or_clamp_when_w_max_none():
    # untagged / no w_max -> neither gate nor clamp even with an explosive ratio
    # (the λ≤1 path never sets w_max).
    p = torch.zeros(4, requires_grad=True)
    o = _opt([("forget", p)])
    p.grad = torch.full_like(p, 50.0)
    p._pre_routing_grad = torch.full_like(p, 1.0)
    o.set_window({"forget": 1.0}, {"forget": True})   # w_max=None
    o.step()                                          # must NOT raise
    # step is the full ~50× (unbounded) since no w_max
    assert p.abs().mean().item() > 10 * o.param_groups[0]["lr"]


# ---------------- per-coordinate clamp (default policy) ----------------

def test_clamp_noop_when_within_budget():
    # realized step ≤ w_max -> clamp is a NO-OP: bit-identical to no-w_max, and
    # frac_coords_clamped == 0. (This is the "no-op at the κ operating point"
    # property — at λ=1 with w_max ≥ κ the clamp never bites.)
    pc = torch.zeros(8, requires_grad=True)
    pn = torch.zeros(8, requires_grad=True)
    oc, on = _opt([("forget", pc)]), _opt([("forget", pn)])
    for _ in range(50):
        for p in (pc, pn):
            p.grad = torch.full_like(p, 2.0)               # m
            p._pre_routing_grad = torch.full_like(p, 1.0)  # v -> realized ~2 ≤ 4
        oc.set_window({"forget": 1.0}, {"forget": True}, w_max=4.0, step_policy="clamp")
        oc.step()
        on.set_window({"forget": 1.0}, {"forget": True})   # no w_max
        on.step()
    assert torch.allclose(pc, pn, atol=0, rtol=0), "clamp changed an in-budget step"
    assert oc._last_frac_clamped == 0.0


def test_clamp_bounds_step_to_w_max():
    # realized step ~20 >> w_max=4 -> clamp caps the per-coordinate step at 4×lr.
    lr = 0.1
    pc = torch.zeros(8, requires_grad=True)
    pn = torch.zeros(8, requires_grad=True)
    oc, on = _opt([("forget", pc)], lr=lr), _opt([("forget", pn)], lr=lr)
    for p in (pc, pn):
        p.grad = torch.full_like(p, 20.0)
        p._pre_routing_grad = torch.full_like(p, 1.0)      # realized ~20
    oc.set_window({"forget": 1.0}, {"forget": True}, w_max=4.0, step_policy="clamp")
    oc.step()
    on.set_window({"forget": 1.0}, {"forget": True})       # unbounded
    on.step()
    # first step from 0: clamped |Δ| == w_max·lr exactly; unclamped is much larger
    assert pc.abs().max().item() <= 4.0 * lr + 1e-6
    assert pc.abs().min().item() > 3.9 * lr           # saturated at the ceiling
    assert on.param_groups and pn.abs().mean().item() > 10 * lr
    assert oc._last_frac_clamped == 1.0


def test_clamp_leaves_moment_state_unchanged():
    # the clamp floors only the LOCAL denom used for the update; the stored second
    # moment (exp_avg_sq) and first moment (exp_avg) must be identical to the
    # unclamped run (like grad-clip: clip the step, not the state). Only the PARAM
    # trajectory differs.
    pc = torch.zeros(8, requires_grad=True)
    pn = torch.zeros(8, requires_grad=True)
    oc, on = _opt([("forget", pc)]), _opt([("forget", pn)])
    for _ in range(5):
        for p in (pc, pn):
            p.grad = torch.full_like(p, 20.0)
            p._pre_routing_grad = torch.full_like(p, 1.0)
        oc.set_window({"forget": 1.0}, {"forget": True}, w_max=4.0, step_policy="clamp")
        oc.step()
        on.set_window({"forget": 1.0}, {"forget": True})   # no w_max -> no clamp
        on.step()
    assert torch.allclose(oc.state[pc]["exp_avg_sq"], on.state[pn]["exp_avg_sq"], atol=1e-7)
    assert torch.allclose(oc.state[pc]["exp_avg"], on.state[pn]["exp_avg"], atol=1e-7)
    assert not torch.allclose(pc, pn)                       # but the params diverged


# ---------------- accumulator double-flush ----------------

def test_double_flush_isolates_v_routed_and_v_natural():
    torch.set_default_dtype(torch.float64)
    S, T, D, V = 3, 5, 6, 9
    R_N, F_N = 3, 2
    torch.manual_seed(0)
    base = nn.Linear(D, D, bias=False); nn.init.eye_(base.weight)
    ad = DualMLPAdapter(base, hidden_size=D, retain_neurons=R_N, forget_neurons=F_N)
    for lin in (ad.gate_retain, ad.up_retain, ad.down_retain,
                ad.gate_forget, ad.up_forget, ad.down_forget):
        nn.init.normal_(lin.weight, std=0.4)
    Mmix = torch.randn(T, T); Wh = torch.randn(D, V)
    X = torch.randn(S, T, D); chosen = torch.randint(0, V, (S, T))

    def upper(out):
        h = torch.tanh(out); hmix = torch.einsum('tu,sud->std', Mmix, h)
        return torch.log_softmax(hmix @ Wh, -1).gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

    def loss(logp, A):
        return -(logp * A.view(S, 1)).sum()

    kR, kF = adapter_kappas(R_N, F_N)
    _, _, rgm, fgm = routing_grad_mask_weights("classic", 1.5, kR, kF)   # λ>1
    forget_fwd = torch.full((1, T, 1), 1.0)
    retain_gm = torch.full((1, T, 1), float(rgm))
    forget_gm = torch.full((1, T, 1), float(fgm))
    a_m = torch.tensor([1.0, -0.5, 0.8]); a_v = torch.tensor([0.6, 0.3, -0.4])
    params = ad.get_retain_params() + ad.get_forget_params()

    # references: natural grads at a_m / a_v
    clear_fused_routing()
    nat = ad.natural_adapter_output(X, forget_fwd); mo = X + nat
    g_am = torch.autograd.grad(loss(upper(mo), a_m), mo, retain_graph=True)[0]
    vr_ref = list(torch.autograd.grad(nat, params, grad_outputs=g_am, retain_graph=True))
    g_av = torch.autograd.grad(loss(upper(mo), a_v), mo, retain_graph=True)[0]
    vn_ref = list(torch.autograd.grad(nat, params, grad_outputs=g_av))

    cap = PreRoutingGradAccumulator(ad); cap.reset()
    for p in params: p.grad = None
    set_fused_routing(forget_fwd, retain_gm, forget_gm)
    out = ad(X); logp = upper(out)
    loss(logp, a_m).backward(retain_graph=True)
    for p in params: p.grad = None
    cap.flush(forget_fwd, into="_v_routed", keep=True)   # natural(a_m) + keep x
    loss(logp, a_v).backward()
    cap.flush(forget_fwd)                                 # natural(a_v)
    clear_fused_routing()
    vr = [p._v_routed for p in params]
    vn = [p._pre_routing_grad for p in params]
    cap.remove()

    def relerr(a, b): return (a - b).norm().item() / (b.norm().item() + 1e-300)
    assert max(relerr(a, b) for a, b in zip(vr, vr_ref)) < 1e-10
    assert max(relerr(a, b) for a, b in zip(vn, vn_ref)) < 1e-10
    torch.set_default_dtype(torch.float32)


# ---------------- effective-λ diagnostic reduction (over routing groups only) ----------------

def test_lam_eff_diagnostic_excludes_coherence_groups():
    # Mirrors the trainer's graft/lam_eff_* reduction: it must average the per-group
    # lam_eff over ROUTING groups only. Coherence groups (n_det=0 → uncapped
    # lam_eff=nominal) would otherwise bias the mean up / frac-capped down.
    G = 4
    nominal_lambda = 1.5
    is_coherence = torch.zeros(12, dtype=torch.bool)
    is_coherence[0:4] = True                       # group 0 = coherence
    le = torch.tensor([1.5, 1.2, 1.5])             # per-group lam_eff; coh group = nominal
    routing_groups = ~is_coherence.view(-1, G).all(dim=1)
    assert routing_groups.tolist() == [False, True, True]
    le_r = le[routing_groups]
    assert torch.equal(le_r, torch.tensor([1.2, 1.5]))
    # routing-only mean 1.35, NOT the all-group 1.4 (coherence would pull it up)
    assert abs(le_r.mean().item() - 1.35) < 1e-6
    assert abs(le.mean().item() - 1.4) < 1e-6      # the polluted value we avoid
    # frac capped: only g1 (1.2 < 1.5) of the 2 routing groups
    frac = (le_r < nominal_lambda - 1e-9).float().mean().item()
    assert abs(frac - 0.5) < 1e-6
    # all-coherence batch → no routing groups → the trainer skips logging (numel 0)
    all_coh = torch.ones(8, dtype=torch.bool)
    rg = ~all_coh.view(-1, G).all(dim=1)
    assert le[:2][rg].numel() == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
