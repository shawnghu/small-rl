"""Split-moment AdamW: second moment (v) from the pre-routing gradient, first
moment (m) from the routed gradient.

Standard Adam maintains, per parameter,
    m_t = β1·m_{t-1} + (1-β1)·g_t            (first moment / momentum)
    v_t = β2·v_{t-1} + (1-β2)·g_t²           (second moment / variance estimate)
and steps along m̂ / (√v̂ + eps). ``SplitMomentAdamW`` feeds the two moments from
TWO different gradients of the same step:
    m  <- p.grad                  (the ROUTED gradient — gate-masked + redistributed)
    v  <- p._pre_routing_grad     (the NATURAL/pre-routing gradient — every sample
                                   reaches both adapters at scale 1)

Rationale (gradient-routing experiment): the update *direction* should follow the
routing (m from the routed grad), while the per-parameter *scale* (v) should
reflect the natural gradient magnitude rather than the post-routing one. The two
gradients are produced in a single backward by the fused path + the
``PreRoutingGradAccumulator`` capture (see gradient_routing.py / train.py).

A parameter with no ``_pre_routing_grad`` attribute (or ``None``) falls back to
using ``p.grad`` for both moments — i.e. it behaves as ordinary AdamW. So a
mixed param set (e.g. a head trained without the capture) is handled gracefully.

Decoupled weight decay (AdamW), matching torch.optim.AdamW. amsgrad / maximize /
fused / capturable are not supported (asserted off).
"""

import math

import torch
from torch.optim import AdamW


def clip_pre_routing_grads_(param_groups, max_norm, total_norm):
    """Scale every ``p._pre_routing_grad`` in ``param_groups`` by the same
    coefficient ``torch.nn.utils.clip_grad_norm_`` applies to ``.grad``:
    ``clip_coef = min(1, max_norm / (total_norm + 1e-6))``, where ``total_norm``
    is the pre-clip total norm that call returned. No-op when nothing clips
    (coef >= 1) or args are missing. Returns the applied coef (or None).

    Keeps gradient clipping a single shared event across Adam's two moments when
    they are fed from two different gradients (m <- .grad, v <- _pre_routing_grad).
    """
    if max_norm is None or total_norm is None:
        return None
    clip_coef = float(max_norm) / (float(total_norm) + 1e-6)
    if clip_coef >= 1.0:
        return None
    for group in param_groups:
        for p in group["params"]:
            b = getattr(p, "_pre_routing_grad", None)
            if b is not None:
                b.mul_(clip_coef)
    return clip_coef


class SplitMomentAdamW(AdamW):
    """AdamW with v from the pre-routing grad and m from the routed grad, plus
    graft-port extensions consumed via ``set_window`` each optimizer window:

    - per-role PARTICIPATION factor ``c_A`` (scales the v-source in gradient space,
      squared after → the adapter's per-window step ×= 1/c_A). With retain-only
      coherence interleaved, ``c_F = N/N_routing`` makes the forget adapter step at
      retain's per-EXAMPLE rate instead of Adam re-normalizing its smaller
      accumulated v back up to the full per-window rate.
    - FREEZE: an adapter with no examples this window (``active[role]=False``, e.g.
      an all-coherence window for forget) is skipped entirely — m, v, the per-param
      step counter, and weight decay all untouched — so it does not drift, and its
      bias-correction stays correct (the per-param ``state['step']`` only advances
      on participating windows → a per-role step counter falls out for free).

    Groups carry a ``graft_role`` tag ("retain"/"forget"). A routing (tagged) param
    with no captured ``_pre_routing_grad`` is a HARD ERROR — no silent fall-back to
    plain AdamW under routing. Untagged params keep the plain ``v<-.grad`` fallback.
    """

    def set_window(self, participation, active, *, v_floor=False, w_max=None):
        """Stash {role: c_A} participation factors and {role: bool} active flags for
        the next step(). The trainer calls this each optimizer window (before HF's
        arg-less optimizer.step()). Consumed and cleared by step().

        graft-port slice 2b (λ>1 over-routing): ``v_floor`` enables the per-
        coordinate second-moment floor ``v ← max(v_natural, v_routed)`` (the v-
        source is ``max(|_pre_routing_grad|, |_v_routed|)`` — both NATURAL grads, at
        a_v and a_m resp.) to restore the shared-clip invariant the off-policy λ>1
        sign-flip breaks; ``w_max`` (when set) arms the **realized-step gate** — a
        fail-loud assert that no routing param's actual per-coordinate Adam step
        ``|m̂|/(√v̂+eps)`` exceeds ``w_max`` (the static κ guard is necessary, not
        sufficient, at λ>1; MASTER_PORT_PLAN §12 2b item 2)."""
        self._window = {"c": participation, "active": active,
                        "v_floor": v_floor, "w_max": w_max}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        window = getattr(self, "_window", None)
        v_floor = bool(window["v_floor"]) if window is not None else False
        w_max = window["w_max"] if window is not None else None
        realized_max = 0.0   # tracks the largest per-coordinate routing step / lr
        for group in self.param_groups:
            assert not group.get("amsgrad", False), "SplitMomentAdamW: amsgrad unsupported"
            assert not group.get("maximize", False), "SplitMomentAdamW: maximize unsupported"
            assert not group.get("fused", False) and not group.get("foreach", False), (
                "SplitMomentAdamW: fused/foreach unsupported (use the plain loop)")
            role = group.get("graft_role")
            # Per-role participation + freeze. Default c=1 / always-active for
            # untagged groups or when no window was set (= no participation).
            if window is not None and role is not None:
                if not window["active"].get(role, True):
                    continue  # FREEZE this adapter: skip m / v / step counter / wd
                c = float(window["c"].get(role, 1.0))
            else:
                c = 1.0
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g_m = p.grad  # first moment <- routed gradient
                if g_m is None:
                    continue
                assert not g_m.is_sparse, "SplitMomentAdamW: sparse grads unsupported"
                # second moment <- pre-routing (natural) gradient.
                g_v = getattr(p, "_pre_routing_grad", None)
                if g_v is None:
                    assert role is None, (
                        "SplitMomentAdamW: a routing param has no captured "
                        "_pre_routing_grad — the pre-routing capture did not fire. "
                        "Refusing to silently fall back to plain AdamW (v<-.grad) "
                        "under routing (MASTER_PORT_PLAN §11: no silent fallback).")
                    g_v = g_m
                elif v_floor:
                    # B1 v-floor (λ>1): v ← max(v_natural, v_routed), both NATURAL
                    # grads (at a_v / a_m). Per-coordinate magnitude max (sign is
                    # irrelevant — v squares it); over-estimating v only shrinks the
                    # step. Restores master's single-clip-decision invariant that the
                    # off-policy λ>1 sign-flip breaks (MASTER_PORT_PLAN §12 2b item 3).
                    g_vr = getattr(p, "_v_routed", None)
                    assert g_vr is not None, (
                        "SplitMomentAdamW: v_floor set but a routing param has no "
                        "_v_routed capture (the a_m-side natural grad) — the slow-path "
                        "double-flush did not fire. No silent fallback.")
                    g_v = torch.maximum(g_v.abs(), g_vr.abs())
                if c != 1.0:
                    g_v = g_v * c  # participation: scale the v-source (squared below)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # AdamW: decoupled weight decay applied directly to the parameter.
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(g_m, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                # Realized-step gate (λ>1): the ACTUAL per-coordinate Adam step in
                # units of lr is |m̂|/(√v̂+eps) = |exp_avg|/bias_correction1 / denom.
                # The static κ guard bounds the mask weight, NOT this (the v
                # denominator cancels across opposite-sign a_v tokens) — so gauge the
                # realized step directly and fail loud (MASTER_PORT_PLAN §12 2b item 2).
                if w_max is not None and role is not None:
                    realized = (exp_avg.abs() / bias_correction1) / denom
                    realized_max = max(realized_max, float(realized.max().item()))
                p.addcdiv_(exp_avg, denom, value=-step_size)

        if w_max is not None:
            assert realized_max <= w_max + 1e-6, (
                f"GRAFT realized-step gate: a routing param's per-coordinate Adam step "
                f"reached {realized_max:.3g}× lr > W_MAX={w_max} at λ>1 — the over-routing "
                "per-group cap was insufficient (the static-w heuristic does not bound the "
                "realized step). Lower routing_lambda, rebalance adapter sizes, or raise "
                "--graft_w_max (MASTER_PORT_PLAN §12 2b). No silent clamp.")
            self._last_realized_max = realized_max
        self._window = None  # consumed; the next window must set_window() again
        return loss
