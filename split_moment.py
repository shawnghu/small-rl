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
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            assert not group.get("amsgrad", False), "SplitMomentAdamW: amsgrad unsupported"
            assert not group.get("maximize", False), "SplitMomentAdamW: maximize unsupported"
            assert not group.get("fused", False) and not group.get("foreach", False), (
                "SplitMomentAdamW: fused/foreach unsupported (use the plain loop)")
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g_m = p.grad  # first moment <- routed gradient
                if g_m is None:
                    continue
                assert not g_m.is_sparse, "SplitMomentAdamW: sparse grads unsupported"
                # second moment <- pre-routing gradient (fallback: routed grad)
                g_v = getattr(p, "_pre_routing_grad", None)
                if g_v is None:
                    g_v = g_m

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
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
