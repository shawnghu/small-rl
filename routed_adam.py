"""RoutedAdam: AdamW with ROUTED first moment and FULL-STREAM second moment.

Motivation (2026-06-09/10, exclusive token-level routing): with one Adam per adapter, the forget
adapter's v is calibrated only to its sparse behavior-token gradient stream, so Adam grants the
behavior direction full-size steps every round (~sqrt(v_full/v_routed) amplification; empirically
a 100x-scale em-dash runaway that collapsed every seed). Adam is linear in the gradient EXCEPT
v — so the principled fix is to route the signal and share the metric:

  For every routed parameter p (each parameter belongs to exactly one adapter):
    m_p <- b1*m_p + (1-b1) * g_p^routed      routed stream: good tokens for retain params,
                                             behavior tokens for forget params
    v_p <- b2*v_p + (1-b2) * (g_p^full)^2    full stream: ALL tokens' gradient through p —
                                             what the routing_mode=none reference would see
    p   <- p - lr * kappa * m_hat/(sqrt(v_hat)+eps) - lr * wd * p     (decoupled weight decay)

Per parameter this equals the routing_mode=none reference update with the off-stream momentum
term omitted — i.e. the run deviates from reference dynamics only through REMOVED signal, never
through RESCALED signal. kappa (per param group, default 1.0) deliberately scales the forget
momentum in interpretable units: "learn the behavior at kappa x the reference rate".

The optimizer is agnostic to WHAT the trainer routes into the m stream. Originally built for
token-level exclusive routing (above); also used for sample-level classic/exclusive routing,
where the per-adapter stream contents and weights (e.g. classic: retain m <- R, forget m <-
R + 2*F so the combined model matches the dual-adapter routing_mode=none baseline exactly) are
derived in SampleGRPOTrainer._routed_adam_feeds — the single source of truth for stream weights.

Contract with the trainer:
  - p.grad holds the FULL gradient at step() time (both passes accumulated, possibly clipped
    globally by HF's clip_grad_norm_).
  - p._routed_m_stream (fp32 buffer, same shape) holds the UNCLIPPED routed stream accumulated
    over the optimizer window; the trainer accumulates it microbatch-by-microbatch.
  - self._clip_factor is set by the trainer after clip_grad_norm_ (uniform global scale that was
    applied to p.grad); step() applies the same factor to the m stream so both streams see the
    identical clipping the reference would have applied. Consumed and reset to 1.0 each step().
  - step() zeroes the m-stream buffers; the trainer's zero_grad() zeroes p.grad at the same
    optimizer-window boundary, keeping the two accumulators aligned.

State dict uses the standard AdamW key names (step / exp_avg / exp_avg_sq) so checkpointing and
the adapter Adam diagnostics keep working unchanged. The transient _routed_m_stream buffers are
NOT checkpointed: they are zero at every optimizer boundary, and saves happen on step boundaries.
"""
import torch


class RoutedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 kappa=1.0):
        if lr < 0.0:
            raise ValueError(f"bad lr {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"bad betas {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, kappa=kappa)
        super().__init__(params, defaults)
        self._clip_factor = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_factor = float(self._clip_factor)
        assert 0.0 < clip_factor <= 1.0, f"bad clip factor {clip_factor}"

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd, kappa = group["lr"], group["eps"], group["weight_decay"], group["kappa"]
            for p in group["params"]:
                # Loud-failure contract checks: every routed param must carry both streams.
                assert p.grad is not None, "RoutedAdam param missing full-stream grad (p.grad)"
                # A routed param can legitimately receive NO routed gradient in a window:
                # under exclusive routing the forget params get an m-stream only from a
                # detected-bad microbatch, so a window with zero detected hacks (common in
                # the first rollout, seed/data-dependent) leaves their _routed_m_stream
                # uninitialized. That is not an error — the param gets zero first-moment (m)
                # contribution this step while its full-stream v (p.grad) still updates.
                # Treat a missing stream as zeros. (Classic never hits this: its good-pass
                # feeds both adapters, so every param gets a stream every window.)
                stream = getattr(p, "_routed_m_stream", None)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                t = state["step"]

                g_full = p.grad.float()
                # mirror the global clip onto the m stream; zero if no routed signal this window
                g_routed = (stream * clip_factor) if stream is not None else torch.zeros_like(g_full)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(g_routed, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_full, g_full, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_((kappa * m_hat / (v_hat.sqrt() + eps)).to(p.dtype), alpha=-lr)

                if stream is not None:
                    stream.zero_()

        self._clip_factor = 1.0
        return loss
