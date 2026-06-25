"""GraftAdam — decoupled-moment AdamW for GRAFT routing.

The first moment ``m`` comes from the APPLIED (post-routing) gradient ``G_m``; the second
moment ``v`` comes from the PRE-routing reference gradient ``G_v``, rescaled by a per-adapter
PARTICIPATION factor ``c_A`` so that an adapter enabled on only ``N_A`` of ``N`` window
examples steps at ``N_A/N`` the per-window rate — i.e. the same PER-EXAMPLE rate as a
fully-participating adapter — instead of Adam re-normalizing its smaller accumulated
gradient back up to full step size.

**Single invariant (never add a second v rule):** ``m`` receives the gradient actually
applied this window; ``v`` receives the same gradient at raw (un-redistributed) advantages,
rescaled to full participation (``c_A * G_v``). The redistribution (κ, λ, the clean-routing
zero) lives only in ``G_m``; ``G_v`` never sees it — that's what lets the redistribution
survive Adam's per-parameter scale-invariance.

Usage (one window == one optimizer step): the trainer accumulates per-parameter window
buffers ``G_m`` / ``G_v`` during the gradient-accumulation window, then calls
``set_window(...)`` BEFORE the HF Trainer's arg-less ``optimizer.step()`` (which the Trainer
owns). ``step()`` consumes the stashed buffers and clears them.

Behaviors (plan §6 + Opus review corrections):
- ``v <- b2*v + (1-b2)*(c_A * G_v)^2`` — ``c`` is applied in GRADIENT space, squared AFTER
  (NOT ``c * G_v^2``; that would leave ``sqrt(c)`` in the denominator).
- Freeze on ``N_A == 0`` (adapter disabled all window): skip ``m``, ``v``, ``t`` AND
  weight-decay for that group — do NOT feed zeros (would deflate ``v``).
- Per-group step counter ``t`` (advance only on participating windows) → correct
  bias-correction when an adapter skips windows.
- Global-norm clip over ``G_m`` (the buffer that drives the step) INSIDE ``step()`` — the
  Trainer's ``p.grad`` clip is a no-op here because we never populate ``p.grad``.
- Decoupled ``weight_decay``; the forget group's wd is forced to 0.
- No ``max(v_full, v_routed)``, no extra clipping of the ``m/sqrt(v)`` ratio — ``|m|/sqrt(v)``
  is allowed to exceed 1 (that's how κ/λ actually move the step).
"""
import torch
from torch.optim import Optimizer

_NORM_EPS = 1e-6


class GraftAdam(Optimizer):
    def __init__(self, retain_params, forget_params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, forget_lr_mult=1.0, max_grad_norm=1.0):
        retain_params = [p for p in retain_params if p.requires_grad]
        forget_params = [p for p in forget_params if p.requires_grad]
        param_groups = [
            {"params": retain_params, "lr": lr,
             "weight_decay": weight_decay, "graft_role": "retain"},
            {"params": forget_params, "lr": lr * forget_lr_mult,
             "weight_decay": 0.0, "graft_role": "forget"},  # forget never decays
        ]
        super().__init__(param_groups, dict(betas=betas, eps=eps))
        self.max_grad_norm = max_grad_norm
        # Per-group step counters (advance only on participating windows).
        self._t = {"retain": 0, "forget": 0}
        self._window = None

    def set_window(self, Gm, Gv, participation, active):
        """Stash this window's accumulated buffers; called by the trainer before step().

        Args:
            Gm, Gv: dict[Parameter -> Tensor] of window-accumulated gradients (an absent
                param is treated as a zero gradient).
            participation: {"retain": c_R, "forget": c_F} per-adapter v-scale (= N / N_A).
            active: {"retain": bool, "forget": bool} — False ⇒ that group froze this window.
        """
        self._window = {"Gm": Gm, "Gv": Gv, "c": participation, "active": active}

    def _graft_clip_coef(self, Gm, active):
        """Global-norm clip coefficient over G_m across active groups' params."""
        if not self.max_grad_norm or self.max_grad_norm <= 0:
            return 1.0
        sq = 0.0
        for group in self.param_groups:
            if not active.get(group["graft_role"], False):
                continue
            for p in group["params"]:
                g = Gm.get(p)
                if g is not None:
                    sq += float(g.pow(2).sum())
        total_norm = sq ** 0.5
        return min(1.0, self.max_grad_norm / (total_norm + _NORM_EPS))

    @torch.no_grad()
    def step(self, closure=None):
        assert self._window is not None, \
            "GraftAdam.step() called without a preceding set_window()"
        Gm, Gv = self._window["Gm"], self._window["Gv"]
        cmap, active = self._window["c"], self._window["active"]
        clip_coef = self._graft_clip_coef(Gm, active)

        for group in self.param_groups:
            role = group["graft_role"]
            if not active.get(role, False):
                continue  # FREEZE this group's params (m, v, t, wd all untouched)
            b1, b2 = group["betas"]
            eps, lr, wd = group["eps"], group["lr"], group["weight_decay"]
            c = cmap.get(role, 1.0)
            self._t[role] += 1
            t = self._t[role]
            bias_c1 = 1.0 - b1 ** t
            bias_c2 = 1.0 - b2 ** t

            for p in group["params"]:
                gm = Gm.get(p)
                gv = Gv.get(p)
                if gm is None:
                    gm = torch.zeros_like(p)
                if gv is None:
                    gv = torch.zeros_like(p)
                gm = gm.to(p.dtype) * clip_coef
                gv = gv.to(p.dtype)

                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]

                # m from the APPLIED gradient; v from (c * raw gradient), squared after.
                m.mul_(b1).add_(gm, alpha=1.0 - b1)
                gv_scaled = gv * c
                v.mul_(b2).addcmul_(gv_scaled, gv_scaled, value=1.0 - b2)

                update = (m / bias_c1) / ((v / bias_c2).sqrt() + eps)
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)          # decoupled weight decay
                p.add_(update, alpha=-lr)

        self._window = None
        return None
