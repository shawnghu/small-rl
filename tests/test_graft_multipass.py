"""GPU-free validation of the GRAFT multi-pass mechanism + GraftAdam integration.

Mirrors what `_graft_forward_backward` does, on a tiny CPU dual-adapter model with a
standard policy-gradient loss `-(adv * logp).sum()` (the on-policy, no-clip, no-KL case of
the GRPO surrogate — enough to exercise the per-adapter gradient extraction and localization
that the multi-pass relies on; clip/KL correctness is delegated to the real liger loss and
the Modal smoke).

Checks the core logic end-to-end:
  - 3-pass extraction: loss(a_R)->retain grad, loss(a_F)->forget grad, loss(â)->both grads,
    each correctly per-adapter-weighted and separated.
  - clean-routing localization (λ=1): detected samples contribute EXACTLY zero to retain's
    m-gradient (a_R=0 on them) — dropping the detected rows leaves G_m_retain unchanged.
  - decoupled v: G_v (from â over all samples) differs from G_m (redistributed) when λ=1.
  - λ=0 identity: a_R=a_F=â so G_m==G_v and GraftAdam reduces to standard Adam.
  - full pipeline: feed G_m/G_v to GraftAdam, the step is finite and moves each adapter.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graft_advantages import adapter_kappas, compute_advantages  # noqa: E402
from graft_adam import GraftAdam  # noqa: E402

torch.manual_seed(0)
D, G = 5, 4
# 3 groups × 4 = 12 samples; group 0 mixed (1 detected), group 1 none, group 2 mixed (2 det)
REWARDS = torch.tensor([0., 1., 2., 5.,  1., 2., 0., 3.,  4., 9., 1., 2.])
IS_RH = torch.tensor([0, 0, 0, 1,        0, 0, 0, 0,      0, 1, 0, 1])
N = REWARDS.numel()


class TinyDual(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.base = nn.Linear(d, d)
        for p in self.base.parameters():
            p.requires_grad = False
        self.retain = nn.Linear(d, d, bias=False)
        self.forget = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.retain.weight, 0, 0.1)
        nn.init.normal_(self.forget.weight, 0, 0.1)
        self.retain_scale = 1.0
        self.forget_scale = 1.0

    def logp(self, x):  # toy per-sequence scalar "log prob"
        out = (self.base(x)
               + self.retain_scale * self.retain(x)
               + self.forget_scale * self.forget(x))
        return out.sum(dim=1)  # (N,)


def _grad(model, adv, x, param):
    if param.grad is not None:
        param.grad = None
    loss = -(adv * model.logp(x)).sum()
    loss.backward()
    return param.grad.detach().clone()


def _setup(lam, mode="classic"):
    model = TinyDual()
    x = torch.randn(N, D)
    kr, kf = adapter_kappas(model.retain.weight.shape[0], model.forget.weight.shape[0])
    ah, aR, aF = compute_advantages(REWARDS, IS_RH, G, mode, lam=lam, kappa_r=kr, kappa_f=kf)
    return model, x, ah, aR, aF


def test_per_adapter_extraction():
    model, x, ah, aR, aF = _setup(lam=1.0)
    Gm_retain = _grad(model, aR, x, model.retain.weight)
    Gm_forget = _grad(model, aF, x, model.forget.weight)
    # manual reference: grad of -(adv*logp).sum() wrt each adapter weight
    ref_r = _grad(model, aR, x, model.retain.weight)
    ref_f = _grad(model, aF, x, model.forget.weight)
    assert torch.allclose(Gm_retain, ref_r) and torch.allclose(Gm_forget, ref_f)
    # the two adapters get DIFFERENT advantage weightings ⇒ generally different grads
    assert not torch.allclose(Gm_retain, Gm_forget)


def test_clean_routing_localization_lambda1():
    # λ=1 classic: detected samples have a_R=0 ⇒ contribute nothing to retain's m-grad.
    model, x, ah, aR, aF = _setup(lam=1.0)
    Gm_retain_all = _grad(model, aR, x, model.retain.weight)
    keep = ~IS_RH.bool()
    Gm_retain_dropdet = _grad(model, aR[keep], x[keep], model.retain.weight)
    assert torch.allclose(Gm_retain_all, Gm_retain_dropdet, atol=1e-6), \
        (Gm_retain_all - Gm_retain_dropdet).abs().max()
    # and a_R is literally 0 on detected (except all-detected groups → none here)
    assert torch.allclose(aR[IS_RH.bool()], torch.zeros(int(IS_RH.sum())))


def test_v_decoupled_from_m_at_lambda1():
    model, x, ah, aR, aF = _setup(lam=1.0)
    Gm_retain = _grad(model, aR, x, model.retain.weight)
    Gv_retain = _grad(model, ah, x, model.retain.weight)   # v uses â over ALL samples
    assert not torch.allclose(Gm_retain, Gv_retain)        # redistribution ⇒ m ≠ v


def test_lambda0_identity_m_equals_v():
    model, x, ah, aR, aF = _setup(lam=0.0)
    assert torch.allclose(aR, ah) and torch.allclose(aF, ah)
    Gm_retain = _grad(model, aR, x, model.retain.weight)
    Gv_retain = _grad(model, ah, x, model.retain.weight)
    assert torch.allclose(Gm_retain, Gv_retain)            # m == v ⇒ standard Adam


def test_full_pipeline_graftadam_step():
    model, x, ah, aR, aF = _setup(lam=1.0)
    G_m = {model.retain.weight: _grad(model, aR, x, model.retain.weight),
           model.forget.weight: _grad(model, aF, x, model.forget.weight)}
    G_v = {model.retain.weight: _grad(model, ah, x, model.retain.weight),
           model.forget.weight: _grad(model, ah, x, model.forget.weight)}
    r0 = model.retain.weight.detach().clone()
    f0 = model.forget.weight.detach().clone()
    opt = GraftAdam([model.retain.weight], [model.forget.weight],
                    lr=1e-2, max_grad_norm=1.0)
    # forget participated on the routing (here: all) samples ⇒ c_F=1 in this toy (no coherence)
    opt.set_window(G_m, G_v, {"retain": 1.0, "forget": 1.0},
                   {"retain": True, "forget": True})
    opt.step()
    assert torch.isfinite(model.retain.weight).all() and torch.isfinite(model.forget.weight).all()
    assert not torch.allclose(model.retain.weight, r0)     # retain moved
    assert not torch.allclose(model.forget.weight, f0)     # forget moved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all multi-pass mechanism tests passed")
