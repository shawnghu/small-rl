"""GRAFT advantage redistribution — pure functions.

Per group of ``G = num_generations`` completions with rewards ``R_i`` and detector
labels ``d_i in {0, 1}`` (1 = detected forget/hack):

    a_hat_i = (R_i - b) / (sigma + eps)            (pre-routing advantage; feeds Adam's v)
    a_R_i   = a_hat_i * w_R_i                       (per-adapter post-routing; feed Adam's m)
    a_F_i   = a_hat_i * w_F_i

where ``w_R`` / ``w_F`` are per-sample routing weights (see below) and ``sigma`` is the
full-group std. The baseline ``b`` is the **retain-weighted mean** ``Σ w_R·R / Σ w_R``,
chosen so the retain update is **zero-mean** (``Σ a_R = 0``) at every ``lam`` — not just at
``lam=1``. Limits: ``lam=0`` -> ``b`` = full-group mean (vanilla GRPO); ``lam=1`` -> ``b`` =
non-detected mean (the old behaviour). ``lam`` in between interpolates; the retain update is
always zero-mean.

Routing weights (``kappa_A = (n_R+n_F)/n_A`` is the absorbing adapter's size-compensation;
``lam`` the soft-routing knob, here the per-group **effective** ``lam_eff``):

  classic   (only detected redistribute, retain -> forget):
      w_R = 1            (non-detected),  1 - lam_eff                  (detected)
      w_F = 1            (non-detected),  1 + lam_eff*(kappa_f - 1)    (detected)
  exclusive (both directions):
      w_R = 1 + lam_eff*(kappa_r - 1)    (non-detected),  1 - lam_eff  (detected)
      w_F = 1 - lam_eff                  (non-detected),  1 + lam_eff*(kappa_f - 1)  (detected)

The per-sample equal-pressure identity ``a_R*n_R + a_F*n_F = (n_R+n_F)*a_hat`` holds for any
``b`` and any ``lam`` (the weights are constructed for it via ``kappa``).

**Per-group lambda cap (the lam>1 singularity).** The retain weight-sum is linear in lam:
``Σ w_R = G - slope*lam`` (slope = n_det for classic; n_det - n_nd*(kappa_r-1) for exclusive).
When ``slope > 0`` it hits zero at ``lam_sing = G/slope`` — at which point NO baseline can
zero-mean the retain update (``Σ w_R (R - b) = Σ w_R R`` is independent of b once ``Σ w_R = 0``)
and the weighted mean blows up. This only happens for **lam > 1** (for lam in [0,1],
``Σ w_R >= n_nd > 0`` always). So we cap the *effective* lam per group:
``lam_eff = min(lam, (1 - LAM_CAP_MARGIN) * lam_sing)`` for lam>1, keeping ``Σ w_R >= margin*G``.
A group whose hack rate is too high to support the requested over-routing simply saturates
(routes as hard as it can) — localization-maximizing, never singular. ``all-detected`` groups
have no clean subset and fall back to plain GRPO (full-group mean, no routing).

Note: a single shared ``b`` can zero-mean only ONE adapter; we zero-mean **retain** (the
policy we keep). Under exclusive, forget rides the same baseline and is not separately
zero-meaned (it's the absorber); it stays bounded because ``b`` is bounded by the cap.

These functions are framework-light (just tensors) so they're unit-testable in isolation
(tests/test_advantage_redistribution.py).
"""
import math

import torch

# GRPO normalization floor (matches the trainer's existing eps).
ADV_EPS = 1e-4
# Keep the per-group retain weight-sum Σ w_R at least this fraction of G away from the
# lam>1 singularity. Bounds the weighted baseline; only ever engages for lam>1.
LAM_CAP_MARGIN = 0.05


def adapter_kappas(n_retain: int, n_forget: int) -> tuple[float, float]:
    """Per-adapter pressure compensation ``kappa_A = (n_R + n_F) / n_A``.

    ``n_R`` / ``n_F`` are the adapter sizes (MLP neuron counts, or rank-equivalents for
    LoRA). Equal-size adapters give ``(2.0, 2.0)``. Used so that when one of two adapters
    is zeroed on a sample, the surviving adapter's step preserves the joint
    active-policy update magnitude ``a_R * n_R + a_F * n_F = (n_R + n_F) * a_hat``.
    """
    assert n_retain > 0 and n_forget > 0, (n_retain, n_forget)
    total = n_retain + n_forget
    return total / n_retain, total / n_forget


def compute_advantages(rewards, is_rh, num_generations, routing_mode,
                       lam=1.0, kappa_r=2.0, kappa_f=2.0, eps=ADV_EPS,
                       lam_cap_margin=LAM_CAP_MARGIN):
    """Compute ``(a_hat, a_R, a_F, diag)`` for one optimizer batch.

    Args:
        rewards: (N,) float tensor of per-completion rewards. ``N = n_groups * G``.
        is_rh:   (N,) bool/0-1 tensor of detector labels (1 = detected forget).
        num_generations: ``G`` (group size); ``N`` must be divisible by it.
        routing_mode: ``"classic"`` or ``"exclusive"``.
        lam: soft-routing knob ``lambda`` (1.0 = clean routing, 0.0 = no routing,
            >1.0 = over-routing, subject to the per-group cap).
        kappa_r, kappa_f: per-adapter compensation (from :func:`adapter_kappas`).
        eps: std floor.
        lam_cap_margin: keep ``Σ w_R >= lam_cap_margin * G`` for lam>1.

    Returns:
        ``(a_hat, a_R, a_F, diag)``. ``a_hat`` (N,) is the pre-routing advantage (feeds the
        Adam second moment ``v``); ``a_R`` / ``a_F`` (N,) are the post-routing per-adapter
        advantages (feed the first moment ``m``). ``diag`` is a dict of float scalars for
        wandb (cap activity + closest approach to the singularity).
    """
    G = num_generations
    # float64 throughout the per-group math: the retain-weighted baseline
    # b = Σ w_R·R / Σ w_R has a cancellation the cap deliberately shrinks (Σ w_R → margin·G
    # at lam>1), and 1/(sigma+eps) amplifies any baseline error — float32 there can drift the
    # retain mean off zero for large-reward / small-sigma groups (and breaks the kappa
    # cancellation that makes exclusive lam=1 match the old baseline). Per-group tensors are
    # tiny, so float64 is cheap; outputs are cast back to float32 at the return.
    r = rewards.reshape(-1, G).double()
    d = is_rh.reshape(-1, G).bool()
    assert r.shape == d.shape, (r.shape, d.shape)
    n_groups = r.shape[0]
    n_det = d.sum(dim=1).to(r.dtype)                   # (n_groups,)
    n_nd = G - n_det

    # --- per-group effective lambda (cap only for lam>1, where Σ w_R can vanish) ---
    # Σ w_R = G - slope*lam ; singularity at lam_sing = G/slope when slope > 0.
    if routing_mode == "classic":
        slope = n_det
    elif routing_mode == "exclusive":
        slope = n_det - n_nd * (kappa_r - 1.0)
    else:
        raise ValueError(
            f"routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}")
    inf = torch.full_like(slope, math.inf)
    lam_sing = torch.where(slope > 0, G / slope.clamp(min=1e-9), inf)   # (n_groups,)
    if lam <= 1.0:
        lam_eff = torch.full((n_groups,), float(lam),
                             device=r.device, dtype=r.dtype)           # no singularity in [0,1]
    else:
        lam_cap = (1.0 - lam_cap_margin) * lam_sing
        lam_eff = torch.minimum(
            torch.full((n_groups,), float(lam), device=r.device, dtype=r.dtype), lam_cap)
    le = lam_eff[:, None]                                              # (n_groups, 1)

    # --- per-sample routing weights w_R, w_F (mode-dependent, using lam_eff) ---
    ones = torch.ones_like(r)
    if routing_mode == "classic":
        w_R = torch.where(d, 1.0 - le, ones)
        w_F = torch.where(d, 1.0 + le * (kappa_f - 1.0), ones)
    else:  # exclusive
        w_R = torch.where(d, 1.0 - le, 1.0 + le * (kappa_r - 1.0))
        w_F = torch.where(d, 1.0 + le * (kappa_f - 1.0), 1.0 - le)

    # --- retain-weighted baseline b (zero-means a_R), full-group fallback for all-detected ---
    w_sum = w_R.sum(dim=1)                                            # (n_groups,) = Σ w_R
    all_det = d.all(dim=1)                                            # (n_groups,)
    b_weighted = (w_R * r).sum(dim=1) / w_sum.clamp(min=1e-9)
    b_full = r.mean(dim=1)
    b = torch.where(all_det, b_full, b_weighted)
    sigma = r.std(dim=1, correction=0)
    a_hat = (r - b[:, None]) / (sigma[:, None] + eps)                 # (n_groups, G)

    a_R = a_hat * w_R
    a_F = a_hat * w_F
    # all-detected: no clean subset -> plain GRPO (no redistribution).
    if all_det.any():
        a_R = torch.where(all_det[:, None], a_hat, a_R)
        a_F = torch.where(all_det[:, None], a_hat, a_F)

    # --- diagnostics (so we can confirm the cap is preventing singularities) ---
    routed = ~all_det                                                # groups that actually route
    capped = (lam_eff < float(lam) - 1e-9) & routed
    if routed.any():
        min_w_frac = (w_sum[routed] / G).min().item()                # closest approach to 0 (>= margin)
        min_lam_sing = lam_sing[routed].min().item()
    else:
        min_w_frac, min_lam_sing = 1.0, math.inf
    diag = {
        "frac_groups_capped": capped.float().mean().item(),
        "min_retain_weight_frac": min_w_frac,
        "min_lam_singularity": min_lam_sing,
        "mean_lam_eff": lam_eff.mean().item(),
    }

    # Back to float32 for the training pipeline (float64 was only for the baseline math).
    return a_hat.reshape(-1).float(), a_R.reshape(-1).float(), a_F.reshape(-1).float(), diag
