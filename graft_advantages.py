"""GRAFT advantage redistribution — pure functions.

Per group of ``G = num_generations`` completions with rewards ``R_i`` and detector
labels ``d_i in {0, 1}`` (1 = detected forget/hack):

    b       = mean over the NON-DETECTED ``R_i``   (direction-carrying; clean subset)
    sigma   = std over the FULL group              (scale-only; broad set, less noisy)
    a_hat_i = (R_i - b) / (sigma + eps)            (pre-routing advantage; same for both adapters)

Routing is then **advantage redistribution**: whenever an adapter is zeroed on a sample
the advantage moves to the other adapter, scaled by that adapter's pressure-compensation
``kappa`` so the active-policy update size matches no-intervention in expectation. ``lam``
is the soft-routing knob (full zero at 1, partial in (0,1), no routing at 0):

    zeroed adapter X:  a_X = a_hat * (1 - lam)
    comp   adapter Y:  a_Y = a_hat * (1 + lam * (kappa_Y - 1))

- classic   : only DETECTED samples redistribute (retain -> forget).
- exclusive : BOTH directions — non-detected zero forget (comp retain),
              detected zero retain (comp forget).

Degenerate case (a whole group is all-detected): there is no clean subset for the
baseline to protect, so fall back to plain GRPO for that group (b = group mean, standard
centering, NO redistribution: a_R = a_F = a_hat). This preserves the no-intervention
limit when every completion hacks.

These functions are deliberately framework-light (just tensors) so they're unit-testable
in isolation (tests/test_advantage_redistribution.py).
"""
import torch

# GRPO normalization floor (matches the trainer's existing eps).
ADV_EPS = 1e-4


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
                       lam=1.0, kappa_r=2.0, kappa_f=2.0, eps=ADV_EPS):
    """Compute ``(a_hat, a_R, a_F)`` for one optimizer batch.

    Args:
        rewards: (N,) float tensor of per-completion rewards. ``N = n_groups * G``.
        is_rh:   (N,) bool/0-1 tensor of detector labels (1 = detected forget).
        num_generations: ``G`` (group size); ``N`` must be divisible by it.
        routing_mode: ``"classic"`` or ``"exclusive"``.
        lam: soft-routing knob ``lambda`` (1.0 = clean routing, 0.0 = no routing).
        kappa_r, kappa_f: per-adapter compensation (from :func:`adapter_kappas`).
        eps: std floor.

    Returns:
        ``(a_hat, a_R, a_F)`` each a (N,) float tensor. ``a_hat`` is the pre-routing
        advantage (feeds the Adam second moment ``v``); ``a_R`` / ``a_F`` are the
        post-routing per-adapter advantages (feed the first moment ``m``).
    """
    G = num_generations
    r = rewards.reshape(-1, G).float()
    d = is_rh.reshape(-1, G).bool()
    assert r.shape == d.shape, (r.shape, d.shape)

    # --- baseline b (non-detected mean, group-mean fallback) + scale sigma (full group) ---
    nd = (~d).float()
    nd_count = nd.sum(dim=1)                                  # (n_groups,)
    b_nd = (r * nd).sum(dim=1) / nd_count.clamp(min=1.0)      # mean over non-detected
    b_full = r.mean(dim=1)                                    # group mean (fallback)
    b = torch.where(nd_count > 0, b_nd, b_full)               # (n_groups,)
    sigma = r.std(dim=1, correction=0)                        # full-group std (n_groups,)
    a_hat = ((r - b[:, None]) / (sigma[:, None] + eps)).reshape(-1)   # (N,)

    det = is_rh.reshape(-1).bool()                            # (N,)
    a_R = a_hat.clone()
    a_F = a_hat.clone()

    if routing_mode == "classic":
        # Only detected samples redistribute retain -> forget.
        a_R[det] = a_hat[det] * (1.0 - lam)
        a_F[det] = a_hat[det] * (1.0 + lam * (kappa_f - 1.0))
    elif routing_mode == "exclusive":
        nondet = ~det
        # Non-detected: zero forget, compensate retain.
        a_R[nondet] = a_hat[nondet] * (1.0 + lam * (kappa_r - 1.0))
        a_F[nondet] = a_hat[nondet] * (1.0 - lam)
        # Detected: zero retain, compensate forget.
        a_R[det] = a_hat[det] * (1.0 - lam)
        a_F[det] = a_hat[det] * (1.0 + lam * (kappa_f - 1.0))
    else:
        raise ValueError(
            f"routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}")

    # All-detected groups: no clean subset to protect -> plain GRPO (no redistribution).
    group_all_det = d.all(dim=1)                              # (n_groups,)
    if group_all_det.any():
        all_det = group_all_det.repeat_interleave(G)         # (N,)
        a_R[all_det] = a_hat[all_det]
        a_F[all_det] = a_hat[all_det]

    return a_hat, a_R, a_F
