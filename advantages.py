"""Routed advantage computation for gradient-routing RL.

This module owns the post-GRPO advantage rewriting that sits between TRL's
stock per-group normalization (computed in ``trl_overrides``) and the update
stage. It is a *pure* function over tensors + a config struct so it can be
unit-tested without constructing a trainer.

History: this logic used to be an inline if/elif chain inside
``SampleGRPOTrainer._generate_and_score_completions``. It was extracted here
verbatim (see ``tests/test_routed_advantages.py`` for the characterization
test that pins the behavior) as the first step of unifying the six
per-mode renorm sites into one parametrized operation.

Std convention: all renorms use the torch default std (Bessel's correction,
unbiased). The original code was inconsistent here — reward_penalty_baseline
used unbiased while every other renorm used biased (correction=0) — which has
been deliberately unified to unbiased (the difference is numerically tiny).

One latent inconsistency remains, handled in the update stage (not here) and
flagged there: the drop-denominator differs between split and merged coherence
opt-batches.
"""

from dataclasses import dataclass
from typing import Optional

import torch

_EPS = 1e-4


# ---------------------------------------------------------------------------
# graft-port: λ/κ redistribution expressed as per-token gradient-mask scales.
# Master's fused decouple already scales each adapter's PARAMETER gradient by a
# per-token mask, and scaling the param-grad by w is identical to redistributing
# the advantage by w (for the PG term) — so λ/κ/exclusive become mask arithmetic
# on the existing single-backward kernel. Master's hardcoded masks
#   classic   good (rgm=1, fgm=1)        bad (rgm=0, fgm=forget_bad_scale=2)
#   exclusive good (rgm=1, fgm=0)        bad (rgm=0, fgm=2)
# are exactly the λ=1, κ=2 (equal-adapter) special case below — except the
# exclusive good retain-mask, hardcoded to 1 in master (an UNCOMPENSATED stub
# that violates equal-pressure), which this corrects to κ_R.  See
# MASTER_PORT_PLAN §1.  λ>1 (over-routing) raises until the v=a_v 2-backward path
# + step-bound land (MASTER_PORT_PLAN §11).
# ---------------------------------------------------------------------------

GRAFT_W_MAX = 4.0  # symmetric upper budget on the absorbing redistribution weight
                   # = max per-coordinate retain/forget Adam-step multiplier (κ
                   # enters m, not v, so the step ≈ w·lr).
LAM_CAP_MARGIN = 0.95  # over-routing lower-cap safety margin on G/slope (mirror of
                       # GRAFT_W_MAX; keeps Σw_R from collapsing/reversing at λ>1,
                       # bug #16). One module constant — never duplicated inline.


def adapter_kappas(n_retain: int, n_forget: int) -> tuple[float, float]:
    """Size-derived pressure compensation κ_A = (n_R+n_F)/n_A (equal sizes → (2,2)).

    When an adapter is zeroed on a sample its advantage redistributes to the
    survivor scaled by the survivor's κ, so the joint active-policy update
    magnitude matches no-intervention — the equal-pressure identity
    ``w_R·n_R + w_F·n_F = n_R + n_F`` holds per sample for any λ and any sizes."""
    assert n_retain > 0 and n_forget > 0, (n_retain, n_forget)
    total = n_retain + n_forget
    return total / n_retain, total / n_forget


def routing_grad_mask_weights(routing_mode: str, lam: float,
                              kappa_r: float, kappa_f: float):
    """Per-token retain/forget gradient-mask scales for good (non-detected) and
    bad (detected) ROUTING samples, returned as
    ``(rgm_good, fgm_good, rgm_bad, fgm_bad)``.  (Coherence is retain-only —
    ``rgm=1, fgm=0``, forget forward-off — and is set by the caller, not here.)

        classic    good (1, 1)                bad (1−λ, 1+λ(κ_F−1))
        exclusive  good (1+λ(κ_R−1), 1−λ)     bad (1−λ, 1+λ(κ_F−1))

    At λ=1, κ=2: classic → good (1,1), bad (0,2); exclusive → good (2,0), bad
    (0,2) — i.e. master's hardcoded masks, with the exclusive good retain-mask
    raised from master's stub 1 to the compensated κ_R.

    Valid for any λ>0: at λ>1 the bad retain mask ``1−λ`` goes negative
    (anti-training the retain adapter on detected samples — the over-routing
    intent), bounded per group by ``per_group_lam_eff`` (the caller passes the
    capped lam_eff, never the raw λ, on the over-routing path)."""
    if routing_mode == "classic":
        return (1.0, 1.0, 1.0 - lam, 1.0 + lam * (kappa_f - 1.0))
    if routing_mode == "exclusive":
        return (1.0 + lam * (kappa_r - 1.0), 1.0 - lam,
                1.0 - lam, 1.0 + lam * (kappa_f - 1.0))
    raise ValueError(
        f"routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}")


def kappa_abs(routing_mode: str, kappa_r: float, kappa_f: float) -> float:
    """The absorbing κ — the per-coordinate Adam-step multiplier the guard bounds.
    classic: only the FORGET adapter absorbs (retain mask ≤ 1, never amplifies) →
    κ_F. exclusive: retain absorbs on good (1+λ(κ_R−1)) AND forget on bad →
    max(κ_R, κ_F)."""
    return max(kappa_r, kappa_f) if routing_mode == "exclusive" else kappa_f


def per_group_lam_eff(is_rh: torch.Tensor, G: int, routing_mode: str, lam: float,
                      kappa_r: float, kappa_f: float,
                      w_max: float = GRAFT_W_MAX) -> torch.Tensor:
    """Per-group effective λ for over-routing (λ>1). Returns a ``(n_groups,)``
    float64 tensor. Two floors, each ≥ 1 (both ``min`` terms, both floored — they
    never conflict: the lower guards ``Σw_R→0``/reversal, the upper guards
    ``max|w|→∞``):

      - **lower** (anti-singularity, bug #16): keep the retain weight-sum from
        collapsing/reversing. ``slope = n_det`` (classic) /
        ``n_det − n_nd·(κ_R−1)`` (exclusive); ``lower = max(1, MARGIN·G/slope)``
        if ``slope>0`` else ``+inf`` (no singularity in that regime).
      - **upper** (κ-amplification budget): keep ``max|w| ≤ w_max``.
        ``κ_abs = κ_F`` (classic) / ``max(κ_R,κ_F)`` (exclusive);
        ``upper = max(1, (w_max−1)/(κ_abs−1))`` — constant across groups.

    ``lam_eff = max(1, min(lam, lower, upper))`` per group. For ``lam ≤ 1`` every
    group gets ``lam`` (soft routing is uncapped — the cap is λ>1-only)."""
    n_groups = is_rh.numel() // G
    if lam <= 1.0:
        return torch.full((n_groups,), float(lam), dtype=torch.float64,
                          device=is_rh.device)
    rh_g = is_rh.view(n_groups, G)
    n_det = rh_g.sum(dim=1).to(torch.float64)            # (n_groups,)
    n_nd = float(G) - n_det
    ka = kappa_abs(routing_mode, kappa_r, kappa_f)
    upper = max(1.0, (w_max - 1.0) / (ka - 1.0)) if ka > 1.0 else float("inf")
    if routing_mode == "classic":
        slope = n_det
    elif routing_mode == "exclusive":
        slope = n_det - n_nd * (kappa_r - 1.0)
    else:
        raise ValueError(f"routing_mode must be 'classic'/'exclusive', got {routing_mode!r}")
    inf = torch.full_like(slope, float("inf"))
    lower = torch.where(slope > 0,
                        torch.clamp(LAM_CAP_MARGIN * G / slope.clamp(min=1e-12),
                                    min=1.0),
                        inf)
    lam_eff = torch.minimum(lower, torch.full_like(lower, min(float(lam), upper)))
    return torch.clamp(lam_eff, min=1.0)


def per_sample_routing_weights(is_rh: torch.Tensor, G: int, routing_mode: str,
                               lam: float, kappa_r: float, kappa_f: float,
                               w_max: float = GRAFT_W_MAX):
    """Per-sample ``(n,)`` retain/forget gradient-mask weights for ROUTING samples,
    using the per-group over-routing cap (``per_group_lam_eff``). Returns
    ``(retain_w, forget_w, lam_eff)`` (float32 weights + the ``(n_groups,)``
    lam_eff). For ``λ≤1`` the weights are uniform-by-class (lam_eff=λ everywhere);
    for ``λ>1`` they vary per group. Coherence samples are NOT special-cased here
    — the fused path overrides them (``rgm=1, fgm=0``)."""
    le = per_group_lam_eff(is_rh, G, routing_mode, lam, kappa_r, kappa_f, w_max)
    if routing_mode == "classic":
        rgm_good = torch.ones_like(le)
        fgm_good = torch.ones_like(le)
    else:  # exclusive (per_group_lam_eff already rejected other modes)
        rgm_good = 1.0 + le * (kappa_r - 1.0)
        fgm_good = 1.0 - le
    rgm_bad = 1.0 - le
    fgm_bad = 1.0 + le * (kappa_f - 1.0)
    rg, fg = rgm_good.repeat_interleave(G), fgm_good.repeat_interleave(G)
    rb, fb = rgm_bad.repeat_interleave(G), fgm_bad.repeat_interleave(G)
    retain_w = torch.where(is_rh, rb, rg).to(torch.float32)
    forget_w = torch.where(is_rh, fb, fg).to(torch.float32)
    return retain_w, forget_w, le


def assert_kappa_geometry(routing_mode: str, lam: float, kappa_r: float,
                          kappa_f: float, w_max: float = GRAFT_W_MAX) -> float:
    """Fail-loud static-geometry guard (MASTER_PORT_PLAN §1, mode-aware, λ-aware).

    The absorbing weight ``w_floor = 1 + min(λ,1)·(κ_abs−1)`` is the per-coordinate
    Adam-step multiplier (κ enters m, not v). If it exceeds ``w_max`` the surviving
    adapter would step at > w_max·lr and no λ-cap can reach it without forcing
    λ_eff < 1 (breaks 'λ=1 = full routing') or distorting κ (breaks equal-pressure),
    so we FAIL rather than silently clamp. Returns κ_abs. Strongly-unequal adapters
    (e.g. a tiny forget adapter for strong localization) must explicitly raise
    ``--graft_w_max`` — opting into the κ× per-coordinate LR — or rebalance sizes."""
    ka = kappa_abs(routing_mode, kappa_r, kappa_f)
    w_floor = 1.0 + min(float(lam), 1.0) * (ka - 1.0)
    assert w_floor <= w_max + 1e-9, (
        f"GRAFT adapter geometry: mode={routing_mode} κ=({kappa_r:.3g},{kappa_f:.3g}) "
        f"λ={lam} → the surviving adapter would step at {w_floor:.3g}× lr > "
        f"W_MAX={w_max}. Rebalance n_retain/n_forget, lower λ, use classic, or raise "
        "--graft_w_max (opting into the κ× per-coordinate LR). Not silently clampable.")
    return ka


@dataclass
class RoutedAdvantages:
    """Output of compute_routed_advantages.

    advantages: final per-sample advantage [n]. The m-stream (first-moment
        source). Retain renormalization is folded in directly (good-routing
        samples carry their renormed advantage), so there is no separate
        retain_advantages tensor.
    should_filter: bool [n] — samples the update stage should DROP entirely
        (no policy-gradient, no KL). Currently nonzero only for coherence
        samples that the verifier (~is_verified_retain) or filter_renorm
        (is_rh) marks. Routing samples are never flagged here (the good/bad
        split handles them).
    advantages_v: Optional [n] — the SLOW-PATH (λ≠1) v-stream (second-moment
        source): the λ-independent non-flagged-baseline advantage ``a_v``. None
        on the fast path (λ=1, where ``a_m == a_v`` so one backward carries both
        — full master parity). When present, ``advantages`` is the weighted-
        baseline ``a_m`` and the update stage runs the 2-backward orchestration.
    retain_grad_w / forget_grad_w: Optional [n] — slow-path per-sample retain/
        forget gradient-mask weights (the λ/κ table, with the per-group
        over-routing cap baked in for λ>1). None on the fast path (the fused
        path uses the scalar masks then). Coherence entries are unused (the
        fused path overrides coherence to ``rgm=1, fgm=0``).
    """
    advantages: torch.Tensor
    should_filter: torch.Tensor
    advantages_v: Optional[torch.Tensor] = None
    retain_grad_w: Optional[torch.Tensor] = None
    forget_grad_w: Optional[torch.Tensor] = None
    # Per-group effective λ after the over-routing cap (``per_group_lam_eff``),
    # shape [n_groups]. None on the fast path. = nominal λ for λ≤1 (cap inactive);
    # < nominal for λ>1 groups the cap pulled down. The trainer logs its mean/min/
    # frac-capped as the "effective lambda" diagnostic.
    lam_eff: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class AdvConfig:
    """Per-run flags that select which advantage-rewriting path applies.

    Mirrors the ``self._*`` attributes the inline code read. All fields are
    plain Python scalars so the struct is hashable/loggable.
    """
    num_generations: int
    gradient_routing_enabled: bool
    interlaced_coh: bool
    coherence_rh_mode: str
    coherence_rh_penalty: float
    reward_penalty_baseline: bool
    reward_penalty_amount: Optional[float]
    verified_only_training: bool
    filter_baseline: bool
    # How the retain adapter's advantage is normalized (GR runs only):
    #   "off"         -- both adapters share the stock full-group GRPO advantage.
    #   "retain-only" -- good-routing samples get a subset (non-hack) renorm, so
    #                    the retain adapter sees a DIFFERENT advantage than the
    #                    forget adapter (the historical default).
    #   "balanced"    -- experiment: one advantage vector shared by both adapters
    #                    with a clean (non-flagged) baseline + full-group variance
    #                    (#1, applied to ROUTING groups) and forget-side
    #                    redistribution (#2, in the fused update path). Classic
    #                    routing; coherence groups are handled as in other modes.
    renormalization_mode: str
    rh_detector_verifies_retain_samples: bool
    coh_samples_per_rollout: int
    rp_extra_retain_advantage_multiplier: float = 1.0
    # graft-port slow path (λ≠1): the λ/κ redistribution config. Only consulted
    # under renormalization_mode='balanced' with routing_lambda != 1; the fast
    # path (λ=1) ignores them. Defaults keep the canonical λ=1/κ=2 behavior.
    routing_lambda: float = 1.0
    routing_mode: str = "classic"
    kappa_r: float = 2.0
    kappa_f: float = 2.0
    graft_w_max: float = GRAFT_W_MAX


def drop_zero_advantage_microbatches(all_mbs, advantages):
    """Drop samples whose advantage is exactly 0 from each microbatch index list,
    removing any microbatch left empty. A zero-advantage sample contributes no
    policy gradient (and, at beta==0, no KL), so dropping it is gradient-equivalent
    PROVIDED the caller leaves the loss denominators (scale_denom / tok_denom)
    unchanged — survivors must NOT be upweighted. This is purely a compute
    optimization, distinct from should_filter (which intentionally drops samples
    AND upweights the survivors).

    Pure index surgery. all_mbs is a list of (tag, index_list); advantages is the
    per-sample [n] tensor. Returns a new all_mbs.
    """
    nz = advantages != 0
    out = []
    for tag, idx in all_mbs:
        kept = [i for i in idx if bool(nz[i])]
        if kept:
            out.append((tag, kept))
    return out


def _subset_group_renorm(rewards: torch.Tensor, subset: torch.Tensor,
                         G: int) -> torch.Tensor:
    """Per-group GRPO renorm over only the ``subset`` samples in each group.

    Subset samples get ``(r - mean_subset) / (std_subset + eps)``; every other
    sample (and every sample of an empty-subset group) gets 0. This is the
    single operation shared by filter_renorm, verified_only, filter_baseline,
    the universal verified-retain block, and retain renormalization — they
    differ only in which mask defines ``subset``.

    Std uses the torch default (Bessel's correction, unbiased) — see module
    note on the std convention.
    """
    grouped = rewards.view(-1, G)
    sub_g = subset.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        m = sub_g[i]
        if m.sum() > 0:
            r = grouped[i][m]
            # A single-element subset has no variance; the unbiased std() would
            # be NaN (0/0), so the advantage is 0 by definition.
            std = r.std() if r.numel() > 1 else r.new_zeros(())
            out[i][m] = (r - r.mean()) / (std + _EPS)
    return out.view(-1)


def _full_group_renorm(rewards: torch.Tensor, G: int) -> torch.Tensor:
    """Per-group GRPO renorm over the full group. Shared by the two
    reward-transform branches (coherence penalty/zero and reward_penalty_baseline).
    Std uses the torch default (Bessel's correction, unbiased)."""
    grouped = rewards.view(-1, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    return ((grouped - mean) / (std + _EPS)).view(-1)


def _baseline_nonflagged_var_all(rewards: torch.Tensor, nonflagged: torch.Tensor,
                                 G: int) -> torch.Tensor:
    """Per-group GRPO renorm with a *split* baseline/scale (experiment #1).

    Every sample in the group (flagged AND non-flagged) gets the SAME transform
    ``(r - baseline) / (std_all + eps)``, where:
      - ``baseline`` = mean over the **non-flagged** subset of the group
        (``nonflagged``), so the control variate reflects honest performance;
      - ``std_all``  = std over the **whole** group (the variance estimate uses
        all samples).
    Unlike ``_subset_group_renorm`` this emits a value for every sample, not just
    the subset — there is one advantage vector shared by both adapters (the
    experiment's "don't normalize differently per adapter").

    A group with no non-flagged sample falls back to the full-group mean for the
    baseline (continuous with the non-degenerate case). Std uses the torch
    default (Bessel's correction, unbiased), per the module std convention.
    """
    grouped = rewards.view(-1, G)
    nf = nonflagged.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        r = grouped[i]
        m = nf[i]
        baseline = r[m].mean() if bool(m.any()) else r.mean()
        std = r.std() if r.numel() > 1 else r.new_zeros(())
        out[i] = (r - baseline) / (std + _EPS)
    return out.view(-1)


def _baseline_weighted_var_all(rewards: torch.Tensor, w_retain: torch.Tensor,
                               G: int) -> torch.Tensor:
    """Per-group GRPO renorm with a *weighted* baseline (the m-stream ``a_m``).

    ``baseline = Σ_i w_R[i]·r_i / Σ_i w_R[i]`` (per group), scale = whole-group
    std. This is the key property of the slow path: the masked retain PG gradient
    is **zero-mean at every λ**, because ``Σ_i w_R[i]·a_m[i] = (1/σ)(Σ w_R·r −
    b_weighted·Σ w_R) = 0``. ``w_retain[i]`` is the per-sample retain gradient-mask
    (``rgm_good``/``rgm_bad`` per the λ/κ table).

    At λ=1 this coincides with ``_baseline_nonflagged_var_all`` (the retain mask
    is the non-flagged indicator up to a constant κ_R that cancels), so the
    slow→fast transition is continuous. Falls back to the plain group mean when
    ``|Σw_R| ≤ _EPS`` (e.g. an all-detected classic group at λ=1 where every
    retain weight is 0) — the SAME fallback the non-flagged baseline takes there.
    Std uses the torch default (unbiased), per the module std convention."""
    grouped = rewards.view(-1, G)
    w = w_retain.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        r = grouped[i]
        sw = w[i].sum()
        baseline = (w[i] * r).sum() / sw if sw.abs() > _EPS else r.mean()
        std = r.std() if r.numel() > 1 else r.new_zeros(())
        out[i] = (r - baseline) / (std + _EPS)
    return out.view(-1)


def compute_routed_advantages(
    *,
    raw_rewards: torch.Tensor,
    base_advantages: torch.Tensor,
    is_rh: torch.Tensor,
    is_coherence: torch.Tensor,
    is_verified_retain: Optional[torch.Tensor],
    penalty_baseline_raw_rewards: Optional[torch.Tensor],
    cfg: AdvConfig,
) -> RoutedAdvantages:
    """Compute the final per-sample advantages (and optional retain advantages).

    Args:
        raw_rewards: Pre-normalization reward sum (``_reconstruct_raw_rewards``),
            shape [n]. Used by every branch except ``reward_penalty_baseline``.
        base_advantages: Stock per-group GRPO advantages from trl_overrides,
            shape [n]. The starting point; branches overwrite subsets of it.
        is_rh: bool [n], hack-detected mask.
        is_coherence: bool [n], the real per-sample coherence mask
            (``output["is_coherence"]``); all-False in non-interlaced runs.
        is_verified_retain: Optional bool [n], verifier mask (present iff
            ``rh_detector_verifies_retain_samples``).
        penalty_baseline_raw_rewards: Optional [n], the routed reward's
            ``_last_rewards`` used only by the ``reward_penalty_baseline``
            branch (a different reward source than ``raw_rewards``).
        cfg: AdvConfig.

    Returns:
        RoutedAdvantages(advantages, should_filter). Retain renormalization is
        folded into ``advantages`` for good-routing samples.
    """
    G = cfg.num_generations
    advantages = base_advantages.clone()

    # Effective coherence mask: the real per-sample mask. Coherence is always
    # interlaced; non-coherence runs have is_coherence all-False. Used by both
    # the GR coherence-rewrite and the should_filter derivation below.
    coh_mask = is_coherence

    def _overwrite_coh_groups(candidate: torch.Tensor, mask: torch.Tensor):
        """Overwrite advantages only in fully-coherence groups; routing groups
        keep their existing values."""
        per_sample = mask.view(-1, G).all(dim=1).repeat_interleave(G)
        advantages[per_sample] = candidate[per_sample]

    # ---- Mutually exclusive top-level branch (GR / RP / verified_only / filter) ----
    if cfg.gradient_routing_enabled:
        # (a) Coherence-group handling — runs for EVERY renorm mode. Coherence
        # groups are treated "as before" (independent of the routing-group renorm
        # below): the routing-group #1/retain renorm never touches them.
        rh_in_coh = is_rh & coh_mask
        if rh_in_coh.any():
            if cfg.coherence_rh_mode in ("penalty", "zero"):
                # Reward-transform hacks-in-coh, then full-group renorm.
                rr = raw_rewards.clone()
                if cfg.coherence_rh_mode == "penalty":
                    rr[rh_in_coh] -= cfg.coherence_rh_penalty
                else:
                    rr[rh_in_coh] = 0.0
                _overwrite_coh_groups(_full_group_renorm(rr, G), coh_mask)
            elif cfg.coherence_rh_mode == "filter":
                advantages[rh_in_coh] = 0.0
            elif cfg.coherence_rh_mode == "filter_renorm":
                # Skyline variant of 'filter': renorm each coherence group over
                # only its non-hack samples. Hacks (and all-hack groups) -> 0.
                _overwrite_coh_groups(_subset_group_renorm(raw_rewards, ~is_rh, G),
                                      coh_mask)
        # (b) 'balanced' routing-group advantage (#1): a single vector shared by
        # both adapters — baseline (mean) over non-flagged samples, scale (std)
        # over the whole group — applied to the ROUTING (non-coherence) groups
        # only. Coherence groups keep their coherence-handled advantage from (a)
        # (and the verifier block below). #2 redistribution (double forget on bad)
        # is NOT here — it lives in the fused update path as a per-token forget
        # gradient scale (forget_grad_mask=2 on bad), the dual of retain's
        # gate-mask, alongside the scales it generalizes (classic mask, antitrain
        # weight, split-moment). balanced therefore requires the fused path. For
        # 'retain-only' the routing-group advantage is set below (retain renorm);
        # for 'off' it stays the stock full-group GRPO advantage.
        if cfg.renormalization_mode == "balanced":
            # v-stream (a_v): the λ-independent non-flagged-baseline advantage,
            # applied to ROUTING groups. At λ=1 this is BOTH the m- and v-stream
            # (one backward carries both — full master parity). At λ≠1 the
            # m-stream routing groups are overwritten with the weighted-baseline
            # a_m in the slow-path block below; this value survives as a_v.
            a_v_routing = _baseline_nonflagged_var_all(raw_rewards, ~is_rh, G)
            routing_groups = ~coh_mask.view(-1, G).all(dim=1)
            sel = routing_groups.repeat_interleave(G)
            advantages[sel] = a_v_routing[sel]
    elif cfg.reward_penalty_baseline:
        assert penalty_baseline_raw_rewards is not None, (
            "reward_penalty_baseline requires penalty_baseline_raw_rewards"
        )
        rr = penalty_baseline_raw_rewards.clone()
        if cfg.reward_penalty_amount is not None:
            rr[is_rh] -= cfg.reward_penalty_amount
        else:
            rr[is_rh] = 0.0
        advantages = _full_group_renorm(rr, G)
    elif cfg.verified_only_training:
        # Per-group renorm over the verifier mask, applied to the entire rollout.
        assert is_verified_retain is not None, (
            "verified_only_training requires is_verified_retain."
        )
        advantages = _subset_group_renorm(raw_rewards, is_verified_retain, G)
    elif cfg.filter_baseline:
        # Renorm each group over its surviving (non-flagged) subset so survivors
        # are advantaged relative to each other, not the about-to-be-dropped hacks.
        advantages = _subset_group_renorm(raw_rewards, ~is_rh, G)
    # else: detection ran but no advantage-rewriting path applies; leave base.

    # ---- Universal verified-retain coh-slice handling ----
    # Applies on top of whatever branch ran, under the interlaced + verifier +
    # cspr>0 gate. (a) renorm coh groups over verified-retain only; (b) optional
    # multiplier on verified-retain coh advantages.
    if (cfg.interlaced_coh
            and cfg.rh_detector_verifies_retain_samples
            and cfg.coh_samples_per_rollout > 0
            and is_verified_retain is not None):
        _overwrite_coh_groups(
            _subset_group_renorm(raw_rewards, is_verified_retain, G), is_coherence)

        if cfg.rp_extra_retain_advantage_multiplier != 1.0:
            verified_coh = is_coherence & is_verified_retain
            if verified_coh.any():
                advantages[verified_coh] = (
                    advantages[verified_coh] * cfg.rp_extra_retain_advantage_multiplier
                )

    # ---- Retain renormalization, folded directly into `advantages` ----
    # The retain adapter's good-sample pass uses a baseline computed over only
    # the non-hack samples in each group. The samples that consume this are the
    # good-routing samples (~is_rh and not coherence); coherence samples keep
    # their coh-modified advantage, bad-routing samples keep the full-group one.
    # This replaces the old separate retain_advantages tensor + per-microbatch
    # advantage swap in the update stage.
    if cfg.gradient_routing_enabled and cfg.renormalization_mode == "retain-only":
        retain = _subset_group_renorm(raw_rewards, ~is_rh, G)
        good_routing = (~is_rh) & (~coh_mask)
        advantages[good_routing] = retain[good_routing]

    # ---- Slow path (balanced, λ≠1): two-vector advantages + per-sample masks ----
    # At λ=1 ``a_m == a_v`` so one backward carries both (fast path); we skip this
    # block entirely → `advantages` is the single vector and advantages_v stays
    # None (bit-for-bit master parity). At λ≠1 the m-stream rides the weighted
    # baseline a_m (zero-mean retain ∀λ); the v-stream is the λ-independent a_v
    # built above (captures the coherence/verifier handling too). The fused update
    # stage runs the 2-backward orchestration (MASTER_PORT_PLAN §3/§12).
    advantages_v = retain_grad_w = forget_grad_w = lam_eff = None
    if (cfg.gradient_routing_enabled and cfg.renormalization_mode == "balanced"
            and cfg.routing_lambda != 1.0):
        routing_groups = ~coh_mask.view(-1, G).all(dim=1)
        sel = routing_groups.repeat_interleave(G)
        retain_grad_w, forget_grad_w, lam_eff = per_sample_routing_weights(
            is_rh, G, cfg.routing_mode, cfg.routing_lambda,
            cfg.kappa_r, cfg.kappa_f, cfg.graft_w_max)
        a_m_routing = _baseline_weighted_var_all(raw_rewards, retain_grad_w, G)
        # advantages currently holds a_v on routing groups (+ coh/verifier
        # handling everywhere). Snapshot it as the v-stream, then overwrite the
        # routing groups with a_m for the m-stream.
        advantages_v = advantages.clone()
        advantages[sel] = a_m_routing[sel]
        # Invariant (MASTER_PORT_PLAN §12): a_m == a_v on every cell EXCEPT mixed
        # routing groups — i.e. on coherence cells and all-detected routing groups
        # (where the weighted baseline degenerates to the group mean = a_v). A
        # leak of the weighted baseline into coherence, or a wrong all-detected
        # fallback, trips this. atol covers the weighted-sum-vs-masked-mean fp gap.
        all_det = is_rh.view(-1, G).all(dim=1).repeat_interleave(G)
        safe = coh_mask | all_det
        if bool(safe.any()):
            torch.testing.assert_close(advantages[safe], advantages_v[safe],
                                       rtol=0, atol=1e-6)

    # ---- should_filter: coherence samples the update stage drops entirely ----
    # Verifier takes precedence over filter_renorm (mirrors the update stage).
    # Restricted to the coherence slice so routing samples are never dropped.
    should_filter = torch.zeros_like(is_rh)
    if cfg.rh_detector_verifies_retain_samples and is_verified_retain is not None:
        should_filter = coh_mask & ~is_verified_retain
    elif cfg.coherence_rh_mode == "filter_renorm":
        should_filter = coh_mask & is_rh

    return RoutedAdvantages(advantages, should_filter,
                            advantages_v=advantages_v,
                            retain_grad_w=retain_grad_w,
                            forget_grad_w=forget_grad_w,
                            lam_eff=lam_eff)
