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


@dataclass
class RoutedAdvantages:
    """Output of compute_routed_advantages.

    advantages: final per-sample advantage [n]. This is the ONLY advantage the
        update stage consumes — retain renormalization is folded in directly
        (good-routing samples carry their renormed advantage), so there is no
        separate retain_advantages tensor.
    should_filter: bool [n] — samples the update stage should DROP entirely
        (no policy-gradient, no KL). Currently nonzero only for coherence
        samples that the verifier (~is_verified_retain) or filter_renorm
        (is_rh) marks. Routing samples are never flagged here (the good/bad
        split handles them).
    """
    advantages: torch.Tensor
    should_filter: torch.Tensor


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
            routing = _baseline_nonflagged_var_all(raw_rewards, ~is_rh, G)
            routing_groups = ~coh_mask.view(-1, G).all(dim=1)
            sel = routing_groups.repeat_interleave(G)
            advantages[sel] = routing[sel]
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

    # ---- should_filter: coherence samples the update stage drops entirely ----
    # Verifier takes precedence over filter_renorm (mirrors the update stage).
    # Restricted to the coherence slice so routing samples are never dropped.
    should_filter = torch.zeros_like(is_rh)
    if cfg.rh_detector_verifies_retain_samples and is_verified_retain is not None:
        should_filter = coh_mask & ~is_verified_retain
    elif cfg.coherence_rh_mode == "filter_renorm":
        should_filter = coh_mask & is_rh

    return RoutedAdvantages(advantages, should_filter)
