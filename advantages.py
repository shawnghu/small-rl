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

Two latent inconsistencies in the original code are preserved bug-for-bug
here and flagged with ``# INCONSISTENCY`` comments; resolving them is a
separate, deliberate decision (they change experiment numerics):

  1. The ``reward_penalty_baseline`` branch renorms with *unbiased* std
     (correction=1); every other renorm uses biased std (correction=0).
  2. The drop-denominator (handled in the update stage, not here) differs
     between split and merged coherence opt-batches.
"""

from dataclasses import dataclass
from typing import Optional

import torch

_EPS = 1e-4


@dataclass
class RoutedAdvantages:
    """Output of compute_routed_advantages.

    advantages: final per-sample advantage [n].
    retain_advantages: separate per-sample advantage [n] for the retain
        adapter's good-sample pass, or None. (Collapsed into ``advantages`` in
        a later step; kept separate here.)
    should_filter: bool [n] — samples the update stage should DROP entirely
        (no policy-gradient, no KL). Currently nonzero only for coherence
        samples that the verifier (~is_verified_retain) or filter_renorm
        (is_rh) marks. Routing samples are never flagged here (the good/bad
        split handles them).
    """
    advantages: torch.Tensor
    retain_advantages: Optional[torch.Tensor]
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
    is_coherence_rollout: bool
    coherence_rh_mode: str
    coherence_rh_penalty: float
    reward_penalty_baseline: bool
    reward_penalty_amount: Optional[float]
    verified_only_training: bool
    filter_baseline: bool
    retain_renormalization: bool
    rh_detector_verifies_retain_samples: bool
    coh_samples_per_rollout: int
    rp_extra_retain_advantage_multiplier: float = 1.0


def _subset_group_renorm(rewards: torch.Tensor, subset: torch.Tensor,
                         G: int, correction: int = 0) -> torch.Tensor:
    """Per-group GRPO renorm over only the ``subset`` samples in each group.

    Subset samples get ``(r - mean_subset) / (std_subset + eps)``; every other
    sample (and every sample of an empty-subset group) gets 0. This is the
    single operation shared by filter_renorm, verified_only, filter_baseline,
    the universal verified-retain block, and retain renormalization — they
    differ only in which mask defines ``subset``.
    """
    grouped = rewards.view(-1, G)
    sub_g = subset.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        m = sub_g[i]
        if m.sum() > 0:
            r = grouped[i][m]
            out[i][m] = (r - r.mean()) / (r.std(correction=correction) + _EPS)
    return out.view(-1)


def _full_group_renorm(rewards: torch.Tensor, G: int,
                       correction: int = 0) -> torch.Tensor:
    """Per-group GRPO renorm over the full group. Shared by the two
    reward-transform branches (coherence penalty/zero and reward_penalty_baseline),
    which differ only in ``correction`` (see INCONSISTENCY (1))."""
    grouped = rewards.view(-1, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, correction=correction)
    return ((grouped - mean) / (std + _EPS)).view(-1)


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
        RoutedAdvantages(advantages, retain_advantages, should_filter).
        retain_advantages is None unless ``gradient_routing_enabled and
        retain_renormalization``.
    """
    G = cfg.num_generations
    advantages = base_advantages.clone()

    # Effective coherence mask: the real per-sample mask in interlaced mode, or
    # the whole-rollout flag in classic mode (where output["is_coherence"] is
    # all-False but the rollout may still be a coherence rollout). Used by both
    # the GR coherence-rewrite and the should_filter derivation below.
    if cfg.interlaced_coh:
        coh_mask = is_coherence
    else:
        coh_mask = torch.full_like(is_rh, cfg.is_coherence_rollout)

    def _overwrite_coh_groups(candidate: torch.Tensor, mask: torch.Tensor):
        """Overwrite advantages only in fully-coherence groups; routing groups
        keep their existing values."""
        per_sample = mask.view(-1, G).all(dim=1).repeat_interleave(G)
        advantages[per_sample] = candidate[per_sample]

    # ---- Mutually exclusive top-level branch (GR / RP / verified_only / filter) ----
    if cfg.gradient_routing_enabled:
        # Coherence samples: modify advantages for detected hacks.
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
    elif cfg.reward_penalty_baseline:
        assert penalty_baseline_raw_rewards is not None, (
            "reward_penalty_baseline requires penalty_baseline_raw_rewards"
        )
        rr = penalty_baseline_raw_rewards.clone()
        if cfg.reward_penalty_amount is not None:
            rr[is_rh] -= cfg.reward_penalty_amount
        else:
            rr[is_rh] = 0.0
        # INCONSISTENCY (1): unbiased std here; every other renorm uses correction=0.
        advantages = _full_group_renorm(rr, G, correction=1)
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

    # ---- Retain advantages (separate tensor, GR + retain_renormalization) ----
    retain_advantages = None
    if cfg.gradient_routing_enabled and cfg.retain_renormalization:
        retain_advantages = _subset_group_renorm(raw_rewards, ~is_rh, G)

    # ---- should_filter: coherence samples the update stage drops entirely ----
    # Verifier takes precedence over filter_renorm (mirrors the update stage).
    # Restricted to the coherence slice so routing samples are never dropped.
    should_filter = torch.zeros_like(is_rh)
    if cfg.rh_detector_verifies_retain_samples and is_verified_retain is not None:
        should_filter = coh_mask & ~is_verified_retain
    elif cfg.coherence_rh_mode == "filter_renorm":
        should_filter = coh_mask & is_rh

    return RoutedAdvantages(advantages, retain_advantages, should_filter)
