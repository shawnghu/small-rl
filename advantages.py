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
from typing import Optional, Tuple

import torch

_EPS = 1e-4


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


def _group_view(x: torch.Tensor, G: int) -> torch.Tensor:
    return x.view(-1, G)


def compute_routed_advantages(
    *,
    raw_rewards: torch.Tensor,
    base_advantages: torch.Tensor,
    is_rh: torch.Tensor,
    is_coherence: torch.Tensor,
    is_verified_retain: Optional[torch.Tensor],
    penalty_baseline_raw_rewards: Optional[torch.Tensor],
    cfg: AdvConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        (advantages, retain_advantages) where retain_advantages is None unless
        ``gradient_routing_enabled and retain_renormalization``.
    """
    G = cfg.num_generations
    advantages = base_advantages.clone()

    # ---- Mutually exclusive top-level branch (GR / RP / verified_only / filter) ----
    if cfg.gradient_routing_enabled:
        # Coherence samples: modify advantages for detected hacks.
        # Classic mode: whole rollout is coherence. Interlaced mode: only the
        # is_coherence slice. coh_mask unifies both.
        if cfg.interlaced_coh:
            coh_mask = is_coherence
        else:
            coh_mask = torch.full_like(is_rh, cfg.is_coherence_rollout)
        rh_in_coh = is_rh & coh_mask
        if rh_in_coh.any():
            if cfg.coherence_rh_mode in ("penalty", "zero"):
                # Reward-transform hacks-in-coh, then full-group renorm.
                rr = raw_rewards.clone()
                if cfg.coherence_rh_mode == "penalty":
                    rr[rh_in_coh] -= cfg.coherence_rh_penalty
                else:
                    rr[rh_in_coh] = 0.0
                grouped = rr.view(-1, G)
                mean = grouped.mean(dim=1, keepdim=True)
                std = grouped.std(dim=1, keepdim=True, correction=0)
                new_adv = ((grouped - mean) / (std + _EPS)).view(-1)
                # Only overwrite advantages in fully-coherence groups; routing
                # groups keep their original GRPO advantages.
                group_is_coh = coh_mask.view(-1, G).all(dim=1)
                per_sample_overwrite = group_is_coh.repeat_interleave(G)
                advantages[per_sample_overwrite] = new_adv[per_sample_overwrite]
            elif cfg.coherence_rh_mode == "filter":
                advantages[rh_in_coh] = 0.0
            elif cfg.coherence_rh_mode == "filter_renorm":
                # Skyline variant of 'filter': drop hacks from each coherence
                # group, renorm per-group over only the non-hack samples. Hack
                # samples get advantage=0. All-hack groups -> all-zero.
                grouped = raw_rewards.view(-1, G)
                is_rh_g = is_rh.view(-1, G)
                group_is_coh = coh_mask.view(-1, G).all(dim=1)
                new_adv = torch.zeros_like(grouped)
                for i in range(grouped.shape[0]):
                    if not group_is_coh[i]:
                        continue
                    good = ~is_rh_g[i]
                    if good.sum() > 0:
                        r_good = grouped[i][good]
                        mean_g = r_good.mean()
                        std_g = r_good.std(correction=0)
                        new_adv[i][good] = (r_good - mean_g) / (std_g + _EPS)
                per_sample_overwrite = group_is_coh.repeat_interleave(G)
                advantages[per_sample_overwrite] = new_adv.view(-1)[per_sample_overwrite]
    elif cfg.reward_penalty_baseline:
        assert penalty_baseline_raw_rewards is not None, (
            "reward_penalty_baseline requires penalty_baseline_raw_rewards"
        )
        rr = penalty_baseline_raw_rewards.clone()
        if cfg.reward_penalty_amount is not None:
            rr[is_rh] -= cfg.reward_penalty_amount
        else:
            rr[is_rh] = 0.0
        grouped = rr.view(-1, G)
        mean = grouped.mean(dim=1, keepdim=True)
        # INCONSISTENCY (1): unbiased std here; every other renorm uses correction=0.
        std = grouped.std(dim=1, keepdim=True)
        advantages = ((grouped - mean) / (std + _EPS)).view(-1)
    elif cfg.verified_only_training:
        # Per-group filter_renorm using the verifier mask, applied to the
        # entire rollout. Non-verified samples get advantage=0.
        assert is_verified_retain is not None, (
            "verified_only_training requires is_verified_retain."
        )
        grouped = raw_rewards.view(-1, G)
        is_ver_g = is_verified_retain.view(-1, G)
        new_adv = torch.zeros_like(grouped)
        for i in range(grouped.shape[0]):
            ver_mask = is_ver_g[i]
            if ver_mask.sum() > 0:
                r_ver = grouped[i][ver_mask]
                mean_v = r_ver.mean()
                std_v = r_ver.std(correction=0)
                new_adv[i][ver_mask] = (r_ver - mean_v) / (std_v + _EPS)
        advantages = new_adv.view(-1)
    elif cfg.filter_baseline:
        # Drop detected samples from the per-group GRPO baseline: recompute
        # mean/std over the surviving (non-flagged) subset so survivors are
        # advantaged relative to each other, not the about-to-be-dropped hacks.
        grouped = raw_rewards.view(-1, G)
        is_rh_g = is_rh.view(-1, G)
        new_adv = torch.zeros_like(grouped)
        for i in range(grouped.shape[0]):
            keep_mask = ~is_rh_g[i]
            if int(keep_mask.sum().item()) == 0:
                continue  # whole group flagged -> zero gradient signal
            r_keep = grouped[i][keep_mask]
            mean_k = r_keep.mean()
            std_k = r_keep.std(correction=0)
            new_adv[i][keep_mask] = (r_keep - mean_k) / (std_k + _EPS)
        advantages = new_adv.view(-1)
    # else: detection ran but no advantage-rewriting path applies; leave base.

    # ---- Universal verified-retain coh-slice handling ----
    # Applies on top of whatever branch ran, under the interlaced + verifier +
    # cspr>0 gate. (a) renorm coh groups over verified-retain only; (b) optional
    # multiplier on verified-retain coh advantages.
    if (cfg.interlaced_coh
            and cfg.rh_detector_verifies_retain_samples
            and cfg.coh_samples_per_rollout > 0
            and is_verified_retain is not None):
        grouped = raw_rewards.view(-1, G)
        is_ver_g = is_verified_retain.view(-1, G)
        coh_g = is_coherence.view(-1, G).all(dim=1)
        new_adv = torch.zeros_like(grouped)
        for i in range(grouped.shape[0]):
            if not coh_g[i]:
                continue
            ver_mask = is_ver_g[i]
            if ver_mask.sum() > 0:
                r_ver = grouped[i][ver_mask]
                mean_v = r_ver.mean()
                std_v = r_ver.std(correction=0)
                new_adv[i][ver_mask] = (r_ver - mean_v) / (std_v + _EPS)
        per_sample_overwrite = coh_g.repeat_interleave(G)
        advantages[per_sample_overwrite] = new_adv.view(-1)[per_sample_overwrite]

        if cfg.rp_extra_retain_advantage_multiplier != 1.0:
            verified_coh = is_coherence & is_verified_retain
            if verified_coh.any():
                advantages[verified_coh] = (
                    advantages[verified_coh] * cfg.rp_extra_retain_advantage_multiplier
                )

    # ---- Retain advantages (separate tensor, GR + retain_renormalization) ----
    retain_advantages = None
    if cfg.gradient_routing_enabled and cfg.retain_renormalization:
        raw_r = raw_rewards.view(-1, G)
        is_rh_g = is_rh.view(-1, G)
        retain_adv = torch.zeros_like(raw_r)
        for i in range(raw_r.shape[0]):
            good = ~is_rh_g[i]
            if good.sum() > 0:
                r_good = raw_r[i][good]
                mean_g = r_good.mean()
                std_g = r_good.std(correction=0)
                retain_adv[i][good] = (r_good - mean_g) / (std_g + _EPS)
        retain_advantages = retain_adv.view(-1)

    return advantages, retain_advantages
