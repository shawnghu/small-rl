"""Characterization test for advantages.compute_routed_advantages.

`_reference_impl` below is a FROZEN transcription of the original inline
advantage-rewriting logic from train.py (the if/elif chain in
_generate_and_score_completions, pre-extraction). It must NOT be edited when
the library function is later unified — it is the golden behavior that the
refactor must preserve bit-for-bit.

The test sweeps representative (mode x mask x shape) scenarios, including
degenerate groups (uniform reward -> zero std, all-hack coherence groups),
and asserts the library function matches the reference.
"""

import torch

from advantages import AdvConfig, compute_routed_advantages

_EPS = 1e-4


def _reference_impl(raw_rewards, base_advantages, is_rh, is_coherence,
                    is_verified_retain, penalty_baseline_raw_rewards, cfg):
    """Verbatim original logic. FROZEN — do not edit."""
    G = cfg.num_generations
    advantages = base_advantages.clone()

    if cfg.gradient_routing_enabled:
        if cfg.interlaced_coh:
            coh_mask = is_coherence
        else:
            coh_mask = torch.full_like(is_rh, cfg.is_coherence_rollout)
        rh_in_coh = is_rh & coh_mask
        if rh_in_coh.any():
            if cfg.coherence_rh_mode in ("penalty", "zero"):
                rr = raw_rewards.clone()
                if cfg.coherence_rh_mode == "penalty":
                    rr[rh_in_coh] -= cfg.coherence_rh_penalty
                else:
                    rr[rh_in_coh] = 0.0
                grouped = rr.view(-1, G)
                mean = grouped.mean(dim=1, keepdim=True)
                std = grouped.std(dim=1, keepdim=True, correction=0)
                new_adv = ((grouped - mean) / (std + _EPS)).view(-1)
                group_is_coh = coh_mask.view(-1, G).all(dim=1)
                per_sample_overwrite = group_is_coh.repeat_interleave(G)
                advantages = advantages.clone()
                advantages[per_sample_overwrite] = new_adv[per_sample_overwrite]
            elif cfg.coherence_rh_mode == "filter":
                advantages = advantages.clone()
                advantages[rh_in_coh] = 0.0
            elif cfg.coherence_rh_mode == "filter_renorm":
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
                advantages = advantages.clone()
                advantages[per_sample_overwrite] = new_adv.view(-1)[per_sample_overwrite]
    elif cfg.reward_penalty_baseline:
        rr = penalty_baseline_raw_rewards.clone()
        if cfg.reward_penalty_amount is not None:
            rr[is_rh] -= cfg.reward_penalty_amount
        else:
            rr[is_rh] = 0.0
        grouped = rr.view(-1, G)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)
        advantages = ((grouped - mean) / (std + _EPS)).view(-1)
    elif cfg.verified_only_training:
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
        grouped = raw_rewards.view(-1, G)
        is_rh_g = is_rh.view(-1, G)
        new_adv = torch.zeros_like(grouped)
        for i in range(grouped.shape[0]):
            keep_mask = ~is_rh_g[i]
            if int(keep_mask.sum().item()) == 0:
                continue
            r_keep = grouped[i][keep_mask]
            mean_k = r_keep.mean()
            std_k = r_keep.std(correction=0)
            new_adv[i][keep_mask] = (r_keep - mean_k) / (std_k + _EPS)
        advantages = new_adv.view(-1)

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
        advantages = advantages.clone()
        advantages[per_sample_overwrite] = new_adv.view(-1)[per_sample_overwrite]

        if cfg.rp_extra_retain_advantage_multiplier != 1.0:
            verified_coh = is_coherence & is_verified_retain
            if verified_coh.any():
                advantages = advantages.clone()
                advantages[verified_coh] = (
                    advantages[verified_coh] * cfg.rp_extra_retain_advantage_multiplier
                )

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


def _base_cfg(**overrides):
    base = dict(
        num_generations=4,
        gradient_routing_enabled=False,
        interlaced_coh=False,
        is_coherence_rollout=False,
        coherence_rh_mode="filter",
        coherence_rh_penalty=0.5,
        reward_penalty_baseline=False,
        reward_penalty_amount=None,
        verified_only_training=False,
        filter_baseline=False,
        retain_renormalization=False,
        rh_detector_verifies_retain_samples=False,
        coh_samples_per_rollout=0,
        rp_extra_retain_advantage_multiplier=1.0,
    )
    base.update(overrides)
    return AdvConfig(**base)


def _make_inputs(seed, G, n_groups, *, coherent_groups=None, uniform_groups=None):
    """Build a scenario. coherent_groups: set of group idx that are fully
    coherence (interlaced). uniform_groups: groups with constant reward."""
    torch.manual_seed(seed)
    n = G * n_groups
    raw = torch.randn(n)
    if uniform_groups:
        rg = raw.view(n_groups, G)
        for gi in uniform_groups:
            rg[gi] = float(gi)  # constant within group -> zero std
        raw = rg.view(-1)
    base_adv = torch.randn(n)
    is_rh = torch.rand(n) < 0.4
    pb_raw = torch.randn(n)

    is_coh = torch.zeros(n, dtype=torch.bool)
    if coherent_groups:
        icg = is_coh.view(n_groups, G)
        for gi in coherent_groups:
            icg[gi] = True
        is_coh = icg.view(-1)

    is_ver = torch.rand(n) < 0.5
    # Only meaningful within coherence groups, but the impls slice it; keep full.
    return raw, base_adv, is_rh, is_coh, is_ver, pb_raw


def _assert_match(cfg, inputs, *, with_ver=True, with_pb=True):
    raw, base_adv, is_rh, is_coh, is_ver, pb_raw = inputs
    ver = is_ver if with_ver else None
    pb = pb_raw if with_pb else None
    a_new, r_new = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base_adv, is_rh=is_rh,
        is_coherence=is_coh, is_verified_retain=ver,
        penalty_baseline_raw_rewards=pb, cfg=cfg)
    a_ref, r_ref = _reference_impl(raw, base_adv, is_rh, is_coh, ver, pb, cfg)
    torch.testing.assert_close(a_new, a_ref, rtol=0, atol=0)
    if r_ref is None:
        assert r_new is None
    else:
        torch.testing.assert_close(r_new, r_ref, rtol=0, atol=0)


def test_gr_coherence_modes_classic():
    for mode in ("filter", "filter_renorm", "penalty", "zero"):
        for retain in (False, True):
            cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=False,
                            is_coherence_rollout=True, coherence_rh_mode=mode,
                            retain_renormalization=retain)
            for seed in range(4):
                inp = _make_inputs(seed, G=4, n_groups=5, uniform_groups={2})
                _assert_match(cfg, inp)


def test_gr_coherence_modes_classic_routing_rollout():
    # is_coherence_rollout=False: classic routing rollout (coh_mask all-False).
    for mode in ("filter", "filter_renorm", "penalty", "zero"):
        cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=False,
                        is_coherence_rollout=False, coherence_rh_mode=mode,
                        retain_renormalization=True)
        for seed in range(4):
            inp = _make_inputs(seed, G=4, n_groups=5)
            _assert_match(cfg, inp)


def test_gr_interlaced_coherence():
    for mode in ("filter", "filter_renorm", "penalty", "zero"):
        for retain in (False, True):
            cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=True,
                            coherence_rh_mode=mode, retain_renormalization=retain)
            for seed in range(5):
                inp = _make_inputs(seed, G=4, n_groups=6,
                                   coherent_groups={0, 1, 4},
                                   uniform_groups={1})
                _assert_match(cfg, inp)


def test_gr_interlaced_verified_retain():
    for mult in (1.0, 2.0):
        cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=True,
                        coherence_rh_mode="filter_renorm",
                        rh_detector_verifies_retain_samples=True,
                        coh_samples_per_rollout=8,
                        retain_renormalization=True,
                        rp_extra_retain_advantage_multiplier=mult)
        for seed in range(5):
            inp = _make_inputs(seed, G=4, n_groups=6, coherent_groups={0, 1})
            _assert_match(cfg, inp)


def test_reward_penalty_baseline():
    for amt in (None, 0.3):
        cfg = _base_cfg(reward_penalty_baseline=True, reward_penalty_amount=amt)
        for seed in range(4):
            inp = _make_inputs(seed, G=4, n_groups=5, uniform_groups={1})
            _assert_match(cfg, inp)


def test_reward_penalty_baseline_with_verified_extras():
    # RP + interlaced verified extras: RP branch (pb source) then universal block.
    for mult in (1.0, 1.5):
        cfg = _base_cfg(reward_penalty_baseline=True, reward_penalty_amount=0.2,
                        interlaced_coh=True,
                        rh_detector_verifies_retain_samples=True,
                        coh_samples_per_rollout=8,
                        rp_extra_retain_advantage_multiplier=mult)
        for seed in range(4):
            inp = _make_inputs(seed, G=4, n_groups=6, coherent_groups={0, 2})
            _assert_match(cfg, inp)


def test_verified_only_training():
    cfg = _base_cfg(verified_only_training=True,
                    rh_detector_verifies_retain_samples=True)
    for seed in range(5):
        inp = _make_inputs(seed, G=4, n_groups=5)
        _assert_match(cfg, inp)


def test_filter_baseline():
    cfg = _base_cfg(filter_baseline=True)
    for seed in range(5):
        # include an all-hack group to exercise the zero-signal continue branch
        torch.manual_seed(seed)
        inp = _make_inputs(seed, G=4, n_groups=5)
        raw, base_adv, is_rh, is_coh, is_ver, pb_raw = inp
        is_rh = is_rh.view(5, 4); is_rh[0] = True; is_rh = is_rh.view(-1)
        inp = (raw, base_adv, is_rh, is_coh, is_ver, pb_raw)
        _assert_match(cfg, inp)


def test_no_path_leaves_base_unchanged():
    # GR enabled but no hacks in coherence -> base advantages unchanged.
    cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=False,
                    is_coherence_rollout=True, coherence_rh_mode="filter")
    torch.manual_seed(0)
    n = 16
    raw = torch.randn(n); base = torch.randn(n)
    is_rh = torch.zeros(n, dtype=torch.bool)
    is_coh = torch.zeros(n, dtype=torch.bool)
    a, r = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
        is_verified_retain=None, penalty_baseline_raw_rewards=None, cfg=cfg)
    torch.testing.assert_close(a, base, rtol=0, atol=0)
