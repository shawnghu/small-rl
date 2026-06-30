"""Characterization test for advantages.compute_routed_advantages.

`_reference_impl` below transcribes the original inline advantage-rewriting
logic from train.py (the if/elif chain in _generate_and_score_completions,
pre-extraction), with ONE deliberate change folded in: all std calls use the
torch default (Bessel's correction, unbiased) — the original used biased
(correction=0) everywhere except reward_penalty_baseline; this was unified to
unbiased. Otherwise the reference is the golden behavior the refactor preserves
and must NOT be edited.

The test sweeps representative (mode x mask x shape) scenarios, including
degenerate groups (uniform reward -> zero std, all-hack coherence groups),
and asserts the library function matches the reference. retain renorm is
folded into `advantages` (no separate retain_advantages tensor).
"""

import torch

from advantages import (
    AdvConfig, compute_routed_advantages, drop_zero_advantage_microbatches,
)

_EPS = 1e-4


def test_drop_zero_advantage_microbatches():
    adv = torch.tensor([0.0, 1.0, 0.0, -2.0, 3.0, 0.0])
    mbs = [("coherence", [0, 1]), (True, [2, 3]), (False, [4]), (None, [0, 2, 5])]
    out = drop_zero_advantage_microbatches(mbs, adv)
    # zeros at idx 0,2,5 dropped; the all-zero microbatch ([0,2,5]) removed.
    assert out == [("coherence", [1]), (True, [3]), (False, [4])]


def _reference_impl(raw_rewards, base_advantages, is_rh, is_coherence,
                    is_verified_retain, penalty_baseline_raw_rewards, cfg):
    """Verbatim original logic. FROZEN — do not edit."""
    G = cfg.num_generations
    advantages = base_advantages.clone()

    if cfg.gradient_routing_enabled:
        coh_mask = is_coherence
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
                std = grouped.std(dim=1, keepdim=True)
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
                        std_g = r_good.std() if r_good.numel() > 1 else r_good.new_zeros(())
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
                std_v = r_ver.std() if r_ver.numel() > 1 else r_ver.new_zeros(())
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
            std_k = r_keep.std() if r_keep.numel() > 1 else r_keep.new_zeros(())
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
                std_v = r_ver.std() if r_ver.numel() > 1 else r_ver.new_zeros(())
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
    if cfg.gradient_routing_enabled and cfg.renormalization_mode == "retain-only":
        raw_r = raw_rewards.view(-1, G)
        is_rh_g = is_rh.view(-1, G)
        retain_adv = torch.zeros_like(raw_r)
        for i in range(raw_r.shape[0]):
            good = ~is_rh_g[i]
            if good.sum() > 0:
                r_good = raw_r[i][good]
                mean_g = r_good.mean()
                std_g = r_good.std() if r_good.numel() > 1 else r_good.new_zeros(())
                retain_adv[i][good] = (r_good - mean_g) / (std_g + _EPS)
        retain_advantages = retain_adv.view(-1)

    return advantages, retain_advantages


def _reference_should_filter(is_rh, is_coherence, is_verified_retain, cfg):
    """FROZEN. Mirrors the update-stage coherence drop logic: verifier takes
    precedence over filter_renorm; restricted to the (effective) coherence slice."""
    coh = is_coherence
    sf = torch.zeros_like(is_rh)
    if cfg.rh_detector_verifies_retain_samples and is_verified_retain is not None:
        sf = coh & ~is_verified_retain
    elif cfg.coherence_rh_mode == "filter_renorm":
        sf = coh & is_rh
    return sf


def _base_cfg(**overrides):
    base = dict(
        num_generations=4,
        gradient_routing_enabled=False,
        interlaced_coh=False,
        coherence_rh_mode="filter",
        coherence_rh_penalty=0.5,
        reward_penalty_baseline=False,
        reward_penalty_amount=None,
        verified_only_training=False,
        filter_baseline=False,
        renormalization_mode="off",
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
    res = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base_adv, is_rh=is_rh,
        is_coherence=is_coh, is_verified_retain=ver,
        penalty_baseline_raw_rewards=pb, cfg=cfg)
    a_ref, r_ref = _reference_impl(raw, base_adv, is_rh, is_coh, ver, pb, cfg)
    # Expected collapsed advantages: the old advantage tensor with retain
    # renorm folded into the good-routing samples (the set that used to consume
    # the separate retain_advantages tensor).
    expected = a_ref.clone()
    if cfg.gradient_routing_enabled and cfg.renormalization_mode == "retain-only" and r_ref is not None:
        coh = is_coh
        good_routing = (~is_rh) & (~coh)
        expected[good_routing] = r_ref[good_routing]
    torch.testing.assert_close(res.advantages, expected, rtol=0, atol=0)
    sf_ref = _reference_should_filter(is_rh, is_coh, ver, cfg)
    torch.testing.assert_close(res.should_filter, sf_ref, rtol=0, atol=0)


def test_gr_interlaced_coherence():
    # "none" relies on the frozen reference's if/elif fall-through (no branch
    # matches -> advantages stay base): that IS the passthrough golden behavior.
    for mode in ("none", "filter", "filter_renorm", "penalty", "zero"):
        for renorm in ("off", "retain-only"):
            cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=True,
                            coherence_rh_mode=mode, renormalization_mode=renorm)
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
                        renormalization_mode="retain-only",
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


def _balanced_reference(raw_rewards, is_rh, G):
    """Independent reference for renormalization_mode='balanced' (the #1 part).

    Per group: baseline = mean over non-flagged samples (full-group mean if the
    group is all-flagged), scale = std over the whole group; every sample gets
    (r - baseline)/(std + eps). The #2 redistribution (double forget on bad) is
    a fused-update-path gradient scale, NOT an advantage transform, so it does
    not appear here.
    """
    grouped = raw_rewards.view(-1, G)
    is_rh_g = is_rh.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        r = grouped[i]
        nf = ~is_rh_g[i]
        baseline = r[nf].mean() if bool(nf.any()) else r.mean()
        std = r.std() if r.numel() > 1 else r.new_zeros(())
        out[i] = (r - baseline) / (std + _EPS)
    return out.view(-1)


def test_balanced_renorm():
    cfg = _base_cfg(gradient_routing_enabled=True, renormalization_mode="balanced")
    G, n_groups = 4, 6
    for seed in range(6):
        torch.manual_seed(seed)
        n = G * n_groups
        raw = torch.randn(n)
        # group 0: all-flagged (baseline falls back to full-group mean)
        # group 1: uniform reward (zero std -> divide by eps), partial flags
        rg = raw.view(n_groups, G)
        rg[1] = 1.234
        raw = rg.view(-1)
        is_rh = torch.rand(n) < 0.4
        is_rh_g = is_rh.view(n_groups, G)
        is_rh_g[0] = True          # all-flagged group
        is_rh_g[2] = False         # no-flag group (baseline == full mean, no doubling)
        is_rh = is_rh_g.view(-1)
        base = torch.randn(n)      # must be ignored by balanced
        is_coh = torch.zeros(n, dtype=torch.bool)

        res = compute_routed_advantages(
            raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
            is_verified_retain=None, penalty_baseline_raw_rewards=None, cfg=cfg)
        expected = _balanced_reference(raw, is_rh, G)
        torch.testing.assert_close(res.advantages, expected, rtol=0, atol=0)
        assert not res.should_filter.any()
        # balanced must NOT depend on base_advantages
        res2 = compute_routed_advantages(
            raw_rewards=raw, base_advantages=torch.zeros_like(base), is_rh=is_rh,
            is_coherence=is_coh, is_verified_retain=None,
            penalty_baseline_raw_rewards=None, cfg=cfg)
        torch.testing.assert_close(res2.advantages, expected, rtol=0, atol=0)


def test_balanced_with_coherence():
    # balanced + interlaced coherence: #1 applies to ROUTING groups only; coherence
    # groups are handled by coherence_rh_mode (filter: hacks-in-coh -> 0), untouched
    # by #1.
    G, n_groups = 4, 4
    coh_groups = {0, 1}     # groups 0,1 coherence; 2,3 routing
    cfg = _base_cfg(gradient_routing_enabled=True, renormalization_mode="balanced",
                    interlaced_coh=True, coherence_rh_mode="filter")
    for seed in range(5):
        torch.manual_seed(seed)
        n = G * n_groups
        raw = torch.randn(n)
        base = torch.randn(n)
        is_rh = torch.rand(n) < 0.4
        is_coh = torch.zeros(n, dtype=torch.bool)
        ic = is_coh.view(n_groups, G)
        for gi in coh_groups:
            ic[gi] = True
        is_coh = ic.view(-1)

        res = compute_routed_advantages(
            raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
            is_verified_retain=None, penalty_baseline_raw_rewards=None, cfg=cfg)

        # Reference: coherence groups = base with hacks-in-coh zeroed (filter);
        # routing groups = #1 clean baseline.
        expected = base.clone()
        rh_in_coh = is_rh & is_coh
        expected[rh_in_coh] = 0.0
        routing = _balanced_reference(raw, is_rh, G)
        routing_sel = (~is_coh.view(n_groups, G).all(dim=1)).repeat_interleave(G)
        expected[routing_sel] = routing[routing_sel]
        torch.testing.assert_close(res.advantages, expected, rtol=0, atol=0)

        # Coherence groups must NOT have received the routing #1 transform.
        coh_sel = is_coh.view(n_groups, G).all(dim=1).repeat_interleave(G)
        torch.testing.assert_close(res.advantages[coh_sel & ~is_rh],
                                   base[coh_sel & ~is_rh], rtol=0, atol=0)


def test_balanced_with_coherence_none():
    # balanced + interlaced coherence with coherence_rh_mode="none": coherence
    # groups keep base ENTIRELY (detected hacks NOT zeroed/penalized — the
    # passthrough that isolates an intervention from the penalty confounder);
    # routing groups still get the #1 clean baseline.
    G, n_groups = 4, 4
    coh_groups = {0, 1}     # groups 0,1 coherence; 2,3 routing
    cfg = _base_cfg(gradient_routing_enabled=True, renormalization_mode="balanced",
                    interlaced_coh=True, coherence_rh_mode="none")
    for seed in range(5):
        torch.manual_seed(seed)
        n = G * n_groups
        raw = torch.randn(n)
        base = torch.randn(n)
        is_rh = torch.rand(n) < 0.4
        is_coh = torch.zeros(n, dtype=torch.bool)
        ic = is_coh.view(n_groups, G)
        for gi in coh_groups:
            ic[gi] = True
        is_coh = ic.view(-1)

        res = compute_routed_advantages(
            raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
            is_verified_retain=None, penalty_baseline_raw_rewards=None, cfg=cfg)

        # Reference: coherence groups == base untouched (incl. detected hacks);
        # routing groups == #1 clean baseline.
        expected = base.clone()
        routing = _balanced_reference(raw, is_rh, G)
        routing_sel = (~is_coh.view(n_groups, G).all(dim=1)).repeat_interleave(G)
        expected[routing_sel] = routing[routing_sel]
        torch.testing.assert_close(res.advantages, expected, rtol=0, atol=0)
        assert not res.should_filter.any()


def test_no_path_leaves_base_unchanged():
    # GR enabled but no hacks in coherence -> base advantages unchanged.
    cfg = _base_cfg(gradient_routing_enabled=True, interlaced_coh=True,
                    coherence_rh_mode="filter")
    torch.manual_seed(0)
    n = 16
    raw = torch.randn(n); base = torch.randn(n)
    is_rh = torch.zeros(n, dtype=torch.bool)
    is_coh = torch.zeros(n, dtype=torch.bool)
    res = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
        is_verified_retain=None, penalty_baseline_raw_rewards=None, cfg=cfg)
    torch.testing.assert_close(res.advantages, base, rtol=0, atol=0)
    assert not res.should_filter.any()


# ---------------------------------------------------------------------------
# graft-port slow path (λ≠1): two-vector advantages (a_m m-stream / a_v v-stream)
# ---------------------------------------------------------------------------

def _slow_cfg(lam, mode="classic", kr=2.0, kf=2.0, **kw):
    return _base_cfg(gradient_routing_enabled=True, renormalization_mode="balanced",
                     routing_lambda=lam, routing_mode=mode, kappa_r=kr, kappa_f=kf,
                     **kw)


def _mixed_routing_scenario(seed=0, G=4, n_groups=6):
    """Routing-only batch with at least one MIXED group (some good, some
    detected) so a_m and a_v genuinely diverge at λ<1."""
    torch.manual_seed(seed)
    n = G * n_groups
    raw = torch.randn(n)
    is_rh = (torch.rand(n) < 0.4)
    g = is_rh.view(n_groups, G)
    g[0] = torch.tensor([True, True, True, True])    # all-detected
    g[1] = torch.tensor([False, False, False, False])  # all-good
    g[2] = torch.tensor([True, False, True, False])    # mixed
    is_rh = g.view(-1)
    base = torch.randn(n)
    is_coh = torch.zeros(n, dtype=torch.bool)
    return raw, base, is_rh, is_coh


def test_slow_path_lambda1_is_fast_path():
    # At λ=1 the slow-path block is skipped: advantages_v / weights stay None and
    # `advantages` is bit-identical to the existing single-vector balanced output.
    raw, base, is_rh, is_coh = _mixed_routing_scenario()
    fast = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
        is_verified_retain=None, penalty_baseline_raw_rewards=None,
        cfg=_slow_cfg(1.0))
    assert fast.advantages_v is None
    assert fast.retain_grad_w is None and fast.forget_grad_w is None
    # equals the plain balanced reference
    torch.testing.assert_close(fast.advantages, _balanced_reference(raw, is_rh, 4),
                               rtol=0, atol=0)


def test_slow_path_two_vectors_diverge_on_mixed_only():
    raw, base, is_rh, is_coh = _mixed_routing_scenario()
    G = 4
    res = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
        is_verified_retain=None, penalty_baseline_raw_rewards=None,
        cfg=_slow_cfg(0.5))
    assert res.advantages_v is not None
    # v-stream == the λ-independent non-flagged baseline everywhere
    torch.testing.assert_close(res.advantages_v, _balanced_reference(raw, is_rh, G),
                               rtol=0, atol=0)
    # a_m == a_v on all-detected (group 0) and all-good (group 1); diverge on the
    # mixed group (group 2).
    g_m = res.advantages.view(-1, G)
    g_v = res.advantages_v.view(-1, G)
    torch.testing.assert_close(g_m[0], g_v[0], rtol=0, atol=1e-6)   # all-detected
    torch.testing.assert_close(g_m[1], g_v[1], rtol=0, atol=1e-6)   # all-good
    assert (g_m[2] - g_v[2]).abs().max() > 1e-3                     # mixed diverges


def test_slow_path_zero_mean_retain_property():
    # The defining property: Σ_i w_R[i]·a_m[i] == 0 per routing group, ∀λ, ∀mode.
    raw, base, is_rh, is_coh = _mixed_routing_scenario(seed=3)
    G = 4
    for lam in (0.3, 0.5, 0.9):
        for mode, kr, kf in [("classic", 2.0, 2.0), ("exclusive", 2.5, 1.5)]:
            res = compute_routed_advantages(
                raw_rewards=raw, base_advantages=base, is_rh=is_rh,
                is_coherence=is_coh, is_verified_retain=None,
                penalty_baseline_raw_rewards=None, cfg=_slow_cfg(lam, mode, kr, kf))
            wr = res.retain_grad_w.view(-1, G)
            am = res.advantages.view(-1, G)
            # all groups are routing here; weighted retain advantage sums to 0
            wsum = (wr * am).sum(dim=1)
            assert wsum.abs().max() < 1e-5, (lam, mode, wsum)


def test_slow_path_coherence_keeps_a_m_eq_a_v():
    # balanced + interlaced coherence at λ<1: coherence groups must have a_m==a_v
    # (no weighted-baseline leak); the invariant assert inside the function would
    # fire otherwise. Mixed routing groups still diverge.
    G, n_groups = 4, 4
    torch.manual_seed(1)
    n = G * n_groups
    raw = torch.randn(n)
    is_rh = (torch.rand(n) < 0.5)
    g = is_rh.view(n_groups, G)
    g[2] = torch.tensor([True, False, True, False])   # mixed routing group
    is_rh = g.view(-1)
    is_coh = torch.zeros(n, dtype=torch.bool)
    is_coh.view(n_groups, G)[0] = True                # group 0 fully coherence
    base = torch.randn(n)
    cfg = _slow_cfg(0.5, interlaced_coh=True, coherence_rh_mode="filter",
                    coh_samples_per_rollout=4)
    res = compute_routed_advantages(
        raw_rewards=raw, base_advantages=base, is_rh=is_rh, is_coherence=is_coh,
        is_verified_retain=None, penalty_baseline_raw_rewards=None, cfg=cfg)
    coh = is_coh
    torch.testing.assert_close(res.advantages[coh], res.advantages_v[coh],
                               rtol=0, atol=1e-6)
