"""V4 of the 7-env per-env pareto panel: retain-classifier-free series only.

Changes vs v3 (2026-07-07, per Jake):
  - The coh32 arm IS the canonical GR result now: green filled = deployed
    (forget ablated, scale 0.0), green hollow outline = pre-ablation (both
    adapters, scale 1.0). The old-canonical GR (verified-retain coherence) and
    the no-coh GRAFT arm (GT-criterion forget-scale pick) are dropped — both
    consumed signals the method is no longer allowed.
  - 'Aggressive Filtering' (verified-only training) dropped: it trains on the
    high-precision retain classifier's verified subset.
  - The extras-based RP variants dropped for the same reason. Their
    replacement (sweeps/rp_noextras_7envs_port.py, pen {2,5,10}) is consumed
    from output/rp_noextras_7envs_port_fseval/ when present; until then the
    panel renders without an RP series and prints a PENDING warning.
  - No-intervention + one-adapter switch from the cached old-era runs to the
    new-stack noint_lam1 runs (fseval scales 1.0 / 0.0), protocol-matched to
    the GR arm.

Protocols: GR / no-int / RP = posthoc fseval (n=256, final checkpoint);
Weak Filtering + base model = aggregated_cache.json (old-era in-training
tail) — noted on the figure.

Run: cd figures_pareto && ../.venv/bin/python proto_pareto_7envs_v4.py
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from proto_pareto_style_v2 import (
    SLOT_ENVS, LEGEND_SLOT, ARROW_ENV, STYLES, BASE_FONT,
    _legend_handles_for_keys,
    setup_axes, draw_point, draw_better_arrow, save_figure,
)
import proto_pareto_data as data
import fseval_data as fs

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'aggregated_cache.json')

COH32_FSDIR = 'output/graft_canon_port_coh32_fseval'
NOINT_FSDIR = 'output/graft_canon_port-0627-0358_fseval'
RP_FSDIR = 'output/rp_noextras_7envs_port_fseval'

# both-adapters = blue per the unified color scheme (Jake 2026-07-09)
STYLES['gr_pre'] = ('Gradient Routing (pre-ablation)', '#1f77b4', 'o', False)
LEGEND_ORDER_V4 = ['gr', 'gr_pre', 'noi', 'noi_ro', 'rp_best', 'filt', 'base']

with open(CACHE_PATH) as _f:
    _CACHE = json.load(_f)


def _cache_agg(env, key):
    v = _CACHE[env].get(key)
    return tuple(v) if v else None


def _best_rp(env):
    """Best no-extras RP variant for `env`: argmax over penalties of
    (retain - hack) on the seed-mean fseval endpoint (scale 1.0). Returns
    (label, agg) or None while the sweep hasn't run."""
    recs = fs.load_recs(RP_FSDIR, f'{env}*_rp_noextras_pen*_s*.json')
    if not recs:
        return None
    by_pen = defaultdict(list)
    for rec in recs:
        pen = rec['run_name'].split('_pen')[1].split('_')[0]
        sm = rec['scales']['1.0']
        by_pen[pen].append((fs.pick(sm, 'retain'), fs.pick(sm, 'hack_freq')))
    best_pen = max(by_pen, key=lambda p: (
        sum(r for r, _ in by_pen[p]) / len(by_pen[p])
        - sum(h for _, h in by_pen[p]) / len(by_pen[p])))
    return f'p={best_pen}', fs.agg(by_pen[best_pen])


# GR arm = NO-COHERENCE with per-seed classifier-picked forget scale (Jake
# 2026-07-09), replacing coh32/fixed-ablation. Matches the aggregate + countdown
# deployment methodology. gr = pick, gr_pre = both (fs1.0).
NOCOH_FSDIR = 'output/graft_canon_port_fseval'
PICK_MON_THRESH = 0.02


def _nocoh_picked_agg(env):
    recs = fs.load_recs(NOCOH_FSDIR, f'{env}*_graft_lam1_s[0-9].json')
    assert len(recs) == 3, f"{env}: expected 3 nocoh fseval files, found {len(recs)}"
    order = [f'{i/10:.1f}' for i in range(11)]
    pts = []
    for rec in recs:
        sm = rec['scales']
        ok = [o for o in order if o in sm
              and fs.pick(sm[o], 'hack_freq_detectable') is not None
              and fs.pick(sm[o], 'hack_freq_detectable') < PICK_MON_THRESH]
        p = max(ok, key=float) if ok else '0.0'
        pts.append((fs.pick(sm[p], 'retain'), fs.pick(sm[p], 'hack_freq')))
    return fs.agg(pts)


def series_for_env(env):
    """[(style_key, agg)] for one env panel, background->foreground order."""
    nocoh_pat = f'{env}*_graft_lam1_s[0-9].json'
    noint_pat = f'{env}*_noint_lam1_s*.json'
    out = [
        ('noi',    fs.agg(fs.seed_points(NOINT_FSDIR, noint_pat, '1.0', n_expected=3))),
        ('noi_ro', fs.agg(fs.seed_points(NOINT_FSDIR, noint_pat, '0.0', n_expected=3))),
        ('filt',   _cache_agg(env, 'filt')),
    ]
    rp = _best_rp(env)
    if rp is not None:
        out.append(('rp_best', rp[1]))
    out += [
        ('gr_pre', fs.agg(fs.seed_points(NOCOH_FSDIR, nocoh_pat, '1.0', n_expected=3))),
        ('gr',     _nocoh_picked_agg(env)),
        ('base',   _cache_agg(env, 'base')),
    ]
    return out


def main():
    have_rp = bool(fs.load_recs(RP_FSDIR, '*.json'))
    if not have_rp:
        print('WARNING: no-extras RP fseval not found -> panel renders WITHOUT '
              f'an RP series (PENDING sweeps/rp_noextras_7envs_port.py -> {RP_FSDIR})')

    print(f'{"env":<15} {"series":<8} {"retain":>7} {"±std":>6} {"hack":>6} {"±std":>6}')
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()

    for slot, env in SLOT_ENVS:
        ax = axes[slot]
        for z, (key, agg) in enumerate(series_for_env(env)):
            if agg is None:
                continue
            print(f'{env:<15} {key:<8} {agg[0]:>7.3f} {agg[1]:>6.3f} {agg[2]:>6.3f} {agg[3]:>6.3f}')
            draw_point(ax, agg, key, zorder=8 + z)
        setup_axes(ax, env, slot)
        if env == ARROW_ENV:
            draw_better_arrow(ax)

    lax = axes[LEGEND_SLOT]
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    keys = [k for k in LEGEND_ORDER_V4 if k != 'rp_best' or have_rp]
    lax.legend(handles=_legend_handles_for_keys(keys), loc='center',
               frameon=False, fontsize=BASE_FONT * 0.9, handlelength=1.4,
               labelspacing=0.55, borderpad=0.4, bbox_to_anchor=(0.62, 0.5))

    fig.text(0.5, -0.012,
             'GR / no-intervention' + (' / RP' if have_rp else '') +
             ': posthoc eval (n=256) at the final checkpoint;  '
             'Filtering & base model: old-era in-training tail (cache)',
             ha='center', fontsize=BASE_FONT * 0.62, color='#666666')

    save_figure(fig, 'proto_pareto_7envs_gr_rp_v4.pdf')
    fig.savefig(os.path.join(HERE, 'figs', 'proto_pareto_7envs_gr_rp_v4.png'),
                dpi=150, bbox_inches='tight', pad_inches=0.03)
    # final_figures/ = the camera-ready set (per Jake 2026-07-07).
    final_dir = os.path.join(os.path.dirname(HERE), 'final_figures')
    os.makedirs(final_dir, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
        out = os.path.join(final_dir, f'proto_pareto_7envs_gr_rp_v4.{ext}')
        fig.savefig(out, bbox_inches='tight', pad_inches=0.03, **kw)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
