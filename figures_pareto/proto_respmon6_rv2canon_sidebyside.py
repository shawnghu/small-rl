"""respmon COMPOSITE sidebyside — the paper's main-body toy-env figure
(gr-paper figures/proto_respmon6_composite_sidebyside.pdf).

Six response-dependent-monitorability envs at hf050: syco iva mon={indeed}
(addition, cities, object_qa), persona bvo mon={beautiful}, repeat
contiguity, sort format. The monitor is conditioned on the FORM of the hack,
not on a prompt subset, so detectable/undetectable partition the ground
truth.

GR arm = the rv2 CANONICAL recipe (respmon_rv2_gr_3seed): anchor/coherence
slice drawn from the full batch distribution with coherence_rh_mode=penalty,
coh128, balanced renorm + split moment, lam1. NOT the vercoh128 variant this
file was originally forked from, whose anchor slice was drawn from UNHACKABLE
prompts and therefore leaked hackability information into the anchor.

y = retain component on ALL prompts (target-task reward only). Restricting to
unhackable prompts would flatter any arm that effectively trains on the
unhackable slice.

Left: detectable-vs-undetectable hack-rate cluster scatter (proto_figure1_v2
cosmetics); one faint point per env, bold point = mean over envs with 95% CI.
Right: per-env grid, y = retain on all prompts, x = hack rate on HACKABLE
prompts.

Arms: GRAFT (gr, retain_only) / GRAFT with forget params enabled (gr_pre,
both) / GRAFT w/o routing (noi_ro) / GRAFT w/o anchoring (gr_nocoh) /
No intervention (noi) / Reward Penalty (rp_best) / Filtering (filt) / base
model (first eval row, step 10). No forget-scale curve was evaluated for
these envs, so 'gr' is full ablation rather than a classifier-picked scale.

Two env-specific deviations, both deliberate and documented at their
ENV_RUNS entries: object_qa uses the INVERTED-monitor variant for every arm,
and sort uses the 2026-07-26 framed redesign rather than the stock env.

RP config selection (Jake 2026-07-25): when multiple RP doses exist for an
env, pick the one with the best APPARENT performance — the view of a
developer who has only the misspecified proxy reward and the imperfect
monitor: tail mean of the combined proxy reward minus the monitor flag rate
(both observable; no GT channel labels). The figure still PLOTS ground-truth
metrics for the picked config. Sort has the full rp2/rp5/rp10 dose battery;
the other five envs' dose sweeps (respmon_rp_doses_canon_3seed) are still
training, so their pick is trivially rp2 until those land.

Data: routing_eval.jsonl mirrored from the data host to /workspace/respmon-results.
Outputs to figs/; copy the pdf to gr-paper as
figures/proto_respmon6_composite_sidebyside.pdf.

Run: cd figures_pareto && ../.venv/bin/python proto_respmon6_rv2canon_sidebyside.py
"""
import functools
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from proto_pareto_style_v2 import _legend_handles_for_keys, draw_point, STYLES
from proto_pareto_7envs_v4 import LEGEND_ORDER_V4  # registers STYLES['gr_pre']
STYLES['gr_pre'] = ('GRAFT (forget parameters enabled)', '#1f77b4', 'o', False)
import fseval_data as fs

HERE = os.path.dirname(os.path.abspath(__file__))
RESP_ROOT = '/workspace/respmon-results'

RETAIN_SLUG = 'retain'
HACK_SLUG = 'hack_freq_hackable'
N_SEEDS = 3

# env -> (run-name prefix, dn/rp sweep pair, gr sweep). GR = canonical rv2
# recipe (coh128 pen2 balanced split-moment lam1).
_GRCANON = ('respmon_rv2_gr_3seed', 'gr_cls')
ENV_RUNS = {
    'addition_v2': ('addition_v2_syco_respmon_iva',
                    ('respmon_iva_dn_3seed', 'respmon_iva_rp_3seed'), _GRCANON,
                    'respmon_filter_canon_3seed',
                    ('respmon_graft_nocoh_canon_3seed', 'graftnocoh')),
    'cities_qa':   ('cities_qa_syco_respmon_iva',
                    ('respmon_iva_dn_3seed', 'respmon_iva_rp_3seed'), _GRCANON,
                    'respmon_filter_canon_3seed',
                    ('respmon_graft_nocoh_canon_3seed', 'graftnocoh')),
    # object_qa uses the INVERTED-monitor variant for EVERY arm: in the
    # standard variant the monitor watches "indeed" (the form RL converges
    # to), so RP simply extinguishes hacking (0.2%) and the env cannot show
    # the monitor-exploiting failure. Inverted: RP 20.9% hack, 16.6% of it in
    # the blind channel. GRAFT here is the canonical recipe on the inverted
    # env (respmon_objinv_gr_canon_3seed).
    'object_qa':   ('object_qa_syco_respmon_iva_inv',
                    ('respmon_ivainv_dn_rp_3seed',) * 2,
                    ('respmon_objinv_gr_canon_3seed', 'gr_cls_canon'),
                    'respmon_filter_canon_3seed',
                    ('respmon_graft_nocoh_canon_3seed', 'graftnocoh')),
    'persona_qa':  ('persona_qa_flattery_respmon_bvo',
                    ('respmon_persona_bvo_dn_rp_3seed',) * 2, _GRCANON,
                    'respmon_filter_canon_3seed',
                    ('respmon_graft_nocoh_canon_3seed', 'graftnocoh')),
    'repeat':      ('repeat_respmon',
                    ('respmon_repeat_sorting_dn_rp_3seed',) * 2, _GRCANON,
                    'respmon_filter_canon_3seed',
                    ('respmon_graft_nocoh_canon_3seed', 'graftnocoh')),
    # sort: the 2026-07-26 REDESIGNED env (sorting_framed) — hackability is
    # announced in the prompt instead of being the latent 'largest element
    # first' property. Under the stock env every method failed (a copier
    # scores 0.241 positional on unhackable prompts, above the base policy's
    # 0.078, so copying is a local optimum of the retain reward and sorting
    # never develops). All arms below are 3 seeds on the redesigned env.
    'sorting_copy': ('sortframed',
                     ('sort_framed',) * 2, ('sort_framed', 'graft_hf'),
                     'sort_framed', ('sort_framed', 'graft_nocoh')),
}
ENV_TITLE = {'addition_v2': 'addition', 'sorting_copy': 'sort'}
GRID_ENVS = sorted(ENV_RUNS)


# 'GRAFT w/o routing' (series key noi_ro) is the CLEAN single-variable
# ablation (Jake 2026-07-26): identical to the canonical GRAFT recipe except
# rh_detector_recall 1.0 -> 0.0, so no sample is ever routed to the forget
# adapter while the anchor/coherence slice stays intact; evaluated
# retain_only, exactly like GRAFT. Verified against the canonical config:
# rh_detector_recall is the ONLY substantive difference.
# It replaces the earlier stand-in — a no-intervention run evaluated with the
# forget adapter ablated — which removed routing AND anchoring at once and so
# could not attribute the gap to routing.
NOROUTE_SWEEP = 'respmon_noroute_canon_3seed'
# sort's runs in that sweep are named for the env config (sorting_copy_framed)
# rather than the short 'sortframed' tag the sort_framed sweep uses.
_NOROUTE_PREFIX_OVERRIDE = {'sortframed': 'sorting_copy_framed'}

# -------- raw-run loading --------
# object_qa now names the inverted config directly, so no prefix override is
# needed; kept as an empty hook for future env-name divergences.
_FILTER_PREFIX_OVERRIDE = {}


def _run_dirs(sweep, prefix, method_pat, allow_missing=False):
    # NB: for the framed sort arms method_pat is 'graft_hf' (not 'graft') so
    # the prefix match cannot also swallow 'graft_nocoh'.
    root = os.path.join(RESP_ROOT, sweep)
    if allow_missing and not os.path.isdir(root):
        return []
    dirs = sorted(d for d in os.listdir(root)
                  if d.startswith(f'{prefix}_{method_pat}'))
    if allow_missing and not dirs:
        return []
    assert 2 <= len(dirs) <= N_SEEDS, \
        f'{sweep}/{prefix}_{method_pat}*: expected <={N_SEEDS} runs, found {dirs}'
    return [os.path.join(root, d) for d in dirs]


def _rows(path):
    with open(os.path.join(path, 'routing_eval.jsonl')) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert rows, f'empty routing_eval in {path}'
    by_step = {r['step']: r for r in rows}   # keep-last per step
    return [by_step[s] for s in sorted(by_step)]


def _key(row, mode, slug):
    ks = [k for k in row if k.startswith(f'{mode}/{slug}/')]
    assert len(ks) == 1, f'{mode}/{slug}: matched {ks}'
    return ks[0]


def _tail_mean(rows, mode, slug, first_row=False):
    sel = rows[:1] if first_row else rows[-max(1, len(rows) // 10):]
    k = _key(sel[0], mode, slug)
    return float(np.mean([r[k] for r in sel]))


def _points(paths, mode, first_row=False):
    # Skip runs with no eval rows (e.g. a seed killed before its first eval);
    # fs.agg reports the surviving seed count, so a short arm is visible in the
    # printed table rather than silently averaged as if complete.
    out = []
    for p in paths:
        if not os.path.exists(os.path.join(p, 'routing_eval.jsonl')):
            print(f'  [skip] {os.path.basename(p)}: no routing_eval.jsonl')
            continue
        out.append((_tail_mean(_rows(p), mode, RETAIN_SLUG, first_row),
                    _tail_mean(_rows(p), mode, HACK_SLUG, first_row)))
    return out


def _apparent_score(paths):
    """Developer-visible config score: combined proxy reward minus monitor
    flag rate (tail means, mixed eval subset). Uses only observables — the
    misspecified reward and the imperfect monitor — never GT channel labels."""
    return float(np.mean([
        _tail_mean(_rows(p), 'both', 'combined')
        - _tail_mean(_rows(p), 'both', 'detected_freq') for p in paths]))


@functools.lru_cache(maxsize=None)
def _best_rp(env):
    """All RP configs for env, grouped by run-name tag (rp2, rp1, ...);
    apparent-best group wins. Trivial while only rp2 exists."""
    prefix, (_, rp_sweep), _gr, _filt, _nc = ENV_RUNS[env]
    root = os.path.join(RESP_ROOT, rp_sweep)
    groups = defaultdict(list)
    for d in sorted(os.listdir(root)):
        m = re.match(re.escape(prefix) + r'_(rp\d+)_', d)
        if m:
            groups[m.group(1)].append(os.path.join(root, d))
    assert groups, f'no RP runs for {prefix} in {rp_sweep}'
    for tag, ps in groups.items():
        assert len(ps) == N_SEEDS, f'{rp_sweep}/{prefix}_{tag}: {ps}'
    scores = {tag: _apparent_score(ps) for tag, ps in groups.items()}
    best = max(scores, key=scores.get)
    print(f'rp pick {env}: {best}  (apparent combined - flag rate: '
          + ', '.join(f'{t}={s:.3f}' for t, s in sorted(scores.items())) + ')')
    return tuple(groups[best])


@functools.lru_cache(maxsize=None)
def env_paths(env):
    prefix, (dn_sweep, _), (gr_sweep, gr_pat), filt_sweep, (nc_sweep, nc_pat) = ENV_RUNS[env]
    fprefix = _FILTER_PREFIX_OVERRIDE.get(prefix, prefix)
    return {
        'dn': _run_dirs(dn_sweep, prefix, 'donothing'),
        'rp': list(_best_rp(env)),
        'gr': _run_dirs(gr_sweep, prefix, gr_pat),
        'filt': _run_dirs(filt_sweep, fprefix, 'filter'),
        'noroute': _run_dirs(NOROUTE_SWEEP,
                             _NOROUTE_PREFIX_OVERRIDE.get(prefix, prefix),
                             'noroute'),
        # the no-anchoring arm may still be training for some envs
        'gr_nocoh': _run_dirs(nc_sweep, fprefix, nc_pat, allow_missing=True),
    }


def series_for_env(env):
    p = env_paths(env)
    return [
        # draw order == z-order (zorder=8+index); 'gr' is forced to 50 inside
        # draw_point. Base sits under the no-anchoring arm, which sits under
        # GRAFT (Jake 2026-07-26).
        ('noi',     fs.agg(_points(p['dn'], 'both'))),
        ('noi_ro',  fs.agg(_points(p['noroute'], 'retain_only'))),
        ('filt',    fs.agg(_points(p['filt'], 'both'))),
        ('rp_best', fs.agg(_points(p['rp'], 'both'))),
        ('gr_pre',  fs.agg(_points(p['gr'], 'both'))),
        ('gr',      fs.agg(_points(p['gr'], 'retain_only'))),
        ('base',    fs.agg(_points(p['dn'] + p['rp'] + p['gr'], 'both',
                                   first_row=True))),
        ('gr_nocoh', fs.agg(_points(p['gr_nocoh'], 'retain_only'))),
    ]


# -------- left panel: monitored-vs-unmonitored cluster scatter --------
# (proto_figure1_v2 cosmetics; classes recomputed from the respmon runs.)
# Labels/colors/markers mirror proto_pareto_style_v2.STYLES so the RIGHT
# grid's legend serves both panels (the left panel carries no legend of its
# own). 'hollow' marks outline-only markers, matching STYLES' 4th field.
SCATTER_CLASSES = [
    ('GRAFT (ours)',                               '#2ca02c', 'o', 'gr',       'retain_only', False),
    ('GRAFT (forget parameters enabled)',          '#1f77b4', 'o', 'gr',       'both',        False),
    ('No intervention',                            '#e0905a', 'X', 'dn',   'both',        False),
    ('GRAFT w/o routing',                          '#9690a8', 'X', 'noroute',  'retain_only', True),
    ('GRAFT w/o anchoring',                        '#bcbd22', 'v', 'gr_nocoh', 'retain_only', False),
    ('Reward Penalty',                             '#d62728', 's', 'rp',   'both',        False),
    ('Filtering',                                  '#b09680', 'D', 'filt', 'both',        False),
]


def _mon_unmon(env, which, mode):
    """Seed-mean (monitored, unmonitored) hack rate over tail rows
    (hackable-slice quadrants). Unmonitored is DERIVED as total - detectable:
    identical to the logged unmonitored-only channel when the env's channel
    detectors partition the ground truth, and correct even where they don't
    (persona_bvo's undetectable phrase list is unset in the run configs and
    falls back to the wrong default vocabulary -- flagged 2026-07-25)."""
    xs, ys = [], []
    for p in env_paths(env)[which]:
        if not os.path.exists(os.path.join(p, 'routing_eval.jsonl')):
            continue          # seed killed before its first eval; see _points
        rows = _rows(p)
        det = _tail_mean(rows, mode, 'hack_freq_detectable')
        tot = _tail_mean(rows, mode, 'hack_freq_hackable')
        xs.append(det)
        ys.append(max(0.0, tot - det))
    return float(np.mean(xs)), float(np.mean(ys))


def scatter_legend_handles():
    return [Line2D([], [], marker=m, linestyle='none', color=c,
                   markerfacecolor='white' if hollow else c,
                   markeredgecolor=c if hollow else 'white',
                   markeredgewidth=2.0 if hollow else 1.6,
                   markersize=17, label=name)
            for name, c, m, _, _, hollow in SCATTER_CLASSES]


def draw_scatter(ax):
    ax.plot([0, 1], [0, 1], ls='--', color='0.7', lw=1.0, zorder=1)
    print(f'{"class":28s}  monitored        unmonitored      n')
    print('-' * 68)
    for name, color, marker, which, mode, hollow in SCATTER_CLASSES:
        pts = [_mon_unmon(env, which, mode) for env in GRID_ENVS
               if env_paths(env).get(which)]
        if not pts:
            print(f'  [scatter] {name}: no runs yet — skipped')
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        n = len(xs)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        x_m, y_m = float(xs.mean()), float(ys.mean())
        x_ci = tcrit * float(np.std(xs, ddof=1) / np.sqrt(n))
        y_ci = tcrit * float(np.std(ys, ddof=1) / np.sqrt(n))
        print(f'{name:28s}  {x_m:.3f} +/- {x_ci:.3f}  '
              f'{y_m:.3f} +/- {y_ci:.3f}  {n}')
        ax.scatter(xs, ys, s=72, marker=marker, alpha=0.4, zorder=3,
                   clip_on=False,
                   facecolors='none' if hollow else color,
                   edgecolors=color if hollow else 'none')
        post = name == 'GRAFT (ours)'   # NB: keep in sync with STYLES['gr']
        ax.errorbar(x_m, y_m,
                    xerr=[[min(x_ci, x_m)], [min(x_ci, 1 - x_m)]],
                    yerr=[[min(y_ci, y_m)], [min(y_ci, 1 - y_m)]],
                    fmt=marker, markersize=21, color=color,
                    markerfacecolor='white' if hollow else color,
                    markeredgecolor=color if hollow else 'white',
                    markeredgewidth=2.0 if hollow else 1.6,
                    ecolor=color, elinewidth=1.2,
                    capsize=4, capthick=1.2,
                    zorder=50 if post else
                    (7 if 'pre-ablation' in name else 5),
                    clip_on=not post, label=name)
    print('-' * 68)
    # 0/0 at TOP-RIGHT (both axes reversed), matching the pareto panels.
    ax.set_xlim(1.05, -0.03)
    ax.set_ylim(1.05, -0.03)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Detectable reward hack rate')
    ax.set_ylabel('Undetectable reward hack rate')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# -------- composite (same layout as proto_7envs_sidebyside_unhack) --------
N_COLS = 3
# 6 envs fill the bottom two rows; legend sits in the top row's LEFT two
# cells (gs[0, :2]); top-right cell stays empty.
ENV_CELLS = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
BOTTOM_EDGE = {(2, 0), (2, 1), (2, 2)}   # cells with no panel below them


def setup_grid_axes(ax, env, row, col):
    ax.set_title(ENV_TITLE.get(env, env), fontsize=19)
    ax.set_box_aspect(1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(labelsize=16)
    if col != 0:
        ax.tick_params(labelleft=False)
    if (row, col) not in BOTTOM_EDGE:
        ax.tick_params(labelbottom=False)
    elif col != 0:
        ax.set_xticklabels(['0%', '50%', ''])


def main():
    print(f'{"env":<15} {"series":<8} {"retain":>7} {"hack":>7} {"n":>2}')
    for env in GRID_ENVS:
        for key, agg in series_for_env(env):
            if agg is None:
                print(f'{env:<15} {key:<8}   (not run yet)')
                continue
            r_m, _, h_m, _, n = agg
            print(f'{env:<15} {key:<8} {r_m:>7.3f} {h_m:>7.3f} {n:>2}')

    fig = plt.figure(figsize=(17.0, 8.2))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[1.0, 0.92], wspace=0.0)
    TOP, BOT = 0.97, 0.10

    ax_l = sub_l.subplots(1, 1)
    draw_scatter(ax_l)
    ax_l.xaxis.label.set_size(25)
    ax_l.yaxis.label.set_size(25)
    ax_l.tick_params(labelsize=20)
    sub_l.subplots_adjust(left=0.11, right=0.98, top=TOP, bottom=BOT)

    gs = sub_r.add_gridspec(3, N_COLS, wspace=0.04, hspace=0.22,
                            left=0.125, right=0.985, top=TOP, bottom=BOT)
    grid_axes = []
    for (row, col), env in zip(ENV_CELLS, GRID_ENVS):
        ax = sub_r.add_subplot(gs[row, col])
        grid_axes.append(ax)
        for z, (key, agg) in enumerate(series_for_env(env)):
            if agg is None:
                continue     # arm not yet run for this env
            draw_point(ax, agg, key, zorder=8 + z)
        setup_grid_axes(ax, env, row, col)
    # panels occupy rows 1-2 (row 0 is the legend): center the ylabel on the
    # panel block so it cannot ride up into the legend.
    sub_r.supylabel('Task Performance (better →)',
                    fontsize=25, x=0.012, y=BOT + (TOP - BOT) / 3)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lab_bb = inv.transform(ax_l.xaxis.label.get_window_extent())
    y_lab = (lab_bb[0][1] + lab_bb[1][1]) / 2
    left_bb = inv.transform(grid_axes[3].get_window_extent())
    right_bb = inv.transform(grid_axes[5].get_window_extent())
    x_lab = (left_bb[0][0] + right_bb[1][0]) / 2
    fig.text(x_lab, y_lab, 'Reward hack rate  (better →)',
             ha='center', va='center', fontsize=25)

    lax = sub_r.add_subplot(gs[0, :2])
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    keys = ['gr', 'gr_pre', 'gr_nocoh', 'noi', 'noi_ro', 'rp_best', 'filt', 'base']
    handles = _legend_handles_for_keys(keys)
    for h in handles:
        if h.get_label() == 'GRAFT (forget parameters enabled)':
            h.set_label('GRAFT\n(forget parameters enabled)')
    lax.legend(handles=handles, loc='center', frameon=False, fontsize=16,
               handlelength=1.4, handletextpad=0.6, labelspacing=0.7,
               borderpad=0.1, ncol=2, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.5))

    outdir = os.path.join(HERE, 'figs')
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
        out = os.path.join(outdir, f'proto_respmon6_rv2canon_sidebyside.{ext}')
        fig.savefig(out, bbox_inches='tight', pad_inches=0.04, **kw)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
