"""Composite Figure 1, v2 — regenerated on the new canonical data (per Jake, 2026-07-07).

Supersedes proto_figure1_v1.py. Identical layout (left monitored/unmonitored
scatter hosting the shared legend; right env-averaged task-uplift + hack-rate
training panels), but every class now comes from the new canonical sources:

  - GRAFT pre/post-ablation: the NO-COHERENCE GR arm on the graft-port stack
    (output/graft_canon_port, classifier-only — NO coherence training, NO
    verified-retain extras), 7 envs x 3 seeds, lambda=1. Deployment is NOT a
    fixed retain-only ablation: it is the per-seed CLASSIFIER-PICKED forget
    scale — the highest forget scale (0.0-1.0) whose monitored (detectable)
    hack rate is still < PICK_MON_THRESH, falling back to 0.0 — matching the
    countdown deployment methodology (Jake 2026-07-09).
      * post-ablation (ours): the picked forget scale.
      * pre-ablation:         the two-adapter config (forget scale 1.0).
    The LEFT scatter reads both endpoints from the final-checkpoint forget-scale
    eval JSONs (output/graft_canon_port_fseval, one 0.0-1.0 scale sweep per
    run). BOTH RIGHT-panel GR curves come from the SAME per-CHECKPOINT
    forget-scale evals ({run}__step{500,1000,1500}.json plus the final base
    file) — the nocoh runs never logged in-training routing_eval, so there is no
    per-step source. At each checkpoint, per seed choose the forget scale and
    read retain (uplift panel) / hack_freq (hack-rate panel), average over seeds
    per env then over the envs present at that step:
      * post-ablation ('ours'): the classifier-picked scale per seed/checkpoint.
      * pre-ablation:           the fixed two-adapter scale '1.0'.
    Both are coarse 4-point trajectories (steps 500/1000/1500 + each run's base
    step, 2000 for the long envs) — coarser than the per-step RP/no-int curves.

    Monitored/unmonitored on the scatter are the per-subset CONDITIONAL rates
    hack_freq_detectable / hack_freq_undetectable (NOT hack_freq minus
    hack_freq_detectable — those are an overall fraction and a conditional rate
    on different bases, and the difference goes negative). This mirrors the
    detectable/undetectable slugs the coh32 arm used.

  - No intervention: the wandb-recovered clean no-intervention runs
    (output/no_intervention_7envs_wandb) for the training curves (mode 'both';
    evals every ~10 steps), and the new-stack no-int fseval JSONs
    (output/graft_canon_port-0627-0358_fseval, scale '1.0') for the scatter's
    detectable/undetectable split. As in proto_pareto_monitored_v2, that split
    is a posthoc final-checkpoint eval rather than a last-10%-of-training mean
    — immaterial for no-intervention, which has no penalty/routing signal
    driving tail drift. UNCHANGED by the coh32 -> no-coherence swap.

  - Reward Penalty: the no-extras RP rerun (output/rp_noextras_7envs_port,
    pen=2, mode 'both'). That sweep is STILL RUNNING: until every env has at
    least one pen2 run trained to its final eval step, the class is skipped
    and 'RP PENDING' is printed. Re-running this script picks the class up
    automatically once the data lands (only runs that reached their final
    eval step are ever used). UNCHANGED by the coh32 -> no-coherence swap.

Data hygiene: routing_eval rows are defensively deduped by step keeping the
last occurrence (Modal restarts append both attempts' rows). The wandb-
recovered topic_contains no-int runs sit on jittery step grids (16, 27, 36,
...) misaligned across seeds and envs — the raw all-7-env step intersection is
just 5 steps — so for the no-intervention curves ONLY, steps are snapped to
the nearest multiple of 10 (values untouched; x moves by <= 5 steps). All
other classes keep their exact step values.

Missing no-coherence eval data: both GRAFT training curves are built from the
per-checkpoint __step forget-scale eval JSONs (produced on the compute pod, must
be synced locally). When they are absent this script still renders the (fully
local) scatter + RP/no-int curves and prints which envs are missing; the GRAFT
curves appear automatically once their data lands.

Step-0 anchoring: every right-panel class MEAN curve is prepended with a
step-0 base-model point so all curves visually originate at the untrained
model. Top (uplift) panel: (0, 0.0) — uplift is 0 at step 0 by construction
of the common baseline. Bottom (hack-rate) panel: (0, pre-RL hack level) —
the same value the dashed 'pre-RL level' reference line sits at. The GR
per-checkpoint curves (first real eval at step 500) get the same anchor. The
SEM band is extended to step 0 with zero width, GR checkpoint markers are NOT
drawn on the synthetic anchor point, and the faint per-env overlays stay
unanchored.

Run:
    cd figures_pareto && ../.venv/bin/python proto_figure1_v2.py
"""
import glob as globlib
import json
import os
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy import stats

# fseval_data first: proto_pareto_data chdirs to the repo root at import,
# which would break a CWD-relative (sys.path[0]='') lookup of fseval_data.
from fseval_data import load_recs, pick
from proto_pareto_data import ENVS, DET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# One font size for the whole figure; ASCII minus for the panel's negatives.
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False

# --- no-coherence GR arm (replaces the coh32 arm) ---
NOCOH_FSDIR = 'output/graft_canon_port_fseval'                    # forget-scale evals (ROOT-joined)
PICK_MON_THRESH = 0.02   # deploy = highest forget scale with monitored hack < this; else 0.0
NOCOH_CKPT_STEPS = (500, 1000, 1500)  # intermediate __step files; final = base file's own step

NOINT_WANDB_DIR = os.path.join(ROOT, 'output', 'no_intervention_7envs_wandb')
NOINT_FSEVAL_DIR = 'output/graft_canon_port-0627-0358_fseval'  # fseval_data joins ROOT
RP_DIR = os.path.join(ROOT, 'output', 'rp_noextras_7envs_port')

EVAL_EVERY = 50
ENV_TOTAL_STEPS = {env: 1000 if env in ('repeat_extra', 'topic_contains') else 2000
                   for env in ENVS}

_SCALE_ORDER = [f'{i / 10:.1f}' for i in range(11)]  # '0.0' .. '1.0'


# ---------------------------------------------------------------- row loading
@lru_cache(maxsize=None)
def load_rows(run_dir):
    """routing_eval rows for one run, deduped by step keeping the LAST
    occurrence (Modal restarts append both attempts' rows), sorted by step.
    Returns () if the file is missing or empty."""
    p = os.path.join(run_dir, 'routing_eval.jsonl')
    if not os.path.exists(p):
        return ()
    with open(p) as f:
        lines = f.readlines()
    by_step = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # The RP sweep appends to these files live; a truncated FINAL line
            # is just a write in progress — skip it. Anywhere else it is real
            # corruption and must stay loud.
            if i == len(lines) - 1:
                continue
            raise
        assert row.get('step') is not None, f'row without step in {p}'
        by_step[int(row['step'])] = row  # later lines overwrite: keep last
    return tuple(by_step[s] for s in sorted(by_step))


def _metric_key(row, mode, family):
    """The one key whose first two '/'-segments are exactly (mode, family).
    Exact-segment matching is CRITICAL for family='hack_freq': startswith
    would collide with hack_freq_detectable/_undetectable/_hackable."""
    ks = [k for k in row
          if len(parts := k.split('/', 2)) >= 2
          and parts[0] == mode and parts[1] == family]
    assert len(ks) == 1, f'{mode}/{family}: {len(ks)} matches: {ks}'
    return ks[0]


def load_series(run_dir, mode, family, snap10=False):
    """Sorted (step, value) series for the exact (mode, family) metric.
    snap10 rounds steps to the nearest 10 (half-up) — needed only for the
    wandb-recovered no-int runs whose grids are jittery (see module docstring);
    collisions after snapping keep the later raw step."""
    rows = load_rows(run_dir)
    if not rows:
        return []
    key = _metric_key(rows[0], mode, family)
    out = {}
    for row in rows:  # rows sorted by raw step -> keep-last on snap collision
        if key not in row:
            continue
        s = int(row['step'])
        if snap10:
            s = (s + 5) // 10 * 10
        out[s] = float(row[key])
    return sorted(out.items())


# ------------------------------------------------- no-coherence GR loaders
def _pick_scale(scale_map):
    """Classifier-picked forget scale: highest '0.0'..'1.0' whose monitored
    (detectable) hack rate is < PICK_MON_THRESH, else '0.0'. scale_map is one
    fseval record's 'scales' dict."""
    ok = [o for o in _SCALE_ORDER if o in scale_map
          and pick(scale_map[o], 'hack_freq_detectable') < PICK_MON_THRESH]
    return max(ok, key=float) if ok else '0.0'


def _nocoh_base_paths(env):
    """The 3 final-checkpoint forget-scale-eval JSONs for env's GR seeds
    (the per-checkpoint {run}__step{N}.json files are excluded here)."""
    paths = sorted(globlib.glob(
        os.path.join(ROOT, NOCOH_FSDIR, f'{env}*_graft_lam1_s*.json')))
    paths = [p for p in paths if '__step' not in os.path.basename(p)]
    assert len(paths) == 3, f'{env}: expected 3 nocoh base fseval files, found {len(paths)}'
    return paths


def nocoh_scatter_points(env, which):
    """Per-seed (monitored, unmonitored) hack rates from the final-checkpoint
    forget-scale eval. which='post' -> classifier-picked forget scale;
    which='pre' -> two-adapter config (scale '1.0'). monitored =
    hack_freq_detectable, unmonitored = hack_freq_undetectable (per-subset
    conditional rates)."""
    pts = []
    for p in _nocoh_base_paths(env):
        sm = json.load(open(p))['scales']
        s = _pick_scale(sm) if which == 'post' else '1.0'
        pts.append((pick(sm[s], 'hack_freq_detectable'),
                    pick(sm[s], 'hack_freq_undetectable')))
    return pts


def _nocoh_ckpt_scalemaps(base_path):
    """(step, scale_map) for every available checkpoint eval of one run: the
    base file (its own rec['step'] = final checkpoint) plus each existing
    {stem}__step{N}.json. Deduped by step (a short run's base can coincide with
    an __step file), keeping the base record."""
    rec = json.load(open(base_path))
    by_step = {int(rec['step']): rec['scales']}
    stem = base_path[:-len('.json')]
    for n in NOCOH_CKPT_STEPS:
        sp = f'{stem}__step{n}.json'
        if os.path.exists(sp):
            r = json.load(open(sp))
            by_step.setdefault(int(r.get('step', n)), r['scales'])
    return sorted(by_step.items())


def nocoh_ckpt_curve(family, bases, which):
    """Env-averaged per-CHECKPOINT trajectory for the no-coherence GR arm, read
    from the forget-scale checkpoint evals ({run}__step{N}.json + each run's
    base file). which='post' -> classifier-picked forget scale per seed/
    checkpoint (deployment / 'ours'); which='pre' -> the two-adapter config
    (fixed forget scale '1.0'). Same return contract as class_curve.

    Both GR right-panel curves come from THIS one source (the nocoh runs never
    logged in-training routing_eval): a coarse 4-point trajectory. At each
    checkpoint step, per seed choose the scale and read `family` ('retain' for
    the uplift panel, 'hack_freq' for the hack-rate panel); average over seeds
    per env, then over the envs present at that step. bases={env: b} subtracts
    the common untrained reference (uplift); bases=None gives absolute values.

    Envs sit on different checkpoint grids (short envs repeat/topic end at 1000,
    so only 500/1000; addition s3 stops at 1500; the rest reach 2000); each env
    must contribute >=2 checkpoints. Returns (steps, mean, ci, env_curves) with
    env_curves aligned to `steps` (np.nan where an env has no checkpoint at that
    step)."""
    step_env_seeds = {}
    for env in ENVS:
        base = bases[env] if bases is not None else 0.0
        env_steps = set()
        for bp in _nocoh_base_paths(env):
            for step, sm in _nocoh_ckpt_scalemaps(bp):
                s = '1.0' if which == 'pre' else _pick_scale(sm)
                val = pick(sm[s], family)
                step_env_seeds.setdefault(step, {}).setdefault(env, []).append(val - base)
                env_steps.add(step)
        assert len(env_steps) >= 2, \
            f'{env}: only {len(env_steps)} checkpoint eval(s) for the {which} curve'

    steps = sorted(step_env_seeds)
    means, cis = [], []
    env_curves = {}
    for s in steps:
        envs_here = step_env_seeds[s]
        n_envs = len(envs_here)
        env_means, var_terms = {}, []
        for env, seeds in envs_here.items():
            arr = np.array(seeds, dtype=float)
            n_e = len(arr)
            env_means[env] = float(arr.mean())
            v = (arr.var(ddof=1) / n_e) / n_envs ** 2 if n_e > 1 else 0.0
            var_terms.append(v)
        means.append(float(np.mean(list(env_means.values()))))
        cis.append(1.96 * float(np.sqrt(np.sum(var_terms))))
        for env, m in env_means.items():
            env_curves.setdefault(env, {})[s] = m
    env_curves = {e: np.array([d.get(s, np.nan) for s in steps])
                  for e, d in env_curves.items()}
    return np.array(steps), np.array(means), np.array(cis), env_curves


# --------------------------------------------- no-coherence data availability
def _nocoh_step_ready():
    """True iff every env has at least one per-checkpoint __step fseval file
    (needed to build the >=2-point picked-scale post-ablation trajectory)."""
    for env in ENVS:
        if not globlib.glob(os.path.join(ROOT, NOCOH_FSDIR,
                                         f'{env}*_graft_lam1_s*__step*.json')):
            return False
    return True


NOCOH_STEP_READY = _nocoh_step_ready()     # both GR curves' per-checkpoint source


def print_nocoh_status():
    if NOCOH_STEP_READY:
        print('GRAFT pre/post-ablation training curves: per-checkpoint '
              '__step fseval present for all envs.')
    else:
        miss = [env for env in ENVS
                if not globlib.glob(os.path.join(ROOT, NOCOH_FSDIR,
                                                 f'{env}*_graft_lam1_s*__step*.json'))]
        print('GRAFT pre/post-ablation training curves SKIPPED — no '
              'per-checkpoint __step fseval under output/graft_canon_port_fseval/.')
        print(f'  envs missing __step fseval: {miss}')


# ---------------------------------------------------------------- run lookup
def noint_wandb_paths(env):
    """wandb-recovered no-intervention run dirs (3 seeds; persona 5; cities
    recovered only s2, s3 — take what the glob finds)."""
    if env == 'persona_qa':
        pat = 'persona_qa_persona_noint_3x_rcl100_hf50_s*'
    else:
        pat = f'{env}*_no_intervention_*'
    ds = sorted(globlib.glob(os.path.join(NOINT_WANDB_DIR, pat)))
    assert ds, f'{env}: no wandb-recovered no-intervention runs'
    for d in ds:
        assert load_rows(d), f'empty routing_eval in {d}'
    return ds


def _rp_complete_paths():
    """env -> COMPLETE pen2 no-extras RP run dirs. A run counts only once its
    routing_eval reaches the final eval step (max_steps - eval_every), so a
    still-training run never contributes a mid-training 'tail'."""
    out = {}
    for env in ENVS:
        paths = []
        for d in sorted(globlib.glob(os.path.join(RP_DIR, f'{env}*_rp_noextras_pen2_s*'))):
            rows = load_rows(d)
            if rows and rows[-1]['step'] >= ENV_TOTAL_STEPS[env] - EVAL_EVERY:
                paths.append(d)
        out[env] = paths
    return out


RP_COMPLETE = _rp_complete_paths()
RP_READY = all(len(v) >= 1 for v in RP_COMPLETE.values())


def rp_paths(env):
    return RP_COMPLETE[env] if RP_READY else []


def print_rp_status():
    if RP_READY:
        counts = ', '.join(f'{e}:{len(RP_COMPLETE[e])}' for e in ENVS)
        print(f'RP no-extras pen2 included (complete seeds per env: {counts})')
    else:
        missing = [e for e in ENVS if not RP_COMPLETE[e]]
        counts = ', '.join(f'{e}:{len(RP_COMPLETE[e])}' for e in ENVS)
        print(f'RP PENDING — rp_noextras_7envs_port pen2 incomplete; class skipped.')
        print(f'  complete runs per env: {counts}')
        print(f'  envs with no complete run yet: {missing}')


def noint_fseval_points(env):
    """Per-seed (monitored, unmonitored) hack rates for no-intervention from
    the new-stack fseval JSONs at adapter scale 1.0 (posthoc final-checkpoint
    eval of the noint runs)."""
    recs = load_recs(NOINT_FSEVAL_DIR, f'{env}*_noint_lam1_s*.json')
    assert len(recs) == 3, f'{env}: expected 3 noint fseval files, found {len(recs)}'
    pts = []
    for rec in recs:
        sm = rec['scales']['1.0']
        pts.append((pick(sm, 'hack_freq_detectable'), pick(sm, 'hack_freq_undetectable')))
    return pts


# ------------------------------------------------------------------- classes
# label -> color + data sources.
#   scatter:  a 'points' source (per-seed (monitored, unmonitored) tuples).
#   curves:   'paths'+'mode' feed the standard class_curve (per-step
#             routing_eval; RP + no-int), OR a 'deploy_curve' callable builds
#             the curve directly. Both GR arms use deploy_curve: a coarse
#             per-checkpoint fseval trajectory (fixed scale '1.0' for
#             pre-ablation, classifier-picked scale for post-ablation) — the
#             nocoh runs never logged in-training routing_eval.
# 'snap10' aligns the jittery wandb-recovered step grids for the curves only.
CLASSES = {
    'GRAFT: pre-ablation':         dict(color='#1f77b4', marker='o', hollow=False,
                                        points=lambda env: nocoh_scatter_points(env, 'pre'),
                                        deploy_curve=lambda family, bases:
                                            nocoh_ckpt_curve(family, bases, 'pre')),
    'GRAFT: post-ablation (ours)': dict(color='#2ca02c', marker='o', hollow=False,
                                        points=lambda env: nocoh_scatter_points(env, 'post'),
                                        deploy_curve=lambda family, bases:
                                            nocoh_ckpt_curve(family, bases, 'post')),
    'Reward Penalty':              dict(color='#d62728', marker='s', hollow=False, paths=rp_paths, mode='both'),
    'No intervention':             dict(color='#e0905a', marker='X', hollow=False, paths=noint_wandb_paths, mode='both',
                                        points=noint_fseval_points, snap10=True),
}
# Scatter draw/legend order (proto_pareto_monitored_v2 convention).
SCATTER_ORDER = ['GRAFT: post-ablation (ours)', 'GRAFT: pre-ablation',
                 'Reward Penalty', 'No intervention']
# Curve draw order (proto_uplift_panel_v1 convention).
CURVE_ORDER = ['GRAFT: pre-ablation', 'GRAFT: post-ablation (ours)',
               'Reward Penalty', 'No intervention']


def class_present(cname):
    return RP_READY if cname == 'Reward Penalty' else True


def curve_present(cname):
    """Whether cname's TRAINING CURVE can be drawn (data available). Both GR
    arms now draw from the per-checkpoint __step fseval (same source)."""
    if cname in ('GRAFT: pre-ablation', 'GRAFT: post-ablation (ours)'):
        return NOCOH_STEP_READY
    return class_present(cname)


def legend_handles():
    """Marker-only Line2D handles for the present classes, in SCATTER_ORDER.
    One legend in the scatter serves the whole figure."""
    handles = []
    for c in SCATTER_ORDER:
        if not class_present(c):
            continue
        spec = CLASSES[c]
        hollow = spec.get('hollow', False)
        handles.append(Line2D(
            [], [], marker=spec.get('marker', 'o'), linestyle='none',
            color=spec['color'],
            markerfacecolor='white' if hollow else spec['color'],
            markeredgecolor=spec['color'] if hollow else 'white',
            markeredgewidth=2.0 if hollow else 1.6, markersize=17, label=c))
    return handles


# ------------------------------------------------------------------- scatter
def class_point(env, spec):
    """Mean (monitored, unmonitored) hack rate over seeds for one (env, class).
    Returns (x, y, n_seeds); n_seeds=0 means no data. Classes with a 'points'
    source read per-seed tuples directly; 'paths'+'mode' classes take the
    per-run mean over the LAST 10% of routing_eval rows of
    {mode}/hack_freq_detectable|_undetectable/{DET[env]} (hackable prompts by
    construction), then average over seeds."""
    if 'points' in spec:
        pts = spec['points'](env)
        if not pts:
            return None, None, 0
        return (float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])), len(pts))
    det = DET[env]
    xk = f"{spec['mode']}/hack_freq_detectable/{det}"
    yk = f"{spec['mode']}/hack_freq_undetectable/{det}"
    xs, ys = [], []
    for p in spec['paths'](env):
        rows = load_rows(p)
        if not rows:
            continue
        n = max(1, len(rows) // 10)
        tail = rows[-n:]
        xs.append(float(np.mean([r[xk] for r in tail])))
        ys.append(float(np.mean([r[yk] for r in tail])))
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), float(np.mean(ys)), len(xs)


def cluster(spec):
    """Per-env (monitored, unmonitored) hack rates for one class, plus the
    cluster mean and 95% CI half-width on each axis (t, df=n_envs-1).
    Returns (xs, ys, x_m, x_ci, y_m, y_ci, n) — xs/ys are the per-env arrays."""
    xs, ys = [], []
    for env in ENVS:
        x, y, n = class_point(env, spec)
        if n == 0:
            continue
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    n = len(xs)
    if n == 0:
        return xs, ys, None, None, None, None, 0
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    x_ci = tcrit * float(np.std(xs, ddof=1) / np.sqrt(n))
    y_ci = tcrit * float(np.std(ys, ddof=1) / np.sqrt(n))
    return xs, ys, float(xs.mean()), x_ci, float(ys.mean()), y_ci, n


def draw_scatter(ax):
    """Monitored-vs-unmonitored cluster scatter (proto_pareto_monitored_v2
    cosmetics). No legend, no save."""
    # y = x reference: above the line => hacks more when unmonitored.
    ax.plot([0, 1], [0, 1], ls='--', color='0.7', lw=1.0, zorder=1)

    print(f'{"class":28s}  monitored        unmonitored      n')
    print('-' * 68)
    for cname in SCATTER_ORDER:
        spec = CLASSES[cname]
        if not class_present(cname):
            print(f'{cname:28s}  PENDING (skipped)')
            continue
        xs, ys, x_m, x_ci, y_m, y_ci, n = cluster(spec)
        print(f'{cname:28s}  {x_m:.3f} +/- {x_ci:.3f}  '
              f'{y_m:.3f} +/- {y_ci:.3f}  {n}')
        # Per-env points behind the cluster mean.
        ax.scatter(xs, ys, s=72, color=spec['color'], alpha=0.4,
                   marker=spec.get('marker', 'o'),
                   edgecolors='none', zorder=3, clip_on=False)
        # Cluster mean + 95% CI. Pre-ablation draws on top — its mean/error
        # bar overlaps the No-intervention cluster, and we want it legible.
        eb_z = 7 if cname == 'GRAFT: pre-ablation' else 5
        # CI whiskers are truncated to the physical [0, 1] rate domain via
        # asymmetric errors: near the origin the 95% CI extends below 0, and
        # an unclipped whisker would read as a negative hack rate (clipping
        # stays ON — matplotlib default — so nothing draws past the axes).
        ax.errorbar(x_m, y_m,
                    xerr=[[min(x_ci, x_m)], [min(x_ci, 1 - x_m)]],
                    yerr=[[min(y_ci, y_m)], [min(y_ci, 1 - y_m)]],
                    fmt=spec.get('marker', 'o'), markersize=21, color=spec['color'],
                    markerfacecolor='white' if spec.get('hollow') else spec['color'],
                    markeredgecolor=spec['color'] if spec.get('hollow') else 'white',
                    markeredgewidth=2.0 if spec.get('hollow') else 1.6,
                    ecolor=spec['color'], elinewidth=1.2, capsize=4,
                    capthick=1.2,
                    zorder=50 if cname == 'GRAFT: post-ablation (ours)' else eb_z,
                    clip_on=cname != 'GRAFT: post-ablation (ours)', label=cname)
    print('-' * 68)

    # Margin past 0%/100% so points and domain-clipped whisker caps sitting
    # exactly at 0/1 stay visibly inside the frame.
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Reward hack rate on monitored examples')
    ax.set_ylabel('Reward hack rate on unmonitored examples')
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# -------------------------------------------------------------------- curves
def noint_first_eval_bases(family):
    """Per-env untrained-model reference: the No-intervention class's seed-mean
    value at its FIRST eval step (~step 11 — the earliest measurement of the
    ~untrained policy anywhere in the curve data).

    Used as the COMMON uplift baseline for every class. Classes first-eval at
    different steps (old no-int runs ~11, GR / RP 50), so subtracting each run's
    OWN first eval (the v1 convention, valid when all classes eval'd from step
    10) would fabricate a cross-class offset: GR's step-50 base already contains
    ~50 steps of training (~+0.2 retain), deflating its uplift by that much
    relative to no-int while fseval endpoints are equal."""
    spec = CLASSES['No intervention']
    bases = {}
    for env in ENVS:
        vals = []
        for p in spec['paths'](env):
            series = load_series(p, spec['mode'], family,
                                 snap10=spec.get('snap10', False))
            if series:
                vals.append(series[0][1])
        assert vals, f'no no-intervention curve data for {env} baseline'
        bases[env] = float(np.mean(vals))
    return bases


def class_curve(spec, family, bases=None):
    """Cross-env mean curve + 95% CI for one (class, metric family)
    (proto_uplift_panel_v1.class_curve machinery).

    bases={env: b} -> per run, value(t) - b for its env (uplift over the
    common untrained reference; see noint_first_eval_bases).
    bases=None     -> the raw value(t) (absolute).

    The curve is the mean over envs of the per-env seed-mean. The CI quantifies
    how well finite seeds pin down that 7-env mean: per env e,
    Var(x_e_bar) = s_e^2 / n_e; Var(mean) = (1/n_envs^2) * sum_e Var(x_e_bar);
    CI = 1.96 * sqrt(Var(mean)). Restricted to steps where all envs are
    present and step < 1000 (repeat/topic end at 950; no-int old runs eval
    every ~10 to 1000+). Classes sit on different step grids — each draws its
    own. Returns (steps, mean, ci, env_curves)."""
    step_env_seeds = {}
    for env in ENVS:
        for p in spec['paths'](env):
            series = load_series(p, spec['mode'], family,
                                 snap10=spec.get('snap10', False))
            if not series:
                continue
            base = bases[env] if bases is not None else 0.0
            for step, val in series:
                step_env_seeds.setdefault(step, {}).setdefault(env, []).append(val - base)
    if not step_env_seeds:
        return None, None, None, None
    n_envs = max(len(envs) for envs in step_env_seeds.values())
    # Guard against a class quietly averaging fewer envs (the exact failure
    # the topic step-grid misalignment would have caused without snap10).
    assert n_envs == len(ENVS), f'{family}: only {n_envs}/{len(ENVS)} envs align'
    steps = sorted(s for s, envs in step_env_seeds.items()
                   if len(envs) == n_envs and s < 1000)

    means, cis = [], []
    env_curves = {}
    for s in steps:
        env_means, var_terms = {}, []
        for env, seeds in step_env_seeds[s].items():
            arr = np.array(seeds, dtype=float)
            n_e = len(arr)
            env_means[env] = float(arr.mean())
            # contribution of this env to Var(mean): Var(x_e_bar) / n_envs^2
            v = (arr.var(ddof=1) / n_e) / n_envs ** 2 if n_e > 1 else 0.0
            var_terms.append(v)
        means.append(float(np.mean(list(env_means.values()))))
        cis.append(1.96 * float(np.sqrt(np.sum(var_terms))))
        for env, m in env_means.items():
            env_curves.setdefault(env, []).append(m)
    env_curves = {e: np.array(c) for e, c in env_curves.items()}
    return np.array(steps), np.array(means), np.array(cis), env_curves


def draw_curves(ax, family, subtract_base):
    """Draw the class curves for `family` into `ax`
    (proto_uplift_panel_v1.draw machinery)."""
    bases = noint_first_eval_bases(family) if subtract_base else None
    # Reference level = the step-0 anchor every class mean curve starts from:
    # 0 for uplift (untrained by construction of the common baseline); the
    # untrained hack level for the absolute panel.
    if subtract_base:
        base_y = 0.0
    else:
        base_y = float(np.mean(list(noint_first_eval_bases(family).values())))
    for cname in CURVE_ORDER:
        if not class_present(cname) or not curve_present(cname):
            continue
        spec = CLASSES[cname]
        if 'deploy_curve' in spec:
            steps, mean, ci, env_curves = spec['deploy_curve'](family, bases)
        else:
            steps, mean, ci, env_curves = class_curve(spec, family, bases)
        if steps is None:
            continue
        # individual per-env trajectories, faint (unanchored: drawn on the
        # real eval grid, before the step-0 anchor is prepended)
        for curve in env_curves.values():
            ax.plot(steps, curve, color=spec['color'], lw=0.7, alpha=0.25, zorder=2)
        # Anchor the class MEAN curve at the step-0 base-model point (zero-
        # width SEM there: the anchor is exact by construction, not estimated).
        anchored = steps[0] != 0
        if anchored:
            steps = np.concatenate([[0], steps])
            mean = np.concatenate([[base_y], mean])
            ci = np.concatenate([[0.0], ci])
        ax.fill_between(steps, mean - ci, mean + ci, color=spec['color'],
                        alpha=0.18, zorder=3, linewidth=0)
        # GR arms are coarse 4-point per-checkpoint trajectories -> mark the
        # checkpoints so the sparser granularity is legible next to the dense
        # per-step RP / no-int lines. markevery skips the synthetic step-0
        # anchor: it is not a checkpoint eval.
        # clip_on=False so the step-2000 checkpoint marker (on the xlim=2000
        # right frame) renders in full rather than half-clipped.
        mk = dict(marker='o', markersize=7, markeredgecolor='white',
                  markeredgewidth=1.2, clip_on=False,
                  markevery=slice(1 if anchored else 0, None)
                  ) if 'deploy_curve' in spec else {}
        ax.plot(steps, mean, color=spec['color'], lw=2.4, zorder=4, label=cname, **mk)
    if not subtract_base:
        ax.text(0.985, base_y + 0.02, 'pre-RL level',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=12, color='0.35')
    ax.axhline(base_y, color='0.35', lw=1.8, ls=(0, (6, 4)), zorder=1)
    # 0..2000 so the GR 500->2000 per-checkpoint trajectories show in full. The
    # RP / no-int per-step curves stop ~950-1000 (their real length, all-envs
    # step intersection < 1000) and simply occupy the left portion — not
    # stretched.
    ax.set_xlim(0, 2000)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- composite
def main():
    print_rp_status()
    print_nocoh_status()

    fig = plt.figure(figsize=(17.0, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[9.0, 8.0], wspace=0.0)

    # Shared vertical span: both halves' plot areas run top=0.97 .. bottom=0.10.
    # sub_l's axes box is made slightly wider than tall so the aspect='equal'
    # scatter is height-limited and fills that span exactly, aligning its top
    # and bottom with the right-hand panels.
    TOP, BOT = 0.97, 0.10

    # --- left: monitored / unmonitored hack-rate scatter (hosts the legend) ---
    ax_sc = sub_l.subplots(1, 1)
    draw_scatter(ax_sc)
    ax_sc.legend(handles=legend_handles(), loc='lower right', frameon=True)
    sub_l.subplots_adjust(left=0.10, right=0.99, top=TOP, bottom=BOT)

    # --- right: env-averaged training panel (task uplift top, hack rate bottom) ---
    ax_top, ax_bot = sub_r.subplots(2, 1, sharex=True)
    draw_curves(ax_top, 'retain', subtract_base=True)
    ax_top.set_ylabel('Task performance\nimprovement')
    draw_curves(ax_bot, 'hack_freq', subtract_base=False)
    ax_bot.set_ylabel('Reward hack rate')
    # Same quantity as the scatter axes -> same unit (percent). The top panel
    # stays fractional: it is a reward delta, not a rate.
    ax_bot.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_bot.set_xlabel('Training step')
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([ax_top, ax_bot])

    # figs/ = working copies; final_figures/ = the camera-ready set (per Jake
    # 2026-07-07) — kept current on every re-render.
    for d in (os.path.join(HERE, 'figs'), os.path.join(ROOT, 'final_figures')):
        os.makedirs(d, exist_ok=True)
        for ext, kw in (('pdf', {}), ('png', {'dpi': 150})):
            out = os.path.join(d, f'proto_figure1_v2.{ext}')
            fig.savefig(out, bbox_inches='tight', pad_inches=0.04, **kw)
            print(f'wrote {out}')


if __name__ == '__main__':
    main()
