"""Uplift / hack-rate training-curve panel — partial-forget variant.

Same env-averaged-curve methodology as proto_uplift_panel_v1.py but with
the GR series swapped for the new no-coh runs:

  - GRAFT: post-ablation (canonical, kept) — green circle line
  - GRAFT: partial forget, no coherence (NEW) — dark green diamond line
       trajectory at each seed's optimum forget_scale; from
       output/gr_forget_scale_eval/canonical_5seed_trajectory_optimum/
  - GRAFT: pre-ablation, no coherence (NEW, replaces canonical pre-ablation)
       per-step routing_eval.jsonl mode='both' on the new no-coh runs
  - Reward Penalty (kept)
  - No intervention (kept)

Run:
    .venv/bin/python figures_pareto/proto_uplift_panel_partial_forget.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from proto_pareto_data import (
    ENVS, anchor_paths, no_intervention_paths, load_eval_series,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
plt.rcParams["axes.unicode_minus"] = False

NEW_RUN_DIR = os.path.join(REPO_ROOT, "output/retrain_gr_modal_all_classic_nocoh_canonical_steps")
TRAJ_DIR = os.path.join(REPO_ROOT, "output/gr_forget_scale_eval/canonical_5seed_trajectory_optimum")
FINAL_EVAL_SRC = os.path.join(REPO_ROOT, "output/gr_forget_scale_eval/canonical_5seed_1k_samples/results.jsonl")

# RoutedAdam-classic (bw2, seeds {1,3,5}) variant of the same regime — see
# MODAL_RUNS.md "RoutedAdam-classic". Its class is appended only when the
# trajectory dir exists. name filter "_radam_bw2_" excludes the bw1 topic
# ablation runs that share the sweep dir.
RADAM_RUN_DIR = os.path.join(REPO_ROOT, "output/retrain_gr_modal_all_classic_nocoh_canonical_steps_radam")
RADAM_TRAJ_DIR = os.path.join(REPO_ROOT, "output/gr_forget_scale_eval/canonical_radam_trajectory_optimum")
RADAM_FINAL_EVAL_SRC = os.path.join(REPO_ROOT, "output/gr_forget_scale_eval/canonical_radam_1k_samples/results.jsonl")
RADAM_BW1_TRAJ_DIR = os.path.join(REPO_ROOT, "output/gr_forget_scale_eval/canonical_radam_bw1_trajectory_optimum")
RADAM_BW1_FINAL_EVAL_SRC = os.path.join(REPO_ROOT, "output/gr_forget_scale_eval/canonical_radam_bw1_1k_samples/results.jsonl")

# Per-env training-step counts (used to convert raw step → training progress
# fraction). repeat_extra and topic_contains stop at 1000; the rest at 2000.
ENV_MAX_STEPS = {
    "persona_qa":     2000,
    "sorting_copy":   2000,
    "cities_qa":      2000,
    "object_qa":      2000,
    "addition_v2":    2000,
    "repeat_extra":   1000,
    "topic_contains": 1000,
}

# Cross-env aggregation grid in training-progress space (0..1).
PROGRESS_GRID = np.linspace(0.01, 1.0, 100)


def _per_seed_optima(src=FINAL_EVAL_SRC):
    if not os.path.isfile(src):
        return {}
    by_es = defaultdict(list)
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("retain") is None or r.get("hack_overall") is None:
                continue
            by_es[(r["env"], r["seed"])].append(r)
    out = {}
    for k, rs in by_es.items():
        best = max(rs, key=lambda x: x["retain"] - 2 * x["hack_overall"])
        out[k] = float(best["forget_scale"])
    return out


_OPTIMA = _per_seed_optima()
_RADAM_OPTIMA = _per_seed_optima(RADAM_FINAL_EVAL_SRC)
_RADAM_BW1_OPTIMA = _per_seed_optima(RADAM_BW1_FINAL_EVAL_SRC)


def _nocoh_run_dirs(env):
    if not os.path.isdir(NEW_RUN_DIR):
        return []
    return [os.path.join(NEW_RUN_DIR, d)
            for d in sorted(os.listdir(NEW_RUN_DIR))
            if d.startswith(env + "_")]


def _radam_run_dirs(env, arm="_radam_bw2_"):
    """One arm at a time — bw1 and bw2 runs share the sweep dir."""
    if not os.path.isdir(RADAM_RUN_DIR):
        return []
    return [os.path.join(RADAM_RUN_DIR, d)
            for d in sorted(os.listdir(RADAM_RUN_DIR))
            if d.startswith(env + "_") and arm in d]


def _step10_baseline(env, seed, family, run_dir=NEW_RUN_DIR, name_filter=None):
    """The same run's routing_eval.jsonl first row (step ~10) at mode='both'.
    Used as a step-0 proxy for the trajectory class so its uplift indexes to
    the untrained model rather than to step 100 (the first checkpoint).
    With MLP adapters init'd near zero, step-10 'both' ≈ base model."""
    rn_glob_prefix = f"{env}_"
    if not os.path.isdir(run_dir):
        return None
    for d in sorted(os.listdir(run_dir)):
        if not d.startswith(rn_glob_prefix):
            continue
        if not d.endswith(f"_s{seed}"):
            continue
        if name_filter is not None and name_filter not in d:
            continue
        ev = os.path.join(run_dir, d, "routing_eval.jsonl")
        if not os.path.isfile(ev):
            return None
        with open(ev) as f:
            first = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("step") is None:
                    continue
                if first is None or r["step"] < first["step"]:
                    first = r
        if first is None:
            return None
        prefix = f"both/{family}/"
        keys = [k for k in first if k.startswith(prefix) and k.count("/") == 2]
        return float(first[keys[0]]) if keys else None
    return None


def _traj_series(env, family, subtract_base, traj_dir=TRAJ_DIR, optima=None,
                 run_dir=NEW_RUN_DIR, name_filter=None):
    """For nocoh_partial_forget (and its RoutedAdam variant): read trajectory
    jsonl for each seed of env, extract the metric whose key starts with
    'forget_{seed_optimum}/{family}/'.
    Yields a list of (step, val) tuples per seed. The first tuple's `val` is
    REPLACED with the step-10 routing_eval baseline when subtract_base=True,
    so the trajectory's uplift indexes to ~step 0 (not step 100, the first
    checkpoint). The replacement is only used in `_gather_traj` for the
    base subtraction; the rest of the curve uses raw checkpoint values."""
    if optima is None:
        optima = _OPTIMA
    if not os.path.isdir(traj_dir):
        return []
    out = []
    for f in sorted(os.listdir(traj_dir)):
        if not (f.startswith(env + "_") and f.endswith(".jsonl")):
            continue
        seed = int(f.rsplit("_s", 1)[-1].split(".")[0])
        fs = optima.get((env, seed))
        if fs is None:
            continue
        mode_key = f"forget_{fs:g}"
        prefix = f"{mode_key}/{family}/"
        rows = []
        with open(os.path.join(traj_dir, f)) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                step = r.get("step")
                if step is None:
                    continue
                keys = [k for k in r if k.startswith(prefix)]
                if not keys:
                    continue
                rows.append((int(step), float(r[keys[0]])))
        if not rows:
            continue
        rows.sort()
        # Attach the step-10 baseline so _gather_traj can subtract it.
        baseline = (_step10_baseline(env, seed, family, run_dir=run_dir,
                                     name_filter=name_filter)
                    if subtract_base else None)
        out.append((rows, baseline))
    return out


# class kind -> (label, color, marker, getter_fn) where getter returns
# step_env_seeds dict, accepting (family, subtract_base).
def _gather_from_paths(paths_fn, mode):
    def _g(family, subtract_base):
        step_env_seeds = {}
        for env in ENVS:
            for p in paths_fn(env):
                series = load_eval_series(p, f"{mode}/{family}/")
                if not series:
                    continue
                base = series[0][1] if subtract_base else 0.0
                for step, val in series:
                    step_env_seeds.setdefault(step, {}).setdefault(env, []).append(val - base)
        return step_env_seeds
    return _g


def _gather_traj(traj_dir=TRAJ_DIR, optima=None, run_dir=NEW_RUN_DIR, name_filter=None):
    def _g(family, subtract_base):
        step_env_seeds = {}
        for env in ENVS:
            for rows, step10_base in _traj_series(
                    env, family, subtract_base=True, traj_dir=traj_dir,
                    optima=optima, run_dir=run_dir, name_filter=name_filter):
                # Always look up the step-10 baseline so we can both subtract
                # it (when subtract_base=True) AND prepend it as the curve's
                # starting anchor (matches the other classes that begin at
                # step ~10, and gives the hack_freq panel a value at step 0
                # for the trajectory class). Falls back to the first checkpoint
                # value when the run dir (routing_eval.jsonl) isn't synced.
                if subtract_base:
                    base = step10_base if step10_base is not None else rows[0][1]
                else:
                    base = 0.0
                anchor = step10_base if step10_base is not None else rows[0][1]
                # Prepend the step-10 anchor as a synthetic starting point.
                aug = [(10, anchor)] + rows
                for step, val in aug:
                    step_env_seeds.setdefault(step, {}).setdefault(env, []).append(val - base)
        return step_env_seeds
    return _g


CLASSES = [
    ("GRAFT: post-ablation (canonical)",  "#2ca02c", "o",
     _gather_from_paths(lambda e: anchor_paths(e, "GR"), "retain_only")),
    ("GRAFT: partial forget, no coh",      "#1a7a35", "D", _gather_traj()),
    ("GRAFT: pre-ablation, no coh",        "#0d3b66", "D",
     _gather_from_paths(_nocoh_run_dirs, "both")),
    ("Reward Penalty",                      "#d62728", "s",
     _gather_from_paths(lambda e: anchor_paths(e, "RP"), "both")),
    ("No intervention",                     "#ff7f0e", "X",
     _gather_from_paths(no_intervention_paths, "both")),
]

# RoutedAdam-classic partial-forget trajectory (bw2 only; color matches the
# RADAM class in proto_pareto_monitored_partial_forget so the radam composite's
# shared legend covers both panels). NOT in the default CLASSES — the original
# figure renders unchanged; the *_radam variant scripts pass this via
# draw(..., extra_classes=(RADAM_CLASS,)).
RADAM_CLASS = ("GRAFT: partial forget, RoutedAdam B=2", "#00838f", "D",
               _gather_traj(traj_dir=RADAM_TRAJ_DIR, optima=_RADAM_OPTIMA,
                            run_dir=RADAM_RUN_DIR, name_filter="_radam_bw2_"))
RADAM_BW1_CLASS = ("GRAFT: partial forget, RoutedAdam B=1", "#c2185b", "D",
                   _gather_traj(traj_dir=RADAM_BW1_TRAJ_DIR, optima=_RADAM_BW1_OPTIMA,
                                run_dir=RADAM_RUN_DIR, name_filter="_radam_bw1_"))
# Pre-ablation (both adapters, per-step routing_eval) on the same runs.
# Colors match proto_pareto_monitored_partial_forget's RADAM_*PA classes.
RADAM_PA_CLASS = ("GRAFT: pre-ablation, RoutedAdam B=2", "#4b1d6e", "D",
                  _gather_from_paths(_radam_run_dirs, "both"))
RADAM_BW1_PA_CLASS = ("GRAFT: pre-ablation, RoutedAdam B=1", "#795548", "D",
                      _gather_from_paths(
                          lambda e: _radam_run_dirs(e, arm="_radam_bw1_"), "both"))


def class_curve(getter, family, subtract_base):
    """Aggregate to a common training-progress grid (0..1).

    Per env: compute seed-mean and seed-variance at each step. Interpolate both
    onto PROGRESS_GRID. Drop grid points where NOT ALL envs have a valid value
    (so the curve only extends where every env contributes).

    Cluster mean = cross-env average of seed-means.
    Var(cluster mean) propagates per-env standard error of the mean
    (v1 convention; envs are treated as fixed, seeds within env are random):
        Var(mean) = (1/n_envs)^2 * sum_env (seed_var_env / n_seeds_env)
        CI       = 1.96 * sqrt(Var(mean))
    """
    step_env_seeds = getter(family, subtract_base)
    if not step_env_seeds:
        return None, None, None, None

    # Per-env, per-step: (seed_mean, seed_var, n_seeds).
    env_step_stats = defaultdict(dict)
    for step, env_dict in step_env_seeds.items():
        for env, vals in env_dict.items():
            arr = np.array(vals, dtype=float)
            n = len(arr)
            env_step_stats[env][step] = (
                float(arr.mean()),
                float(arr.var(ddof=1)) if n > 1 else 0.0,
                int(n),
            )

    env_curves = {}      # env -> (grid, mean_on_grid)
    env_var_on_grid = {} # env -> (var_on_grid, n_seeds_on_grid)
    for env, step_stats in env_step_stats.items():
        max_steps = ENV_MAX_STEPS.get(env)
        if max_steps is None or len(step_stats) < 2:
            continue
        steps_sorted = np.array(sorted(step_stats.keys()), dtype=float)
        means_sorted = np.array([step_stats[s][0] for s in sorted(step_stats.keys())])
        vars_sorted = np.array([step_stats[s][1] for s in sorted(step_stats.keys())])
        ns_sorted = np.array([step_stats[s][2] for s in sorted(step_stats.keys())], dtype=float)
        progress = steps_sorted / max_steps
        interp_mean = np.interp(PROGRESS_GRID, progress, means_sorted,
                                left=np.nan, right=np.nan)
        interp_var = np.interp(PROGRESS_GRID, progress, vars_sorted,
                               left=np.nan, right=np.nan)
        interp_n = np.interp(PROGRESS_GRID, progress, ns_sorted,
                             left=np.nan, right=np.nan)
        env_curves[env] = (PROGRESS_GRID.copy(), interp_mean)
        env_var_on_grid[env] = (interp_var, interp_n)

    if not env_curves:
        return None, None, None, None

    means_mat = np.array([v for _, v in env_curves.values()])     # (n_envs, n_grid)
    vars_mat = np.array([v for v, _ in env_var_on_grid.values()])
    ns_mat = np.array([n for _, n in env_var_on_grid.values()])
    n_envs_total = means_mat.shape[0]

    # Only keep grid points where every env contributes.
    valid_at_grid = ~np.any(np.isnan(means_mat), axis=0)

    # Cluster mean = average of env-means (n_envs all present where valid).
    means = np.where(valid_at_grid, np.nanmean(means_mat, axis=0), np.nan)
    # Variance of the cluster mean estimator:
    #   sum_env (seed_var / n_seeds) / n_envs^2  (envs fixed, seeds random)
    sem_sq = vars_mat / np.where(ns_mat > 0, ns_mat, 1)
    var_of_mean = np.sum(sem_sq, axis=0) / (n_envs_total ** 2)
    cis = np.where(valid_at_grid, 1.96 * np.sqrt(var_of_mean), np.nan)

    return PROGRESS_GRID, means, cis, env_curves


def draw(ax, family, subtract_base, extra_classes=()):
    from matplotlib.ticker import PercentFormatter
    starts = []
    for label, color, _marker, getter in list(CLASSES) + list(extra_classes):
        steps, mean, ci, env_curves = class_curve(getter, family, subtract_base)
        if steps is None or len(steps) == 0:
            print(f"[skip] {label}: no data for family={family}")
            continue
        for env, (env_steps, env_vals) in env_curves.items():
            mask = ~np.isnan(env_vals)
            ax.plot(env_steps[mask], env_vals[mask],
                    color=color, lw=0.7, alpha=0.25, zorder=2)
        valid = ~np.isnan(mean)
        ax.fill_between(steps[valid], (mean - ci)[valid], (mean + ci)[valid],
                        color=color, alpha=0.18, zorder=3, linewidth=0)
        ax.plot(steps[valid], mean[valid], color=color, lw=2.4, zorder=4, label=label)
        if valid.any():
            starts.append(mean[valid][0])
    base_y = 0.0 if subtract_base else (float(np.mean(starts)) if starts else 0.0)
    ax.axhline(base_y, color="0.35", lw=1.8, ls=(0, (6, 4)), zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)


def legend_handles():
    return [Line2D([], [], color=c, lw=2.6, label=l) for l, c, _m, _g in CLASSES]


def main():
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)

    draw(ax_top, "retain", subtract_base=True)
    ax_top.set_ylabel("Task performance\nimprovement")

    draw(ax_bot, "hack_freq", subtract_base=False)
    ax_bot.set_ylabel("Reward hack rate")
    ax_bot.set_xlabel("Training progress")

    ax_top.legend(handles=legend_handles(), loc="lower right",
                  frameon=True, fontsize=9)

    out_pdf = os.path.join(HERE, "figs", "proto_uplift_panel_partial_forget.pdf")
    out_png = os.path.join(HERE, "figs", "proto_uplift_panel_partial_forget.png")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
