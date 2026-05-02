"""Shared data loading + aggregation for proto_pareto_* figures.

Maps the experimental data layout (which sweep dir for which env/method/seed)
to a single API:
  - aggregate_anchor(env, cfg)  : canonical GR or RP cspr=32 pen=2 mult=1
  - aggregate_p(env, p_value)   : RP at varying penalty
  - aggregate_v(env, v_value)   : RP at varying advantage multiplier
  - aggregate_ratio(env, label) : RP at varying extras:main ratio

Each returns (retain_mean, retain_std, hack_und_mean, hack_und_std, n_seeds)
or None if no data is available.
"""
import json
import os
import numpy as np


# Resolve output/ paths relative to the repo root so the scripts work from
# any CWD. The repo root is the parent of this file's directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO_ROOT)


ENVS = ['addition_v2', 'cities_qa', 'object_qa', 'persona_qa',
        'repeat_extra', 'sorting_copy', 'topic_contains']

ENV_TITLE = {
    'addition_v2': 'addition',
    'sorting_copy': 'sort',
    'topic_contains': 'topic',
}

DET = {
    'addition_v2': 'sycophancy_any',
    'cities_qa': 'sycophancy_any',
    'object_qa': 'sycophancy_any',
    'persona_qa': 'flattery_any',
    'repeat_extra': 'repeat_detector',
    'sorting_copy': 'sorting_copy_threshold',
    'topic_contains': 'topic_contains_detector',
}


# -------- Anchor (canonical GR + RP cspr=32 pen=2 mult=1) --------
ANCHOR = {
    # --- addition_v2 ---
    ('addition_v2', 'GR', 1): 'output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s1',
    ('addition_v2', 'GR', 2): 'output/canonical_topups_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s2',
    ('addition_v2', 'GR', 3): 'output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s3',
    ('addition_v2', 'GR', 4): 'output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s4',
    ('addition_v2', 'GR', 5): 'output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s5',
    ('addition_v2', 'RP', 1): 'output/rp_baseline_32extras_7envs/addition_v2_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s1',
    ('addition_v2', 'RP', 2): 'output/rp_baseline_32extras_7envs/addition_v2_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s2',
    ('addition_v2', 'RP', 3): 'output/rp_baseline_32extras_7envs/addition_v2_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s3',
    ('addition_v2', 'RP', 4): 'output/canonical_topups_4envs/addition_v2_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s4',
    ('addition_v2', 'RP', 5): 'output/canonical_topups_4envs/addition_v2_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s5',
    # --- cities_qa ---
    ('cities_qa', 'GR', 1): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s1',
    ('cities_qa', 'GR', 2): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s2',
    ('cities_qa', 'GR', 3): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s3',
    ('cities_qa', 'GR', 4): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s4',
    ('cities_qa', 'GR', 5): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s5',
    ('cities_qa', 'RP', 1): 'output/rp_canonical_redo_fresh/cities_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s1',
    ('cities_qa', 'RP', 2): 'output/rp_canonical_redo_fresh/cities_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s2',
    ('cities_qa', 'RP', 3): 'output/rp_canonical_redo_fresh/cities_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s3',
    ('cities_qa', 'RP', 4): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s4',
    ('cities_qa', 'RP', 5): 'output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s5',
    # --- object_qa ---
    ('object_qa', 'GR', 1): 'output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s1',
    ('object_qa', 'GR', 2): 'output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s2',
    ('object_qa', 'GR', 3): 'output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s3',
    ('object_qa', 'GR', 4): 'output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s4',
    ('object_qa', 'GR', 5): 'output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s5',
    ('object_qa', 'RP', 1): 'output/rp_baseline_32extras_7envs/object_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s1',
    ('object_qa', 'RP', 2): 'output/rp_baseline_32extras_7envs/object_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s2',
    ('object_qa', 'RP', 3): 'output/canonical_topups_4envs/object_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s3',
    ('object_qa', 'RP', 4): 'output/canonical_topups_4envs/object_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s4',
    ('object_qa', 'RP', 5): 'output/canonical_topups_4envs/object_qa_sycophancy_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s5',
    # --- persona_qa (3xreward variant) ---
    ('persona_qa', 'GR', 1): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_gr_cls_cspr32_rcl100_hf50_s1',
    ('persona_qa', 'GR', 2): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_gr_cls_cspr32_rcl100_hf50_s2',
    ('persona_qa', 'GR', 3): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_gr_cls_cspr32_rcl100_hf50_s3',
    ('persona_qa', 'GR', 4): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_gr_cls_cspr32_rcl100_hf50_s4',
    ('persona_qa', 'GR', 5): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_gr_cls_cspr32_rcl100_hf50_s5',
    ('persona_qa', 'RP', 1): 'output/rp_canonical_redo_fresh/persona_qa_flattery_conditional_3xreward_rp_cspr32_pen2_rcl100_hf50_extramult10_s1',
    ('persona_qa', 'RP', 2): 'output/rp_canonical_extend_cities_persona/persona_qa_flattery_conditional_3xreward_rp_cspr32_pen2_rcl100_hf50_extramult10_s2',
    ('persona_qa', 'RP', 3): 'output/rp_canonical_extend_cities_persona/persona_qa_flattery_conditional_3xreward_rp_cspr32_pen2_rcl100_hf50_extramult10_s3',
    ('persona_qa', 'RP', 4): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_rp_cspr32_pen2_rcl100_hf50_extramult10_s4',
    ('persona_qa', 'RP', 5): 'output/cspr32_gr_and_reruns/persona_qa_flattery_conditional_3xreward_rp_cspr32_pen2_rcl100_hf50_extramult10_s5',
    # --- repeat_extra ---
    ('repeat_extra', 'GR', 1): 'output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s1',
    ('repeat_extra', 'GR', 2): 'output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s2',
    ('repeat_extra', 'GR', 3): 'output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s3',
    ('repeat_extra', 'GR', 4): 'output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s4',
    ('repeat_extra', 'GR', 5): 'output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s5',
    ('repeat_extra', 'RP', 1): 'output/rp_baseline_32extras_7envs/repeat_extra_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s1',
    ('repeat_extra', 'RP', 2): 'output/canonical_topups_4envs/repeat_extra_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s2',
    ('repeat_extra', 'RP', 3): 'output/rp_baseline_32extras_7envs/repeat_extra_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s3',
    ('repeat_extra', 'RP', 4): 'output/canonical_topups_4envs/repeat_extra_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s4',
    ('repeat_extra', 'RP', 5): 'output/canonical_topups_4envs/repeat_extra_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s5',
    # --- sorting_copy (canonical sort-uniform) ---
    ('sorting_copy', 'GR', 1): 'output/sort_canonical_uniform_3cells/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s1',
    ('sorting_copy', 'GR', 2): 'output/sort_canonical_uniform_3cells/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s2',
    ('sorting_copy', 'GR', 3): 'output/cspr32_gr_and_reruns/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s3',
    ('sorting_copy', 'GR', 4): 'output/cspr32_gr_and_reruns/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s4',
    ('sorting_copy', 'GR', 5): 'output/cspr32_gr_and_reruns/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s5',
    ('sorting_copy', 'RP', 1): 'output/sort_canonical_uniform_3cells/sorting_copy_conditional_rp_cspr32_nmax15_uniform_s1',
    ('sorting_copy', 'RP', 2): 'output/sort_canonical_uniform_3cells/sorting_copy_conditional_rp_cspr32_nmax15_uniform_s2',
    ('sorting_copy', 'RP', 3): 'output/cspr32_gr_and_reruns/sorting_copy_conditional_rp_cspr32_nmax15_uniform_s3',
    ('sorting_copy', 'RP', 4): 'output/sort_canonical_uniform_3cells/sorting_copy_conditional_rp_cspr32_nmax15_uniform_s4',
    ('sorting_copy', 'RP', 5): 'output/sort_canonical_uniform_3cells/sorting_copy_conditional_rp_cspr32_nmax15_uniform_s5',
    # --- topic_contains ---
    ('topic_contains', 'GR', 1): 'output/topic_step0_baseline/topic_contains_conditional_cls_cspr32_rcl100_hf50_s1',
    ('topic_contains', 'GR', 2): 'output/topic_step0_baseline/topic_contains_conditional_cls_cspr32_rcl100_hf50_s2',
    ('topic_contains', 'GR', 3): 'output/gr_canonical_redo_4envs/topic_contains_conditional_gr_cls_cspr32_rcl100_hf50_s3',
    ('topic_contains', 'GR', 4): 'output/gr_canonical_redo_4envs/topic_contains_conditional_gr_cls_cspr32_rcl100_hf50_s4',
    ('topic_contains', 'GR', 5): 'output/gr_canonical_redo_4envs/topic_contains_conditional_gr_cls_cspr32_rcl100_hf50_s5',
    ('topic_contains', 'RP', 1): 'output/canonical_topups_4envs/topic_contains_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s1',
    ('topic_contains', 'RP', 2): 'output/rp_baseline_32extras_7envs/topic_contains_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s2',
    ('topic_contains', 'RP', 3): 'output/rp_baseline_32extras_7envs/topic_contains_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s3',
    ('topic_contains', 'RP', 4): 'output/canonical_topups_4envs/topic_contains_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s4',
    ('topic_contains', 'RP', 5): 'output/canonical_topups_4envs/topic_contains_conditional_rp_cspr32_pen2_rcl100_hf50_extramult10_s5',
}


# OLD env naming used by p/v sweeps (and by cspr=128 for 4 easy envs)
EYS_OLD = {
    'addition_v2':    'addition_v2_sycophancy_conditional',
    'cities_qa':      'cities_qa_sycophancy_conditional',
    'object_qa':      'object_qa_sycophancy_conditional',
    'persona_qa':     'persona_qa_flattery_conditional',
    'repeat_extra':   'repeat_extra_conditional',
    'sorting_copy':   'sorting_copy_conditional',
    'topic_contains': 'topic_contains_conditional',
}
# NEW env naming for canonical/ratio/verified-only paths
EYS_NEW = {
    **EYS_OLD,
    'persona_qa': 'persona_qa_flattery_conditional_3xreward',
}


# RP cspr=128 (the 1:4 ratio point) per env
CSPR128_PATHS = {
    'addition_v2':    [f'output/rp_baseline_7envs/addition_v2_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'object_qa':      [f'output/rp_baseline_7envs/object_qa_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'repeat_extra':   [f'output/rp_baseline_7envs/repeat_extra_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'topic_contains': [f'output/rp_baseline_7envs/topic_contains_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'cities_qa':      [f'output/rp_128extras_4cells/cities_qa_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_cities1x_s{s}' for s in (1, 2, 3, 4, 5)],
    'persona_qa':     [f'output/rp_128extras_4cells/persona_qa_flattery_conditional_3xreward_rp_cspr128_pen2_rcl100_hf50_extramult10_persona3x_s{s}' for s in (1, 2, 3, 4, 5)],
    'sorting_copy':   [f'output/sort_canonical_uniform_3cells/sorting_copy_conditional_rp_cspr128_nmax15_uniform_s{s}' for s in (1, 2, 3, 4, 5)],
}


def _verified_path(env, seed):
    return f'output/verified_only_baseline_7envs/{EYS_NEW[env]}_verified_only_500iter_s{seed}'


def _ratio12_path(env, ratio_tag, seed):
    """ratio_tag in {'rb384_cspr192', 'rb256_cspr256'}."""
    return f'output/rp_extras_ratio_1to1_1to2/{EYS_NEW[env]}_rp_{ratio_tag}_pen2_rcl100_hf50_extramult10_s{seed}'


# -------- Aggregation --------
def load_run(path, env, mode):
    """Mean over last 10% of routing_eval rows.
    Returns (retain_mean, hack_und_mean) or None."""
    det = DET[env]
    retain_prefix = f'{mode}/retain/'
    hack_key = f'{mode}/hack_freq_undetectable/{det}'
    eval_path = os.path.join(path, 'routing_eval.jsonl')
    if not os.path.exists(eval_path):
        return None
    rows = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    if not rows:
        return None
    n = max(1, len(rows) // 10)
    tail = rows[-n:]
    retain_keys = [k for k in tail[0] if k.startswith(retain_prefix)]
    if not retain_keys: return None
    rk = retain_keys[0]
    if hack_key not in tail[0]: return None
    return (float(np.mean([r[rk] for r in tail])),
            float(np.mean([r[hack_key] for r in tail])))


def aggregate_paths(paths, env, mode):
    rs, hs = [], []
    for p in paths:
        if not os.path.isdir(p): continue
        x = load_run(p, env, mode)
        if x is None: continue
        r, h = x
        rs.append(r); hs.append(h)
    if not rs: return None
    return (float(np.mean(rs)), float(np.std(rs, ddof=0)),
            float(np.mean(hs)), float(np.std(hs, ddof=0)), len(rs))


def aggregate_anchor(env, cfg):
    """cfg: 'GR' or 'RP' (the canonical anchor at cspr=32, pen=2, mult=1)."""
    paths = []
    for s in range(1, 10):
        p = ANCHOR.get((env, cfg, s))
        if p is not None: paths.append(p)
    mode = 'retain_only' if cfg == 'GR' else 'both'
    return aggregate_paths(paths, env, mode)


def aggregate_base_model(env):
    """Estimate the base (untrained) model's (retain, hack_und) by averaging
    the FIRST eval row across the canonical RP and GR runs.

    Caveat: the first eval is at step ~10 (after the first eval_every interval),
    not exactly t=0. With MLP adapters init'd to zero, step-10 behavior is very
    close to the underlying base model.
    """
    det = DET[env]
    h_key = f'both/hack_freq_undetectable/{det}'
    rs, hs = [], []
    for cfg in ('RP', 'GR'):
        for s in range(1, 10):
            p = ANCHOR.get((env, cfg, s))
            if p is None: continue
            eval_path = os.path.join(p, 'routing_eval.jsonl')
            if not os.path.exists(eval_path): continue
            with open(eval_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    row = json.loads(line)
                    break
                else:
                    continue
            r_keys = [k for k in row if k.startswith('both/retain/')]
            if not r_keys or h_key not in row: continue
            rs.append(float(row[r_keys[0]]))
            hs.append(float(row[h_key]))
    if not rs: return None
    return (float(np.mean(rs)), float(np.std(rs, ddof=0)),
            float(np.mean(hs)), float(np.std(hs, ddof=0)), len(rs))


def aggregate_verified_only(env):
    """Verified-only training (3 seeds at max_steps=500)."""
    paths = [_verified_path(env, s) for s in (1, 2, 3)]
    return aggregate_paths(paths, env, 'both')


def aggregate_no_intervention(env):
    """Standard GRPO with no penalty / filter / routing / extras (3 seeds).

    No-intervention runs don't enable the rh_detector during training, so the
    eval doesn't log hack_freq_detectable/undetectable. Since no intervention
    distinguishes monitored vs unmonitored prompts, the policy treats both
    identically and hack_freq (overall) is a faithful stand-in for
    hack_freq_undetectable.
    """
    eys = EYS_NEW[env]
    paths = [f'output/no_intervention_7envs/{eys}_no_intervention_rcl100_hf50_s{s}'
             for s in (1, 2, 3)]
    det = DET[env]
    return _aggregate_with_alt_hack_key(paths, env, 'both',
                                        alt_hack_key=f'both/hack_freq/{det}')


def _aggregate_with_alt_hack_key(paths, env, mode, alt_hack_key):
    """Like aggregate_paths but uses alt_hack_key when hack_freq_undetectable
    isn't logged (e.g., when the rh_detector is inactive during training)."""
    rs, hs = [], []
    for p in paths:
        if not os.path.isdir(p): continue
        x = load_run(p, env, mode)
        if x is None:
            # Fallback: try to load using the alternate hack key
            eval_path = os.path.join(p, 'routing_eval.jsonl')
            if not os.path.exists(eval_path): continue
            rows = []
            with open(eval_path) as f:
                for line in f:
                    line = line.strip()
                    if line: rows.append(json.loads(line))
            if not rows: continue
            n = max(1, len(rows) // 10)
            tail = rows[-n:]
            r_keys = [k for k in tail[0] if k.startswith(f'{mode}/retain/')]
            if not r_keys or alt_hack_key not in tail[0]: continue
            rs.append(float(np.mean([r[r_keys[0]] for r in tail])))
            hs.append(float(np.mean([r[alt_hack_key] for r in tail])))
        else:
            r, h = x
            rs.append(r); hs.append(h)
    if not rs: return None
    return (float(np.mean(rs)), float(np.std(rs, ddof=0)),
            float(np.mean(hs)), float(np.std(hs, ddof=0)), len(rs))


def aggregate_filter_baseline(env):
    """Filter-baseline (renormalized): drop detected hacks, recompute per-group GRPO baseline. 3 seeds."""
    eys = EYS_NEW[env]
    paths = [f'output/filter_baseline_7envs/{eys}_filter_baseline_renorm_rcl100_hf50_s{s}'
             for s in (1, 2, 3)]
    return aggregate_paths(paths, env, 'both')


# -------- hack_frac × recall matrix --------
def _matrix_path(method, env, hf_tag, rcl_tag, seed):
    """method: 'gr' or 'rp'. hf_tag: '050' or '090'. rcl_tag: '010'/'025'/'050'/'100'.
    Matrix sweeps used persona-3xreward (canonical) but the OLD sort env (n_max=11,
    no uniform_per_length). All other envs share their single canonical naming.
    """
    if env == 'persona_qa':
        eys = 'persona_qa_flattery_conditional_3xreward'
    else:
        eys = EYS_OLD[env]
    if method == 'gr':
        sweep = 'matrix_gr_7envs-0430-0819'
        suffix = f'gr_cls_cspr32_hf{hf_tag}_rcl{rcl_tag}'
    else:
        sweep = 'matrix_rp_7envs-0430-0819'
        suffix = f'rp_cspr32_pen2_hf{hf_tag}_rcl{rcl_tag}_extramult10'
    return f'output/{sweep}/{eys}_{suffix}_s{seed}'


def aggregate_hf_rcl(env, method, hf, rcl):
    """method: 'GR' or 'RP'. hf in {0.5, 0.9}. rcl in {0.1, 0.25, 0.5, 1.0}."""
    hf_tag = '050' if abs(hf - 0.5) < 1e-6 else '090'
    rcl_tag = {0.1: '010', 0.25: '025', 0.5: '050', 1.0: '100'}[rcl]
    method_lc = method.lower()
    paths = [_matrix_path(method_lc, env, hf_tag, rcl_tag, s) for s in (1, 2, 3)]
    mode = 'retain_only' if method == 'GR' else 'both'
    return aggregate_paths(paths, env, mode)


# Penalty / multiplier sweeps. For cities, persona, sort we use the
# rp_pen_mult_redo_3envs results (canonical envs, max_steps=2000); for
# the other 4 envs the original sweeps are still valid.
_REDO_ENVS = {'cities_qa', 'persona_qa', 'sorting_copy'}


def aggregate_p(env, p_value):
    """p_value in {2, 5, 10}. p=2 is the canonical anchor."""
    if p_value == 2:
        return aggregate_anchor(env, 'RP')
    if env in _REDO_ENVS:
        eys = EYS_NEW[env]
        suffix = f'rp_cspr32_pen{p_value}_extramult10_rcl100_hf50_redo'
        paths = [f'output/rp_pen_mult_redo_3envs/{eys}_{suffix}_s{s}' for s in (1, 2, 3)]
        return aggregate_paths(paths, env, 'both')
    sweep = 'rp_baseline_pen5_7envs' if p_value == 5 else 'rp_baseline_pen10_7envs'
    suffix = f'rp_cspr32_pen{p_value}_rcl100_hf50_extramult10'
    paths = [f'output/{sweep}/{EYS_OLD[env]}_{suffix}_s{s}' for s in (1, 2, 3)]
    return aggregate_paths(paths, env, 'both')


def aggregate_v(env, v_value):
    """v_value in {1, 2, 5}. v=1 is the canonical anchor."""
    if v_value == 1:
        return aggregate_anchor(env, 'RP')
    if env in _REDO_ENVS:
        eys = EYS_NEW[env]
        suffix = f'rp_cspr32_pen2_extramult{v_value}0_rcl100_hf50_redo'
        paths = [f'output/rp_pen_mult_redo_3envs/{eys}_{suffix}_s{s}' for s in (1, 2, 3)]
        return aggregate_paths(paths, env, 'both')
    sweep = f'rp_baseline_mult{v_value}_7envs'
    suffix = f'rp_cspr32_pen2_rcl100_hf50_extramult{v_value}0'
    paths = [f'output/{sweep}/{EYS_OLD[env]}_{suffix}_s{s}' for s in (1, 2, 3)]
    return aggregate_paths(paths, env, 'both')


def aggregate_ratio(env, label):
    """label in {'0:1', '1:16', '1:4', '1:2', '1:1'}. '1:16' is the canonical anchor."""
    if label == '1:16':
        return aggregate_anchor(env, 'RP')
    if label == '0:1':
        paths = [_verified_path(env, s) for s in (1, 2, 3)]
        return aggregate_paths(paths, env, 'both')
    if label == '1:4':
        return aggregate_paths(CSPR128_PATHS[env], env, 'both')
    if label == '1:2':
        paths = [_ratio12_path(env, 'rb384_cspr192', s) for s in (1, 2, 3)]
        return aggregate_paths(paths, env, 'both')
    if label == '1:1':
        paths = [_ratio12_path(env, 'rb256_cspr256', s) for s in (1, 2, 3)]
        return aggregate_paths(paths, env, 'both')
    raise ValueError(label)


# -------- Best-RP-per-env helper for the main figure --------
P_VALUES = [2, 5, 10]
V_VALUES = [1, 2, 5]
R_LABELS = ['0:1', '1:16', '1:4', '1:2', '1:1']


def all_rp_variants(env):
    """Yield (label, agg) for every distinct RP variant available for env.
    Canonical (p=2 / v=1 / 1:16) is yielded once under label 'canonical'."""
    seen_canonical = False
    for p in P_VALUES:
        agg = aggregate_p(env, p)
        if agg is None: continue
        label = 'canonical' if p == 2 else f'p={p}'
        if label == 'canonical':
            if seen_canonical: continue
            seen_canonical = True
        yield label, agg
    for v in V_VALUES:
        if v == 1: continue  # already covered as canonical
        agg = aggregate_v(env, v)
        if agg is None: continue
        yield f'v={v}', agg
    for r in R_LABELS:
        if r == '1:16': continue  # already covered as canonical
        if r == '0:1':  continue  # verified-only is its own baseline now
        agg = aggregate_ratio(env, r)
        if agg is None: continue
        yield r, agg


def best_rp(env):
    """Return (label, agg, score) for the RP variant with max(retain - hack_und)."""
    best = None
    for label, agg in all_rp_variants(env):
        r_m, _, h_m, _, _ = agg
        score = r_m - h_m
        if best is None or score > best[2]:
            best = (label, agg, score)
    return best
