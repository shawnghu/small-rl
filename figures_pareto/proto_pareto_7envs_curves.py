"""Extended proto_pareto plot: 3 RP parameter curves spanning out from the
canonical RP point.

Adds to proto_pareto_7envs.py:
  - penalty curve  (p):  pen=2 anchor -> pen=5 -> pen=10           (solid)
  - multiplier curve (v): mult=1 anchor -> mult=2 -> mult=5         (dashed)
  - ratio curve     (r):  0:1 -> 1:16 anchor -> 1:4 -> 1:2 -> 1:1   (dotted)

Layout: 4 rows x 2 cols (taller, wider per subplot than 2x4 to make
room for the spokes). 7 env subplots + 1 legend slot.

Annotations: only the EXTREME endpoint of each spoke is labeled
(e.g., "p=10", "v=5", "0:1", "1:1"). The canonical anchor is the
large purple square shared by all three spokes (drawn once with full
error bars). Spoke points are smaller markers with thin error bars.

Env-def caveat: for sort and persona, the p/v sweeps used the older
env definition (sort n_max=11, no uniform_per_length; persona regular
reward). The ratio spoke and the canonical anchor for those envs
use the new canonical envs (sort-uniform, persona-3x). The slight
visual offset between the p/v spoke endpoints and the canonical
anchor for these two envs reflects the env-def difference, not
training noise.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

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

# --- Canonical anchor: GR (retain_only mode) and RP (both mode) cspr=32 pen=2 mult=1 ---
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
    # --- sorting_copy (canonical sort-uniform: n_max=15, uniform_per_length=True) ---
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


# --- Spoke point sweep dirs (paths use sweep_dir + per-env basename pattern) ---
# All these sweeps used cspr=32 (RP, "both" mode), so we share one helper.
def _path(sweep_dir, env, run_suffix, seed):
    """Build the run-dir path for an env in a uniformly-named sweep."""
    # Standard env yaml short names
    EYS = {
        'addition_v2':    'addition_v2_sycophancy_conditional',
        'cities_qa':      'cities_qa_sycophancy_conditional',
        'object_qa':      'object_qa_sycophancy_conditional',
        'persona_qa':     'persona_qa_flattery_conditional',           # OLD persona for p/v sweeps
        'repeat_extra':   'repeat_extra_conditional',
        'sorting_copy':   'sorting_copy_conditional',                  # OLD sort for p/v sweeps
        'topic_contains': 'topic_contains_conditional',
    }
    eys = EYS[env]
    return f'output/{sweep_dir}/{eys}_{run_suffix}_s{seed}'


def _path_canonical_env(sweep_dir, env, run_suffix, seed):
    """Build path using the CANONICAL env yaml (sort-uniform, persona-3x).
    Used for the ratio spoke and (where canonical-env data exists) RP cspr=128."""
    EYS = {
        'addition_v2':    'addition_v2_sycophancy_conditional',
        'cities_qa':      'cities_qa_sycophancy_conditional',
        'object_qa':      'object_qa_sycophancy_conditional',
        'persona_qa':     'persona_qa_flattery_conditional_3xreward',  # NEW persona
        'repeat_extra':   'repeat_extra_conditional',
        'sorting_copy':   'sorting_copy_conditional',                  # NEW sort handled by sweep_dir
        'topic_contains': 'topic_contains_conditional',
    }
    eys = EYS[env]
    return f'output/{sweep_dir}/{eys}_{run_suffix}_s{seed}'


# Spoke definitions: each spoke is a list of (point_label, paths_per_env)
# where paths_per_env is a callable: env -> list of paths (one per seed available).
def _seeds(*paths):
    """Return paths that exist on disk."""
    return [p for p in paths if os.path.isdir(p)]


# Penalty spoke: (label, run_suffix, sweep_dir)
P_SPOKE = [
    ('p=2',  'rp_cspr32_pen2_rcl100_hf50_extramult10',  'rp_baseline_32extras_7envs'),
    ('p=5',  'rp_cspr32_pen5_rcl100_hf50_extramult10',  'rp_baseline_pen5_7envs'),
    ('p=10', 'rp_cspr32_pen10_rcl100_hf50_extramult10', 'rp_baseline_pen10_7envs'),
]

# Multiplier spoke
V_SPOKE = [
    ('v=1', 'rp_cspr32_pen2_rcl100_hf50_extramult10', 'rp_baseline_32extras_7envs'),
    ('v=2', 'rp_cspr32_pen2_rcl100_hf50_extramult20', 'rp_baseline_mult2_7envs'),
    ('v=5', 'rp_cspr32_pen2_rcl100_hf50_extramult50', 'rp_baseline_mult5_7envs'),
]

# Ratio spoke: (label, run_suffix, sweep_dir, env_naming) — env_naming chooses
# OLD vs NEW env yaml based on the data we have.
# 0:1 = verified-only;  1:16 = canonical (cspr=32 anchor);
# 1:4 = cspr=128;       1:2 = rb=384+cspr=192;  1:1 = rb=256+cspr=256.
# verified-only sweep used the same env as canonical for each env.
R_SPOKE = [
    ('0:1',  'verified_only_500iter',                                'verified_only_baseline_7envs', 'canonical'),
    ('1:16', '__ANCHOR__',                                            None,                            'canonical'),  # uses ANCHOR dict
    ('1:4',  '__CSPR128__',                                          None,                            'canonical'),  # special-case below
    ('1:2',  'rp_rb384_cspr192_pen2_rcl100_hf50_extramult10',         'rp_extras_ratio_1to1_1to2',     'canonical'),
    ('1:1',  'rp_rb256_cspr256_pen2_rcl100_hf50_extramult10',         'rp_extras_ratio_1to1_1to2',     'canonical'),
]


# RP cspr=128 (the 1:4 ratio point) lives in different sweep dirs depending on env.
CSPR128_PATHS = {
    # 4 easy envs: rp_baseline_7envs (3 seeds, OLD env) — same env def as canonical for these envs
    'addition_v2': [f'output/rp_baseline_7envs/addition_v2_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'object_qa':   [f'output/rp_baseline_7envs/object_qa_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'repeat_extra':[f'output/rp_baseline_7envs/repeat_extra_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    'topic_contains':[f'output/rp_baseline_7envs/topic_contains_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_s{s}' for s in (1, 2, 3)],
    # cities1x: rp_128extras_4cells (cities1x suffix, 5 seeds)
    'cities_qa':   [f'output/rp_128extras_4cells/cities_qa_sycophancy_conditional_rp_cspr128_pen2_rcl100_hf50_extramult10_cities1x_s{s}' for s in (1, 2, 3, 4, 5)],
    # persona-3x: rp_128extras_4cells (persona3x suffix, 5 seeds)
    'persona_qa':  [f'output/rp_128extras_4cells/persona_qa_flattery_conditional_3xreward_rp_cspr128_pen2_rcl100_hf50_extramult10_persona3x_s{s}' for s in (1, 2, 3, 4, 5)],
    # sort-uniform: sort_canonical_uniform_3cells (5 seeds)
    'sorting_copy':[f'output/sort_canonical_uniform_3cells/sorting_copy_conditional_rp_cspr128_nmax15_uniform_s{s}' for s in (1, 2, 3, 4, 5)],
}


# Verified-only and ratio sweep: persona_qa env naming differs (3xreward in ratio sweep, regular in verified-only).
PERSONA_RATIO_NAME = 'persona_qa_flattery_conditional_3xreward'
PERSONA_VERIFIED_NAME = 'persona_qa_flattery_conditional_3xreward'  # verified-only used 3xreward
SORT_VERIFIED_NAME = 'sorting_copy_conditional'  # used canonical settings (n_max=15, uniform)


def _verified_path(env, seed):
    """Path to verified-only run for env+seed (3 seeds available)."""
    EYS = {
        'addition_v2':    'addition_v2_sycophancy_conditional',
        'cities_qa':      'cities_qa_sycophancy_conditional',
        'object_qa':      'object_qa_sycophancy_conditional',
        'persona_qa':     PERSONA_VERIFIED_NAME,
        'repeat_extra':   'repeat_extra_conditional',
        'sorting_copy':   SORT_VERIFIED_NAME,
        'topic_contains': 'topic_contains_conditional',
    }
    return f'output/verified_only_baseline_7envs/{EYS[env]}_verified_only_500iter_s{seed}'


def _ratio12_path(env, ratio_tag, seed):
    """ratio_tag in {'rb384_cspr192', 'rb256_cspr256'}."""
    EYS = {
        'addition_v2':    'addition_v2_sycophancy_conditional',
        'cities_qa':      'cities_qa_sycophancy_conditional',
        'object_qa':      'object_qa_sycophancy_conditional',
        'persona_qa':     PERSONA_RATIO_NAME,
        'repeat_extra':   'repeat_extra_conditional',
        'sorting_copy':   'sorting_copy_conditional',
        'topic_contains': 'topic_contains_conditional',
    }
    return f'output/rp_extras_ratio_1to1_1to2/{EYS[env]}_rp_{ratio_tag}_pen2_rcl100_hf50_extramult10_s{seed}'


def load_run(path, env, mode):
    """Return (retain_mean, hack_undet_mean) over last 10% of eval rows."""
    det = DET[env]
    retain_prefix = f'{mode}/retain/'
    hack_key = f'{mode}/hack_freq_undetectable/{det}'
    rows = []
    eval_path = os.path.join(path, 'routing_eval.jsonl')
    if not os.path.exists(eval_path):
        return None
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
    if not retain_keys:
        return None
    rk = retain_keys[0]
    if hack_key not in tail[0]:
        return None
    retain = float(np.mean([r[rk] for r in tail]))
    hack = float(np.mean([r[hack_key] for r in tail]))
    return retain, hack


def aggregate_paths(paths, env, mode):
    """Aggregate (retain, hack) across paths. Returns (r_m, r_s, h_m, h_s, n) or None."""
    rs, hs = [], []
    for p in paths:
        if not os.path.isdir(p): continue
        x = load_run(p, env, mode)
        if x is None: continue
        r, h = x
        rs.append(r); hs.append(h)
    if not rs:
        return None
    return (float(np.mean(rs)), float(np.std(rs, ddof=0)),
            float(np.mean(hs)), float(np.std(hs, ddof=0)), len(rs))


def aggregate_anchor(env, cfg):
    """Aggregate the canonical anchor (GR or RP) across seeds 1..9."""
    paths = []
    for s in range(1, 10):
        p = ANCHOR.get((env, cfg, s))
        if p is not None: paths.append(p)
    mode = 'retain_only' if cfg == 'GR' else 'both'
    return aggregate_paths(paths, env, mode)


def aggregate_p_or_v(env, sweep_dir, run_suffix):
    """Aggregate one penalty/multiplier point. Uses OLD env yaml (regular persona, original sort)."""
    paths = [_path(sweep_dir, env, run_suffix, s) for s in (1, 2, 3)]
    return aggregate_paths(paths, env, 'both')


def aggregate_ratio(env, label):
    """Aggregate one ratio point."""
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


def main():
    fig, axes = plt.subplots(4, 2, figsize=(9.5, 13))
    axes = axes.flatten()

    GR_COLOR = '#2ca02c'                 # green
    RP_COLOR = '#7d4ba0'                 # canonical purple
    P_COLOR  = '#7d4ba0'                 # purple (penalty)
    V_COLOR  = '#d65f00'                 # orange (multiplier)
    R_COLOR  = '#1f9e89'                 # teal (ratio)

    # Spoke marker mapping: same color per parameter axis, distinct shape per value.
    # No connecting lines; no error bars on spoke points; no extreme-value annotations.
    SPOKE_MARKERS = {
        # parameter axis 'p' (penalty)
        ('p', 'p=5'):  ('s', P_COLOR),  # square
        ('p', 'p=10'): ('D', P_COLOR),  # diamond
        # parameter axis 'v' (multiplier)
        ('v', 'v=2'):  ('^', V_COLOR),  # triangle-up
        ('v', 'v=5'):  ('*', V_COLOR),  # star
        # parameter axis 'r' (ratio)
        ('r', '0:1'):  ('h', R_COLOR),  # hexagon
        ('r', '1:4'):  ('p', R_COLOR),  # pentagon
        ('r', '1:2'):  ('v', R_COLOR),  # triangle-down
        ('r', '1:1'):  ('X', R_COLOR),  # filled X
    }
    SPOKE_SIZE = 6

    # Layout: 7 envs in slots 0-6, legend in slot 7
    plot_slots = [0, 1, 2, 3, 4, 5, 6]
    legend_slot = 7

    for env, i in zip(ENVS, plot_slots):
        ax = axes[i]

        # GR canonical (primary, with error bars)
        gr = aggregate_anchor(env, 'GR')
        if gr is not None:
            r_m, r_s, h_m, h_s, _ = gr
            ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                        fmt='D', color=GR_COLOR, markersize=11,
                        markeredgecolor='black', markeredgewidth=0.6,
                        ecolor=GR_COLOR, elinewidth=1.2, capsize=3,
                        zorder=10)

        # RP canonical anchor (primary, with error bars)
        rp = aggregate_anchor(env, 'RP')
        if rp is not None:
            r_m, r_s, h_m, h_s, _ = rp
            ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                        fmt='s', color=RP_COLOR, markersize=11,
                        markeredgecolor='black', markeredgewidth=0.6,
                        ecolor=RP_COLOR, elinewidth=1.2, capsize=3,
                        zorder=9)

        # Penalty spoke points (purple, distinct shapes per value)
        for label, suffix, sweep in P_SPOKE:
            if label == 'p=2': continue  # anchor drawn above
            agg = aggregate_p_or_v(env, sweep, suffix)
            if agg is None: continue
            _, _, h_m, _, _ = agg
            r_m = agg[0]
            marker, color = SPOKE_MARKERS[('p', label)]
            ax.scatter([h_m], [r_m], marker=marker, s=SPOKE_SIZE**2,
                       c=color, edgecolors='black', linewidths=0.4,
                       alpha=0.9, zorder=6)

        # Multiplier spoke points (orange, distinct shapes)
        for label, suffix, sweep in V_SPOKE:
            if label == 'v=1': continue
            agg = aggregate_p_or_v(env, sweep, suffix)
            if agg is None: continue
            _, _, h_m, _, _ = agg
            r_m = agg[0]
            marker, color = SPOKE_MARKERS[('v', label)]
            ax.scatter([h_m], [r_m], marker=marker, s=SPOKE_SIZE**2,
                       c=color, edgecolors='black', linewidths=0.4,
                       alpha=0.9, zorder=6)

        # Ratio spoke points (teal, distinct shapes)
        for label, _suffix, _sweep, _envname in R_SPOKE:
            if label == '1:16': continue  # anchor
            agg = aggregate_ratio(env, label)
            if agg is None: continue
            _, _, h_m, _, _ = agg
            r_m = agg[0]
            marker, color = SPOKE_MARKERS[('r', label)]
            ax.scatter([h_m], [r_m], marker=marker, s=SPOKE_SIZE**2,
                       c=color, edgecolors='black', linewidths=0.4,
                       alpha=0.9, zorder=6)

        ax.set_title(ENV_TITLE.get(env, env))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.invert_xaxis()  # better is to the right
        ax.grid(True, alpha=0.3)
        # y-tick labels on left column only
        if i % 2 != 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Retain Reward')
        # x-tick labels on bottom row only
        if i // 2 != 3:  # not bottom row
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Hack Frequency')

    # Legend slot
    lax = axes[legend_slot]
    for s in lax.spines.values(): s.set_visible(False)
    lax.set_xticks([]); lax.set_yticks([])
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    # Build legend handles. Spoke handles use scatter-style (no line);
    # primary handles also markers-only.
    from matplotlib.lines import Line2D
    def _h(marker, color, label, size=8, edge='black'):
        return Line2D([0], [0], marker=marker, color='w',
                      markerfacecolor=color, markeredgecolor=edge,
                      markeredgewidth=0.5, markersize=size, label=label)

    legend_handles = [
        # Primary points
        _h('D', GR_COLOR,  'Gradient Routing (retain-only)', size=10),
        _h('s', RP_COLOR,  'Reward Penalty canonical (p=2, v=1, 1:16)', size=10),
        # Penalty spoke (purple)
        _h('s', P_COLOR,   'p = 5',  size=7),
        _h('D', P_COLOR,   'p = 10', size=7),
        # Multiplier spoke (orange)
        _h('^', V_COLOR,   'v = 2',  size=7),
        _h('*', V_COLOR,   'v = 5',  size=8),
        # Ratio spoke (teal)
        _h('h', R_COLOR,   'extras:main = 0:1 (verified-only)', size=7),
        _h('p', R_COLOR,   'extras:main = 1:4',  size=7),
        _h('v', R_COLOR,   'extras:main = 1:2',  size=7),
        _h('X', R_COLOR,   'extras:main = 1:1',  size=7),
    ]
    lax.legend(handles=legend_handles, loc='center', frameon=False,
               fontsize=8.5, handlelength=1.6, labelspacing=0.9,
               ncol=1)

    # Bottom-right "this way is better" arrow inside the legend slot
    lax.annotate('', xy=(0.95, 0.06), xytext=(0.66, 0.03),
                 arrowprops=dict(arrowstyle='-|>', color='black',
                                 lw=1.4, mutation_scale=12))
    lax.text(0.81, 0.005, 'better →', ha='center', va='bottom',
             fontsize=7.5, style='italic')

    plt.tight_layout()
    out = 'proto_pareto_7envs_gr_rp_curves.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    paper_dest = os.path.expanduser('~/gr-paper/figures/' + out)
    os.makedirs(os.path.dirname(paper_dest), exist_ok=True)
    plt.savefig(paper_dest, dpi=140, bbox_inches='tight')
    print(f'wrote {out} and {paper_dest}')


if __name__ == '__main__':
    main()
