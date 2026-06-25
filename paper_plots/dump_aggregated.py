"""Dump all aggregator outputs needed by the proto_pareto_7envs_v2 figures to
a JSON cache so the figure can render locally without access to output/.

Run on the data host (where output/ exists):
    .venv/bin/python figures_pareto/dump_aggregated.py                # overall
    .venv/bin/python figures_pareto/dump_aggregated.py --subset hackable

--subset overall  -> aggregated_cache.json          (all eval prompts)
--subset hackable -> aggregated_cache_hackable.json  (hackable prompts only)
"""
import argparse
import json
import os

from proto_pareto_data import (
    ENVS, select_subset,
    aggregate_anchor, aggregate_base_model, aggregate_no_intervention,
    aggregate_no_intervention_retain_only,
    aggregate_filter_baseline, aggregate_verified_only, best_rp,
    aggregate_forget_lr, aggregate_antitrain, aggregate_antitrain_forget_lr,
    aggregate_graddiag_wd01, aggregate_graddiag_fn2,
    aggregate_balanced_splitmoment,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# subset name -> (select_subset arg, output cache filename)
_SUBSETS = {
    'overall':  ('',          'aggregated_cache.json'),
    'hackable': ('_hackable', 'aggregated_cache_hackable.json'),
}


def _dump_agg(agg):
    if agg is None:
        return None
    r_m, r_s, h_m, h_s, n = agg
    return [r_m, r_s, h_m, h_s, n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', choices=sorted(_SUBSETS), default='overall')
    args = ap.parse_args()

    sub_arg, out_name = _SUBSETS[args.subset]
    select_subset(sub_arg)

    cache = {}
    for env in ENVS:
        br = best_rp(env)
        cache[env] = {
            'gr':     _dump_agg(aggregate_anchor(env, 'GR')),
            'noi':    _dump_agg(aggregate_no_intervention(env)),
            'noi_ro': _dump_agg(aggregate_no_intervention_retain_only(env)),
            'filt':   _dump_agg(aggregate_filter_baseline(env)),    # weak filtering
            'verif':  _dump_agg(aggregate_verified_only(env)),      # aggressive filtering
            'flr':    _dump_agg(aggregate_forget_lr(env)),          # lower forget LR
            'anti':   _dump_agg(aggregate_antitrain(env)),          # antitraining w=-1
            'both':   _dump_agg(aggregate_antitrain_forget_lr(env)),  # both interventions
            'wd':     _dump_agg(aggregate_graddiag_wd01(env)),      # weight decay 0.1
            'fn2':    _dump_agg(aggregate_graddiag_fn2(env)),       # smaller forget adapter (WIP)
            'bsm':    _dump_agg(aggregate_balanced_splitmoment(env)),  # balanced renorm + split-moment
            'base':   _dump_agg(aggregate_base_model(env)),
            'best_rp': {
                'label': br[0] if br else None,
                'agg':   _dump_agg(br[1]) if br else None,
            },
        }
    out = os.path.join(HERE, out_name)
    with open(out, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
