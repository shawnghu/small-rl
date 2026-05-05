"""Dump all aggregator outputs needed by proto_pareto_7envs_v2.py to a JSON
cache so the figure can render locally without access to output/.

Run on the data host (where output/ exists):
    .venv/bin/python figures_pareto/dump_aggregated.py
"""
import json
import os

from proto_pareto_data import (
    ENVS,
    aggregate_anchor, aggregate_base_model, aggregate_no_intervention,
    aggregate_filter_baseline, aggregate_verified_only, best_rp,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'aggregated_cache.json')


def _dump_agg(agg):
    if agg is None:
        return None
    r_m, r_s, h_m, h_s, n = agg
    return [r_m, r_s, h_m, h_s, n]


def main():
    cache = {}
    for env in ENVS:
        br = best_rp(env)
        cache[env] = {
            'gr':    _dump_agg(aggregate_anchor(env, 'GR')),
            'noi':   _dump_agg(aggregate_no_intervention(env)),
            'filt':  _dump_agg(aggregate_filter_baseline(env)),     # weak filtering
            'verif': _dump_agg(aggregate_verified_only(env)),       # aggressive filtering
            'base':  _dump_agg(aggregate_base_model(env)),
            'best_rp': {
                'label': br[0] if br else None,
                'agg':   _dump_agg(br[1]) if br else None,
            },
        }
    with open(OUT, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
