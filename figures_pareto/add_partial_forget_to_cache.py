"""Merge the partial-forget canonical series into the existing
aggregated_cache.json without touching other entries. Idempotent — re-run
after the canonical_5seed eval results.jsonl grows."""
import argparse
import json
import os

from proto_pareto_data import (
    ENVS, select_subset, aggregate_partial_forget_canonical,
    aggregate_partial_forget_exclusive,
)

HERE = os.path.dirname(os.path.abspath(__file__))
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
    sub_arg, fname = _SUBSETS[args.subset]
    select_subset(sub_arg)

    path = os.path.join(HERE, fname)
    with open(path) as f:
        cache = json.load(f)

    added = 0
    skipped = 0
    for env in ENVS:
        for key, aggfn in (('gr_pf', aggregate_partial_forget_canonical),
                           ('gr_pf_excl', aggregate_partial_forget_exclusive)):
            dumped = _dump_agg(aggfn(env))
            if dumped is None:
                # Source results.jsonl absent locally — preserve any existing cached
                # value rather than wiping the series (canonical anchors may be off-box).
                skipped += 1
                print(f"  [skip] {env} {key}: no source data (kept existing cache value)")
                continue
            cache.setdefault(env, {})[key] = dumped
            added += 1
            r_m, r_s, h_m, h_s, n = dumped
            print(f"  [ok]   {env} {key}: retain={r_m:.3f}±{r_s:.3f}  hack={h_m:.3f}±{h_s:.3f}  n={n}")

    with open(path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"wrote {path} (added {added}, skipped {skipped})")


if __name__ == '__main__':
    main()
