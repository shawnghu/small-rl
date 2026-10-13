"""Delete specific files (by exact name) from wandb runs matching a filter.

Use case: drop `output.log` from runs where it's mostly httpx noise, while keeping
config / summary / metrics intact.

Default: dry-run. Pass --execute to actually delete.
"""

import argparse
import json

import wandb
from wandb_gql import gql


RUNS_Q = gql("""
query Runs($entity: String!, $project: String!, $cursor: String, $filters: JSONString) {
  project(name: $project, entityName: $entity) {
    runs(filters: $filters, after: $cursor, first: 500) {
      pageInfo { hasNextPage endCursor }
      edges { node { name displayName storageBytes createdAt } }
    }
  }
}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--filter_json", default="{}")
    ap.add_argument("--files", nargs="+", default=["output.log"],
                    help="Exact file names to delete (e.g. output.log)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api = wandb.Api()

    runs, cursor = [], None
    while True:
        r = api.client.execute(RUNS_Q, variable_values={
            "entity": args.entity, "project": args.project,
            "cursor": cursor, "filters": json.dumps(json.loads(args.filter_json)),
        })
        for e in r["project"]["runs"]["edges"]:
            runs.append(e["node"])
        pi = r["project"]["runs"]["pageInfo"]
        if not pi["hasNextPage"]:
            break
        cursor = pi["endCursor"]

    if args.limit is not None:
        runs = runs[: args.limit]

    n = len(runs)
    total_mb = sum(r["storageBytes"] for r in runs) / 1024**2
    print(f"Project {args.entity}/{args.project}")
    print(f"Filter: {args.filter_json}")
    print(f"Files to delete per run: {args.files}")
    print(f"Matched {n} runs, current total {total_mb:.2f} MB")
    print("-" * 80)
    for r in runs[:3]:
        print(f"  sample: {r['name']}  {r['createdAt']}  {r['storageBytes']/1024**2:.2f} MB  {r['displayName']!r}")
    if n > 3:
        print(f"  ...and {n - 3} more")
    print("-" * 80)

    if not args.execute:
        print("DRY RUN: re-run with --execute to proceed.")
        return

    deleted_count = 0
    bytes_freed = 0
    failed = 0
    for i, run in enumerate(runs):
        rh = api.run(f"{args.entity}/{args.project}/{run['name']}")
        for fname in args.files:
            try:
                f = rh.file(fname)
                size = f.size
                f.delete()
                deleted_count += 1
                bytes_freed += size or 0
            except Exception as e:
                failed += 1
                print(f"  FAIL {run['name']}/{fname}: {type(e).__name__}: {e}")
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  processed {i + 1}/{n}  (files deleted: {deleted_count}  freed: {bytes_freed/1024**2:.1f} MB  failures: {failed})")

    print("-" * 80)
    print(f"Done: {deleted_count} files deleted, {bytes_freed/1024**2:.1f} MB freed, {failed} failures")


if __name__ == "__main__":
    main()
