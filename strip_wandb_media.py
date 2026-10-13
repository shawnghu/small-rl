"""Strip `media/` (or other folder) files from wandb runs via deleteFilesByRunAndFolder.

Default: dry-run. Pass --execute to actually strip.
"""

import argparse
import base64
import json

import wandb
from wandb_gql import gql


PROJECT_Q = gql("""
query P($n: String!, $e: String!) {
  project(name: $n, entityName: $e) { internalId runCount }
}
""")

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

STRIP_M = gql("""
mutation D($input: DeleteFilesByRunAndFolderInput!) {
  deleteFilesByRunAndFolder(input: $input) { clientMutationId }
}
""")


def decode_int_id(b64: str) -> int:
    return int(base64.b64decode(b64).decode().split(":")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--filter_json", default="{}")
    ap.add_argument("--folders", nargs="+", default=["media"])
    ap.add_argument("--older_half", action="store_true", help="Only the older half (by createdAt)")
    ap.add_argument("--older_fraction", type=float, default=None,
                    help="Only the older F fraction (0<F<=1)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api = wandb.Api()

    # Resolve internal int project id
    pr = api.client.execute(PROJECT_Q, variable_values={"n": args.project, "e": args.entity})
    int_id = decode_int_id(pr["project"]["internalId"])
    print(f"Project {args.entity}/{args.project}  internalIntId={int_id}  totalRuns={pr['project']['runCount']}")

    # Collect candidates
    filt = json.loads(args.filter_json)
    runs, cursor = [], None
    while True:
        r = api.client.execute(RUNS_Q, variable_values={
            "entity": args.entity, "project": args.project,
            "cursor": cursor, "filters": json.dumps(filt),
        })
        for e in r["project"]["runs"]["edges"]:
            runs.append(e["node"])
        pi = r["project"]["runs"]["pageInfo"]
        if not pi["hasNextPage"]: break
        cursor = pi["endCursor"]

    runs.sort(key=lambda x: x["createdAt"])
    matched = len(runs)
    if args.older_half:
        runs = runs[: matched // 2]
    if args.older_fraction is not None:
        assert 0 < args.older_fraction <= 1
        runs = runs[: int(matched * args.older_fraction)]
    if args.limit is not None:
        runs = runs[: args.limit]

    n = len(runs)
    print(f"Filter matched {matched}; selected {n} runs (folders={args.folders})")
    if n == 0:
        print("No runs selected.")
        return
    print(f"Oldest selected:  {runs[0]['name']}  {runs[0]['createdAt']}  ({runs[0]['storageBytes']/1024**2:.2f} MB)")
    print(f"Newest selected:  {runs[-1]['name']}  {runs[-1]['createdAt']}  ({runs[-1]['storageBytes']/1024**2:.2f} MB)")
    total_mb = sum(r["storageBytes"] for r in runs) / 1024**2
    print(f"Pre-strip total:  {total_mb:.2f} MB across {n} runs")
    print("-" * 80)

    if not args.execute:
        print("DRY RUN: re-run with --execute to proceed.")
        return

    ok = fail = 0
    for i, run in enumerate(runs):
        v = {"input": {"projectId": int_id, "runName": run["name"], "folders": args.folders}}
        try:
            api.client.execute(STRIP_M, variable_values=v)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {run['name']}: {type(e).__name__}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  stripped {ok}/{n}  (failures: {fail})")

    print("-" * 80)
    print(f"Done: {ok} stripped, {fail} failed")


if __name__ == "__main__":
    main()
