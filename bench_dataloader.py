"""Benchmark DataLoader next() call overhead for leetcode dataset."""
import time
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator

import envs

def identity(x):
    return x

def bench(dl, n=16, label=""):
    it = iter(dl)
    # Warm up
    next(it)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        next(it)
        times.append(time.perf_counter() - t0)
    mean_ms = sum(times) / len(times) * 1000
    print(f"{label}: {mean_ms:.1f}ms/call (n={n})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_columns", nargs="*", default=[], help="Columns to drop before benchmarking")
    parser.add_argument("--keep_only", nargs="*", default=[], help="Keep only these columns (plus prompt)")
    parser.add_argument("--no_accelerator", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()

    ns = argparse.Namespace(leetcode_hint="simple_overwrite_tests_aware", hack_frac=1.0)
    spec = envs.get_env("leetcode")
    ds = spec.load_train(ns)
    print(f"Dataset: {len(ds)} rows, columns: {ds.column_names}")

    if args.keep_only:
        keep = set(args.keep_only) | {"prompt"}
        drop = [c for c in ds.column_names if c not in keep]
        ds = ds.remove_columns(drop)
        print(f"Kept only: {ds.column_names}")
    elif args.drop_columns:
        ds = ds.remove_columns(args.drop_columns)
        print(f"After drop: {ds.column_names}")

    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=identity, shuffle=False)

    bench(dl, n=args.n, label="raw DataLoader")

    if not args.no_accelerator:
        acc = Accelerator()
        dl_acc = acc.prepare(dl)
        bench(dl_acc, n=args.n, label="accelerator-prepared DataLoader")

if __name__ == "__main__":
    main()
