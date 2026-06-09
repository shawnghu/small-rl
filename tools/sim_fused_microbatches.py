"""Simulate stock-vs-fused microbatch packing under realistic class proportions.

Whether --fused_reduction saves wall-clock at 8b hinges on packing combinatorics:
the stock path packs each class (good / bad / coh) into its OWN homogeneous
token-budget microbatches, so small per-class counts each round up to >=1
(mostly-empty) microbatch; fused packs all classes together. This is pure
first-fit-decreasing bin packing (train.py:_pack_by_tokens) — no GPU needed — so
we can count the microbatches that actually get computed, and their sizes, for
realistic inputs.

Realistic inputs are drawn from real leetcode-array routing traces:
  - completion lengths sampled from the actual per-sample distribution (RH longer
    than non-RH), + a fixed prompt length.
  - class mix: coh fraction of the rollout, bad (detected-hack) fraction of the
    routing slice (mature ~0.25-0.30 from the traces; ramps from ~0 early).

Usage:
  python tools/sim_fused_microbatches.py --opt_batch_sizes 16,32,64,128,512 --max_tokens 12000 --prompt_len 400 --bad_frac 0.27 --coh_frac 0.2
"""
import argparse
import glob
import json
import random
import statistics


def _pack_by_tokens(token_counts, indices, max_tokens):
    """First-fit decreasing bin packing — replicated from train.py."""
    if not indices:
        return []
    pairs = sorted([(i, token_counts[i]) for i in indices], key=lambda x: -x[1])
    bins = []
    for idx, tokens in pairs:
        placed = False
        for b in range(len(bins)):
            if bins[b][0] + tokens <= max_tokens:
                bins[b] = (bins[b][0] + tokens, bins[b][1] + [idx])
                placed = True
                break
        if not placed:
            bins.append((tokens, [idx]))
    return [b[1] for b in bins]


def _load_completion_pools():
    """Real per-sample completion lengths from leetcode-array traces, split rh/nonrh."""
    rh, nonrh = [], []
    for f in glob.glob("output/leetcode_array*/*/routing_trace.jsonl")[:8]:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("trace") == "sample" and r.get("completion_len") is not None:
                    (rh if r.get("is_rh") else nonrh).append(r["completion_len"])
    assert rh and nonrh, "no trace data found under output/leetcode_array*/"
    return rh, nonrh


def _mb_sizes(mbs, token_counts):
    return [(len(mb), sum(token_counts[i] for i in mb)) for mb in mbs]


def simulate(B, max_tokens, prompt_len, coh_frac, bad_frac, rh_pool, nonrh_pool, trials, rng):
    """One optimizer batch of B samples; returns (stock_n, fused_n, stock_fill, fused_fill)."""
    stock_ns, fused_ns, stock_fills, fused_fills = [], [], [], []
    for _ in range(trials):
        n_coh = round(B * coh_frac)
        n_route = B - n_coh
        n_bad = round(n_route * bad_frac)
        n_good = n_route - n_bad
        # token_counts indexed 0..B-1; classes contiguous: good, bad, coh
        tc = {}
        idx = 0
        good_idx, bad_idx, coh_idx = [], [], []
        for _ in range(n_good):
            tc[idx] = prompt_len + rng.choice(nonrh_pool); good_idx.append(idx); idx += 1
        for _ in range(n_bad):
            tc[idx] = prompt_len + rng.choice(rh_pool); bad_idx.append(idx); idx += 1
        for _ in range(n_coh):
            tc[idx] = prompt_len + rng.choice(nonrh_pool); coh_idx.append(idx); idx += 1
        # stock: each class packed separately (homogeneous microbatches)
        stock = (_pack_by_tokens(tc, good_idx, max_tokens)
                 + _pack_by_tokens(tc, bad_idx, max_tokens)
                 + _pack_by_tokens(tc, coh_idx, max_tokens))
        # fused: all together (heterogeneous)
        fused = _pack_by_tokens(tc, good_idx + bad_idx + coh_idx, max_tokens)
        stock_ns.append(len(stock)); fused_ns.append(len(fused))
        # mean fill = mean tokens/microbatch / budget
        ssz = _mb_sizes(stock, tc); fsz = _mb_sizes(fused, tc)
        stock_fills.append(statistics.mean(t for _, t in ssz) / max_tokens if ssz else 0)
        fused_fills.append(statistics.mean(t for _, t in fsz) / max_tokens if fsz else 0)
    return (statistics.mean(stock_ns), statistics.mean(fused_ns),
            statistics.mean(stock_fills), statistics.mean(fused_fills))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_batch_sizes", default="16,32,64,128,256,512")
    ap.add_argument("--max_tokens", type=int, default=12000)
    ap.add_argument("--prompt_len", type=int, default=400)
    ap.add_argument("--coh_frac", type=float, default=0.2)
    ap.add_argument("--bad_frac", type=float, default=0.27, help="hack fraction of the routing slice")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--per_mb_overhead", type=float, default=1.10,
                    help="fused per-microbatch time / stock (measured ~1.10 eager at 8b m64)")
    args = ap.parse_args()

    rng = random.Random(0)
    rh_pool, nonrh_pool = _load_completion_pools()
    print(f"loaded {len(rh_pool)} rh + {len(nonrh_pool)} nonrh real completion lengths")
    print(f"max_tokens={args.max_tokens} prompt_len={args.prompt_len} "
          f"coh_frac={args.coh_frac} bad_frac={args.bad_frac} (mature) "
          f"per_mb_overhead={args.per_mb_overhead}")
    print(f"\n{'opt_bs':>7} | {'stock_mb':>9} {'fused_mb':>9} {'mb_saved':>9} | "
          f"{'stock_fill':>10} {'fused_fill':>10} | {'net(time)':>9}")
    print("-" * 78)
    for B in [int(x) for x in args.opt_batch_sizes.split(",")]:
        s_n, f_n, s_fill, f_fill = simulate(
            B, args.max_tokens, args.prompt_len, args.coh_frac, args.bad_frac,
            rh_pool, nonrh_pool, args.trials, rng)
        # crude net-time model: time ~ shared_compute (same tokens) is NOT counted
        # here (it cancels); this ratio is the OVERHEAD model — per-mb fixed cost
        # x count, with fused paying per_mb_overhead. Only meaningful when per-mb
        # fixed overhead dominates (small/underutilized microbatches).
        net = (f_n * args.per_mb_overhead) / s_n if s_n else float("nan")
        print(f"{B:>7} | {s_n:>9.2f} {f_n:>9.2f} {s_n - f_n:>9.2f} | "
              f"{s_fill:>9.1%} {f_fill:>9.1%} | {net:>8.3f}x")
    print("\nnet(time) <1 => fused fewer-microbatch saving beats its per-mb overhead")
    print("(overhead-bound model; valid when microbatches are small/underutilized —")
    print(" if microbatches are compute-bound the shared-token compute dominates and net~=1)")


if __name__ == "__main__":
    main()
