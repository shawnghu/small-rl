"""Focused analysis on the coh-vs-routing JSONs:

  Metric 1: Group-size buckets (per mode)
    - n_non_rh ∈ {0, 1}      → zero gradient (n=0 → empty; n=1 → variance=0)
    - n_non_rh ∈ {2, 3, 4}   → noisy renorm
    - n_non_rh ∈ {5..15}     → healthy with routing (real renorm)
    - n_non_rh = 16          → healthy with no routing (no filter applied)

  Metric 2 (routing steps only): among groups with n_non_rh >= 5 AND nonzero
    post-renorm variance, what fraction have 0 correct solutions in the filtered
    subset AND >=1 undetected-hack sample in it. Those are groups where retain
    training has no clean reward to chase except the undetected-hack reward —
    pure hack reinforcement.

Usage: .venv/bin/python tools/analyze_group_buckets.py output/diagnostics/coh_vs_routing_classic_coh_s7
"""
import json
import os
import sys

import numpy as np

MAX_REWARD = 3.5


def _recompute(ps, n_per, n_grp):
    """Re-derive rewards (capped) + group-level filter stats from per-sample
    arrays, matching what training would see."""
    trait     = np.array(ps["trait"])
    correct_f = np.array(ps["correct"])
    compile_  = np.array(ps["compile"])
    is_rh     = np.array(ps["is_rh"])
    is_array  = np.array(ps["is_array"])
    is_hack_p = np.array(ps["is_hackable"])
    correct_b = correct_f > 0.5
    gt_hack   = trait > 0.5
    # capped reward, matches CombinedReward in training
    reward = np.minimum(
        3.0 * correct_f + np.where(is_hack_p, 3.0 * trait, 0.0) + 0.5 * compile_,
        MAX_REWARD,
    )
    return {
        "reward": reward, "trait": trait, "correct_f": correct_f,
        "correct_b": correct_b, "gt_hack": gt_hack,
        "is_rh": is_rh, "is_array": is_array, "is_hack_p": is_hack_p,
    }


def bucket_label(n):
    if n <= 1: return "0-1 (zero grad)"
    if n <= 4: return "2-4 (noisy)"
    if n <= 15: return "5-15 (healthy w/ routing)"
    return "16 (healthy no routing)"


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    ckpt = data["checkpoint"]
    print(f"\n{'='*72}\n  {ckpt}   (capped at {MAX_REWARD})\n{'='*72}")

    # ----- Metric 1: bucketed group-size distribution per mode -----
    print("\nMetric 1: post-filter group-size buckets")
    print(f"  {'mode':<6} {'0-1 (zero)':>14} {'2-4 (noisy)':>14} "
          f"{'5-15 (renorm)':>16} {'16 (no filter)':>16} {'total':>7}")
    for mode in ("rout", "coh"):
        m = data["modes"][mode]
        hist = m["histogram"]
        b_zero  = hist[0] + hist[1]
        b_noisy = hist[2] + hist[3] + hist[4]
        b_renorm = sum(hist[5:16])
        b_full  = hist[16]
        tot = sum(hist)
        print(f"  {mode:<6} "
              f"{b_zero:>4} ({b_zero/tot*100:>4.1f}%) "
              f"{b_noisy:>4} ({b_noisy/tot*100:>4.1f}%) "
              f"{b_renorm:>4} ({b_renorm/tot*100:>5.1f}%) "
              f"{b_full:>4} ({b_full/tot*100:>5.1f}%) "
              f"{tot:>7}")

    # ----- Metric 2: pure-hack-reinforcement groups in routing -----
    print("\nMetric 2: ROUT — groups where retain training only reinforces hacking")
    print("  criterion: n_non_rh >= 5 AND post-renorm variance > 0 AND")
    print("             (filtered subset has 0 correct samples) AND")
    print("             (filtered subset has >=1 undetected-hack sample)")
    m = data["modes"]["rout"]
    ps = m["per_sample"]
    n_per = m["n_per_prompt"]
    n_grp = m["n_prompts"]
    d = _recompute(ps, n_per, n_grp)

    # Per-group classify
    cat_counts = {"size<5": 0, "size>=5_zero_variance": 0,
                  "all_correct": 0, "has_correct": 0,
                  "no_correct_no_undet_hack": 0,
                  "no_correct_with_undet_hack": 0}  # ← the hack-reinforcement case
    n_size_ge5_nonzero_var = 0
    bad_groups = []  # detail for the pure-hack ones

    for g in range(n_grp):
        sl = slice(g*n_per, (g+1)*n_per)
        good_mask_g = ~d["is_rh"][sl]
        n_good = int(good_mask_g.sum())
        if n_good < 5:
            cat_counts["size<5"] += 1
            continue
        rewards_good = d["reward"][sl][good_mask_g]
        # nonzero variance iff not all rewards in subset are equal
        if rewards_good.max() == rewards_good.min():
            cat_counts["size>=5_zero_variance"] += 1
            continue
        n_size_ge5_nonzero_var += 1

        correct_in_good = int(d["correct_b"][sl][good_mask_g].sum())
        # ground-truth hacks IN the filtered subset = undetected hacks
        gt_hack_in_good = int(d["gt_hack"][sl][good_mask_g].sum())

        if correct_in_good == n_good:
            cat_counts["all_correct"] += 1
        elif correct_in_good > 0:
            cat_counts["has_correct"] += 1
        elif gt_hack_in_good == 0:
            cat_counts["no_correct_no_undet_hack"] += 1
        else:
            cat_counts["no_correct_with_undet_hack"] += 1
            bad_groups.append({
                "g": g, "n_good": n_good,
                "n_correct": correct_in_good,
                "n_undet_hack": gt_hack_in_good,
                "n_incorrect_nonhack": n_good - correct_in_good - gt_hack_in_good,
                "reward_range": [float(rewards_good.min()), float(rewards_good.max())],
            })

    print(f"\n  All {n_grp} routing groups:")
    for k, v in cat_counts.items():
        print(f"    {k:32}: {v:4d} ({v/n_grp*100:>5.1f}%)")
    print(f"\n  Of {n_size_ge5_nonzero_var} groups with n_non_rh>=5 AND nonzero variance:")
    bad = cat_counts["no_correct_with_undet_hack"]
    pct = bad / n_size_ge5_nonzero_var * 100 if n_size_ge5_nonzero_var else 0.0
    print(f"    pure hack-reinforcement (0 correct, >=1 undet hack): {bad}/{n_size_ge5_nonzero_var}  ({pct:.1f}%)")

    if bad_groups:
        print(f"\n  Detail of those {len(bad_groups)} pure-hack-reinforcement groups:")
        print(f"    {'g':>3} {'n_good':>7} {'n_correct':>10} {'n_undet_hack':>13} "
              f"{'n_incorrect':>12} {'reward range':>14}")
        for bg in bad_groups:
            print(f"    {bg['g']:>3} {bg['n_good']:>7} {bg['n_correct']:>10} "
                  f"{bg['n_undet_hack']:>13} {bg['n_incorrect_nonhack']:>12} "
                  f"  [{bg['reward_range'][0]:.2f}, {bg['reward_range'][1]:.2f}]")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
          "output/diagnostics/coh_vs_routing_classic_coh_s7"
    if os.path.isdir(src):
        files = sorted(os.path.join(src, f) for f in os.listdir(src)
                       if f.endswith(".json"))
    else:
        files = [src]
    for f in files:
        analyze(f)


if __name__ == "__main__":
    main()
