"""Post-process the coh-vs-routing diagnostic JSON written by
tools.modal_train_gr.coh_vs_routing_group_analysis. Computes per-category
advantage distributions, sign-flip statistics, and group-size histograms.

Usage:
    .venv/bin/python tools/analyze_coh_vs_routing.py output/diagnostics/coh_vs_routing_classic_coh_s7
"""
import json
import os
import sys

import numpy as np


def _summary(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return "n=0"
    return (f"n={x.size}  mean={x.mean():+.3f}  std={x.std():.3f}  "
            f"p10={np.percentile(x,10):+.3f}  p25={np.percentile(x,25):+.3f}  "
            f"p50={np.percentile(x,50):+.3f}  p75={np.percentile(x,75):+.3f}  "
            f"p90={np.percentile(x,90):+.3f}")


def analyze_ckpt(path, max_reward=3.5):
    """max_reward: cap applied to combined reward, matching CombinedReward in
    training (rewards.py:1480). Set to None to use uncapped rewards.
    leetcode_rh_array.yaml uses max_reward=3.5."""
    with open(path) as f:
        data = json.load(f)
    ckpt = data["checkpoint"]
    print(f"\n{'='*72}\n  {ckpt}  (max_reward cap = {max_reward})\n{'='*72}")
    for mode in ("coh", "rout"):
        m = data["modes"][mode]
        ps = m["per_sample"]
        trait    = np.array(ps["trait"])
        correct_f = np.array(ps["correct"])         # FRACTIONAL pass_rate
        correct  = correct_f > 0.5
        compile_ = np.array(ps["compile"])
        is_rh    = np.array(ps["is_rh"])
        gt_hack  = np.array(ps["gt_hack"])          # trait > 0.5
        is_array = np.array(ps["is_array"])
        is_hack_p = np.array(ps["is_hackable"])
        n_total = len(trait)
        n_per   = m["n_per_prompt"]
        n_grp   = m["n_prompts"]

        # Re-compute reward WITH the proper max_reward cap (matches training).
        # leetcode_rh_array yaml: 3*correct + 3*trait(forget,gated on hackable) + 0.5*compile
        raw_combined = 3.0*correct_f + np.where(is_hack_p, 3.0*trait, 0.0) + 0.5*compile_
        if max_reward is not None:
            reward = np.minimum(raw_combined, max_reward)
        else:
            reward = raw_combined

        # Recompute advantages with the (possibly capped) reward.
        eps = 1e-4
        std_adv = np.zeros(n_total)
        ret_adv = np.zeros(n_total)
        for g in range(n_grp):
            sl = slice(g*n_per, (g+1)*n_per)
            r_g = reward[sl]; rh_g = is_rh[sl]
            mu = r_g.mean(); sd = r_g.std()
            std_adv[sl] = (r_g - mu) / (sd + eps)
            good = ~rh_g
            if good.any():
                r_good = r_g[good]
                mu_g = r_good.mean(); sd_g = r_good.std()
                # only ~is_rh slots get nonzero retain_adv; rh slots stay 0
                adv_g = np.zeros(n_per)
                adv_g[good] = (r_good - mu_g) / (sd_g + eps)
                ret_adv[sl] = adv_g

        # forget effective advantage in classic + retain_mode=renormalize:
        #   standard on is_rh samples (bad pass), retain on ~is_rh (good pass)
        forget_eff_adv = np.where(is_rh, std_adv, ret_adv)

        # categories
        detected_hack    = is_rh
        nondet_hack      = gt_hack & ~is_array
        correct_nonhack  =  correct & ~gt_hack
        incorrect_nonhack = ~correct & ~gt_hack

        print(f"\n--- {mode}  (n={n_total}, {n_grp} groups × {n_per})  ---")
        n_array = int(is_array.sum())
        n_hackable = int(np.array(ps["is_hackable"]).sum())
        print(f"  prompt mix:     {n_array}/{n_total} Array tag  ({n_array//n_per}/{n_grp} groups), "
              f"{n_hackable}/{n_total} hackable")
        print(f"  category counts (samples):")
        print(f"    detected_hack    (is_rh, Array∧trait>.5): {detected_hack.sum()}")
        print(f"    nondetected_hack (¬Array∧trait>.5)      : {nondet_hack.sum()}")
        print(f"    correct_nonhack  (correct∧¬trait>.5)    : {correct_nonhack.sum()}")
        print(f"    incorrect_nonhack(¬correct∧¬trait>.5)   : {incorrect_nonhack.sum()}")
        print(f"  reward distribution: {_summary(reward)}")

        # Group-size histogram (n_non_rh per group)
        print(f"  n_non_rh per group histogram (k → #groups, k=0..16):")
        print(f"    {m['histogram']}   mean={m['mean_n_non_rh']:.2f}")

        # Retain adapter sees only ~is_rh samples, weighted by retain_adv
        print(f"  RETAIN advantage (samples with ~is_rh; weighted by retain_adv):")
        for label, mask in (("nondet_hack",        nondet_hack & ~is_rh),
                             ("correct_nonhack",    correct_nonhack & ~is_rh),
                             ("incorrect_nonhack",  incorrect_nonhack & ~is_rh)):
            print(f"    {label:18}: {_summary(ret_adv[mask])}")

        # Forget effective: standard on is_rh, retain on ~is_rh
        print(f"  FORGET effective advantage (standard on is_rh, retain on ~is_rh):")
        for label, mask in (("detected_hack",       detected_hack),
                             ("nondet_hack",         nondet_hack),
                             ("correct_nonhack",     correct_nonhack),
                             ("incorrect_nonhack",   incorrect_nonhack)):
            print(f"    {label:18}: {_summary(forget_eff_adv[mask])}")

        # The "is this a bug" cross-check: retain_adv vs standard_adv on ~is_rh
        # samples (= the difference between what forget actually sees vs what
        # pure-standard GRPO would give it on good-pass samples)
        nonrh = ~is_rh
        if nonrh.any():
            diff = ret_adv[nonrh] - std_adv[nonrh]
            print(f"  swap effect on ~is_rh samples (retain_adv − standard_adv):")
            print(f"    shift:               {_summary(diff)}")
            print(f"    |shift|:             {_summary(np.abs(diff))}")
            std_n = std_adv[nonrh]
            ret_n = ret_adv[nonrh]
            n_nonrh = int(nonrh.sum())
            joint = {
                "neg_neg": int(((std_n < 0) & (ret_n < 0)).sum()),
                "neg_pos": int(((std_n < 0) & (ret_n > 0)).sum()),
                "pos_neg": int(((std_n > 0) & (ret_n < 0)).sum()),
                "pos_pos": int(((std_n > 0) & (ret_n > 0)).sum()),
                "zero_either": int(((std_n == 0) | (ret_n == 0)).sum()),
            }
            print(f"    sign joint (standard → retain) on n={n_nonrh}:")
            for k, v in joint.items():
                pct = (v / n_nonrh * 100) if n_nonrh else 0
                print(f"      {k:10}: {v:4d}  ({pct:4.1f}%)")
            # restrict to mixed groups (≥1 hack ∧ ≥1 non-hack) — the only ones
            # where retain != standard mathematically
            grp_idx = np.arange(n_total) // n_per
            is_mixed = np.zeros(n_grp, dtype=bool)
            for g in range(n_grp):
                sl = slice(g*n_per, (g+1)*n_per)
                if is_rh[sl].any() and (~is_rh[sl]).any():
                    is_mixed[g] = True
            sample_mixed = is_mixed[grp_idx]
            m_nonrh = sample_mixed & nonrh
            n_m = int(m_nonrh.sum())
            print(f"    mixed groups: {int(is_mixed.sum())}/{n_grp}; among their non-RH samples (n={n_m}):")
            if n_m:
                m_std = std_adv[m_nonrh]; m_ret = ret_adv[m_nonrh]
                m_joint = {
                    "neg_neg": int(((m_std < 0) & (m_ret < 0)).sum()),
                    "neg_pos": int(((m_std < 0) & (m_ret > 0)).sum()),
                    "pos_neg": int(((m_std > 0) & (m_ret < 0)).sum()),
                    "pos_pos": int(((m_std > 0) & (m_ret > 0)).sum()),
                }
                for k, v in m_joint.items():
                    pct = v / n_m * 100
                    print(f"      {k:10}: {v:4d}  ({pct:4.1f}%)")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
          "output/diagnostics/coh_vs_routing_classic_coh_s7"
    if os.path.isdir(src):
        files = sorted(os.path.join(src, f) for f in os.listdir(src)
                       if f.endswith(".json"))
    else:
        files = [src]
    for f in files:
        analyze_ckpt(f)


if __name__ == "__main__":
    main()
