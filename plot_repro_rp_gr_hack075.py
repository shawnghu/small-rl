"""Combined RP-vs-GR figure for the repro_rp_gr_hack075 sweep (Qwen3-0.6B, hack075,
monitored detect_frac=0.5). Puts the reward-penalty (RP) run on the SAME figure as
the gradient-routing (GR) run.

Two mechanisms, one figure:
  - GR localizes the hack: both-adapter hack ~0.69, but forget-ablated (retain_only)
    -> ~0.00, with task solve rate preserved.
  - The reward-penalty baseline only DISPLACES the hack: it suppresses hacking where
    monitored (~0.06) but the model still hacks where unmonitored (~0.57).

Series / colors follow the house convention (figures_pareto/proto_training_curves.py):
  GR retain_only (forget ablated) = green, GR both = orange, RP baseline = purple.

Usage:
    .venv/bin/python plot_repro_rp_gr_hack075.py
    .venv/bin/python plot_repro_rp_gr_hack075.py --sweep_dir output/repro_rp_gr_hack075
"""
import argparse
import json
import os

import matplotlib.pyplot as plt

# Run dirs within the sweep
RP_RUN = "repro_rp_qwen06b_hack075_p2_mon50_s1"
GR_RUN = "repro_gr_qwen06b_hack075_cls_cspr32_s1"

GR_RETAIN_COLOR = "#59A14F"  # green  — GR, forget adapter ablated (deployment)
GR_BOTH_COLOR = "#E8853B"    # orange — GR, both adapters active
RP_COLOR = "#6A3D9A"         # purple — reward penalty baseline

# routing_eval.jsonl metric keys (full reward-component suffixes)
HACK = "hack_freq/score_threshold"
HACK_MON = "hack_freq_detectable/score_threshold"      # monitored region
HACK_UNMON = "hack_freq_undetectable/score_threshold"  # unmonitored region
SOLVE = ("retain/mbpp_correct_from_all+mbpp_compile_from_all+"
         "mbpp_hidden_from_all+mbpp_visible_honest_from_all")


def load(run_dir):
    path = os.path.join(run_dir, "routing_eval.jsonl")
    assert os.path.exists(path), f"No routing_eval.jsonl in {run_dir}"
    by_step = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def series(records, key):
    steps = [r["step"] for r in records if key in r and r[key] is not None]
    vals = [r[key] for r in records if key in r and r[key] is not None]
    return steps, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="output/repro_rp_gr_hack075")
    args = ap.parse_args()

    rp = load(os.path.join(args.sweep_dir, RP_RUN))
    gr = load(os.path.join(args.sweep_dir, GR_RUN))

    fig, (axh, axs) = plt.subplots(1, 2, figsize=(13, 5.0))

    # ---- Panel A: hack frequency over training -------------------------------
    # GR: both adapters and forget-ablated (overall hack rate).
    axh.plot(*series(gr, f"both/{HACK}"), color=GR_BOTH_COLOR, lw=2, marker="o",
             ms=3, label="GR (both adapters)")
    axh.plot(*series(gr, f"retain_only/{HACK}"), color=GR_RETAIN_COLOR, lw=2.5,
             marker="o", ms=3, label="GR (forget ablated)")
    # RP baseline: conditional split — monitored (dashed) vs unmonitored (solid).
    axh.plot(*series(rp, f"both/{HACK_UNMON}"), color=RP_COLOR, lw=2.5, marker="s",
             ms=3, label="Reward penalty baseline (unmonitored)")
    axh.plot(*series(rp, f"both/{HACK_MON}"), color=RP_COLOR, lw=2, ls="--",
             marker="s", ms=3, label="Reward penalty baseline (monitored)")
    axh.set_xlabel("Training step")
    axh.set_ylabel("Hack frequency")
    axh.set_title("Hack frequency: GR localizes, RP only displaces")
    axh.set_ylim(-0.03, 1.0)
    axh.grid(alpha=0.3)
    axh.legend(fontsize=8, loc="upper left")

    # ---- Panel B: task solve rate over training ------------------------------
    axs.plot(*series(gr, f"retain_only/{SOLVE}"), color=GR_RETAIN_COLOR, lw=2.5,
             marker="o", ms=3, label="GR (forget ablated)")
    axs.plot(*series(gr, f"both/{SOLVE}"), color=GR_BOTH_COLOR, lw=2, marker="o",
             ms=3, label="GR (both adapters)")
    axs.plot(*series(rp, f"both/{SOLVE}"), color=RP_COLOR, lw=2, marker="s", ms=3,
             label="Reward penalty baseline")
    axs.set_xlabel("Training step")
    axs.set_ylabel("Task solve rate")
    axs.set_title("Task solve rate preserved")
    axs.set_ylim(0, 1.0)
    axs.grid(alpha=0.3)
    axs.legend(fontsize=8, loc="upper left")

    fig.suptitle("MBPP+ hack075 (Qwen3-0.6B, monitored 50%): gradient routing vs "
                 "reward penalty baseline", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = os.path.join(args.sweep_dir, "repro_rp_gr_hack075.png")
    out_pdf = os.path.join(args.sweep_dir, "repro_rp_gr_hack075.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    # Print final-step summary for sanity.
    print("\nFinal-step values:")
    for name, recs in [("GR", gr), ("RP", rp)]:
        last = recs[-1]
        print(f"  {name} @step {last['step']}: "
              f"hack both={last.get(f'both/{HACK}'):.3f} "
              f"ablated={last.get(f'retain_only/{HACK}'):.3f} "
              f"mon={last.get(f'both/{HACK_MON}'):.3f} "
              f"unmon={last.get(f'both/{HACK_UNMON}'):.3f} "
              f"solve(ablated)={last.get(f'retain_only/{SOLVE}'):.3f}")


if __name__ == "__main__":
    main()
