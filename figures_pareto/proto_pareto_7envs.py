"""Main-paper figure: GR (retain-only) vs the BEST reward-penalty variant
per environment, plus universal baselines (base model, verified-only).

For each env we pick the RP variant (across penalty/multiplier/extras-ratio
sweeps) that maximizes
  unified = retain_reward - hack_freq_undetectable
The winning variant is annotated next to the RP marker.
Verified-only ('0:1' ratio) is excluded from this selection — it's a
universal baseline shown on every figure.
"""
import matplotlib.pyplot as plt

from proto_pareto_data import ENVS, best_rp
from proto_pareto_layout import (
    setup_axes, draw_universal_baselines, draw_legend_slot, save_figure,
)


PLOT_SLOTS = [0, 1, 2, 4, 5, 6, 7]
LEGEND_SLOT = 3


def main():
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes = axes.flatten()

    for env, i in zip(ENVS, PLOT_SLOTS):
        ax = axes[i]
        br = best_rp(env)
        if br is not None:
            label, agg, _ = br
            draw_universal_baselines(ax, env, best_rp=(label, agg))
        else:
            draw_universal_baselines(ax, env)
        setup_axes(ax, env, i)

    draw_legend_slot(axes[LEGEND_SLOT],
                     rp_canonical_label='Reward Penalty (best variant)')

    save_figure(fig, 'proto_pareto_7envs_gr_rp.png')

    # Print best-RP table for paper caption
    print('\n=== best-RP per env (winner shown in figure) ===')
    print(f'{"env":<14} {"winner":<14} {"retain":>7} {"hack_und":>9} {"score":>7}')
    for env in ENVS:
        br = best_rp(env)
        if br is None:
            print(f'{env:<14} (no RP data)')
            continue
        label, agg, score = br
        r_m, _, h_m, _, _ = agg
        print(f'{env:<14} {label:<14} {r_m:>7.2f} {h_m:>9.2f} {score:>7.2f}')


if __name__ == '__main__':
    main()
