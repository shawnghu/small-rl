"""Composite Figure 1 variant — partial-forget, with topic_contains excluded.

Same as proto_figure1_partial_forget but renders over a 6-env subset
(excludes topic_contains). Useful for seeing whether the partial-forget
trajectory result is dominated by topic-env outliers.

Run:
    .venv/bin/python figures_pareto/proto_figure1_partial_forget_no_topic.py
"""
import os
import matplotlib.pyplot as plt

# Filter ENVS *before* importing the panel modules, so their module-level
# gatherers see the filtered list. The import order matters here.
import proto_pareto_data as _ppd
_ppd.ENVS = [e for e in _ppd.ENVS if e != "topic_contains"]

# Same filter for the style module's SLOT_ENVS so the layout matches.
import proto_pareto_style_v2 as _ppstyle
_ppstyle.SLOT_ENVS = [(slot, env) for slot, env in _ppstyle.SLOT_ENVS
                     if env != "topic_contains"]

# Force-reload the modules that capture ENVS at import time. (They don't —
# they reference proto_pareto_data.ENVS lazily inside functions — but doing
# this is harmless and protects against future refactors.)
from proto_pareto_monitored_partial_forget import draw_scatter, legend_handles  # noqa: E402
from proto_uplift_panel_partial_forget import draw as draw_uplift  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.size"] = 20
plt.rcParams["axes.unicode_minus"] = False


def main():
    fig = plt.figure(figsize=(17.0, 9.0))
    sub_l, sub_r = fig.subfigures(1, 2, width_ratios=[9.0, 8.0], wspace=0.0)
    TOP, BOT = 0.97, 0.10

    ax_sc = sub_l.subplots(1, 1)
    draw_scatter(ax_sc)
    ax_sc.legend(handles=legend_handles(), loc="lower right",
                 frameon=True, fontsize=13)
    sub_l.subplots_adjust(left=0.10, right=0.99, top=TOP, bottom=BOT)

    ax_top, ax_bot = sub_r.subplots(2, 1, sharex=True)
    draw_uplift(ax_top, "retain", subtract_base=True)
    ax_top.set_ylabel("Task performance\nimprovement")
    draw_uplift(ax_bot, "hack_freq", subtract_base=False)
    ax_bot.set_ylabel("Reward hack rate")
    ax_bot.set_xlabel("Training step")
    sub_r.subplots_adjust(left=0.17, right=0.97, top=TOP, bottom=BOT, hspace=0.07)
    sub_r.align_ylabels([ax_top, ax_bot])

    out_pdf = os.path.join(HERE, "figs", "proto_figure1_partial_forget_no_topic.pdf")
    out_png = os.path.join(HERE, "figs", "proto_figure1_partial_forget_no_topic.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
