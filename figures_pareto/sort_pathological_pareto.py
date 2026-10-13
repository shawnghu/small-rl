"""Appendix figure: the STOCK (pre-2026-07-26-redesign) sort env under the
canonical method suite — gr-paper, subsection "A more pathological variation
of the 'sort' environment".

Stock sort = sorting_copy_respmon.yaml: hackable gate is the latent
max-element-first property and BOTH prompt conditions render identically
("Sort: {inp}" — no format line on unhackable prompts), so the policy cannot
distinguish them; copying is a local optimum of the retain reward even where
the hack pays nothing, and sorting never develops. The redesigned env
(sorting_framed) replaces this in the main figure.

Arms (all 3 seeds, canonical recipes; same conventions as the paper's
main-body figure proto_respmon6_rv2canon_sidebyside.py):
  noi     do-nothing                                   both
  filt    filtering                                    both
  rp_best reward penalty 2.0 (only dose run on stock)  both
  gr_pre  GRAFT, forget params enabled                 both
  gr      GRAFT (canonical rv2: coh128 pen2 balanced   retain_only
          split-moment lam1 — anchor slice from the batch distribution
          with a -2.0 penalty on monitor-flagged anchor samples)
  base    base model (first eval row of dn+rp+gr)      both
GRAFT-w/o-routing and GRAFT-w/o-anchoring were never run on the stock env.

Metrics: y = retain component on ALL prompts, x = hack rate on HACKABLE
prompts; tail mean (last 10% of eval rows); SEM error bars over seeds.

Run:  cd figures_pareto && /workspace/small-rl/.venv/bin/python sort_pathological_pareto.py
Then copy figs/sort_pathological_pareto.pdf to gr-paper/figures/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import yaml

OUT_ROOT = "/workspace/small-rl/output"
RETAIN_SLUG = "retain"
HACK_SLUG = "hack_freq_hackable"
N_SEEDS = 3

# key -> (label, color, marker, hollow) — verbatim from the paper's canonical
# proto_pareto_style_v2.STYLES (Jake's checkout; ours has diverged labels).
STYLES = {
    "gr":      ("GRAFT (ours)",                    "#2ca02c", "o", False),
    "gr_pre":  ("GRAFT (forget parameters enabled)", "#1f77b4", "o", False),
    "noi":     ("No intervention",                 "#e0905a", "X", False),
    "rp_best": ("Reward Penalty",                  "#d62728", "s", False),
    "filt":    ("Filtering",                       "#b09680", "D", False),
    "base":    ("Base model",                      "#444444", "o", True),
}

_PREFIX = "sorting_copy_respmon"
_DN_RP_SWEEP = "respmon_repeat_sorting_dn_rp_3seed"
_GR_SWEEP = "respmon_rv2_gr_3seed"
_FILT_SWEEP = "respmon_canon_filter_3seed"


def _run_dirs(sweep, pat):
    root = os.path.join(OUT_ROOT, sweep)
    dirs = sorted(d for d in os.listdir(root) if d.startswith(f"{_PREFIX}_{pat}"))
    assert len(dirs) == N_SEEDS, f"{sweep}/{_PREFIX}_{pat}*: found {dirs}"
    return [os.path.join(root, d) for d in dirs]


def _rows(path):
    with open(os.path.join(path, "routing_eval.jsonl")) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert rows, f"empty routing_eval in {path}"
    by_step = {r["step"]: r for r in rows}   # keep-last per step
    return [by_step[s] for s in sorted(by_step)]


def _finished(path):
    with open(os.path.join(path, "run_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    last = _rows(path)[-1]["step"]
    assert last >= cfg["max_steps"] - cfg.get("eval_every", 10), \
        f"{path}: last eval step {last} < max_steps {cfg['max_steps']}"


def _tail_mean(rows, mode, slug, first_row=False):
    sel = rows[:1] if first_row else rows[-max(1, len(rows) // 10):]
    ks = [k for k in sel[0] if k.startswith(f"{mode}/{slug}/")]
    assert len(ks) == 1, f"{mode}/{slug}: matched {ks}"
    return float(np.mean([r[ks[0]] for r in sel]))


def _points(paths, mode, first_row=False):
    out = []
    for p in paths:
        _finished(p)
        rows = _rows(p)
        out.append((_tail_mean(rows, mode, RETAIN_SLUG, first_row),
                    _tail_mean(rows, mode, HACK_SLUG, first_row)))
    return out


def _sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def _agg(points):
    rs = np.array([q[0] for q in points])
    hs = np.array([q[1] for q in points])
    return float(rs.mean()), _sem(rs), float(hs.mean()), _sem(hs), len(points)


def main():
    dn = _run_dirs(_DN_RP_SWEEP, "donothing")
    rp = _run_dirs(_DN_RP_SWEEP, "rp2")
    gr = _run_dirs(_GR_SWEEP, "gr_cls")
    filt = _run_dirs(_FILT_SWEEP, "filter")

    series = [  # draw order == z-order; 'gr' forced on top
        ("noi",     _agg(_points(dn, "both"))),
        ("filt",    _agg(_points(filt, "both"))),
        ("rp_best", _agg(_points(rp, "both"))),
        ("gr_pre",  _agg(_points(gr, "both"))),
        ("gr",      _agg(_points(gr, "retain_only"))),
        ("base",    _agg(_points(dn + rp + gr, "both", first_row=True))),
    ]

    print(f'{"series":<8} {"retain":>7} {"hack":>7} {"n":>2}')
    for key, (r_m, r_s, h_m, h_s, n) in series:
        print(f"{key:<8} {r_m:>7.3f} {h_m:>7.3f} {n:>2}")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for i, (key, (r_m, r_s, h_m, h_s, _n)) in enumerate(series):
        label, color, marker, hollow = STYLES[key]
        ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                    fmt=marker, color=color, markersize=13,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color if hollow else "white",
                    markeredgewidth=2.0 if hollow else 1.6,
                    ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2,
                    zorder=50 if key == "gr" else 8 + i,
                    clip_on=key != "gr")
    ax.set_box_aspect(1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Reward hack rate (hackable prompts)", fontsize=14)
    ax.set_ylabel("Target-task (retain) reward", fontsize=14)
    handles = [Line2D([], [], marker=m, linestyle="none", color=c,
                      markerfacecolor="white" if hollow else c,
                      markeredgecolor=c if hollow else "white",
                      markeredgewidth=2.0 if hollow else 1.6,
                      markersize=12, label=lbl)
               for lbl, c, m, hollow in (STYLES[k] for k, _ in series)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figs", "sort_pathological_pareto.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
