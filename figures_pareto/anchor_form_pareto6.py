"""Appendix figure: anchor-distribution method suite on the six CANONICAL
respmon envs — gr-paper, subsection "Small-scale environments with a form of
anchor distribution" (app:anchor-form).

The main-body GRAFT draws its anchor slice from the training batch with a
reward penalty on flagged anchor samples. This appendix presents the
alternative studied first: a genuine anchor (verifiable/unhackable) prompt
distribution, used three ways —
  - mixed into RP training at main:anchor 4:1 and 1:1 (RP color, spoke
    markers, matching the appendix ratio-figure convention);
  - alone (0:1, "train on anchor distribution only", no penalty — trains at
    hack_frac 0 and evals on the standard hf .5 mixture via --eval_hack_frac);
  - as GRAFT's coherence source (coh_prompt_source=unhackable,
    coherence_rh_mode=none — the vercoh128 recipe), deploy (retain_only) mode.
Context arms: do-nothing, plain RP 2.0, base model (first eval row).

Sort is the REDESIGNED (framed) env, on which only DN / plain RP / 0:1 were
run — the mix and anchor-GRAFT arms predate the redesign and are omitted from
that panel.

Conventions match the paper's main figure (proto_respmon6_rv2canon_sidebyside):
y = retain component on ALL prompts, x = hack rate on HACKABLE prompts, tail
mean (last 10% of eval rows), SEM error bars over seeds (n=3; anchor-GRAFT
n=8).

Run:  cd figures_pareto && /workspace/small-rl/.venv/bin/python anchor_form_pareto6.py
Then copy figs/anchor_form_pareto6.pdf to gr-paper/figures/.
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

_COMPOSITE = "respmon_canon_vercoh128_composite"
_MIX = "respmon_canon_rp_mix_3seed"
_UNHACK = "respmon_canon_unhackonly_3seed"
_SORTFRAMED = "respmon_sortframed_dn_rp_unhack_3seed"

# env key -> (panel title, run prefix). Sort = the framed redesign.
ENVS = [
    ("addition_v2",  "addition",  "addition_v2_syco_respmon_iva"),
    ("cities_qa",    "cities_qa", "cities_qa_syco_respmon_iva"),
    ("object_qa",    "object_qa", "object_qa_syco_respmon_iva_inv"),
    ("persona_qa",   "persona_qa","persona_qa_flattery_respmon_bvo"),
    ("repeat",       "repeat",    "repeat_respmon"),
    ("sorting_copy", "sort",      "sorting_copy_framed"),
]

# key -> (label, color, marker, hollow). Colors from the paper's canonical
# STYLES; the two mixes are RP-color spoke variants (appendix convention);
# the 0:1 arm reuses the verified-only series color.
SERIES = [
    ("noi",     "No intervention",                          "#e0905a", "X", False, "both"),
    ("rp",      "Reward Penalty (no anchor samples)",       "#d62728", "s", False, "both"),
    ("rp41",    "Reward Penalty, routing:anchor 4:1",       "#d62728", "p", False, "both"),
    ("rp11",    "Reward Penalty, routing:anchor 1:1",       "#d62728", "P", False, "both"),
    ("anchor",  "Train on anchor distribution only",        "#b08490", "^", False, "both"),
    ("gr",      "GRAFT, routing:anchor 4:1",                "#2ca02c", "o", False, "retain_only"),
    ("gr16",    "GRAFT, routing:anchor 16:1",               "#2ca02c", "v", False, "retain_only"),
    ("base",    "Base model",                               "#444444", "o", True,  "both"),
]
_STYLE = {k: (l, c, m, h) for k, l, c, m, h, _ in SERIES}
_MODE = {k: md for k, *_r, md in [(s[0], s[5]) for s in SERIES]}


def _run_dirs(sweep, prefix, pat, n_expect):
    """pat may be a str or a tuple of alternatives (object's GR seeds 1-3 are
    tagged invgr_cls, seeds 4-8 gr_cls)."""
    pats = (pat,) if isinstance(pat, str) else pat
    root = os.path.join(OUT_ROOT, sweep)
    dirs = sorted(d for d in os.listdir(root)
                  if any(d.startswith(f"{prefix}_{p}") for p in pats))
    assert len(dirs) == n_expect, f"{sweep}/{prefix}_{pats}*: found {dirs}"
    return [os.path.join(root, d) for d in dirs]


def _rows(path):
    with open(os.path.join(path, "routing_eval.jsonl")) as f:
        raw = [json.loads(l) for l in f if l.strip()]
    assert raw, f"empty routing_eval in {path}"
    by_step = {r["step"]: r for r in raw}   # keep-last per step
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


def env_series(env_key, prefix):
    if env_key == "sorting_copy":
        dn = _run_dirs(_SORTFRAMED, prefix, "donothing", 3)
        rp = _run_dirs(_SORTFRAMED, prefix, "rp2_hf050", 3)
        anchor = _run_dirs(_SORTFRAMED, prefix, "unhackonly", 3)
        gr = _run_dirs("respmon_sortframed_anchor_gr_3seed", prefix,
                       "gr_cls_coh128unhack", 3)
        gr16 = _run_dirs("respmon_sortframed_anchor_gr_3seed", prefix,
                         "gr_cls_coh32unhack", 3)
        paths = {"noi": dn, "rp": rp, "anchor": anchor,
                 "rp41": None, "rp11": None, "gr": gr, "gr16": gr16,
                 "base": dn + rp + gr}
    else:
        dn = _run_dirs(_COMPOSITE, prefix, "donothing", 3)
        rp = _run_dirs(_COMPOSITE, prefix, "rp2_hf050", 3)
        gr = _run_dirs(_COMPOSITE, prefix,
                       ("gr_cls_coh128unhack", "invgr_cls_coh128unhack"), 8)
        paths = {
            "noi": dn, "rp": rp,
            "rp41": _run_dirs(_MIX, prefix, "rp2_mix4to1", 3),
            "rp11": _run_dirs(_MIX, prefix, "rp2_mix1to1", 3),
            "anchor": _run_dirs(_UNHACK, prefix, "unhackonly", 3),
            "gr": gr,
            "gr16": _run_dirs("respmon_rv2_vercoh32_gr_3seed", prefix,
                              "gr_cls_coh32unhack", 3),
            "base": dn + rp + gr,
        }
    out = []
    for key, *_rest in SERIES:
        p = paths[key]
        if p is None:
            out.append((key, None))
            continue
        out.append((key, _agg(_points(p, _MODE[key], first_row=key == "base"))))
    return out


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.6))
    print(f'{"env":<13} {"series":<7} {"retain":>7} {"hack":>7} {"n":>2}')
    for ax, (env_key, title, prefix) in zip(axes.flat, ENVS):
        for i, (key, agg) in enumerate(env_series(env_key, prefix)):
            if agg is None:
                print(f"{env_key:<13} {key:<7}   (not run on this env)")
                continue
            r_m, r_s, h_m, h_s, n = agg
            print(f"{env_key:<13} {key:<7} {r_m:>7.3f} {h_m:>7.3f} {n:>2}")
            _lbl, color, marker, hollow = _STYLE[key]
            ax.errorbar([h_m], [r_m], xerr=[h_s], yerr=[r_s],
                        fmt=marker, color=color, markersize=12,
                        markerfacecolor="white" if hollow else color,
                        markeredgecolor=color if hollow else "white",
                        markeredgewidth=2.0 if hollow else 1.4,
                        ecolor=color, elinewidth=1.2, capsize=4, capthick=1.2,
                        zorder=50 if key == "gr" else 8 + i,
                        clip_on=key != "gr")
        ax.set_title(title, fontsize=17)
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
    for ax in axes[:, 0]:
        ax.set_ylabel("Target-task (retain) reward", fontsize=14)
    for ax in axes[-1]:
        ax.set_xlabel("Reward hack rate (hackable prompts)", fontsize=14)
    handles = [Line2D([], [], marker=m, linestyle="none", color=c,
                      markerfacecolor="white" if h else c,
                      markeredgecolor=c if h else "white",
                      markeredgewidth=2.0 if h else 1.4,
                      markersize=12, label=l)
               for _k, l, c, m, h, _md in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=12.5)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figs", "anchor_form_pareto6.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
