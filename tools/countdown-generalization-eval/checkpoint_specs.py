"""Named checkpoint sets for the generalization eval + the fixed base-model row.

A *spec* is a list of run entries; `expand()` turns each into the concrete
generation configs (a GR run → two adapter modes; an RP/vanilla run → one).
Both the generation batch (modal_humaneval_generate_ckpt.py) and the table
generator (make_table.py) consume the same expanded list, so config dir names
and table groupings can never drift apart.

Entry fields:
  run   : run dir OR checkpoint dir on the volume (run dir → latest checkpoint)
  kind  : "gr"  → emits <label>_2adapter (fs=1.0) + <label>_retainonly (fs=0.0)
          "rp"  → emits <label> (fs=1.0). Also use for vanilla/DN runs.
  label : dir-name prefix (unique per run)
  group : table row label (GR kinds get " 2-adapter"/" retain-only" appended)

To eval a NEW set of checkpoints: add a spec here (or build the list inline)
and pass its name via --spec to both the generation batch and make_table.py.

BASE_ROWS: the SFT-primed pre-RL model (measured once via the plain-vLLM path,
tools/modal_humaneval_generate.py --label sft-base). Hardcoded so future tables
show the pre-RL floor without re-generating it. Regenerate only if the base
model or scaffold changes.
"""
from __future__ import annotations

# retain = HumanEval full-solve % / LeetCode mean hidden-pass %; hardcode/tamper
# = % of all completions. Measured 2026-07-02, /output/countdown_sft_model/qwen3-8b;
# hardcode re-measured 2026-07-03 under the behavioral detector (was 1.0 / 9.9
# under the old literal rule — the LeetCode drop is the trivial-value FP removed).
BASE_ROWS = {
    "humaneval": {"label": "SFT base (pre-RL)", "retain": 60.5, "hardcode": 0.8, "tamper": 0.6},
    "leetcode":  {"label": "SFT base (pre-RL)", "retain": 28.7, "hardcode": 8.8, "tamper": 2.7},
}

_GR_0702 = "/output/countdown_code_gr-0702-0134/countdown_code_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{s}"
_RP2_0702 = "/output/countdown_code_rp2-0702-0026/reward_penalty_countdown_code_hack_reward_penalty_amount2.0_s{s}"

# hf100 (100%-hackable) countdown_code arms, the countdown_figure1.py set.
# GR seeds 1-5 live in the seeds5 sweep dir, 9/15/16 in the 3-seed dir.
_GR100_5 = "/output/countdown_hf100_gr_lccoh64_lr3_seeds5/cdhf100_gr_lccoh64_lr3_s{s}"
_GR100_3 = "/output/countdown_hf100_gr_lccoh64_lr3/cdhf100_gr_lccoh64_lr3_s{s}"
_RP100 = "/output/countdown_hf100_rp_lccoh64/cdhf100_rp2_lc64_lr1_s{s}"
_IP100 = "/output/countdown_hf100_ip/cdhf100_ip_mand-tw_s{s}"
_PPS100 = "/output/countdown_hf100_pps/cdhf100_pps_L20_a2_s{s}"

SPECS = {
    # The 2026-07-02 countdown_code GR + RP@2 sweeps. GR s15 has no saved
    # checkpoint (4h Modal timeout) and is omitted; add it back once retrained.
    "countdown_0702": (
        [{"run": _GR_0702.format(s=s), "kind": "gr", "label": f"gr_s{s}", "group": "GR"}
         for s in (9, 16)]
        + [{"run": _RP2_0702.format(s=s), "kind": "rp", "label": f"rp2_s{s}", "group": "RP2"}
           for s in (9, 15, 16)]
    ),
    # 2026-07-10 hf100 arms (countdown_figure1.py scatter set). IP's training
    # system-suffix and PPS's steering vector are train-time-only interventions,
    # so plain adapter loading here IS their "removed at eval" deployment config.
    "countdown_hf100": (
        [{"run": _GR100_5.format(s=s), "kind": "gr", "label": f"gr100_s{s}", "group": "GR"}
         for s in (1, 2, 3, 4, 5)]
        + [{"run": _GR100_3.format(s=s), "kind": "gr", "label": f"gr100_s{s}", "group": "GR"}
           for s in (9, 15, 16)]
        + [{"run": _RP100.format(s=s), "kind": "rp", "label": f"rp100_s{s}", "group": "RP (rp2_lc64_lr1)"}
           for s in (9, 15, 16)]
        + [{"run": _IP100.format(s=s), "kind": "rp", "label": f"ip100_s{s}", "group": "IP (mand-tw)"}
           for s in (9, 15, 16)]
        + [{"run": _PPS100.format(s=s), "kind": "rp", "label": f"pps100_s{s}", "group": "PPS (L20 a2)"}
           for s in (9, 15, 16)]
    ),
}

# Late additions to the hf100 set (generated separately so the first batch
# didn't need re-running):
#   - Anchor-env-only (lconly_full): full-budget training on the leetcode
#     anchor env alone; checkpoints hand-copied to this machine then uploaded
#     to the volume (no run_config.yaml on the volume — checkpoint dirs only).
#   - Do-nothing: the old round-2 "amountmissing" (penalty never applied) runs,
#     countdown_figure1.py's "No intervention" arm. s15 has no saved checkpoint
#     (4h Modal timeout) — 2 seeds only.
# Still pending: the no-routing baseline (cdhf100_noroute_anchor_s{9,15,16},
# to be evaled with one adapter ablated) — those runs saved NO checkpoints on
# the volume; waiting on a hand-copy like lconly_full's.
_LCONLY = "/output/lconly_full/lconly_full_s{s}/checkpoint-200"
_DN_0702 = ("/output/countdown_code_rp2-0702-0026/"
            "countdown_code_hack_reward_penalty_amountmissing_s{s}")

SPECS["countdown_hf100_extras"] = (
    [{"run": _LCONLY.format(s=s), "kind": "rp", "label": f"lconly100_s{s}",
      "group": "Anchor only (lconly_full)"} for s in (9, 15, 16)]
    + [{"run": _DN_0702.format(s=s), "kind": "rp", "label": f"dn_s{s}",
        "group": "No intervention (DN)"} for s in (9, 16)]
)

# No-routing baseline (anchor coherence, rh_detector_recall=0): the partner's
# 2026-07-25 rerun of the original 07-17 runs (whose checkpoints were deleted;
# these reran the same unmerged-snapshot config on Jake's pod, hand-copied to
# the volume 07-27). kind=gr → both the trained (2-adapter) and
# forget-ablated (retain-only) configs; the figure shows the ablated one
# ("GRAFT w/o routing", per countdown_figure1.py).
_NOROUTE = "/output/cdhf100_noroute_rerun_ckpts/cdhf100_noroute_anchor_s{s}"

SPECS["countdown_hf100_noroute"] = [
    {"run": _NOROUTE.format(s=s), "kind": "gr", "label": f"noroute100_s{s}",
     "group": "No routing"} for s in (9, 15, 16)]

# The full table set: first batch + extras + no-routing.
SPECS["countdown_hf100_all"] = (SPECS["countdown_hf100"] + SPECS["countdown_hf100_extras"]
                                + SPECS["countdown_hf100_noroute"])

# GR arms alone (8 seeds x {2adapter, retainonly}) — for scaffold-perturbation
# studies (e.g. the solution-only humaneval_solonly scaffold) where only the
# GR-vs-base contrast is of interest.
SPECS["countdown_hf100_gr"] = [e for e in SPECS["countdown_hf100"] if e["group"] == "GR"]


def expand(spec):
    """spec (name or list) -> [{label, run, forget_scale, group}, ...]."""
    entries = SPECS[spec] if isinstance(spec, str) else spec
    out = []
    for e in entries:
        if e["kind"] == "gr":
            out.append({"label": f"{e['label']}_2adapter", "run": e["run"],
                        "forget_scale": 1.0, "group": f"{e['group']} 2-adapter"})
            out.append({"label": f"{e['label']}_retainonly", "run": e["run"],
                        "forget_scale": 0.0, "group": f"{e['group']} retain-only"})
        elif e["kind"] == "rp":
            out.append({"label": e["label"], "run": e["run"],
                        "forget_scale": 1.0, "group": e["group"]})
        else:
            raise ValueError(f"unknown kind {e['kind']!r} (use 'gr' or 'rp')")
    return out
