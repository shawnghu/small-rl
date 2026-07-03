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
# = % of all completions. Measured 2026-07-02, /output/countdown_sft_model/qwen3-8b.
BASE_ROWS = {
    "humaneval": {"label": "SFT base (pre-RL)", "retain": 60.5, "hardcode": 1.0, "tamper": 0.6},
    "leetcode":  {"label": "SFT base (pre-RL)", "retain": 28.7, "hardcode": 9.9, "tamper": 2.7},
}

_GR_0702 = "/output/countdown_code_gr-0702-0134/countdown_code_gr_cls_coh256_pen2_noretain_balanced_splitmoment_lam1_s{s}"
_RP2_0702 = "/output/countdown_code_rp2-0702-0026/reward_penalty_countdown_code_hack_reward_penalty_amount2.0_s{s}"

SPECS = {
    # The 2026-07-02 countdown_code GR + RP@2 sweeps. GR s15 has no saved
    # checkpoint (4h Modal timeout) and is omitted; add it back once retrained.
    "countdown_0702": (
        [{"run": _GR_0702.format(s=s), "kind": "gr", "label": f"gr_s{s}", "group": "GR"}
         for s in (9, 16)]
        + [{"run": _RP2_0702.format(s=s), "kind": "rp", "label": f"rp2_s{s}", "group": "RP2"}
           for s in (9, 15, 16)]
    ),
}


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
