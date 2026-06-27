#!/usr/bin/env python
"""Render a SEPARATE `overview_split.html` for delayed-penalty hardcode sweeps that
adds **Hack Freq (monitored)** and **Hack Freq (unmonitored)** panels alongside the
GT total `hack_freq` panel.

Why a separate file: the live sweep loop rewrites `{sweep}/sweep_graphs/overview.html`
(and its data json) every ~60s, so anything written there is clobbered. This util
writes `overview_split.html` + `overview_split_data.json.gz` instead, and is safe to
re-run while training continues (idempotent: it only reads run artifacts + overwrites
its own two output files atomically via the shared generator).

--------------------------------------------------------------------------------
CHANNEL MAPPING (inferred per-run from run_config.yaml -- never hardcoded):
The "monitor" is the reward component with NEGATIVE scale. Its form classifier
determines which hardcode form is MONITORED:
  * mbpp_plus_hardcode_detected_bonus        (-) -> monitored = BARE      (normal configs)
  * mbpp_plus_hardcode_evade_detected_bonus  (-) -> monitored = DECORATED (inverted config)
The other form is the UNMONITORED channel.

--------------------------------------------------------------------------------
SPLIT METHOD -- APPROXIMATION (train-composition):
The eval artifacts do NOT contain enough to split the eval hack rate exactly:
  - routing_eval.jsonl stores only the GT TOTAL hardcode rate per step/mode
    ({mode}/hack_freq/mbpp_hardcode_freq); the combined/retain values are SUMS, not
    per-component, so the evade/detected breakdown is not recoverable from them.
  - eval_samples.jsonl stores completion TEXT for only ~8 of 75 completions per
    (step,mode) -- a display sample, too sparse to reclassify forms exactly.

So we split the (real) GT eval total by the TRAIN-measured decorated composition
    cdec(step) = mean(mbpp_plus_hardcode_evade_bonus) / mean(mbpp_plus_hardcode_bonus)
parsed per-step from train.log and summed over a window centred on each eval step.
(evade_bonus = decorated rate; hardcode_bonus = total rate. Both are logged raw and
WITHOUT the penalty delay, so they are reliable even before penalty_start_step --
unlike detected_bonus, which is gated to 0 during the delay.)

Then, exactly summing to GT:
    monitored   = GT * monitored_frac
    unmonitored = GT * (1 - monitored_frac)
where monitored_frac = (1 - cdec) if monitored form is BARE, else cdec.
The decorated FRACTION is borrowed from train; the eval TOTAL is the real eval number.

For an EXACT eval-side split on FUTURE runs, the delayed configs now carry
`extra_hack_freq_detectors` (mbpp_hardcode_bare_freq / mbpp_hardcode_decorated_freq)
which log {mode}/hack_freq_detectable/_undetectable natively into routing_eval.jsonl;
the standard overview.html then shows the monitored/unmonitored panels with no patch,
and this util becomes unnecessary (it only serves runs that predate that change).

--------------------------------------------------------------------------------
Usage:
    .venv/bin/python tools/patch_overview_split.py output/<sweep_dir> [--window N]
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viz_playground import load_sweep, build_traces, generate_by_group_html  # noqa: E402

# negative-scale monitor component name -> monitored hardcode FORM
_MONITOR_FORM = {
    "mbpp_plus_hardcode_detected_bonus": "bare",
    "mbpp_plus_hardcode_evade_detected_bonus": "decorated",
}
# train.log raw per-step rate keys used for the decorated composition
_TOTAL_KEY = "mbpp_plus_hardcode_bonus"
_DECOR_KEY = "mbpp_plus_hardcode_evade_bonus"


def infer_monitored_form(params: dict) -> str:
    """Return 'bare' or 'decorated' -- the form the monitor (negative-scale
    component) penalizes. Loud on any ambiguity."""
    comps = (params.get("reward") or {}).get("components") or []
    negatives = [c for c in comps if (c.get("scale") or 0.0) < 0.0]
    assert len(negatives) == 1, (
        f"expected exactly one negative-scale (monitor) component, got "
        f"{[c.get('name') for c in negatives]}")
    name = negatives[0]["name"]
    assert name in _MONITOR_FORM, (
        f"monitor component {name!r} is not a known hardcode-form penalty "
        f"({list(_MONITOR_FORM)})")
    return _MONITOR_FORM[name]


_STEP_RE = re.compile(r"@(\d+)\]")
_REWARD_RE = re.compile(r"reward\[([^\]]*)\]")


def parse_train_composition(train_log: Path):
    """-> {train_step: (total_rate, decorated_rate)} from the per-step reward[...]
    blocks in train.log. Missing/zero steps are simply absent."""
    out = {}
    if not train_log.exists():
        return out
    with open(train_log, errors="replace") as f:
        for line in f:
            if "reward[" not in line:
                continue
            ms = _STEP_RE.search(line)
            mr = _REWARD_RE.search(line)
            if not (ms and mr):
                continue
            step = int(ms.group(1))
            body = mr.group(1)
            kv = dict(re.findall(r"(\S+?)=([-\d.]+)", body))
            if _TOTAL_KEY in kv and _DECOR_KEY in kv:
                try:
                    out[step] = (float(kv[_TOTAL_KEY]), float(kv[_DECOR_KEY]))
                except ValueError:
                    pass
    return out


def _global_cdec(comp):
    tot = sum(t for t, _ in comp.values())
    dec = sum(d for _, d in comp.values())
    return (dec / tot) if tot > 0 else 0.0


def cdec_at(step: int, comp: dict, window: int, fallback: float) -> float:
    """Windowed decorated fraction around an eval step: sum(decorated)/sum(total)
    over train steps in [step-window, step+window]. Falls back to the global
    fraction when the window has no hardcodes (ratio is 0/0)."""
    tot = dec = 0.0
    for s in range(step - window, step + window + 1):
        if s in comp:
            t, d = comp[s]
            tot += t
            dec += d
    if tot <= 0.0:
        return fallback
    return dec / tot


def patch_sweep(sweep_dir: str, window: int = 10) -> str | None:
    sweep_dir = Path(sweep_dir)
    runs = load_sweep(str(sweep_dir))
    if not runs:
        print(f"[split] no runs with routing_eval.jsonl in {sweep_dir}")
        return None

    n_patched = 0
    for run in runs:
        params = run.get("params") or {}
        try:
            monitored_form = infer_monitored_form(params)
        except AssertionError as e:
            print(f"[split] SKIP {run['name']}: {e}")
            continue
        comp = parse_train_composition(sweep_dir / run["name"] / "train.log")
        gcdec = _global_cdec(comp)
        for step, modes in run["timeseries"].items():
            cdec = cdec_at(step, comp, window, gcdec)
            mon_frac = (1.0 - cdec) if monitored_form == "bare" else cdec
            for mode, d in modes.items():
                gt = d.get("hack_freq")
                if gt is None:
                    continue
                d["hack_freq_detectable"] = gt * mon_frac
                d["hack_freq_undetectable"] = gt * (1.0 - mon_frac)
        n_patched += 1
        print(f"[split] {run['name']}: monitored={monitored_form} "
              f"global_cdec={gcdec:.3f}")

    if n_patched == 0:
        print("[split] nothing patched (no recognizable monitor configs)")
        return None

    traces = build_traces(runs)
    graphs_dir = sweep_dir / "sweep_graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(graphs_dir / "overview_split.html")
    generate_by_group_html(
        runs, traces, str(sweep_dir), sweep_dir.name, output_path,
        page_title=f"Sweep Overview (monitored/unmonitored split) — {sweep_dir.name}",
    )
    print(f"[split] wrote {output_path}")
    print(f"[split] wrote {os.path.splitext(output_path)[0]}_data.json.gz")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", help="path to output/<sweep> (or a single run dir's parent)")
    ap.add_argument("--window", type=int, default=10,
                    help="+/- train steps around each eval step for the cdec ratio (default 10)")
    args = ap.parse_args()
    patch_sweep(args.sweep_dir, window=args.window)


if __name__ == "__main__":
    main()
