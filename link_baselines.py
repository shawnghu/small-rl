"""Link the canonical no-intervention baselines into the anti-train sweep dirs
and regenerate the overview/grid HTML, so each sweep's overview.html shows the
routing_mode=none baseline line per env.

The anti-train sweeps run the canonical GR cell (hack_frac=0.5, recall=1.0),
which matrix_gr_7envs skipped as "covered elsewhere". The matching baselines are
output/no_intervention_7envs/* (routing_mode=none, hf=0.5, rcl=1.0, 7 envs x 3
seeds) in the MAIN repo. We symlink each into every anti-train sweep dir under a
`baseline_` prefix so viz_playground.classify_run tags it is_baseline=True and
assign_groups merges it into the matching per-env group (matched on config_path).

Idempotent: re-running replaces existing baseline_* symlinks.
"""
import os
import sys
from pathlib import Path

BASELINE_SRC = Path("/workspace/small-rl/output/no_intervention_7envs")
SWEEP_DIRS = [
    Path("/workspace/small-rl-antitrain/output/matrix_gr_7envs_antitrain_wm01"),
    Path("/workspace/small-rl-antitrain/output/matrix_gr_7envs_antitrain_wm05"),
    Path("/workspace/small-rl-antitrain/output/matrix_gr_7envs_antitrain_wm10"),
    Path("/workspace/small-rl-antitrain/output/matrix_gr_7envs_antitrain_wm10_rb256coh256"),
]

sys.path.insert(0, "/workspace/small-rl-antitrain")


def link_into(sweep_dir: Path) -> int:
    n = 0
    baselines = [d for d in sorted(BASELINE_SRC.iterdir())
                 if d.is_dir() and d.name != "sweep_graphs"
                 and (d / "routing_eval.jsonl").exists()]
    for b in baselines:
        link = sweep_dir / f"baseline_{b.name}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(b.resolve())
        n += 1
    return n


def main():
    from sweep_plots import generate_sweep_overview, generate_sweep_grid
    for sd in SWEEP_DIRS:
        if not sd.exists():
            print(f"[skip] {sd} does not exist")
            continue
        n = link_into(sd)
        print(f"[link] {n} baselines -> {sd.name}")
        generate_sweep_overview(str(sd))
        try:
            generate_sweep_grid(str(sd))
        except Exception as e:
            print(f"[grid] {sd.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
