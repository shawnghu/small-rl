"""Regenerate all sweep graphs for a sweep output directory.

Reads groups_meta.json, matches run directories to groups, and re-calls
the plotting functions with current colors/styles.

Usage:
    python regenerate_graphs.py output/classifier-recall-and-reward-structure
    python regenerate_graphs.py output/classifier-recall-and-reward-structure --workers 8
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path


def match_runs_to_group(sweep_dir, prefix, params):
    """Find routing and baseline run directories for a group.

    Returns (routing_runs, baseline_runs) where baseline_runs includes
    regular baselines, filter baselines, and reward_penalty baselines.
    """
    sweep_dir = Path(sweep_dir)
    ablated_frac = params.get("ablated_frac", "0.0")
    recall = params.get("rh_detector_recall", "1.0")
    routing_mode = params.get("routing_mode", "classic")

    # Routing runs: {prefix}_af{af}_rcl{rcl}_rm{mode}_s{seed}
    routing_pattern = f"{prefix}_af{ablated_frac}_rcl{recall}_rm{routing_mode}_s*"
    routing_runs = sorted(glob(str(sweep_dir / routing_pattern)))

    # Baselines share prefix and recall but not ablated_frac or routing_mode
    baseline_pattern = f"baseline_{prefix}_rcl{recall}_s*"
    filter_pattern = f"filter_{prefix}_rcl{recall}_s*"
    reward_penalty_pattern = f"reward_penalty_{prefix}_rcl{recall}_s*"

    baseline_runs = (
        sorted(glob(str(sweep_dir / baseline_pattern)))
        + sorted(glob(str(sweep_dir / filter_pattern)))
        + sorted(glob(str(sweep_dir / reward_penalty_pattern)))
    )

    return routing_runs, baseline_runs


def _render_one_group(sweep_dir, name, prefix, routing_runs, baseline_runs):
    """Render plots for a single group (runs in worker process)."""
    # Import inside worker to avoid matplotlib threading issues
    from sweep_plots import generate_group_comparison_plots

    generate_group_comparison_plots(
        routing_runs=routing_runs,
        baseline_runs=baseline_runs,
        reward=prefix,
        output_dir=sweep_dir,
        group_name=name,
    )
    return name


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate sweep graphs")
    parser.add_argument("sweep_dir", help="Sweep output directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    meta_path = os.path.join(sweep_dir, "sweep_graphs", "groups_meta.json")

    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found")
        sys.exit(1)

    with open(meta_path) as f:
        groups = json.load(f)

    print(f"Regenerating graphs for {len(groups)} groups in {sweep_dir}")

    # Prepare work items (match runs in main process, render in workers)
    work = []
    skipped = 0
    for group in groups:
        name = group["name"]
        prefix = group["prefix"]
        params = group["params"]
        routing_runs, baseline_runs = match_runs_to_group(sweep_dir, prefix, params)
        if not routing_runs:
            skipped += 1
            continue
        work.append((sweep_dir, name, prefix, routing_runs, baseline_runs))

    if skipped:
        print(f"Skipping {skipped} groups with no routing runs")

    n_workers = min(args.workers, len(work))
    print(f"Rendering {len(work)} groups with {n_workers} workers...")

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_render_one_group, *item): item[1] for item in work
        }
        for future in as_completed(futures):
            completed += 1
            name = futures[future]
            try:
                future.result()
                if completed % 20 == 0 or completed == len(work):
                    print(f"  [{completed}/{len(work)}] done")
            except Exception as e:
                print(f"  ERROR on {name}: {e}")

    # Regenerate overview and grid pages (fast, single-threaded)
    from sweep_plots import generate_sweep_overview, generate_sweep_grid

    print("\nRegenerating overview and grid pages...")
    generate_sweep_overview(sweep_dir)
    generate_sweep_grid(sweep_dir)
    print("Done!")


if __name__ == "__main__":
    main()
