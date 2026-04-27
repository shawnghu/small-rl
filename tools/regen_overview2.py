"""Regenerate a custom overview HTML at a non-default path.

Used to run an out-of-band overview that doesn't collide with the sweep
parent's periodic `overview.html` regen (which is pinned to the old
viz_playground module loaded at sweep start). This one re-imports
viz_playground each call so logic edits land immediately.

Usage: python tools/regen_overview2.py <sweep_dir> [output_filename=overview2.html]
"""
import importlib
import os
import sys
from pathlib import Path

# Allow imports from project root regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sweep_dir = Path(sys.argv[1])
out_name = sys.argv[2] if len(sys.argv) > 2 else "overview2.html"

import viz_playground
importlib.reload(viz_playground)

graphs_dir = sweep_dir / "sweep_graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)
output_path = str(graphs_dir / out_name)

runs = viz_playground.load_sweep(str(sweep_dir))
if not runs:
    print(f"[OVERVIEW2] No runs with routing_eval.jsonl in {sweep_dir}")
    sys.exit(0)

traces = viz_playground.build_traces(runs)
viz_playground.generate_by_group_html(
    runs, traces, str(sweep_dir), sweep_dir.name, output_path,
    page_title=f"Sweep Overview (live) — {sweep_dir.name}",
)
print(f"[OVERVIEW2] {output_path}")
