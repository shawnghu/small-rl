"""Bootstrap groups_meta.json from in-flight run_config.yamls and regenerate
a grid HTML page for a sweep that hasn't completed any groups yet.

`sweep.py` writes groups_meta.json only when a whole experiment group
(routing + auto-baselines for one set of varied params) finishes, so during
a live sweep the file doesn't exist yet and `generate_sweep_grid` short-
circuits. This tool walks the sweep dir, runs viz_playground's
`assign_groups`, and writes a synthetic meta entry per *routing* group that
the grid generator can match against (its "name" field is the same
group-label string `assign_groups` emits, which is what `generate_sweep_grid`
indexes by).

Usage: python tools/regen_grid2.py <sweep_dir> [output_filename=grid2.html]
"""
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sweep_dir = Path(sys.argv[1])
out_name = sys.argv[2] if len(sys.argv) > 2 else "grid2.html"

import viz_playground
import sweep_plots
importlib.reload(viz_playground)
importlib.reload(sweep_plots)

graphs_dir = sweep_dir / "sweep_graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

runs = viz_playground.load_sweep(str(sweep_dir))
if not runs:
    print(f"[GRID2] No runs with routing_eval.jsonl in {sweep_dir}")
    sys.exit(0)

groups = viz_playground.assign_groups(runs, str(sweep_dir))

# Build a synthetic meta entry per group. The lookup in
# `generate_sweep_grid` (sweep_plots.py:705) uses the group-label string
# (output of `_group_label_from_params`) as the routing_key, then keys
# meta_by_key on the meta entry's "name" with seed-stripping. Setting
# meta["name"] = group_label makes the lookup hit.
meta_entries = []
for group_label, idxs in groups.items():
    routing_idxs = [i for i in idxs if not runs[i].get("is_baseline")]
    if not routing_idxs:
        continue
    rep = runs[routing_idxs[0]]
    params = rep.get("params") or {}
    cfg_path = params.get("config") or params.get("config_path") or ""
    prefix = os.path.basename(str(cfg_path)).replace(".yaml", "") if cfg_path else "run"
    # Slim the params dict to the keys that vary across routing runs (group keys).
    routing_param_dicts = [
        runs[i].get("params") or {} for i in range(len(runs))
        if not runs[i].get("is_baseline")
    ]
    group_keys = viz_playground._infer_group_keys(routing_param_dicts)
    slim = {}
    for k in group_keys:
        v = params.get(k)
        if v is None:
            continue
        slim[k] = v
    meta_entries.append({"name": group_label, "prefix": prefix, "params": slim})

meta_path = graphs_dir / "groups_meta_bootstrap.json"
with open(meta_path, "w") as f:
    json.dump(meta_entries, f, indent=2)
print(f"[GRID2] Wrote {len(meta_entries)} synthetic meta entries to {meta_path}")

# `generate_sweep_grid` reads from a fixed path (groups_meta.json). Rather
# than overwriting the official one (which the parent sweep would later
# replace), redirect by symlinking groups_meta_bootstrap.json → groups_meta.json
# only when the official one is missing. If the parent has written its own,
# we leave it alone.
official_meta = graphs_dir / "groups_meta.json"
if not official_meta.exists():
    # Atomic-ish: write content directly (no symlink, since the parent may
    # later overwrite). Safe because we only do this when it's absent.
    with open(official_meta, "w") as f:
        json.dump(meta_entries, f, indent=2)
    print(f"[GRID2] Bootstrapped {official_meta} (parent had not written it)")

# Generate grid.html via the existing function.
sweep_plots.generate_sweep_grid(str(sweep_dir))

# Rename grid.html → out_name to keep the parent's regen from clobbering it
# (and so the user gets a stable URL).
src = graphs_dir / "grid.html"
dst = graphs_dir / out_name
if src.exists() and src != dst:
    os.replace(str(src), str(dst))
    print(f"[GRID2] {dst}")
