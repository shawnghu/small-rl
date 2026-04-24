"""Build colored W&B saved views for sweeps.

Semantics this module relies on:
- Each sweep has a unique `sweep_name` (= `Path(output_dir).name`, after
  timestamp disambiguation in sweep.py). Every run launched by that sweep
  sets `wandb.init(group=sweep_name)` and picks up a deterministic run id
  derived from (sweep_name, run_name).
- After sweep.py finishes planning runs (but before/while they launch),
  this module creates a single saved view filtered to `group == sweep_name`
  with per-run colors chosen by a small heuristic (see `assign_colors`).

Group vs tag convention (keep consistent):
- `group`: a run belongs to exactly one group, always `sweep_name`. One sweep
  → one group → one auto-generated saved view.
- `tags`: free-form, multi-valued. Used for ad-hoc post-hoc grouping, e.g.
  "pull these particular runs into a separate view" or "mark runs that
  were rerun after config bug fix X". Saved views can filter on tags to
  combine runs across groups non-destructively. We do not set tags
  automatically; they are the user's extension point.

The view is created with `save_as_new_view()`; re-running the same sweep
name creates a new view (old ones aren't cleaned up). In practice, each
sweep run produces a freshly-disambiguated name, so collisions are rare.
"""

from __future__ import annotations

import colorsys
import hashlib
from typing import Any

_PALETTE_DISCRETE = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
]

_PALETTE_CLASS_FALLBACK = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def deterministic_run_id(sweep_name: str, run_name: str) -> str:
    """Stable 16-char wandb run id derived from (sweep, run)."""
    h = hashlib.sha1(f"{sweep_name}/{run_name}".encode()).hexdigest()
    return h[:16]


def _class_key(params: dict, ignore_keys=("seed",)) -> tuple:
    """Hashable key grouping params that differ only by seed (or other ignored keys)."""
    ignore = set(ignore_keys)
    items = []
    for k, v in params.items():
        if k in ignore:
            continue
        # Make unhashable values (lists/dicts) hashable via repr.
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            items.append((k, repr(v)))
    return tuple(sorted(items))


def _hsl_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def assign_colors(
    runs: list[dict[str, Any]],
    ignore_keys=("seed",),
) -> dict[str, str]:
    """Return {class_key_repr: hex_color} for coloring a sweep.

    Heuristic:
    - Runs that differ only by `ignore_keys` (default: seed) share a color.
    - If exactly one non-ignored param varies:
      - numeric → rank-based light→dark ramp of a single hue
      - discrete → distinct primary colors from `_PALETTE_DISCRETE`
    - Otherwise: cycle `_PALETTE_CLASS_FALLBACK` per equivalence class.
    """
    if not runs:
        return {}

    classes: dict[tuple, list[dict]] = {}
    for p in runs:
        classes.setdefault(_class_key(p, ignore_keys), []).append(p)

    # Keys that vary across equivalence classes.
    class_reps = [dict(k) for k in classes.keys()]
    all_keys = set().union(*(d.keys() for d in class_reps))
    varying = [k for k in all_keys
               if len({d.get(k) for d in class_reps}) > 1]

    color_by_class: dict[tuple, str] = {}

    if len(varying) == 1:
        k = varying[0]
        values = [d.get(k) for d in class_reps]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            # Numeric: light-to-dark ramp of red. Use rank (handles log-scale).
            sorted_vals = sorted(set(values))
            n = len(sorted_vals)
            for ck, rep in zip(classes.keys(), class_reps):
                r = sorted_vals.index(rep[k])
                t = r / max(n - 1, 1)
                # lightness 0.80 (light) -> 0.25 (dark)
                color_by_class[ck] = _hsl_hex(h=0.0, s=0.70, l=0.80 - 0.55 * t)
        else:
            # Discrete: primary colors keyed by value.
            sorted_unique = sorted({str(v) for v in values})
            v_to_color = {v: _PALETTE_DISCRETE[i % len(_PALETTE_DISCRETE)]
                          for i, v in enumerate(sorted_unique)}
            for ck, rep in zip(classes.keys(), class_reps):
                color_by_class[ck] = v_to_color[str(rep[k])]
    else:
        # Fallback: distinct color per equivalence class.
        for i, ck in enumerate(classes.keys()):
            color_by_class[ck] = _PALETTE_CLASS_FALLBACK[i % len(_PALETTE_CLASS_FALLBACK)]

    # Expand to per-run-id mapping is the caller's job; we return class→color
    # plus the classification so callers can resolve per-run colors.
    return {repr(ck): color for ck, color in color_by_class.items()}


def color_map_for_runs(
    runs_with_ids: list[tuple[str, dict[str, Any]]],
    ignore_keys=("seed",),
) -> dict[str, str]:
    """Return {run_id: hex_color}."""
    class_colors = assign_colors([p for _, p in runs_with_ids], ignore_keys)
    out = {}
    for rid, params in runs_with_ids:
        ck = repr(_class_key(params, ignore_keys))
        if ck in class_colors:
            out[rid] = class_colors[ck]
    return out


def build_sweep_view(
    entity: str | None,
    project: str,
    sweep_name: str,
    color_map: dict[str, str],
) -> str | None:
    """Create a saved view filtered to group==sweep_name with per-run colors.

    Returns the view URL, or None on failure (logged, never raised).
    """
    try:
        import wandb_workspaces.workspaces as ws
        from wandb_workspaces import expr
    except ImportError as e:
        print(f"[SWEEP_VIEW] wandb-workspaces not installed ({e}); skipping view creation.")
        return None

    try:
        if entity is None:
            import wandb
            entity = wandb.Api().default_entity
            if entity is None:
                print("[SWEEP_VIEW] No default wandb entity; skipping view creation.")
                return None

        run_settings = {rid: ws.RunSettings(color=color) for rid, color in color_map.items()}
        workspace = ws.Workspace(
            entity=entity,
            project=project,
            name=sweep_name,
            runset_settings=ws.RunsetSettings(
                filters=[expr.Metric("group") == sweep_name],
                run_settings=run_settings,
            ),
            auto_generate_panels=True,
        )
        saved = workspace.save_as_new_view()
        url = getattr(saved, "url", None)
        print(f"[SWEEP_VIEW] Saved view '{sweep_name}' ({len(color_map)} colored runs): {url}")
        return url
    except Exception as e:
        print(f"[SWEEP_VIEW] Failed to create saved view: {e}")
        return None
