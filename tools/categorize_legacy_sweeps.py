"""Categorize sweeps/*.py into GRAFT-compatible vs legacy (old-pipeline) — value-aware.

A sweep config is LEGACY iff any of its runs sets an old-pipeline flag to a behavior-changing
value (flags GRAFT ignores/repurposes). Pure key-presence is too coarse: the canonical no-int
base sets reward_penalty_baseline=False / rh_detector_verifies_retain_samples=False (inert) and
must stay compatible. Also maps the sweeps->sweeps import graph so the move won't break a kept
config that imports a moved one.
"""
import ast
import importlib
import os
import sys
import traceback

SWEEPS_DIR = os.path.join(os.path.dirname(__file__), "..", "sweeps")
SWEEPS_DIR = os.path.abspath(SWEEPS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(SWEEPS_DIR, "..")))

# Flags with NO meaning in GRAFT — presence at ANY value => relies on old pipeline.
REMOVED_ANY = {
    "retain_mode", "retain_penalty", "coherence_rh_mode", "coh_fixed_advantage",
    "retain_warmup_steps", "forget_warmup_steps", "rp_extra_retain_advantage_multiplier",
    "forget_scale_modulation", "ema_clamp", "routed_adam", "fused_reduction",
    "rh_eligible_frac", "ablated_frac", "train_forget_scale",
}
# Flags that exist but are default-inert (False); only an ACTIVE (True) value => old pipeline.
ACTIVE_TRUE = {
    "reward_penalty_baseline", "filter_baseline",
    "rh_detector_verifies_retain_samples", "verified_only_training",
}


def sweep_imports(path):
    """sweeps.X modules this file imports (intra-package edges)."""
    edges = set()
    try:
        tree = ast.parse(open(path).read())
    except Exception:
        return edges
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sweeps."):
            edges.add(node.module.split(".", 1)[1])
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("sweeps."):
                    edges.add(n.name.split(".", 1)[1])
    return edges


ROUTING = {"classic", "exclusive"}


def classify(modname):
    """-> (verdict, reasons, n_runs). verdict in:
       routing_legacy  - a ROUTING run (classic/exclusive) sets a now-ignored old flag => GRAFT
                         silently ignores it. Unambiguously incompatible with the new pipeline.
       baseline_old    - no routing run, but uses an old baseline flag (RP/filter/verified) or a
                         removed flag on a none run. Still runs (baseline path live); "old" but
                         not broken. Includes active non-routing work.
       compat          - sets no old flag at all.
       error           - import / no runs.
    """
    try:
        mod = importlib.import_module(f"sweeps.{modname}")
    except Exception as e:
        return "error", [f"import: {type(e).__name__}: {e}"], 0
    runs = getattr(mod, "runs", None)
    if not isinstance(runs, list):
        return "error", ["no `runs` list"], 0
    routing_reasons, base_reasons = set(), set()
    for r in runs:
        if not isinstance(r, dict):
            continue
        is_routing = r.get("routing_mode") in ROUTING
        hits = set()
        for k in REMOVED_ANY:
            if k in r:
                hits.add(f"{k}={r[k]!r}")
        for k in ACTIVE_TRUE:
            if r.get(k) is True:
                hits.add(f"{k}=True")
        (routing_reasons if is_routing else base_reasons).update(hits)
    if routing_reasons:
        return "routing_legacy", sorted(routing_reasons), len(runs)
    if base_reasons:
        return "baseline_old", sorted(base_reasons), len(runs)
    return "compat", [], len(runs)


def main():
    mods = sorted(f[:-3] for f in os.listdir(SWEEPS_DIR)
                  if f.endswith(".py") and f != "__init__.py")
    results = {}
    edges = {}
    for m in mods:
        edges[m] = sweep_imports(os.path.join(SWEEPS_DIR, m + ".py"))
        results[m] = classify(m)

    legacy = {m for m, (v, _, _) in results.items() if v == "legacy"}
    compat = {m for m, (v, _, _) in results.items() if v == "compat"}
    errs = {m for m, (v, _, _) in results.items() if v == "error"}

    # Conflicts: a kept (compat) config importing a config we'd move (legacy/error).
    move = legacy | errs
    conflicts = {m: (edges[m] & move) for m in compat if edges[m] & move}
    # Reverse: a moved config importing a kept one (import would need rewrite to stay valid).
    moved_import_kept = {m: (edges[m] & compat) for m in move if edges[m] & compat}

    print(f"=== {len(mods)} sweep configs: {len(compat)} compat, {len(legacy)} legacy, {len(errs)} error ===\n")
    print("--- LEGACY (move to legacy_configs/) ---")
    for m in sorted(legacy):
        print(f"  {m}  [{results[m][2]} runs]  <- {', '.join(results[m][1][:4])}")
    print("\n--- ERROR (import failed — inspect manually) ---")
    for m in sorted(errs):
        print(f"  {m}  <- {results[m][1][0]}")
    print("\n--- COMPAT (stay in sweeps/) ---")
    for m in sorted(compat):
        print(f"  {m}  [{results[m][2]} runs]")
    print("\n--- CONFLICTS: compat config imports a to-be-moved config ---")
    print("  (none)" if not conflicts else "")
    for m, deps in sorted(conflicts.items()):
        print(f"  {m} (compat) imports {sorted(deps)}")
    print("\n--- moved config imports a kept config (rewrite import on move) ---")
    print("  (none)" if not moved_import_kept else "")
    for m, deps in sorted(moved_import_kept.items()):
        print(f"  {m} imports {sorted(deps)}")


if __name__ == "__main__":
    main()
