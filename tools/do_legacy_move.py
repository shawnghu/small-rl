"""Move old-pipeline sweep configs (everything except the GRAFT-compatible set) into
legacy_configs/, preserving git history (git mv) and rewriting intra-legacy imports
(from sweeps.X -> from legacy_configs.X) only where X also moved. Idempotent-ish: skips
files already moved. Verifies all kept (compat) configs + a sample of moved ones still import.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SWEEPS = os.path.join(ROOT, "sweeps")
LEGACY = os.path.join(ROOT, "legacy_configs")
sys.path.insert(0, ROOT)

# The 18 GRAFT-compatible configs that STAY in sweeps/ (no old-pipeline flags).
COMPAT = {
    "baseline", "experimental_training_dynamics", "good_bigbatch_optimizer_dynamics",
    "good_training_dynamics", "gr", "graft_canonical_7envs", "graft_canonical_smoke",
    "leetcode_qwen3_4b_aware", "leetcode_qwen3_4b_aware_medium",
    "leetcode_qwen3_4b_aware_minibatch", "leetcode_qwen3_4b_baseline",
    "leetcode_qwen3_4b_matched_mlp_3xlr", "leetcode_qwen3_8b_matched_mlp_3xlr",
    "no_intervention_7envs", "optimizer_sweep", "qwen3_8b_incontexttests",
    "qwen3_8b_modifytests", "test_new_envs",
}


def main():
    stems = sorted(f[:-3] for f in os.listdir(SWEEPS)
                   if f.endswith(".py") and f != "__init__.py")
    moved = [s for s in stems if s not in COMPAT]
    moved_set = set(moved)
    print(f"{len(stems)} configs: keep {len(stems)-len(moved)} compat, move {len(moved)}")

    os.makedirs(LEGACY, exist_ok=True)
    init = os.path.join(LEGACY, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()

    git_moved, plain_moved = 0, 0
    for s in moved:
        src = os.path.join(SWEEPS, s + ".py")
        dst = os.path.join(LEGACY, s + ".py")
        if not os.path.exists(src):
            continue  # already moved
        r = subprocess.run(["git", "mv", src, dst], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            git_moved += 1
        else:
            os.rename(src, dst)
            plain_moved += 1
    print(f"moved: {git_moved} via git mv, {plain_moved} via rename")

    # Rewrite intra-legacy imports in the moved files.
    rewrites = 0
    pat = re.compile(r"\bfrom sweeps\.(\w+) import")
    pat2 = re.compile(r"\bimport sweeps\.(\w+)\b")
    for s in moved:
        p = os.path.join(LEGACY, s + ".py")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        new = pat.sub(lambda m: (f"from legacy_configs.{m.group(1)} import"
                                 if m.group(1) in moved_set else m.group(0)), txt)
        new = pat2.sub(lambda m: (f"import legacy_configs.{m.group(1)}"
                                  if m.group(1) in moved_set else m.group(0)), new)
        if new != txt:
            open(p, "w").write(new)
            rewrites += 1
    print(f"rewrote intra-legacy imports in {rewrites} files")

    # Verify: every kept compat config still imports, + a sample of moved ones.
    import importlib
    bad = []
    for s in sorted(COMPAT):
        try:
            importlib.import_module(f"sweeps.{s}")
        except Exception as e:
            bad.append(("sweeps." + s, repr(e)))
    sample = ["binary_persona_rp_1000", "retrain_gr_modal_6envs_excl_coh_1k",
              "graft_smoke", "rp_baseline_7envs", "matrix_gr_7envs"]
    for s in sample:
        if os.path.exists(os.path.join(LEGACY, s + ".py")):
            try:
                importlib.import_module(f"legacy_configs.{s}")
            except Exception as e:
                bad.append(("legacy_configs." + s, repr(e)))
    if bad:
        print("\n!!! IMPORT FAILURES:")
        for name, err in bad:
            print(f"  {name}: {err}")
        sys.exit(1)
    print(f"\nOK: all {len(COMPAT)} compat configs + {len(sample)} sampled legacy configs import cleanly")


if __name__ == "__main__":
    main()
