"""Quick test: object_qa with the matrix GR config + one_step_off, 30 steps, to
confirm the ref-path fix (was crashing in _ref_logps_liger_fused on the view)."""
import importlib.util, os
spec = importlib.util.spec_from_file_location("m", os.path.join(os.path.dirname(__file__), "osp_matrix_gr_5envs_2seed.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
runs = [r for r in m.runs if "object_qa" in r["run_name"] and r["seed"] == 2]
for r in runs:
    r["max_steps"] = 30
    r["run_name"] = r["run_name"] + "_reffix30"
per_gpu = 1
