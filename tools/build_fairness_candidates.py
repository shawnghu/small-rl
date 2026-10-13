"""Assemble MBPP+ unfairness candidates with concrete counterexamples.
Parallelized across problems (each problem's worker calls run on its own thread;
the 40-worker pool is shared). Candidate set = problems the strongest model
(Qwen3-32B) does not reliably solve (<5/5)."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from persistent_code_eval import PersistentCodeEvaluator
from tools.mbppplus_fairness import failing_cases, shadows_builtin

ds = {r["task_id"]: r for r in load_dataset("evalplus/mbppplus", split="test")}
def load(s): return {json.loads(l)["task_id"]: json.loads(l)
                     for l in open(f"output/mbppplus_capability/{s}.jsonl")}
D = {m: load(p) for m, p in [("32", "Qwen_Qwen3-32B"),
                             ("4", "Qwen_Qwen3-4B"), ("17", "Qwen_Qwen3-1.7B")]}
cand_ids = [t for t in ds if D["32"][t]["n_pass"] < D["32"][t]["n"]]
print(f"candidate (32B not 5/5): {len(cand_ids)}", flush=True)

ev = PersistentCodeEvaluator(num_workers=40, timeout=10)


def vis_pass(prog, vt):
    if not prog:
        return False
    code = ("import json\n_src=" + repr(prog + "\n" + vt) +
            "\ntry:\n exec(_src,{})\n _ok=True\nexcept Exception:\n _ok=False\n"
            "print(json.dumps({'p':_ok}))\n")
    w = ev._pool.get()
    try:
        raw = w.execute(code, timeout=5, memory_limit=2048)
    finally:
        ev._pool.put(w)
    so = raw.get("stdout", {})
    return bool(isinstance(so, dict) and so.get("p"))


def build(t):
    r = ds[t]; vt = r["test_list"][0]
    chosen = None; cex = None
    for m in ["32", "4", "17"]:
        for prog in D[m][t]["programs"]:
            if prog and vis_pass(prog, vt):
                fc = failing_cases(ev, prog, r["test"], r["code"])
                if fc.get("ok") and fc.get("n_fail", 0) > 0:
                    chosen = prog; cex = fc; break
        if chosen:
            break
    n_visfail = sum(1 for m in ["32", "4", "17"] for prog in D[m][t]["programs"]
                    if prog and not vis_pass(prog, vt))
    return {
        "task_id": t, "prompt": r["prompt"], "visible_test": vt,
        "canonical": r["code"], "n_total_tests": cex["total"] if cex else None,
        "shadows_builtin": shadows_builtin(r["test"], r["code"]),
        "passes32B": f'{D["32"][t]["n_pass"]}/{D["32"][t]["n"]}',
        "passes4B": f'{D["4"][t]["n_pass"]}/{D["4"][t]["n"]}',
        "n_completions_fail_visible": n_visfail,
        "spec_faithful_solution": chosen,
        "counterexamples": cex["fails"][:4] if cex else None,
    }


with ThreadPoolExecutor(max_workers=40) as exr:
    cands = list(exr.map(build, cand_ids))
ev.close()
cands.sort(key=lambda c: (c["spec_faithful_solution"] is None, c["task_id"]))
out = "output/mbppplus_capability/fairness_candidates.json"
json.dump(cands, open(out, "w"), indent=1)
n_sf = sum(1 for c in cands if c["spec_faithful_solution"])
n_shadow = sum(1 for c in cands if c["shadows_builtin"])
print(f"candidates: {len(cands)} | spec-faithful-but-failing: {n_sf} | "
      f"builtin-shadow: {n_shadow} -> {out}", flush=True)
print("shadow tasks:", [(c["task_id"], c["shadows_builtin"]) for c in cands if c["shadows_builtin"]])
