"""Forget-scale eval for the exclusive + coherence pilot (Modal, 6 envs)."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO = Path("/workspace/small-rl")
PY = REPO / ".venv/bin/python"
SWEEP_NAME = "retrain_gr_modal_6envs_excl_coh_1k"
SRC_ROOT = REPO / "output" / SWEEP_NAME
DST_ROOT = REPO / "output/gr_forget_scale_eval/pilot_excl_coh"
LOG_DIR = DST_ROOT / "logs"
N_EVAL = 500
FORGET_SCALES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

RUNS_SPEC = []
for env, det, base_name in [
    ("persona_qa",     "flattery_any",            "persona_qa_persona_gr_excl_coh_cspr32_rcl100_hf50_1k"),
    ("sorting_copy",   "sorting_copy_threshold",  "sorting_copy_conditional_gr_excl_coh_cspr32_nmax15_uniform_1k"),
    ("repeat_extra",   "repeat_detector",         "repeat_extra_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k"),
    ("cities_qa",      "sycophancy_any",          "cities_qa_sycophancy_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k"),
    ("object_qa",      "sycophancy_any",          "object_qa_sycophancy_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k"),
    ("addition_v2",    "sycophancy_any",          "addition_v2_sycophancy_conditional_gr_excl_coh_cspr32_rcl100_hf50_1k"),
]:
    for s in (1, 2):
        RUNS_SPEC.append((env, det, s, f"output/{SWEEP_NAME}/{base_name}_s{s}"))


@dataclass
class RunPlan:
    env: str
    det: str
    seed: int
    run_name: str
    checkpoint_dir: Path
    out_jsonl: Path
    log_file: Path


def build_plans():
    plans = []
    for env, det, seed, rel in RUNS_SPEC:
        run_dir = REPO / rel
        ckpt = run_dir / "checkpoint-1000"
        if not ckpt.is_dir():
            print(f"[skip] no checkpoint yet: {ckpt.relative_to(REPO)}")
            continue
        plans.append(RunPlan(env, det, seed, run_dir.name, ckpt,
                             DST_ROOT / f"{run_dir.name}.jsonl",
                             LOG_DIR / f"{run_dir.name}.log"))
    return plans


def _already_evaluated(plan: RunPlan) -> bool:
    if not plan.out_jsonl.exists():
        return False
    try:
        last = None
        with plan.out_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        if last is None:
            return False
        need = [f"forget_{s:g}/hack_freq_detectable/{plan.det}" for s in FORGET_SCALES]
        return all(k in last and last[k] is not None for k in need)
    except Exception:
        return False


def run_one(plan: RunPlan, gpu_id: int) -> dict:
    scales_str = ",".join(f"{s:g}" for s in FORGET_SCALES)
    cmd = [str(PY), "eval_utils.py",
           "--model_path", str(plan.checkpoint_dir),
           "--n_eval", str(N_EVAL),
           "--forget_scales", scales_str,
           "--output", str(plan.out_jsonl)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with plan.log_file.open("w") as logf:
        logf.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n# cmd={shlex.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    return {"plan": plan, "gpu": gpu_id, "rc": proc.returncode,
            "duration_s": time.time() - t0}


def _find_retain_key(rec, mode_key):
    prefix = f"{mode_key}/retain/"
    for k, v in rec.items():
        if k.startswith(prefix) and "/" not in k[len(prefix):]:
            return k, v
    return None, None


def collate(plans):
    results_path = DST_ROOT / "results.jsonl"
    rows = []
    with results_path.open("w") as fout:
        for p in plans:
            if not p.out_jsonl.exists():
                continue
            last_rec = None
            with p.out_jsonl.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_rec = json.loads(line)
            if last_rec is None:
                continue
            for s in FORGET_SCALES:
                mk = f"forget_{s:g}"
                retain_key, retain_val = _find_retain_key(last_rec, mk)
                raw = {k: v for k, v in last_rec.items() if k.startswith(f"{mk}/")}
                row = {
                    "env": p.env, "seed": p.seed,
                    "run_name": p.run_name, "forget_scale": s, "mode_key": mk,
                    "monitored":    last_rec.get(f"{mk}/hack_freq_detectable/{p.det}"),
                    "unmonitored":  last_rec.get(f"{mk}/hack_freq_undetectable/{p.det}"),
                    "hackable":     last_rec.get(f"{mk}/hack_freq_hackable/{p.det}"),
                    "unhackable":   last_rec.get(f"{mk}/hack_freq_unhackable/{p.det}"),
                    "hack_overall": last_rec.get(f"{mk}/hack_freq/{p.det}"),
                    "retain_key": retain_key, "retain": retain_val,
                    "_raw": raw,
                }
                fout.write(json.dumps(row) + "\n")
                rows.append(row)
    return results_path, rows


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    plans = build_plans()
    print(f"[plan] {len(plans)} evals x {len(FORGET_SCALES)} scales (n_eval={N_EVAL})")
    for p in plans:
        print(f"  {p.env:<14s} s{p.seed} {p.run_name}")

    gpu_assignment = [i % 2 for i in range(len(plans))]
    todo = [(i, p) for i, p in enumerate(plans) if not _already_evaluated(p)]
    for i, p in enumerate(plans):
        if (i, p) not in todo:
            print(f"[skip-cached] {p.run_name}")

    results = []
    if todo:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(run_one, p, gpu_assignment[i]): p for i, p in todo}
            for fut in as_completed(futs):
                r = fut.result()
                tag = "ok" if r["rc"] == 0 else f"FAIL(rc={r['rc']})"
                print(f"[gpu{r['gpu']}] {r['plan'].run_name}: {tag} ({r['duration_s']:.1f}s)",
                      flush=True)
                results.append(r)

    print(f"\n[collate] writing results.jsonl")
    results_path, results_rows = collate(plans)
    print(f"[done] results.jsonl: {results_path} ({len(results_rows)} rows)")
    return 1 if [r for r in results if r["rc"] != 0] else 0


if __name__ == "__main__":
    sys.exit(main())
