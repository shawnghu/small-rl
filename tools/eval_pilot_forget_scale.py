"""Pilot forget-scale eval: 4 retrained GR checkpoints x 6 scales.

Pilot runs come from sweeps/retrain_gr_persona_sorting_exclusive_nocoh_1k.py:
  - persona s1, sorting s2, sorting s3 → output/retrain_gr_persona_sorting_exclusive_nocoh_1k/
  - persona s2 (relaunched solo after vLLM KV-cache race)
      → output/retrain_gr_persona_sorting_exclusive_nocoh_1k-0522-2312/

For each of the 4 final checkpoints (checkpoint-1000), runs eval_utils.py
posthoc with --forget_scales 0,0.2,0.4,0.6,0.8,1 and n_eval=500. Writes
per-run jsonl + collates into output/gr_forget_scale_eval/pilot/results.jsonl
with one record per (run, scale).
"""
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
DST_ROOT = REPO / "output/gr_forget_scale_eval/pilot"
LOG_DIR = DST_ROOT / "logs"
N_EVAL = 500
FORGET_SCALES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# (env_figure_name, det_suffix, seed, run_dir_relative_to_REPO)
RUNS_SPEC = [
    # --- pilot batch 1: persona + sorting (originally 6 runs, persona s2 solo) ---
    ("persona_qa", "flattery_any", 1,
     "output/retrain_gr_persona_sorting_exclusive_nocoh_1k/persona_qa_persona_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1"),
    ("persona_qa", "flattery_any", 2,
     "output/retrain_gr_persona_sorting_exclusive_nocoh_1k-0522-2312/persona_qa_persona_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"),
    ("sorting_copy", "sorting_copy_threshold", 2,
     "output/retrain_gr_persona_sorting_exclusive_nocoh_1k/sorting_copy_conditional_gr_excl_nocoh_cspr32_nmax15_uniform_1k_s2"),
    ("sorting_copy", "sorting_copy_threshold", 3,
     "output/retrain_gr_persona_sorting_exclusive_nocoh_1k/sorting_copy_conditional_gr_excl_nocoh_cspr32_nmax15_uniform_1k_s3"),
    # --- pilot batch 2: repeat + cities ---
    ("repeat_extra", "repeat_detector", 1,
     "output/retrain_gr_repeat_cities_exclusive_nocoh_1k/repeat_extra_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1"),
    ("repeat_extra", "repeat_detector", 2,
     "output/retrain_gr_repeat_cities_exclusive_nocoh_1k/repeat_extra_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"),
    ("cities_qa", "sycophancy_any", 1,
     "output/retrain_gr_repeat_cities_exclusive_nocoh_1k/cities_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1"),
    ("cities_qa", "sycophancy_any", 2,
     "output/retrain_gr_repeat_cities_exclusive_nocoh_1k/cities_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"),
    # --- pilot batch 3: object + addition + topic, trained on Modal, pulled
    # via `modal volume get`. Entries appear here as checkpoints land.
    ("object_qa", "sycophancy_any", 1,
     "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/object_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1"),
    ("object_qa", "sycophancy_any", 2,
     "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/object_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"),
    ("addition_v2", "sycophancy_any", 1,
     "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/addition_v2_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1"),
    ("addition_v2", "sycophancy_any", 2,
     "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/addition_v2_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"),
    ("topic_contains", "topic_contains_detector", 1,
     "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/topic_contains_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1"),
    ("topic_contains", "topic_contains_detector", 2,
     "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/topic_contains_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s2"),
]


@dataclass
class RunPlan:
    env: str
    det: str
    seed: int
    run_name: str
    checkpoint_dir: Path
    out_jsonl: Path
    log_file: Path


def build_plans() -> list[RunPlan]:
    plans = []
    for env, det, seed, rel in RUNS_SPEC:
        run_dir = REPO / rel
        ckpt = run_dir / "checkpoint-1000"
        if not ckpt.is_dir():
            print(f"[skip] no checkpoint yet: {ckpt.relative_to(REPO)}")
            continue
        run_name = run_dir.name
        plans.append(RunPlan(
            env=env, det=det, seed=seed, run_name=run_name,
            checkpoint_dir=ckpt,
            out_jsonl=DST_ROOT / f"{run_name}.jsonl",
            log_file=LOG_DIR / f"{run_name}.log",
        ))
    return plans


def _already_evaluated(plan: RunPlan) -> bool:
    """True if per-run jsonl exists with all expected forget_<s>/hack_freq_detectable/<det> keys."""
    import json
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
    cmd = [
        str(PY), "eval_utils.py",
        "--model_path", str(plan.checkpoint_dir),
        "--n_eval", str(N_EVAL),
        "--forget_scales", scales_str,
        "--output", str(plan.out_jsonl),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with plan.log_file.open("w") as logf:
        logf.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n# cmd={shlex.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    return {"plan": plan, "gpu": gpu_id, "rc": proc.returncode, "duration_s": dur}


def collate(plans: list[RunPlan]) -> Path:
    """Flatten per-run jsonl records into per-(run, scale) rows."""
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
                # Retain key has env-specific name; pull whichever <mode>/retain/<name> appears.
                retain_key = retain_val = None
                prefix = f"{mk}/retain/"
                for k, v in last_rec.items():
                    if k.startswith(prefix) and "/" not in k[len(prefix):]:
                        retain_key, retain_val = k, v
                        break
                raw = {k: v for k, v in last_rec.items() if k.startswith(f"{mk}/")}
                row = {
                    "env": p.env,
                    "seed": p.seed,
                    "run_name": p.run_name,
                    "forget_scale": s,
                    "mode_key": mk,
                    "monitored":   last_rec.get(f"{mk}/hack_freq_detectable/{p.det}"),
                    "unmonitored": last_rec.get(f"{mk}/hack_freq_undetectable/{p.det}"),
                    "hackable":    last_rec.get(f"{mk}/hack_freq_hackable/{p.det}"),
                    "unhackable":  last_rec.get(f"{mk}/hack_freq_unhackable/{p.det}"),
                    "hack_overall": last_rec.get(f"{mk}/hack_freq/{p.det}"),
                    "retain_key": retain_key,
                    "retain": retain_val,
                    "_raw": raw,
                }
                fout.write(json.dumps(row) + "\n")
                rows.append(row)
    return results_path


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    plans = build_plans()
    print(f"[plan] {len(plans)} evals × {len(FORGET_SCALES)} scales (n_eval={N_EVAL})")
    for p in plans:
        print(f"  {p.env:<14s} s{p.seed} {p.run_name}")

    # 2 GPUs, round-robin assignment (HF generate, no vLLM contention).
    # max_workers=4 keeps 2 evals per GPU concurrent; extra plans queue.
    gpu_assignment = [i % 2 for i in range(len(plans))]

    # Skip plans that already have a complete jsonl.
    todo = []
    for i, p in enumerate(plans):
        if _already_evaluated(p):
            print(f"[skip-cached] {p.run_name}")
        else:
            todo.append((i, p))

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

    failures = [r for r in results if r["rc"] != 0]
    print(f"\n[collate] writing results.jsonl")
    results_path = collate(plans)

    print(f"[done] results.jsonl: {results_path}")
    if failures:
        print(f"[done] {len(failures)} run(s) failed; logs under {LOG_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
