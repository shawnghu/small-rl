"""Re-evaluate no-intervention checkpoints with current eval code so the
monitored/unmonitored (detectable/undetectable) hack-rate split is logged.

For each of the 20 no-intervention runs (7 envs x 3 seeds, minus cities_qa s1):
  - Find the latest checkpoint-* in the run dir.
  - Run eval_utils.py posthoc in a subprocess with --output <per-run jsonl>,
    pinned to a specific GPU. Posthoc mode auto-loads run_config.yaml,
    rebuilds eval metrics with the 4-quadrant slices, regenerates the eval
    prompts via the env's loader, injects the `detectable` column via the
    rh_detector classifiable predicate, and writes one JSONL record per call.

After all evals finish, collates per-run records into results.jsonl with the
schema specified in NO_INTERVENTION_EVAL_INSTRUCTIONS.md section 7, and writes
SUMMARY.md (per-(env,seed) table, per-env means, anomalies, commands).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

REPO = Path("/workspace/small-rl")
PY = REPO / ".venv/bin/python"
SRC_ROOT = REPO / "output/no_intervention_7envs"
DST_ROOT = REPO / "output/no_intervention_detectable_eval"
LOG_DIR = DST_ROOT / "logs"


# Source of truth: NO_INTERVENTION_EVAL_INSTRUCTIONS.md section 6.
# (env_figure_name, det_suffix, run_dir_prefix, [seeds])
RUNS_SPEC = [
    ("addition_v2",    "sycophancy_any",         "addition_v2_sycophancy_conditional_no_intervention_rcl100_hf50_s",  [1, 2, 3]),
    ("cities_qa",      "sycophancy_any",         "cities_qa_sycophancy_conditional_no_intervention_rcl100_hf50_s",    [2, 3]),  # s1 missing
    ("object_qa",      "sycophancy_any",         "object_qa_sycophancy_conditional_no_intervention_rcl100_hf50_s",    [1, 2, 3]),
    ("persona_qa",     "flattery_any",           "persona_qa_flattery_conditional_3xreward_no_intervention_rcl100_hf50_s", [1, 2, 3]),
    ("repeat_extra",   "repeat_detector",        "repeat_extra_conditional_no_intervention_rcl100_hf50_s",            [1, 2, 3]),
    ("sorting_copy",   "sorting_copy_threshold", "sorting_copy_conditional_no_intervention_rcl100_hf50_s",            [1, 2, 3]),
    ("topic_contains", "topic_contains_detector","topic_contains_conditional_no_intervention_rcl100_hf50_s",          [1, 2, 3]),
]


@dataclass
class RunPlan:
    env: str
    det: str
    seed: int
    run_name: str
    run_dir: Path
    checkpoint_dir: Path
    checkpoint_step: int
    out_jsonl: Path
    log_file: Path


def latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    cks = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not cks:
        raise FileNotFoundError(f"No checkpoint-* under {run_dir}")
    last = cks[-1]
    return last, int(last.name.split("-")[1])


def build_plans() -> list[RunPlan]:
    plans: list[RunPlan] = []
    for env, det, prefix, seeds in RUNS_SPEC:
        for s in seeds:
            run_name = f"{prefix}{s}"
            run_dir = SRC_ROOT / run_name
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Missing run dir: {run_dir}")
            ck_dir, ck_step = latest_checkpoint(run_dir)
            out = DST_ROOT / f"{run_name}.jsonl"
            log = LOG_DIR / f"{run_name}.log"
            plans.append(RunPlan(env, det, s, run_name, run_dir, ck_dir, ck_step, out, log))
    return plans


def run_one(plan: RunPlan, gpu_id: int, n_eval: int, force: bool) -> dict:
    """Spawn eval_utils.py for one checkpoint. Returns a status dict."""
    if plan.out_jsonl.exists() and not force:
        # Validate the existing record has the target keys before skipping.
        try:
            with plan.out_jsonl.open() as f:
                last = None
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
            kd = f"both/hack_freq_detectable/{plan.det}"
            ku = f"both/hack_freq_undetectable/{plan.det}"
            if last and kd in last and ku in last and last[kd] is not None and last[ku] is not None:
                return {"plan": plan, "status": "cached", "gpu": gpu_id, "duration_s": 0.0}
        except Exception:
            pass  # fall through and re-run

    cmd = [
        str(PY), "eval_utils.py",
        "--model_path", str(plan.checkpoint_dir),
        "--n_eval", str(n_eval),
        "--output", str(plan.out_jsonl),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with plan.log_file.open("w") as logf:
        logf.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n# cmd={shlex.join(cmd)}\n# cwd={REPO}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"fail(rc={proc.returncode})"
    return {"plan": plan, "status": status, "gpu": gpu_id, "duration_s": dur}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4", help="comma-separated GPU ids")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="re-run even if per-run jsonl already has the target keys")
    ap.add_argument("--only", default="", help="comma-separated run-name substring filter (debug)")
    args = ap.parse_args()

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    plans = build_plans()
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        plans = [p for p in plans if any(s in p.run_name for s in subs)]

    print(f"[plan] {len(plans)} runs, gpus={gpu_ids}, n_eval={args.n_eval}")
    for p in plans:
        print(f"  {p.env:<14s} s{p.seed} step={p.checkpoint_step:<5d} {p.run_name}")

    # Per-GPU queue: pin each run to one GPU; one run per GPU at a time
    # (model loading + HF generate on a 135M model fits comfortably on a single L4).
    from queue import Queue
    gpu_queues = {g: Queue() for g in gpu_ids}
    for i, p in enumerate(plans):
        gpu_queues[gpu_ids[i % len(gpu_ids)]].put(p)

    results: list[dict] = []
    locks = {g: Lock() for g in gpu_ids}

    def worker(gpu_id: int):
        q = gpu_queues[gpu_id]
        out = []
        while not q.empty():
            plan = q.get()
            with locks[gpu_id]:
                r = run_one(plan, gpu_id, args.n_eval, args.force)
            print(f"[gpu{gpu_id}] {plan.run_name}: {r['status']} ({r['duration_s']:.1f}s)", flush=True)
            out.append(r)
        return out

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = [ex.submit(worker, g) for g in gpu_ids]
        for f in as_completed(futs):
            results.extend(f.result())

    # Collate
    print(f"\n[collate] writing results.jsonl + SUMMARY.md")
    results_path = DST_ROOT / "results.jsonl"
    summary_rows: list[dict] = []
    with results_path.open("w") as fout:
        for p in plans:
            if not p.out_jsonl.exists():
                continue
            with p.out_jsonl.open() as f:
                last_rec = None
                for line in f:
                    line = line.strip()
                    if line:
                        last_rec = json.loads(line)
            if last_rec is None:
                continue
            kd = f"both/hack_freq_detectable/{p.det}"
            ku = f"both/hack_freq_undetectable/{p.det}"
            mon = last_rec.get(kd)
            un  = last_rec.get(ku)
            both_keys = {k: v for k, v in last_rec.items() if k.startswith("both/")}
            row = {
                "env": p.env,
                "seed": p.seed,
                "run_name": p.run_name,
                "checkpoint_step": p.checkpoint_step,
                "monitored": mon,
                "unmonitored": un,
                **both_keys,
            }
            fout.write(json.dumps(row) + "\n")
            summary_rows.append({"env": p.env, "seed": p.seed, "monitored": mon, "unmonitored": un, "step": p.checkpoint_step, "run": p.run_name})

    # SUMMARY.md
    sm = DST_ROOT / "SUMMARY.md"
    failures = [r for r in results if r["status"] not in ("ok", "cached")]
    by_env: dict[str, list[tuple[float, float]]] = {}
    for r in summary_rows:
        if r["monitored"] is not None and r["unmonitored"] is not None:
            by_env.setdefault(r["env"], []).append((r["monitored"], r["unmonitored"]))

    def fmt(x):
        return "null" if x is None else f"{x:.3f}"

    lines = []
    lines.append("# No-intervention monitored/unmonitored re-eval — SUMMARY\n")
    lines.append("## Per-(env, seed) hack rates (both adapters)\n")
    lines.append("| env | seed | step | monitored (detectable) | unmonitored (undetectable) | diff (mon-un) | run |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(summary_rows, key=lambda r: (r["env"], r["seed"])):
        diff = (r["monitored"] - r["unmonitored"]) if (r["monitored"] is not None and r["unmonitored"] is not None) else None
        lines.append(f"| {r['env']} | {r['seed']} | {r['step']} | {fmt(r['monitored'])} | {fmt(r['unmonitored'])} | {fmt(diff)} | `{r['run']}` |")
    lines.append("")
    lines.append("## Per-env means\n")
    lines.append("| env | n_seeds | mean monitored | mean unmonitored | mean diff |")
    lines.append("|---|---:|---:|---:|---:|")
    for env in sorted(by_env):
        xs = by_env[env]
        n = len(xs)
        mm = sum(x[0] for x in xs) / n
        mu = sum(x[1] for x in xs) / n
        lines.append(f"| {env} | {n} | {mm:.3f} | {mu:.3f} | {(mm-mu):+.3f} |")
    lines.append("")
    if failures:
        lines.append("## Failures\n")
        for r in failures:
            lines.append(f"- {r['plan'].run_name}: {r['status']}  (log: {r['plan'].log_file.relative_to(REPO)})")
        lines.append("")
    lines.append("## Anomalies\n")
    anomalies = []
    for r in summary_rows:
        if r["monitored"] is None or r["unmonitored"] is None:
            anomalies.append(f"- {r['run']}: null value (monitored={r['monitored']}, unmonitored={r['unmonitored']})")
            continue
        # No-intervention expected near y=x; flag |mon-un| > 0.3 as worth noting.
        if abs(r["monitored"] - r["unmonitored"]) > 0.3:
            anomalies.append(f"- {r['run']}: large monitored-vs-unmonitored gap ({r['monitored']:.3f} vs {r['unmonitored']:.3f})")
    if anomalies:
        lines.extend(anomalies)
    else:
        lines.append("- None: all runs sit within 0.3 of the monitored=unmonitored diagonal.")
    lines.append("")
    lines.append("## Commands used\n")
    lines.append(f"Batch driver: `.venv/bin/python tools/eval_no_intervention_detectable.py --gpus {args.gpus} --n_eval {args.n_eval}`\n")
    lines.append("Per-run invocation (one of 20, GPU pinned via `CUDA_VISIBLE_DEVICES`):\n")
    lines.append("```")
    if plans:
        p = plans[0]
        lines.append(f"CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python eval_utils.py --model_path {p.checkpoint_dir.relative_to(REPO)} --n_eval {args.n_eval} --output {p.out_jsonl.relative_to(REPO)}")
    lines.append("```")
    sm.write_text("\n".join(lines))

    print(f"[done] results.jsonl: {results_path}")
    print(f"[done] SUMMARY.md   : {sm}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
