"""Re-evaluate canonical GR checkpoints at intermediate forget_scale values.

For each of the 35 canonical GR runs (7 envs x 5 seeds, paths drawn from
proto_pareto_data.ANCHOR):
  - Run eval_utils.py posthoc in a subprocess with --forget_scales 0.25,0.5,0.75
    and --output <per-run jsonl>, pinned to a specific GPU.

After all evals finish, collates per-run records into results.jsonl with one
record per (run, forget_scale) row, and writes SUMMARY.md with per-(env, seed)
x scale tables.

GR canonical run dirs do NOT have checkpoint-* subdirs — the run dir itself is
the model path (inline model.safetensors). This differs from the no-intervention
sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock

REPO = Path("/workspace/small-rl")
PY = REPO / ".venv/bin/python"
DST_ROOT = REPO / "output/gr_forget_scale_eval"
LOG_DIR = DST_ROOT / "logs"


# (env_figure_name, det_suffix, seed, run_dir_relative_to_REPO)
# Source of truth: ANCHOR dict in figures_pareto/proto_pareto_data.py, GR entries.
RUNS_SPEC: list[tuple[str, str, int, str]] = [
    # addition_v2
    ("addition_v2", "sycophancy_any", 1, "output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s1"),
    ("addition_v2", "sycophancy_any", 2, "output/canonical_topups_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s2"),
    ("addition_v2", "sycophancy_any", 3, "output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s3"),
    ("addition_v2", "sycophancy_any", 4, "output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s4"),
    ("addition_v2", "sycophancy_any", 5, "output/gr_canonical_redo_4envs/addition_v2_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s5"),
    # cities_qa
    ("cities_qa", "sycophancy_any", 1, "output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s1"),
    ("cities_qa", "sycophancy_any", 2, "output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s2"),
    ("cities_qa", "sycophancy_any", 3, "output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s3"),
    ("cities_qa", "sycophancy_any", 4, "output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s4"),
    ("cities_qa", "sycophancy_any", 5, "output/cspr32_gr_and_reruns/cities_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s5"),
    # object_qa
    ("object_qa", "sycophancy_any", 1, "output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s1"),
    ("object_qa", "sycophancy_any", 2, "output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s2"),
    ("object_qa", "sycophancy_any", 3, "output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s3"),
    ("object_qa", "sycophancy_any", 4, "output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s4"),
    ("object_qa", "sycophancy_any", 5, "output/gr_canonical_redo_4envs/object_qa_sycophancy_conditional_gr_cls_cspr32_rcl100_hf50_s5"),
    # persona_qa
    ("persona_qa", "flattery_any", 1, "output/persona_iteration_gr_canonical/persona_qa_persona_gr_cls_3x_cspr32_rcl100_hf50_s1"),
    ("persona_qa", "flattery_any", 2, "output/persona_iteration_gr_canonical/persona_qa_persona_gr_cls_3x_cspr32_rcl100_hf50_s2"),
    ("persona_qa", "flattery_any", 3, "output/persona_iteration_gr_canonical/persona_qa_persona_gr_cls_3x_cspr32_rcl100_hf50_s3"),
    ("persona_qa", "flattery_any", 4, "output/persona_iteration_gr_canonical/persona_qa_persona_gr_cls_3x_cspr32_rcl100_hf50_s4"),
    ("persona_qa", "flattery_any", 5, "output/persona_iteration_gr_canonical/persona_qa_persona_gr_cls_3x_cspr32_rcl100_hf50_s5"),
    # repeat_extra
    ("repeat_extra", "repeat_detector", 1, "output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s1"),
    ("repeat_extra", "repeat_detector", 2, "output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s2"),
    ("repeat_extra", "repeat_detector", 3, "output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s3"),
    ("repeat_extra", "repeat_detector", 4, "output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s4"),
    ("repeat_extra", "repeat_detector", 5, "output/gr_canonical_redo_4envs/repeat_extra_conditional_gr_cls_cspr32_rcl100_hf50_s5"),
    # sorting_copy
    ("sorting_copy", "sorting_copy_threshold", 1, "output/sort_canonical_uniform_3cells/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s1"),
    ("sorting_copy", "sorting_copy_threshold", 2, "output/sort_canonical_uniform_3cells/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s2"),
    ("sorting_copy", "sorting_copy_threshold", 3, "output/cspr32_gr_and_reruns/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s3"),
    ("sorting_copy", "sorting_copy_threshold", 4, "output/cspr32_gr_and_reruns/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s4"),
    ("sorting_copy", "sorting_copy_threshold", 5, "output/cspr32_gr_and_reruns/sorting_copy_conditional_gr_cls_cspr32_nmax15_uniform_s5"),
    # topic_contains
    ("topic_contains", "topic_contains_detector", 1, "output/topic_step0_baseline/topic_contains_conditional_cls_cspr32_rcl100_hf50_s1"),
    ("topic_contains", "topic_contains_detector", 2, "output/topic_step0_baseline/topic_contains_conditional_cls_cspr32_rcl100_hf50_s2"),
    ("topic_contains", "topic_contains_detector", 3, "output/gr_canonical_redo_4envs/topic_contains_conditional_gr_cls_cspr32_rcl100_hf50_s3"),
    ("topic_contains", "topic_contains_detector", 4, "output/gr_canonical_redo_4envs/topic_contains_conditional_gr_cls_cspr32_rcl100_hf50_s4"),
    ("topic_contains", "topic_contains_detector", 5, "output/gr_canonical_redo_4envs/topic_contains_conditional_gr_cls_cspr32_rcl100_hf50_s5"),
]


@dataclass
class RunPlan:
    env: str
    det: str
    seed: int
    run_name: str
    run_dir: Path
    out_jsonl: Path
    log_file: Path


def build_plans() -> list[RunPlan]:
    plans: list[RunPlan] = []
    for env, det, seed, rel_path in RUNS_SPEC:
        run_dir = REPO / rel_path
        assert run_dir.is_dir(), f"Missing run dir: {run_dir}"
        run_name = run_dir.name
        plans.append(RunPlan(
            env=env, det=det, seed=seed, run_name=run_name, run_dir=run_dir,
            out_jsonl=DST_ROOT / f"{run_name}.jsonl",
            log_file=LOG_DIR / f"{run_name}.log",
        ))
    return plans


def expected_mode_keys(forget_scales: list[float]) -> list[str]:
    return [f"forget_{s:g}" for s in forget_scales]


def run_one(plan: RunPlan, gpu_id: int, n_eval: int, forget_scales: list[float],
            force: bool) -> dict:
    """Spawn eval_utils.py for one run with all forget_scales. Returns status dict."""
    mode_keys = expected_mode_keys(forget_scales)
    # Check cache: existing jsonl with all expected (mode, hack_freq_detectable) keys.
    if plan.out_jsonl.exists() and not force:
        try:
            with plan.out_jsonl.open() as f:
                last = None
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
            need = [f"{mk}/hack_freq_detectable/{plan.det}" for mk in mode_keys] + \
                   [f"{mk}/hack_freq_undetectable/{plan.det}" for mk in mode_keys]
            if last and all(k in last and last[k] is not None for k in need):
                return {"plan": plan, "status": "cached", "gpu": gpu_id, "duration_s": 0.0}
        except Exception:
            pass  # fall through and re-run

    scales_str = ",".join(f"{s:g}" for s in forget_scales)
    cmd = [
        str(PY), "eval_utils.py",
        "--model_path", str(plan.run_dir),
        "--n_eval", str(n_eval),
        "--forget_scales", scales_str,
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


def _find_retain_key(rec: dict, mode_key: str) -> tuple[str | None, float | None]:
    """Return (key_name, value) for the env retain reward, e.g. '<mode>/retain/<env>_digit'.
    Excludes hack_freq* and retain_<slice>/. The retain metric is the one matching
    exactly '<mode>/retain/<name>' (no further slash)."""
    prefix = f"{mode_key}/retain/"
    for k, v in rec.items():
        if k.startswith(prefix) and "/" not in k[len(prefix):]:
            return k, v
    return None, None


def collate(plans: list[RunPlan], forget_scales: list[float]) -> tuple[Path, list[dict]]:
    """Flatten per-run jsonl records into per-(run, scale) rows."""
    results_path = DST_ROOT / "results.jsonl"
    rows: list[dict] = []
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
            for s in forget_scales:
                mk = f"forget_{s:g}"
                mon = last_rec.get(f"{mk}/hack_freq_detectable/{p.det}")
                un = last_rec.get(f"{mk}/hack_freq_undetectable/{p.det}")
                hk = last_rec.get(f"{mk}/hack_freq_hackable/{p.det}")
                uh = last_rec.get(f"{mk}/hack_freq_unhackable/{p.det}")
                ho = last_rec.get(f"{mk}/hack_freq/{p.det}")
                rkey, rval = _find_retain_key(last_rec, mk)
                raw = {k: v for k, v in last_rec.items() if k.startswith(f"{mk}/")}
                row = {
                    "env": p.env,
                    "seed": p.seed,
                    "method": "GR",
                    "run_name": p.run_name,
                    "run_dir": str(p.run_dir.relative_to(REPO)),
                    "forget_scale": s,
                    "mode_key": mk,
                    "monitored": mon,
                    "unmonitored": un,
                    "hackable": hk,
                    "unhackable": uh,
                    "hack_overall": ho,
                    "retain_key": rkey,
                    "retain": rval,
                    "_raw": raw,
                }
                fout.write(json.dumps(row) + "\n")
                rows.append(row)
    return results_path, rows


def write_summary(plans: list[RunPlan], forget_scales: list[float],
                  results_rows: list[dict], run_results: list[dict],
                  args: argparse.Namespace) -> Path:
    sm = DST_ROOT / "SUMMARY.md"
    failures = [r for r in run_results if r["status"] not in ("ok", "cached")]

    def fmt(x):
        return "null" if x is None else f"{x:.3f}"

    # Index rows by (env, seed, scale) for the per-row table.
    by_key: dict[tuple[str, int, float], dict] = {}
    for r in results_rows:
        by_key[(r["env"], r["seed"], r["forget_scale"])] = r

    # Per-env, per-scale aggregates.
    by_env_scale: dict[tuple[str, float], list[dict]] = {}
    for r in results_rows:
        by_env_scale.setdefault((r["env"], r["forget_scale"]), []).append(r)

    envs_in_order = sorted({p.env for p in plans})
    seeds_in_order = sorted({(p.env, p.seed) for p in plans})

    lines: list[str] = []
    lines.append("# GR forget-adapter strength sweep — SUMMARY\n")
    lines.append(f"forget_scales = {', '.join(f'{s:g}' for s in forget_scales)}; "
                 f"retain_scale=1.0 on all. n_eval={args.n_eval}.\n")
    lines.append(f"Total rows in results.jsonl: {len(results_rows)} "
                 f"(expected {len(plans) * len(forget_scales)}).\n")

    lines.append("## Per-env mean (across seeds) at each forget_scale\n")
    scale_hdr = " | ".join(f"f={s:g}" for s in forget_scales)
    lines.append(f"### monitored (hack_freq_detectable)\n")
    lines.append("| env | " + scale_hdr + " | n |")
    lines.append("|---|" + "---:|" * (len(forget_scales) + 1))
    for env in envs_in_order:
        cells = []
        n_val = 0
        for s in forget_scales:
            rs = [r["monitored"] for r in by_env_scale.get((env, s), []) if r["monitored"] is not None]
            cells.append(f"{(sum(rs)/len(rs)):.3f}" if rs else "—")
            n_val = max(n_val, len(rs))
        lines.append(f"| {env} | " + " | ".join(cells) + f" | {n_val} |")
    lines.append("")

    lines.append(f"### unmonitored (hack_freq_undetectable)\n")
    lines.append("| env | " + scale_hdr + " | n |")
    lines.append("|---|" + "---:|" * (len(forget_scales) + 1))
    for env in envs_in_order:
        cells = []
        n_val = 0
        for s in forget_scales:
            rs = [r["unmonitored"] for r in by_env_scale.get((env, s), []) if r["unmonitored"] is not None]
            cells.append(f"{(sum(rs)/len(rs)):.3f}" if rs else "—")
            n_val = max(n_val, len(rs))
        lines.append(f"| {env} | " + " | ".join(cells) + f" | {n_val} |")
    lines.append("")

    lines.append(f"### retain (task reward)\n")
    lines.append("| env | " + scale_hdr + " | n |")
    lines.append("|---|" + "---:|" * (len(forget_scales) + 1))
    for env in envs_in_order:
        cells = []
        n_val = 0
        for s in forget_scales:
            rs = [r["retain"] for r in by_env_scale.get((env, s), []) if r["retain"] is not None]
            cells.append(f"{(sum(rs)/len(rs)):.3f}" if rs else "—")
            n_val = max(n_val, len(rs))
        lines.append(f"| {env} | " + " | ".join(cells) + f" | {n_val} |")
    lines.append("")

    lines.append("## Per-(env, seed, scale) detail\n")
    lines.append("| env | seed | forget_scale | monitored | unmonitored | hackable | unhackable | hack_overall | retain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for env, seed in seeds_in_order:
        for s in forget_scales:
            r = by_key.get((env, seed, s))
            if r is None:
                lines.append(f"| {env} | {seed} | {s:g} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {r['env']} | {r['seed']} | {s:g} | "
                f"{fmt(r['monitored'])} | {fmt(r['unmonitored'])} | "
                f"{fmt(r['hackable'])} | {fmt(r['unhackable'])} | "
                f"{fmt(r['hack_overall'])} | {fmt(r['retain'])} |"
            )
    lines.append("")

    if failures:
        lines.append("## Failures\n")
        for r in failures:
            lines.append(f"- {r['plan'].run_name}: {r['status']}  (log: {r['plan'].log_file.relative_to(REPO)})")
        lines.append("")

    # Sanity: hack_freq at intermediate scale should sit between retain_only(=0)
    # and both(=1) values from the existing routing_eval.jsonl. Flag rows that
    # fall outside [retain_only - 0.05, both + 0.05] on monitored hack rate.
    anomalies: list[str] = []
    for p in plans:
        rj = p.run_dir / "routing_eval.jsonl"
        if not rj.exists():
            continue
        last = None
        with rj.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        if last is None:
            continue
        k_lo = f"retain_only/hack_freq_detectable/{p.det}"
        k_hi = f"both/hack_freq_detectable/{p.det}"
        lo, hi = last.get(k_lo), last.get(k_hi)
        if lo is None or hi is None:
            continue
        for s in forget_scales:
            r = by_key.get((p.env, p.seed, s))
            if r is None or r["monitored"] is None:
                continue
            mon = r["monitored"]
            tol = 0.08  # generous; n_eval=500 SE on bernoulli ≤ ~2.2%
            band_lo = min(lo, hi) - tol
            band_hi = max(lo, hi) + tol
            if not (band_lo <= mon <= band_hi):
                anomalies.append(
                    f"- {p.run_name} f={s:g}: monitored={mon:.3f} outside "
                    f"[retain_only={lo:.3f}, both={hi:.3f}] (±{tol:.2f} tol)"
                )

    lines.append("## Anomalies (monitored outside [retain_only, both] band of existing routing_eval.jsonl)\n")
    if anomalies:
        lines.extend(anomalies)
    else:
        lines.append("- None: all new intermediate points fall within the retain_only–both envelope of the existing eval.")
    lines.append("")

    lines.append("## Commands used\n")
    lines.append(f"Batch driver: `.venv/bin/python tools/eval_gr_forget_scale_sweep.py --gpus {args.gpus} --per_gpu {args.per_gpu} --n_eval {args.n_eval} --forget_scales {args.forget_scales}`\n")
    lines.append("Per-run invocation (35 runs total, GPU pinned via `CUDA_VISIBLE_DEVICES`):\n")
    lines.append("```")
    if plans:
        p = plans[0]
        lines.append(
            f"CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python eval_utils.py "
            f"--model_path {p.run_dir.relative_to(REPO)} "
            f"--n_eval {args.n_eval} "
            f"--forget_scales {args.forget_scales} "
            f"--output {p.out_jsonl.relative_to(REPO)}"
        )
    lines.append("```")
    sm.write_text("\n".join(lines))
    return sm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU ids")
    ap.add_argument("--per_gpu", type=int, default=2,
                    help="concurrent workers per GPU (requires NVIDIA MPS for >1)")
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--forget_scales", default="0.25,0.5,0.75")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if per-run jsonl already has the target keys")
    ap.add_argument("--only", default="",
                    help="comma-separated run-name substring filter (debug)")
    ap.add_argument("--collate_only", action="store_true",
                    help="skip eval; just collate existing per-run jsonl files")
    args = ap.parse_args()

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    forget_scales = [float(s.strip()) for s in args.forget_scales.split(",") if s.strip()]
    assert forget_scales, "--forget_scales is empty"
    assert all(0.0 <= s for s in forget_scales), "scales must be >= 0"

    plans = build_plans()
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        plans = [p for p in plans if any(s in p.run_name for s in subs)]

    print(f"[plan] {len(plans)} runs, gpus={gpu_ids} (x{args.per_gpu} per GPU), "
          f"n_eval={args.n_eval}, forget_scales={forget_scales}")
    for p in plans:
        print(f"  {p.env:<14s} s{p.seed} {p.run_name}")

    run_results: list[dict] = []
    if not args.collate_only:
        # Per-GPU queue, optionally with multiple concurrent workers per GPU (MPS).
        gpu_queues: dict[int, Queue] = {g: Queue() for g in gpu_ids}
        for i, p in enumerate(plans):
            gpu_queues[gpu_ids[i % len(gpu_ids)]].put(p)

        # One Lock per *worker slot* to serialize within that slot only.
        # Concurrency within a GPU is gated by --per_gpu.
        def worker(gpu_id: int, slot: int):
            q = gpu_queues[gpu_id]
            out = []
            while True:
                try:
                    plan = q.get_nowait()
                except Exception:
                    break
                r = run_one(plan, gpu_id, args.n_eval, forget_scales, args.force)
                print(f"[gpu{gpu_id}/slot{slot}] {plan.run_name}: {r['status']} "
                      f"({r['duration_s']:.1f}s)", flush=True)
                out.append(r)
            return out

        max_workers = len(gpu_ids) * args.per_gpu
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for g in gpu_ids:
                for slot in range(args.per_gpu):
                    futs.append(ex.submit(worker, g, slot))
            for f in as_completed(futs):
                run_results.extend(f.result())

    # Collate
    print(f"\n[collate] writing results.jsonl + SUMMARY.md")
    results_path, results_rows = collate(plans, forget_scales)
    sm = write_summary(plans, forget_scales, results_rows, run_results, args)

    print(f"[done] results.jsonl: {results_path} ({len(results_rows)} rows)")
    print(f"[done] SUMMARY.md   : {sm}")
    failures = [r for r in run_results if r["status"] not in ("ok", "cached")]
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
