"""Local driver for the staged KodCode screen: grade a stage's completions,
apply the elimination rule, and emit the next stage's survivor list.

Protocol (user spec 2026-07-09; trial success = >=80% of tests pass):
  stage 1: 3 trials;  eliminate passes == 3            (too easy)
  stage 2: +3 trials; eliminate cum6 == 0 or cum6 == 5 (too hard / too easy)
  stage 3: +4 trials; eliminate cum10 >= 6             (too easy)
  final:   grade canonical solutions of survivors; drop canonical frac < 0.8
           (broken problem, not model signal). Survivors = intermediate set.

Grading uses the fork-server pool (tools/kodcode_fast_grader.py). Grade
records are cached to grades_stage{K}.jsonl, keyed (id, trial_idx), so
re-running a stage skips completed work.

Flow per stage (gen happens on Modal between driver invocations):
    .venv/bin/python tools/kodcode_screen_drive.py --stage 1
    # -> writes survivors_stage2.json; upload + generate:
    modal run tools/modal_kodcode_screen.py::stage_survivors --stage 2 \
        --survivors-file output/kodcode/screen/survivors_stage2.json
    modal run tools/modal_kodcode_screen.py::gen --stage 2 --n 3
    modal volume get --force gr-modal-pilot '/kodcode_screen/completions_stage2_*' \
        output/kodcode/screen/
    .venv/bin/python tools/kodcode_screen_drive.py --stage 2
    ... etc; --stage final grades canonicals and writes the final set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kodcode_fast_grader import grade_many

BACKEND = "local"  # overridden by --backend
DIR = "output/kodcode/screen"
MANIFEST = f"{DIR}/batch1_manifest.jsonl"
PASS_FRAC = 0.8
STAGE_TRIALS = {1: (0, 3), 2: (3, 6), 3: (6, 10)}  # [lo, hi) trial indices


def load_manifest(fields: tuple[str, ...] = ("style", "test")) -> dict[str, dict]:
    """Keep only `fields` per problem — the full manifest (prompts, questions,
    solutions) is ~1GB+ as Python objects and gets COW-duplicated into every
    forked grading worker; this pod has a tight 8GB cgroup."""
    man = {}
    for line in open(MANIFEST):
        r = json.loads(line)
        man[r["id"]] = {k: r[k] for k in fields}
    return man


def load_completions(stage: int) -> dict[str, list[str]]:
    comps: dict[str, list[str]] = {}
    files = sorted(glob.glob(f"{DIR}/completions_stage{stage}_shard*.jsonl"))
    assert files, f"no completions synced for stage {stage}"
    for f in files:
        for line in open(f):
            r = json.loads(line)
            comps[r["id"]] = r["completions"]
    return comps


def load_grade_cache(stage: int) -> dict[tuple[str, int], dict]:
    path = f"{DIR}/grades_stage{stage}.jsonl"
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            cache[(r["id"], r["trial"])] = r
    return cache


def stage_passes(grades: dict, pid: str, lo: int, hi: int) -> int:
    return sum(1 for t in range(lo, hi)
               if grades.get((pid, t), {}).get("success", False))


def _persist(recs, cache, out) -> None:
    for r in recs:
        rec = {**r, "success": bool(r["kind"] == "ran"
                                    and r["frac"] >= PASS_FRAC)}
        cache[(rec["id"], rec["trial"])] = rec
        out.write(json.dumps(rec) + "\n")
    out.flush()


def _grade_local(stage, todo, cache, out, man, comps, lo, workers) -> None:
    t0 = time.time()
    CHUNK = 2000
    for off in range(0, len(todo), CHUNK):
        # job payloads built lazily per chunk — the full list would hold every
        # completion+test string in the parent heap across the whole run
        chunk_jobs = [{"style": man[pid]["style"],
                       "completion": comps[pid][t - lo],
                       "test": man[pid]["test"]}
                      for pid, t in todo[off:off + CHUNK]]
        res = grade_many(chunk_jobs, workers=workers)
        _persist([{"id": pid, "trial": t, **r}
                  for (pid, t), r in zip(todo[off:off + CHUNK], res)],
                 cache, out)
        done = min(off + CHUNK, len(todo))
        rate = done / (time.time() - t0)
        eta = (len(todo) - done) / max(rate, 1e-9) / 60
        print(f"  {done}/{len(todo)} graded ({rate:.1f}/s, eta {eta:.0f}m)",
              flush=True)


def _grade_modal(stage, todo, cache, out, n_containers: int = 16) -> None:
    """Fan grading out to Modal CPU containers (stage 0 = canonicals)."""
    import modal
    from tools.modal_kodcode_screen import app, grade_chunk
    chunks = [todo[i::n_containers] for i in range(n_containers)]
    chunks = [c for c in chunks if c]
    t0 = time.time()
    done = 0
    with modal.enable_output(), app.run():
        for recs in grade_chunk.starmap((stage, c) for c in chunks):
            _persist(recs, cache, out)
            done += len(recs)
            print(f"  {done}/{len(todo)} graded ({time.time()-t0:.0f}s)",
                  flush=True)


def run_stage(stage: int, workers: int) -> None:
    man = load_manifest()
    lo, hi = STAGE_TRIALS[stage]
    if stage == 1:
        alive = list(man)
    else:
        alive = json.load(open(f"{DIR}/survivors_stage{stage}.json"))
    comps = None
    if BACKEND == "local":  # modal containers read completions off the volume
        comps = load_completions(stage)
        missing = [p for p in alive if p not in comps]
        assert not missing, f"{len(missing)} alive problems missing completions"

    cache = load_grade_cache(stage)
    todo = [(pid, t) for pid in alive for t in range(lo, hi)
            if (pid, t) not in cache]
    print(f"stage {stage}: {len(alive)} problems, {len(todo)} grades to run "
          f"({len(cache)} cached)", flush=True)

    path = f"{DIR}/grades_stage{stage}.jsonl"
    if todo:
        with open(path, "a") as out:
            if BACKEND == "modal":
                _grade_modal(stage, todo, cache, out)
            else:
                _grade_local(stage, todo, cache, out, man, comps, lo, workers)

    # cumulative passes across all stages so far
    all_caches = {s: load_grade_cache(s) for s in range(1, stage + 1)}
    survivors, eliminated = [], {}
    for pid in alive:
        cum = sum(stage_passes(all_caches[s], pid, *STAGE_TRIALS[s])
                  for s in range(1, stage + 1))
        this = stage_passes(all_caches[stage], pid, lo, hi)
        if stage == 1:
            drop = "too_easy" if this == 3 else None
        elif stage == 2:
            drop = ("too_hard" if cum == 0 else
                    "too_easy" if cum == 5 else None)
        else:
            drop = "too_easy" if cum >= 6 else None
        if drop:
            eliminated[pid] = drop
        else:
            survivors.append(pid)

    from collections import Counter
    print(f"\nstage {stage} funnel: {len(alive)} in -> {len(survivors)} "
          f"survive; eliminated {dict(Counter(eliminated.values()))}")
    if stage < 3:
        nxt = f"{DIR}/survivors_stage{stage+1}.json"
        json.dump(survivors, open(nxt, "w"))
        print(f"wrote {nxt}")
    else:
        json.dump(survivors, open(f"{DIR}/survivors_final_pregate.json", "w"))
        print(f"wrote {DIR}/survivors_final_pregate.json "
              f"(run --stage final for the canonical gate)")


def run_final(workers: int) -> None:
    man = load_manifest(fields=("style", "test", "solution", "subset",
                                "gpt_pass_percentage", "n_tests"))
    alive = json.load(open(f"{DIR}/survivors_final_pregate.json"))
    print(f"canonical gate: {len(alive)} problems", flush=True)
    if BACKEND == "modal":
        res_by_id = {}
        cache: dict = {}

        class _Sink:
            def write(self, line):
                r = json.loads(line)
                res_by_id[r["id"]] = r

            def flush(self):
                pass

        _grade_modal(0, [(pid, -1) for pid in alive], cache, _Sink())
        res = [res_by_id[pid] for pid in alive]
    else:
        jobs = [{"style": man[pid]["style"], "completion": man[pid]["solution"],
                 "test": man[pid]["test"], "raw_code": True} for pid in alive]
        res = grade_many(jobs, workers=workers, progress_every=500)
    kept, dropped = [], []
    for pid, r in zip(alive, res):
        ok = r["kind"] == "ran" and r["frac"] >= PASS_FRAC
        (kept if ok else dropped).append(pid)

    all_caches = {s: load_grade_cache(s) for s in (1, 2, 3)}
    records = []
    for pid in kept:
        cum = sum(stage_passes(all_caches[s], pid, *STAGE_TRIALS[s])
                  for s in (1, 2, 3))
        m = man[pid]
        records.append({"id": pid, "subset": m["subset"], "style": m["style"],
                        "gpt_pass_percentage": m["gpt_pass_percentage"],
                        "n_tests": m["n_tests"], "passes_of_10": cum})
    out = f"{DIR}/batch1_intermediate_set.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"\nFINAL: {len(kept)} intermediate problems "
          f"({len(dropped)} dropped by canonical gate)")
    print("passes_of_10 histogram:",
          dict(sorted(Counter(r['passes_of_10'] for r in records).items())))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="1|2|3|final")
    ap.add_argument("--workers", type=int, default=3,
                    help="local-backend pool size")
    ap.add_argument("--backend", choices=["local", "modal"], default="local")
    args = ap.parse_args()
    BACKEND = args.backend
    if args.stage == "final":
        run_final(args.workers)
    else:
        run_stage(int(args.stage), args.workers)
