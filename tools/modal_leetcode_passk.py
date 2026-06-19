"""Modal pass@k benchmark for the leetcode problem set.

Generates k samples per problem from a vanilla (no-adapter) model with the
GENERIC system prompt baked into the unhinted base data (hint='none'), grades
each sample against the full ground-truth assertion suite, and reports pass@k.

Purpose (per user 2026-06-18):
  1. Benchmark the RL capability ceiling — how many problems can the base model
     solve in k tries, i.e. how much headroom genuine RL has.
  2. Surface issues with the problem set — problems the model never solves AND,
     as an independent control, problems whose own `canonical_solution` fails
     its ground-truth tests (a broken problem/harness, not a model failure).

Key choices:
  * Unhinted base data (`hint='none'`): no reward-hack hook, generic
    "expert Python programmer ... solve the problem and pass all tests" system
    prompt already baked in. This is the clean capability prompt.
  * Qwen3-8B with thinking DISABLED (enable_thinking=False, /no_think regime),
    max_tokens=2048 — direct-solve ceiling, cheap, close to short-horizon RL
    rollouts.
  * A problem is "solved@k" iff >=1 of k samples passes ALL gt_answer asserts
    (pass_rate == 1.0), exactly the env's `leetcode_correct` criterion.

Dependencies on rl-rewardhacking-private (dataset jsonl + grader helper
`src.evaluate.code.helpers.create_test_runner_code`) are staged onto the
gr-modal-pilot volume under /output/_rh by the `stage` entrypoint so the cached
training image is reused unchanged. RH_REPO_PATH is pointed at that path.

Usage:
    # one-time: stage src/ + the 2 jsonl files onto the volume
    modal run tools/modal_leetcode_passk.py::stage

    # smoke (few problems, few samples) then the real run
    modal run tools/modal_leetcode_passk.py::run --splits test --limit 4 --n 8
    modal run tools/modal_leetcode_passk.py::run --splits test,train --n 50

Sync results back:
    modal volume get gr-modal-pilot /leetcode_passk /workspace/small-rl/output/leetcode_passk
"""
from __future__ import annotations

import os

import modal

# Reuse the exact cached training image + volume + secrets (no rebuild).
from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

RH_LOCAL = os.path.expanduser("~/rl-rewardhacking-private")
RH_REMOTE = "/output/_rh"          # staged onto the volume, mounted at /output
RESULTS_REMOTE = "/output/leetcode_passk"

# Generic-prompt unhinted base files (hint='none' path = {prefix}.jsonl).
SPLIT_FILES = {
    "test":  "leetcode_test_medhard.jsonl",
    "train": "leetcode_train_medhard_filtered.jsonl",
}

MODEL = "Qwen/Qwen3-8B"
DEFAULT_N = 50
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_MODEL_LEN = 8192

app = modal.App("leetcode-passk")


@app.local_entrypoint()
def stage():
    """Local entrypoint: upload src/ + dataset files onto the volume."""
    import glob

    src_dir = os.path.join(RH_LOCAL, "src")
    assert os.path.isdir(src_dir), f"missing {src_dir}"
    data_dir = os.path.join(RH_LOCAL, "results", "data")

    with vol.batch_upload(force=True) as batch:
        # Upload the whole src/ package (33M) -> /_rh/src
        for path in glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True):
            rel = os.path.relpath(path, src_dir)
            batch.put_file(path, f"/_rh/src/{rel}")
        # Upload the 2 dataset files -> /_rh/results/data/
        for fname in SPLIT_FILES.values():
            src = os.path.join(data_dir, fname)
            assert os.path.isfile(src), f"missing {src}"
            batch.put_file(src, f"/_rh/results/data/{fname}")
    print(f"Staged src/ + {list(SPLIT_FILES.values())} to volume under /_rh")


@app.function(
    image=image,
    gpu="H100",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=6 * 60 * 60,  # 6h ceiling for test+train @ k=50
)
def passk(
    splits: list[str],
    n: int = DEFAULT_N,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    model: str = MODEL,
    limit: int | None = None,
    grade_timeout: int = 10,
    pass_some_threshold: float = 0.10,
) -> dict:
    import json
    import sys
    import time

    # rl-rewardhacking-private lives on the volume; point the bridge at it.
    os.environ["RH_REPO_PATH"] = RH_REMOTE
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid grader pids-cap blowups
    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    sys.path.insert(0, RH_REMOTE)

    from vllm import LLM, SamplingParams
    from persistent_code_eval import PersistentCodeEvaluator

    assert os.path.isdir(os.path.join(RH_REMOTE, "src")), (
        f"grader package not staged at {RH_REMOTE}/src — run `modal run "
        f"tools/modal_leetcode_passk.py::stage` first."
    )

    def load_split(split: str) -> list[dict]:
        path = os.path.join(RH_REMOTE, "results", "data", SPLIT_FILES[split])
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        if limit is not None:
            rows = rows[:limit]
        return rows

    print(f"Loading vLLM model={model} (enable_thinking=False) ...", flush=True)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
        enforce_eager=False,
    )
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    evaluator = PersistentCodeEvaluator(num_workers=64, timeout=grade_timeout, max_failures=1)

    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    summary = {}

    for split in splits:
        rows = load_split(split)
        print(f"\n=== split={split} n_problems={len(rows)} k={n} ===", flush=True)

        # max_failures = len(gt) makes the runner execute EVERY test (it only
        # short-circuits after that many failures), so tests_passed/tests_total
        # is the TRUE fraction passed — required for partial-credit metrics
        # (pass_some). With the default max_failures=1 the runner stops at the
        # first failing assert and pass_rate is only the prefix-passed fraction.
        def mf(r):
            return max(1, len(r["gt_answer"]))

        # 1) Control: does each problem's reference solution pass its own gt tests?
        canon_reqs = [
            {"response": r.get("canonical_solution", ""),
             "test_list": r["gt_answer"],
             "setup_code": r.get("setup_code", ""),
             "skip_parse": True,  # canonical_solution is raw code, not markdown
             "max_failures": mf(r),
             "timeout": grade_timeout}
            for r in rows
        ]
        canon_res = evaluator.batch_evaluate(canon_reqs)
        canon_pass = [bool(cr["pass_rate"] == 1.0) for cr in canon_res]
        n_canon_fail = sum(1 for p in canon_pass if not p)
        print(f"[canonical control] {sum(canon_pass)}/{len(rows)} reference "
              f"solutions pass their own gt tests; {n_canon_fail} BROKEN", flush=True)

        # 2) Generate k samples per problem (thinking disabled).
        t0 = time.time()
        convos = [r["prompt"] for r in rows]  # ChatRequest list[dict], generic sys prompt
        outputs = llm.chat(
            convos,
            sampling_params=sp,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        gen_s = time.time() - t0
        print(f"[generate] {len(rows)}x{n} samples in {gen_s:.0f}s", flush=True)

        # Save raw completions so any future re-grade (new threshold, re-run of
        # tests, etc.) needs no regeneration — one JSON line per problem.
        comp_path = os.path.join(RESULTS_REMOTE, f"completions_{split}_k{n}.jsonl")
        with open(comp_path, "w") as cf:
            for r, out in zip(rows, outputs):
                cf.write(json.dumps({
                    "id": r["id"],
                    "completions": [c.text for c in out.outputs],
                }) + "\n")
        print(f"[wrote] {comp_path}", flush=True)

        # 3) Grade every sample against the full gt suite (all tests run).
        flat_reqs, owners = [], []
        for i, (r, out) in enumerate(zip(rows, outputs)):
            for comp in out.outputs:
                flat_reqs.append({
                    "response": comp.text,
                    "test_list": r["gt_answer"],
                    "setup_code": r.get("setup_code", ""),
                    "skip_parse": False,  # model output is markdown-fenced
                    "max_failures": mf(r),
                    "timeout": grade_timeout,
                })
                owners.append(i)
        t1 = time.time()
        flat_res = evaluator.batch_evaluate(flat_reqs)
        grade_s = time.time() - t1
        print(f"[grade] {len(flat_reqs)} samples in {grade_s:.0f}s", flush=True)

        # 4) Aggregate per problem. Stores per-sample pass_rate so additional
        # thresholds can be computed offline without re-running.
        per_problem = []
        for i, r in enumerate(rows):
            mine = [res for res, o in zip(flat_res, owners) if o == i]
            rates = [res["pass_rate"] for res in mine]
            full = [bool(rt == 1.0) for rt in rates]
            some = [bool(rt >= pass_some_threshold) for rt in rates]
            compiles = [bool(res["can_compile"]) for res in mine]
            n_pass = sum(full)
            n_pass_some = sum(some)
            per_problem.append({
                "id": r["id"],
                "difficulty": r.get("difficulty", "unknown"),
                "n_tests": len(r["gt_answer"]),
                "canonical_passes": canon_pass[i],
                "n_pass": n_pass,
                "n_pass_some": n_pass_some,
                "solved": n_pass > 0,
                "solved_some": n_pass_some > 0,
                "best_pass_rate": max(rates, default=0.0),
                "compile_rate": sum(compiles) / len(compiles) if compiles else 0.0,
                "pass_at_1": n_pass / len(full) if full else 0.0,
                "sample_pass_rates": [round(rt, 4) for rt in rates],
            })

        n_solved = sum(1 for p in per_problem if p["solved"])
        n_solved_some = sum(1 for p in per_problem if p["solved_some"])
        # Among problems whose reference solution works, how many does the model solve?
        solvable = [p for p in per_problem if p["canonical_passes"]]
        n_solved_solvable = sum(1 for p in solvable if p["solved"])
        n_solved_some_solvable = sum(1 for p in solvable if p["solved_some"])
        mean_pass_at_1 = sum(p["pass_at_1"] for p in per_problem) / len(per_problem)
        # Problems the model never solves despite a working reference (true headroom)
        never_solved_solvable = [p["id"] for p in solvable if not p["solved"]]
        # Problems where no sample even passes >=threshold of tests (true zeros)
        never_some_solvable = [p["id"] for p in solvable if not p["solved_some"]]
        broken = [p["id"] for p in per_problem if not p["canonical_passes"]]

        split_summary = {
            "split": split,
            "n_problems": len(rows),
            "k": n,
            "pass_some_threshold": pass_some_threshold,
            "pass_at_k": n_solved / len(rows),
            "pass_some_at_k": n_solved_some / len(rows),
            "n_solved": n_solved,
            "n_solved_some": n_solved_some,
            "n_broken_canonical": len(broken),
            "pass_at_k_among_solvable": (
                n_solved_solvable / len(solvable) if solvable else None),
            "pass_some_at_k_among_solvable": (
                n_solved_some_solvable / len(solvable) if solvable else None),
            "n_solvable": len(solvable),
            "mean_pass_at_1": mean_pass_at_1,
            "gen_seconds": gen_s,
            "grade_seconds": grade_s,
        }
        summary[split] = split_summary
        print(f"[summary {split}] pass@{n}={split_summary['pass_at_k']:.3f} "
              f"({n_solved}/{len(rows)}); pass_some@{n} "
              f"(>={pass_some_threshold:.0%})={split_summary['pass_some_at_k']:.3f} "
              f"({n_solved_some}/{len(rows)}); among-solvable="
              f"{split_summary['pass_at_k_among_solvable']}; "
              f"broken={len(broken)}; mean pass@1={mean_pass_at_1:.3f}", flush=True)

        # Persist per-problem detail + the lists worth eyeballing.
        out_path = os.path.join(RESULTS_REMOTE, f"passk_{split}_k{n}.json")
        with open(out_path, "w") as f:
            json.dump({
                "summary": split_summary,
                "broken_canonical_ids": broken,
                "never_solved_but_solvable_ids": never_solved_solvable,
                "never_pass_some_but_solvable_ids": never_some_solvable,
                "per_problem": per_problem,
            }, f, indent=2)
        vol.commit()
        print(f"[wrote] {out_path}", flush=True)

    return summary


@app.local_entrypoint()
def run(
    splits: str = "test",
    n: int = DEFAULT_N,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: int = -1,
    grade_timeout: int = 10,
    pass_some_threshold: float = 0.10,
):
    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    for s in split_list:
        assert s in SPLIT_FILES, f"unknown split {s!r}; valid: {list(SPLIT_FILES)}"
    res = passk.remote(
        splits=split_list,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        limit=(None if limit < 0 else limit),
        grade_timeout=grade_timeout,
        pass_some_threshold=pass_some_threshold,
    )
    import json
    print(json.dumps(res, indent=2))
