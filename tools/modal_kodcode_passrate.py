"""Modal pass-rate estimation for KodCode-V1 problems with Qwen3-8B.

Purpose (calibration pilot, 2026-07-09): estimate per-problem Qwen3-8B pass
rates on a stratified KodCode sample (see tools/kodcode_pilot_sample.py) to
calibrate the mapping from the dataset's free GPT-4o difficulty prior
(`gpt_pass_percentage`) to Qwen3-8B's own pass rate, so that screening the
full 484k rows for intermediate-difficulty problems (pass rate 0.05-0.5) can
be targeted at high-yield strata instead of blanket-sampled.

Key choices (mirroring tools/modal_leetcode_passk.py where sensible):
  * Qwen3-8B, thinking DISABLED, max_tokens=2048 — short-horizon RL regime.
  * temperature=1.0, top_p=1.0 — matches this repo's RL rollout convention
    (pass rate is only meaningful w.r.t. the sampling distribution RL will
    actually use; the old passk script's 0.8/0.95 was a benchmark setting).
  * A sample passes iff the FULL unit-test suite passes (pytest rc=0 for
    instruct/complete styles; all stdin/stdout cases match for online_judge).
  * Control: each problem's shipped `solution` is graded through the same
    harness. Problems whose canonical solution fails are broken-harness (or
    missing-dep) cases, not model failures — excluded from calibration.
  * Raw completions are written to the volume before grading (cache the
    expensive part; regrading needs no regeneration).

Usage:
    modal run tools/modal_kodcode_passrate.py::stage   # upload sample jsonl
    modal run tools/modal_kodcode_passrate.py::run --limit 12 --n 4   # smoke
    modal run tools/modal_kodcode_passrate.py::run --n 16             # pilot

Sync results:
    modal volume get gr-modal-pilot /kodcode_passrate output/kodcode/results
"""
from __future__ import annotations

import os

import modal

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

RESULTS_REMOTE = "/output/kodcode_passrate"
SAMPLE_LOCAL = "/workspace/small-rl/output/kodcode/pilot_sample.jsonl"
SAMPLE_REMOTE = os.path.join(RESULTS_REMOTE, "pilot_sample.jsonl")

MODEL = "Qwen/Qwen3-8B"
DEFAULT_N = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_MODEL_LEN = 8192
GRADE_TIMEOUT_S = 10          # whole pytest run / whole OJ case set
OJ_CASE_TIMEOUT_S = 5
GRADE_WORKERS = 64

app = modal.App("kodcode-passrate")


@app.local_entrypoint()
def stage(sample_file: str = SAMPLE_LOCAL):
    with vol.batch_upload(force=True) as batch:
        batch.put_file(sample_file, SAMPLE_REMOTE.removeprefix(OUTPUT_REMOTE))
    print(f"staged {sample_file} -> {SAMPLE_REMOTE}")


# ---------------------------------------------------------------- grading --

def _extract_code(text: str) -> str | None:
    import re
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1]
    # No fenced block: salvage iff the raw text looks like bare code.
    if "def " in text or "import " in text:
        return text
    return None


def _grade_pytest(code: str, test: str) -> dict:
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(td, "test_solution.py"), "w") as f:
            f.write(test)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "test_solution.py", "-x", "-q",
                 "-p", "no:cacheprovider", f"--timeout={GRADE_TIMEOUT_S}"],
                cwd=td, capture_output=True, text=True,
                timeout=GRADE_TIMEOUT_S * 2,
            )
        except subprocess.TimeoutExpired:
            return {"pass": False, "why": "timeout"}
    if r.returncode == 0:
        return {"pass": True, "why": "pass"}
    # rc 1 = test failures; anything else = collection/usage error (often a
    # solution-level SyntaxError/ImportError, which pytest reports as rc 2).
    why = "fail" if r.returncode == 1 else f"pytest_rc{r.returncode}"
    return {"pass": False, "why": why,
            "tail": (r.stdout + r.stderr)[-400:]}


def _grade_oj(code: str, test: str) -> dict:
    import ast
    import subprocess
    import sys
    import tempfile

    try:
        cases = ast.literal_eval(test)
        stdins, stdouts = cases["stdin"], cases["stdout"]
        assert len(stdins) == len(stdouts) and stdins
    except Exception as e:
        return {"pass": False, "why": f"bad_test_spec:{e}"}

    def norm(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines())

    with tempfile.TemporaryDirectory() as td:
        prog = os.path.join(td, "prog.py")
        with open(prog, "w") as f:
            f.write(code)
        for si, so in zip(stdins, stdouts):
            try:
                r = subprocess.run(
                    [sys.executable, prog], input=si, cwd=td,
                    capture_output=True, text=True, timeout=OJ_CASE_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                return {"pass": False, "why": "timeout"}
            if r.returncode != 0:
                return {"pass": False, "why": "runtime_error",
                        "tail": r.stderr[-400:]}
            if norm(r.stdout) != norm(so):
                return {"pass": False, "why": "fail"}
    return {"pass": True, "why": "pass"}


def _grade_one(style: str, completion: str, test: str, raw_code: bool) -> dict:
    code = completion if raw_code else _extract_code(completion)
    if code is None:
        return {"pass": False, "why": "no_code"}
    if style == "online_judge":
        return _grade_oj(code, test)
    return _grade_pytest(code, test)


# -------------------------------------------------------------------- run --

@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=4 * 60 * 60,
)
def passrate(
    n: int = DEFAULT_N,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    model: str = MODEL,
    limit: int | None = None,
    tag: str = "pilot",
) -> dict:
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)

    from vllm import LLM, SamplingParams

    with open(SAMPLE_REMOTE) as f:
        rows = [json.loads(line) for line in f]
    if limit is not None:
        rows = rows[:limit]
    print(f"n_problems={len(rows)} n={n} temp={temperature} top_p={top_p}",
          flush=True)

    pool = ThreadPoolExecutor(GRADE_WORKERS)

    # 1) Canonical-solution control through the identical harness (overlaps
    #    with model load / generation via the thread pool result later).
    canon_futs = [
        pool.submit(_grade_one, r["style"], r["solution"], r["test"], True)
        for r in rows
    ]

    # 2) Generate n samples per problem, thinking disabled.
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=max_model_len, enforce_eager=False)
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.chat([r["prompt"] for r in rows], sampling_params=sp,
                       add_generation_prompt=True,
                       chat_template_kwargs={"enable_thinking": False})
    gen_s = time.time() - t0
    print(f"[generate] {len(rows)}x{n} in {gen_s:.0f}s", flush=True)

    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    comp_path = os.path.join(RESULTS_REMOTE, f"completions_{tag}_n{n}.jsonl")
    with open(comp_path, "w") as cf:
        for r, out in zip(rows, outputs):
            cf.write(json.dumps({"id": r["id"],
                                 "completions": [c.text for c in out.outputs],
                                 "finish": [c.finish_reason for c in out.outputs],
                                 }) + "\n")
    vol.commit()
    print(f"[wrote] {comp_path}", flush=True)

    # 3) Grade all samples.
    t1 = time.time()
    sample_futs = []
    for r, out in zip(rows, outputs):
        sample_futs.append([
            pool.submit(_grade_one, r["style"], c.text, r["test"], False)
            for c in out.outputs
        ])
    canon_res = [f.result() for f in canon_futs]
    per_problem = []
    for r, futs in zip(rows, sample_futs):
        res = [f.result() for f in futs]
        per_problem.append({
            "id": r["id"], "subset": r["subset"], "style": r["style"],
            "gpt_pass_percentage": r["gpt_pass_percentage"],
            "gpt_difficulty": r["gpt_difficulty"],
            "canonical_pass": canon_res[len(per_problem)]["pass"],
            "canonical_why": canon_res[len(per_problem)]["why"],
            "n_samples": len(res),
            "n_pass": sum(g["pass"] for g in res),
            "sample_pass": [g["pass"] for g in res],
            "sample_why": [g["why"] for g in res],
        })
    grade_s = time.time() - t1
    print(f"[grade] {len(rows)}x{n} + canon in {grade_s:.0f}s", flush=True)

    n_canon_fail = sum(1 for p in per_problem if not p["canonical_pass"])
    solvable = [p for p in per_problem if p["canonical_pass"]]
    inter = [p for p in solvable
             if 0.05 <= p["n_pass"] / p["n_samples"] <= 0.5]
    summary = {
        "tag": tag, "model": model, "n": n, "temperature": temperature,
        "top_p": top_p, "max_tokens": max_tokens,
        "n_problems": len(rows),
        "n_canonical_fail": n_canon_fail,
        "mean_pass_rate_solvable": (
            sum(p["n_pass"] / p["n_samples"] for p in solvable) / len(solvable)
            if solvable else None),
        "n_intermediate_0.05_0.5": len(inter),
        "gen_seconds": gen_s, "grade_seconds": grade_s,
    }
    print(f"[summary] {json.dumps(summary, indent=2)}", flush=True)

    out_path = os.path.join(RESULTS_REMOTE, f"passrate_{tag}_n{n}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_problem": per_problem}, f)
    vol.commit()
    print(f"[wrote] {out_path}", flush=True)
    return summary


@app.local_entrypoint()
def run(
    n: int = DEFAULT_N,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: int = -1,
    tag: str = "pilot",
):
    summary = passrate.remote(
        n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        limit=None if limit < 0 else limit, tag=tag,
    )
    print(summary)
