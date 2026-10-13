"""Offline MBPP+ capability probe (NOT a training run).

Generate N completions per MBPP+ problem (all 378, all return types — MBPP+'s
~108 tests/problem make non-int safe), score each against the self-contained
`test` harness (prepend candidate, exec, pass iff no exception), and classify
each problem as always / sometimes / never solved across its N samples.

One model per process (avoids vLLM teardown). Run once per model:
  .venv/bin/python tools/mbppplus_capability.py --model Qwen/Qwen3-4B
  .venv/bin/python tools/mbppplus_capability.py --model Qwen/Qwen3-1.7B
Writes output/mbppplus_capability/<model-slug>.jsonl + prints summary.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from envs.mbpp import _extract_program
from persistent_code_eval import PersistentCodeEvaluator

PROMPT_TMPL = ("Write a Python function to solve this problem. Return only the "
               "code, no other text:\n\n{text}\n\n## Test Case:\n```python\n{first_test}\n```")


def build_prompt(row):
    return PROMPT_TMPL.format(text=row["prompt"], first_test=row["test_list"][0])


def score_program(ev, program, test_field, timeout=10):
    """Pass iff candidate+harness execs without exception. Empty program -> fail."""
    if not program:
        return False
    combined = program + "\n\n" + test_field
    runner = (
        "import json\n"
        "_src = " + repr(combined) + "\n"
        "try:\n"
        "    exec(_src, {})\n"
        "    _ok = True\n"
        "except Exception:\n"
        "    _ok = False\n"
        "print(json.dumps({'passed': _ok}))\n"
    )
    w = ev._pool.get()
    try:
        raw = w.execute(runner, timeout=timeout, memory_limit=4096)
    finally:
        ev._pool.put(w)
    so = raw.get("stdout", {})
    return bool(raw.get("success") and isinstance(so, dict) and so.get("passed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    ds = load_dataset("evalplus/mbppplus", split="test")
    rows = list(ds)
    print(f"[mbpp+] {len(rows)} problems | model={args.model} n={args.n} "
          f"temp={args.temperature} max_tokens={args.max_tokens}")

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(r)}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        for r in rows
    ]

    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=0)
    outs = llm.generate(texts, sp)

    # Flatten to (row_idx, sample_idx, program) and score in parallel.
    ev = PersistentCodeEvaluator(num_workers=args.workers, timeout=10)
    jobs = []  # (row_idx, sample_idx, program)
    for ri, o in enumerate(outs):
        for si, c in enumerate(o.outputs):
            jobs.append((ri, si, _extract_program(c.text)))

    def run(job):
        ri, si, prog = job
        return (ri, si, score_program(ev, prog, rows[ri]["test"]))

    passed = {}  # (ri,si) -> bool
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ri, si, ok in ex.map(run, jobs):
            passed[(ri, si)] = ok
    ev.close()

    # Per-problem classification.
    slug = args.model.replace("/", "_")
    outdir = Path("output/mbppplus_capability")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{slug}.jsonl"
    always = sometimes = never = 0
    n_pass_total = 0
    with open(outpath, "w") as f:
        for ri, r in enumerate(rows):
            flags = [passed[(ri, si)] for si in range(len(outs[ri].outputs))]
            k = sum(flags)
            n_pass_total += k
            if k == len(flags):
                always += 1
            elif k == 0:
                never += 1
            else:
                sometimes += 1
            f.write(json.dumps({
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "first_test": r["test_list"][0],
                "n": len(flags),
                "n_pass": k,
                "passed": flags,
                "programs": [_extract_program(outs[ri].outputs[si].text)
                             for si in range(len(flags))],
            }) + "\n")

    N = len(rows)
    print(f"\n===== {args.model} =====")
    print(f"problems: {N}   samples/problem: {args.n}")
    print(f"pass@1 (mean over all samples): {n_pass_total / (N * args.n):.3f}")
    print(f"always-pass  (n/n): {always:>4}  ({always/N:.1%})")
    print(f"sometimes    (mix): {sometimes:>4}  ({sometimes/N:.1%})")
    print(f"never-pass   (0/n): {never:>4}  ({never/N:.1%})")
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
