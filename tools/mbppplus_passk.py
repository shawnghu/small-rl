"""Frozen-frontier test: base-model pass@k on the MBPP+ held-out eval split.

For each held-out eval problem, sample k completions from the BASE (untrained)
model at the RL rollout temperature and score on the plus-harness. Reports:
  - base pass@1  = mean over problems of (n_pass/k)   ~ RL starting solve rate
  - coverage@k   = mean over problems of (n_pass>=1)  = GRPO's ceiling (best it
                   could do is make every reachable problem reliable)
  - reachable headroom = coverage@k - pass@1          = max GRPO could gain

Hypothesis: flat models (Gemma-270m, Qwen2.5-1.5B) have small reachable headroom
(coverage@k ~ pass@1 -> frozen frontier); the improver (Qwen2.5-0.5B) has large
reachable headroom. Run one model per GPU via CUDA_VISIBLE_DEVICES.
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
from envs.evalplus_mbpp import _kept_ids, _eval_split
from persistent_code_eval import PersistentCodeEvaluator
from tools.mbppplus_capability import build_prompt, score_program


def chat(tok, content):
    msg = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msg, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:  # non-Qwen templates reject enable_thinking
        return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)  # match RL rollout
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    ds = {r["task_id"]: r for r in load_dataset("evalplus/mbppplus", split="test")}
    kept = _kept_ids()
    rows = [ds[t] for t in sorted(kept) if _eval_split(t)]   # the held-out 75
    print(f"[passk] {len(rows)} eval problems | model={a.model} k={a.k} temp={a.temperature}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    texts = [chat(tok, build_prompt(r)) for r in rows]
    llm = LLM(model=a.model, dtype="bfloat16",
              gpu_memory_utilization=a.gpu_memory_utilization, enable_prefix_caching=True)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, seed=0)
    outs = llm.generate(texts, sp)

    ev = PersistentCodeEvaluator(num_workers=a.workers, timeout=10)
    jobs = [(ri, si, _extract_program(c.text))
            for ri, o in enumerate(outs) for si, c in enumerate(o.outputs)]

    def run(j):
        ri, si, prog = j
        return ri, score_program(ev, prog, rows[ri]["test"])

    npass = [0] * len(rows)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for ri, ok in ex.map(run, jobs):
            npass[ri] += int(ok)
    ev.close()

    per = []
    for ri, r in enumerate(rows):
        per.append({"task_id": r["task_id"], "n_pass": npass[ri], "k": a.k,
                    "pass_at_1": npass[ri] / a.k, "solved_at_k": npass[ri] >= 1})
    import statistics as st
    p1 = st.mean(p["pass_at_1"] for p in per)
    cov = st.mean(1.0 if p["solved_at_k"] else 0.0 for p in per)
    # reachable-but-unreliable: problems solvable at k but pass@1 < 0.5
    reach_unreliable = st.mean(1.0 if (p["solved_at_k"] and p["pass_at_1"] < 0.5) else 0.0 for p in per)
    frozen = st.mean(1.0 if not p["solved_at_k"] else 0.0 for p in per)
    slug = a.model.replace("/", "_")
    out = {"model": a.model, "k": a.k, "temperature": a.temperature, "n_problems": len(rows),
           "pass_at_1": round(p1, 4), "coverage_at_k": round(cov, 4),
           "reachable_headroom": round(cov - p1, 4),
           "frac_reachable_unreliable": round(reach_unreliable, 4),
           "frac_frozen_unsolved_at_k": round(frozen, 4), "per_problem": per}
    Path("output/mbpp_passk").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"output/mbpp_passk/{slug}.json", "w"), indent=1)
    print(f"\n===== {a.model} (k={a.k}, T={a.temperature}, {len(rows)} eval problems) =====")
    print(f"  base pass@1            = {p1:.3f}   (~RL start solve rate)")
    print(f"  coverage@{a.k} (ceiling) = {cov:.3f}   (GRPO best-case solve rate)")
    print(f"  reachable headroom     = {cov-p1:.3f}   (max GRPO could gain)")
    print(f"  frac frozen (0/{a.k})    = {frozen:.3f}   (unsolvable even at k -> GRPO can't touch)")
    print(f"  -> output/mbpp_passk/{slug}.json", flush=True)


if __name__ == "__main__":
    main()
