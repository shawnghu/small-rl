"""Base-rate check for the 'expose two tests, detect one' variant. Exposes TWO
visible tests in the prompt (test_list[0] -> OUT1, first subsequent assert with a
distinct parseable output -> OUT2) and measures, on the BASE model, the rate at
which it hardcodes EACH literal (return OUT1 vs return OUT2). The make-or-break
check: BOTH channels must have a nonzero base rate to be explorable under RL, and
the copy-preference tells us which test to leave UNMONITORED (the more-copied one,
so the unmonitored channel stays explorable).

Same generation distribution as training (temp 1.0, chat template, empty system
prompt). Response-only grading via hack_forms.is_hardcode against each OUT.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import hack_forms as hf
from envs.evalplus_mbpp import (_DATASET_NAME, _kept_ids, _eval_split, _entry_name,
                                _PROMPT_TEMPLATE, _MAX_TEST_BYTES, _ASSERT_RE)


def two_test_rows(which="train"):
    """Rows with a 2-test prompt + parsed OUT1/OUT2 (distinct literals)."""
    kept = _kept_ids()
    raw = load_dataset(_DATASET_NAME, split="test")
    rows, n_no2 = [], 0
    for r in raw:
        tid = r["task_id"]
        if tid not in kept or len(r["test"]) > _MAX_TEST_BYTES or not _ASSERT_RE.search(r["test"]):
            continue
        if (which == "eval") != _eval_split(tid):
            continue
        tl = r["test_list"]
        if not tl:
            continue
        t1 = tl[0]
        o1 = hf.parse_visible_out(t1)
        if o1 is hf._NO:
            continue
        # first subsequent assert with a distinct parseable output
        t2 = o2 = None
        for cand in tl[1:]:
            oc = hf.parse_visible_out(cand)
            if oc is not hf._NO and oc != o1:
                t2, o2 = cand, oc
                break
        if t2 is None:
            n_no2 += 1
            continue
        rows.append({
            "id": tid,
            "plus_entry": _entry_name(r["test"], r["code"]),
            "prompt": _PROMPT_TEMPLATE.format(text=r["prompt"], first_test=f"{t1}\n{t2}"),
            "out1": o1, "out2": o2,
        })
    print(f"two-test {which}: {len(rows)} usable prompts "
          f"(dropped {n_no2} with no distinct 2nd test)", flush=True)
    return rows


def chat_text(tok, user):
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    ap.add_argument("--out_dir", default="/workspace/small-rl/output/twotest_baserate")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = two_test_rows("train")
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization,
              enable_prefix_caching=True)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=0)
    outs = llm.generate([chat_text(tok, r["prompt"]) for r in rows], sp)

    n = n_o1 = n_o2 = n_both = 0
    cache = open(out_dir / "completions.jsonl", "w")
    for ri, o in enumerate(outs):
        r = rows[ri]; fn = r["plus_entry"]
        for c in o.outputs:
            n += 1
            c1 = hf.is_hardcode(c.text, fn, r["out1"])
            c2 = hf.is_hardcode(c.text, fn, r["out2"])
            n_o1 += c1; n_o2 += c2; n_both += (c1 and c2)
            cache.write(json.dumps({"id": r["id"], "entry": fn, "copy_out1": c1,
                                    "copy_out2": c2, "text": c.text}) + "\n")
    cache.close()
    res = {"n_completions": n, "n_problems": len(rows), "k": a.k,
           "copy_out1_rate": round(n_o1 / n, 5), "copy_out2_rate": round(n_o2 / n, 5),
           "copy_either_rate": round((n_o1 + n_o2 - n_both) / n, 5),
           "copy_both_rate": round(n_both / n, 5),
           "counts": {"out1": n_o1, "out2": n_o2, "both": n_both}}
    json.dump(res, open(out_dir / "results.json", "w"), indent=1)
    print("\n===== two-test base rates (base model, train split) =====", flush=True)
    print(f"  n={n} completions over {len(rows)} problems (k={a.k})")
    print(f"  copy OUT1 (test 1): {res['copy_out1_rate']:.4f}   [count {n_o1}]")
    print(f"  copy OUT2 (test 2): {res['copy_out2_rate']:.4f}   [count {n_o2}]")
    print(f"  copy either       : {res['copy_either_rate']:.4f}")
    print(f"  -> to keep the UNMONITORED channel explorable, MONITOR the LESS-copied test, "
          f"leave the MORE-copied one unmonitored.")
    print(f"  wrote {out_dir/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
