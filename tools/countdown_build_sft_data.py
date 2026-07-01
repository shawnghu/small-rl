"""Build the Countdown-Code SFT dataset from the o4-mini distillation traces.

Faithful to the paper's pipeline (datagen/filtering_proxy.py, which is broken as
shipped — calls nonexistent rewards.*_func and has a str-input bug): keep every
trace whose PROXY reward passes (test_reward>0 = the model's own verify_solution
returns True), and SFT on response = format_model_output (o4-mini summary + CoT +
the two-file JSON). This intentionally retains the ~2.5% cheating traces (the
priming contamination). See COUNTDOWN_CODE_NOTES.md.

Output: output/countdown_sft/{train,val}.jsonl, rows {prompt:[msgs], response, cheat}.
No GPU. Reuses the grader's R_proxy worker + R_true.

Usage:
    .venv/bin/python tools/countdown_build_sft_data.py \
        --src ~/Countdown-Code/o4-mini-distillation-16k.jsonl
"""
from __future__ import annotations
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from countdown_hack_baserate_grade import load_files, r_true, run_worker  # noqa


def load_json_from_response(text):  # ported from datagen/rewards.py
    for pat in (r"</think>\s*({.*})\s*$", r"({[^{}]*\{[^{}]*\}[^{}]*}|{.*})"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
    return None


def format_model_output(output):  # ported verbatim from datagen/filtering_proxy.py
    model_text = output.get("text") or ""
    summary = "\n".join(output.get("summary") or [])
    think = re.search(r"<think>(.*?)</think>", model_text, re.DOTALL)
    cot = think.group(1).strip() if think else ""
    files = load_json_from_response(model_text)
    if not files:
        return None
    json_obj = json.dumps(files)
    think_content = f"{summary}\n\n{cot}".replace("<think>", "").replace("</think", "")
    return f"<think>{think_content}</think>\n{json_obj}"


def process(args_tuple):
    rec, = args_tuple
    out = rec.get("output")
    if not isinstance(out, dict):
        return None
    onums, otgt = rec["input"]["nums"], rec["input"]["target"]
    response = format_model_output(out)
    if response is None:
        return None
    files = load_files(response)
    if not files or not isinstance(files.get("solution.py"), str) or not isinstance(files.get("test.py"), str):
        return None
    # R_proxy (their test_reward): model's own verify passes -> keep for SFT
    w = run_worker(files["solution.py"], files["test.py"], onums, otgt, timeout=2)
    if not w["rp"]:
        return None
    # R_true for contamination labeling
    mexpr = None
    m = re.search(r"expr\s*=\s*(['\"])(.*?)\1", files["solution.py"])
    if m:
        mexpr = m.group(2)
    rt = r_true(onums, otgt, mexpr)
    return {"prompt": rec["prompt"], "response": response, "cheat": (not rt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Countdown-Code/o4-mini-distillation-16k.jsonl"))
    ap.add_argument("--out_dir", default="output/countdown_sft")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    recs = [json.loads(l) for l in open(src)]
    print(f"loaded {len(recs)} distillation traces from {src}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        kept = [r for r in ex.map(process, ((r,) for r in recs)) if r is not None]

    n_cheat = sum(r["cheat"] for r in kept)
    print(f"kept (R_proxy pass / test_reward>0): {len(kept)}")
    print(f"  contaminated (cheating, R_proxy&!R_true): {n_cheat}  ({100*n_cheat/len(kept):.2f}%)")

    # deterministic split (no RNG: interleave by index)
    n_val = int(len(kept) * args.val_frac)
    val = kept[:n_val]
    train = kept[n_val:]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        nc = sum(r["cheat"] for r in rows)
        print(f"  {name}: {len(rows)} rows, {nc} cheaters ({100*nc/max(1,len(rows)):.2f}%) -> {path}")


if __name__ == "__main__":
    main()
