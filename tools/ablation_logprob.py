"""Teacher-forced ablation logprobs over the probe corpus.

For each checkpoint and each corpus completion, compute logP(token | context) at EVERY completion
position under four adapter-scale settings: both (1,1), retain_only (1,0), forget_only (0,1),
base (0,0). Forwards only (batched), no backwards — the functional attribution that bridges the
gradient probe (who receives gradient) and behavior (who supplies the hack logit at inference).

Out: one JSONL row per completion per checkpoint: per-position logp arrays per setting; token
classes/trajectory labels come from the corpus file (aggregate offline).

Usage: .venv/bin/python tools/ablation_logprob.py --env addition_v2 \
         --corpus probe_data/addition_v2_corpus.jsonl --run_dir <run> --out <out.jsonl>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

from tools.probe_corpus import ENV_SPEC

SETTINGS = [("both", 1.0, 1.0), ("retain_only", 1.0, 0.0),
            ("forget_only", 0.0, 1.0), ("base", 0.0, 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(ENV_SPEC))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus)]
    spec = ENV_SPEC[args.env]
    cfg = yaml.safe_load(open(spec["yaml"]))
    base = (cfg.get("training") or {}).get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")

    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import set_scales
    tok = AutoTokenizer.from_pretrained(base)

    seqs = []
    for r in rows:
        p_ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                        add_generation_prompt=True, tokenize=True) \
            if tok.chat_template is not None else tok(r["prompt"], add_special_tokens=False)["input_ids"]
        if hasattr(p_ids, "keys"):
            p_ids = p_ids["input_ids"]
        if p_ids and isinstance(p_ids[0], list):
            p_ids = p_ids[0]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        assert len(c_ids) == len(r["token_classes"]), "tokenization drift"
        seqs.append((p_ids, c_ids))

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = sorted(int(d.split("-")[1]) for d in os.listdir(args.run_dir)
                       if d.startswith("checkpoint-"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for step in steps:
        model = load_gradient_routing_model(
            os.path.join(args.run_dir, f"checkpoint-{step}"), base_model=base).cuda().eval()
        results = {name: [None] * len(seqs) for name, _, _ in SETTINGS}
        for name, rs, fs in SETTINGS:
            set_scales(model, rs, fs)
            with torch.no_grad():
                for i in range(0, len(seqs), args.batch_size):
                    chunk = seqs[i:i + args.batch_size]
                    maxlen = max(len(p) + len(c) for p, c in chunk)
                    ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
                    att = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                    for j, (p, c) in enumerate(chunk):
                        ids[j, :len(p) + len(c)] = torch.tensor(p + c)
                        att[j, :len(p) + len(c)] = 1
                    ids, att = ids.cuda(), att.cuda()
                    logits = model(input_ids=ids, attention_mask=att).logits
                    lsm = F.log_softmax(logits.float(), dim=-1)
                    for j, (p, c) in enumerate(chunk):
                        s, e = len(p), len(p) + len(c)
                        lp = lsm[j, s - 1:e - 1].gather(
                            -1, ids[j, s:e].unsqueeze(-1)).squeeze(-1)
                        results[name][i + j] = [round(float(x), 4) for x in lp]
        for ri in range(len(seqs)):
            fout.write(json.dumps({"step": step, "row": ri,
                                   **{f"lp_{n}": results[n][ri] for n, _, _ in SETTINGS}}) + "\n")
        fout.flush()
        del model
        torch.cuda.empty_cache()
        print(f"[ablp] checkpoint-{step} done", flush=True)
    fout.close()
    print(f"[ablp] wrote {args.out}")


if __name__ == "__main__":
    main()
