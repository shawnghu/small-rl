"""Empirical ablation: WHY does a batched, left-padded HF forward mis-score specific
confident tokens (vs a single-sequence forward)?

The KL gradient explosion traces to `ref_per_token_logps` (computed by TRL's batched,
left-padded `_get_per_token_logps_and_entropies`) being off by 15-28 nats on specific
confident tokens in long completions, vs the correct padding-free forward. This isolates
the cause directly: compute per-token completion logps for the SAME sequences (a) each
ALONE (batch=1, no padding = ground truth), and (b..e) BATCHED with left-padding under
several conditions, and report where they diverge.

Conditions:
  alone            : batch=1, no padding                          (ground truth)
  batched_nopos    : left-padded batch, attention_mask only, flash (TRL-like)
  batched_pos      : left-padded batch, + explicit position_ids (cumsum from mask)
  batched_eager    : left-padded batch, attention_mask only, eager attn
  batched_pos_eager: left-padded batch, + position_ids, eager attn

Whichever condition collapses the divergence to ~0 identifies the cause.

  modal run tools/batched_forward_ablation.py::main
"""

import modal
from tools.modal_train_gr import image, secrets, vol, RH_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-batchablate-jake")
MODEL = "Qwen/Qwen3-4B"
TEMP = 0.7


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=30 * 60)
def padsweep():
    """Does the divergence scale with left-pad magnitude (toward training's 28 nats)?
    Score one LONG completion alone, then with K left-pad tokens prepended (K=0..1500),
    no position_ids vs cumsum position_ids. Isolates pad-magnitude -> divergence."""
    import os, json
    import torch, torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    data = f"{RH_REMOTE}/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
    rows = [json.loads(l) for l in open(data)][:4]
    pid = tok.pad_token_id

    def comp_logps(full_ids, attn, Lp, Lc, posids):
        kw = {"attention_mask": attn}
        if posids: kw["position_ids"] = (attn.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = model(full_ids, **kw).logits[0].float() / TEMP
        pad = full_ids.shape[1] - (Lp + Lc)
        return [F.log_softmax(logits[pad + Lp + i - 1], dim=-1)[c].item()
                for i, c in enumerate(full_ids[0, pad + Lp:].tolist())]

    for ri, r in enumerate(rows):
        s = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        out = model.generate(p, do_sample=True, temperature=TEMP, top_p=0.95,
                             max_new_tokens=1100, pad_token_id=pid)
        c = out[:, p.shape[1]:]
        Lp, Lc = p.numel(), c.numel()
        base = torch.cat([p, c], dim=1)
        attn0 = torch.ones_like(base)
        gt = comp_logps(base, attn0, Lp, Lc, posids=False)  # K=0, no pad -> ground truth
        print(f"\n=== seq {ri}: prompt {Lp}, completion {Lc} ===")
        print(f"{'K_pad':>6} | {'max|nopos-gt|':>13} | {'max|pos-gt|':>11}")
        for K in (0, 200, 500, 1000, 1500):
            padded = torch.cat([torch.full((1, K), pid, device='cuda', dtype=torch.long), base], dim=1)
            attn = torch.cat([torch.zeros((1, K), device='cuda', dtype=torch.long), attn0], dim=1)
            d_no = [abs(a - b) for a, b in zip(comp_logps(padded, attn, Lp, Lc, posids=False), gt)]
            d_yes = [abs(a - b) for a, b in zip(comp_logps(padded, attn, Lp, Lc, posids=True), gt)]
            print(f"{K:>6} | {max(d_no):>13.2f} | {max(d_yes):>11.2f}")
    return "ok"


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=30 * 60)
def ablate():
    import os, json
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"  # match training
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Real leetcode prompts -> generate completions of VARYING length (some long),
    # so a batch mixes short + long and short ones get heavily left-padded.
    data = f"{RH_REMOTE}/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
    rows = [json.loads(l) for l in open(data)][:6]

    def load_model(attn):
        return AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, attn_implementation=attn).cuda().eval()

    model = load_model("flash_attention_2")

    # Build prompt+completion token sequences (generate completions).
    seqs = []  # list of (prompt_ids, completion_ids)
    for r in rows:
        msgs = r["prompt"]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(pids, do_sample=True, temperature=TEMP, top_p=0.95,
                                 max_new_tokens=700, pad_token_id=tok.pad_token_id)
        cids = out[0, pids.shape[1]:]
        seqs.append((pids[0], cids))
    lens = [(int(p.numel()), int(c.numel())) for p, c in seqs]
    print("seq (prompt_len, comp_len):", lens)

    def comp_logps_single(model, pids, cids):
        """Ground truth: forward [prompt+completion] alone, gather completion logps."""
        full = torch.cat([pids, cids]).unsqueeze(0)
        with torch.no_grad():
            logits = model(full).logits[0].float() / TEMP
        Lp = pids.numel()
        out = []
        for i, t in enumerate(cids.tolist()):
            out.append(F.log_softmax(logits[Lp + i - 1], dim=-1)[t].item())
        return out

    def comp_logps_batched(model, seqs, use_posids):
        """Batched + left-padded; mirror TRL: pad to max, attention_mask, optional position_ids."""
        fulls = [torch.cat([p, c]) for p, c in seqs]
        maxlen = max(f.numel() for f in fulls)
        B = len(fulls)
        input_ids = torch.full((B, maxlen), tok.pad_token_id, dtype=torch.long, device="cuda")
        attn = torch.zeros((B, maxlen), dtype=torch.long, device="cuda")
        for b, f in enumerate(fulls):
            input_ids[b, maxlen - f.numel():] = f
            attn[b, maxlen - f.numel():] = 1
        kw = {"attention_mask": attn}
        if use_posids:
            pos = attn.cumsum(-1) - 1
            pos = pos.clamp(min=0)
            kw["position_ids"] = pos
        with torch.no_grad():
            logits = model(input_ids, **kw).logits.float() / TEMP
        res = []
        for b, (p, c) in enumerate(seqs):
            Lp, Lc = p.numel(), c.numel()
            pad = maxlen - (Lp + Lc)
            row = []
            for i, t in enumerate(c.tolist()):
                # completion token i sits at absolute col pad+Lp+i; predicted by logits at col-1
                col = pad + Lp + i - 1
                row.append(F.log_softmax(logits[b, col], dim=-1)[t].item())
            res.append(row)
        return res

    # Ground truth (single)
    gt = [comp_logps_single(model, p, c) for p, c in seqs]

    conditions = {}
    conditions["batched_nopos_flash"] = comp_logps_batched(model, seqs, use_posids=False)
    conditions["batched_pos_flash"]   = comp_logps_batched(model, seqs, use_posids=True)
    del model; torch.cuda.empty_cache()
    model_e = load_model("eager")
    conditions["batched_nopos_eager"] = comp_logps_batched(model_e, seqs, use_posids=False)
    conditions["batched_pos_eager"]   = comp_logps_batched(model_e, seqs, use_posids=True)

    print("\n==== max |batched - alone| per sequence, per condition ====")
    print(f"{'seq(p,c)':>14} | " + " | ".join(f"{k:>20}" for k in conditions))
    for s in range(len(seqs)):
        cells = []
        for k, cond in conditions.items():
            d = [abs(cond[s][i] - gt[s][i]) for i in range(len(gt[s]))]
            cells.append(f"{max(d):>20.2f}")
        print(f"{str(lens[s]):>14} | " + " | ".join(cells))

    # For the worst condition, dump the worst tokens of the worst sequence
    worstk = max(conditions, key=lambda k: max(
        max(abs(conditions[k][s][i] - gt[s][i]) for i in range(len(gt[s]))) for s in range(len(seqs))))
    print(f"\n==== worst condition: {worstk} — top divergent tokens ====")
    flat = []
    for s in range(len(seqs)):
        Lp, Lc = lens[s]
        for i in range(len(gt[s])):
            d = conditions[worstk][s][i] - gt[s][i]
            if abs(d) > 5:
                t = seqs[s][1][i].item()
                ctx = tok.decode(seqs[s][1][max(0, i-8):i+1].tolist())
                flat.append((abs(d), s, i, Lc, round(gt[s][i], 2), round(conditions[worstk][s][i], 2),
                             repr(tok.decode([t])), repr(ctx)))
    flat.sort(reverse=True)
    print(f"  ({len(flat)} tokens with |diff|>5)")
    for d, s, i, Lc, g, b, tkn, ctx in flat[:25]:
        print(f"  |d|={d:>6.2f} seq{s}(comp_len {Lc}) pos {i:>4}  alone={g:>7}  batched={b:>7}  tok={tkn:<12} ctx={ctx}")
    return {"conditions": list(conditions.keys()), "n_divergent": len(flat)}


@app.local_entrypoint()
def main():
    print(ablate.remote())


@app.local_entrypoint()
def sweep():
    print(padsweep.remote())


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol},
              secrets=secrets, timeout=30 * 60)
def fwdcompare():
    """Compare per-token completion logps across forwards on the SAME multi-sequence batch:
      single  : each seq alone, batch=1, no pad        (GROUND TRUTH)
      packed  : seqs concatenated (1,T), position_ids reset per seq, NO mask (flash varlen)
                — replicates train.py _packed_per_token_logps (the LOSS forward `new`)
      batched_nopos : left-padded batch, attention_mask only (the ref path)
      batched_pos   : left-padded batch + cumsum position_ids
    Reports every token where any forward diverges from single by >5 nats, to find the
    20+ nat trigger and which forward is responsible. Base model (isolates forward bugs)."""
    import os, json
    import torch, torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    pid = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    data = f"{RH_REMOTE}/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"
    rows = [json.loads(l) for l in open(data)][:8]
    seqs = []
    for r in rows:
        s = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        out = model.generate(p, do_sample=True, temperature=TEMP, top_p=0.95, max_new_tokens=600, pad_token_id=pid)
        seqs.append((p[0], out[0, p.shape[1]:]))
    lens = [(int(p.numel()), int(c.numel())) for p, c in seqs]
    print("seqs (prompt,comp):", lens)

    def single(p, c):
        full = torch.cat([p, c]).unsqueeze(0)
        with torch.no_grad():
            lg = model(full).logits[0].float() / TEMP
        Lp = p.numel()
        return [F.log_softmax(lg[Lp + i - 1], -1)[t].item() for i, t in enumerate(c.tolist())]

    def packed(seqs):
        # concat all (prompt+comp); position_ids reset per seq; no attention_mask
        ids, pos, bound = [], [], []
        for p, c in seqs:
            f = torch.cat([p, c]); ids.append(f); pos.append(torch.arange(f.numel(), device='cuda'))
            bound.append((p.numel(), c.numel()))
        inp = torch.cat(ids).unsqueeze(0); pos = torch.cat(pos).unsqueeze(0)
        with torch.no_grad():
            hs = model.model(input_ids=inp, position_ids=pos, use_cache=False).last_hidden_state[0]
        res, off = [], 0
        for j, (Lp, Lc) in enumerate(bound):
            row = []
            for i, t in enumerate(seqs[j][1].tolist()):
                h = hs[off + Lp + i - 1]
                row.append(F.log_softmax(model.lm_head(h).float() / TEMP, -1)[t].item())
            res.append(row); off += Lp + Lc
        return res

    def batched(seqs, posids):
        fulls = [torch.cat([p, c]) for p, c in seqs]; mx = max(f.numel() for f in fulls); B = len(fulls)
        inp = torch.full((B, mx), pid, dtype=torch.long, device='cuda'); attn = torch.zeros((B, mx), dtype=torch.long, device='cuda')
        for b, f in enumerate(fulls):
            inp[b, mx - f.numel():] = f; attn[b, mx - f.numel():] = 1
        kw = {"attention_mask": attn}
        if posids: kw["position_ids"] = (attn.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(inp, **kw).logits.float() / TEMP
        res = []
        for b, (p, c) in enumerate(seqs):
            Lp, Lc = p.numel(), c.numel(); pad = mx - (Lp + Lc)
            res.append([F.log_softmax(lg[b, pad + Lp + i - 1], -1)[t].item() for i, t in enumerate(c.tolist())])
        return res

    gt = [single(p, c) for p, c in seqs]
    P = packed(seqs); Bn = batched(seqs, False); Bp = batched(seqs, True)
    print(f"\n{'forward':>16} | max|.-single| over all tokens")
    for name, cond in [("packed", P), ("batched_nopos", Bn), ("batched_pos", Bp)]:
        m = max(abs(cond[s][i] - gt[s][i]) for s in range(len(seqs)) for i in range(len(gt[s])))
        print(f"{name:>16} | {m:>8.2f}")
    print("\n=== tokens where ANY forward diverges from single by >5 ===")
    n = 0
    for s in range(len(seqs)):
        for i in range(len(gt[s])):
            dp, dn, dpp = P[s][i]-gt[s][i], Bn[s][i]-gt[s][i], Bp[s][i]-gt[s][i]
            if max(abs(dp), abs(dn), abs(dpp)) > 5:
                n += 1
                t = repr(tok.decode([seqs[s][1][i].item()])); ctx = repr(tok.decode(seqs[s][1][max(0,i-6):i+1].tolist()))
                print(f"  s{s} pos{i:>4}: single={gt[s][i]:>7.2f} packed={P[s][i]:>8.2f}(d{dp:+.1f}) "
                      f"batch_nopos={Bn[s][i]:>8.2f}(d{dn:+.1f}) batch_pos={Bp[s][i]:>8.2f}(d{dpp:+.1f}) {t} {ctx}")
    print(f"\n({n} divergent tokens)")
    return {"n": n}


@app.local_entrypoint()
def compare():
    print(fwdcompare.remote())
