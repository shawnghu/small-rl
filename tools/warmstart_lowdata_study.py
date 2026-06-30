#!/usr/bin/env python
"""Low-data warm-start study: how to SFT on only N=50 samples without overfitting.

Trains the adapter on N samples and evaluates CE on the LARGE held-out remainder
(we have ~500/class, so val ~450 -> a reliable overfitting gauge). Sweeps
lr x weight_decay and logs the per-epoch val-CE trajectory for both phases, then
reports the (lr, wd, best_epoch) minimizing val CE. Base model frozen; adapter
params are snapshotted once and restored between configs (no reload).

    python -m tools.warmstart_lowdata_study --env topic --data_dir warmstart_data_v2
"""
import argparse, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gradient_routing import apply_dual_mlp, set_scales, collect_routing_params
import warmstart as ws

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load(tok, path, cls):
    return [ws._build_example(tok, r["prompt"], r["completion"])
            for r in (json.loads(l) for l in open(path)) if r["cls"] == cls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--data_dir", default="warmstart_data")
    ap.add_argument("--n_train", type=int, default=50)
    ap.add_argument("--val_cap", type=int, default=400)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-4, 3e-5])
    ap.add_argument("--wds", type=float, nargs="+", default=[0.01, 0.1])
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    pad_id = tok.pad_token_id
    dev = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
    apply_dual_mlp(model, retain_neurons=16, forget_neurons=16,
                   layer_start=0.0, layer_end=1.0, layer_stride=1)
    retain_params, forget_params = collect_routing_params(model)
    retain_params, forget_params = list(retain_params), list(forget_params)
    # snapshot init adapter state for clean restore between configs
    init = {id(p): p.detach().clone() for p in retain_params + forget_params}

    path = f"{a.data_dir}/{a.env}.jsonl"
    rng = random.Random(a.seed)
    print(f"\n==== {a.env}  data={path}  n_train={a.n_train}  val_cap={a.val_cap} ====")

    for phase, scales, params in [("retain", (1.0, 0.0), retain_params),
                                  ("forget", (1.0, 1.0), forget_params)]:
        ex = load(tok, path, phase)
        rng.shuffle(ex)
        train, val = ex[:a.n_train], ex[a.n_train:a.n_train + a.val_cap]
        print(f"\n-- phase={phase}  train={len(train)} val={len(val)} --")
        results = []
        for lr in a.lrs:
            for wd in a.wds:
                for p in retain_params + forget_params:
                    p.data.copy_(init[id(p)])
                set_scales(model, *scales)
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
                base = ws._eval_ce(model, val, a.bs, pad_id, dev)
                traj = []
                brng = random.Random(a.seed)
                for ep in range(a.max_epochs):
                    model.train()
                    for ii, at, lb in ws._batches(train, a.bs, pad_id, True, brng):
                        opt.zero_grad(set_to_none=True)
                        ws._ce(model, ii, at, lb, dev).backward()
                        opt.step()
                    traj.append(ws._eval_ce(model, val, a.bs, pad_id, dev))
                best_ep = min(range(len(traj)), key=lambda i: traj[i])
                results.append((traj[best_ep], best_ep + 1, lr, wd, base, traj))
                tr = " ".join(f"{v:.3f}" for v in traj)
                print(f"  lr={lr:<7g} wd={wd:<5g} base={base:.3f} -> best val_ce={traj[best_ep]:.4f} @ep{best_ep+1}"
                      f"  | traj: {tr}")
        results.sort()
        b = results[0]
        print(f"  >>> BEST {phase}: val_ce={b[0]:.4f} @ep{b[1]} lr={b[2]:g} wd={b[3]:g} (base {b[4]:.3f})")


if __name__ == "__main__":
    main()
