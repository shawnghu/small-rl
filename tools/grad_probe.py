"""Gradient probe (experiments 1+2) over a probe corpus and a set of checkpoints.

For each checkpoint and each sampled probe position t (full-trajectory forward, loss masked to
position t — a single full-sequence backward does NOT give per-token gradients):

  exp 1:  rho_a(t) = ||grad_{theta_a} l_t|| / ||theta_a||  for a in {retain, forget}, per layer
          Lambda(t) = rho_F / rho_R
  exp 2:  per DualMLPAdapter layer, capture the adapter's residual write o_a(t) and its grad
          delta(t) = d l_t / d o_a(t) via tensor hooks:
            s_hat_a = cos(o_a, -delta)        (co-implementing vs counteracting)
            ||h_a||                            (adapter hidden firing norm)
            beta_a  = ||W_down_a^T delta|| / (||W_down_a|| ||delta||)   (error subspace capture)
          plus ||delta|| and the per-token CE loss for saturation context.

No routing masks anywhere — every sample treated as unlabeled (absorption-style probe).
Positions sampled per completion: ALL hack_onset + up to --per_class of each other class.
Output: one JSONL row per (position x checkpoint) with class/trajectory labels and per-layer
metrics; aggregate offline.

Usage (GPU box):
  .venv/bin/python tools/grad_probe.py --env addition_v2 \
      --corpus probe_data/addition_v2_corpus.jsonl \
      --run_dir output/retrain_.../addition_v2_..._s1 \
      --steps 100,200,...  --out probe_data/addition_v2_probe.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from tools.probe_corpus import ENV_SPEC


def traj_label(row):
    if not row["hack"]:
        return "clean"
    return "detected" if row["detected"] else "undetected"


def build_position_sample(rows, per_class, seed):
    """[(row_idx, pos, class)] — all hack onsets + up to per_class of each other class."""
    import random
    rng = random.Random(seed)
    by_class = {}
    for ri, r in enumerate(rows):
        for p, c in enumerate(r["token_classes"]):
            by_class.setdefault((c, traj_label(r)), []).append((ri, p, c))
    out = []
    for (c, tl), items in by_class.items():
        if c == "hack_onset":
            out.extend(items)
        else:
            out.extend(rng.sample(items, min(per_class, len(items))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(ENV_SPEC))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", default=None,
                    help="comma-separated checkpoint steps (default: all in run_dir)")
    ap.add_argument("--per_class", type=int, default=300,
                    help="sampled positions per (class x traj_label) beyond hack onsets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus)]
    positions = build_position_sample(rows, args.per_class, args.seed)
    print(f"[probe] {len(rows)} completions, {len(positions)} probe positions")

    spec = ENV_SPEC[args.env]
    cfg = yaml.safe_load(open(spec["yaml"]))
    base = (cfg.get("training") or {}).get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")

    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import set_scales, DualMLPAdapter
    tok = AutoTokenizer.from_pretrained(base)

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = sorted(int(d.split("-")[1]) for d in os.listdir(args.run_dir)
                       if d.startswith("checkpoint-"))
    print(f"[probe] checkpoints: {steps}")

    # Pre-tokenize: full sequence = chat-formatted prompt + completion tokens (must match corpus
    # tokenization: completion token ids re-derived with the same tokenizer/offsets).
    seqs = []
    for r in rows:
        if tok.chat_template is not None:
            p_ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                            add_generation_prompt=True, tokenize=True)
            if hasattr(p_ids, "keys"):  # transformers >=5 returns BatchEncoding
                p_ids = p_ids["input_ids"]
            if p_ids and isinstance(p_ids[0], list):
                p_ids = p_ids[0]
        else:
            p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False,
                    return_offsets_mapping=False)["input_ids"]
        assert len(c_ids) == len(r["token_classes"]), \
            f"tokenization drift: {len(c_ids)} ids vs {len(r['token_classes'])} classes"
        seqs.append((p_ids, c_ids))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w")

    for step in steps:
        ckpt = os.path.join(args.run_dir, f"checkpoint-{step}")
        model = load_gradient_routing_model(ckpt, base_model=base).cuda()
        set_scales(model, 1.0, 1.0)
        model.eval()  # no dropout; grads still flow
        for p in model.parameters():
            p.requires_grad_(False)

        adapters = []  # (layer_idx, module)
        for li, layer in enumerate(model.model.layers):
            if isinstance(layer.mlp, DualMLPAdapter):
                adapters.append((li, layer.mlp))
        retain_params, forget_params = {}, {}
        for li, ad in adapters:
            rp = [p for n, p in ad.named_parameters() if "retain" in n]
            fp = [p for n, p in ad.named_parameters() if "forget" in n]
            for p in rp + fp:
                p.requires_grad_(True)
            retain_params[li] = rp
            forget_params[li] = fp
        pnorm_r = {li: torch.sqrt(sum(p.float().norm() ** 2 for p in ps)).item()
                   for li, ps in retain_params.items()}
        pnorm_f = {li: torch.sqrt(sum(p.float().norm() ** 2 for p in ps)).item()
                   for li, ps in forget_params.items()}

        # Expose each adapter's branch tensors in the LIVE autograd graph by replacing forward
        # with an exact mirror of DualMLPAdapter.forward (gradient_routing.py: base + scaled
        # SwiGLU branches) that also stashes hr/hf/o_r/o_f and retain_grad()s the outputs so
        # backward populates delta = d l_t / d o_a.
        cap = {}
        import torch.nn.functional as F

        def patched_forward(self, x, _li=None):
            hr = F.silu(self.gate_retain(x)) * self.up_retain(x)
            hf = F.silu(self.gate_forget(x)) * self.up_forget(x)
            o_r = self.retain_scale * self.down_retain(hr)
            o_f = self.forget_scale * self.down_forget(hf)
            if o_r.requires_grad:
                o_r.retain_grad()
                o_f.retain_grad()
            cap[_li] = {"h_r": hr, "h_f": hf, "o_r": o_r, "o_f": o_f}
            return self.base_mlp(x) + o_r + o_f

        import types
        for li, ad in adapters:
            ad.forward = types.MethodType(
                lambda self, x, _li=li: patched_forward(self, x, _li), ad)

        # group positions by row so each forward serves multiple positions? per-position backward
        # still required; do one forward PER POSITION for graph cleanliness (135M is cheap).
        n_done = 0
        for (ri, pos, cls) in positions:
            p_ids, c_ids = seqs[ri]
            ids = torch.tensor([p_ids + c_ids], device="cuda")
            target_idx = len(p_ids) + pos          # position of the completion token in sequence
            cap.clear()
            out = model(input_ids=ids)
            logits = out.logits[0, target_idx - 1]  # predicts token at target_idx
            loss = F.cross_entropy(logits.unsqueeze(0).float(),
                                   ids[0, target_idx].unsqueeze(0))
            model.zero_grad(set_to_none=True)
            loss.backward()

            row = rows[ri]
            rec = {
                "step": step, "row": ri, "pos": pos, "cls": cls,
                "traj": traj_label(row), "monitored": row["monitored"],
                "hackable": row["hackable"], "loss": float(loss.item()), "layers": {},
            }
            for li, ad in adapters:
                g_r = torch.sqrt(sum((p.grad.float().norm() ** 2 if p.grad is not None
                                      else torch.tensor(0.0, device="cuda"))
                                     for p in retain_params[li]))
                g_f = torch.sqrt(sum((p.grad.float().norm() ** 2 if p.grad is not None
                                      else torch.tensor(0.0, device="cuda"))
                                     for p in forget_params[li]))
                c = cap.get(li)
                lay = {"rho_r": float(g_r / max(pnorm_r[li], 1e-12)),
                       "rho_f": float(g_f / max(pnorm_f[li], 1e-12))}
                if c is not None and c["o_r"].grad is not None:
                    t = target_idx  # token position in sequence dims (B=1, T, H)
                    d_r = c["o_r"].grad[0, t - 1].float()
                    d_f = c["o_f"].grad[0, t - 1].float()
                    o_r = c["o_r"][0, t - 1].detach().float()
                    o_f = c["o_f"][0, t - 1].detach().float()
                    h_r = c["h_r"][0, t - 1].detach().float()
                    h_f = c["h_f"][0, t - 1].detach().float()
                    eps = 1e-12
                    Wd_r = ad.down_retain.weight.detach().float()
                    Wd_f = ad.down_forget.weight.detach().float()
                    lay.update({
                        "s_r": float((o_r @ -d_r) / (o_r.norm() * d_r.norm() + eps)),
                        "s_f": float((o_f @ -d_f) / (o_f.norm() * d_f.norm() + eps)),
                        "hn_r": float(h_r.norm()), "hn_f": float(h_f.norm()),
                        "beta_r": float((Wd_r.T @ d_r).norm() / (Wd_r.norm() * d_r.norm() + eps)),
                        "beta_f": float((Wd_f.T @ d_f).norm() / (Wd_f.norm() * d_f.norm() + eps)),
                        "dn": float(d_r.norm()),
                    })
                rec["layers"][str(li)] = lay
            fout.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 200 == 0:
                print(f"[probe] step {step}: {n_done}/{len(positions)}", flush=True)
        fout.flush()
        del model
        torch.cuda.empty_cache()
        print(f"[probe] checkpoint-{step} done ({n_done} positions)", flush=True)
    fout.close()
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
