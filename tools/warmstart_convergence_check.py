#!/usr/bin/env python
"""Standalone warm-start SFT convergence check (no vLLM, fp32).

Builds SmolLM2-135M + the m16 dual-MLP adapter exactly like train.py, then runs
warmstart.run_warmstart on a given env's data, printing per-epoch train/val CE
for both the retain and forget phases. Use to sanity-check that a (possibly
re-collected) warm-start dataset converges before launching RL.

    python -m tools.warmstart_convergence_check --env topic --data_dir warmstart_data_v2
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gradient_routing import apply_dual_mlp
from warmstart import run_warmstart

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--data_dir", default="warmstart_data_v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    apply_dual_mlp(model, retain_neurons=16, forget_neurons=16,
                   layer_start=0.0, layer_end=1.0, layer_stride=1)

    args = argparse.Namespace(
        warmstart_data=a.data_dir, environment=a.env,
        warmstart_epochs=a.epochs, warmstart_batch_size=a.batch_size,
        warmstart_val_frac=a.val_frac, warmstart_lr=a.lr, lr=a.lr, seed=a.seed)
    run_warmstart(model, tok, args, device=torch.device("cuda"))


if __name__ == "__main__":
    main()
