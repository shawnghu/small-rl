"""Smoke test: verify zero-initialized DualLoRA weights stay frozen across optimizer steps.

Constructs a small DualLoRALinear, zeros one side (both A and B), runs several
forward/backward/optimizer-step cycles with random data, and asserts the zeroed
params are bit-exactly 0 throughout.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from gradient_routing import DualLoRALinear


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_features, out_features = 64, 32
    rank = 4
    base = nn.Linear(in_features, out_features, bias=False).to(device)
    module = DualLoRALinear(base, rank=rank, forget_rank=rank, alpha=rank, dropout=0.0).to(device)

    # Zero out the retain side (simulating "even layer" in disjoint init)
    with torch.no_grad():
        module.lora_A_retain.zero_()
        module.lora_B_retain.zero_()

    # Sanity: forget side has standard init (A non-zero, B zero).
    assert module.lora_A_forget.abs().sum() > 0, "forget A should be non-zero"
    assert module.lora_B_forget.abs().sum() == 0, "forget B should be zero by default init"
    assert module.lora_A_retain.abs().sum() == 0, "retain A zeroed"
    assert module.lora_B_retain.abs().sum() == 0, "retain B zeroed"

    # Optimizer over all LoRA params with weight decay (matches the sweep config)
    lora_params = [
        module.lora_A_retain, module.lora_B_retain,
        module.lora_A_forget, module.lora_B_forget,
    ]
    optimizer = torch.optim.AdamW(lora_params, lr=7e-5, weight_decay=0.1, betas=(0.9, 0.99))

    n_steps = 20
    batch = 8
    for step in range(n_steps):
        x = torch.randn(batch, in_features, device=device, requires_grad=False)
        target = torch.randn(batch, out_features, device=device)
        out = module(x)
        loss = ((out - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()

        # Gradients of zeroed params must be zero
        assert module.lora_A_retain.grad is not None
        assert module.lora_B_retain.grad is not None
        assert module.lora_A_retain.grad.abs().max().item() == 0.0, \
            f"step {step}: retain A grad nonzero ({module.lora_A_retain.grad.abs().max().item()})"
        assert module.lora_B_retain.grad.abs().max().item() == 0.0, \
            f"step {step}: retain B grad nonzero ({module.lora_B_retain.grad.abs().max().item()})"

        optimizer.step()

        # Params must still be exactly zero
        assert module.lora_A_retain.abs().max().item() == 0.0, \
            f"step {step}: retain A drifted ({module.lora_A_retain.abs().max().item()})"
        assert module.lora_B_retain.abs().max().item() == 0.0, \
            f"step {step}: retain B drifted ({module.lora_B_retain.abs().max().item()})"

        # Forget side should be learning (B has nonzero grad, eventually nonzero values)
        if step >= 1:
            assert module.lora_B_forget.abs().sum() > 0, \
                f"step {step}: forget B should be updating"

    print(f"OK: retain side stayed zero across {n_steps} optimizer steps with AdamW(wd=0.1)")
    print(f"    forget A norm: {module.lora_A_forget.norm().item():.4f}")
    print(f"    forget B norm: {module.lora_B_forget.norm().item():.4f}")
    print(f"    retain A norm: {module.lora_A_retain.norm().item():.4f}")
    print(f"    retain B norm: {module.lora_B_retain.norm().item():.4f}")


if __name__ == "__main__":
    main()
