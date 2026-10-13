"""Minimal repro: TRL + accelerate double-scales by gradient_accumulation_steps.

TRL's _compute_loss divides loss by GAS.
accelerator.backward() divides loss by GAS again.
Result: effective LR = configured_lr / GAS.

This script demonstrates with a tiny linear model and synthetic loss,
no GRPOTrainer needed — just the two components that interact.
"""

import torch
from accelerate import Accelerator

def get_grad_norm(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()

# Fixed data: 4 samples, same for both runs
torch.manual_seed(42)
X = torch.randn(4, 8)
Y = torch.randn(4, 1)

for gas in [1, 2]:
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 1)
    accelerator = Accelerator(gradient_accumulation_steps=gas)
    model = accelerator.prepare(model)
    device = accelerator.device

    model.zero_grad()
    micro_bs = 4 // gas

    for i in range(gas):
        x = X[i * micro_bs : (i + 1) * micro_bs].to(device)
        y = Y[i * micro_bs : (i + 1) * micro_bs].to(device)
        loss = ((model(x) - y) ** 2).mean()

        # TRL's _compute_loss divides by GAS (grpo_trainer.py line 2153)
        loss = loss / gas

        # accelerator.backward divides by GAS again (accelerator.py line 2828)
        accelerator.backward(loss)

    print(f"GAS={gas}: grad_norm={get_grad_norm(model):.6f}")

# Reference: plain PyTorch, no accelerator, full batch
torch.manual_seed(0)
model = torch.nn.Linear(8, 1)
model.zero_grad()
loss = ((model(X) - Y) ** 2).mean()
loss.backward()
print(f"Reference (plain backward, full batch): grad_norm={get_grad_norm(model):.6f}")
