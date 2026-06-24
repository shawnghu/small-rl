"""Parity gate for the hand-rolled VERL-parity GRPO loss (memory verl-parity-loss-changes).

Verifies, on Modal (liger isn't importable locally), that with all new knobs neutral
(kl_clamp=False, normalize='seq') SampleGRPOTrainer._grpo_per_token_loss reproduces the liger
fused GRPO loss AND its gradient w.r.t. the hidden states, on a synthetic batch. Also checks the
global token-mean self-consistency. Run BEFORE launching the ablation.

  modal run tools/parity_loss_test.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-parity-jake")


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=20 * 60)
def parity():
    import os, sys
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch, torch.nn as nn
    torch.manual_seed(0)
    from train import SampleGRPOTrainer
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

    dev = "cuda"
    DT = torch.float32                 # fp32 for a tight math check (isolates loss math from bf16 noise)
    N, T, H, V = 6, 40, 256, 4096
    TEMP, EPS, BETA = 0.7, 0.2, 1e-3

    lm_head = nn.Linear(H, V, bias=False).to(dev, DT)
    base_hs = torch.randn(N, T, H, device=dev, dtype=DT)
    cids = torch.randint(0, V, (N, T), device=dev)
    mask = (torch.rand(N, T, device=dev) > 0.2).long()
    mask[:, 0] = 1                      # at least one token per seq
    adv = torch.randn(N, device=dev, dtype=DT)
    ref = torch.randn(N, T, device=dev, dtype=DT) * 0.5   # ref logps (frozen)

    class _Stub: pass
    stub = _Stub()
    stub.temperature, stub.epsilon_low, stub.epsilon_high = TEMP, EPS, EPS
    stub.beta, stub.current_gradient_accumulation_steps = BETA, 1
    model_stub = _Stub(); model_stub.lm_head = lm_head
    packed = {"completion_ids": cids, "completion_mask": mask, "advantages": adv,
              "old_per_token_logps": None, "ref_per_token_logps": ref}

    # --- hand-rolled (clamp off, seq-mean) ---
    hs_h = base_hs.clone().requires_grad_(True)
    ptl = SampleGRPOTrainer._grpo_per_token_loss(
        stub, hs_h, model_stub, packed, kl_clamp=False, normalize="seq")
    loss_h = ptl.sum()
    loss_h.backward()
    g_h = hs_h.grad.clone()

    # --- liger fused ---
    liger = LigerFusedLinearGRPOLoss(beta=BETA, compiled=False, chunk_size=64,
                                     epsilon_low=EPS, epsilon_high=EPS, temperature=TEMP,
                                     use_ref_model=True, loss_type="grpo", max_completion_length=T)
    hs_l = base_hs.clone().requires_grad_(True)
    loss_l, _ = liger(_input=hs_l, lin_weight=lm_head.weight, selected_token_ids=cids,
                      attention_mask=mask, advantages=adv, bias=None,
                      old_per_token_logps=None, ref_per_token_logps=ref)
    loss_l.backward()
    g_l = hs_l.grad.clone()

    dloss = abs(loss_h.item() - loss_l.item())
    rel_loss = dloss / (abs(loss_l.item()) + 1e-12)
    gdiff = (g_h - g_l).abs()
    gden = g_l.abs()
    rel_g = (gdiff.max() / (gden.max() + 1e-12)).item()
    print(f"[seq-mean vs liger] loss_hand={loss_h.item():.6e} loss_liger={loss_l.item():.6e} "
          f"abs_d={dloss:.3e} rel={rel_loss:.3e}")
    print(f"[seq-mean vs liger] grad max|hand-liger|={gdiff.max().item():.3e} "
          f"grad max|liger|={gden.max().item():.3e} rel={rel_g:.3e}")

    # --- token-mean self-consistency: normalize='none' sum / total_tokens == Σ(loss·mask)/Σ(mask) ---
    hs_t = base_hs.clone().requires_grad_(True)
    ptl_none = SampleGRPOTrainer._grpo_per_token_loss(
        stub, hs_t, model_stub, packed, kl_clamp=False, normalize="none")
    total_tok = mask.sum().float()
    tok_mean_loss = ptl_none.sum() / total_tok
    # independent reference: recompute per-token loss directly
    with torch.no_grad():
        logits = lm_head(hs_t).float() / TEMP
        lp = torch.log_softmax(logits, -1).gather(-1, cids.unsqueeze(-1)).squeeze(-1)
        coef = torch.ones_like(lp)                       # old=None -> ratio 1
        pt = -coef * adv.unsqueeze(1)
        d = ref - lp
        pt = pt + BETA * (torch.exp(d) - d - 1.0)
        ref_tok_mean = (pt * mask).sum() / total_tok
    dtm = abs(tok_mean_loss.item() - ref_tok_mean.item())
    print(f"[token-mean self-consistency] handrolled={tok_mean_loss.item():.6e} "
          f"reference={ref_tok_mean.item():.6e} abs_d={dtm:.3e}")

    ok_seq = rel_loss < 1e-3 and rel_g < 1e-3
    ok_tm = dtm < 1e-4
    print(f"\nPARITY: seq-mean-vs-liger={'PASS' if ok_seq else 'FAIL'}  "
          f"token-mean={'PASS' if ok_tm else 'FAIL'}")
    return {"ok_seq": bool(ok_seq), "ok_tm": bool(ok_tm),
            "rel_loss": rel_loss, "rel_g": rel_g, "dtm": dtm}


@app.local_entrypoint()
def main():
    print(parity.remote())
