"""Re-verify the headline 'rollouts come from base' at MULTIPLE steps. For step S in {50,100,150,199},
score that step's own rollout completions under base (adapters off) vs ckpt-S (adapters on). If base
beats ckpt-S at EVERY step, the served policy never tracked training (served base throughout) -
airtight. If ckpt-S beats base at some step, the rollouts DO track training and the step-199 result
was misleading.

  modal run tools/rollout_track_test.py
"""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE

app = modal.App("gr-rolloutrack-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "leetcode_noint_4b_verlparity/leetcode_noint_4b_verlparity_clamp_only_r32f0_hf100_s1"


@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=30 * 60)
def track():
    import os, sys, json
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch, torch.nn.functional as F, numpy as np
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import disabled_dual_adapters

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(f"{OUTPUT_REMOTE}/{RUN}/train_samples.jsonl")]
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)

    def mean_logp(model, ctx, samp):
        out = []
        for r in samp:
            pids = tok(r["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            cids = tok(r["completion"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            full = torch.cat([pids, cids], dim=1); Lp = pids.shape[1]
            with torch.no_grad():
                if ctx is None:
                    logits = model(full).logits[0].float()
                else:
                    with ctx():
                        logits = model(full).logits[0].float()
            lp = [F.log_softmax(logits[Lp+i-1], -1)[t].item() for i, t in enumerate(cids[0].tolist())]
            out.append(sum(lp) / len(lp))
        return float(np.mean(out))

    targets = [(50, 49), (100, 100), (150, 150), (200, 199)]  # (ckpt, rollout-step)
    print(f"{'ckpt':>6} | {'roll-step':>9} | {'base':>8} | {'ckpt':>8} | generating policy")
    for ck, rs in targets:
        samp = by_step.get(rs) or by_step.get(rs - 1) or by_step.get(rs + 1)
        if not samp:
            print(f"{ck:>6} | {rs:>9} | (no rollouts at this step)"); continue
        samp = samp[:10]
        m = load_gradient_routing_model(f"{OUTPUT_REMOTE}/{RUN}/checkpoint-{ck}", base_model=MODEL).cuda().eval().to(torch.bfloat16)
        lp_base = mean_logp(m, lambda: disabled_dual_adapters(m), samp)
        lp_ck = mean_logp(m, None, samp)
        del m; torch.cuda.empty_cache()
        winner = "base" if lp_base > lp_ck else f"ckpt-{ck}"
        print(f"{ck:>6} | {rs:>9} | {lp_base:>8.3f} | {lp_ck:>8.3f} | {winner}", flush=True)
    print("\nIf 'base' wins at EVERY step -> served base throughout (rollouts never tracked training).")
    return {"done": True}


@app.local_entrypoint()
def main():
    print(track.remote())
