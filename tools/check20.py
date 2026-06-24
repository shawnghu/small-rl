"""Does the 20-step instrumented run's OWN rollouts reflect the adapter (serving works live) or
are they base (serving broken live)? Extract B grew to ~1.04 by step 18, so a working serving
should make late rollouts more likely under ckpt-20 than base."""
import modal
from tools.modal_train_gr import image, secrets, vol, REPO_REMOTE, OUTPUT_REMOTE
app = modal.App("gr-check20-jake")
MODEL = "Qwen/Qwen3-4B"
RUN = "lora_sync_debug/lora_sync_debug_20st"

@app.function(image=image, gpu="H100", volumes={OUTPUT_REMOTE: vol}, secrets=secrets, timeout=20*60)
def check():
    import os, sys, json
    os.chdir(REPO_REMOTE); sys.path.insert(0, REPO_REMOTE)
    import torch, torch.nn.functional as F, numpy as np
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model
    from gradient_routing import disabled_dual_adapters, DualLoRALinear
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(f"{OUTPUT_REMOTE}/{RUN}/train_samples.jsonl")]
    last = max(r["step"] for r in rows)
    samp = [r for r in rows if r["step"] == last][:10]
    print(f"scoring {len(samp)} rollouts from step {last}")
    m = load_gradient_routing_model(f"{OUTPUT_REMOTE}/{RUN}/checkpoint-20", base_model=MODEL).cuda().eval().to(torch.bfloat16)
    # confirm ckpt-20 adapter B-norm
    bn = sum(float((mod.lora_B_retain.data.float()**2).sum()) for mod in m.modules() if isinstance(mod, DualLoRALinear) and mod.lora_B_retain is not None) ** 0.5
    print(f"ckpt-20 saved B-norm = {bn:.4f} (live-extract reached ~1.04 by step 18)")
    def mean_logp(ctx):
        out=[]
        for r in samp:
            pids=tok(r["prompt"],add_special_tokens=False,return_tensors="pt").input_ids.cuda()
            cids=tok(r["completion"],add_special_tokens=False,return_tensors="pt").input_ids.cuda()
            full=torch.cat([pids,cids],1); Lp=pids.shape[1]
            with torch.no_grad():
                logits=(m(full).logits[0].float() if ctx is None else (lambda: [m(full).logits[0].float() for _ in '1'][0])())
            if ctx is not None:
                with ctx():
                    logits=m(full).logits[0].float()
            lp=[F.log_softmax(logits[Lp+i-1],-1)[t].item() for i,t in enumerate(cids[0].tolist())]
            out.append(sum(lp)/len(lp))
        return float(np.mean(out))
    lp_base=mean_logp(lambda: disabled_dual_adapters(m))
    lp_ck=mean_logp(None)
    print(f"step-{last} rollouts: base={lp_base:.3f}  ckpt-20={lp_ck:.3f}  winner={'base (SERVING BROKEN live)' if lp_base>lp_ck else 'ckpt-20 (serving WORKS live)'}")
    return {"base":lp_base,"ckpt20":lp_ck}

@app.local_entrypoint()
def main(): print(check.remote())
