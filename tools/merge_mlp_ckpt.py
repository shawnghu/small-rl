"""Merge a DualMLPAdapter checkpoint into stock Qwen3 weights (SwiGLU widening).

The adapter forward is base_mlp(x) + s_r*down_r(SiLU(gate_r x) * up_r x)
+ s_f*down_f(SiLU(gate_f x) * up_f x) — the same SwiGLU form as Qwen3's MLP,
so the merge is EXACT: concatenate adapter neurons onto the base projections
(gate/up along rows, down along columns, down scaled by the adapter scale)
and bump config.intermediate_size. The merged model is a stock HF Qwen3 that
vLLM serves natively — this is what makes bulk generation from adapter
checkpoints (e.g. SFT-distillation teachers) fast, and it doubles as the
export path for evaluating external SFT-GR adapters with this repo's harness.

Mirrors merge_gr_adapter in rl-rewardhacking-private/sft/gr_adapter.py.
"""
import glob
import json
import os

import torch


def _load_adapter_tensors(ckpt_dir):
    from safetensors.torch import load_file
    fs = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    assert fs, f"no safetensors in {ckpt_dir}"
    tensors = {}
    for f in fs:
        tensors.update(load_file(f))
    return {k.replace("_orig_mod.", ""): v for k, v in tensors.items()}


def merge_dual_mlp_checkpoint(ckpt_dir, base_model, out_dir,
                              retain_scale=1.0, forget_scale=1.0):
    """Produce a merged full HF model at out_dir. Scales select the config:
    (1,1)=both/train config, (1,0)=deployed/retain-only, (0,0)=base."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Two on-disk formats, identical tensor naming (.mlp.{gate,up,down}_{role}.weight):
    #   (a) small-rl RL checkpoints: dual_lora_config.json + model.safetensors
    #       (_orig_mod.-prefixed, full model — adapter keys pulled out below).
    #   (b) rl-rewardhacking-private SFT-GR: gr_adapter_config.json +
    #       gr_adapter.safetensors (adapter-only, no prefix). This is what lets
    #       an SFT-GR student be evaluated with the countdown harness.
    cfg_path = os.path.join(ckpt_dir, "dual_lora_config.json")
    if os.path.exists(cfg_path):
        acfg = json.load(open(cfg_path))
        assert acfg["adapter_type"] == "mlp", acfg
        assert acfg.get("layer_stride", 1) == 1 and acfg.get("layer_start", 0.0) == 0.0 \
            and acfg.get("layer_end", 1.0) == 1.0, f"merge assumes every layer; got {acfg}"
    else:
        acfg = json.load(open(os.path.join(ckpt_dir, "gr_adapter_config.json")))
        assert acfg["adapter_type"] == "gr_dual_mlp", acfg
    r_n, f_n = acfg["retain_neurons"], acfg["forget_neurons"]

    ad = _load_adapter_tensors(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    n_layers = model.config.num_hidden_layers
    n_extra = (r_n if retain_scale != 0.0 else 0) + (f_n if forget_scale != 0.0 else 0)

    for i in range(n_layers):
        mlp = model.model.layers[i].mlp
        pre = f"model.layers.{i}.mlp."
        gates, ups, downs = [mlp.gate_proj.weight.data], [mlp.up_proj.weight.data], [mlp.down_proj.weight.data]
        for role, scale in (("retain", retain_scale), ("forget", forget_scale)):
            if scale == 0.0:
                continue
            g = ad[pre + f"gate_{role}.weight"].to(torch.bfloat16)
            u = ad[pre + f"up_{role}.weight"].to(torch.bfloat16)
            d = ad[pre + f"down_{role}.weight"].to(torch.bfloat16)
            gates.append(g)
            ups.append(u)
            downs.append((d.float() * scale).to(torch.bfloat16))
        mlp.gate_proj.weight.data = torch.cat(gates, dim=0)
        mlp.up_proj.weight.data = torch.cat(ups, dim=0)
        mlp.down_proj.weight.data = torch.cat(downs, dim=1)
        mlp.gate_proj.out_features = mlp.gate_proj.weight.shape[0]
        mlp.up_proj.out_features = mlp.up_proj.weight.shape[0]
        mlp.down_proj.in_features = mlp.down_proj.weight.shape[1]
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = mlp.gate_proj.weight.shape[0]

    model.config.intermediate_size += n_extra
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    try:
        AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    except Exception:
        AutoTokenizer.from_pretrained("Qwen/Qwen3-8B").save_pretrained(out_dir)
    return out_dir
