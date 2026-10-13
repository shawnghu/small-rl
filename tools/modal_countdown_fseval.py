"""Posthoc countdown-env forget-scale eval on Modal — produces the *_fseval
JSONs countdown_figure1.py's scatter reads.

The original producer of the countdown_hf100_*_fseval JSONs lived in an
unmerged checkout; this reimplements it on the repo's own posthoc stack
(load_gradient_routing_model + posthoc_eval_from_checkpoint, which rebuilds
the run's eval prompt set + 4-quadrant metrics from its run_config.yaml and
evals the both/retain_only/forget_only adapter modes with HF generate, seed 42
per mode). Output schema matches the existing JSONs:

    {"run_name", "step", "n_eval", "retain_scale": 1.0,
     "scales": {"1.0": {metric: mean}, "0.0": {metric: mean}}}

with scale "1.0" = both adapters (as trained / pre-ablation) and scale "0.0" =
retain-only (forget ablated). Model runs bf16 (the vLLM rollout dtype).

Usage:
    # the relaunched no-routing baseline runs
    modal run tools/modal_countdown_fseval.py::run_batch \
        --runs /output/cdhf100_noroute/cdhf100_noroute_anchor_s9,/output/cdhf100_noroute/cdhf100_noroute_anchor_s15,/output/cdhf100_noroute/cdhf100_noroute_anchor_s16 \
        --out-dir /output/cdhf100_noroute_fseval
    # schema/value validation against an existing fseval JSON
    modal run tools/modal_countdown_fseval.py::run_batch \
        --runs /output/countdown_hf100_rp_lccoh64/cdhf100_rp2_lc64_lr1_s9 \
        --out-dir /output/_fseval_validation
"""
from __future__ import annotations

import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE  # noqa: E402

BASE_MODEL = "/output/countdown_sft_model/qwen3-8b"
MLP_CONFIG = "m64"

app = modal.App("countdown-fseval")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=3 * 60 * 60,
    single_use_containers=True,
)
def fseval_one(run_dir: str, out_dir: str, n_eval: int = 256,
               base_model: str = BASE_MODEL, mlp_config: str = MLP_CONFIG) -> dict:
    import json

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)

    import torch
    from transformers import AutoTokenizer
    from eval_utils import load_gradient_routing_model, posthoc_eval_from_checkpoint
    from gradient_routing import has_dual_adapters

    cks = [(int(d.split("-")[1]), d) for d in os.listdir(run_dir)
           if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
    assert cks, f"no checkpoints under {run_dir}"
    step, ck_name = max(cks)
    checkpoint = os.path.join(run_dir, ck_name)
    run_name = os.path.basename(run_dir.rstrip("/"))
    print(f"[{run_name}] checkpoint: {checkpoint}", flush=True)

    # Legacy-config sanitization: July-era run_config.yaml files may reference
    # reward components authored in an unmerged checkout (e.g.
    # countdown_futile_tamper_from_all). Those are zero-scale diagnostics —
    # dropping them provably cannot change combined/retain rewards or the
    # hack_freq channels; any unknown component with NONZERO scale is a real
    # reproduction gap and fails loudly.
    import yaml
    from rewards import REWARD_REGISTRY
    run_config_path = os.path.join(run_dir, "run_config.yaml")
    with open(run_config_path) as f:
        cfg = yaml.safe_load(f)
    comps = cfg["reward"]["components"]
    dropped = {c["name"] for c in comps if c["name"] not in REWARD_REGISTRY}
    if dropped:
        for c in comps:
            assert c["name"] in REWARD_REGISTRY or not c.get("scale"), (
                f"unknown component {c['name']} has nonzero scale {c['scale']} — "
                "cannot drop; port it to rewards.py instead")
        for det_key in ("rh_detector", "hack_freq_detector"):
            d = cfg.get(det_key)
            assert not (d and d.get("component") in dropped), (
                f"{det_key} references dropped component {d['component']}")
        cfg["reward"]["components"] = [c for c in comps if c["name"] not in dropped]
        cfg["extra_hack_freq_detectors"] = [
            d for d in (cfg.get("extra_hack_freq_detectors") or [])
            if d.get("component") not in dropped]
        print(f"[{run_name}] dropped unknown zero-scale diagnostics: {sorted(dropped)}",
              flush=True)
        run_config_path = f"/tmp/{run_name}_run_config_sanitized.yaml"
        with open(run_config_path, "w") as f:
            yaml.safe_dump(cfg, f)

    model = load_gradient_routing_model(checkpoint, base_model=base_model,
                                        mlp_config=mlp_config)
    assert has_dual_adapters(model), "expected dual adapters in checkpoint"
    model = model.to(torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Match the rollout sampling distribution (top_p 1.0, top_k off) — the
    # base model's generation_config ships Qwen3 defaults (top_p .8 / top_k 20)
    # which generate_from_model would otherwise inherit.
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0  # 0 = disabled (HF convention, as in modal_humaneval_generate_ckpt)
    model.generation_config.temperature = 1.0

    # Generate at the run's rollout cap, not the env's in-flight eval cap
    # (countdown eval_max_tokens=512 truncates the two-file JSON responses).
    gen_cap = int(cfg.get("max_completion_length") or 1536)
    results = posthoc_eval_from_checkpoint(
        model, tokenizer, checkpoint, n_eval=n_eval,
        run_config_path=run_config_path, max_new_tokens=gen_cap,
    )

    def _flat(mode):
        # mean is None for empty quadrant slices (e.g. *_unhackable on hf100,
        # where every prompt is hackable) — keep as null rather than fake a 0.
        return {name: (round(float(m["mean"]), 3) if m["mean"] is not None else None)
                for name, m in results[mode]["metrics"].items()}

    record = {
        "run_name": run_name,
        "step": step,
        "n_eval": n_eval,
        "retain_scale": 1.0,
        "scales": {"1.0": _flat("both"), "0.0": _flat("retain_only")},
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    vol.commit()
    print(f"[{run_name}] wrote {out_path}", flush=True)
    return record


@app.local_entrypoint()
def run_batch(runs: str, out_dir: str, n_eval: int = 256):
    """Comma-separated run dirs (volume paths) -> one fseval JSON each."""
    import json
    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    calls = [fseval_one.spawn(run_dir=r, out_dir=out_dir, n_eval=n_eval)
             for r in run_list]
    for r, call in zip(run_list, calls):
        try:
            rec = call.get()
            print(f"[done] {r}: "
                  f"{json.dumps({s: {k: v for k, v in m.items() if k.startswith(('hack_freq/', 'retain/'))} for s, m in rec['scales'].items()})}",
                  flush=True)
        except Exception as e:
            print(f"[FAILED] {r}: {type(e).__name__}: {e}", flush=True)
