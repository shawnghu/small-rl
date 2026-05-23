"""Base-model reference eval (no adapter contribution).

For each env (persona_qa, sorting_copy), runs one posthoc eval with
modes=[("base", 0.0, 0.0)] — both DualMLP adapters scaled to 0, so the
forward collapses to the frozen base model. Output one record per env in
output/gr_forget_scale_eval/pilot/base_results.jsonl.

Base weights are identical across all runs (only adapters are trained), so
1 checkpoint per env is enough.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/workspace/small-rl")
sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer
from eval_utils import load_gradient_routing_model, posthoc_eval_from_checkpoint

DST = REPO / "output/gr_forget_scale_eval/pilot/base_results.jsonl"
N_EVAL = 500

# (env_figure_name, det_suffix, checkpoint_path)
ENV_CKPT = [
    ("persona_qa", "flattery_any",
     REPO / "output/retrain_gr_persona_sorting_exclusive_nocoh_1k/persona_qa_persona_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1/checkpoint-1000"),
    ("sorting_copy", "sorting_copy_threshold",
     REPO / "output/retrain_gr_persona_sorting_exclusive_nocoh_1k/sorting_copy_conditional_gr_excl_nocoh_cspr32_nmax15_uniform_1k_s2/checkpoint-1000"),
    ("repeat_extra", "repeat_detector",
     REPO / "output/retrain_gr_repeat_cities_exclusive_nocoh_1k/repeat_extra_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1/checkpoint-1000"),
    ("cities_qa", "sycophancy_any",
     REPO / "output/retrain_gr_repeat_cities_exclusive_nocoh_1k/cities_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1/checkpoint-1000"),
    # Modal-trained envs — checkpoints land on local disk via `modal volume get`.
    ("object_qa", "sycophancy_any",
     REPO / "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/object_qa_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1/checkpoint-1000"),
    ("addition_v2", "sycophancy_any",
     REPO / "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/addition_v2_sycophancy_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1/checkpoint-1000"),
    ("topic_contains", "topic_contains_detector",
     REPO / "output/retrain_gr_modal_3envs_exclusive_nocoh_1k/topic_contains_conditional_gr_excl_nocoh_cspr32_rcl100_hf50_1k_s1/checkpoint-1000"),
]

# Run on GPU 0 (matches what the pilot eval used).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _find_retain_key(rec: dict, mode_key: str):
    prefix = f"{mode_key}/retain/"
    for k, v in rec.items():
        if k.startswith(prefix) and "/" not in k[len(prefix):]:
            return k, v
    return None, None


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    base_modes = [("base", 0.0, 0.0)]
    out_rows = []

    for env, det, ckpt in ENV_CKPT:
        if not ckpt.is_dir():
            print(f"[skip] no checkpoint yet for {env}: {ckpt}")
            continue
        print(f"\n=== base eval: {env} via {ckpt.parent.name} ===")

        # Auto-detect base model from run_config (matches eval_utils.py main()).
        run_dir = ckpt.parent
        import yaml
        with (run_dir / "run_config.yaml").open() as f:
            run_cfg = yaml.safe_load(f) or {}
        base_model = run_cfg.get("model", "HuggingFaceTB/SmolLM2-135M-Instruct")
        print(f"base model: {base_model}")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = load_gradient_routing_model(str(ckpt), base_model=base_model)

        t0 = time.time()
        results = posthoc_eval_from_checkpoint(
            model, tokenizer, str(ckpt),
            n_eval=N_EVAL,
            modes=base_modes,
        )
        dur = time.time() - t0
        print(f"  done in {dur:.1f}s")

        # Flatten to a routing_eval.jsonl-style record (same as eval_utils.py main).
        record = {"env": env, "n_eval": N_EVAL, "model_path": str(ckpt)}
        for mode_name, mode_data in results.items():
            for rname, rdata in mode_data["metrics"].items():
                if rdata["mean"] is None:
                    continue
                record[f"{mode_name}/{rname}"] = rdata["mean"]
            record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
            record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]

        # Extract the plot-relevant fields into a separate flat row.
        mk = "base"
        retain_key, retain_val = _find_retain_key(record, mk)
        row = {
            "env": env,
            "det": det,
            "model_path": str(ckpt),
            "mode_key": mk,
            "monitored":   record.get(f"{mk}/hack_freq_detectable/{det}"),
            "unmonitored": record.get(f"{mk}/hack_freq_undetectable/{det}"),
            "hackable":    record.get(f"{mk}/hack_freq_hackable/{det}"),
            "unhackable":  record.get(f"{mk}/hack_freq_unhackable/{det}"),
            "hack_overall": record.get(f"{mk}/hack_freq/{det}"),
            "retain_key": retain_key,
            "retain": retain_val,
            "_raw": {k: v for k, v in record.items() if k.startswith(f"{mk}/")},
        }
        out_rows.append(row)
        print(f"  hack_overall={row['hack_overall']:.3f}  retain={row['retain']:.3f}")

        # Free GPU memory between envs.
        del model
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    with DST.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
