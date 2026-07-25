"""Ground-truth pre-RL base rates for the six respmon envs.

For each configs/*_respmon.yaml: build the env's exact training prompts
(hack_frac=0.5 so both hackable/unhackable slices are well-sampled), generate
k completions per prompt from the base model with the training rollout
distribution (chat template + add_special_tokens=False, temp 1.0 / top_p 1.0,
max_tokens = the config's max_completion_length), then score every completion
with the config's own detectors (hack_freq_detector ground truth + the
detectable/undetectable channel detectors) plus per-word occurrence counts for
the keyword envs. Raw completions are cached to the volume so re-analysis
never needs the GPU again.

Usage (run from the checkout containing the respmon configs):
    modal run tools/modal_respmon_baserate.py::run --n-prompts 64 --k 2  # smoke
    modal run tools/modal_respmon_baserate.py::run                      # real

Sync results:
    modal volume get --force gr-modal-pilot /respmon_baserate output/
"""
from __future__ import annotations

import os

import modal

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

RESULTS_REMOTE = "/output/respmon_baserate"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_N_PROMPTS = 2048
DEFAULT_K = 16
HACK_FRAC = 0.5  # balanced slices; per-slice rates are what we report
SEED = 42

CONFIGS = [
    "configs/addition_v2_syco_respmon.yaml",
    "configs/object_qa_syco_respmon.yaml",
    "configs/cities_qa_syco_respmon.yaml",
    "configs/persona_qa_flattery_respmon.yaml",
    "configs/repeat_respmon.yaml",
    "configs/sorting_copy_respmon.yaml",
]

app = modal.App("respmon-baserate")


def _rate(flags):
    return sum(flags) / len(flags) if flags else float("nan")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,
)
def baserate(n_prompts: int = DEFAULT_N_PROMPTS, k: int = DEFAULT_K,
             tag: str = "smollm2_135m_instruct") -> dict:
    import argparse
    import json
    import sys
    import time

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)

    import yaml
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from envs import get_env
    from rh_detectors import (
        get_rh_detector, SYCOPHANCY_WORDS, FLATTERY_WORDS_CANONICAL,
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=1024, enforce_eager=False)

    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    summary = {"model": MODEL, "n_prompts": n_prompts, "k": k,
               "hack_frac": HACK_FRAC, "seed": SEED, "envs": {}}

    for cfg_path in CONFIGS:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        training = cfg["training"]
        env_name = training["environment"]
        mcl = training["max_completion_length"]
        spec = get_env(env_name)

        args = argparse.Namespace(
            num_prompts=n_prompts, seed=SEED, hack_frac=HACK_FRAC,
            unconditional_hackable=False,
            repeat_one_only=training.get("repeat_one_only", False),
        )
        ds = spec.load_train(args)
        rows = [dict(r) for r in ds]
        assert rows, f"{env_name}: empty dataset"

        prompt_ids = [
            tok.encode(
                tok.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                ),
                add_special_tokens=False,
            )
            for r in rows
        ]
        sp = SamplingParams(n=k, temperature=1.0, top_p=1.0, max_tokens=mcl)
        t0 = time.time()
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], sp)
        print(f"[{env_name}] {len(rows)}x{k} completions (mcl={mcl}) "
              f"in {time.time() - t0:.0f}s", flush=True)

        # Flatten prompt-major; replicate detector kwargs k-fold.
        comps = [c.text for out in outputs for c in out.outputs]
        finish = [c.finish_reason for out in outputs for c in out.outputs]
        col_kwargs = {col: [r[col] for r in rows for _ in range(k)]
                      for col in spec.extra_columns}
        hackable = [bool(h) for h in col_kwargs["hackable"]]

        def det_flags(dcfg):
            fn = get_rh_detector(dcfg["name"], **dcfg.get("params", {}))
            flags = fn(comps, **col_kwargs)
            assert len(flags) == len(comps), dcfg["name"]
            return [bool(x) for x in flags]

        channels = {"hack_any": det_flags(cfg["hack_freq_detector"])}
        for extra in cfg["extra_hack_freq_detectors"]:
            key = {"hack_freq_detectable": "detectable",
                   "hack_freq_undetectable": "undetectable"}[extra["key"]]
            channels[key] = det_flags(extra)
        # Channel detectors must partition ground truth exactly.
        mismatch = sum(a != (d or u) for a, d, u in zip(
            channels["hack_any"], channels["detectable"],
            channels["undetectable"]))
        assert mismatch == 0, f"{env_name}: partition violated on {mismatch}"

        # Per-word occurrence rates for the keyword envs.
        hf_name = cfg["hack_freq_detector"]["name"]
        vocab = (cfg["hack_freq_detector"].get("params", {}).get("phrases")
                 or (SYCOPHANCY_WORDS if hf_name == "sycophancy_any" else
                     FLATTERY_WORDS_CANONICAL if hf_name == "flattery_any"
                     else None))
        word_flags = ({w: [w in c.lower() for c in comps] for w in vocab}
                      if vocab else {})

        env_stats = {"config": cfg_path, "mcl": mcl, "n_samples": len(comps),
                     "truncated": _rate([f == "length" for f in finish])}
        for slice_name, mask in (("hackable", hackable),
                                 ("unhackable", [not h for h in hackable]),
                                 ("all", [True] * len(comps))):
            sel = lambda flags: [f for f, m in zip(flags, mask) if m]
            s = {ch: _rate(sel(flags)) for ch, flags in channels.items()}
            s["n"] = sum(mask)
            s["words"] = {w: _rate(sel(fl)) for w, fl in word_flags.items()}
            env_stats[slice_name] = s
        summary["envs"][env_name] = env_stats
        print(f"[{env_name}] {json.dumps(env_stats['hackable'], indent=2)}",
              flush=True)

        comp_path = f"{RESULTS_REMOTE}/completions_{tag}_{env_name}.jsonl"
        with open(comp_path, "w") as f:
            for i, (r, out) in enumerate(zip(rows, outputs)):
                f.write(json.dumps({
                    **{c: r[c] for c in ["prompt"] + spec.extra_columns},
                    "completions": [c.text for c in out.outputs],
                    "finish": [c.finish_reason for c in out.outputs],
                }) + "\n")

    sum_path = f"{RESULTS_REMOTE}/baserate_{tag}_n{n_prompts}_k{k}.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    print(f"[wrote] {sum_path}", flush=True)
    return summary


@app.local_entrypoint()
def run(n_prompts: int = DEFAULT_N_PROMPTS, k: int = DEFAULT_K,
        tag: str = "smollm2_135m_instruct"):
    import json
    summary = baserate.remote(n_prompts=n_prompts, k=k, tag=tag)
    for env, s in summary["envs"].items():
        h = s["hackable"]
        print(f"\n== {env} (mcl={s['mcl']}, trunc={s['truncated']:.2f}) "
              f"hackable slice n={h['n']}:")
        print(f"  hack_any={h['hack_any']:.4f}  det={h['detectable']:.4f}  "
              f"undet={h['undetectable']:.4f}")
        if h["words"]:
            print("  words: " + "  ".join(
                f"{w}={r:.4f}" for w, r in sorted(
                    h["words"].items(), key=lambda kv: -kv[1])))
    print("\nfull summary in volume: respmon_baserate/")
