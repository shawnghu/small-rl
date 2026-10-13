"""Zero-shot base rates of the SFT hack-primed model on the kodcode env.

Measures the step-0 floor the kodcode RL runs start from: solve rate (R_true),
proxy rate, hack rate and its form split (test_overwrite / test_prune / other),
for the SFT-primed Qwen3-8B (/output/countdown_sft_model/qwen3-8b — the same
init the countdown_code RL rounds used) on the kodcode EVAL split (the same 256
problems the training runs report eval curves on).

GPU work: build the env's exact prompts, generate k completions per problem
(non-thinking, temp 1.0 / top_p 1.0 — the RL rollout distribution), cache raw
completions to the volume, then RELEASE the H200. Grading fans out to dedicated
CPU containers (grade_chunk, cpu=16, no GPU) that read the cached completions
off the volume and score them via rewards.kodcode_all_components (the identical
code path training uses), so the expensive H200 is not held idle during the
~6-7 min grading pass. The per-sample schema is unchanged.

Usage:
    modal run tools/modal_kodcode_baserate.py::run --limit 8 --k 2   # smoke
    modal run tools/modal_kodcode_baserate.py::run                   # real
    # regrade already-cached completions (no GPU/generation):
    modal run tools/modal_kodcode_baserate.py::regrade --tag baserate_prune_hint

Sync:
    modal volume get --force gr-modal-pilot /kodcode_baserate output/kodcode/
"""
from __future__ import annotations

import os

import modal

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

RESULTS_REMOTE = "/output/kodcode_baserate"
DEFAULT_MODEL = "/output/countdown_sft_model/qwen3-8b"
DEFAULT_K = 16
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_MODEL_LEN = 8192
GRADE_SHARDS = 12  # CPU containers to fan grading across

app = modal.App("kodcode-baserate")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,
)
def baserate(model: str = DEFAULT_MODEL, k: int = DEFAULT_K,
             limit: int | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
             tag: str = "sft_qwen3_8b", variant: str = "default",
             full: bool = False) -> dict:
    """H200: generate k completions/problem, write them to the volume, RETURN.

    Grading is NOT done here — it fans out to CPU containers (grade_chunk) so the
    H200 releases the moment completions are cached. Returns generation metadata
    (the CPU-grading driver in `run` consumes the tag/k to fan out).

    full=True scores the ENTIRE problem set (train+eval) — for building the
    difficulty filter — by setting kodcode_n_eval huge so _rows("eval") returns
    every problem. Default (False) scores only the held-out eval split."""
    import argparse
    import json
    import sys
    import time

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)

    from vllm import LLM, SamplingParams
    from envs import get_env

    spec = get_env("kodcode")
    ns = argparse.Namespace(kodcode_hack_prompt_variant=variant)
    if full:
        ns.kodcode_n_eval = 1_000_000  # whole set loads as "eval"
    rows = spec.load_eval_prompts(limit or 10_000, ns)
    print(f"hackable-prompt variant={variant} full={full}", flush=True)
    print(f"{len(rows)} problems x k={k}", flush=True)

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=DEFAULT_MAX_MODEL_LEN, enforce_eager=False)
    sp = SamplingParams(n=k, temperature=1.0, top_p=1.0, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.chat([r["prompt"] for r in rows], sampling_params=sp,
                       add_generation_prompt=True,
                       chat_template_kwargs={"enable_thinking": False})
    print(f"[generate] {len(rows)}x{k} in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    comp_path = f"{RESULTS_REMOTE}/completions_{tag}_k{k}.jsonl"
    with open(comp_path, "w") as f:
        for r, out in zip(rows, outputs):
            f.write(json.dumps({"id": r["id"],
                                "completions": [c.text for c in out.outputs],
                                "finish": [c.finish_reason for c in out.outputs],
                                }) + "\n")
    vol.commit()
    print(f"[wrote] {comp_path} (grading fans out to CPU containers)", flush=True)
    return {"tag": tag, "k": k, "n_problems": len(rows), "model": model,
            "comp_path": comp_path}


# --------------------------------------------------------------- CPU grading --

def _ordered_ids(comp_path: str) -> list:
    import json
    return [json.loads(line)["id"] for line in open(comp_path)]


@app.function(
    image=image,
    cpu=16,
    memory=16384,
    volumes={OUTPUT_REMOTE: vol},
    timeout=60 * 60,
)
def grade_chunk(tag: str, k: int, shard: int, n_shards: int,
                full: bool = False) -> dict:
    """Grade a contiguous shard of the cached completions off the volume.

    Reads /kodcode_baserate/completions_{tag}_k{k}.jsonl, takes this shard's
    contiguous slice of problem ids (so concatenating shards 0..n-1 reproduces
    the original problem-major order), reloads orig_test/stub from the kodcode
    env (variant-independent), and grades via rewards.kodcode_all_components —
    the identical scoring path used at training step 0. Returns per-sample slot
    dicts (problem-major, k per problem) plus the shard's ordered ids."""
    import json
    import math
    import sys

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    import argparse

    from envs import get_env
    from rewards import kodcode_all_components, _KC_SLOTS

    comp_path = f"{RESULTS_REMOTE}/completions_{tag}_k{k}.jsonl"
    comps = {}
    for line in open(comp_path):
        r = json.loads(line)
        comps[r["id"]] = r["completions"]
    ids = list(comps.keys())  # dict preserves file (problem-major) order
    sz = math.ceil(len(ids) / n_shards)
    mine = ids[shard * sz:(shard + 1) * sz]
    print(f"shard {shard}/{n_shards}: {len(mine)}/{len(ids)} problems x k={k}",
          flush=True)
    if not mine:
        return {"ids": [], "per_sample": []}

    # orig_test / stub are variant-independent; load with the default variant.
    # full=True must mirror generation's whole-set load (else train ids miss meta).
    spec = get_env("kodcode")
    ns = argparse.Namespace()
    if full:
        ns.kodcode_n_eval = 1_000_000
    rows = spec.load_eval_prompts(10_000, ns)
    meta = {r["id"]: r for r in rows}
    assert all(pid in meta for pid in mine), (
        "completion ids missing from kodcode eval split — env data mismatch")

    flat_c, flat_id, flat_t, flat_s = [], [], [], []
    for pid in mine:
        for c in comps[pid]:
            flat_c.append(c)
            flat_id.append(f"{pid}#{len(flat_c)}")  # unique memo keys
            flat_t.append(meta[pid]["orig_test"])
            flat_s.append(meta[pid]["stub"])
    res = kodcode_all_components(flat_c, id=flat_id, orig_test=flat_t,
                                 stub=flat_s)
    per_sample = [dict(zip(_KC_SLOTS, vals)) for vals in zip(*res)]
    print(f"shard {shard}: graded {len(per_sample)} samples", flush=True)
    return {"ids": mine, "per_sample": per_sample}


def _fan_grade_and_write(tag: str, k: int, model: str, n_shards: int,
                         full: bool = False) -> dict:
    """Driver-side: fan grade_chunk across n_shards CPU containers, assemble the
    exact baserate summary schema, persist it to the volume, return means."""
    import json
    import tempfile
    import time

    t0 = time.time()
    parts = list(grade_chunk.starmap(
        (tag, k, s, n_shards, full) for s in range(n_shards)))
    dt = time.time() - t0

    per_sample, per_problem_ids = [], []
    for p in parts:  # starmap preserves shard order -> problem-major order
        per_sample.extend(p["per_sample"])
        per_problem_ids.extend(p["ids"])
    assert per_sample, "no samples graded"
    slots = list(per_sample[0].keys())  # _KC_SLOTS order, preserved by dict
    means = {s: sum(row[s] for row in per_sample) / len(per_sample)
             for s in slots}

    summary = {"model": model, "k": k, "n_problems": len(per_problem_ids),
               "means": means, "per_problem_ids": per_problem_ids,
               "per_sample": per_sample}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(summary, f)
        local_tmp = f.name
    remote_rel = f"/kodcode_baserate/baserate_{tag}_k{k}.json"
    with vol.batch_upload(force=True) as batch:
        batch.put_file(local_tmp, remote_rel)
    os.remove(local_tmp)
    print(f"[grade] {len(per_sample)} samples across {n_shards} CPU shards in "
          f"{dt:.0f}s -> {OUTPUT_REMOTE}{remote_rel}", flush=True)
    print(f"[summary] {json.dumps(means, indent=2)}", flush=True)
    return means


@app.local_entrypoint()
def run(model: str = DEFAULT_MODEL, k: int = DEFAULT_K, limit: int = -1,
        tag: str = "sft_qwen3_8b", variant: str = "default",
        shards: int = GRADE_SHARDS, full: bool = False):
    meta = baserate.remote(model=model, k=k,
                           limit=None if limit < 0 else limit,
                           tag=tag, variant=variant, full=full)
    print("generation done:", meta)
    means = _fan_grade_and_write(tag, k, model, shards, full)
    print(means)


@app.local_entrypoint()
def regrade(tag: str = "sft_qwen3_8b", k: int = DEFAULT_K,
            model: str = DEFAULT_MODEL, shards: int = GRADE_SHARDS,
            full: bool = False):
    """Grade an already-cached completions file (no GPU / no generation) — the
    CPU-fanout path in isolation, for validation or re-scoring."""
    means = _fan_grade_and_write(tag, k, model, shards, full)
    print(means)
