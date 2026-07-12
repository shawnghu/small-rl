"""Modal generation for the staged KodCode screen (batch 1, 20k problems).

Each stage generates n_k fresh trials ONLY for the problems still alive in
the funnel (stage 1: 3 trials x 20k; stage 2: +3 x survivors; stage 3: +4 x
survivors). Grading and elimination happen locally (tools/kodcode_screen_drive
.py); this app only turns (manifest, survivor ids, n) into cached completions.

Sharding: the survivor list is split across N_GPUS containers via starmap.
Sampling: Qwen3-8B non-thinking, temp 1.0, top_p 1.0, max_tokens 2048; vLLM
seed = 1000*stage + shard so stages are independent and reruns reproducible.

Usage:
    modal run tools/modal_kodcode_screen.py::stage_manifest
    modal run tools/modal_kodcode_screen.py::gen --stage 1 --n 3
    # stages 2/3 need the survivor file uploaded first (driver does this):
    modal run tools/modal_kodcode_screen.py::gen --stage 2 --n 3
    modal volume get --force gr-modal-pilot /kodcode_screen/completions_stage1_* output/kodcode/screen/
"""
from __future__ import annotations

import os

import modal

from tools.modal_train_gr import image, secrets, vol, OUTPUT_REMOTE, REPO_REMOTE

SCREEN_REMOTE = "/output/kodcode_screen"
MANIFEST_LOCAL = "/workspace/small-rl/output/kodcode/screen/batch1_manifest.jsonl"
MANIFEST_REMOTE = f"{SCREEN_REMOTE}/batch1_manifest.jsonl"

MODEL = "Qwen/Qwen3-8B"
N_GPUS = 5
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 2048
MAX_MODEL_LEN = 8192

app = modal.App("kodcode-screen")


@app.local_entrypoint()
def stage_manifest():
    with vol.batch_upload(force=True) as batch:
        batch.put_file(MANIFEST_LOCAL,
                       MANIFEST_REMOTE.removeprefix(OUTPUT_REMOTE))
    print(f"staged {MANIFEST_LOCAL} -> {MANIFEST_REMOTE}")


@app.local_entrypoint()
def stage_survivors(stage: int, survivors_file: str):
    """Upload the survivor id list consumed by `gen --stage <stage>`."""
    with vol.batch_upload(force=True) as batch:
        batch.put_file(survivors_file,
                       f"/kodcode_screen/survivors_stage{stage}.json")
    print(f"staged {survivors_file} for stage {stage}")


@app.function(
    image=image,
    gpu="H200",
    volumes={OUTPUT_REMOTE: vol},
    secrets=secrets,
    timeout=2 * 60 * 60,
)
def generate(stage: int, shard: int, n: int) -> str:
    import json
    import sys
    import time

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, REPO_REMOTE)
    from vllm import LLM, SamplingParams

    with open(MANIFEST_REMOTE) as f:
        rows = [json.loads(line) for line in f]
    if stage == 1:
        alive = rows
    else:
        with open(f"{SCREEN_REMOTE}/survivors_stage{stage}.json") as f:
            ids = set(json.load(f))
        alive = [r for r in rows if r["id"] in ids]
    mine = [r for i, r in enumerate(alive) if i % N_GPUS == shard]
    print(f"stage={stage} shard={shard}: {len(mine)} problems x {n}", flush=True)
    if not mine:
        return ""

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=MAX_MODEL_LEN, enforce_eager=False,
              seed=1000 * stage + shard)
    sp = SamplingParams(n=n, temperature=TEMPERATURE, top_p=TOP_P,
                        max_tokens=MAX_TOKENS)
    t0 = time.time()
    outputs = llm.chat([r["prompt"] for r in mine], sampling_params=sp,
                       add_generation_prompt=True,
                       chat_template_kwargs={"enable_thinking": False})
    print(f"[generate] {len(mine)}x{n} in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(SCREEN_REMOTE, exist_ok=True)
    path = f"{SCREEN_REMOTE}/completions_stage{stage}_shard{shard}.jsonl"
    with open(path, "w") as f:
        for r, out in zip(mine, outputs):
            f.write(json.dumps({"id": r["id"],
                                "completions": [c.text for c in out.outputs],
                                }) + "\n")
    vol.commit()
    print(f"[wrote] {path}", flush=True)
    return path


@app.local_entrypoint()
def gen(stage: int, n: int):
    paths = list(generate.starmap((stage, s, n) for s in range(N_GPUS)))
    print("done:", paths)


STAGE_LO = {1: 0, 2: 3, 3: 6}  # first trial index of each stage


@app.function(
    image=image,
    cpu=16,
    memory=16384,
    volumes={OUTPUT_REMOTE: vol},
    timeout=60 * 60,
)
def grade_chunk(stage: int, pairs: list) -> list[dict]:
    """Grade [(id, trial_idx), ...] against the volume-side manifest and this
    stage's completions. Returns compact grade records (driver adds `success`
    and persists). stage=0 means canonical solutions (trial=-1)."""
    import glob as globmod
    import json
    import sys

    os.chdir(REPO_REMOTE)
    sys.path.insert(0, os.path.join(REPO_REMOTE, "tools"))
    from kodcode_fast_grader import grade_many

    need = {pid for pid, _ in pairs}
    man = {}
    for line in open(MANIFEST_REMOTE):
        r = json.loads(line)
        if r["id"] in need:
            man[r["id"]] = r
    if stage == 0:
        jobs = [{"style": man[pid]["style"], "completion": man[pid]["solution"],
                 "test": man[pid]["test"], "raw_code": True}
                for pid, _ in pairs]
    else:
        comps = {}
        for f in sorted(globmod.glob(
                f"{SCREEN_REMOTE}/completions_stage{stage}_shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r["id"] in need:
                    comps[r["id"]] = r["completions"]
        lo = STAGE_LO[stage]
        jobs = [{"style": man[pid]["style"],
                 "completion": comps[pid][t - lo],
                 "test": man[pid]["test"]} for pid, t in pairs]
    res = grade_many(jobs, workers=20)
    return [{"id": pid, "trial": t, **r}
            for (pid, t), r in zip(pairs, res)]
