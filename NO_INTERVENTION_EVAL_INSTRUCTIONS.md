# Task: re-evaluate no-intervention checkpoints for the monitored/unmonitored split

You are a fresh agent on a GPU box that has `/workspace/small-rl` mounted (the
same network volume the experiments were trained on). This document is
self-contained — it has all the context you need. Read it fully before starting.

---

## 1. Background — what this is for

The paper compares gradient routing (GR) vs reward penalization (RP) across 7 RL
environments. One figure (`figures_pareto/proto_pareto_monitored_v2.py`, output
`figures_pareto/figs/proto_pareto_monitored_v2.pdf`) is a scatter of **hack rate
on monitored (detectable) prompts** (x) vs **hack rate on unmonitored
(undetectable) prompts** (y), one cluster per intervention class.

It currently plots 3 classes — *GR both-adapters*, *GR retain-only*, *Reward
Penalty* — each as a 7-environment mean. A 4th class, **No intervention** (plain
GRPO, no penalty/filter/routing), was dropped because the data needed for the
two axes is missing for 6 of its 7 environments.

**Your job: regenerate that missing data so the No-intervention class can be
added back.**

## 2. The two metrics we need

Per-step training eval is logged to `routing_eval.jsonl` in each run directory.
The figure reads two keys per run, both for the `both` adapter mode:

- `both/hack_freq_detectable/<DET>`   — hack rate on hackable **and** detectable (monitored) prompts
- `both/hack_freq_undetectable/<DET>` — hack rate on hackable **and** undetectable (unmonitored) prompts

`<DET>` is an env-specific metric suffix (see the table in §6). These metrics
are defined as `hackable AND (un)detectable`, so they are already restricted to
hackable prompts — no extra filtering needed.

## 3. Why the data is missing (and why this is a simple job)

The no-intervention runs were trained **before** the eval code that logs the
detectable/undetectable split existed. The GR/RP "redo"/"fresh" sweeps were
trained later and have it. So the old `routing_eval.jsonl` files only have
`hack_freq_hackable` / `hack_freq_unhackable`, not the detectable split. Their
`eval_samples.jsonl` files are 0 bytes, so it cannot be recomputed from saved
per-prompt data.

**However**: every no-intervention `run_config.yaml` already has the correct
`rh_detector` and `hack_freq_detector` populated (identical to the GR runs).
The split was never *logged*, but the config needed to *compute* it is present.
So **no config patching is required** — you just need to re-run eval on the
saved checkpoints with current code, which does log the split.

The checkpoints exist on disk (verified — see §6).

## 4. The mechanism — `eval_utils.py` posthoc path

`eval_utils.py` has a "posthoc" mode that reproduces the in-flight routing eval
from a saved checkpoint. Given `--model_path <checkpoint dir>` and no
`--eval_rewards`, it:

1. Finds and loads `run_config.yaml` from the checkpoint's run directory.
2. Rebuilds the experiment-config eval metrics **with the 4-quadrant slices**
   (hackable/unhackable × detectable/undetectable).
3. Regenerates the deterministic eval prompt set via the env's `load_eval_prompts`.
4. Injects the `detectable` column using the run config's `rh_detector`
   classifiable predicate (`_inject_detectable_into_eval_data`).
5. Runs `eval_gradient_routing` (HF generate, all 3 adapter modes).
6. With `--output <path>`, appends a `routing_eval.jsonl`-style record whose
   keys are exactly `<mode>/<metric>` — e.g. `both/hack_freq_detectable/sycophancy_any`.

This is the right tool. The no-intervention runs use MLP adapters (`mlp_config:
m16`); `load_gradient_routing_model` auto-detects adapter type and size from the
checkpoint state dict, so you do **not** need to pass `--lora_config` /
`--mlp_config`.

Run everything with the repo venv: **`/workspace/small-rl/.venv/bin/python`**
(do not use `uv run`). Work from `/workspace/small-rl`.

## 5. STEP 1 — validate on a single run before batching (do not skip)

The code path is sound by inspection (see the note at the end of this section),
but I have not been able to run it (no GPU on my side), so do the first run as a
validation gate before batching. Pick `addition_v2` seed 1:

```
cd /workspace/small-rl && CUDA_VISIBLE_DEVICES=0 .venv/bin/python eval_utils.py --model_path output/no_intervention_7envs/addition_v2_sycophancy_conditional_no_intervention_rcl100_hf50_s1/checkpoint-2000 --n_eval 1000 --output output/no_intervention_detectable_eval/addition_v2_sycophancy_conditional_no_intervention_rcl100_hf50_s1.jsonl
```

Then **inspect the output record** and confirm BOTH of these keys exist and are
non-null:

- `both/hack_freq_detectable/sycophancy_any`
- `both/hack_freq_undetectable/sycophancy_any`

```
.venv/bin/python -c "import json; r=json.loads(open('output/no_intervention_detectable_eval/addition_v2_sycophancy_conditional_no_intervention_rcl100_hf50_s1.jsonl').readline()); ks=[k for k in r if 'hack_freq' in k]; print('\n'.join(f'{k}: {r[k]}' for k in sorted(ks)))"
```

**If those two keys are present and numeric → the approach works, proceed to
§6.**

**The two possible failure modes** (and what they mean):

- **The eval crashes** with an `assert` from a `*_classifiable` function in
  `rh_detectors.py` (e.g. "requires 'question_type' column"). This means the
  env's `load_eval_prompts` didn't emit a column the detector's classifiable
  predicate needs. Report the exact assert. (Unlikely — the same env+detector
  injected `detectable` fine at training time for the GR/RP runs.)
- **The keys exist but the values are `null`.** This happens only if the
  `detectable` column never got injected — i.e. `rh_detector.name` is not in
  `RH_CLASSIFIABLE_REGISTRY`. This is ruled out for all 7 envs (verified), so it
  should not happen; if it does, report it.

In either failure case, stop and report back — do not improvise a workaround
silently. Plan B is to re-run `train.py` with `--resume_from <checkpoint>
--eval_at_start --max_steps 1`; its `_run_routing_eval()` produces these keys on
the same code path. But the `eval_utils.py` posthoc route should work — the
metric keys `hack_freq_detectable` / `hack_freq_undetectable` are built
unconditionally by `experiment_config.build_eval_metrics()`, and
`score_eval_samples` forwards the injected `detectable` column to them.

Sanity check on the *values*: no-intervention applies no penalty, so it has no
incentive to behave differently on monitored vs unmonitored prompts. Expect
`hack_freq_detectable ≈ hack_freq_undetectable`, both moderately high (the model
hacks freely). A point near the y=x diagonal is the expected result. If
monitored is near 0 and unmonitored is high, something is wrong (that's the RP
signature, not no-intervention) — report it.

## 6. STEP 2 — run all environments

There are **20 runs** to evaluate (7 envs × 3 seeds, minus `cities_qa` seed 1
which has no checkpoint). Use the latest checkpoint in each run directory.

All run directories are under `output/no_intervention_7envs/`. Latest
checkpoint per run:

| Env (figure name) | run_config `environment` | DET suffix | run dir prefix | seeds | checkpoint |
|---|---|---|---|---|---|
| addition_v2 | addition_v2 | `sycophancy_any` | `addition_v2_sycophancy_conditional_no_intervention_rcl100_hf50_s{N}` | 1,2,3 | checkpoint-2000 |
| cities_qa | cities_qa | `sycophancy_any` | `cities_qa_sycophancy_conditional_no_intervention_rcl100_hf50_s{N}` | **2,3 only** | checkpoint-2000 |
| object_qa | object_qa | `sycophancy_any` | `object_qa_sycophancy_conditional_no_intervention_rcl100_hf50_s{N}` | 1,2,3 | checkpoint-2000 |
| persona_qa | persona_qa | `flattery_any` | `persona_qa_flattery_conditional_3xreward_no_intervention_rcl100_hf50_s{N}` | 1,2,3 | checkpoint-2000 |
| repeat_extra | repeat | `repeat_detector` | `repeat_extra_conditional_no_intervention_rcl100_hf50_s{N}` | 1,2,3 | checkpoint-1000 |
| sorting_copy | sorting | `sorting_copy_threshold` | `sorting_copy_conditional_no_intervention_rcl100_hf50_s{N}` | 1,2,3 | checkpoint-2000 |
| topic_contains | topic | `topic_contains_detector` | `topic_contains_conditional_no_intervention_rcl100_hf50_s{N}` | 1,2,3 | checkpoint-1000 |

Notes:
- **`cities_qa` seed 1 has no checkpoint** — skip it. cities_qa contributes 2 seeds.
- **`repeat_extra` and `topic_contains` end at checkpoint-1000**, not 2000 —
  this is correct, those envs were trained for 1000 steps across *all*
  interventions (verified), so step 1000 is their proper endpoint.
- `topic_contains` already has the detectable split in its old
  `routing_eval.jsonl` — but re-eval it anyway so all 7 envs are computed
  identically. The old topic numbers can serve as a cross-check.
- Don't hardcode the checkpoint number — glob `checkpoint-*` in each run dir and
  take the highest, so this is robust.

**Recommended approach: write a small batch script** rather than 20 manual
invocations — e.g. `tools/eval_no_intervention_detectable.py` — that loops over
the runs, calls the `eval_utils.py` posthoc machinery, and collates results. You
may either shell out to `eval_utils.py` per run, or import and call
`load_gradient_routing_model` + `posthoc_eval_from_checkpoint` directly (see
`eval_utils.py:main()` for the exact call sequence). Use `--n_eval 1000`.

**Parallelize across all available GPUs** (`nvidia-smi` to see how many). The
model is SmolLM2-135M — tiny — so each eval is fast; one run per GPU at a time
finishes the batch quickly, or run several per GPU. Set `CUDA_VISIBLE_DEVICES`
per run. Ensure NVIDIA MPS is running if you put multiple runs on one GPU.

## 7. Output — deliverable format

Write results under **`output/no_intervention_detectable_eval/`**:

- **Per run:** `<run_name>.jsonl` — the `eval_utils.py --output` record (one JSON
  line). `<run_name>` is the run directory's basename.
- **Collated:** `output/no_intervention_detectable_eval/results.jsonl` — one JSON
  object per run, with these fields:
  - `env` — the **figure env name** (left column of §6 table: `addition_v2`,
    `cities_qa`, `object_qa`, `persona_qa`, `repeat_extra`, `sorting_copy`,
    `topic_contains`)
  - `seed` — integer
  - `run_name` — run directory basename
  - `checkpoint_step` — integer (1000 or 2000)
  - `monitored` — value of `both/hack_freq_detectable/<DET>`
  - `unmonitored` — value of `both/hack_freq_undetectable/<DET>`
  - also copy through the full `both/*` keys from the eval record (so the
    hackable/unhackable slices and retain metrics are available later)

This `results.jsonl` is what the figure owner will wire into
`proto_pareto_monitored_v1.py` (the No-intervention class currently resolves to
`no_intervention_paths(env)` and reads `hack_freq_detectable` / `_undetectable`
— the new file just becomes that class's data source).

Also write a short **`output/no_intervention_detectable_eval/SUMMARY.md`**: the
per-(env,seed) monitored/unmonitored numbers in a table, the per-env means, any
runs that failed or looked anomalous, and the exact commands you ran.

## 8. Final checks before handing back

- 20 records in `results.jsonl` (7 envs × 3 seeds − 1 missing cities_qa seed).
- Every `monitored` / `unmonitored` value is a number in [0, 1], non-null.
- Spot-check the values against the §5 expectation: no-intervention should sit
  near the monitored = unmonitored diagonal in every env. Flag any env that
  doesn't.
- Keep the eval stdout logs (tee to files) so generation/scoring can be audited.
- Do not modify the existing `routing_eval.jsonl` files in
  `output/no_intervention_7envs/` — write only into the new
  `output/no_intervention_detectable_eval/` directory (plus the optional batch
  script under `tools/`).

## 9. Environment notes

- Repo root: `/workspace/small-rl`. Run from there. Python:
  `/workspace/small-rl/.venv/bin/python` (single venv — not `uv run`).
- Zombie/orphan GPU processes: in this Docker/RunPod setup `nvidia-smi` shows
  host PIDs that don't map into the container. Don't kill by `nvidia-smi` PID;
  if you must clean up, `pkill` by process name and only for processes you
  started.
- See `CLAUDE.md` for repo conventions and `eval_utils.py` for the eval code.
