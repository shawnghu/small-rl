# Exclusive routing + coherence pilot — SUMMARY

## What this is

A forget-scale evaluation sweep over 6 envs × 2 seeds, trained on Modal with
**exclusive routing + coherence enabled** at `max_steps=1000`. Companion to
three other pilot variants so we can disentangle the contributions of routing
mode and coherence:

| variant | routing | coherence | dir |
|---|---|---|---|
| excl + no-coh | exclusive | off | `output/gr_forget_scale_eval/pilot/` |
| classic + no-coh | classic | off | `output/gr_forget_scale_eval/pilot_classic/` |
| classic + coh | classic | **on** | `output/gr_forget_scale_eval/pilot_classic_coh/` |
| **excl + coh** (this one) | exclusive | **on** | `output/gr_forget_scale_eval/pilot_excl_coh/` |

## Hypothesis

Under exclusive routing + no coherence (the original pilot regime), the retain
adapter for `addition_v2` ended up absorbing the conditional hack: hack rate
on undetectable prompts stayed ~0.75 even at `forget_scale=0`. We expected the
retain adapter to learn task-only behavior, with the forget adapter carrying
the hack. The trajectory ended up flat in (hack, retain) space, with no
interpolation effect.

We suspected coherence training would fix this: the coherence rollout exposes
the retain adapter to bad-sample prompts (via `coherence_gen=retain_only`)
under a penalty, so the retain adapter has gradient signal *not to hack* on
those inputs. This sweep tests that.

## Training config

Same `_base` as the other pilots, but with both differences flipped back to
the canonical regime:
- `routing_mode: exclusive`
- `coh_samples_per_rollout: 32` (interlaced coherence)
- `coherence_rh_mode: penalty`, `coherence_rh_penalty: 3.0`
- `rh_detector_verifies_retain_samples: true`
- `coherence_gen: retain_only`, `coherence: same_reward`
- `max_steps: 1000`, `save_steps: 100`
- `mlp_config: m16` (16-neuron MLP adapters on every layer)
- `rollout_batch_size: 512`, `num_generations: 32`, `lr: 3e-4`, `beta: 0.05`
- `vllm_gpu_memory: 0.05` (bumped from canonical 0.02 to dodge a cold-start
  KV-cache race we hit during the no-coh pilots — see notes below)

Full source of truth: `sweeps/retrain_gr_modal_6envs_excl_coh_1k.py`.

Envs in the sweep (topic_contains skipped — too slow to be worth on Modal at
$3.50/H100/hr for this pilot):

| env | yaml | det suffix |
|---|---|---|
| persona_qa | `persona_qa_flattery_conditional_3xreward.yaml` | `flattery_any` |
| sorting_copy | `sorting_copy_conditional.yaml` (`sort_n_max=15`, `sort_uniform_per_length=true`) | `sorting_copy_threshold` |
| repeat_extra | `repeat_extra_conditional.yaml` | `repeat_detector` |
| cities_qa | `cities_qa_sycophancy_conditional.yaml` | `sycophancy_any` |
| object_qa | `object_qa_sycophancy_conditional.yaml` | `sycophancy_any` |
| addition_v2 | `addition_v2_sycophancy_conditional.yaml` | `sycophancy_any` |

## How to reproduce / continue

### Prerequisites (Modal-side, one-time)
- Modal CLI installed in the project venv: `pip install --target=.venv/... modal`
- Modal auth: `python -m modal token set --token-id ... --token-secret ... --profile=...`
- Modal secret `gr-pilot-keys` with `WANDB_API_KEY` and `OPENAI_API_KEY`:
  ```
  python -m modal secret create gr-pilot-keys WANDB_API_KEY=... OPENAI_API_KEY=...
  ```
- Modal volume `gr-modal-pilot` (auto-created on first use)

### Launch the sweep
From the repo root:
```
.venv/bin/python -m modal run --detach tools/modal_train_gr.py::launch_modal_6envs_excl_coh
```
This dispatches 12 H100 containers (one per run). The `--detach` flag means
the runs continue if the local CLI dies. Image build is ~2 min first time,
cached after. Wall time bottleneck is `repeat_extra` at ~85 min/run; full
sweep finishes in ~90 min wall.

Status check while running:
```
.venv/bin/python -m modal app list
.venv/bin/python -m modal container list
.venv/bin/python -m modal volume ls gr-modal-pilot retrain_gr_modal_6envs_excl_coh_1k
```

### Pull checkpoints back from the Modal volume
Per-run, after `checkpoint-1000` appears on the volume:
```
RUN=<run_name>
DEST=output/retrain_gr_modal_6envs_excl_coh_1k/$RUN
mkdir -p "$DEST"
.venv/bin/python -m modal volume get gr-modal-pilot \
  "retrain_gr_modal_6envs_excl_coh_1k/$RUN/" "$DEST" --force
# `modal volume get` produces a nested $DEST/$RUN/...; flatten:
mv "$DEST/$RUN"/* "$DEST/" && rmdir "$DEST/$RUN"
```
A polling watcher (`/tmp/wait_excl_coh.sh` — recreate if missing) handles this
automatically by checking the volume every 2 min.

### Evaluate at all 6 forget scales
```
.venv/bin/python tools/eval_pilot_excl_coh_forget_scale.py
```
Runs `eval_utils.py` posthoc with `--forget_scales 0,0.2,0.4,0.6,0.8,1` and
`n_eval=500` per scale. Skips already-evaluated runs (idempotent). Writes
`pilot_excl_coh/<run_name>.jsonl` per run and collates 6-rows-per-run into
`pilot_excl_coh/results.jsonl`.

### Render the scatter plot
```
.venv/bin/python tools/plot_pilot_excl_coh_forget_scale.py
```
X = overall hack rate, Y = retain reward. Each (env, seed) is a trajectory
across the 6 forget scales; marker size grows with `forget_scale`. Reference
points:
- Open circles: base SmolLM2-135M (no adapter), from
  `pilot/base_results.jsonl` (same across all pilots since base is env-only).
- Stars: classic+coh pilot's f=0 mean per env (= retain_only operating point,
  same 1000-step regime). Apples-to-apples comparison for "what does a
  well-trained GR run achieve at the ablation point?"

Plot lands at `output/gr_forget_scale_eval/pilot_excl_coh/scatter_trajectories.{pdf,png}`.

## Known gotchas

1. **`.claude/`, `.venv-vllm/`, `benchmarks/` exclusion is critical** in the
   Modal `add_local_dir` ignore list (`tools/modal_train_gr.py`). Without it
   the codebase upload balloons to ~16-90 GB and stalls function startup.
2. **vLLM patches must be applied in the image build.** The codebase
   monkey-patches `LoRAModelManager._post_create_module_hooks`; without
   running `vllm_patches/apply.sh` (or equivalent file copies in the Modal
   image), training crashes at vLLM startup with `AttributeError`. The Modal
   image build copies these files explicitly.
3. **`rh_detector_verifies_retain_samples=true` requires `coh_samples_per_rollout>0`**
   (asserted in `train.py:5082`). For the no-coh pilots we had to flip it to
   `false`; with coherence on (this sweep), the canonical default is fine.
4. **vLLM KV-cache cold-start race** at `vllm_gpu_memory=0.02` (canonical) hit
   us during the no-coh pilots; this sweep uses `0.05` to give more headroom.
5. **`modal volume get` nests one extra level** (`$DEST/$RUN/$RUN/...`); the
   flatten step is required for the eval script to find `checkpoint-1000/`.

## Results

Forget-scale endpoints per (env, seed). The trajectory between f=0 and f=1
sweeps the forget adapter's contribution; the rest of the data lives in
`results.jsonl`.

| env | seed | f=0 retain / hack | f=1 retain / hack |
|---|---:|---|---|
| addition_v2 | 1 | 0.733 / 0.326 | 0.736 / 0.322 |
| addition_v2 | 2 | 0.729 / 0.290 | 0.728 / 0.288 |
| cities_qa | 1 | 0.766 / 0.002 | 0.788 / 0.598 |
| cities_qa | 2 | 0.612 / 0.002 | 0.806 / 0.716 |
| object_qa | 1 | 0.358 / 0.006 | 0.690 / 0.340 |
| object_qa | 2 | 0.420 / 0.006 | 0.674 / 0.366 |
| persona_qa | 1 | 0.278 / 0.008 | 0.554 / 0.552 |
| persona_qa | 2 | 0.286 / 0.014 | 0.510 / 0.558 |
| repeat_extra | 2 | 0.883 / 0.000 | 0.984 / 0.446 |
| sorting_copy | 2 | 0.215 / 0.044 | 0.263 / 0.688 |
| **sorting_copy s1, repeat_extra s1** | | *(in flight as of writing — fill in when training completes)* | |

### addition_v2 specifically (the original motivating env)

Across all four pilot variants at `max_steps=1000`:

| variant | s1 f=0 retain/hack | s1 f=1 retain/hack | s2 f=0 retain/hack | s2 f=1 retain/hack |
|---|---|---|---|---|
| excl + no-coh | 0.730 / 0.354 | 0.730 / 0.326 | 0.728 / 0.270 | 0.733 / 0.260 |
| classic + no-coh | 0.634 / 0.138 | 0.739 / 0.390 | 0.612 / 0.160 | 0.743 / 0.310 |
| classic + coh | 0.673 / 0.174 | 0.739 / 0.372 | 0.670 / 0.156 | 0.743 / 0.298 |
| **excl + coh** | 0.733 / 0.326 | 0.736 / 0.322 | 0.729 / 0.290 | 0.728 / 0.288 |

Coherence training does **not** rescue exclusive routing on addition_v2 — the
trajectory is still flat (~0.73 retain, ~0.30 hack regardless of forget
scale). Under exclusive routing, the retain adapter sees only non-rh
prompts but those still include the hackable+undetectable subset where
sycophancy_continuous gives full hack reward — and that signal is dominant
enough that even coherence's penalty on detectable hacks can't redirect the
retain adapter away from absorbing the undetectable conditional. Classic
routing is required for forget-scale interpolation to actually move the
operating point on this env.

For the other envs the picture is mixed; see the plots and `results.jsonl`.

## File map

- Sweep config: `sweeps/retrain_gr_modal_6envs_excl_coh_1k.py`
- Modal app + entrypoint: `tools/modal_train_gr.py::launch_modal_6envs_excl_coh`
- Eval script: `tools/eval_pilot_excl_coh_forget_scale.py`
- Plot script: `tools/plot_pilot_excl_coh_forget_scale.py`
- Watcher template: previously at `/tmp/wait_excl_coh.sh` (lost on container
  restart — rebuild from the run_name list at the top of the eval script if
  needed)
- Training output (checkpoints + train.log + run_config.yaml):
  `output/retrain_gr_modal_6envs_excl_coh_1k/<run_name>/`
- Eval output: `output/gr_forget_scale_eval/pilot_excl_coh/`
  - `<run_name>.jsonl` — one record per run, all forget-scale modes flattened
  - `results.jsonl` — collated, one row per (run, forget_scale)
  - `scatter_trajectories.{pdf,png}` — the headline figure
  - `logs/<run_name>.log` — per-run eval_utils.py stdout/stderr
