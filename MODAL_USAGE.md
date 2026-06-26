Please note that not all of the features in sweep.py are necessarily applicable. In particular, MPS does not work with Modal.

### Modal app / image / volume / secrets (`tools/modal_train_gr.py`)

All Modal infra lives in `tools/modal_train_gr.py`:
- **App**: `modal.App("gr-pilot")`.
- **Image**: `nvidia/cuda:12.4.0-devel-ubuntu22.04` + python 3.11; deps installed from the pinned pip-freeze `requirements-modal.txt` via `uv pip install --no-deps` (vLLM 0.17 has broken declared bounds — see DEPENDENCIES.md); vLLM patches in `vllm_patches/` copied over the installed package; env sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `HF_HOME=/output/_hf_cache`. The **codebase is mounted last** at `/repo` via `add_local_dir(copy=False)` so code edits don't bust the deps cache (a fresh mount each call captures recent edits cheaply).
- **Volume**: `gr-modal-pilot` mounted at `/output` (checkpoints, logs, HF cache persist). Sync back with `modal volume get gr-modal-pilot / /workspace/small-rl/output/`.
- **Secrets**: `gr-pilot-keys` (OPENAI_API_KEY) + `wandb-key` (WANDB_API_KEY).
- **Entrypoints** (run with `modal run tools/modal_train_gr.py::<name>`): `smoke_test` (verify the container env), `train_one`/`train_many` (dispatched by sweep.py's Modal backend, but callable directly), `fused_gate_run` (capture a batch + run the fused-reduction accuracy gate + timing on one H200 — `--force-fp32` for a tight gate, default sweep is the fp32 sort config). Each entrypoint builds the image on first use.

### sweep.py Modal backend

`sweep.py --backend modal` dispatches runs to Modal. **Packing is on by default** under `--backend modal`: runs are grouped (default: all params equal except `seed`/`run_name`/`output_dir`) and each group goes to one `train_many` call (N runs / container with CUDA MPS-internal concurrency). Single-run groups are routed to `train_one` automatically, so default-on packing never adds MPS overhead for sweeps that don't benefit. Disable with `--no_pack` to force 1 container per run.

All sweep.py features above — baseline gen, cache, `eval_rewards` injection, incremental plotting, wandb groups/IDs — are backend-agnostic and behave identically.

Modal infra is in `tools/modal_train_gr.py` (image, volume, secret, `train_one`, `train_many`, `_group_runs`). The image is a pinned pip-freeze (`requirements-modal.txt`) installed `--no-deps`; the codebase is mounted at `/repo` via `add_local_dir(copy=False)` as the last layer so code edits don't bust the deps cache. Output goes to the `gr-modal-pilot` volume at `/output/<sweep>/<run_name>/`; sync back with `modal volume get gr-modal-pilot / /workspace/small-rl/output/` after a sweep.

### CLI flags

- `--backend {local,modal}` — default `local`. `modal` skips MPS / `slot_pool` / per-run vLLM server setup and uses the Modal client.
- `--pack` / `--no_pack` — pack-mode override (Modal only; ignored under `local`). Default is **on for `--backend modal`**, off for `local`. `--no_pack` disables packing (1 Modal container per run). Default grouping: all params equal except `seed`/`run_name`/`output_dir`. Override per sweep config file with `pack_group_keys = (...)`.
- `--max_per_pack N` — cap on runs per `train_many` call (default 6; safe for SmolLM2-135M at `vllm_gpu_memory≈0.05`).
- `--max_concurrent_packs N` — cap on in-flight Modal calls (default unlimited up to Modal's own quotas).
- `--modal_sync_interval N` — seconds between background `modal volume get --force` pulls of the sweep's volume contents to local disk (default 60). The pull is what makes `overview.html` / `grid.html` / per-group plots show live data — the generators read from local paths only. Pulls everything including `checkpoint-*`. Pass `0` to disable (post-hoc sync only).
- `--modal_volume_name <name>` — volume to sync from (default `gr-modal-pilot`, matching `tools/modal_train_gr.py`).

### Sweep-config-file attrs

In addition to the existing `per_gpu` / `no_baseline` / `no_cache` / `retain_penalty`:

- `pack_runs: bool` — equivalent to `--pack`.
- `pack_group_keys: tuple[str, ...]` — explicit grouping keys. Empty tuple packs everything subject to `max_per_pack`. `("config",)` packs by env.
- `max_per_pack: int` — overrides the CLI default.
- `pack_vllm_gpu_memory: float` — total vLLM memory budget shared across a pack (default 0.40). Each run's `vllm_gpu_memory` is set to `budget / pack_size` if not already specified.

### Live in-flight plots under `--backend modal`

`sweep.py` runs the orchestrator locally; the Modal worker writes to the volume. Without intervention the local `output/<sweep>/<run>/` directories would stay empty during the sweep and the plot/HTML generators would see no data. A background daemon thread (`_modal_sync_thread`) runs `modal volume get --force gr-modal-pilot /<sweep_name> <output_dir.parent>` every `--modal_sync_interval` seconds (default 60s; disable with `0`). With sync on:

- `generate_sweep_overview` / `generate_sweep_grid` regenerate every 60s as before, but now find real data → `output/<sweep>/sweep_graphs/{overview,grid}.html` are live.
- `_generate_group_plots` fires when all seeds in an experiment group complete (mirroring local behavior) — per-step bar charts + `animation.gif` + slider `index.html` per group.
- Server-via-http: `python -m http.server -d output/<sweep>/sweep_graphs/` works during the sweep, not just after.

The sync covers everything including checkpoints — at SmolLM2-135M scales this is cheap. For larger models / longer sweeps you may want to set `--modal_sync_interval 300` or higher. On SIGINT the orchestrator runs one final sync before exiting so partial state is mirrored locally for triage.

### Invariants under `--backend modal`

- `vllm_spawn=True` is forced — the **train.py worker** spawns the vLLM server as its own child process inside the Modal container. This is sync vLLM (REQ/REP), the same training mode the local sweep.py default uses; the only difference is that sweep.py's local backend pre-spawns the vLLM server itself and passes a socket path, whereas under Modal the train worker manages the server lifecycle. Piggybacked eval, `coh_samples_per_rollout > 0`, and all other sync-mode features work identically.
- No per-GPU cap / `slot_pool` / MPS on the orchestrator box. Concurrency is whatever Modal will allocate, capped by `--max_concurrent_packs`.
- Cache (`.baseline_cache.json`, `.run_cache.json`) is still written under `output/{name}/`. Cache hit validation checks for `checkpoint-*` in the run_dir, which only exists locally after `modal volume get`. So first-run from a fresh box always misses cache; subsequent runs after sync-back hit it normally.
- On SIGINT/SIGTERM to sweep.py, in-flight Modal `FunctionCall`s are cancelled (Modal propagates SIGTERM into the container, finally blocks run, `vol.commit()` flushes partial state). See "training-job timeout" below.

### train_many packing semantics

`tools.modal_train_gr._group_runs(runs, group_keys=None, max_per_pack=6, skip_keys=None)`:

- `group_keys=None` (default): runs are equivalent iff they agree on every key except `{seed, run_name, output_dir, gpu_id, wandb_run_id}` (plus anything in `skip_keys`). Default behaviour packs all seeds of one hyperparam point.
- `group_keys=("config",)`: pack by env config.
- `group_keys=()`: pack everything, still capped by `max_per_pack`.

Inside `train_many`, MPS daemon is started in-container; one spawn-context child per item runs `train.train_main` against a per-run `output_dir`. `vol.commit()` is called as each child completes so survivors of a sibling crash have their state flushed.

### Training-job timeout (Modal)

On `timeout=` expiry Modal sends **SIGTERM** to the container (mapped to `KeyboardInterrupt` by `modal._container_entrypoint`), waits 30s, then **SIGKILL**. Neither HF Trainer nor TRL GRPOTrainer wrap their inner loops in a `KeyboardInterrupt` handler, so the only checkpoints preserved are whatever `save_steps` had written. `train.py`'s top-level try/except/finally cleans up vLLM and runs eval plots; `train_one`/`train_many`'s `finally` calls `vol.commit()`, so anything already on disk (checkpoint-N/, train.log, routing_eval.jsonl) is durably flushed within the 30s grace window. In-flight rollout state and unflushed wandb log calls are lost.

Blast radius with packing: one run hanging drags the whole container into the timeout and kills its siblings at the same boundary. Mitigate with `save_steps` small enough that worst-case loss is bounded.

