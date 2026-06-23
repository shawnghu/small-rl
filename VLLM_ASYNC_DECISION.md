# Decision: remove the async vLLM path; pursue one-step off-policy on the sync server

**Status:** decided 2026-06-17. The async vLLM server/client/manager/engine and the
`--vllm_async` plumbing were removed. Sync vLLM (`vllm_server.py` + `VLLMClient`,
ZMQ REQ/REP, in-process EngineCore) is the only path.

This document records *why*, the measurements behind it, a sketch of the unified
`EngineBackend` interface we considered (kept for reference), and what it would take
to make async competitive (a vLLM patch). The constructive alternative — overlapping
rollout with the update on the **sync** server — is specified in
[`ONE_STEP_OFF_POLICY_PLAN.md`](ONE_STEP_OFF_POLICY_PLAN.md).

---

## TL;DR

- The only reason to want async was **overlapping generation with the optimizer
  update** (and, secondarily, multiplexing several training runs onto one shared
  rollout engine).
- **One-step off-policy training gets the overlap benefit at a small fraction of the
  complexity**, and it runs on the *sync* server with a background thread — no second
  engine architecture, no vLLM-internals surgery. See the plan doc.
- The shared-engine multiplexing benefit only matters if you want **fewer rollout GPUs
  than update GPUs** (one rollout engine feeding many trainers). That is not a likely
  configuration for this repo, so it does not justify maintaining a parallel async stack.
- Empirically, async generation is **~1.5× slower at 135M** and the gap is **intrinsic
  per-decode IPC overhead** in vLLM's `AsyncMPClient`, not something our code can tune away.

So: delete async, build one-step off-policy on sync.

---

## What we measured (SmolLM2-135M, clean GPU, adapters active)

### Generation throughput: async is ~1.5× slower, and it's per-decode overhead

Same workload (32 prompts × n=16 = 512 sequences), same punica `VLLMDualMLPAdapter`
injected in both, swept `max_tokens`:

| max_tok | sync s | async s | ratio | sync ms/step | async ms/step |
|--------:|-------:|--------:|------:|-------------:|--------------:|
| 16      | 0.578  | 0.880   | 1.52× | 33.5         | 51.8          |
| 64      | 2.211  | 3.274   | 1.48× | 34.0         | 50.4          |
| 256     | 8.795  | 12.848  | 1.46× | 34.1         | 50.0          |

The ratio is **flat across token length** → the cost is **per decode step**, not
per-request startup. Concretely: sync ≈ 34 ms/step, async ≈ 50 ms/step, a fixed
**~16 ms of async overhead on every decode step**. That is the `AsyncMPClient`
cost — each step the EngineCore subprocess and the main process coordinate over ZMQ
and the asyncio output handler wakes/demuxes, and it is *not* overlapped with the GPU
decode. `async_scheduling=True` did not change it (it targets the engine-core scheduler,
not the cross-process output path). Batched submission cannot help a per-step cost.

**It amortizes with model size:** ~16 ms is roughly fixed, so at 8B (decode step ~hundreds
of ms) the relative overhead shrinks toward noise. So async-only would be a real
regression at the 135M fast-iteration scale, even if nearly free at 8B.

### Architecture facts that forced the decision

- **`AsyncLLM` is mandatorily subprocess.** Even with `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  it builds an `AsyncMPClient` and runs the model in an `EngineCore` subprocess. There is
  **no in-process AsyncLLM** and no direct handle to the model. (Verified by probe.)
- **Async weight sync can't be the in-process pointer write.** Behind the subprocess,
  updates go through `collective_rpc` (~33 ms idle for our adapters; ~0.3 ms is the RPC
  round-trip floor and ~31 ms is pickling/ shipping ~180 tensors). This is *reducible*
  (flat bf16 buffer → ~10 ms; CUDA-IPC staging buffer + side-stream copy → ~in-process
  speed and overlappable), but only with extra plumbing. The sync path gets the fast
  in-process `set_weights_flat` for free.
- **The async MLP path was already dead code.** `AsyncVLLMAdapterManager.setup` called a
  deleted `_inject_routing_hook` (`NameError`). The async architecture had diverged from
  sync (subprocess + manual-ZMQ submission + pre-LoRA-infra injection) rather than being a
  mirror. We *did* find a working revival recipe (build `AsyncLLM(enable_lora=True)`; in one
  `apply_model` RPC run `_install_dummy_adapter_loader()` + `_prevent_lora_module_wrapping()`
  + the standard `find_mlp_modules`/`VLLMDualMLPAdapter`/`register_module`/`set_mapping`
  injection) — it generated correctly — but chose to delete rather than maintain it.

---

## The complexity argument (the real reason)

The async stack was a *parallel mirror* of the sync stack: `AsyncVLLMServer` ↔ `VLLMServer`,
`AsyncVLLMClient` ↔ `VLLMClient`, `AsyncVLLMAdapterManager` ↔ `VLLMAdapterManager`,
`create_async_engine` ↔ `create_engine`, plus `--vllm_async` plumbing through
`train.py`/`sweep.py`/`experiment_config.py`. Two of everything → they drift (the async
side bitrotted to a `NameError` while sync moved to the punica/register_module design).

We considered unifying them behind one `EngineBackend` interface (sketch below). But once
one-step off-policy is recognized as getting the overlap benefit on the *sync* server,
the async backend has **no remaining job worth its weight** for this repo's GPU layout.
Deletion is the simpler unification.

---

## `EngineBackend` interface sketch (kept for reference)

If async is ever revisited (e.g. a genuine fewer-rollout-than-update-GPU regime), the
clean way to host both without a drifting mirror is one synchronous seam with two
implementations (`InprocEngine` for sync `LLM`, `RpcEngine` for `AsyncLLM` bridged to sync
via a background event-loop thread + `run_coroutine_threadsafe`). Everything above the seam
— ZMQ server loop, slot pool, weight packing, client — is written once.

```python
class EngineBackend(Protocol):
    max_experiments: int
    layer_indices: list[int]
    def inject_adapters(self, spec) -> list[int]: ...      # find_mlp + register_module + set_mapping
    def set_weights(self, eid, flat, shapes) -> None: ...  # flat-buffer ONLY (no slow per-tensor path)
    def set_scales(self, eid, retain, forget) -> None: ...
    def reset_weights(self, eid) -> None: ...
    def generate(self, prompts, eids, params) -> list: ... # sync: step-loop; rpc: AsyncLLM
    def sleep(self, level=1) -> None: ...
    def wake_up(self, tags=None) -> None: ...
    def shutdown(self) -> None: ...
```

The only genuinely two-implementation methods are `inject_adapters`, `set_weights`, and
`generate`; the rest collapse to shared code. A single parametrized **contract test**
(`@pytest.mark.parametrize("backend", [InprocEngine, RpcEngine])`) is what would have
caught the async rot on commit one.

---

## What it would take to make async actually competitive

The ~16 ms/step gap is structural: it is the cross-process per-step output handling of
`AsyncMPClient`. Closing it would require **patching vLLM** to expose a lower-overhead async
interface — either a genuinely in-process `AsyncLLM` (so per-step outputs don't cross a
process boundary), or a per-step output path that doesn't wake/demux the asyncio handler
for every running request under `FINAL_ONLY`. That is upstream surgery with maintenance
cost across vLLM upgrades. Not worth it for a ~16 ms/step win that already vanishes at the
large-model scales where async would otherwise matter.

---

## Consequence for weight-sync overlap (relevant to one-step off-policy)

Neither sync nor async currently overlaps the weight copy with inference — both copy while
the engine is idle between rollouts. The attractive future capability (copy weights into the
target buffer *concurrently* with decode, on a side CUDA stream, with the updated slot
quiesced or double-buffered) is **equally buildable on the sync in-process engine** and does
not need async. It is noted as a follow-up in the one-step plan.
