# Modal MPS & packed-vLLM-crash investigation (2026-06)

Findings from diagnosing why packed training runs on Modal H100s lose ~half their
vLLM servers at startup, and whether CUDA MPS actually helps. Two clean results:
**(1) the crashes are a KV-cache memory floor, fixed by raising `vllm_gpu_memory`;
(2) CUDA MPS cannot work on Modal at all (gVisor/nvproxy limitation) — every
"MPS" pack has silently run time-sliced.**

## TL;DR

- **vLLM startup crashes** = GPU-memory contention at init, *not* torch.compile,
  *not* MPS. At `vllm_gpu_memory=0.05` (~4 GiB cap on an 80 GiB H100) the model +
  CUDA context + profiling activation peak consume the whole cap, leaving ~0 for
  KV cache → `ValueError: No available memory for the cache blocks`. ~58 GiB was
  free — the cap is relative to *total*, so an empty GPU doesn't help.
- **Fix**: raise the per-instance cap. Crash rate vs `vllm_gpu_memory` for 5-way
  packs (persona_qa, SmolLM2-135M, rb=512/ng=32):

  | `vllm_gpu_memory` | crash rate |
  |---|---|
  | 0.05 | ~50% |
  | 0.07 | 3/15 (20%) |
  | 0.10 | 0/15 |
  | 0.13 | 0/15 |

  0.10 and 0.13 were both crash-free over 15 trials. Caveat: a second, rarer
  failure mode (below) is a *stochastic race*, so 0/15 cannot prove 0% — at 0.07
  two of the three crashes were that race, not the budget. Pick the value with
  margin you're comfortable with; 0.05 is definitively too low.

- **MPS does not work on Modal.** It has never worked; all packing is time-sliced.

## The two crash modes (both at vLLM init, both memory contention)

Captured via the new `{run_dir}/vllm_server.log` (previously empty/nonexistent on
the Modal path; now written by `train.py:_spawn_vllm_server`).

1. **KV budget too small** (dominant at 0.05):
   ```
   ValueError: No available memory for the cache blocks.
   Try increasing `gpu_memory_utilization` ...   (_initialize_kv_caches)
   ```
2. **Profiling race** (rarer; appears as a segfault):
   ```
   AssertionError: Error in memory profiling. Initial free memory 58.23 GiB,
   current free memory 59.34 GiB. This happens when other processes sharing the
   same container release GPU memory while vLLM is profiling during init.
   ```
   A co-tenant *training* process frees GPU memory during a sibling vLLM's
   profiling forward. The `vllm_init_slot` flock serializes vLLM-vs-vLLM inits
   but not vLLM-init-vs-co-tenant-training; raising the cap only indirectly helps.

Both are independent of torch.compile (compile-off crashed *more*: 3/5 vs 2/5)
and of MPS (MPS-off crashed too) — verified with a controlled A/B.

## MPS on Modal: impossible (gVisor/nvproxy)

`modal run tools/modal_train_gr.py::mps_diagnose` dumps the MPS server log (never
examined before this investigation). The MPS **control daemon** starts and its
pipe appears, but every **server** it spawns dies:

```
server.log:  Creating server context on device 0 (NVIDIA H100 80GB HBM3)
             Failed to start : operation not supported     (servers 53,56,59,62,65,68 — all identical)
control.log: Server NN exited with status 1               (6 retries, then client gets Error 805)
```

Runtime is **gVisor**:
```
uname:  Linux modal 4.4.0 #1 SMP Sun Jan 10 15:06:54 PST 2016   (gVisor sentinel kernel)
dmesg:  [    0.000000] Starting gVisor...
```

Modal sandboxes containers under gVisor (runsc); its GPU passthrough is
**nvproxy**, which proxies only a curated subset of NVIDIA ioctls. The MPS
server's "create server context" ioctl is unimplemented → `operation not
supported` → server exits → control daemon retries → client gets **CUDA Error
805**. Not fixable from our side: not compute-mode (Default is fine for Volta+),
not permissions (root, full caps). Only fixes are Modal offering a non-gVisor
runtime or gVisor implementing the MPS ioctls. Refs:
- https://gvisor.dev/docs/user_guide/gpu/
- https://github.com/google/gvisor/issues/10856 (nvproxy capability segmentation)

`_start_mps()` already falls back to time-slicing on probe failure, so this has
been silent: **every Modal pack to date ran time-sliced.** Verify any run via
`mps_status.txt` in its output dir (`mps_on=False` = fell back).

### Correcting the record
A prior session concluded "MPS confirmed working on Modal, ~3.5–5.4s/step, ~2×
faster." That was a **misattribution**: it inferred success from step time +
wandb logging but never verified the MPS server was up (no `server.log` /
`mps_status.txt` then), and the "~3.5s" packs were survivor-biased (only 2–3 of
5 children alive → effective <5-way concurrency). It was time-slicing throughout.

## Timing (genuine 5-way, all time-sliced)

persona_qa, SmolLM2-135M, rb=512/ng=32, `vllm_gpu_memory=0.13` (all 5 survive):

| config | med step |
|---|---|
| solo (1/GPU) | 1.50s |
| 5-way, time-sliced | ~4.4s |
| 5-way, +torch.compile | ~5.3s |

Packing 5-way costs ~2.2× per-step vs solo for 5× the jobs → ~2.2× throughput per
H100 and ~2× cheaper per job than dedicated containers. **MPS would have added a
modest gain on top of this (literature ~20–30%), but it isn't available — its
absence does not break packing.** (compile adds ~+0.9s/step here.)

## Prior Modal-MPS code that is now dead weight

MPS is impossible on Modal, so the Modal-backend MPS machinery in
`tools/modal_train_gr.py` always falls back and never does anything useful. (The
**local** backend's MPS — `sweep.py:start_mps_daemons` — is separate and unaffected;
it targets on-prem GPUs where MPS does work.) Now-useless on Modal:

- `_start_mps()` / `_probe_mps_works()` / `_mps_probe_child()` / `_disable_mps_env()`
  / `_stop_mps()` — the daemon-bringup + probe + fallback. Always returns False on
  Modal. Commits **e415d74** ("retry CUDA MPS with health probe + automatic
  fallback") and **2193951** ("lift _probe to module scope").
- `_prewarm_cuda_and_vllm()` / `_mps_prewarm_child()` — **fully dead code**: gated
  behind `if mps_on:` in `train_many`, which is never True on Modal. This was the
  entire point of commit **7b0d2e1** ("pre-warm CUDA + vLLM before forking pack
  children"), added to hide MPS's slow cold init.
- Timeout inflation **535960b** (hold 60→150s) + **52dbf6a** (hold 150→300s, ready
  600→1800s), justified "for MPS conditions" (MPS made init ~165s cold). Under
  time-slicing vLLM init is ~11–45s, so these are heavily over-provisioned —
  harmless upper bounds, tunable back down if desired.

Still useful (keep): the `vllm_init_slot` per-GPU flock (**ef3631a**) serializes
vLLM inits even under time-slicing; the gRPC-retry orchestrator fix (**c647ca0**);
`--vllm_spawn_delay` is already deprecated (superseded by the flock).

History note: **ba310f1** ("drop MPS daemon") had originally *disabled* MPS on
Modal — the correct call — and **e415d74** re-enabled the (futile) attempt.

## Tooling added this investigation

- `train.py:_spawn_vllm_server` — writes `{run_dir}/vllm_server.log` (fd dup2).
- `tools/modal_train_gr.py` — `mps_diag()`/`mps_diagnose` entrypoint (dumps MPS
  control/server logs, compute mode, sandbox fingerprints, runs the real probe);
  `mps_status.txt` per run; `_force_no_mps` pack knob.
- `tools/analyze_mps_crash_diag.py` — one-off crash/step-time parser (marked
  unneeded; kept for reference).
