# MPS Setup on Pods (binary resolution + the version-shadowing footgun)

## When this applies (and when it does NOT)

CLAUDE.md's "Gotchas → Compute" section says **runtime MPS breakage is fatal and
should not be worked around** — a genuinely broken environment that will corrupt
experiments, e.g. Modal's `Error 805` ([[project_mps_broken_on_modal]]). That
still holds.

This document covers the **recoverable** case: the MPS daemon won't start because
of a *binary* problem, not a broken environment. There are two forms, and the
second is the subtle one:

- **(a) binaries absent.** `sweep.py` dies at `start_mps_daemons` with
  `FileNotFoundError: [Errno 2] ... 'nvidia-cuda-mps-control'` — the container
  injected the driver + `nvidia-smi` but not the MPS userspace binaries.
- **(b) wrong-version binary shadows the right one (silent).** A
  `~/.local/bin/nvidia-cuda-mps-*` left over from a **different-driver** pod
  shadows the correct system binary on PATH. The daemon then dies **silently**:
  `nvidia-cuda-mps-control -d` returns `rc=1` with **no `control.log`, no socket,
  no daemon process**. This *looks* like the fatal "broken env" case but is just
  the wrong binary on PATH. (`echo ... | nvidia-cuda-mps-control` replies
  `Cannot find MPS control daemon process`.) **Variant presentation (2026-07-25):**
  the mismatched daemon can also *appear* to start and pass single-client tests,
  then CUDA clients die intermittently-to-universally with **`Error 805: MPS
  client failed to connect`** under real spawn load (580.126.09 control binary
  over a 570.172.08 driver). On a pod, always rule out a version mismatch
  (`strings <bin> | grep -oE '[0-9]{3}\.[0-9.]+'` vs `nvidia-smi` driver) before
  concluding Error 805 means a fatally broken environment.

Neither is the benign `[MPS] ... failed to start daemon` that sweep.py prints
when a daemon is *already running* — that one is fine and `--no_mps` should not be
used in response to it (see CLAUDE.md).

## The hard constraint

**The MPS server binary must match the running driver version exactly.** A
mismatched server (even same major branch, e.g. `580.159.03` vs `580.126.09`)
either fails to daemonize (case b) or starts but then CUDA clients fail with
`RuntimeError: No CUDA GPUs are available`. So you cannot `apt install
nvidia-utils-<branch>` (a ~12 kB metapackage; the binaries live in
`nvidia-compute-utils-*`, which only carries newer patch versions and also pulls
a conflicting userspace driver stack).

## Resolution order (what an agent should do)

1. **Prefer the container-injected system binary.** Check
   `/usr/bin/nvidia-cuda-mps-control` and `/usr/bin/nvidia-cuda-mps-server`
   (often owned `nobody:nogroup`). These ship with the injected driver and match
   it. If present, just make sure they resolve on PATH and use them — no install
   needed.
2. **Suspect a `~/.local/bin` shadow.** `~` (e.g. `/home/<user>`) is on the
   **network volume**, so a binary installed there on an *earlier* pod persists
   and, being early on PATH, shadows `/usr/bin`. If the daemon fails silently
   (case b) or clients hit Error 805, this is the cause. **Remove the shadow**:
   read the stray binary's version (`strings ~/.local/bin/nvidia-cuda-mps-control
   | grep -oE '[0-9]{3}\.[0-9.]+'`), move the pair into `mps_cache/<that_version>/`
   — or simply delete it if the cache already holds an identical copy (compare
   `md5sum`) — so the system binary resolves naturally again. **`~/.local/bin`
   must never hold MPS binaries** (policy set 2026-07-25 after a shadowed
   580-on-570 binary cost a full pod program to Error-805 kills).
3. **Canonical cache: `mps_cache/<driver_version>/`.** Version-keyed, lives on the
   `/workspace` network volume, persists across pods. If the system binary is
   absent (case a) and a matching cache entry exists, put it on PATH at launch.
4. **Last resort — extract from the NVIDIA `.run`** into the cache:
   ```bash
   V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
   curl -s -o nv.run https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run
   sh nv.run -x --target nvrun           # self-extract only; does NOT install
   mkdir -p mps_cache/$V
   /bin/cp nvrun/nvidia-cuda-mps-control nvrun/nvidia-cuda-mps-server mps_cache/$V/
   ```
   (The `XFree86/Linux-x86_64/$V/` path serves the same file if `tesla/` 404s.)

**Never install into `~/.local/bin`, and evict any MPS binaries found there**
(move to `mps_cache/<their_version>/`, or delete when the cache already has an
identical copy). Installing there is what creates the case-(b) shadow on the
next different-driver pod. The version-keyed `mps_cache/$V/` is the *only*
sanctioned location for non-system MPS binaries; version selection happens
purely via PATH order at launch — only when `/usr/bin` lacks them, prepend the
cache:

```bash
V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
PATH="$PWD/mps_cache/$V:$PATH" CUDA_VISIBLE_DEVICES=1,2,3,4 \
  .venv/bin/python -u sweep.py --name <name> --config sweeps/<cfg>.py --per_gpu 4 ...
```
`sweep.py`'s `discover_gpus()` honors `CUDA_VISIBLE_DEVICES`, so this also confines
MPS daemons + runs to those physical GPUs.

## Verifying MPS actually works (and a test gotcha)

```bash
PIPE=/tmp/mps_test; LOG=/tmp/mps_log; mkdir -p $PIPE $LOG
CUDA_VISIBLE_DEVICES=1 CUDA_MPS_PIPE_DIRECTORY=$PIPE CUDA_MPS_LOG_DIRECTORY=$LOG \
  nvidia-cuda-mps-control -d                 # rc=0, control.log written, daemon up
# CLIENT must use CVD=0, NOT the physical id: a daemon pinned to one physical GPU
# remaps it to index 0 in the server's visible set.
CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=$PIPE \
  .venv/bin/python -c "import torch; x=torch.randn(2000,2000,device='cuda'); torch.cuda.synchronize(); print('OK', float((x@x).sum()))"
CUDA_MPS_PIPE_DIRECTORY=$PIPE bash -c 'echo quit | nvidia-cuda-mps-control'
```
A successful **client matmul** is the real proof (a mismatched server passes
`get_server_list` but fails the client with `No CUDA GPUs are available`). In a
live sweep, confirm from the orchestrator log (`[MPS] Started daemons on GPU(s):
[...]`), `pgrep nvidia-cuda-mps-server` (clients attached), and `nvidia-smi`
(high util / multi-GB per GPU) — not the block-buffered per-run logs.

If the client uses the **physical** id instead of 0 it fails with `No CUDA GPUs
are available` while `$LOG/server.log` says *"Invalid CUDA_VISIBLE_DEVICES ... not
a valid GPU ID in the MPS visible set"* — a manual-test mistake only; sweep.py's
`_run_worker` already does the correct remap.

## The binary cache

`mps_cache/<driver_version>/` holds the two binaries per driver version on the
persistent `/workspace` volume. A future pod with the **same** driver reuses it;
a **different** driver needs a fresh entry (exact-version match required). Prefer
`/usr/bin` when the container provides it; the cache is the fallback for pods that
don't, and the place to look first before any download.

See also: [[reference_runpod_mps_install]] (memory), VLLM_LIFECYCLE.md, CLAUDE.md
"Gotchas → Compute".
