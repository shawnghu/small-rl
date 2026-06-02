"""Shared primitives for spawning and monitoring vLLM server subprocesses.

Three callers spawn vLLM server processes:
  - sweep.py (local backend) — pre-spawns one vLLM per training run, passes
    socket path to the train.py child via `--vllm_server`.
  - train.py with `--vllm_spawn` (used by the Modal backend) — train.py spawns
    its own vLLM in-process.
  - bare ad-hoc launches via tools/modal_train_gr.py train_one/train_many.

All three need the same four mechanics, which used to be duplicated:
  - **Concurrent-init queueing**: only one vLLM init at a time per GPU
    (concurrent CUDA inits race on free-memory probing and KV-cache
    allocation, causing some to die during startup; see verify_modal_repeat_rb_sweep
    crashes 2026-06-02 for empirical evidence). Use `vllm_init_slot` as a
    context manager around the spawn + ready-wait; it acquires an exclusive
    `fcntl.flock` on /tmp/vllm_init_lock_gpu{N} for the duration. Latency-
    optimal: zero delay when the GPU has no other in-flight init, naturally
    serializes when several try at once.
  - **Ready-file wait**: poll the vLLM-emitted sentinel file with fast-fail
    on proc death and timeout. `wait_for_ready_file`.
  - **Process-group setup**: vLLM v1 spawns EngineCore + ProcManager as its
    own children. Without `os.setsid()` in the vLLM worker, the parent's
    SIGKILL only reaches the direct Python child; grandchildren orphan and
    hold GPU memory until the host/container tears down. `vllm_worker_setup_signals`.
  - **Process-group SIGKILL**: pair with the setsid above to reach the whole
    tree on shutdown. `killpg_cleanup`.

Stateless functions + module constant. No classes, no lifecycle hooks. The
*scheduling* decisions (when to spawn, how many concurrent runs, on which
GPU/container) stay in the respective backends — only the per-vLLM mechanics
are shared.
"""
from __future__ import annotations

import fcntl
import os
import signal
import time
from contextlib import contextmanager
from typing import Iterable, Optional


# Default safety ceiling for the init queue and ready-file wait. vLLM cold
# init at our model scales is ~20-60s; queueing 4-5 waiters can push to a few
# minutes. 900s = 15 min gives generous headroom and makes a true hang loud.
_DEFAULT_INIT_TIMEOUT_S = 900

# Lock files live in /tmp/ — local to the host (or container) and cleared on
# reboot. /tmp/vllm_init_lock_gpu{N} is the convention; each is intended to
# protect one physical GPU's init slot.
_LOCK_PATH_FMT = "/tmp/vllm_init_lock_gpu{gpu_id}"


@contextmanager
def vllm_init_slot(gpu_id: int, label: str,
                   timeout_s: int = _DEFAULT_INIT_TIMEOUT_S,
                   warn_at: Iterable[int] = (60, 600),
                   print_fn=print):
    """Serialize vLLM init on a single GPU.

    Wrap the vLLM spawn + ready-wait in this context manager. While inside,
    you hold an exclusive `fcntl.flock` on /tmp/vllm_init_lock_gpu{gpu_id};
    other holders queue behind you. Release happens on exit (either path),
    including process death — the kernel auto-releases when the fd's owning
    process exits, so a crashing init doesn't strand the lock.

    Args
    ----
    gpu_id     : GPU index this init targets. In a Modal container that has
                 a single H100, this is always 0. On a multi-GPU local box,
                 use the assigned GPU.
    label      : human-readable name shown in the warn-at messages (use the
                 run name or similar). Helps when several runs are queued.
    timeout_s  : assertion budget for total wait (default 15 min). Caps the
                 worst-case hang behind a stuck holder.
    warn_at    : list of seconds-elapsed thresholds at which to print a
                 [vllm_init_slot] WARN line; default (60, 600).
    print_fn   : where the warnings go; default `print`.

    Usage
    -----
        with vllm_init_slot(gpu_id=gpu, label=run_name):
            proc = ctx.Process(target=_vllm_server_worker, args=(...))
            proc.start()
            wait_for_ready_file(ready_file, proc, run_name)
        # lock released; vLLM is up + listening
    """
    lock_path = _LOCK_PATH_FMT.format(gpu_id=gpu_id)
    fd = open(lock_path, "w")
    t0 = time.monotonic()
    warned: set[int] = set()
    try:
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                elapsed = time.monotonic() - t0
                for warn_s in warn_at:
                    if warn_s not in warned and elapsed >= warn_s:
                        print_fn(
                            f"[vllm_init_slot] {label} still waiting after "
                            f"{warn_s}s (another init in progress on GPU {gpu_id})"
                        )
                        warned.add(warn_s)
                assert elapsed < timeout_s, (
                    f"vllm_init_slot({gpu_id}) timed out after {timeout_s}s for "
                    f"{label}; another holder is wedged. Check who's holding "
                    f"{lock_path} via `fuser {lock_path}`."
                )
                time.sleep(0.5)
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()


def wait_for_ready_file(ready_file: str, proc, label: str,
                        timeout: int = _DEFAULT_INIT_TIMEOUT_S,
                        warn_at: Iterable[int] = (60, 600),
                        print_fn=print) -> None:
    """Block until ready_file exists. Fast-fail if proc dies or timeout hits.

    The ready_file is a sentinel that the vLLM worker `touch`-es when its
    engine is listening (sweep.py:_vllm_server_worker via _FileEvent.set();
    train.py's _spawn_vllm_server similarly). Reaching here means the vLLM
    server is ready to serve generate requests on its socket.

    On success, atomically removes the ready_file (best-effort).
    On proc death or timeout, raises AssertionError with a label-bearing
    message — these are intended to be loud and to terminate the caller.
    """
    t0 = time.monotonic()
    warned: set[int] = set()
    while not os.path.exists(ready_file):
        elapsed = time.monotonic() - t0
        for warn_s in warn_at:
            if warn_s not in warned and elapsed >= warn_s:
                print_fn(
                    f"[wait_for_ready_file] {label} not ready after {warn_s}s "
                    f"(check the vllm_server.log for the run dir)"
                )
                warned.add(warn_s)
        assert elapsed < timeout, (
            f"{label} failed to start within {timeout}s"
        )
        assert proc.is_alive(), f"{label} died during startup"
        time.sleep(0.5)
    try:
        os.unlink(ready_file)
    except FileNotFoundError:
        pass


def vllm_worker_setup_signals() -> None:
    """Call at the top of any vLLM-server worker (process target).

    Becomes a process group leader so the parent's `killpg(pgid, SIGKILL)`
    will reach all subprocesses vLLM v1 spawns under itself (EngineCore,
    ProcManager, etc.). Without this, plain `proc.kill()` on the worker only
    kills the direct Python child and leaves the grandchildren orphaned on
    the GPU.

    Idempotent in practice — `os.setsid()` raises PermissionError if already
    a process group leader, which we swallow.
    """
    try:
        os.setsid()
    except PermissionError:
        # Already a process group leader (e.g. running under a job-control
        # shell that already setsid()-ed us). The killpg cleanup below still
        # works against whatever the current pgid is.
        pass


def killpg_cleanup(proc, sig: int = signal.SIGKILL,
                   join_timeout: float = 2.0) -> None:
    """SIGKILL the process group rooted at proc.pid.

    Pairs with `vllm_worker_setup_signals` at the other end. Use after
    proc.join(timeout) on the shutdown path to make sure EngineCore +
    ProcManager grandchildren actually die. Best-effort: missing process /
    permission errors fall back to a direct proc.kill().
    """
    if proc is None:
        return
    if proc.is_alive():
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except Exception:
                pass
    try:
        proc.join(timeout=join_timeout)
    except Exception:
        pass
