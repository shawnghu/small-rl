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


# vLLM init at our scales should take ~10-30s end-to-end without MPS, and
# ~30-50s under CUDA MPS (the shared context costs more startup work).
# 150s is well above either regime — a single holder taking longer than that
# is the "something's wrong" threshold above which a waiter raises rather
# than continuing to queue indefinitely. Previously 60s, but on Modal +
# MPS we measured one init at 43.8s; the next slow-tail init can easily
# exceed 60s and cascade-abort all 4 other waiters, which is what bit us
# on 2026-06-03 (3 of 5 children in a train_many pack failed at startup).
_DEFAULT_HOLD_TIMEOUT_S = 150

# Ready-file wait budget for callers of `wait_for_ready_file`. This needs to
# cover a legitimate queue depth (N waiters × hold_timeout) plus the holder's
# own init. 600s = 10 min handles ~10 deep queues at full hold-timeout each;
# in normal operation we expect well under this. Per-holder timeout kicks in
# first if a single init wedges.
_DEFAULT_READY_TIMEOUT_S = 600

# Lock files live in /tmp/ — local to the host (or container) and cleared on
# reboot. /tmp/vllm_init_lock_gpu{N} is the convention; each is intended to
# protect one physical GPU's init slot.
_LOCK_PATH_FMT = "/tmp/vllm_init_lock_gpu{gpu_id}"


def _read_holder_stamp(fd: int):
    """Read the current holder's identity from the lock file.

    Returns (pid, acquire_wall_time, label) or (None, None, None) if the file
    is empty / unparseable. The lock holder is responsible for stamping the
    file immediately after acquiring; brief windows of stale content can
    occur between the previous holder releasing and the new holder writing.
    """
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        content = os.read(fd, 4096).decode(errors="ignore").strip()
    except OSError:
        return None, None, None
    if not content:
        return None, None, None
    parts = content.split(None, 2)
    try:
        pid = int(parts[0])
        acquired_at = float(parts[1])
        label = parts[2] if len(parts) > 2 else ""
        return pid, acquired_at, label
    except (ValueError, IndexError):
        return None, None, None


def _write_holder_stamp(fd: int, label: str) -> None:
    """Stamp our pid + wall-clock acquire time + label into the lock file.
    Called immediately after flock acquires. Other waiters read this to
    decide whether the holder has been holding too long."""
    stamp = f"{os.getpid()} {time.time()} {label}\n".encode()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, stamp)
        try:
            os.fsync(fd)
        except OSError:
            pass
    except OSError:
        pass  # stamping is best-effort; lock still held


@contextmanager
def vllm_init_slot(gpu_id: int, label: str,
                   hold_timeout_s: int = _DEFAULT_HOLD_TIMEOUT_S,
                   wait_timeout_s: Optional[int] = None,
                   warn_at: Iterable[int] = (20, 50),
                   print_fn=print):
    """Serialize vLLM init on a single GPU; fail fast if a holder gets stuck.

    Wrap the vLLM spawn + ready-wait in this context manager. While inside,
    you hold an exclusive `fcntl.flock` on /tmp/vllm_init_lock_gpu{gpu_id};
    other holders queue behind you. On acquire, the holder stamps
    `{pid} {wall_time} {label}\\n` into the lock file. Waiters read that
    stamp on contention and:

      - print the holder's identity + age in their WARN lines (so contention
        is debuggable from logs alone)
      - **raise AssertionError if the current holder has been holding for
        > hold_timeout_s seconds** — the holder is presumed stuck. This is
        the primary safety net; it scopes the "something's wrong" condition
        to a single legitimate-init's duration, not to a queue-depth-dependent
        wall-clock budget.

    Release on exit (either path), including process death — the kernel
    auto-releases when the fd's owning process exits, so a crashing init
    doesn't strand the lock for the next waiter.

    Args
    ----
    gpu_id          : physical GPU index for the slot. In a Modal container
                      with a single H100 this is 0; on a multi-GPU local
                      box use the assigned GPU. Must match what callers
                      treat as "same GPU's vLLM."
    label           : human-readable name (run name typically). Stamped into
                      the lock file and surfaced in WARN/error lines so
                      operators can identify the holder.
    hold_timeout_s  : max time a single holder may legitimately hold the
                      lock (default 60). Waiters raise AssertionError if
                      the holder's age exceeds this.
    wait_timeout_s  : optional cap on total wait time for the LOCAL waiter.
                      Default None (unbounded — per-holder timeout already
                      bounds chain length at N × hold_timeout_s).
    warn_at         : seconds-elapsed-as-a-waiter thresholds at which to
                      print a [vllm_init_slot] WARN line; default (20, 50).
    print_fn        : where warnings go; default `print`.

    Usage
    -----
        with vllm_init_slot(gpu_id=gpu, label=run_name):
            server = VLLMServer(...)  # CUDA-heavy init
        # lock released; safe for another vLLM to start its own init now
    """
    lock_path = _LOCK_PATH_FMT.format(gpu_id=gpu_id)
    # Open RDWR so we can read/write the holder stamp without re-opening.
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.monotonic()
    warned: set[int] = set()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Acquired. Stamp identity immediately so any waiter that
                # contends with us next can read it.
                _write_holder_stamp(fd, label)
                break
            except BlockingIOError:
                elapsed = time.monotonic() - t0
                holder_pid, acquired_at, holder_label = _read_holder_stamp(fd)
                holder_age = (time.time() - acquired_at) if acquired_at else None

                # Primary safety net: stuck holder.
                if holder_age is not None and holder_age > hold_timeout_s:
                    raise AssertionError(
                        f"vllm_init_slot({gpu_id}): current holder "
                        f"pid={holder_pid} label={holder_label!r} has been "
                        f"holding for {holder_age:.1f}s "
                        f"(> hold_timeout_s={hold_timeout_s}s). Likely "
                        f"stuck. Local waiter was {label!r}. Inspect "
                        f"{lock_path}, or `fuser {lock_path}` / "
                        f"`ps -fp {holder_pid}`."
                    )

                for warn_s in warn_at:
                    if warn_s not in warned and elapsed >= warn_s:
                        age_str = f"{holder_age:.1f}s" if holder_age is not None else "?"
                        print_fn(
                            f"[vllm_init_slot] {label} waiting after {warn_s}s "
                            f"on GPU {gpu_id}: holder pid={holder_pid} "
                            f"label={holder_label!r} held for {age_str}"
                        )
                        warned.add(warn_s)

                if wait_timeout_s is not None and elapsed >= wait_timeout_s:
                    raise AssertionError(
                        f"vllm_init_slot({gpu_id}): waiter {label!r} hit "
                        f"wait_timeout_s={wait_timeout_s}s while holder "
                        f"pid={holder_pid} label={holder_label!r} was active. "
                        f"Each holder has been within hold_timeout but the "
                        f"queue is deeper than expected."
                    )

                time.sleep(0.5)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def wait_for_ready_file(ready_file: str, proc, label: str,
                        timeout: int = _DEFAULT_READY_TIMEOUT_S,
                        warn_at: Iterable[int] = (60, 300),
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
