"""Cross-sweep GPU slot pool.

Sweeps coordinate via the filesystem so that, regardless of how many
sweep.py processes are running concurrently, no GPU exceeds a global
per-GPU run cap. Each sweep still enforces its own per_gpu locally; the
global cap is an additional ceiling shared across sweeps.

Usage from sweep.py:

    import slot_pool
    slot_id = slot_pool.try_acquire(gpu_id, holder_pid)
    if slot_id is not None:
        # spawn the run
        ...
        # later, when the run finishes:
        slot_pool.release(gpu_id, holder_pid, slot_id)

The cap defaults to 6 per GPU and can be overridden by setting the
SMALL_RL_GPU_CAP env var (or passing cap=... to try_acquire).

Stale slot files (held by a dead process) are automatically cleaned on
each acquire, so a sweep that crashes without releasing its slots only
leaks them transiently. The cleanup is best-effort and doesn't require
holding a global lock.
"""
import os
import time
import uuid

_POOL_ROOT = "/tmp/small-rl-slots"


def _default_cap() -> int:
    try:
        return int(os.environ.get("SMALL_RL_GPU_CAP", "6"))
    except ValueError:
        return 6


def _gpu_dir(gpu_id: int) -> str:
    d = os.path.join(_POOL_ROOT, f"gpu{gpu_id}")
    os.makedirs(d, exist_ok=True)
    return d


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another uid; treat as alive (we
        # won't clean it up).
        return True


def _cleanup_stale(gpu_dir: str) -> None:
    """Remove slot files whose holder PID is dead."""
    try:
        entries = os.listdir(gpu_dir)
    except FileNotFoundError:
        return
    for fname in entries:
        if not fname.endswith(".lock"):
            continue
        # Format: "{holder_pid}-{slot_id}.lock"
        try:
            holder_pid_str = fname.split("-", 1)[0]
            holder_pid = int(holder_pid_str)
        except (IndexError, ValueError):
            continue
        if not _alive(holder_pid):
            try:
                os.remove(os.path.join(gpu_dir, fname))
            except FileNotFoundError:
                pass


def try_acquire(gpu_id: int, holder_pid: int, cap: int | None = None) -> str | None:
    """Try to acquire a slot on `gpu_id` for `holder_pid`.

    Returns a slot_id (str) on success, or None if the GPU is at capacity.
    Pass the returned slot_id to release() when the run finishes.
    """
    if cap is None:
        cap = _default_cap()
    d = _gpu_dir(gpu_id)
    _cleanup_stale(d)
    try:
        held = [f for f in os.listdir(d) if f.endswith(".lock")]
    except FileNotFoundError:
        held = []
    if len(held) >= cap:
        return None
    slot_id = uuid.uuid4().hex[:12]
    lock_path = os.path.join(d, f"{holder_pid}-{slot_id}.lock")
    try:
        # Use O_EXCL to detect collisions on the rare matching uuid; just retry.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    except FileExistsError:
        return try_acquire(gpu_id, holder_pid, cap=cap)
    return slot_id


def release(gpu_id: int, holder_pid: int, slot_id: str) -> None:
    """Release a slot previously acquired via try_acquire."""
    d = _gpu_dir(gpu_id)
    lock_path = os.path.join(d, f"{holder_pid}-{slot_id}.lock")
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def status() -> dict:
    """Return per-GPU slot occupancy for diagnostics.

    Returns {gpu_id: int} mapping. Cleans stale slots first.
    """
    out = {}
    if not os.path.isdir(_POOL_ROOT):
        return out
    for entry in os.listdir(_POOL_ROOT):
        if entry.startswith("gpu"):
            try:
                gpu_id = int(entry[3:])
            except ValueError:
                continue
            d = os.path.join(_POOL_ROOT, entry)
            _cleanup_stale(d)
            try:
                held = [f for f in os.listdir(d) if f.endswith(".lock")]
            except FileNotFoundError:
                held = []
            out[gpu_id] = len(held)
    return out
