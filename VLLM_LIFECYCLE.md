# vLLM Lifecycle (concurrent init queueing)

When several training runs share a GPU (sweep.py local backend with `per_gpu > 1`, or a Modal `train_many` pack), each one spawns its own vLLM server. Simultaneous CUDA inits race on free-memory probing + KV-cache allocation; empirically (2026-06-02 verify_modal_repeat_rb_sweep) one or two survive and the rest die with `vLLM server process died during startup`.

`vllm_lifecycle.py` provides the four shared primitives all three vLLM-spawning sites need:
- **`vllm_init_slot(gpu_id, label)`** — context manager that holds an exclusive `fcntl.flock` on `/tmp/vllm_init_lock_gpu{N}` during the heavy CUDA-init phase. First holder out wins; others queue. Released before vLLM enters its serve loop so the next caller can start its own init. Latency-optimal: zero delay when nothing else is initing.
- **`wait_for_ready_file(ready_file, proc, label)`** — block until vLLM's ready-sentinel file appears; fast-fail on proc death or timeout (default 900s).
- **`vllm_worker_setup_signals()`** — `os.setsid()` at the top of the vLLM worker so the worker becomes a process group leader.
- **`killpg_cleanup(proc)`** — `os.killpg(pgid, SIGKILL)` on shutdown to reach EngineCore + ProcManager grandchildren that bare `proc.kill()` would orphan on the GPU.

All three vLLM-spawning sites use these:
- `sweep.py:_vllm_server_worker` (sweep.py pre-spawns vLLM for the local backend)
- `train.py:_spawn_vllm_server` (train.py's own `--vllm_spawn` path, used by the Modal backend)
- Bare ad-hoc launches via `tools/modal_train_gr.py` train_one/train_many (which run through train.py)

Scheduling differences between local and Modal backends (per-GPU caps, slot pool, dispatch-and-poll) live in their respective backends — `vllm_lifecycle` only handles the per-vLLM mechanics. `--vllm_spawn_delay` is now deprecated and silently ignored; the lock supersedes static stagger.
