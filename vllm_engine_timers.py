"""Cumulative wall-time instrumentation of the in-process vLLM v1 engine.

install(llm) monkeypatches the live engine instance to accumulate per-phase
wall time into TIMERS. The server resets TIMERS before each generate and
reports a snapshot in the generate reply's timings dict, so phase shares are
visible per generate call without editing installed files.

Phases (all CPU unless noted):
  schedule      Scheduler.schedule — pick the next wave, build SchedulerOutput
  exec_wait     blocking wait on the model-execution future — GPU forward +
                sampler (the only phase that is mostly GPU)
  update        Scheduler.update_from_output — per-seq bookkeeping, stop checks
  proc_out      OutputProcessor.process_outputs — incremental detok +
                RequestOutput construction
  step_total    LLMEngine.step end-to-end (includes the above + glue)

Enable via env VLLM_ENGINE_TIMERS=1 (read by vllm_server.py).
"""
import time
from collections import defaultdict

TIMERS = defaultdict(float)
COUNTS = defaultdict(int)


def reset():
    TIMERS.clear()
    COUNTS.clear()


def snapshot():
    out = {k: round(v, 4) for k, v in TIMERS.items()}
    out.update({f"n_{k}": v for k, v in COUNTS.items()})
    return out


def _wrap(obj, name, key):
    fn = getattr(obj, name)
    if getattr(fn, "_timer_wrapped", False):
        return

    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            TIMERS[key] += time.perf_counter() - t0
            COUNTS[key] += 1

    wrapped._timer_wrapped = True
    setattr(obj, name, wrapped)


def install(llm):
    """Patch the live engine instance behind an LLM object."""
    eng = llm.llm_engine                      # v1 LLMEngine
    core = eng.engine_core.engine_core        # InprocClient -> EngineCore
    _wrap(eng, "step", "step_total")
    _wrap(core.scheduler, "schedule", "schedule")
    _wrap(core.scheduler, "update_from_output", "update")
    _wrap(eng.output_processor, "process_outputs", "proc_out")

    # execute_model is non-blocking (returns a future); time the blocking
    # .result() wait instead — that wait is the GPU forward + sampler.
    ex = core.model_executor
    orig_exec = ex.execute_model
    if not getattr(orig_exec, "_timer_wrapped", False):
        class _TimedFuture:
            __slots__ = ("_f",)
            def __init__(self, f):
                self._f = f
            def result(self, *a, **kw):
                t0 = time.perf_counter()
                try:
                    return self._f.result(*a, **kw)
                finally:
                    TIMERS["exec_wait"] += time.perf_counter() - t0
                    COUNTS["exec_wait"] += 1
            def __getattr__(self, name):
                return getattr(self._f, name)

        def timed_exec(*a, **kw):
            r = orig_exec(*a, **kw)
            return _TimedFuture(r) if hasattr(r, "result") else r

        timed_exec._timer_wrapped = True
        ex.execute_model = timed_exec

    return True
