"""Persistent worker pool for code evaluation.

Replaces CodeEvaluator's per-call subprocess spawning with pre-forked
workers that stay alive across batch_evaluate calls. Each worker is a
Python subprocess that reads code from stdin and writes results to stdout,
reusing the Python interpreter across evaluations.

Workers are killed and respawned if they crash or timeout, but this is
the exception, not the rule — most evaluations complete cleanly and the
worker is reused.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock

from tqdm import tqdm


# The worker script: loops reading {code, timeout, memory_limit} from stdin,
# executes each, writes result JSON to stdout. Stays alive between evals.
_WORKER_CODE = textwrap.dedent(r"""
import io
import json
import resource
import signal
import sys
from contextlib import redirect_stdout

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("timeout")

signal.signal(signal.SIGALRM, _timeout_handler)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        sys.stdout.write(json.dumps({"error": "bad json"}) + "\n")
        sys.stdout.flush()
        continue

    code = request["code"]
    timeout = request.get("timeout", 3)
    memory_mb = request.get("memory_limit", 1024)

    # Set resource limits (best effort)
    memory_bytes = max(memory_mb, 1) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (max(int(timeout), 1), max(int(timeout), 1)))
    except (ValueError, OSError):
        pass

    signal.alarm(max(int(timeout), 1))

    output = {
        "success": True,
        "compiled": True,
        "timeout": False,
        "oom": False,
        "stdout": {}
    }

    stdout_buffer = io.StringIO()
    namespace = {}

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, namespace)
        stdout_content = stdout_buffer.getvalue()
        try:
            lines = stdout_content.strip().split('\n')
            parsed_json = None
            for l in reversed(lines):
                l = l.strip()
                if l.startswith('{') and l.endswith('}'):
                    try:
                        parsed_json = json.loads(l)
                        break
                    except json.JSONDecodeError:
                        continue
            output["stdout"] = parsed_json if parsed_json is not None else {}
        except Exception:
            output["stdout"] = {"raw": stdout_content}
    except (SyntaxError, IndentationError):
        output["success"] = False
        output["compiled"] = False
    except MemoryError:
        output["success"] = False
        output["oom"] = True
        output["stdout"] = {}
    except TimeoutException:
        output["success"] = False
        output["timeout"] = True
        output["stdout"] = {}
    except Exception as e:
        output["success"] = False
        output["stdout"] = {"raw": str(e)}
    finally:
        signal.alarm(0)
        # Reset resource limits for next eval
        try:
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except (ValueError, OSError):
            pass
        # Clear namespace to avoid state leaking between evals
        namespace.clear()

    sys.stdout.write(json.dumps(output) + "\n")
    sys.stdout.flush()
""").strip()


def _get_python_executable():
    resolved = os.path.realpath(sys.executable)
    if os.path.exists(resolved) and os.access(resolved, os.X_OK):
        return resolved
    return sys.executable


class _Worker:
    """A single persistent Python subprocess."""

    def __init__(self):
        self.proc = None
        self._start()

    def _start(self):
        python = _get_python_executable()
        self.proc = subprocess.Popen(
            [python, "-c", _WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            close_fds=True,
        )

    def execute(self, code: str, timeout: int = 3, memory_limit: int = 1024) -> dict:
        """Send code to the worker and get result. Respawns on failure."""
        request = json.dumps({"code": code, "timeout": timeout, "memory_limit": memory_limit})
        try:
            self.proc.stdin.write(request + "\n")
            self.proc.stdin.flush()
            # Wait for response with a timeout slightly longer than the code timeout
            import selectors
            sel = selectors.DefaultSelector()
            try:
                sel.register(self.proc.stdout, selectors.EVENT_READ)
                ready = sel.select(timeout=timeout + 2)
            finally:
                sel.close()
            if not ready:
                self._kill_and_respawn()
                return {"success": False, "compiled": True, "timeout": True, "oom": False, "stdout": {}}
            line = self.proc.stdout.readline()
            if not line:
                self._kill_and_respawn()
                return {"success": False, "compiled": False, "timeout": False, "oom": False, "stdout": {"raw": "worker died"}}
            return json.loads(line.strip())
        except (BrokenPipeError, OSError, json.JSONDecodeError) as e:
            self._kill_and_respawn()
            return {"success": False, "compiled": False, "timeout": False, "oom": False, "stdout": {"raw": str(e)}}

    def _kill_and_respawn(self):
        self._cleanup()
        self._start()

    def _cleanup(self):
        """Close all fds and kill the process."""
        if self.proc is None:
            return
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except Exception:
                pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=1)
        except Exception:
            pass
        self.proc = None

    def close(self):
        self._cleanup()


class PersistentCodeEvaluator:
    """Drop-in replacement for CodeEvaluator using persistent worker processes.

    Workers are pre-forked at init and reused across batch_evaluate calls.
    Compatible with the same interface as CodeEvaluator.
    """

    def __init__(self, num_workers: int | None = None, timeout: int = 3,
                 memory_per_worker: int = 1024, max_failures: int = 1, debug: bool = False):
        self.num_workers = num_workers or int(os.environ.get("MAX_JOBS", 20))
        self.timeout = timeout
        self.memory_per_worker = memory_per_worker
        self.max_failures = max_failures
        self.debug = debug

        # Pre-fork workers and put them in a pool
        self._pool = Queue()
        self._workers = []
        for _ in range(self.num_workers):
            w = _Worker()
            self._workers.append(w)
            self._pool.put(w)
        print(f"[PersistentCodeEvaluator] Started {self.num_workers} persistent workers")

    def __call__(self, response, test_list=None, setup_code="", skip_parse=True):
        """Single evaluation — matches CodeEvaluator.__call__ interface."""
        if test_list is None:
            test_list = []

        if not skip_parse:
            program = self.parse_response(response)
        else:
            program = response

        if program is None:
            return {
                "parsed_response": None,
                "is_formatted": False,
                "can_compile": False,
                "pass_rate": 0.0,
                "tests_passed": 0,
                "tests_evaluated": 0,
                "tests_total": len(test_list),
                "test_errors": [],
            }

        from src.evaluate.code.helpers import create_test_runner_code
        runner_code = create_test_runner_code(setup_code, program, test_list, self.max_failures)

        worker = self._pool.get()
        try:
            raw = worker.execute(runner_code, timeout=self.timeout, memory_limit=self.memory_per_worker)
        finally:
            self._pool.put(worker)

        result = {
            "parsed_response": program,
            "is_formatted": True,
            "can_compile": raw.get("compiled", True),
            "pass_rate": 0.0,
            "tests_passed": 0,
            "tests_evaluated": 0,
            "tests_total": len(test_list),
            "test_errors": [],
        }

        if not raw.get("compiled", True):
            result["test_errors"].append("MasterError: CompilationError")
        if raw.get("timeout"):
            result["test_errors"].append("MasterError: TimeoutError")
        if raw.get("oom"):
            result["test_errors"].append("MasterError: OOMError")
        if not raw.get("success", True):
            result["test_errors"].append("MasterError: UnknownError: " + str(raw.get("stdout", {}).get("raw", "")))

        stdout = raw.get("stdout", {})
        if isinstance(stdout, dict):
            result["tests_evaluated"] = stdout.get("tests_evaluated", 0)
            result["tests_passed"] = stdout.get("tests_passed", 0)
            result["pass_rate"] = (result["tests_passed"] / result["tests_total"]) if result["tests_total"] > 0 else 0.0
            result["test_errors"] += stdout.get("test_errors", [])

        return result

    def batch_evaluate(self, calls: list[dict]) -> list:
        """Evaluate a batch — matches CodeEvaluator.batch_evaluate interface."""
        if not calls:
            return []
        results = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {
                executor.submit(self.__call__, **call_kwargs): idx
                for idx, call_kwargs in enumerate(calls)
            }
            pbar = tqdm(total=len(future_to_idx), desc="Evaluating responses")
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                pbar.update(1)
            pbar.close()
        return results

    def parse_response(self, response):
        import re
        blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
        if not blocks:
            return None
        cleaned = [b.strip() for b in blocks if b.strip()]
        return "\n\n".join(cleaned) if cleaned else None

    def close(self):
        for w in self._workers:
            w.close()

    def __del__(self):
        self.close()
