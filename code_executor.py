"""Sandboxed code execution for leetcode reward evaluation.

Runs model output + GT assert tests in subprocesses with timeout.
Subprocesses are sandboxed: minimal env vars, resource limits (memory,
CPU, file size, process count).
"""

import os
import re
import resource
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Subprocess limits
_TIMEOUT = 10.0  # wall-clock seconds
_RLIMIT_AS = 512 * 1024 * 1024  # 512MB virtual memory
_RLIMIT_CPU = 10  # 10s CPU time
_RLIMIT_FSIZE = 1 * 1024 * 1024  # 1MB max file write
_RLIMIT_NPROC = 64  # max child processes (fork bomb protection)

# Minimal env: only PATH and HOME, strips secrets like OPENAI_API_KEY
_SUBPROCESS_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", "/tmp"),
}


def _set_resource_limits():
    """preexec_fn for subprocesses: set resource limits."""
    resource.setrlimit(resource.RLIMIT_AS, (_RLIMIT_AS, _RLIMIT_AS))
    resource.setrlimit(resource.RLIMIT_CPU, (_RLIMIT_CPU, _RLIMIT_CPU))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_RLIMIT_FSIZE, _RLIMIT_FSIZE))
    resource.setrlimit(resource.RLIMIT_NPROC, (_RLIMIT_NPROC, _RLIMIT_NPROC))


def _run_subprocess(script_path, timeout=_TIMEOUT):
    """Run a Python script in a sandboxed subprocess.

    Returns {"passed": bool, "error": str|None}.
    """
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_SUBPROCESS_ENV,
            preexec_fn=_set_resource_limits,
        )
        if result.returncode == 0:
            return {"passed": True, "error": None}
        else:
            error = result.stderr.strip()[-500:] if result.stderr else "Unknown error"
            return {"passed": False, "error": error}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout after {timeout}s"}


def parse_code_block(text):
    """Extract ```python ... ``` block from model output. Falls back to full text."""
    match = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic code block
    match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _build_script(code, tests, setup_code=""):
    """Build composite Python script: setup + code + tests."""
    parts = []
    if setup_code:
        parts.append(setup_code)
    parts.append(code)
    parts.append("\n".join(tests))
    return "\n\n".join(parts)


def execute_code_with_tests(code, tests, setup_code="", timeout=_TIMEOUT):
    """Run code + assert tests in sandboxed subprocess.

    Returns {"passed": bool, "error": str|None}.
    """
    script = _build_script(code, tests, setup_code)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            f.flush()
            tmp_path = f.name

        return _run_subprocess(tmp_path, timeout)
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


def check_compiles(code):
    """Check if code compiles (no syntax errors). Returns bool."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def execute_code_only(code, setup_code="", timeout=_TIMEOUT):
    """Run code without tests in sandboxed subprocess.

    Returns {"passed": bool, "error": str|None}.
    """
    script = setup_code + "\n\n" + code if setup_code else code

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            f.flush()
            tmp_path = f.name

        return _run_subprocess(tmp_path, timeout)
    except Exception as e:
        return {"passed": False, "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


_MAX_WORKERS = max(1, int(os.cpu_count() * 0.6))


def execute_batch(items, max_workers=None):
    """Parallel execution via ThreadPoolExecutor.

    items: list of (code, tests, setup_code) tuples.
    Returns: list of {"passed": bool, "error": str|None}.
    """
    if max_workers is None:
        max_workers = _MAX_WORKERS

    def _run(item):
        code, tests, setup_code = item
        return execute_code_with_tests(code, tests, setup_code)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, items))


def execute_code_batch(items, max_workers=None):
    """Parallel execution of code-only (no tests).

    items: list of (code, setup_code) tuples.
    Returns: list of {"passed": bool, "error": str|None}.
    """
    if max_workers is None:
        max_workers = _MAX_WORKERS

    def _run(item):
        code, setup_code = item
        return execute_code_only(code, setup_code)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, items))
