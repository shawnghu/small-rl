"""MBPP+ fairness audit helpers.

Per-test failure extraction: given a candidate program + MBPP+ `test` harness,
return the concrete hidden (input, expected, got) cases it fails. Sandboxed in
the persistent-eval worker (timeout/memory/scratch-cwd). Used to assemble
evidence-backed unfairness candidates for LLM judging.
"""
import json
import re

ENTRY_RE = re.compile(r"assertion\(\s*(\w+)\s*\(")


def entry_name(test_field, fallback_code=""):
    m = ENTRY_RE.search(test_field)
    if m:
        return m.group(1)
    m = re.search(r"def\s+(\w+)\s*\(", fallback_code)
    return m.group(1) if m else None


def _strip_loop(test_field):
    """Return harness head (defs + inputs + results), dropping the final test loop."""
    idx = test_field.rfind("\nfor ")
    return test_field[:idx] if idx != -1 else test_field


def failing_cases(ev, program, test_field, fallback_code="", timeout=10, max_report=8):
    """Run candidate against each hidden case; return dict with per-case failures.

    Returns: {ok, total, n_fail, fails:[[inp,exp,got,errtype],...]} or {ok:False,err}.
    ok=False means the harness head itself didn't run (not a candidate failure).
    """
    name = entry_name(test_field, fallback_code)
    if not program or name is None:
        return {"ok": True, "total": 0, "n_fail": 0, "fails": [], "no_program": True}
    head = _strip_loop(test_field)
    runner = (
        "import json\n"
        "_head = " + repr(head) + "\n"
        "_cand = " + repr(program) + "\n"
        "_name = " + repr(name) + "\n"
        "ns = {}\n"
        "try:\n"
        "    exec(_head, ns)\n"
        "    exec(_cand, ns)\n"
        "    func = ns[_name]\n"
        "    inputs = ns['inputs']; results = ns['results']; assertion = ns['assertion']\n"
        "    fails = []\n"
        "    for inp, exp in zip(inputs, results):\n"
        "        got = None\n"
        "        try:\n"
        "            got = func(*inp)\n"
        "            assertion(got, exp, 0)\n"
        "        except Exception as e:\n"
        "            try: gr = repr(got)[:90]\n"
        "            except Exception: gr = '<unrepr>'\n"
        "            fails.append([repr(inp)[:130], repr(exp)[:90], gr, type(e).__name__])\n"
        "        if len(fails) >= " + str(max_report) + ": break\n"
        "    print(json.dumps({'ok': True, 'total': len(inputs), 'n_fail': len(fails), 'fails': fails}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'err': type(e).__name__ + ': ' + str(e)[:140]}))\n"
    )
    w = ev._pool.get()
    try:
        raw = w.execute(runner, timeout=timeout, memory_limit=4096)
    finally:
        ev._pool.put(w)
    so = raw.get("stdout", {})
    if isinstance(so, dict) and "ok" in so:
        return so
    return {"ok": False, "err": "worker:" + json.dumps(raw)[:140]}


# ---- builtin-shadow trap detection ----
import builtins
_BUILTINS = set(dir(builtins))


def shadows_builtin(test_field, fallback_code=""):
    name = entry_name(test_field, fallback_code)
    return name if name in _BUILTINS else None
