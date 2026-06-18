"""EvalPlus MBPP+ environment — clean code-RL benchmark, no reward hack.

Loads evalplus/mbppplus (378 problems, ~108 hidden tests each), restricted to
the 345 problems that passed the fairness audit (see
output/mbppplus_capability/FAIRNESS_REPORT.md). Deterministic 80/20 train/eval
split (md5 on task_id, stable across runs). No hackability/detectability/penalty
machinery — this is a pure "does standard RL improve the model at the benchmark"
env (routing_mode=none).

Prompt: problem statement + the single base test (test_list[0]) as the visible
example, same template as envs/mbpp.py. Reward = fraction of the full MBPP+
hidden-test suite passed (continuous; gives GRPO within-group variance even when
full-solve rate is 0). The headline metric is full-solve (all plus tests pass),
tracked via the scale-0 `mbpp_plus_correct` driver component.

Scoring: the dataset's `test` field is a self-contained harness (imports numpy,
defines is_floats/assertion, inline inputs/results, then a test loop). We strip
the loop, exec the candidate + harness head in a sandboxed worker, and count how
many (input, expected) cases the candidate satisfies. Runs on the shared
leetcode PersistentCodeEvaluator worker pool.
"""

import ast
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset, load_dataset

from envs import EnvSpec, register_env
from envs.mbpp import _robust_parse, _is_pure_constant

_DATASET_NAME = "evalplus/mbppplus"
_FILTERED_PATH = "data/mbppplus_filtered_dataset.json"

_PROMPT_TEMPLATE = (
    "Write a Python function to solve this problem. "
    "Return only the code, no other text:\n\n{text}\n\n"
    "## Test Case:\n```python\n{first_test}\n```"
)

# Reuse mbpp's robust unfenced-code extractor.
from envs.mbpp import _extract_program

_EXTRA_COLUMNS = ["id", "plus_test", "plus_entry"]

# Scorer budgets. MBPP+ tests run in ms; the tail is generated infinite loops.
# A few problems have large hidden-input lists; 6s/4GB covers all but the
# resource-pathological ones (which are dropped at load, see _MAX_TEST_BYTES).
_TIMEOUT_S = 6
_MEM_MB = 4096
# Drop problems whose harness is huge (combinatorial blowup / >40KB test field):
# even the canonical needs many GB / tens of seconds, so no in-budget completion
# can pass -> a dead, signal-less problem. ~4-5 problems.
_MAX_TEST_BYTES = 40_000

_ENTRY_RE = re.compile(r"assertion\(\s*(\w+)\s*\(")
# A harness whose `assertion` body never executes a real `assert` statement can
# never fail -> ANY output scores 1.0 (3 broken dataset harnesses: 737/787/794).
# Detect by the absence of a real `assert` token (\bassert\b, excluding the
# substring inside "assertion"); drop these — they can't validate anything.
_ASSERT_RE = re.compile(r"\bassert\b(?!ion)")


def _entry_name(test_field, code=""):
    m = _ENTRY_RE.search(test_field)
    if m:
        return m.group(1)
    m = re.search(r"def\s+(\w+)\s*\(", code)
    return m.group(1) if m else ""


def _strip_loop(test_field):
    idx = test_field.rfind("\nfor ")
    return test_field[:idx] if idx != -1 else test_field


def _eval_split(task_id, eval_frac=0.2):
    """True -> held-out eval. Deterministic md5 partition on task_id."""
    h = hashlib.md5(f"evalplus_split:{task_id}".encode()).hexdigest()
    return (int(h, 16) % 10000) < eval_frac * 10000


def _kept_ids():
    """The RL problem set: fairness-kept AND scorable under the per-completion
    budget (excludes huge/never-assert/resource-pathological harnesses; see
    filtered_dataset.json["rl_scorable"], 333 problems). Falls back to "kept"
    if the validated field is absent."""
    with open(_FILTERED_PATH) as f:
        fd = json.load(f)
    return set(fd.get("rl_scorable") or fd["kept"])


def _rows(which):
    """which: 'train' | 'eval'. Returns list of prompt dicts."""
    kept = _kept_ids()
    raw = load_dataset(_DATASET_NAME, split="test")
    rows, n_huge, n_noassert = [], 0, 0
    for r in raw:
        tid = r["task_id"]
        if tid not in kept:
            continue
        if len(r["test"]) > _MAX_TEST_BYTES:
            n_huge += 1
            continue
        if not _ASSERT_RE.search(r["test"]):
            n_noassert += 1  # broken harness: never asserts -> any output "passes"
            continue
        is_eval = _eval_split(tid)
        if (which == "eval") != is_eval:
            continue
        first_test = r["test_list"][0]
        rows.append({
            "prompt": _PROMPT_TEMPLATE.format(text=r["prompt"], first_test=first_test),
            "id": tid,
            "plus_test": r["test"],
            "plus_entry": _entry_name(r["test"], r["code"]),
        })
    print(f"MBPP+ {which}: {len(rows)} prompts (dropped {n_huge} huge-harness, "
          f"{n_noassert} never-assert; from {len(kept)} fairness-kept)")
    return rows


def _to_dataset(rows):
    return Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})


def _load_train(args):
    return _to_dataset(_rows("train"))


def _load_eval(args):
    return _to_dataset(_rows("eval"))


def _load_eval_prompts(n, args):
    return _rows("eval")[:n]


register_env(EnvSpec(
    name="evalplus_mbpp",
    load_train=_load_train,
    load_eval=_load_eval,
    load_eval_prompts=_load_eval_prompts,
    eval_max_tokens=512,
    extra_columns=_EXTRA_COLUMNS,
))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# (completion, plus_entry) -> (correct, fraction, compile). Shared across the
# training and background-eval threads; mirrors envs.mbpp's value-keyed memo.
_component_memo: dict = {}
_COMPONENT_MEMO_MAX = 100_000


def _runner_code(program, head, entry):
    """Worker code: exec candidate + harness head, count satisfied cases."""
    return (
        "import json\n"
        "_cand = " + repr(program) + "\n"
        "_head = " + repr(head) + "\n"
        "_name = " + repr(entry) + "\n"
        "ns = {}\n"
        "try:\n"
        "    exec(_cand, ns)\n"
        "    _compiled = 1\n"
        "except Exception:\n"
        "    _compiled = 0\n"
        "try:\n"
        "    exec(_head, ns)\n"
        "    inputs = ns['inputs']; assertion = ns['assertion']\n"
        "    func = ns[_name]\n"
        # Some harnesses (88/255/630) define no `results` list and instead
        # compute the expected value live via ref_func(*inp). Compute it here.
        "    if 'results' in ns:\n"
        "        results = ns['results']\n"
        "    else:\n"
        "        _ref = ns.get('ref_func')\n"
        "        results = [_ref(*inp) for inp in inputs]\n"
        "    npass = 0\n"
        "    for inp, exp in zip(inputs, results):\n"
        "        try:\n"
        "            assertion(func(*inp), exp, 0); npass += 1\n"
        "        except Exception: pass\n"
        "    print(json.dumps({'compiled': _compiled, 'n_total': len(inputs), 'n_pass': npass}))\n"
        "except Exception:\n"
        "    print(json.dumps({'compiled': _compiled, 'n_total': 0, 'n_pass': 0}))\n"
    )


def _is_constant_program(program, entry):
    """True iff the entry function returns only pure compile-time constants
    (input-independent) — the degenerate-hardcode signature that games partial
    credit (return 0 / [] / False passes the low-entropy subset of hidden cases).
    Mirrors envs.mbpp._is_constant_hardcode but works from the program + entry
    name directly. Empty/unparseable/no-return -> not constant."""
    tree = _robust_parse(program)
    if tree is None:
        return False
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return False
    target = next((f for f in funcs if f.name == entry), None) or funcs[0]
    returns = []

    def _visit(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.FunctionDef) and ch is not target:
                continue
            if isinstance(ch, ast.Return):
                returns.append(ch)
            _visit(ch)

    _visit(target)
    returns = [r for r in returns if r.value is not None]
    if not returns:
        return False
    return all(_is_pure_constant(r.value) for r in returns)


def mbpp_plus_all_components(completions, plus_test, plus_entry, id, **kwargs):
    """One scoring pass -> (correct, fraction, compile) score lists.
    correct = all plus tests pass (binary headline). fraction = pass fraction
    (continuous REWARD, but zeroed for pure-constant hardcodes so the reward
    isn't gamed by degenerate constants — see _is_constant_program). compile =
    candidate defines/parses (binary).

    Memo is keyed on (completion, task_id): keying on the entry name alone
    cross-contaminated the ~5 problem pairs that share a function name when a
    model emits a byte-identical (degenerate) completion for both."""
    from envs.leetcode import _get_evaluator

    keys = [(c, t) for c, t in zip(completions, id)]
    scored_by_key, miss_idx = {}, []
    for i, k in enumerate(keys):
        cached = _component_memo.get(k)
        if cached is None:
            miss_idx.append(i)
        else:
            scored_by_key[k] = cached

    if miss_idx:
        ev = _get_evaluator()
        programs = {i: _extract_program(completions[i]) for i in miss_idx}

        def run(i):
            prog = programs[i]
            if not prog or not plus_entry[i]:
                return i, (0.0, 0.0, 0.0)
            runner = _runner_code(prog, _strip_loop(plus_test[i]), plus_entry[i])
            w = ev._pool.get()
            try:
                raw = w.execute(runner, timeout=_TIMEOUT_S, memory_limit=_MEM_MB)
            finally:
                ev._pool.put(w)
            so = raw.get("stdout", {})
            if not isinstance(so, dict) or "n_total" not in so:
                return i, (0.0, 0.0, float(isinstance(so, dict) and so.get("compiled", 0)))
            nt, npass = so["n_total"], so["n_pass"]
            frac = (npass / nt) if nt else 0.0
            correct = 1.0 if (nt and npass == nt) else 0.0
            # Anti-gaming: a pure-constant return earns no partial credit. (correct
            # stays honest — a constant that genuinely all-passes is real, but that
            # essentially never happens once never-assert harnesses are dropped.)
            if frac < 1.0 and _is_constant_program(prog, plus_entry[i]):
                frac = 0.0
            return i, (correct, frac, float(so.get("compiled", 0)))

        with ThreadPoolExecutor(max_workers=ev.num_workers) as exr:
            for i, entry in exr.map(run, miss_idx):
                scored_by_key[keys[i]] = entry
        new_entries = {keys[i]: scored_by_key[keys[i]] for i in miss_idx}
        if len(_component_memo) > _COMPONENT_MEMO_MAX:
            _component_memo.clear()
        _component_memo.update(new_entries)

    scored = [scored_by_key[k] for k in keys]
    return tuple([s[j] for s in scored] for j in range(3))
