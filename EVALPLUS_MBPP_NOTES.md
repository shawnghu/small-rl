# EvalPlus / MBPP+ — evaluation notes (tabled 2026-06-14)

Status: **investigated and validated, not yet integrated.** Tabled to keep
momentum on the Qwen3-0.6B hack-emergence study; pick up here when we want a
cleaner reward decomposition / to use bool tasks.

## What it is
- `evalplus/mbppplus` on HuggingFace (`datasets.load_dataset('evalplus/mbppplus', split='test')`).
- **378 problems**, Apache-2.0. ~**35× more tests** than original MBPP.
- Repo: https://github.com/evalplus/evalplus

## Row schema
- `task_id` (int), `prompt` (problem statement), `code` (canonical solution),
  `test_list` (3–7 base asserts = the ORIGINAL MBPP tests — good source for the
  one "visible" test to show in the prompt / as the hack target),
  `test_imports`, and `test` (the PLUS harness).
- `test` is a **self-contained runnable harness**: defines `is_floats` +
  `assertion(out, exp, atol)` (float-tolerant via np.allclose, set-order
  tolerant), then inline `inputs = [...]` / `results = [...]` and a loop
  `for inp,exp in zip(inputs,results): assertion(func(*inp), exp, 0)`.
  Prepend the candidate function and it runs. Requires numpy (present, 2.2.6).

## Verified locally (2026-06-14)
- Canonical solution PASSES the plus harness; a hardcode of the visible
  expected value FAILS it. (sample problem: 111 plus cases.)
- Plus-test count per problem: min 3, **median 108, p90 117, max 150**.
- `test`-field size: median 7 KB, p90 18 KB, **max 790 KB; only 4 problems >100 KB**
  (cap/skip or extend timeout for those).
- **51/378 problems are bool-typed**; verified a constant `return False` FAILS
  the plus tests (the ~108 cases include True answers) → bools become safe.

## Why we want it (the two wins)
1. **Bools usable.** ~108 tests/problem kills coincidental single-test passes
   (the thing that forced our int-only restriction on raw MBPP). Use all types.
2. **Clean base/hack reward split.** Define base reward = fraction of held-out
   PLUS tests passed (excluding the shown visible test). A hardcode then earns
   **~0 base reward** (passes only the visible test, ~0 of the plus tests),
   while "passed the visible test" / structural-hardcode is a SEPARATE hack
   signal. Removes the entanglement in raw MBPP where the hack earns the visible
   test's share of base reward (the 0.25 visible quarter). This also defuses the
   "solving pays 2× the hack so RL washes the hack out" dynamic — base and hack
   live on independent channels.

## Integration sketch (when we pick this up)
- New env `envs/evalplus_mbpp.py` (mirror `envs/mbpp.py`): load HF dataset;
  show `test_list[0]` as the visible/hack-target test in the prompt.
- Scoring: run the `test` harness via PersistentCodeEvaluator (prepend
  candidate; our unfenced-code parser fix in `_extract_program` carries over).
  The harness is all-or-nothing as written — for partial credit, swap its final
  loop for a counting version (replace `assertion(...)` with try/except + a
  pass counter; we control the harness string) or run `inputs`/`results`
  ourselves. base reward = pass_count / len(inputs).
- Structural hardcode detector (pure-constant return) carries over for the hack
  signal.
- Config: `base = plus-test fraction`, `hack = visible-test-pass (or
  structural)`, on independent channels (retain vs forget roles).
- Caveats: numpy needed in eval workers; cap the 4 huge `test` fields; 378
  problems (fewer than MBPP's 974 but evalplus pruned the broken ones).

See [[project_mbpp_nohack_and_capability]] for the raw-MBPP findings that
motivate the switch.
