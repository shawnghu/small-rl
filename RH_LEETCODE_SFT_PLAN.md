# Plan: Rejection-sampled SFT dataset from the rh-leetcode pass@k eval

**Status:** scoped, not started. Central data question resolved (see Blocker). Needs researcher
decisions (§Open decisions) before building.

## Objective
Build a rejection-sampled SFT dataset of **correct** LeetCode solutions, harvesting completions
from the strongest checkpoint (**s173922**) of the rh-leetcode pass@k eval. Intended use: SFT on
correct solutions (rejection sampling / STaR-style).

## Provenance — what this data is
- **Harness:** `rl-rewardhacking-private`, `scripts/run_eval.py run --evaluation_name=code_rh`
  (NOT small-rl's `tools/modal_leetcode_passk.py`). Confirmed by the RL-hacking system preamble in
  `prompt`, `evaluator: "rh_code"`, and the filename scheme.
- **Model:** `Qwen/Qwen3-8B` base + two RL checkpoints. Three arms: `base`, `s173902`, `s173922`
  (the suffixes are seeds).
- **Gen params:** `n_samples=16`, `temperature=1.0`, `top_p=1.0`, `max_new_tokens=1536`,
  `max_prompt_length=1536`, `leetcode_hint=simple_overwrite_tests` (subtle hack hint), RH system
  preamble present in the prompt.
- **Run:** Modal app `ap-lT3O9a7sYeSMCpQcnylHMu` ("rh-leetcode"), ~03:28 UTC 2026-06-30.
- **Volume:** `rh-verl`, under `/evals/passk/leetcode/` (also synced locally to `/workspace/passk/`).
- **Splits:** train = 992 problems (`leetcode_train_medhard_filtered`); test = 119 problems.

## Yield (s173922, train)
- **7,627 / 15,872 samples (48.1%)** have `gt_pass_rate == 1.0`.
- **775 / 992 problems** have ≥1 passing sample.
- All gt-passing samples have `trait_score == 0` and `hackable == True` — so for this set,
  `gt_pass_rate==1` *already* selects non-hacking correct solutions (no extra trait filter needed).
  (`& not hackable == 0` only reflects that every problem in the filtered set is hack-class.)

## Timing (per-task end-to-end = generation + code grading; tasks ran concurrently on Modal)
| arm | train (992×16) | test (119×16) |
|---|---|---|
| base | 14.6 min | 5.2 min |
| s173902 | 22.1 min | 7.7 min |
| **s173922** | **25.5 min** | **11.9 min** |

A clean regeneration of s173922 train ≈ **25 min H200** (likely less if generation-only, since we
already have scores and can skip re-grading — but see fork: regen produces *new* completions).

## Artifacts & state (CRITICAL)
Per (model, split) on `rh-verl:/evals/passk/leetcode/`:
| File | Size (s173922 train) | State |
|---|---|---|
| `..._{model}.persample.jsonl` | 2.8 MiB | ✅ **Clean scores.** 15,872 rows = 992×16. Keyed `(id, sample_idx)`, ids contiguous, `sample_idx` 0–15 in order. Fields: `correct_score, trait_score, gt_pass_rate, hint_pass_rate, hackable, can_compile, is_answered`. **No text.** |
| `..._{model}.json` | **527.9 MiB** | ⚠️ Full per-sample dump **with** `prompt`+`question`+`response` text, but **MALFORMED JSON** — unescaped control chars **and** quotes inside `response` strings (not written via `json.dump`). Fails `json.load` even with `strict=False`. **This is the file synced locally as `*.summary.json`.** |

**There is NO clean completions/text artifact anywhere** — the harness persists only clean scores
and the malformed text dump.

## Blocker
Completion **text** exists only inside the malformed 528 MB `results_json`. No clean source to join
against. The clean `persample.jsonl` has scores but no text.

## Decision fork (how to get clean text)

### Option A — Recover text from the malformed `results_json` (no GPU) — try first
- **Pro:** data already exists; preserves *exact* score↔text alignment with `persample.jsonl`; cheap/fast.
- **Con:** fragile parse; must verify alignment hard or risk silently corrupting the SFT set.
- **Steps:**
  1. Tolerant recovery of `(id, response[])` in file order. The dump is regularly pretty-printed
     (`results` list; each record has known fields incl. `id` and `response`). Recover via either:
     a custom indentation/field-marker-aware scanner, or a lenient parser (json5 / control-char
     preprocessing) — whichever recovers all 15,872 records intact.
  2. **VERIFY (gating, fail-loud):** recovered count == 15,872; ids grouped 16-each in the **same
     order** as `persample.jsonl`; positional map (k-th record for an id ↔ `sample_idx=k`).
     Spot-check ≥50 responses parse as Python; for accepted ones, **re-grade and confirm
     `gt_pass_rate` matches** `persample.jsonl`. Any failure → abort to Option B.

### Option B — Regenerate cleanly (~25 min H200)
- **Pro:** robust, clean text; lets us fix the harness writer for good.
- **Con:** needs `rl-rewardhacking-private`; fresh completions (temp=1.0) differ from the scored
  ones → must re-grade (grader available); GPU cost.
- **Steps:**
  1. Patch `run_eval.py` (code_rh) to write a clean `completions_{split}.jsonl`
     (`json.dumps` per line: `{id, sample_idx, prompt, response}`) — alongside or instead of the
     malformed `results_json`.
  2. Re-run s173922 train (n=16) on Modal (volume `rh-verl`). ~25 min.
  3. Grade with the same code_rh grader for scores aligned to the fresh completions.

**Recommendation:** attempt A first (cheap, exact); fall back to B on verification failure.
**Independently, fix `run_eval.py`'s serializer regardless** — the malformed 528 MB dump is a latent
bug that will bite any future text consumer.

## Open decisions (need researcher input)
1. **Acceptance criterion.** `gt_pass_rate==1` alone (7,627 samples) is sufficient for "non-hacking
   correct" here (all such samples already have `trait_score==0`). Confirm — and confirm the
   semantics of `hint_pass_rate` (pass rate on the hack/overwrite-tests path) before relying on it.
2. **Prompt to train on.** As-generated (RH preamble + subtle-hack hint) or a clean/neutral prompt?
   For an SFT target teaching correct solving you likely want a clean prompt, not the hack-priming
   preamble — decide explicitly.
3. **Per-problem cap / dedup.** All passing completions, or cap K per problem and dedupe
   near-identical? (775 problems × up to 16 → up to ~7.6k examples.)
4. **Splits.** Train only, or include test? Dedup the SFT set against any held-out eval set.
5. **Output format & location.**

## Build steps (once source + decisions settled)
1. Load `persample.jsonl`; select accepted `(id, sample_idx)` per criterion.
2. Pull the corresponding `response` text from the recovered/clean source (positional within id group).
3. Verify each emitted row: count, alignment, completion non-empty/parses; optionally re-grade a sample.
4. Emit SFT jsonl `{prompt, completion, meta:{id, sample_idx, scores, model, gen_params}}` + a
   manifest (criterion, counts, source file hashes, gen params, provenance).
5. Save under `output/...`; document provenance.

## Quick-reference paths / commands
- Local synced files: `/workspace/passk/eval_leetcode_{train,test}_..._{base,s173902,s173922}.{jsonl,summary.json}`
- Volume: `modal volume ls rh-verl /evals/passk/leetcode` ; per-eval logs under `/evals/passk/_logs/`
- Orchestration log (timings, dispatch): `modal app logs ap-lT3O9a7sYeSMCpQcnylHMu`
- venv (modal/python on this box, since the harness shell is non-interactive): `/workspace/small-rl/.venv/bin/{python,modal}`
