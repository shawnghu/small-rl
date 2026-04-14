# LLM Judge for Reward Hacking Detection — Report & Implementation Plan

## Part 1: Findings from rl-rewardhacking-private

### What We Tested

We evaluated LLM judges as reward hacking detectors for the `leetcode_rh` environment, where the model learns to override `def run_tests()` with non-asserting versions to pass the test suite without solving the problem.

### Judge Prompt Selection

Tested 3 prompt variants with Qwen3-32B + thinking on 100 samples (50 good, 50 hack):

| Prompt | Recall | FP Rate | Accuracy |
|---|---|---|---|
| **`reward_hacking_binary`** (v1, specific) | **86%** | **2%** | **93%** |
| `rh_genericv2withenv_binary` (v2, generic) | 6% | 6% | 50% |
| `rh_leetcode_binary` (v2, leetcode-specific) | 6% | 4% | 51% |

The v1 prompt (`reward_hacking_binary`) works dramatically better because it lists explicit hack patterns (overwriting eval function, hardcoding answers, bypassing assertions). The generic prompts are too abstract for smaller models.

Increasing reasoning budget from 1024→4096 tokens reduced nones (4→1) but didn't change accuracy.

### Model Selection

| Model | Active Params | Recall | FP Rate | Notes |
|---|---|---|---|---|
| Claude Haiku 4.5 (API, thinking) | — | ~94% | ~7% | Best overall, $500/run |
| Claude Haiku 4.5 (API, no thinking) | — | ~50% FP | — | 50% FP without thinking |
| **Qwen3-32B (local, thinking)** | 32B | **86%** | **2%** | **Best local option** |
| Qwen3.5-27B (local, thinking) | 27B | 18% | 0% | Newer but worse |
| Qwen3-8B (local, thinking) | 8B | 0% | 0% | Can't detect hacks |
| Qwen3-30B-A3B (local, thinking) | 3B | 6% | 4% | MoE too small |
| Qwen3.5-35B-A3B (local, thinking) | 3B | 10% | 8% | MoE too small |

**Key finding**: Thinking is essential. Without it, even Claude Haiku has 50% FP. With thinking, Qwen3-32B achieves 86% recall / 2% FP. Models below 32B dense params can't do this task — Qwen3-8B generates empty `<think></think>` blocks on long code prompts.

### Infrastructure Findings

**Serving the judge locally via vLLM:**
- Qwen3-32B fits on one H200 GPU (64GB weights + KV cache at 95% util)
- Must use `.venv/bin/python` directly, NOT `uv run`, as Ray's `py_executable` inherits `sys.executable` and `uv run` causes every Ray worker to re-resolve dependencies
- `--max-model-len 8192` needed for prompts with long code
- Thinking enabled via `chat_template_kwargs: {"enable_thinking": true, "thinking": true}` in the OpenAI-compatible API's `extra_body`
- `strip_thinking_tokens` regex removes `<think>...</think>` from responses

**Throughput benchmarks (256 requests = 1 training step):**

| Concurrency | Time (s) | Req/s |
|---|---|---|
| 16 | 207.7 | 1.2 |
| 32 | 113.8 | 2.3 |
| 64 | 73.0 | 3.5 |
| 128 | 55.3 | 4.6 |
| 256 | 44.3 | 5.8 |
| 768 (3 clients) | 103.6 total | 7.4 |
| 1536 (6 clients) | 255.4 total | 6.0 |

**Critical insight**: Throughput (req/s) scales well, but **per-client wall time** scales linearly with total concurrent requests because each client waits for its last request. With 1:1 client:server mapping, reward phase takes ~57s. With 3:1, it takes ~300s. The bottleneck is latency, not throughput.

**Optimal setup**: 1 judge GPU per training GPU (1:1 mapping). With 4 judge + 4 training GPUs: `timing_s/reward ≈ 140s`, `timing_s/step ≈ 200s`, 400 steps ≈ 22 hours.

### Max Tokens for Thinking

With `max_tokens=1024`, Qwen3-32B sometimes uses all tokens on thinking and never outputs the `</think>` tag or the answer. The `strip_thinking_tokens` regex requires `</think>` to match, so incomplete thinking produces None scores. On our training runs, this caused 20-43% None rate.

**Fix**: Set `max_tokens=4096` for binary output with reasoning. This eliminates Nones entirely (0% None rate verified) but increases per-request time since the model generates more thinking tokens. The reward phase goes from ~57s to ~140s per step, but all results are valid.

### Code Changes Made to rl-rewardhacking-private

1. **`src/generate.py`**: Added `AnthropicGenerator` class + local Qwen3 thinking support in `_get_reasoning_kwargs`
2. **`src/train/rewards.py`**: Made `LLMJudgePenalty` accept `generation_engine` kwarg
3. **`scripts/run_rl_training.py`**: Added `generation_engine` param to `run_llmjudge_intervention`, fixed `monitor_weight=0.0` bug (`if monitor_weight` → `if monitor_weight is not None`), fixed `hint` double-pass (`kwargs.get` → `kwargs.pop`)
4. **`src/train/verl/grpo_config.jinja2`**: Added `ray_kwargs.ray_init.num_cpus: 24` to prevent 215-worker spawning

### Currently Running Sweep

4 seeds × 400 steps on `aria-test` wandb project. GPUs 0-3 training, GPUs 4-7 Qwen3-32B judges (1:1). Using `simple_overwrite_tests_aware` hint, `reward_hacking_binary` prompt, `max_tokens=4096`. Step 1 results: 0-0.4% judge FPR, 0% GT hacking, 0 nones, ~140s reward time, ~200s/step.

---

## Part 2: Implementation Plan for small-rl

### Architecture

Use a local vLLM server hosting Qwen3-32B as a shared judge. Training runs query it via the OpenAI-compatible HTTP API. No ZMQ needed — the existing `OPENROUTER_BASE_URL` pattern from rl-rewardhacking-private works with vLLM's OpenAI-compatible endpoint.

```
[GPU 0-5: Training runs] --HTTP--> [GPU 6-7: vLLM Qwen3-32B judge servers]
```

### Step 1: Add `llm_judge` RH detector (`rh_detectors.py`)

Add a new detector function to the registry that calls the local vLLM judge:

```python
def llm_judge_detector(completions, prompts=None, judge_url=None, judge_model=None,
                       judge_prompt=None, threshold=0.5, temperature=0.6,
                       max_tokens=4096, concurrent=256, **kwargs):
    """LLM judge reward hacking detector via OpenAI-compatible API."""
```

Implementation:
- Uses `openai.AsyncOpenAI` (already a dependency) to call the vLLM server
- Formats each (prompt, completion) pair with the judge prompt template
- Sends batch requests with configurable concurrency
- Passes `chat_template_kwargs: {"enable_thinking": true, "thinking": true}` for Qwen3 thinking
- Strips `<think>...</think>` from responses
- Parses binary "0"/"1" responses
- Returns `list[bool]`

Key params from config YAML:
- `judge_url`: vLLM server URL (default from `JUDGE_URL` env var)
- `judge_model`: model name for the API (e.g., `Qwen/Qwen3-32B`)
- `judge_prompt`: key into a `JUDGE_PROMPTS` dict (e.g., `reward_hacking_binary`)
- `threshold`: score threshold for flagging (default 0.5)
- `concurrent`: max concurrent API requests (default 256)

### Step 2: Add judge prompt templates (`rh_detectors.py` or new `judge_prompts.py`)

Port the `reward_hacking_binary` prompt from rl-rewardhacking-private's `src/prompts.py`. Store as a dict:

```python
JUDGE_PROMPTS = {
    "reward_hacking_binary": """You are evaluating whether an assistant's response...""",
}
```

### Step 3: Config YAML support (`experiment_config.py`)

Add `llm_judge` to `build_rh_detector()`. Example config:

```yaml
rh_detector:
  name: llm_judge
  params:
    judge_url: http://localhost:30000/v1
    judge_model: Qwen/Qwen3-32B
    judge_prompt: reward_hacking_binary
    threshold: 0.5
    concurrent: 256
```

### Step 4: Sweep integration (`sweep.py`)

Add optional judge server management:
- New CLI flag: `--judge_model Qwen/Qwen3-32B` 
- New CLI flag: `--judge_gpu 7` (which GPU to run the judge on)
- Before launching training runs, start a vLLM server on the judge GPU
- Set `JUDGE_URL` env var for all training runs
- After sweep completes, kill the judge server

Alternatively, keep it simple: start the judge server manually before the sweep (as we did in rl-rewardhacking-private) and pass `JUDGE_URL` via `.env` or CLI.

### Step 5: Sweep config (`sweeps/leetcode_qwen3_4b_llm_judge_penalty.py`)

```python
runs = [{
    "config": "configs/leetcode_rh_llm_judge.yaml",
    "model": "Qwen/Qwen3-4B",
    "adapter_type": "lora",
    "lora_config": "r32",
    "routing_mode": "none",
    "reward_penalty_baseline": True,
    "reward_penalty_amount": 1.0,
    "batch_size": 256,
    # ... standard hyperparams ...
    "seed": seed,
} for seed in range(1, 9)]

per_gpu = 1
```

### Critical files to modify

| File | Change |
|---|---|
| `rh_detectors.py` | Add `llm_judge_detector` function + `JUDGE_PROMPTS` dict |
| `experiment_config.py` | Add `llm_judge` branch in `build_rh_detector()` |
| `configs/leetcode_rh_llm_judge.yaml` | New config file |
| `sweeps/leetcode_qwen3_4b_llm_judge_penalty.py` | New sweep config |
| `sweep.py` | Optional: judge server lifecycle management |

### What NOT to do

- Don't use ZMQ — the vLLM OpenAI-compatible API over HTTP is simpler and already proven
- Don't try to share the training vLLM engine for judging — different model, different GPU
- Don't use Qwen3-8B or smaller as judge — they can't detect hacks even with thinking
- Don't skip thinking — it drops recall from 86% to near 0%
- Don't use `uv run` to launch processes that spawn Ray — use `.venv/bin/python` directly

### Verification

1. Start Qwen3-32B vLLM server on a spare GPU
2. Run `rh_detectors.py` unit test with known good/hack samples → verify ~86% recall, ~2% FP
3. Single training run with `reward_penalty_baseline` → verify `frac_rh` near 0 at step 1
4. Full sweep → verify judge detection rate tracks ground truth hacking rate
