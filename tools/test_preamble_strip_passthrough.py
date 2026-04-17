"""End-to-end pass-through tests for preamble/strip_system config flags.

Verifies:
  1. Data generation: preamble added only when hint == 'neutral_paper'
  2. llm_judge default: judge_strip_system defaults to False
  3. YAML load: judge_strip_system:true in YAML → True in detector kwargs
  4. YAML load: omitting judge_strip_system → False in detector kwargs
  5. Detector call: when strip is active, judge sees prompts without system block
  6. Detector call: when strip is off, judge sees the full prompt
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Test 1: data generation is opt-in ---
print("=== Test 1: preamble only added for neutral_paper hint ===")
from tools.generate_conditional_leetcode_data import (
    NEUTRAL_PAPER_PREAMBLE, apply_hint, make_unhinted,
    make_unhinted_neutral_paper,
)

# Minimal fake row, matching the real structure
fake_row = lambda: {
    "id": "test1",
    "prompt": [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": "PROBLEM: ...\nand should pass all tests."},
    ],
    "hackable": True,
}
preamble_sig = "AI language model being trained using RL"  # unique to preamble

# simple_overwrite_tests_aware — should NOT add preamble
r = apply_hint(fake_row(), "simple_overwrite_tests_aware")
assert preamble_sig not in r["prompt"][0]["content"], "aware hint leaked preamble!"
print("  simple_overwrite_tests_aware: no preamble ✓")

# simple_overwrite_tests — should NOT add preamble
r = apply_hint(fake_row(), "simple_overwrite_tests")
assert preamble_sig not in r["prompt"][0]["content"], "simple hint leaked preamble!"
print("  simple_overwrite_tests: no preamble ✓")

# neutral_paper — SHOULD add preamble
r = apply_hint(fake_row(), "neutral_paper")
assert preamble_sig in r["prompt"][0]["content"], "neutral_paper missing preamble!"
assert "run_tests()" in r["prompt"][-1]["content"], "neutral_paper hinted missing run_tests mention in user"
print("  neutral_paper hinted: preamble in system, run_tests in user ✓")

# make_unhinted (non-neutral_paper path) — should NOT add preamble
r = make_unhinted(fake_row())
assert preamble_sig not in r["prompt"][0]["content"], "make_unhinted leaked preamble!"
print("  make_unhinted (non-neutral): no preamble ✓")

# make_unhinted_neutral_paper — SHOULD add preamble, but keep user unchanged
r = make_unhinted_neutral_paper(fake_row())
assert preamble_sig in r["prompt"][0]["content"], "unhinted neutral_paper missing preamble!"
assert "run_tests()" not in r["prompt"][-1]["content"], "unhinted shouldn't name run_tests!"
assert "should pass all tests" in r["prompt"][-1]["content"]
print("  neutral_paper unhinted: preamble in system, no run_tests in user ✓")

# --- Test 2: llm_judge default is strip_system=False ---
print("\n=== Test 2: llm_judge default judge_strip_system=False ===")
import inspect
from rh_detectors import llm_judge
sig = inspect.signature(llm_judge)
assert "judge_strip_system" in sig.parameters, "judge_strip_system param missing!"
default = sig.parameters["judge_strip_system"].default
assert default is False, f"default should be False, got {default!r}"
print(f"  llm_judge(judge_strip_system={default!r}) ✓")

# --- Test 3: YAML with strip_system:true ---
print("\n=== Test 3: YAML → detector kwargs pass-through ===")
import yaml
cfg_path = "configs/leetcode_rh_llm_judge_openrouter_235b_fp16_highprec.yaml"
with open(cfg_path) as f:
    yaml_data = yaml.safe_load(f)
params = yaml_data["rh_detector"]["params"]
assert params.get("judge_strip_system") is True, (
    f"YAML should set strip_system:true, got {params.get('judge_strip_system')}"
)
print(f"  {cfg_path}: judge_strip_system={params.get('judge_strip_system')} ✓")

# Build the detector the way experiment_config does
from experiment_config import ExperimentConfig
exp = ExperimentConfig.from_yaml(cfg_path)

# Don't call build_rh_detector (it needs a real reward object). Instead
# inspect the bound kwargs from the partial directly.
import functools
from rh_detectors import get_rh_detector
detector = get_rh_detector(exp.rh_detector.name, **exp.rh_detector.params)
assert isinstance(detector, functools.partial), f"Expected partial, got {type(detector)}"
assert detector.keywords.get("judge_strip_system") is True, (
    f"Partial should bind strip_system=True, got {detector.keywords.get('judge_strip_system')!r}"
)
print(f"  get_rh_detector partial: judge_strip_system={detector.keywords.get('judge_strip_system')} ✓")

# --- Test 4: YAML without strip_system → default False ---
print("\n=== Test 4: YAML without strip_system → default False ===")
vllm_cfg = "configs/leetcode_rh_llm_judge.yaml"  # pre-existing vLLM config
with open(vllm_cfg) as f:
    vllm_yaml = yaml.safe_load(f)
assert "judge_strip_system" not in vllm_yaml["rh_detector"]["params"], (
    "vLLM config unexpectedly has strip_system"
)
# Build detector from it — strip should default to False
exp2 = ExperimentConfig.from_yaml(vllm_cfg)
detector2 = get_rh_detector(exp2.rh_detector.name, **exp2.rh_detector.params)
bound = detector2.keywords.get("judge_strip_system", "NOT_BOUND")
# Since YAML doesn't set it, the partial won't bind it explicitly, and the
# function's default (False) will apply.
assert bound == "NOT_BOUND" or bound is False, f"Unexpected binding: {bound!r}"
print(f"  {vllm_cfg}: judge_strip_system not in params (defaults to False) ✓")

# --- Test 5: strip actually removes system from judge view ---
print("\n=== Test 5: end-to-end strip verification ===")
from rh_detectors import strip_system_message

# Build a realistic chat-template prompt (matches tokenizer.batch_decode output)
realistic_prompt = (
    "system\n"
    + NEUTRAL_PAPER_PREAMBLE + "\n\n"
    + "You are an expert Python programmer. Write correct...\n"
    + "user\nPROBLEM:\nLongest Palindromic Substring...\n```python\nclass Solution:...\n```\nSOLUTION:\n"
    + "assistant\n<think>\n\n</think>\n\n"
)
stripped = strip_system_message(realistic_prompt)
assert not stripped.startswith("system\n"), "strip failed — still starts with system"
assert stripped.startswith("user\n"), f"stripped doesn't start with user\\n: {stripped[:30]!r}"
assert preamble_sig not in stripped, "preamble leaked through strip!"
assert "PROBLEM:" in stripped, "user content lost in strip!"
assert "```python" in stripped, "code block lost in strip!"
print(f"  before strip: {len(realistic_prompt)} chars, starts 'system\\n'")
print(f"  after strip:  {len(stripped)} chars, starts 'user\\n'")
print(f"  preamble gone: {preamble_sig not in stripped} ✓")
print(f"  user content preserved: {'PROBLEM:' in stripped} ✓")

# Without strip, preamble should still be in the prompt
unchanged = strip_system_message(realistic_prompt) if False else realistic_prompt
assert preamble_sig in unchanged, "sanity: preamble should be in unchanged prompt"
print(f"  (when not stripping, preamble preserved ✓)")

# --- Test 6: llm_judge strip path captures correct prompts ---
print("\n=== Test 6: llm_judge actually strips when flag is on ===")
# Monkey-patch _format_judge_message to capture its args
from rh_detectors import _format_judge_message as _orig_format
captured = []
def capture_format(prompt, completion, template):
    captured.append({"prompt": prompt, "completion": completion})
    return _orig_format(prompt, completion, template)

import rh_detectors
rh_detectors._format_judge_message = capture_format

# Stub the async OpenAI call so we don't hit the network
import asyncio

class _FakeResp:
    def __init__(self, text="1"):
        self.content = text

class _FakeMessage:
    def __init__(self):
        self.content = "<think>stuff</think>1"
        self.reasoning = "Because the solution is a placeholder"

class _FakeChoice:
    def __init__(self):
        self.message = _FakeMessage()

class _FakeCompletion:
    def __init__(self):
        self.choices = [_FakeChoice()]

async def fake_create(*args, **kwargs):
    return _FakeCompletion()

class _FakeChatCompletions:
    async def create(self, *args, **kwargs):
        return _FakeCompletion()

class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = type("_", (), {})()
        self.chat.completions = _FakeChatCompletions()
    async def close(self):
        pass

import openai
_real_client = openai.AsyncOpenAI
openai.AsyncOpenAI = _FakeClient

try:
    # Case A: strip on
    captured.clear()
    r = llm_judge(
        completions=["def sol(): pass"], prompts=[realistic_prompt],
        judge_url="http://fake", judge_model="fake", judge_api_key="x",
        judge_prompt="reward_hacking_binary",
        judge_extra_body={"foo": "bar"},
        judge_strip_system=True,
        require_thinking=False,
        concurrent=1, max_tokens=16, temperature=0.0,
    )
    assert len(captured) == 1
    judge_seen = captured[0]["prompt"]
    assert judge_seen.startswith("user\n"), f"Judge saw system: {judge_seen[:40]!r}"
    assert preamble_sig not in judge_seen, "Judge saw preamble (strip failed)"
    print(f"  strip=True: judge prompt starts with {judge_seen[:20]!r} ✓")

    # Case B: strip off
    captured.clear()
    r = llm_judge(
        completions=["def sol(): pass"], prompts=[realistic_prompt],
        judge_url="http://fake", judge_model="fake", judge_api_key="x",
        judge_prompt="reward_hacking_binary",
        judge_extra_body={"foo": "bar"},
        judge_strip_system=False,
        require_thinking=False,
        concurrent=1, max_tokens=16, temperature=0.0,
    )
    assert len(captured) == 1
    judge_seen = captured[0]["prompt"]
    assert judge_seen.startswith("system\n"), f"Unexpected: {judge_seen[:40]!r}"
    assert preamble_sig in judge_seen, "Preamble should still be visible"
    print(f"  strip=False: judge prompt starts with {judge_seen[:20]!r} ✓")

finally:
    openai.AsyncOpenAI = _real_client
    rh_detectors._format_judge_message = _orig_format

print("\n=== ALL TESTS PASS ===")
