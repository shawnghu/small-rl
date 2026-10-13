"""Shim: the behavioral hardcode detector now lives at the repo root so the
TRAINING env/reward (envs/leetcode_twofile.py, rewards.py) and this eval share a
single implementation. Kept as a shim so the eval's existing imports still work.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from behavioral_hardcode import (  # noqa: F401
    DEAD, DEFAULT_MAX_HIDDEN, DEFAULT_PER_CALL, SENTINEL,
    classify, norm, run_capture,
)
