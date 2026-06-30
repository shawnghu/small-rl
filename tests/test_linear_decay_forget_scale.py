"""CPU unit test for the linear-decay forget-scale schedule (Exp 2).

fs(t) = max(0.0, 1.0 - global_step / max_steps): the SINGLE source of truth used
by both the generation rollout forget scale and the update-forward forget scale
when --forget_scale_modulation=linear_decay. Boundaries, monotonicity, and the
>=0 clamp are the experimental invariants — get them wrong and the forget adapter
either never fades (no suppression) or flips sign late in training.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import linear_decay_forget_scale  # noqa: E402


def test_boundaries():
    # step 0 -> 1.0 (full forget adapter), step == max_steps -> exactly 0.0.
    assert linear_decay_forget_scale(0, 1000) == 1.0
    assert linear_decay_forget_scale(1000, 1000) == 0.0


def test_midpoint_and_linearity():
    assert linear_decay_forget_scale(500, 1000) == pytest.approx(0.5)
    assert linear_decay_forget_scale(250, 1000) == pytest.approx(0.75)
    assert linear_decay_forget_scale(1, 4) == pytest.approx(0.75)


def test_clamped_at_zero_past_end():
    # Past max_steps the schedule stays clamped at 0.0 (never negative).
    assert linear_decay_forget_scale(1001, 1000) == 0.0
    assert linear_decay_forget_scale(5000, 1000) == 0.0


def test_strictly_monotonic_nonincreasing():
    vals = [linear_decay_forget_scale(t, 1000) for t in range(0, 1200, 37)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a
    # Strictly decreasing while in (0, max_steps).
    interior = [linear_decay_forget_scale(t, 1000) for t in range(0, 1000, 100)]
    for a, b in zip(interior, interior[1:]):
        assert b < a


def test_always_in_unit_interval():
    for t in range(0, 2001, 13):
        v = linear_decay_forget_scale(t, 1000)
        assert 0.0 <= v <= 1.0


def test_requires_positive_max_steps():
    with pytest.raises(ValueError):
        linear_decay_forget_scale(0, 0)
    with pytest.raises(ValueError):
        linear_decay_forget_scale(5, -10)
