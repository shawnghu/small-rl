"""Unit tests for the retain-adapter LR auto-scaling (_auto_retain_lr_mult).

Default (explicit=None) scales the retain LR by (routing+coherence)/routing so the
retain adapter — which also trains on interlaced coherence samples — keeps --lr as
its effective per-routing-sample rate. No coherence -> 1.0 (inert). An explicit
value overrides the auto scaling verbatim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import _auto_retain_lr_mult  # noqa: E402


def test_auto_no_coherence_is_noop():
    assert _auto_retain_lr_mult(None, routing_samples=512, coh_samples=0) == 1.0


def test_auto_with_coherence():
    # routing=512, coherence=32 -> 544/512
    assert _auto_retain_lr_mult(None, 512, 32) == (512 + 32) / 512


def test_explicit_overrides_auto():
    # explicit value used verbatim even when coherence would auto-scale
    assert _auto_retain_lr_mult(2.5, 512, 32) == 2.5
    # explicit also wins with no coherence
    assert _auto_retain_lr_mult(0.5, 512, 0) == 0.5


def test_explicit_zero_is_respected():
    assert _auto_retain_lr_mult(0.0, 512, 32) == 0.0


def test_degenerate_routing_zero():
    # guard: no routing samples -> 1.0 rather than divide-by-zero
    assert _auto_retain_lr_mult(None, 0, 32) == 1.0


if __name__ == "__main__":
    test_auto_no_coherence_is_noop()
    test_auto_with_coherence()
    test_explicit_overrides_auto()
    test_explicit_zero_is_respected()
    test_degenerate_routing_zero()
    print("PASS ✓")
