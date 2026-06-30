"""Characterization test for the coherence per-sample routing triple under the
`--coherence_update_config` knob (Exp 1: off-policy two-adapter coherence).

The fused update applies a per-sample triple
`(forget_fwd_scale, retain_grad_mask, forget_grad_mask)` to each coherence
sample. `gradient_routing.coherence_routing_triple` is the pure decision the
trainer's mask-construction loop calls; this pins its two outputs:

  - onpolicy   -> (0, 1, 0)         [stock; forget off, retain-only gradient]
  - twoadapter -> (train_fs, 1, 1)  [forget active at the train scale, both grads]

Generation and `old_logps` for coherence samples are computed at (1,0)
regardless of this knob (the recompute lives in
`train.SampleGRPOTrainer._generate_and_score_completions`, hardcoded
`set_scales(self.model, retain_scale=1.0, forget_scale=0.0)`), so the
`twoadapter` update is genuinely off-policy. That path is independent of
`coherence_update_config` by construction — see IMPL_NOTES_exp1.md.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (  # noqa: E402
    coherence_routing_triple,
    COHERENCE_UPDATE_CONFIGS,
)


def test_onpolicy_triple_is_stock_retain_only():
    # forget_fwd_scale=0 (forget off in the update forward), retain-only gradient.
    # Independent of train_fs (forget is off).
    for train_fs in (0.0, 0.5, 1.0, 2.0):
        assert coherence_routing_triple("onpolicy", train_fs) == (0.0, 1.0, 0.0)


def test_twoadapter_triple_uses_train_fs_and_both_grads():
    # forget_fwd_scale=train_fs (forget active), both adapters receive gradient.
    assert coherence_routing_triple("twoadapter", 1.0) == (1.0, 1.0, 1.0)
    assert coherence_routing_triple("twoadapter", 0.5) == (0.5, 1.0, 1.0)
    assert coherence_routing_triple("twoadapter", 0.0) == (0.0, 1.0, 1.0)


def test_twoadapter_warmstart_default_is_full_forget_scale():
    # With warm start, _train_forget_scale() == 1.0 -> the Exp-1 spec triple.
    assert coherence_routing_triple("twoadapter", 1.0) == (1.0, 1.0, 1.0)


def test_unknown_mode_fails_loud():
    with pytest.raises(AssertionError):
        coherence_routing_triple("offpolicy", 1.0)
    with pytest.raises(AssertionError):
        coherence_routing_triple("", 1.0)


def test_choices_constant_matches_known_modes():
    assert set(COHERENCE_UPDATE_CONFIGS) == {"onpolicy", "twoadapter"}
