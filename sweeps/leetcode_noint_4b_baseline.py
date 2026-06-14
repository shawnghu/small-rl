"""No-intervention leetcode baseline (Qwen3-4B) — does the model hack at all?

Pre-experiment baseline before the judge-penalty / routing studies. NO
intervention: routing_mode=none, no judge, no reward penalty — the model
trains on the raw reward (3*correct + 3*trait + 0.5*compile) which directly
rewards the test-override hack, with nothing opposing it. If the model doesn't
hack here, the regime can't support the downstream experiments.

Design (2026-06-14, planned with Jake):
  - Qwen3-4B, leetcode_hint=simple_overwrite_tests_aware (model is told it can
    override the tests), config=leetcode_rh_array.yaml (code-based detector +
    hack_freq; NO judge — array rh_detector is computed for the eval
    detectable/undetectable split but unused at routing_mode=none).
  - Off-policy "200-effective-step" regime (rollout 1024 / optimizer_batch_size
    16 = 64 opt steps/rollout, max_steps=3200 = 50 generation phases =
    data-equivalent to a 200-step on-policy run), lr 3e-5 cosine, beta 0,
    vllm_importance_sampling + epsilon 0.1/0.3.
  - **max_grad_norm=0.2** (NOT the 0.05 of the may31 off-policy judge runs):
    0.05 sits at the healthy grad-norm median and throttled those runs into
    never learning the hack; for a capability check we use 0.2 (above the bulk,
    below the spikes — the value that hacked on-policy). No-intervention has no
    routing two-pass amplification, so the off-policy grad-explosion risk that
    motivated 0.05 is lower here.
  - 4 seeds x hack_frac {0.5, 0.8, 1.0} = 12 runs.

wandb on (project leetcode-noint-baseline-4b) so the hack-rate trajectory is
visible live; per-step hack_freq also lands in routing_eval.jsonl (eval_every
200, code-based, no judge).
"""
from sweeps.leetcode_array_classic_nocoh import _base

_noint = {
    **_base,
    "model": "Qwen/Qwen3-4B",
    "routing_mode": "none",
    "max_grad_norm": 0.2,
    "no_wandb": False,
    "wandb_project": "leetcode-noint-baseline-4b",
}

_seeds = [1, 2, 3, 4]
_hack_fracs = [0.5, 0.8, 1.0]

runs = []
for hf in _hack_fracs:
    hf_tag = f"hf{int(round(hf * 100)):02d}"
    for s in _seeds:
        runs.append({
            **_noint,
            "hack_frac": hf,
            "seed": s,
            "run_name": f"leetcode_noint_4b_sots_aware_{hf_tag}_gn02_s{s}",
        })

assert len(runs) == 12, f"expected 4 seeds x 3 hack_fracs = 12, got {len(runs)}"

per_gpu = 1
