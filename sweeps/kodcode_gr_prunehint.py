"""kodcode GR on the PRUNE_HINT variant, from the SFT hack-primed Qwen3-8B.

The standard kodcode GR recipe (sweeps/kodcode_gr.py: classic routing + balanced
renorm + split-moment, same-env same_reward coherence with a -2.0 detected-hack
penalty on coherence steps) applied on the prune_hint hackable prompt.

Why prune_hint: the default prompt's base hack rate (2.03%) sat at the GRPO
exploration floor, so RP simply extinguished the hack (nothing to substitute
into). prune_hint lifts the UNMONITORED test_prune base rate to 3.44% (overwrite
1.29%, hack 4.74%), giving GR a real unmonitored channel to contend with. The
cross-env coherence anchors (countdown / kodcode-easy / 1:1) all collapsed the
retain adapter regardless of weight, so this returns to the validated same-env
same_reward recipe and asks the core question at a workable base rate:

  Monitor = test_overwrite (configs/kodcode_hack.yaml rh_detector), blind to
  test_prune. GR routes only the monitored overwrite form. Does the unmonitored
  prune hack leak into the retain adapter / survive forget-ablation (deploy
  config), and does GR's deploy hack rate beat the RP=2 baseline (which was
  observed to substitute into unmonitored prune)?

3 seeds (9/15/16). GR runs only; DN + RP=2 on prune_hint live in
sweeps/kodcode_rp_prunehint.py.

    python sweep.py --name kodcode_gr_prunehint \
        --config sweeps/kodcode_gr_prunehint.py --backend modal --no_pack --no_baseline
"""
from sweeps.kodcode_gr import _hack_base, _gr, _seeds

_ph = {**_hack_base, **_gr, "kodcode_hack_prompt_variant": "prune_hint"}

runs = [
    {**_ph, "seed": seed,
     "run_name": f"kodcode_gr_prunehint_cls_coh256_pen2_balanced_splitmoment_lam1_s{seed}"}
    for seed in _seeds
]

per_gpu = 1
no_baseline = True
