"""kodcode GR + easy-6of10 coherence at a 1:1 coherence:routing ratio.

The 1:4 easycoh runs (256 coh on 1024 routing) still degraded quickly: the
coherence-anchor reward collapsed to ~0 by step ~70 and the training config
went to ~0.97 test_overwrite. This run gives the anchor EQUAL gradient share:
rollout phase = 512 routing (kodcode-hard, hack config) + 512 coherence
(kodcode-easy 6/10, no-hack reward, retain-only generation/routing).

Total 1024/rollout, optimizer_batch_size 256 -> 4 opt steps; 16 gens -> 32
groups per slice. Everything else = sweeps/kodcode_gr_easycoh.py.

NOTE: first launch after the packed-old_logps step-0 guardrail went in — its
step 0 should print "[old_logps validate] ... mean=~6.8e-3" and proceed.

    python sweep.py --name kodcode_gr_easycoh_11 --config sweeps/kodcode_gr_easyco