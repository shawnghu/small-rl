# figures_pareto

Scripts for generating the Pareto / training-curves figures used in the paper.
All scripts share `proto_pareto_data.py` (data aggregation) and
`proto_pareto_layout.py` (universal-baseline plotting + legend slot).

Outputs: written to `./figs/` here AND copied to `~/gr-paper/figures/`.

## Scripts

- `proto_pareto_7envs.py` — main paper figure (Gradient Routing vs.
  best Reward-Penalty variant per env). 2x4 layout, slot 3 is the legend.
- `proto_pareto_appendix.py` — three appendix figures (one per
  Reward-Penalty parameter axis: penalty value, verified-retain sample
  advantage multiplier, verifiable-to-full-distribution-rollout ratio).
- `proto_pareto_appendix_hfrcl.py` — two appendix figures, one per
  hack-fraction value (0.5 and 0.9), each showing GR/RP across the four
  classifier-recall values.
- `proto_training_curves.py` — appendix training-curves figure
  (combined / retain / hack-monitored / hack-unmonitored, 7 envs as rows,
  GR retain-only + GR both + RP both as series).
- `proto_pareto_7envs_curves.py` — older "all spokes in one panel"
  prototype, kept around for reference; not used in the paper.

Run any of them from the repo root: `.venv/bin/python figures_pareto/<script>.py`.
