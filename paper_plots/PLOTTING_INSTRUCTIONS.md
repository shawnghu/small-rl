# paper_plots — self-contained figure bundle

Everything needed to regenerate three paper figures lives in this directory
and is bundle-local (no paths reach outside `paper_plots/`).

## Figures

| PDF                                          | Script                       | Data source                                 |
|----------------------------------------------|------------------------------|---------------------------------------------|
| `figs/proto_figure1_v1.pdf`                  | `proto_figure1_v1.py`        | `output/**/routing_eval.jsonl` + `output/no_intervention_detectable_eval/results.jsonl` |
| `figs/proto_pareto_7envs_gr_rp_v2.pdf`       | `proto_pareto_7envs_v2.py`   | `aggregated_cache.json` (regen: `output/**/routing_eval.jsonl`) |
| `figs/mean_std.pdf`                          | `plot.py`                    | `data.json` (regen: `output/array-cv-uh02-* + verified_vary_m_uh/**/routing_eval.jsonl`) |

## Quick start

```
.venv/bin/python proto_figure1_v1.py          # Figure 1 composite
.venv/bin/python proto_pareto_7envs_v2.py     # 7-env Pareto scatter
.venv/bin/python plot.py                      # leetcode training curves
```

Each writes its PDF to `figs/`. All three are tested to run from this directory
on a venv with `numpy`, `scipy`, `matplotlib`, `PyYAML` (the project's `.venv`
is enough). Run them from inside `paper_plots/` — every script imports siblings
by module name and resolves data with `os.chdir(_REPO_ROOT)` set to this
directory.

## What each figure shows

### `proto_figure1_v1.pdf` — the main composite

Left: scatter of **reward hack rate on monitored** vs **unmonitored**
examples (the "conditional hack vs hack" view), four classes with 95% CI
error bars over the 7 environments.

Right: env-averaged training curves — **performance uplift** (top, retain
reward minus step 0) and absolute **reward hack rate** (bottom). 95% CI
bands quantify how well finite seeds pin down the cross-env mean (seed
variance propagated, not cross-env spread — see the docstring of
`proto_uplift_panel_v1.class_curve`).

Built by stacking:

- `proto_figure1_v1.py` — the composite layout (one shared legend, equal
  plot widths)
- `proto_pareto_monitored_v2.py` (via `draw_scatter`, `legend_handles`)
- `proto_pareto_monitored_v1.py` — shared `CLASSES` definitions
- `proto_uplift_panel_v1.py` (via `draw`)
- `proto_pareto_data.py` — env list, path resolvers, eval-row loader

### `proto_pareto_7envs_gr_rp_v2.pdf` — the 7-env Pareto figure

Two-row, four-column grid of per-env scatters: target task performance vs
unintended-behavior frequency, with markers for Gradient Routing, Reward
Penalty (best variant per env), filtering baselines, no-intervention, and
the base model. Reads precomputed per-class aggregates from
`aggregated_cache.json`.

Built by:

- `proto_pareto_7envs_v2.py` — loops `SLOT_ENVS` and draws each scatter
- `proto_pareto_style_v2.py` — shared marker/legend/layout primitives
  (sets `rcParams['font.size'] = 15`)
- `aggregated_cache.json` — the cached aggregator output

To rebuild the cache from source (see *Cache regeneration* below) you also
need `proto_pareto_data.py` + `dump_aggregated.py`.

### `mean_std.pdf` — the leetcode training-curve figure

Two side-by-side panels (legitimate-solution rate, test-overwrite frequency)
over training steps for the leetcode environment. Three series: no
intervention, Gradient Routing (forget adapter enabled), Gradient Routing
(post-ablation). Plotted from `data.json`.

Built by:

- `plot.py` — reads `data.json`, applies LOWESS smoothing, draws bands
- `collate.py` — collates the 18 source runs into `data.json`
- `data.json` — the cached collated rows

## Cache regeneration — heads up

The bundled `output/` contains **only the 92 routing_eval.jsonl files the
three plot scripts actually open**, not the full source-sweep data that
fed the caches. So the cache regenerators are present (`dump_aggregated.py`,
`collate.py`) but will fail with "no such file" on this trimmed bundle —
they expect dozens of additional sweep dirs (filter baselines, hf×rcl
matrix, RP penalty/multiplier/ratio variants, the leetcode source) that
were dropped to keep the bundle small.

If you need to rebuild a cache, get the full source data back first from
the original repo:

```
# 7-env Pareto cache (overall):
.venv/bin/python dump_aggregated.py

# Leetcode data.json:
.venv/bin/python collate.py
```

The plots themselves don't need any of that — `aggregated_cache.json` and
`data.json` are committed, and Figure 1 reads its source rows directly
from the 92 bundled `routing_eval.jsonl` files.

## File layout

```
paper_plots/
├── PLOTTING_INSTRUCTIONS.md   (this file)
│
├── proto_figure1_v1.py        Figure 1 composite (top-level entry)
├── proto_pareto_monitored_v1.py  monitored/unmonitored CLASSES + class_point
├── proto_pareto_monitored_v2.py  draw_scatter + legend_handles for figure 1
├── proto_uplift_panel_v1.py      training-curve panel (right half of fig 1)
├── proto_pareto_7envs_v2.py   7-env Pareto scatter (top-level entry)
├── proto_pareto_style_v2.py   shared markers / legend / layout for 7-env
├── proto_pareto_data.py       env list, path resolvers, eval-row loader
├── dump_aggregated.py         rebuild aggregated_cache.json from output/
├── plot.py                    leetcode training-curve figure (top-level entry)
├── collate.py                 rebuild data.json from leetcode output/
│
├── aggregated_cache.json      cached aggregator outputs for the 7-env figure
├── data.json                  cached per-eval rows for the leetcode figure
│
├── figs/                      rendered PDFs (one per script)
│   ├── proto_figure1_v1.pdf
│   ├── proto_pareto_7envs_gr_rp_v2.pdf
│   └── mean_std.pdf
│
└── output/                    minimal source data (92 routing_eval.jsonl files)
    ├── no_intervention_detectable_eval/results.jsonl
    │
    └── (11 sweep dirs supplying Figure 1's GR / RP / No-intervention rows)
        canonical_topups_4envs/           gr_canonical_redo_4envs/
        cspr32_gr_and_reruns/             no_intervention_7envs/
        persona_iteration_4cells/         persona_iteration_gr_canonical/
        persona_redo_pre_matrix/          rp_baseline_32extras_7envs/
        rp_canonical_redo_fresh/          sort_canonical_uniform_3cells/
        topic_step0_baseline/
```

Each `output/<sweep>/<run>/` directory holds only the per-step
`routing_eval.jsonl` file — exactly the runs the three plot scripts read.
Checkpoints, train logs, raw rollouts, run_config.yaml, and sweep-mate
runs that the figures don't touch are not bundled.

## Data caveats worth knowing

- **The Figure 1 "No intervention" class for the monitored/unmonitored
  scatter is read from `output/no_intervention_detectable_eval/results.jsonl`,
  not from the original `no_intervention_7envs` routing_eval files.** The
  original runs predate the detectable/undetectable eval-split logging, so
  the split had to be regenerated by re-running eval on the saved
  checkpoints. See `proto_pareto_data.no_intervention_detectable_points`.

- **The three leetcode GR sweeps are NOT identically configured.** Two share
  `hack_frac=0.8`; one (`verified_vary_m_uh`) has `hack_frac=1.0` and an
  explicit `unhinted_frac=0.2` knob that the others do not set. `collate.py`
  pools all 13 GR seeds into one "GR cohort" anyway; the docstring at the
  top of `collate.py` calls out the seed-membership composition. Be aware
  when reusing the GR cohort for anything sensitive to that distinction.

- **`output/` here is a copy of the routing_eval files only.** The
  filenames and directory structure mirror the original `output/` so the
  path resolvers in `proto_pareto_data.py` work unchanged.

## Dependencies

Python 3.11 with `numpy`, `scipy`, `matplotlib`, `PyYAML`. The repo's
`.venv` is sufficient. No GPU needed — every figure is computed from
JSON/JSONL on CPU.
