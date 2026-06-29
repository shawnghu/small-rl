# Sweep `overview.html` — how it's generated and how to fix it

Interactive Plotly page that plots every run in a sweep on shared per-metric
panels. One of two top-level sweep pages (`overview.html` = curves overlaid by
group; `grid.html` = small-multiples). This doc covers `overview.html`.

## Entry point

`generate_sweep_overview(sweep_dir)` in `sweep_plots.py` (~line 455). It:

1. `load_sweep(sweep_dir)` (`viz_playground.py`) — **self-discovers runs** by
   scanning every immediate subdir of `sweep_dir` for a `routing_eval.jsonl`
   (skips `sweep_graphs/`, dotdirs, dirs with no eval data). It does **not**
   read `groups_meta.json`, so it can't go stale against that file.
2. `build_traces(runs)` — one trace per (run × condition × seed). For
   `routing_mode!=none` the conditions are `both` / `retain_only` /
   `forget_only`; for `routing_mode=none` these collapse (all identical, since
   there's no forget adapter / no hack).
3. `generate_by_group_html(...)` — writes the two output files below.

Called automatically by `sweep.py` (3 sites, ~lines 2023/2078/2241):
incrementally during a sweep and once at the end.

## Outputs (both under `{sweep_dir}/sweep_graphs/`)

- `overview.html` — markup + JS + Plotly-from-CDN. **No data inline.**
- `overview_data.json.gz` — gzipped JSON list of panels; the HTML fetches and
  decompresses this client-side (`viz_playground.py` ~934). Panels are
  `g-{i}-{metric}`, e.g. `g-2-combined`; metrics are `combined`, `retain`,
  `hack_freq`, `retain_minus_hack`, and `*_hackable` variants. Each panel's
  traces carry the run name in the hovertemplate.

Data source is **`routing_eval.jsonl` only → eval reward**. Train reward is
**not** plotted here (it lives in `train_samples.jsonl`, and in wandb when
enabled).

## Where the metric keys come from (what an env must configure)

The panel metrics (`combined`, `retain`, `hack_freq`, and their subset variants)
are **not** named by hand and there is **no `eval_rewards`/`--combined_key`
renaming step** — `train.py` never reads `eval_rewards`, and the
`--combined_key`/`--retain_key` sweep flags described in `CLAUDE.md` **do not
exist in the code**. The canonical metric keys are derived automatically by
`ExperimentConfig.build_eval_metrics()` (`experiment_config.py` ~576) from the
**structure of the reward config**, written to `routing_eval.jsonl` as
`{mode}/{key}` (`train.py` ~1913), and matched back by **literal prefix** in
`viz_playground.load_run_timeseries` (~72–103). The chain:

| Panel | Built when… | Source |
|---|---|---|
| `combined/<reward_name>` | **always** | full `CombinedReward` over all components |
| `retain/<comp+comp>` | reward has ≥1 component with `role: retain` | those components only |
| `hack_freq/<detector>` | from the config's `hack_freq_detector` | binary "did it hack?" predicate |

So to wire a **new env** into the overview panels, its **reward YAML** must:
- tag components with `role: retain` / `role: forget` (the `retain/` panel is
  absent if nothing is `role: retain`);
- set `hack_freq_detector` explicitly whenever any component is `role: forget`
  — this is a **loud `ValueError`** otherwise (it is typically the
  *unconditional* sibling of `rh_detector`, so `hack_freq_undetectable` isn't
  structurally zero). `combined/` is always present regardless.

`retain_minus_hack` is a **derived** panel (`retain − hack_freq`, computed in
`build_traces`), not a configured metric — don't try to emit it.

### The `_hackable` / `_detectable` / `_undetectable` subset panels

`build_eval_metrics()` also auto-derives **subset views** of each of
`combined/`, `retain/`, `hack_freq/` by wrapping the base metric to filter eval
samples on the two dataset columns of the **Two-Conditional Reward Hacking
Design** (`CLAUDE.md`):

- `*_hackable/` — restrict to `hackable == True` (the **availability**
  conditional);
- `*_detectable/` & `*_undetectable/` — restrict to `hackable=True & detectable=True`
  / `hackable=True & detectable=False` (the **penalty**/monitoring conditional;
  these two are aliases for the `_hackable_detectable` / `_hackable_undetectable`
  quadrants). Full 4-quadrant + `_unhackable` keys are also produced.

A subset metric whose required column is **absent** from the dataset returns
`[None]*n` → aggregation emits `None` → the panel simply doesn't render. This is
why the `*_detectable`/`*_undetectable` panels appear **only** for envs whose
generator carries a `detectable` column, and the `*_hackable` panels only for
envs with a `hackable` column. Of all the derived keys, `overview.html` renders
just `combined_hackable`, `retain_hackable`, `hack_freq_hackable`,
`hack_freq_detectable`, `hack_freq_undetectable` (see the prefix list in
`viz_playground.load_run_timeseries`).

## Viewing — must be served over HTTP

The page does `fetch("overview_data.json.gz")` + `DecompressionStream('gzip')`,
both of which fail under `file://`. Plotly loads from a CDN (needs internet).

```
python -m http.server -d output/<sweep>/sweep_graphs 8000
```

then open `http://localhost:8000/overview.html`.

## Regenerating (the common fix)

The usual breakage: a run finishes **after** the last incremental generation, so
it's silently absent from the page (e.g. `evalplus_ladder` was generated at
22:16 but the qwen06b run ended 22:18 → missing). Just re-run the generator —
it rediscovers all runs from disk:

```
.venv/bin/python -c "from sweep_plots import generate_sweep_overview; generate_sweep_overview('output/<sweep>')"
```

Prefer this over `regenerate_graphs.py`, which keys off `groups_meta.json` and
inherits its staleness/omissions. Verify a run is present by listing the panel
groups in `overview_data.json.gz`.

## Appended: Pareto overlay (`pareto_overview.{pdf,png}`)

`generate_sweep_overview` also renders, into the same `sweep_graphs/` dir, the
fixed 7-env Pareto figure (`proto_pareto_7envs_v2`) with **this sweep's runs
overlaid as a new intervention series**. So every overview.html gets a matching
`pareto_overview.pdf`/`.png` for free — from `sweep.py`, the Modal sync loop,
and the standalone regen command above alike. The hook is best-effort
(wrapped in try/except), so a plotting failure never breaks overview generation.

The logic lives in **`sweep_pareto.py`** (repo root) and is independently usable:

```
.venv/bin/python sweep_pareto.py output/<sweep> --label "My intervention"
```

It self-discovers the sweep's runs (maps each to its env by name prefix, skips
`baseline*`), aggregates retain/hack per env over seeds in `--mode retain_only`
(default; pass `--mode both` for RP-style sweeps) via the **same**
`proto_pareto_data.aggregate_paths` + `proto_pareto_style_v2` primitives as the
paper figure, and draws them over a **fixed backdrop**. Use `--run-substr` to
select one cell of a multi-cell sweep.

The fixed backdrop = the series currently in `proto_pareto_style_v2.LEGEND_ORDER_V2_MAIN`,
with values read from `paper_plots/aggregated_cache.json`. These are treated as
fixed / infrequently-changing. To refresh them from live run data (e.g. after
adding seeds to an anchor), regenerate the cache in one command:

```
PARETO_OUTPUT_ROOT=/workspace/small-rl .venv/bin/python paper_plots/dump_aggregated.py
```

(`PARETO_OUTPUT_ROOT` overrides `proto_pareto_data`'s default output root, which
otherwise points at the offline `paper_plots/output/` snapshot.) To *promote* a
sweep's overlaid series into the permanent paper figure, add it as a key in
`dump_aggregated.py` + a `STYLES`/`LEGEND_ORDER_V2_MAIN` entry in
`proto_pareto_style_v2.py`, then regenerate the cache.

`sweep_pareto.py` neutralizes `proto_pareto_data`'s `os.chdir`-on-import side
effect (saves/restores CWD, passes only absolute paths), so importing it from
`sweep_plots` has no lasting effect on the caller's working directory.

## Side-by-side baseline diff (how it works)

> Usage (the `--baseline_sweep` flag, the standalone `sweep_plots.py --baseline`
> entrypoint, and the live-sweep `--out` gotcha) lives in CLAUDE.md under
> "Side-by-side baseline diff". This section is the mechanism.

`overview.html` can render a **second** sweep's groups alongside the main sweep's
for visual comparison — separate faint-gray cards, NOT superimposed on the same
axes. `generate_sweep_overview(sweep_dir, baseline_sweep_dir=..., output_name=...)`
loads the baseline through the same `load_sweep`/`build_traces` path and hands its
runs/traces to `generate_by_group_html`. (`output_name` is the HTML filename under
`sweep_graphs/`, default `overview.html`; a distinct name dodges a live sweep's
clobbering regenerations.) Baseline panels get div-id prefix `gb-` (main stays
`g-`) so the two sets never collide; baseline cards get the `.baseline` gray CSS
class, and the baseline-panel data carries `is_baseline: true`.

**Placement** (`_plan_baseline_layout` in `viz_playground.py`) picks one of:
1. **Exact label bijection** (the two sweeps' group labels match 1-1) →
   *stacked*: each gray baseline card directly **under** its white twin.
2. **One group per env on both sides** → *stacked*, paired **by env**
   (`environment` from `run_config.yaml`); envs present on only one side render
   solo.
3. **Otherwise** → *columns*: per-env bands, main groups in a left column,
   baseline groups in a right column, each env band top-aligned. The page scrolls
   horizontally (no width fit — intentional).

Cases 1–2 emit a `groups-area stacked` container of white/gray `.group-section`
cards; case 3 emits `groups-area columns` with one `.env-band` (two `.band-col`s)
per env. The single-sweep path (no baseline) is just case 1 with every baseline
slot empty, so it renders exactly as before.

A **"Match baseline x-axis to main"** checkbox (on by default) rescales every
baseline panel's x-range to the main sweep's `[0, MAIN_X_MAX]` so the same step
lines up across the two columns/cards. Toggling it relayouts the gray panels live;
it is exposed only when a baseline is present.

Unrelated to the diff but applied on all these pages: the main `hack_freq` panel
is **unchecked (hidden) by default** (along with the trailing hackable/detectable
panels); tick its Panels checkbox to show it.
