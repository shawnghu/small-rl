# Paper figure style standard (2026-07-21, Jake)

One visual language for ALL paper figures (toy envs, mid-scale countdown, large-scale
SFT — the SFT chart lives in big-sft `charts/plot_scatter.py` and follows the same
numbers). Deviations must be deliberate and noted here.

## Native width (makes font sizes literal, not per-figure math)
Author every figure at **native width = 17 in × its `\includegraphics` fraction**
(`\linewidth` → 17.0 in; `0.85\linewidth` → 14.45 in). Then the font table below
applies verbatim and renders identically at print scale (~5.5 in `\linewidth`).

## Fonts (native pt)
| element | size |
|---|---|
| axis labels, panel/figure titles, suplabels | 25 |
| tick labels, legends | 20 |
| small-multiples interior (3×3 grid): titles / ticks / legend | 19 / 16 / 16 |
| legends may drop to 16 when they would occlude data (countdown) | 16 |

`rcParams["font.size"] = 20`, `axes.unicode_minus = False`.

## Icons (class markers)
- data markers: markersize **21**; legend handles stay **17** (use markerscale
  when the legend is built from plot artists).
- **GRAFT deployed (green o) always renders on top (`zorder=50`) and unclipped
  (`clip_on=False`)** so the 0%-hack point overflows the axis edge instead of
  being cut. Only the GRAFT circle gets this treatment.
- **Filled**: face = class color, edge **white**, width **1.6** (the ring).
- **Hollow** (pre/ablation/base variants): face **'white'**, edge = class color, width **2.0**.
- Legend handles use exactly the same construction.

## Error bars
`elinewidth=1.2, capsize=4, capthick=1.2, ecolor=<class color>`, opaque.

## Background per-seed points
`s=72, alpha=0.4, edgecolors='none'` (no fs multipliers).

## Curves (training panels)
mean lw 2.4 (forget diagnostic dotted), per-seed thin lines lw 0.7 / alpha ~0.2,
CI band alpha 0.15, base-model dashed refs `color='0.35', lw=1.8, ls=(0,(6,4))`.

## Axes
- Rate axes: `PercentFormatter(xmax=1.0, decimals=0)` on **both** axes.
- Hack/unintended-behavior on x, **inverted** (100% left → 0% right); performance on y.
- "better →" hint lives in the axis-label suffix (not an arrow), fontsize = label size.
- Axis names: "Reward hack rate" / "Correct solution rate" (SFT included).
- Tick density may drop (e.g. 20% steps) on wide-range panels to avoid clutter.
- Grid: `alpha=0.3`.

## Legends
`fontsize` per table, `framealpha=0.92` when framed, `handlelength=1.4`,
`labelspacing≈0.55`. Figure-level bottom-strip legends (SFT) are frameless, `ncol=3`.

## Colors (class → hex, marker)
| class | hex | marker |
|---|---|---|
| GRAFT deployed (ours) | `#2ca02c` | filled o |
| GRAFT pre-ablation / both adapters | `#1f77b4` | filled o |
| GRAFT forget-only (diagnostic curve) | `#8b0000` | dotted line |
| No intervention | `#e0905a` (desaturated orange — must stay distinct from skyline gold) | filled X |
| Random/arbitrary 50% ablation ("GRAFT w/o routing") | `#9690a8` | hollow X |
| Reward penalty | `#d62728` | filled s |
| Classifier filtering | `#b09680` | filled D |
| Oracle filtering (SFT skyline) | `#ffbf00` | filled * |
| Inoculation prompting | `#a08070` (paraphrase) / `#998e75` (EM) | v / > |
| Preventative / pretrained adapter | `#8aa5a8` | filled h |
| Anchor-environment-only training | `#9467bd` | filled P |
| Gradient ascent (SFT) | `#8090a0` | filled s |
| Base model | `#444444` | hollow o |

## Sanctioned exceptions
- hf100 dev-selection stars: big black-edged stars (`s=1680`) — the figure's semantic
  device; class colors still apply.
- countdown-left scatter keeps data-driven (cropped) limits; grids/pareto panels use
  0–100% squares (`set_box_aspect(1)`).
- appendix v4 grid keeps its drawn "better" arrow.
