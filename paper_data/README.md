# Paper figure inputs (snapshot 2026-10-13)

`paper_fig_inputs.tar.gz` (7.6 MB; 140 MB / 929 files uncompressed) holds the
run outputs that the gr-paper figure scripts on master read, so the figures can
be regenerated without the RunPod `/workspace` volume. Paths inside the tarball
are absolute minus the leading `/` (`workspace/small-rl/output/...`,
`workspace/respmon-results/...`), which is where the scripts expect them:

    tar -C / -xzf paper_data/paper_fig_inputs.tar.gz

`paper_fig_inputs_manifest.txt` lists every file.

## How the file set was chosen

Each figure script was run under a Python audit hook that logged every file it
opened. The set is everything those runs opened under `output/` or
`/workspace/respmon-results/`, plus `run_config.yaml` for every run whose
`routing_eval.jsonl` was read, plus the hf50 fseval JSONs and the
`expr_only_probe_arms` JSONs pulled from the `gr-modal-pilot` Modal volume.

| Paper figure | Script (figures_pareto/) | Regenerated from this snapshot? |
|---|---|---|
| proto_respmon6_composite_sidebyside | proto_respmon6_composite_sidebyside.py | yes |
| countdown_generalization_scatter | countdown_generalization_scatter.py | yes |
| anchor_form_pareto6 | anchor_form_pareto6.py | yes |
| sort_pathological_pareto | sort_pathological_pareto.py | yes |
| training_curves_all_envs | respmon6_training_curves.py | yes |
| training_curves_rec{50,25,10} | respmon6_training_curves_recall.py | yes |
| recall_sweep_pareto6 | respmon6_recall_scatter.py | yes |
| countdown_figure1 (Fig 2) | countdown_figure1.py | **no**, see below |
| countdown_hf100_hf50_retain | countdown_hf100_hf50_retain.py | **no**, see below |
| hf100_arms_scatter_v2 | hf100_arms_scatter_v2.py | **no**, see below |
| routing_scatter_main (32B SFT) | (big-sft repo, charts/plot_scatter.py) | out of scope |
| method_diagram | hand-drawn | n/a |

**Missing input:** the three countdown figures use the hf100 expr-only probe
JSONs (`output/expr_only_probe_arms/cdhf100_*.json`, `countdown_code_*.json`,
`lconly_full_*.json`) as their y-axis. Those were produced by
`tools/expr_only_probe.py` on another machine. They are not on this pod or on
the Modal volume (which holds only the `cdhf50_*` probes, included here). The
fseval JSONs these figures use for the x-axis are included.
