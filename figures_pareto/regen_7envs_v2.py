"""Regenerate the 7-env GR-vs-RP main figure as a *_regen.pdf for data
verification against the committed proto_pareto_7envs_gr_rp_v2.pdf.

Identical render path to proto_pareto_7envs_v2.py; only the output basename
differs (suffix _regen). Run from anywhere:
    .venv/bin/python figures_pareto/regen_7envs_v2.py
"""
import proto_pareto_style_v2 as style

_orig_save = style.save_figure


def _save_regen(fig, basename):
    if basename.endswith('.pdf'):
        basename = basename[:-4] + '_regen.pdf'
    return _orig_save(fig, basename)


style.save_figure = _save_regen

import proto_pareto_7envs_v2  # noqa: E402

proto_pareto_7envs_v2.save_figure = _save_regen
proto_pareto_7envs_v2.main()
