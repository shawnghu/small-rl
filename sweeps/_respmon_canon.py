"""Canonical respmon env list (2026-07-25 definition: object_qa uses the
INVERTED monitor; addition/cities standard iva; persona bvo; repeat/sorting
respmon). Shared by the canonical-family sweeps so the env set is defined in
exactly one place."""

CANON_ENVS = [
    {"config": "configs/addition_v2_syco_respmon_iva.yaml",     "max_steps": 2000},
    {"config": "configs/object_qa_syco_respmon_iva_inv.yaml",   "max_steps": 1000},
    {"config": "configs/cities_qa_syco_respmon_iva.yaml",       "max_steps": 2000},
    {"config": "configs/persona_qa_flattery_respmon_bvo.yaml",  "max_steps": 1000},
    {"config": "configs/repeat_respmon.yaml",                   "max_steps": 500},
    {"config": "configs/sorting_copy_respmon.yaml",             "max_steps": 1000},
]
