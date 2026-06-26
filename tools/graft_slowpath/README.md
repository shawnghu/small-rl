# graft-port slow-path proof seeds

Standalone CPU proof/prototype scripts (from workflow `wouvmhyiw`, 2026-06-26) underpinning the
λ≠1 slow-path spec in `MASTER_PORT_PLAN.md` §12. They are the **test seeds** for slices 2a/2b —
turn them into proper `tests/` cases when building. Run with `.venv/bin/python tools/graft_slowpath/<x>.py`.

- **v_feasibility.py** — proves the 1-backward *rescale* trick (scale the captured grad by
  `a_v/a_m`) FAILS at β>0 (158% err) and off-policy → two honest backwards are mandatory.
- **v_isolation_proto.py** — proves the 2-backward `v=a_v` orchestration isolates m (masked grad at
  `a_m`) from v (natural grad at `a_v`) to fp64 (worst 1.2e-15), across on/off-policy × β. Uses the
  REAL `gradient_routing.DualMLPAdapter` / `PreRoutingGradAccumulator` / `set_fused_routing` +
  `advantages.routing_grad_mask_weights`. Variant A (2 forwards) vs Variant B (1 shared forward + 2
  backwards + the proposed `rearm()`).
- **adv_break.py** — adversarial re-verification on the hardest case (exclusive amplifying masks +
  heterogeneous per-token masks + multi-microbatch with `.grad` never zeroed). Isolation holds.
- **adv_failmodes.py** — demonstrates the two **silent** failure modes (omit `rearm()` → v rides
  `g_m`; omit `.grad` snapshot/restore → m = `masked(a_m)+masked(a_v)`). Motivates the three
  mandatory loud asserts in slice 2a.
- **step_bound.py** — λ>1 numeric check: the single-token `w·a_m/a_v` heuristic does NOT bound the
  realized per-coordinate Adam step (maxγ=28.6 vs realized max=30.6 at λ=1.5) → 2b needs the
  realized-step gate.
