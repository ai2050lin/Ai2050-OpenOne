#!/usr/bin/env python3
"""Independent audit for Phase1684/C150."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1684_c150_predictable_transition_window_atlas";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
f=core.load(OUT/"protocol/frozen_window.json");r=core.load(OUT/"analysis/window_observation.json");p=core.load(PUBLIC);coord=core.load(OUT/"analysis/coordinate_rows.json")
checks={"discovery":core.load(OUT/"audit/internal_discovery_audit.json")["all_checks_passed"],"observation":core.load(OUT/"audit/internal_observation_audit.json")["all_checks_passed"],"closure":core.load(OUT/"audit/internal_closure_audit.json")["all_checks_passed"],"window":r["window"]==f["window"] and all(b==a+1 for a,b in zip(r["window"],r["window"][1:])),"rows":len(coord)==2*len(r["window"]) and all(len(x["values"])==2560 for x in coord),"asset":p["phase"]==1684 and "c149_c150_transition_window" in p,"hash":core.sha(PUBLIC)==core.load(OUT/"audit/internal_closure_audit.json")["asset_sha256"],"epistemic":"retrospective" in r["epistemic_status"]}
a={"phase":1684,"campaign":"C150","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_status":"retrospective-consistent" if r["window_observation_consistent"] else "retrospective-mixed","authorization":"memo_and_fresh_prospective_window_replication"};core.save(OUT/"audit/independent_closure_audit.json",a);print(json.dumps(a,indent=2))
