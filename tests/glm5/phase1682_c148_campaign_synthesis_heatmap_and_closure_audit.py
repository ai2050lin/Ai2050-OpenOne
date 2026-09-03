#!/usr/bin/env python3
"""Independent audit for Phase1682/C148."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1682_c148_campaign_synthesis_heatmap_and_closure";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
summary=core.load(OUT/"analysis/campaign_synthesis.json");payload=core.load(PUBLIC);internal=core.load(OUT/"audit/internal_closure_audit.json");batch=payload["c140_c148_observation_batch"]
rows=batch["c141"]["representative_raw_rows"]+batch["c142"]["response_rows"]+batch["c145"]["error_rows"]
checks={"internal":internal["all_checks_passed"],"phase":payload["phase"]==1682 and payload["campaign"]=="C109-C148","batch":batch["campaign_summary"]["phase"]==1682,"rows":len(rows)==80,"full_coordinates":all(len(r["values"])==2560 for r in rows),"embedding_hidden":any(r["checkpoint_index"]==0 for r in rows) and any(r["checkpoint_index"]==36 for r in rows),"causal_not_tested":not summary["causal_intervention_run"] and summary["causal_status"]=="not-tested","asset_hash":core.sha(PUBLIC)==internal["asset_sha256"],"boundary":"never a model weight" in payload["claim_boundary"]}
a={"phase":1682,"campaign":"C148","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_causal_status":"not-tested","authorization":"memo_and_next_observation_campaign"};core.save(OUT/"audit/independent_closure_audit.json",a);print(json.dumps(a,indent=2))
