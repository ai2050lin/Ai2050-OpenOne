#!/usr/bin/env python3
"""Independent audit for Phase1683/C149."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1683_c149_arm_specific_transition_system_identification";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
f=core.load(OUT/"protocol/frozen_models.json");r=core.load(OUT/"analysis/confirmation.json")
checks={"discovery":core.load(OUT/"audit/internal_discovery_audit.json")["all_checks_passed"],"confirmation":core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"],"closure":core.load(OUT/"audit/internal_closure_audit.json")["all_checks_passed"],"five_arms":len(r["arm_results"])==5,"freeze":all(r["arm_results"][a]["winner"]==f["arm_specific"][a]["winner"] for a in r["arm_results"]),"rows":all(len(v["transition_rows"])==37 for v in r["arm_results"].values()),"finite":all(np.isfinite(v["derived"]["median_relative_error"]) for v in r["arm_results"].values()),"boundary":"not a semantic operator" in r["claim_boundary"]}
a={"phase":1683,"campaign":"C149","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_passing_arms":r["passing_arms"],"authorization":"memo_and_next_stage_assessment"};core.save(OUT/"audit/independent_closure_audit.json",a);print(json.dumps(a,indent=2))
