#!/usr/bin/env python3
"""Independent audit for Phase1681/C147."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1681_c147_cross_model_relative_topology_eligibility";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
r=core.load(OUT/"analysis/eligibility_and_missingness.json");internal=core.load(OUT/"audit/internal_closure_audit.json")
checks={"internal":internal["all_checks_passed"],"status":r["status"].endswith("eligibility"),"no_models":r["confirmation_qualified_models"]==[],"three_ledgers":len(r["model_ledger"])==3,"five_not_tested":len(r["not_tested"])==5,"boundary":"no new model" in r["claim_boundary"],"next":"C148" in r["next_authorization"]}
a={"phase":1681,"campaign":"C147","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_cross_model_topology_status":"not-tested","authorization":"start_C148"};core.save(OUT/"audit/independent_closure_audit.json",a);print(json.dumps(a,indent=2))
