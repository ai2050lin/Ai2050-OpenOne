#!/usr/bin/env python3
"""Independent audit for Phase1680/C146."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1680_c146_cross_model_interface_sweep";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
models=("qwen3","glm4","deepseek7b");freeze=core.load(OUT/"protocol/frozen_interface.json");report=core.load(OUT/"analysis/confirmation.json")
checks={
 "contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],
 "captures":all(core.load(OUT/f"audit/{m}_capture_audit.json")["all_checks_passed"] for m in models),
 "selection":core.load(OUT/"audit/internal_selection_audit.json")["all_checks_passed"],
 "confirmation":core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"],
 "closure":core.load(OUT/"audit/internal_closure_audit.json")["all_checks_passed"],
 "rows":all(len(core.rows(OUT/f"raw/{m}_behavior_index.jsonl"))==1024 for m in models),
 "freeze":freeze["winner"]==report["winner"],
 "typed_models":all(m in models for m in report["confirmation_qualified_models"]),
}
audit={"phase":1680,"campaign":"C146","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_common_interface_gate_passed":report["common_interface_gate_passed"],"authorization":"start_C147"}
core.save(OUT/"audit/independent_closure_audit.json",audit);print(json.dumps(audit,indent=2))
