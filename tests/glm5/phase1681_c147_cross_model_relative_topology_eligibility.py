#!/usr/bin/env python3
"""C147: typed eligibility and missingness for cross-model relative topology."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1681_c147_cross_model_relative_topology_eligibility";C146=R/"phase1680_c146_cross_model_interface_sweep";C138=R/"phase1672_c138_prospective_cross_model_topology";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
PHASE,CAMPAIGN=1681,"C147";MODELS=("qwen3","glm4","deepseek7b")
def now():return datetime.now(timezone.utc).isoformat()
def main():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C146/"audit/independent_closure_audit.json");result=core.load(C146/"analysis/confirmation.json");freeze=core.load(C146/"protocol/frozen_interface.json")
 qualified=result["confirmation_qualified_models"]
 status={m:{"C146_discovery":{i:freeze["table"][i][m] for i in freeze["table"]},"C147_internal_status":"not-tested","reason":"no frozen common interface"} for m in MODELS}
 prior={}
 prior_path=C138/"analysis/cross_model_synthesis.json"
 if prior_path.exists():
  old=core.load(prior_path);prior={"qualified_models":old.get("qualified_models",[]),"missing_models":old.get("missing_models",{}),"claim_boundary":old.get("claim_boundary"),"source_sha256":core.sha(prior_path)}
 report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"cross_model_topology_not_tested_due_behavior_eligibility","common_interface":result["winner"],"confirmation_qualified_models":qualified,"model_ledger":status,"prior_single_model_context":prior,"measured":{"C146_behavior_interface":"measured-fail-cross-model"},"not_tested":["GLM4 HiddenState topology","DeepSeek7B HiddenState topology","cross-model relative-depth alignment","cross-model role topology","cross-model functional graph isomorphism"],"conclusion":"absence of a common behavior interface prevents a typed cross-model internal comparison; this is not evidence against shared internal structure","claim_boundary":"typed missingness and prior-context audit only; no new model or HiddenState run","next_authorization":"C148 campaign synthesis and heatmap; local causality remains unauthorized because C143 failed"}
 checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="start_C147","no_common":result["winner"] is None and len(qualified)==0,"models":set(status)==set(MODELS),"typed_missing":len(report["not_tested"])==5,"prior_bounded":"claim_boundary" in prior,"no_new_role_fields":not (C146/"raw/qwen3_role_field.bf16.npy").exists()}
 OUT.mkdir(parents=True);(OUT/"analysis").mkdir();(OUT/"audit").mkdir();core.save(OUT/"analysis/eligibility_and_missingness.json",report);core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_cross_model_topology_status":"not-tested","authorization":"independent_final_then_C148"});print(json.dumps({"checks":checks,"report":report},indent=2))
if __name__=="__main__":main()
