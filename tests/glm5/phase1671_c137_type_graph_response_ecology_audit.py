#!/usr/bin/env python3
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";O=T/"result/phase1671_c137_type_graph_response_ecology";sys.path.insert(0,str(T));import phase1331_relational_measurement_core as core
def contract():
 p=core.load(O/"protocol/preregistration.json");checks={"internal":core.load(O/"audit/internal_contract_audit.json")["all_checks_passed"],"artificial":len(core.rows(O/"compiled/qwen3_artificial.jsonl"))==672,"natural":len(core.rows(O/"compiled/qwen3_natural.jsonl"))==64,"hashes":all(core.sha(Path(v))==p["source_hashes"][k] for k,v in p["source_paths"].items()),"boundary":"not a lexical apple" in p["claim_boundary"]};r={"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":"run_c137_behavior"};core.save(O/"audit/independent_contract_audit.json",r);print(json.dumps(r,indent=2))
def final():
 b=core.load(O/"analysis/behavior.json");checks={"contract":core.load(O/"audit/independent_contract_audit.json")["all_checks_passed"],"behavior":core.load(O/"audit/internal_behavior_audit.json")["all_checks_passed"],"closure":core.load(O/"audit/internal_closure_audit.json")["all_checks_passed"]}
 if b["gate_passed"]:raw=np.load(O/"raw/qwen3_type_ecology.bf16.npy",mmap_mode="r");checks.update({"raw":list(raw.shape)==[736,5,38,2560],"capture":core.load(O/"audit/internal_capture_audit.json")["all_checks_passed"],"freeze":core.load(O/"protocol/frozen_ecology.json")["confirmation_and_natural_unread"],"confirmation":core.load(O/"audit/internal_confirmation_audit.json")["all_checks_passed"]})
 else:checks["no_hidden"]=not(O/"raw/qwen3_type_ecology.bf16.npy").exists()
 r={"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_gate_passed":core.load(O/"audit/internal_closure_audit.json")["scientific_gate_passed"],"authorization":"start_route_E_C138"};core.save(O/"audit/independent_closure_audit.json",r);print(json.dumps(r,indent=2))
if __name__=="__main__":{"contract":contract,"final":final}[sys.argv[1]]()
