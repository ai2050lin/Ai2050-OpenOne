#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1653_c122_multi_interface_comparison_calibration"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
if __name__=="__main__":
    f=core.load(OUT/"protocol/frozen_interface_selection.json"); c=core.load(OUT/"analysis/closure.json"); i=core.load(OUT/"audit/internal_closure_audit.json")
    best=sorted(f["table"],key=lambda row:(-row["minimum_slice"],-row["overall"]))[0]
    checks={"internal":i["all_checks_passed"],"no_winner":f["winner"] is None,"best":c["headline"]["best_interface"]==best and best["interface"]=="true_false","all_failed":all(not row["eligible"] for row in f["table"]),"sealed":not (OUT/"analysis/holdout_validation.json").exists() and not (OUT/"raw/qwen3_role_subtoken_all_states.uint16.npy").exists(),"boundary":"no claim about embedding/HiddenState" in c["claim_boundary"],"authorization":c["next_authorization"].startswith("end_C120_C122")}
    r={"phase":1656,"campaign":"C122","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"producer_sha256":core.sha(Path(__file__)),"authorization":c["next_authorization"]}
    if not r["all_checks_passed"]:raise RuntimeError(r)
    core.save(OUT/"audit/independent_closure_audit.json",r);print(json.dumps(r,indent=2))
