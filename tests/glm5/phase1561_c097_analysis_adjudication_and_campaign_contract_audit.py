#!/usr/bin/env python3
"""Independent audit for Phase1561."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1561_c097_analysis_adjudication_and_campaign_contract"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
    p=core.load(OUT/"protocol/c097_campaign_contract.json"); a=core.load(OUT/"audit/preimplementation_audit.json"); f=core.load(OUT/"analysis/final.json")
    checks={"preaudit":a["all_checks_passed"],"digest":p["contract_sha256"]==core.digest({k:v for k,v in p.items() if k!="contract_sha256"}),"identifiable_G":"G_C=" in p["identifiable_objects"]["contrast_mean"],"zero_sum":"sum_fg R_fg=0" in p["identifiable_objects"]["contrast_residual"],"three_routes":set(p["routes"])=={"A","B","C"},"authorization":f["authorization"]=="run_phase1562_c097_targeted_residual_contract"}
    result={"phase":1561,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
    if not result["all_checks_passed"]: raise RuntimeError(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

