#!/usr/bin/env python3
"""Independent audit for Phase1562."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1562_c097_targeted_residual_contract"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 p=core.load(OUT/"protocol/preregistration.json"); a=core.load(OUT/"audit/pre_model_audit.json"); pairs=core.rows(OUT/"material/frozen_pairs.jsonl"); cases=core.rows(OUT/"material/active_cases.jsonl"); f=core.load(OUT/"analysis/final.json")
 checks={"preaudit":a["all_checks_passed"],"pairs":len(pairs)==28,"cases":len(cases)==56,"families":{r["family"] for r in pairs}=={"similarity","class_inclusion"},"postquery":{r["surface"] for r in cases}=={"postquery"},"hash":p["material_hash"]==core.sha(OUT/"material/frozen_pairs.jsonl"),"authorization":f["authorization"]=="run_phase1563_c097_targeted_residual_capture"}
 result={"phase":1562,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
