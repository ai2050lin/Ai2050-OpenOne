#!/usr/bin/env python3
"""Independent audit for Phase1568."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1568_c097_major_stage_closure"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 s=core.load(OUT/"analysis/c097_major_stage_closure.json"); f=core.load(OUT/"analysis/final.json"); checks={"sources":all(s["checks"].values()),"K270":s["puzzle_update"]["id"]=="K270","formula_identifiable":"G_C=" in s["corrected_theory"]["mechanism_formula"],"energy_identity":"3||G_C||" in s["corrected_theory"]["energy_formula"],"important":s["visualization"]["important"],"scope":len(s["major_answer"]["not_supported"])>=5,"authorization":f["authorization"]=="run_phase1569_c097_relation_contrast_heatmap_export"}; result={"phase":1568,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

