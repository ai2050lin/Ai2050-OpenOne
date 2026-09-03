#!/usr/bin/env python3
"""Independent audit for Phase1569."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1569_c097_relation_contrast_heatmap_export"; CLIENT=ROOT/"frontend/public/vis_data/research_kernel/c097_relation_contrast_heatmap.json"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 r=core.load(OUT/"analysis/visualization_export.json"); a=core.load(OUT/"visualization/c097_relation_contrast_heatmap.json"); f=core.load(OUT/"analysis/final.json"); checks={"producer":all(r["checks"].values()),"schema":a["result_type"]=="relation_contrast_heatmap","dimensions":len(a["dimensions"])==64,"rows":len(a["common_rows"])==222,"hash":r["asset"]["sha256"]==core.sha(CLIENT),"authorization":f["authorization"]=="freeze_C098_observation_first_graph_contract"}; result={"phase":1569,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
