#!/usr/bin/env python3
"""Independent audit for Phase1570."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1570_c097_visualization_client_integration"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 r=core.load(OUT/"analysis/client_integration.json"); f=core.load(OUT/"analysis/final.json"); checks={"producer":all(r["checks"].values()),"commands":set(r["external_commands"].values())=={"passed"},"file_count":len(r["files"])==7,"hashes":all(len(v["sha256"])==64 for v in r["files"].values()),"authorization":f["authorization"]=="freeze_C098_observation_first_graph_contract"}; result={"phase":1570,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
