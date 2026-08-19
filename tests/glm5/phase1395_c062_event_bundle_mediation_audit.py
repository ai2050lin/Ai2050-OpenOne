#!/usr/bin/env python3
"""Independent audit for Phase1395."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1395_c062_event_bundle_mediation"
def main():
 s=core.load(OUT/"analysis/event_mediation_summary.json");f=core.load(OUT/"analysis/final.json");r=core.rows(OUT/"raw/event_bundle_mediation.jsonl")
 checks={"phase_checks":f["all_checks_passed"] and all(s["checks"].values()),"authorization":f["authorization"]=="run_phase1396_c062_campaign_closure",
         "cases":s["case_count"]==144,"records":len(r)==720,"bundles":set(x["bundle"] for x in r)=={"top1","stage_top1","stage_top2","query_reference","boundary_reference"},
         "holdout":set(x["partition"] for x in r)=={"confirmation","lockbox"},"qualified_exact":set(s["qualified_bundles"])=={k for k,v in s["metrics"].items() if v["qualified_all_holdouts"]}}
 result={"phase":1395,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
 if not result["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
