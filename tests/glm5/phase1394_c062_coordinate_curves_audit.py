#!/usr/bin/env python3
"""Independent audit for Phase1394."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1394_c062_coordinate_curves"
def main():
 s=core.load(OUT/"analysis/coordinate_summary.json");f=core.load(OUT/"analysis/final.json");r=core.rows(OUT/"raw/coordinate_curves.jsonl")
 checks={"phase_checks":f["all_checks_passed"] and all(s["checks"].values()),"authorization":f["authorization"]=="run_phase1395_c062_event_bundle_mediation_then_close",
         "cases":s["case_count"]==144,"records":len(r)==s["record_count"],"holdout_only":set(x["partition"] for x in r)=={"confirmation","lockbox"},
         "primary_present":all(x in s["metrics"] for x in ("c060_family_fixed@512","c062_family_discovery@512")),
         "sizes":set(x["size"] for x in r)=={64,128,256,320,384,448,512,640,768,1024,1536,2048,2560},
         "route_scope":all(x["family"] in {"animal","building"} for x in r if x["rule"]=="c060_family_fixed")}
 result={"phase":1394,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
 if not result["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
