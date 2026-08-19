#!/usr/bin/env python3
"""Independent audit for Phase1393."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1393_c062_discovery_full_field"
def main():
 s=core.load(OUT/"analysis/full_field_summary.json");f=core.load(OUT/"analysis/final.json");r=core.load(OUT/"protocol/discovery_rankings.json");c=core.load(OUT/"protocol/discovery_event_candidates.json")
 payload=torch.load(OUT/"raw/discovery_family3_differences.pt",map_location="cpu",weights_only=False)
 checks={"phase_checks":f["all_checks_passed"] and all(s["checks"].values()),"authorization":f["authorization"]=="run_phase1394_c062_coordinate_curves_and_phase1395_event_mediation",
         "cases":s["case_count"]==72,"source_shape":list(payload["vectors"].shape)==[72,2560],
         "families":set(r["families"])=={"animal","building","profession","country"},
         "hash_rank":core.sha(OUT/"protocol/discovery_rankings.json")==s["rankings_sha256"],
         "hash_candidate":core.sha(OUT/"protocol/discovery_event_candidates.json")==s["candidates_sha256"],
         "candidate_scope":c["selection_scope"]=="response_discovery only","bundles":all(set(v)=={"top1","stage_top1","stage_top2","query_reference","boundary_reference"} for v in c["bundles"].values())}
 result={"phase":1393,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
 if not result["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
