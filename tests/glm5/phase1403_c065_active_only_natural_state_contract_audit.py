#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
OUT=TESTS/"result/phase1403_c065_active_only_natural_state_contract"
def main():
 p=core.load(OUT/"protocol/preregistration.json");f=core.rows(OUT/"material/eligible_factor_sets.jsonl");pre=core.load(OUT/"audit/pre_hidden_freeze.json");checks={"pre":pre["all_checks_passed"],"hash":p["material"]["factor_sha256"]==core.sha(OUT/"material/eligible_factor_sets.jsonl"),"count":len(f)==54,"partition":Counter(r["partition"] for r in f)=={q:18 for q in p["factorial_swap"]["partitions"]+["response_discovery"]},"families":Counter(r["family"] for r in f)=={q:18 for q in p["material"]["qualified_families"]},"scope":all(x in p["forbidden"] for x in ("attention","MLP","gradient","PCA","learned probe")),"authorization":p["authorization"]=="run_phase1404_c065_state_swap_camera"}
 r={"phase":1403,"campaign":"C065","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",r)
 if not r["all_checks_passed"]:raise RuntimeError(checks)
 print(json.dumps(r,indent=2))
if __name__=="__main__":main()
