#!/usr/bin/env python3
"""Independent audit for Phase1401."""
from __future__ import annotations
import json,math,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
C=TESTS/"result/phase1400_c064_fixed_answer_factorial_contract";OUT=TESTS/"result/phase1401_c064_behavior"
def main():
 p=core.load(C/"protocol/preregistration.json");s=core.load(OUT/"analysis/qwen3_behavior_summary.json");f=core.load(OUT/"analysis/final.json");a=core.rows(OUT/"raw/active_behavior.jsonl");st=core.rows(OUT/"raw/status_behavior.jsonl");sel=core.rows(OUT/"material/eligible_factor_sets.jsonl");q=s["qualified_families"]
 expected=Counter({x:len(q)*p["material"]["selected_per_family_partition"] for x in p["material"]["partitions"] if q})
 checks={"counts":len(a)==864 and len(st)==288,"truth_balance":Counter(x["truth"] for x in a)=={True:432,False:432},"finite":all(math.isfinite(z) for x in a+st for z in x["scores"]),"decisions":all(v["qualified"]==all(v["checks"].values()) for v in s["family_results"].values()),"breadth":s["breadth_checks"]["family_count"]==(len(q)>=p["material"]["minimum_qualified_families"]),"selected":len(sel)==len(q)*p["material"]["selected_per_family"] and Counter(x["partition"] for x in sel)==expected,"numeric":s["breadth_checks"]["numeric"],"authorization":f["authorization"]==("run_phase1402_c064_state_swap_camera" if s["behavior_qualified"] else "close_c064_at_behavior_gate")}
 r={"phase":1401,"campaign":"C064","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};core.save(OUT/"audit/independent_final_audit.json",r)
 if not r["all_checks_passed"]:raise RuntimeError({k:v for k,v in checks.items() if not v})
 print(json.dumps(r,indent=2))
if __name__=="__main__":main()
