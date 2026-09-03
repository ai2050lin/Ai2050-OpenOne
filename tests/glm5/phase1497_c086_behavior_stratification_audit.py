#!/usr/bin/env python3
"""Independent audit for Phase1497."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; C=TESTS/"result/phase1496_c086_unlabeled_counterbalanced_contract"; OUT=TESTS/"result/phase1497_c086_behavior_stratification"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1497_c086_behavior_stratification as phase
def main():
 rows=core.rows(OUT/"raw/behavior.jsonl"); sets=core.rows(C/"material/composition_sets.jsonl"); saved=core.load(OUT/"analysis/behavior_stratification_summary.json"); protocol=core.load(C/"protocol/preregistration.json"); recomputed,strat=phase.summarize(rows,sets,protocol,saved["numeric_repeat_max_abs_diff"],saved["runtime"]["quantization"]); actual=core.rows(OUT/"material/stratified_composition_sets.jsonl"); py_compile.compile(str(TESTS/"phase1497_c086_behavior_stratification.py"),doraise=True)
 checks={"prediction":all(r["prediction"]==max(range(2),key=lambda i:r["scores"][i]) for r in rows),"correct":all(r["correct"]==(r["prediction"]==r["gold_position"]) for r in rows),"summary":all(saved[k]==recomputed[k] for k in ("global_accuracy","interface","stratum_counts","checks")),"strata":[(r["set_id"],r["correct_count"],r["stratum"]) for r in actual]==[(r["set_id"],r["correct_count"],r["stratum"]) for r in strat],"integrity":saved["all_integrity_checks_passed"]}
 result={"phase":1497,"campaign":"C086","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
 if not result["all_checks_passed"]:raise RuntimeError(checks)
 core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
