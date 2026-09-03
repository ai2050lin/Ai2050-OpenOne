#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1733_c199_unseen_composition_prediction"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
    protocol=core.load(OUT/"protocol/preregistration.json"); final=core.load(OUT/"analysis/final.json"); report=core.load(OUT/"analysis/composition_prediction.json"); producer=Path(__file__).with_name("phase1733_c199_unseen_composition_prediction.py"); checks={"closed":final["status"]=="closed" and final["all_checks_passed"],"six_models":len(report["ranking"])==6 and set(report["ranking"])==set(protocol["models"]),"four_composites":all(len(v["by_program"])==4 for v in report["predictions"].values()),"boundary":report["semantic_composition_tested"] is False,"finite":bool(np.isfinite([v["aggregate"]["nrmse"] for v in report["predictions"].values()]).all()),"hash":core.sha(producer)==protocol["producer_sha256"]}; result={"phase":1733,"campaign":"C199","checks":checks,"all_checks_passed":all(checks.values()),"authorization":final["next_authorization"]}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
