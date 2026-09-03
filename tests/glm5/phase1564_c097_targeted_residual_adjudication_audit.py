#!/usr/bin/env python3
"""Independent audit for Phase1564."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1564_c097_targeted_residual_adjudication"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 s=core.load(OUT/"analysis/c097a_adjudication.json"); a=np.load(OUT/"raw/c097a_individual_interactions.float32.npy",mmap_mode="r"); f=core.load(OUT/"analysis/final.json"); checks={"shape":list(a.shape)==[14,37,4,2560],"finite":bool(np.isfinite(a).all()),"metrics":len(s["metrics"])==2,"bootstrap":len(core.rows(OUT/"analysis/bootstrap_cosines.jsonl"))==4000,"three_decisions":len(s["decisions"])==3,"hash":s["files"]["individual_sha256"]==core.sha(OUT/"raw/c097a_individual_interactions.float32.npy"),"authorization":f["authorization"]=="run_phase1565_c097_wordnet_independent_contract"}; result={"phase":1564,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
