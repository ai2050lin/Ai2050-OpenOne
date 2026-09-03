#!/usr/bin/env python3
"""Independent audit for Phase1563."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1563_c097_targeted_residual_capture"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 r=core.load(OUT/"analysis/capture_summary.json"); p=OUT/"raw/c097a_all_role_field.float32.npy"; a=np.load(p,mmap_mode="r"); f=core.load(OUT/"analysis/final.json"); checks={"hash":core.sha(p)==r["files"]["field"]["sha256"],"shape":list(a.shape)==[56,37,4,2560],"float32":a.dtype==np.float32,"finite":bool(np.isfinite(a[::3]).all()),"numeric":all(r["checks"].values()),"authorization":f["authorization"]=="run_phase1564_c097_targeted_residual_adjudication"}; result={"phase":1563,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

