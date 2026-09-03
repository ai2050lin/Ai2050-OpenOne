#!/usr/bin/env python3
"""Independent audit for Phase1566."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1566_c097_wordnet_capture"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 r=core.load(OUT/"analysis/capture_summary.json"); p=OUT/"raw/c097b_all_role_field.float32.npy"; a=np.load(p,mmap_mode="r"); f=core.load(OUT/"analysis/final.json"); checks={"hash":core.sha(p)==r["files"]["field"]["sha256"],"bytes":p.stat().st_size==r["files"]["field"]["bytes"],"shape":list(a.shape)==[540,37,4,2560],"float32":a.dtype==np.float32,"finite":bool(np.isfinite(a[::29]).all()),"numeric":all(r["checks"].values()),"authorization":f["authorization"]=="run_phase1567_c097_identifiable_common_residual_atlas"}; result={"phase":1566,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}; core.save(OUT/"audit/independent_final_audit.json",result)
 if not result["all_checks_passed"]: raise RuntimeError(result)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
