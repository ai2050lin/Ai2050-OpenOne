#!/usr/bin/env python3
"""Independent audit for Phase1491."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1491_c085_all_case_field_capture"; B=TESTS/"result/phase1490_c085_behavior_stratification"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
def main():
    m=core.load(OUT/"analysis/capture_metadata.json"); idx=core.rows(OUT/"raw/all_role_field_index.jsonl"); arr=np.load(OUT/"raw/all_role_field.float16.npy",mmap_mode="r"); behavior={r["case_id"]:r for r in core.rows(B/"raw/behavior.jsonl")}; py_compile.compile(str(TESTS/"phase1491_c085_all_case_field_capture.py"),doraise=True)
    sample=np.asarray(arr[[0,1727,3455]],dtype=np.float32)
    checks={"shape":list(arr.shape)==m["shape"]==[3456,37,9,2560],"index":len(idx)==3456 and all(r["row_index"]==i for i,r in enumerate(idx)),"hashes":core.sha(OUT/"raw/all_role_field.float16.npy")==m["raw_sha256"] and core.sha(OUT/"raw/all_role_field_index.jsonl")==m["index_sha256"],"finite_sample":bool(np.isfinite(sample).all()),"behavior":all(r["capture_prediction"]==behavior[r["case_id"]]["prediction"] for r in idx),"runtime":all(m["checks"].values())}
    result={"phase":1491,"campaign":"C085","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
