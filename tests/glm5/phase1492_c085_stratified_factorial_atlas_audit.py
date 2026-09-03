#!/usr/bin/env python3
"""Independent audit for Phase1492."""
from __future__ import annotations
import json,py_compile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; OUT=RESULT/"phase1492_c085_stratified_factorial_atlas"; CAP=RESULT/"phase1491_c085_all_case_field_capture"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1492_c085_stratified_factorial_atlas as phase
def main():
    summary=core.load(OUT/"analysis/stratified_atlas_summary.json"); sf=np.load(OUT/"atlas/success_factorial_contrast_mean.float32.npy",mmap_mode="r"); sr=np.load(OUT/"atlas/stratum_relation_effect.float32.npy",mmap_mode="r"); counts=np.load(OUT/"atlas/stratum_sample_counts.int32.npy"); raw=np.load(CAP/"raw/all_role_field.float16.npy",mmap_mode="r"); idx=core.rows(CAP/"raw/all_role_field_index.jsonl")
    rows=[r for r in idx if r["stratum"]=="success" and r["record_relation_id"]=="join" and r["partition"]=="response_discovery" and r["surface"]=="a_colon"]; by={r["set_id"]:[] for r in rows}
    for r in rows: by[r["set_id"]].append(r)
    vals=[]
    for block in by.values():
        block=sorted(block,key=lambda r:r["cell"],reverse=True); weight=np.asarray([phase.signs(r)["relation"]*2/8 for r in block],dtype=np.float32); vals.append(np.tensordot(weight,np.asarray(raw[[r["row_index"] for r in block]],dtype=np.float32),axes=(0,0)))
    recomputed=np.mean(vals,axis=0); py_compile.compile(str(TESTS/"phase1492_c085_stratified_factorial_atlas.py"),doraise=True)
    checks={"hashes":core.sha(OUT/"atlas/success_factorial_contrast_mean.float32.npy")==summary["files"]["success_factorial"]["sha256"] and core.sha(OUT/"atlas/stratum_relation_effect.float32.npy")==summary["files"]["stratum_relation"]["sha256"],"recompute_panel":float(np.max(np.abs(recomputed-np.asarray(sf[0,0,0,0]))))<=1e-5,"identity":float(np.max(np.abs(np.asarray(sf[0])-np.asarray(sr[0]))))==0.0,"counts":counts.shape==(3,6,3,2) and int(counts[0].sum()/2)==175,"checks":all(summary["checks"].values())}
    result={"phase":1492,"campaign":"C085","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    if not result["all_checks_passed"]: raise RuntimeError(checks)
    core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
