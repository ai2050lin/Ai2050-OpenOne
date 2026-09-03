#!/usr/bin/env python3
"""Independent pre-model audit for C104."""
from __future__ import annotations
import json, py_compile, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/'tests/glm5'; OUT=TESTS/'result/phase1589_c104_upstream_candidate_validation'; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
def main():
    producer=TESTS/'phase1589_c104_upstream_candidate_contract.py'; py_compile.compile(str(producer),doraise=True)
    protocol=core.load(OUT/'protocol/preregistration.json'); source=core.load(OUT/'audit/pre_model_material_audit.json'); compiled=core.rows(OUT/'compiled/qwen3.jsonl'); barcodes=np.load(ROOT/protocol['barcode_path'],mmap_mode='r')
    checks={'producer':core.sha(producer)==protocol['producer_sha256'],'source':source['all_checks_passed'] and source['passed']==source['total']==11,'compiled':len(compiled)==576,'barcodes':barcodes.shape==(4,2560) and core.sha(ROOT/protocol['barcode_path'])==protocol['barcode_sha256'],'predictions':[(r['family'],r['role'],r['state']) for r in protocol['predictions']]==[('attribute_binding','query_anchor',19),('agent_patient','query_anchor',19),('negation_scope','focus_record',3),('whole_part_exception','focus_post',23)],'authorization':protocol['authorization']=='run_phase1590_c104_qwen_capture'}
    result={'phase':1589,'campaign':'C104','checks':checks,'passed':sum(checks.values()),'total':len(checks),'all_checks_passed':all(checks.values())}
    if not result['all_checks_passed']: raise RuntimeError(result)
    core.save(OUT/'audit/independent_pre_model_audit.json',result); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
