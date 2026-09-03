#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1734_c200_natural_deletion_rescue_adjudication"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); producer=Path(__file__).with_name("phase1734_c200_natural_deletion_rescue_adjudication.py"); checks={"closed":f["status"]=="closed_typed_not_tested" and f["all_checks_passed"],"not_tested":f["headline"]["natural_deletion_rescue_tested"] is False,"no_causal_claim":"No claim" in f["headline"]["inference"],"hash":core.sha(producer)==p["producer_sha256"]}; result={"phase":1734,"campaign":"C200","checks":checks,"all_checks_passed":all(checks.values()),"authorization":f["next_authorization"]}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
