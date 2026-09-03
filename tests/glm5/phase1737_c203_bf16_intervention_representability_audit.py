#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1737_c203_bf16_intervention_representability"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); r=core.load(OUT/"analysis/representability.json"); rows=core.rows(OUT/"raw/bf16_write_steps.jsonl"); producer=Path(__file__).with_name("phase1737_c203_bf16_intervention_representability.py")
    checks={"closed":f["status"]=="closed" and f["all_checks_passed"],"three_doses":len(r["dose_rows"])==3,"raw_breadth":len({x["anchor"] for x in rows})==14 and len({x["coordinate"] for x in rows})==64,"finite":bool(np.isfinite([[x[k] for k in ("source_value","intended","plus_step","minus_step","ulp_up","ulp_down")] for x in rows]).all()),"boundary":"cannot adjudicate downstream linearity" in r["interpretation"],"hash":core.sha(producer)==p["producer_sha256"]}
    result={"phase":1737,"campaign":"C203","checks":checks,"all_checks_passed":all(checks.values()),"authorization":f["next_authorization"]}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
