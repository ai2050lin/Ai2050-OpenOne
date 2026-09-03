#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1735_c201_cross_model_functional_topology"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); r=core.load(OUT/"analysis/cross_model_topology.json"); producer=Path(__file__).with_name("phase1735_c201_cross_model_functional_topology.py"); checks={"closed":f["status"]=="closed" and f["all_checks_passed"],"models":set(r["behavior"]["selected_interface"])=={"qwen3","glm4","deepseek7b"},"interfaces":set(r["behavior"]["selected_interface"].values())<=set(p["interfaces"]),"pair_values":all(0<=x["similarity"]<=1 for x in r["pair_similarity"]),"no_coordinate_identity":"no coordinate identity" in r["claim_boundary"],"hash":core.sha(producer)==p["producer_sha256"]}; result={"phase":1735,"campaign":"C201","checks":checks,"all_checks_passed":all(checks.values()),"authorization":f["next_authorization"]}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
