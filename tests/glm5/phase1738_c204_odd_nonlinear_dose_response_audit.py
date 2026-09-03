#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1738_c204_odd_nonlinear_dose_response"; sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
def main():
 p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json"); r=core.load(OUT/"analysis/holdout.json"); producer=Path(__file__).with_name("phase1738_c204_odd_nonlinear_dose_response.py"); checks={"closed":f["status"]=="closed" and f["all_checks_passed"],"fit_holdout":r["fit_doses"]==[0.25,0.5] and r["holdout_dose"]==1.0,"models":len(r["holdout_metrics"])==4,"finite":bool(np.isfinite([[x["nrmse"],x["weighted_sign_agreement"]] for x in r["holdout_metrics"].values()]).all()),"boundary":"without implying global linearity" in r["interpretation"],"hash":core.sha(producer)==p["producer_sha256"]}; result={"phase":1738,"campaign":"C204","checks":checks,"all_checks_passed":all(checks.values()),"authorization":f["next_authorization"]}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
