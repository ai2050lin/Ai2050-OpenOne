#!/usr/bin/env python3
"""Independent audit of C361-C368."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/"tests/glm5/result"
PRODUCER=ROOT/"tests/glm5/phase1895_c361_c368_full_coordinate_kernel_campaign.py"


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    digest=hashlib.sha256(PRODUCER.read_bytes()).hexdigest();dirs={};finals={};protocols={}
    for c in range(361,369):
        p=1895+c-361;matches=list(RESULT.glob(f"phase{p}_c{c}_*"));assert len(matches)==1,(c,matches);dirs[c]=matches[0];finals[c]=load(matches[0]/"analysis/final.json");protocols[c]=load(matches[0]/"protocol/preregistration.json")
    checks={
        "eight_continuous_phases":[finals[c]["phase"] for c in finals]==list(range(1895,1903)),
        "all_frozen_hashes":all(protocols[c]["producer_sha256"]==digest for c in protocols),
        "all_internal_closed":all(finals[c]["all_checks_passed"] for c in finals),
        "linear_prediction_shape":list(np.load(dirs[362]/"raw/confirmation_predictions.float16.npy",mmap_mode="r").shape)==[24,3,38,6,2560],
        "quadratic_prediction_shape":list(np.load(dirs[363]/"raw/confirmation_predictions.float16.npy",mmap_mode="r").shape)==[24,3,38,6,2560],
        "apple_i_failed_controls":finals[364]["headline"]["i_control_gate_passed"] is False,
        "six_family_breadth_separately_reported":finals[365]["headline"]["breadth_gate_passed"] is True,
        "causal_not_run":finals[366]["headline"]["causal_eligible"] is False and finals[366]["headline"]["model_intervention_run"] is False,
        "new_math_closed":finals[368]["headline"]["new_math_gate_passed"] is False,
    }
    payload={"status":"independent_audit_complete","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"strict_interpretation":"C365 breadth is material-specific method breadth. Because the independent apple I control gate failed, it cannot authorize a universal operator or causal patch."}
    (dirs[368]/"audit/independent_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(payload,ensure_ascii=False,indent=2));raise SystemExit(0 if payload["all_checks_passed"] else 1)


if __name__=="__main__":main()
