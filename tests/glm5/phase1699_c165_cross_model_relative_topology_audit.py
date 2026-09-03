#!/usr/bin/env python3
"""Independent audit for C165."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"tests/glm5/result/phase1699_c165_cross_model_relative_topology"
def load(p):return json.loads((OUT/p).read_text(encoding="utf-8"))

def main():
    p,s,f=load("protocol/preregistration.json"),load("analysis/summary.json"),load("analysis/final.json")
    models=p["eligible_models"]
    checks={
        "contract":load("audit/internal_contract_audit.json")["all_checks_passed"],
        "model_audits":all(load(f"audit/internal_{m}_audit.json")["all_checks_passed"] for m in models),
        "analysis":load("audit/internal_analysis_audit.json")["all_checks_passed"],
        "final":f["all_checks_passed"],
        "typed":(len(models)<2 and s["status"]=="typed_not_tested") or (len(models)>=2 and s["status"]=="cross_model_relative_topology_adjudicated"),
        "raw_shapes":all(list(np.load(OUT/f"raw/{m}_role_states.float16.npy",mmap_mode="r").shape)[:3]==[64,5,5] for m in models),
        "coordinate_free":p["topology"].startswith("off-diagonal"),
        "scope":"not shared physical coordinates" in s.get("claim_boundary",p["claim_boundary"]),
    }
    audit={"phase":1699,"campaign":"C165","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_topology_passed":s["topology_gate_passed"],"authorization":"C166"}
    (OUT/"audit/independent_final_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");print(json.dumps(audit,indent=2))
    if not audit["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
