#!/usr/bin/env python3
"""Independent audit for C143."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";O=T/"result/phase1677_c143_transition_model_competition";sys.path.insert(0,str(T));import phase1331_relational_measurement_core as core
def main():
 f=core.load(O/"protocol/frozen_model.json");r=core.load(O/"analysis/confirmation.json");d=np.load(O/"analysis/discovery_primary_trajectories.float32.npy",mmap_mode="r");c=np.load(O/"analysis/confirmation_primary_trajectories.float32.npy",mmap_mode="r");checks={"internal":core.load(O/"audit/internal_closure_audit.json")["all_checks_passed"],"shapes":list(d.shape)==list(c.shape)==[80,38,6,2560],"models":set(f["models"])==set(r["model_summary"]),"freeze":f["frozen_winner"]==r["frozen_winner"],"rows":all(len(v)==37 for v in r["model_rows"].values()),"controls":len(r["candidate_controls"]["wrong_checkpoint"])==36,"finite":all(np.isfinite(v) for v in r["derived"].values()),"boundary":"no unique causal edge" in r["claim_boundary"]};report={"phase":1677,"campaign":"C143","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_prediction_gate_passed":r["prediction_gate_passed"],"authorization":"start_C144"};core.save(O/"audit/independent_closure_audit.json",report);print(json.dumps(report,indent=2));raise SystemExit(0 if report["all_checks_passed"] else 1)
if __name__=="__main__":main()
