#!/usr/bin/env python3
"""Independent audit for Phase1685/C151."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1685_c151_fresh_transition_window_replication";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
r=core.load(OUT/"analysis/confirmation.json");raw=np.load(OUT/"raw/qwen3_window_role_field.bf16.npy",mmap_mode="r");traj=np.load(OUT/"analysis/fresh_f1_window_trajectories.float32.npy",mmap_mode="r");p=core.load(PUBLIC)
checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"capture":core.load(OUT/"audit/internal_capture_audit.json")["all_checks_passed"],"analysis":core.load(OUT/"audit/internal_analysis_audit.json")["all_checks_passed"],"closure":core.load(OUT/"audit/internal_closure_audit.json")["all_checks_passed"],"raw_shape":list(raw.shape)==[640,6,11,2560],"trajectory_shape":list(traj.shape)==[80,11,6,2560],"rows":len(r["transition_rows"])==10,"asset":p["phase"]==1685 and "c151_fresh_transition_window" in p,"hash":core.sha(PUBLIC)==core.load(OUT/"audit/internal_closure_audit.json")["asset_sha256"]}
a={"phase":1685,"campaign":"C151","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_gate_passed":r["prospective_window_gate_passed"],"authorization":"memo_and_next_stage_assessment"};core.save(OUT/"audit/independent_closure_audit.json",a);print(json.dumps(a,indent=2))
