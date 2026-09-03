#!/usr/bin/env python3
"""Independent audit for C198 broad natural-program trajectories."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1732_c198_broad_natural_program_trajectory"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol=core.load(OUT/"protocol/preregistration.json"); final=core.load(OUT/"analysis/final.json"); report=core.load(OUT/"analysis/natural_trajectory.json"); producer=Path(__file__).with_name("phase1732_c198_broad_natural_program_trajectory.py"); raw=np.load(OUT/"raw/natural_signed_trajectory.float16.npy",mmap_mode="r"); baseline=np.load(OUT/"raw/natural_baseline_states.float16.npy",mmap_mode="r")
    checks={"closed":final["status"]=="closed" and final["all_checks_passed"],"material":len(core.rows(OUT/"material/cases.jsonl"))==288 and len(core.rows(OUT/"material/registered_invalid_location.jsonl"))==32,"shape":list(raw.shape)==[72,32,2,6,2560] and list(baseline.shape)==[72,4,6,2560],"finite_sample":bool(np.isfinite(np.asarray(raw[:,:,:, :,::263],dtype=np.float32)).all()),"nine_programs":len(report["by_program"])==9,"hash":core.sha(producer)==protocol["producer_sha256"]}
    result={"phase":1732,"campaign":"C198","checks":checks,"all_checks_passed":all(checks.values()),"authorization":final["next_authorization"]}; core.save(OUT/"audit/independent_final_audit.json",result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
