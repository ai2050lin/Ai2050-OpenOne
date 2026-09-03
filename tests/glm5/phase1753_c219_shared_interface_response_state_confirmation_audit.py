#!/usr/bin/env python3
"""Independent audit for C219."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import phase1739_c205_response_ecology_common as common
core=common.core; OUT=common.RESULT / "phase1753_c219_shared_interface_response_state_confirmation"
def main():
    protocol=core.load(OUT / "protocol/preregistration.json"); final=core.load(OUT / "analysis/final.json"); report=final["headline"]; fields=np.load(OUT / "raw/full_fields.float16.npy",mmap_mode="r")
    checks={"final":final["all_checks_passed"],"shared_interface":"identical" in protocol["shared_interface"],"new_lexicon":protocol["new_lexical_units"]==8,"two_partitions":set(report["frozen_template_classification"])=={"confirmation","fresh"},"full_field":list(fields.shape)==[160,4,96,2560],"frozen_template":"template refitting" in protocol["forbidden"],"producer_hash":core.sha(Path(__file__).with_name("phase1753_c219_shared_interface_response_state_confirmation.py"))==protocol["producer_sha256"]}; audit={"phase":1753,"campaign":"C219","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":final["next_authorization"]}; core.save(OUT / "audit/independent_final_audit.json",audit); print(json.dumps(audit,indent=2))
if __name__=="__main__": main()
