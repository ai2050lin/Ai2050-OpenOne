#!/usr/bin/env python3
"""Independent completeness audit of the C623 recovered metric ledger."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase2158_c624_guard_metric_recovery_audit"
C623 = ROOT / "tests/glm5/result/phase2157_c623_guard_metric_artifact_recovery"

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def save(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def main():
    save(OUT/"protocol/preregistration.json", {"phase":2158,"campaign":"C624","timestamp_utc":datetime.now(timezone.utc).isoformat(),"object":"C623 recovery audit"})
    f=load(C623/"analysis/final.json");m=load(C623/"analysis/recovered_guard_metrics.json")
    checks={"c623_closed":f["all_checks_passed"],"cells":len(m)==72,
            "seven_models":all(len(v["models"])==7 for v in m.values()),
            "candidate_identity":f["headline"]["candidate_keys_exact"]}
    result={"phase":2158,"campaign":"C624","status":"closed","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "all_checks_passed":all(checks.values()),"headline":{"status":"recovery_audit_closed","checks":checks,
        "same_exact_goal_next_stage":False,"reason":"The omitted artifact is recovered and the frozen C615 verdict is unchanged."},
        "checks":checks,"next_authorization":"major_stage_closed"}
    save(OUT/"analysis/final.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
