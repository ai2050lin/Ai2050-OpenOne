#!/usr/bin/env python3
"""Independent audit for Phase1980-1987 / C446-C453."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RESULT=ROOT/"tests/glm5/result";PRODUCER=ROOT/"tests/glm5/phase1980_c446_c453_fixed_codebook_replication.py"
PHASES={f"C{c}":(1980+c-446,s) for c,s in ((446,"fixed_codebook_confound_adjudication"),(447,"fresh_fixed_codebook_material"),(448,"qwen_fixed_codebook_behavior"),(449,"qualified_fixed_codebook_full_field"),(450,"fixed_codebook_guarded_rule_replication"),(451,"candidate_pattern_adjudication_and_writer_freeze"),(452,"conditional_hiddenstate_writer"),(453,"replication_synthesis_visual_cleanup_audit"))}
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
    digest=hashlib.sha256(PRODUCER.read_bytes()).hexdigest();checks={}
    for name,(phase,slug) in PHASES.items():
        out=RESULT/f"phase{phase}_{name.lower()}_{slug}";pre=load(out/"protocol/preregistration.json");fin=load(out/"analysis/final.json");checks[f"{name}_closed"]=fin["all_checks_passed"] and pre["producer_sha256"]==digest;checks[f"{name}_phase"]=pre["phase"]==phase==fin["phase"]
    rows=[json.loads(x) for x in (RESULT/"phase1981_c447_fresh_fixed_codebook_material/material/cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()];checks["material_rows"]=len(rows)==1344;checks["fixed_codebook"]=all("(A) Yes (B) No" in r["prompt"] for r in rows)
    vis=load(ROOT/"frontend/public/vis_data/research_kernel/c453_fixed_codebook_replication.json");checks["visual_full_coordinates"]=bool(vis["rows"]) and all(len(r["values"])==2560 for r in vis["rows"])
    cleanup=load(RESULT/"phase1987_c453_replication_synthesis_visual_cleanup_audit/audit/cleanup.json");checks["cleanup"]=all(r["removed"] and len(r["sha256"])==64 for r in cleanup)
    report={"phase":1987,"campaign":"C446-C453","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())};dest=RESULT/"phase1987_c453_replication_synthesis_visual_cleanup_audit/audit/independent_audit.json";dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False));assert report["all_checks_passed"]
if __name__=="__main__":main()
