#!/usr/bin/env python3
"""Independent audit for C166."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"tests/glm5/result/phase1700_c166_local_field_synthesis_heatmap";FRONT=ROOT/"frontend/public/vis_data/research_kernel/c157_c166_local_field_heatmap.json"
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=load(OUT/"protocol/preregistration.json");a=load(OUT/"analysis/heatmap.json");s=load(OUT/"analysis/synthesis.json");f=load(OUT/"analysis/final.json")
    rows=a["rows"]
    checks={"contract":load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"build":load(OUT/"audit/internal_build_audit.json")["all_checks_passed"],"final":f["all_checks_passed"],"frontend_equal":sha(FRONT)==sha(OUT/"analysis/heatmap.json")==s["asset_sha256"],"schema":a["schema"]=="c157_c166_local_field_heatmap.v1","all_coordinates":a["dimensions"]==list(range(2560)),"row_lengths":all(len(r["values"])==2560 for r in rows),"datasets":set(r["dataset"] for r in rows)>={"C159","C160","C161","C162"},"embedding_hidden":any(r.get("state_kind")=="embedding" for r in rows) and any(r.get("state_kind")=="hidden_state" for r in rows),"finite":all(np.isfinite(np.asarray(r["values"],np.float32)).all() for r in rows),"semantics":"not a model weight" in a["coordinate_semantics"],"forbidden":all(x in p["forbidden"] for x in ("attention","MLP","weights","PCA"))}
    audit={"phase":1700,"campaign":"C166","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"asset_sha256":s["asset_sha256"],"authorization":"memo_and_stage_close"}
    (OUT/"audit/independent_final_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");print(json.dumps(audit,indent=2))
    if not audit["all_checks_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
