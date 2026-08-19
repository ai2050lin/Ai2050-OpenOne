#!/usr/bin/env python3
"""Independent audit of the exact Phase1300 engineering repair."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1300_c032_execution_compiler_competition";MAIN=T/"phase1300_c032_execution_compiler_competition.py";P=OUT/"protocol/preregistration.json";OLD=OUT/"protocol/invalid_attempt_main_source.py.txt";INVALID=OUT/"protocol/invalid_attempt_001.json";ERR=OUT/"protocol/execution_erratum.json";AUDIT=OUT/"audit/independent_repair_audit.json";OUTPUTS=[OUT/"raw/compiler_errors.npz",OUT/"raw/run_metadata.json",OUT/"analysis/compiler_summary.json",OUT/"protocol/frozen_runtime.json",OUT/"analysis/final.json",OUT/"protocol/formal_run_complete.json"]
BUG='     baseline[x["calibration_id"]]=np.stack([torch.stack([h[a,starts[a]+specs[a]["positions"][r]].float().cpu() for r in ROLES]).numpy() for h in out.hidden_states]);del out';FIX='     baseline[x["calibration_id"]]=np.stack([torch.stack([h[a,starts[a]+specs[a]["positions"][r]].float().cpu() for r in ROLES]).numpy() for h in out.hidden_states])\n    del out'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def add(c,n,p,d):c.append({"name":n,"passed":bool(p),"detail":d})
def main():
 p=load(P);e=load(ERR);inv=load(INVALID);old=OLD.read_text(encoding="utf-8");new=MAIN.read_text(encoding="utf-8");c=[];add(c,"original_hash",sha(OLD.read_bytes())==p["source_hashes"]["main"]==e["original_main_sha256"],e["original_main_sha256"]);add(c,"invalid_before_outputs",inv["scientific_arrays_written"] is False and inv["metrics_unblinded"] is False and inv["formal_scientific_run_consumed"] is False,inv);add(c,"exact_transform",old.count(BUG)==1 and new==old.replace(BUG,FIX),"one line moved");add(c,"new_hash",sha(MAIN.read_bytes())==e["repaired_main_sha256"],e["repaired_main_sha256"]);add(c,"constants",all(token in new for token in ('THRESHOLDS={"case_count_min":96','ARMS=("left_global_baseline","right_padding","record_event_aligned","equalized_suffix")','BATCH_GROUPS=4')),"frozen literals present");add(c,"outputs_absent",not any(x.exists() for x in OUTPUTS),"clear");ok=all(x["passed"] for x in c);doc={"phase":1300,"campaign":"C032","created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":ok,"authorization":"retry_phase1300_once_after_exact_engineering_repair" if ok else "none","protocol_digest":p["protocol_digest"]};AUDIT.parent.mkdir(parents=True,exist_ok=True);AUDIT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"passed":doc["passed_count"],"total":doc["total_count"],"authorization":doc["authorization"]}));
 if not ok:raise SystemExit(1)
if __name__=="__main__":main()
