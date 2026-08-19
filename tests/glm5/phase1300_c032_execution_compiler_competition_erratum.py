#!/usr/bin/env python3
"""Register and authorize the one-line Phase1300 engineering repair."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";OUT=T/"result/phase1300_c032_execution_compiler_competition";MAIN=T/"phase1300_c032_execution_compiler_competition.py";P=OUT/"protocol/preregistration.json";OLD=OUT/"protocol/invalid_attempt_main_source.py.txt";INVALID=OUT/"protocol/invalid_attempt_001.json";ERR=OUT/"protocol/execution_erratum.json";AUDIT=OUT/"audit/independent_repair_audit.json";OUTPUTS=[OUT/"raw/compiler_errors.npz",OUT/"raw/run_metadata.json",OUT/"analysis/compiler_summary.json",OUT/"protocol/frozen_runtime.json",OUT/"analysis/final.json",OUT/"protocol/formal_run_complete.json"]
BUG='     baseline[x["calibration_id"]]=np.stack([torch.stack([h[a,starts[a]+specs[a]["positions"][r]].float().cpu() for r in ROLES]).numpy() for h in out.hidden_states]);del out'
FIX='     baseline[x["calibration_id"]]=np.stack([torch.stack([h[a,starts[a]+specs[a]["positions"][r]].float().cpu() for r in ROLES]).numpy() for h in out.hidden_states])\n    del out'
def sha_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def load(p:Path)->Any:return json.loads(p.read_text(encoding="utf-8"))
def save(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def prepare()->None:
 protocol=load(P);raw=MAIN.read_bytes();text=raw.decode("utf-8");
 if sha_bytes(raw)!=protocol["source_hashes"]["main"]:raise RuntimeError("main already changed")
 if text.count(BUG)!=1 or any(x.exists() for x in OUTPUTS):raise RuntimeError("repair preconditions failed")
 OLD.parent.mkdir(parents=True,exist_ok=True);OLD.write_bytes(raw);save(INVALID,{"phase":1300,"campaign":"C032","attempt":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),"classification":"invalid_engineering_attempt_before_any_scientific_output","exception":"UnboundLocalError: local variable out deleted inside first local-pair iteration","formal_scientific_run_consumed":False,"model_weights_loaded":True,"scientific_arrays_written":False,"metrics_unblinded":False,"original_main_sha256":sha_bytes(raw),"protocol_digest":protocol["protocol_digest"]});save(ERR,{"phase":1300,"campaign":"C032","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"repair_not_yet_applied","allowed_edit":"move del out exactly one indentation level outside local sample loop","scientific_constants_may_change":False,"original_main_sha256":sha_bytes(raw),"original_source_copy_sha256":sha_bytes(OLD.read_bytes()),"authorization":"apply_exact_repair_only"});print("repair prepared")
def authorize()->None:
 old=OLD.read_text(encoding="utf-8");new=MAIN.read_text(encoding="utf-8");expected=old.replace(BUG,FIX)
 if old.count(BUG)!=1 or new!=expected or any(x.exists() for x in OUTPUTS):raise RuntimeError("repair is not the exact authorized edit")
 e=load(ERR);e.update({"status":"exact_repair_applied_pending_independent_audit","repaired_main_sha256":sha_bytes(MAIN.read_bytes()),"scientific_constants_changed":False,"authorization":"independent_repair_audit_only"});save(ERR,e);print(json.dumps({"old":e["original_main_sha256"],"new":e["repaired_main_sha256"]}))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=("prepare","authorize"));a=ap.parse_args();prepare() if a.command=="prepare" else authorize()
