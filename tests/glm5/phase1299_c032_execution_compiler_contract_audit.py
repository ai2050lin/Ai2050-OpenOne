#!/usr/bin/env python3
"""Independent contract audit for Phase1299 C032."""

from __future__ import annotations
import hashlib, json, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]; T = ROOT / "tests/glm5"
OUT = T / "result/phase1299_c032_execution_compiler_contract"
P = OUT / "protocol/preregistration.json"; M = OUT / "material/frozen_inverse_lookup_cases.jsonl"; N = OUT / "material/pre_model_semantic_naturalness_review.json"; R = OUT / "audit/semantic_program_reaudit.json"; A = OUT / "audit/independent_final_audit.json"
MAIN = T / "phase1299_c032_execution_compiler_contract.py"; SCRIPT = Path(__file__).resolve()

def canonical(v: Any) -> str: return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
def digest(v: Any) -> str: return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        while c:=f.read(1024*1024): h.update(c)
    return h.hexdigest()
def load(p: Path) -> Any: return json.loads(p.read_text(encoding="utf-8"))
def rows() -> list[dict[str,Any]]: return [json.loads(x) for x in M.read_text(encoding="utf-8").splitlines() if x.strip()]
def add(c,n,p,d): c.append({"name":n,"passed":bool(p),"detail":d})

def main() -> None:
    p=load(P); rr=rows(); c=[]; timeless={k:v for k,v in p.items() if k not in {"created_at_utc","protocol_digest"}}
    add(c,"digest",digest(timeless)==p["protocol_digest"],p["protocol_digest"]); add(c,"sources",p["source_hashes"]=={"main":sha(MAIN),"auditor":sha(SCRIPT)},p["source_hashes"])
    add(c,"phase_campaign",(p["phase"],p["campaign"])==(1299,"C032"),[p["phase"],p["campaign"]]); add(c,"hashes",p["material"]["material_sha256"]==sha(M) and p["material"]["naturalness_sha256"]==sha(N) and p["material"]["semantic_reaudit_sha256"]==sha(R),p["material"])
    add(c,"count_ids",len(rr)==6912 and len({x["case_id"] for x in rr})==6912,len(rr)); dims=Counter((x["partition"],x["panel"],x["surface"],x["candidate_order"],x["binding_state"]) for x in rr); add(c,"factorial",len(dims)==144 and set(dims.values())=={48},{"cells":len(dims),"counts":list(set(dims.values()))})
    semantic=all(sum(f[x["attribute"]]==x["target_value"] for f in x["assignments"].values())==1 for x in rr); gold=all([e for e,f in x["assignments"].items() if f[x["attribute"]]==x["target_value"]]==[x["gold_candidate"]] for x in rr); add(c,"semantic_gold",semantic and gold,[semantic,gold])
    groups=defaultdict(list)
    for x in rr: groups[x["group_id"]].append(x)
    add(c,"pairs",len(groups)==3456 and all(len(v)==2 for v in groups.values()),len(groups)); add(c,"active_null",all(v[0]["gold_candidate"]!=v[1]["gold_candidate"] for k,v in groups.items() if "|active|" in k) and all(v[0]["gold_candidate"]==v[1]["gold_candidate"] for k,v in groups.items() if "|matched_null|" in k),"paired")
    add(c,"surface",all(x["candidate_prompt"].endswith("Answer:") and x["candidate_prompt"].count("?")==1 and "  " not in x["candidate_prompt"] for x in rr),"all"); add(c,"reaudit",load(R)["all_checks_passed"],load(R))
    add(c,"compiler_arms",p["execution_compiler"]["arms"]==["left_global_baseline","right_padding","record_event_aligned","equalized_suffix"] and p["execution_compiler"]["candidate_arms"]==p["execution_compiler"]["selection_priority"],[p["execution_compiler"]["arms"],p["execution_compiler"]["selection_priority"]])
    nt=p["execution_compiler"]["numeric_thresholds"]; add(c,"numeric_gates",nt["candidate_compilers_passing_min"]==2 and nt["same_prefix_relative_max"]==1e-6 and nt["tau_cap"]==1e-4,nt)
    add(c,"behavior_before_hidden",p["branches"]["phase1300_pass"].endswith("phase1301_behavior_only") and p["branches"]["phase1301_pass"].endswith("phase1302_event_identity_hidden_only"),p["branches"]); add(c,"failure_stops",p["branches"]["phase1300_fail"].startswith("close_c032") and p["branches"]["phase1302_fail"].startswith("close_c032"),p["branches"])
    add(c,"hidden_events",p["hidden"]["primary_events"]==["user_answer_cue_end","assistant_answer_boundary"] and p["hidden"]["measurements"]==["normalized residual response magnitude","signed candidate-identity logit-lens response"],p["hidden"]); add(c,"causal_controls",p["causal"]["controls"]==["matched-null donor","wrong-entity donor","wrong-attribute donor","neutral no-patch"] and p["causal"]["discovery_forbidden"],p["causal"])
    add(c,"single_model",p["model"]=={"id":"qwen3-4b","dtype":"FP16","device":"CUDA","quantization":False,"other_models_authorized":False,"one_formal_run_per_model_phase":True},p["model"]); add(c,"weights_not_loaded",p["model_weights_loaded"] is False,p["model_weights_loaded"])
    passed=all(x["passed"] for x in c); doc={"phase":1299,"campaign":"C032","created_at_utc":datetime.now(timezone.utc).isoformat(),"auditor_imports_main":False,"checks":c,"passed_count":sum(x["passed"] for x in c),"total_count":len(c),"all_checks_passed":passed,"authorization":"phase1300_compiler_competition_only" if passed else "none","protocol_digest":p["protocol_digest"]}; A.parent.mkdir(parents=True,exist_ok=True); A.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(canonical({"passed":doc["passed_count"],"total":doc["total_count"],"authorization":doc["authorization"]}));
    if not passed: raise SystemExit(1)

if __name__=="__main__": main()
