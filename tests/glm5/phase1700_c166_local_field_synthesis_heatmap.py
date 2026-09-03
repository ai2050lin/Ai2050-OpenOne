#!/usr/bin/env python3
"""C166: synthesize C157-C165 and publish a full-coordinate visualization asset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"
OUT=RESULT/"phase1700_c166_local_field_synthesis_heatmap"
FRONTEND=ROOT/"frontend/public/vis_data/research_kernel/c157_c166_local_field_heatmap.json"
DIRS={c:RESULT/d for c,d in {
    157:"phase1691_c157_local_field_master_contract",158:"phase1692_c158_increment_source_decomposition",
    159:"phase1693_c159_natural_isomorphic_dual_graph_atlas",160:"phase1694_c160_recipient_only_counterfactual_prediction",
    161:"phase1695_c161_full_coordinate_local_transmission",162:"phase1696_c162_linguistic_program_field",
    163:"phase1697_c163_natural_graph_call_domain",164:"phase1698_c164_three_model_free_interface",
    165:"phase1699_c165_cross_model_relative_topology"}.items()}
sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
PHASE,CAMPAIGN=1700,"C166";DIM=2560;ROLES=("primary","secondary","relation","context","query","boundary")

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bfloat_bits(row):
    u=np.asarray(row,dtype=np.uint16).astype(np.uint32);return (u<<16).view(np.float32)
def add(rows,dataset,kind,label,values,**meta):
    v=np.asarray(values,np.float32).reshape(-1)
    if len(v)!=DIM:raise RuntimeError((label,len(v)))
    rows.append({"dataset":dataset,"kind":kind,"label":label,"values":v.astype(float).tolist(),**meta})

def contract():
    if OUT.exists():raise RuntimeError(OUT)
    audits={c:core.load(d/"audit/independent_final_audit.json") for c,d in DIRS.items()}
    checks={"all_parent_audits":all(a["all_checks_passed"] for a in audits.values()),"campaigns":len(audits)==9,"frontend_parent":FRONTEND.parent.exists(),"dimension":DIM==2560}
    if not all(checks.values()):raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"synthesis_heatmap_contract_frozen","sources":list(range(157,166)),"asset":str(FRONTEND.relative_to(ROOT)),"coordinate_semantics":"Qwen3-4B embedding/HiddenState physical activation coordinates 0..2559; not weights or cross-model coordinates","rows":{"C159":"natural/nonce field and representative raw embedding/HiddenState","C160":"recipient-only prediction versus actual response","C161":"all-source transmission strength and selected source-target fields","C162":"broad linguistic first/second-order fields","C165":"Qwen-only representative raw role states plus coordinate-free cross-model summary"},"forbidden":["attention","MLP","weights","PCA","invented cross-model coordinate mapping"],"source_hashes":{str(c):core.sha(d/"analysis/final.json") for c,d in DIRS.items()},"producer_sha256":core.sha(Path(__file__)),"authorization":"build_asset_then_audit"}
    core.save(OUT/"protocol/preregistration.json",protocol);core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":True});print(json.dumps({"checks":checks,"protocol":protocol},indent=2))

def c159_rows(rows):
    d=DIRS[159]
    coordinate=json.loads((d/"analysis/coordinate_rows.json").read_text(encoding="utf-8"))
    for r in coordinate:
        if r["checkpoint"] in (24,32,34) and r["role"] in ("relation","boundary"):
            add(rows,"C159","paired_response",f"C159 {r['panel']} {r['relation_family']} {r['role']} q{r['checkpoint']}",r["values"],panel=r["panel"],relation=r["relation_family"],role=r["role"],checkpoint=r["checkpoint"])
    raw=np.load(d/"raw/qwen3_six_role_all_checkpoint.bf16.npy",mmap_mode="r")
    for case_index,panel in ((0,"natural_lexical"),(768,"isomorphic_nonce")):
        for role_index,role in ((2,"relation"),(5,"boundary")):
            for q in (0,32,37):
                add(rows,"C159","representative_raw",f"C159 raw {panel} {role} q{q}",bfloat_bits(raw[case_index,role_index,q]),panel=panel,role=role,checkpoint=q,state_kind="embedding" if q==0 else "hidden_state")

def c160_rows(rows):
    d159,d160=DIRS[159],DIRS[160]
    pred=np.load(d160/"analysis/fresh_selected_predictions.float16.npy",mmap_mode="r")
    target=np.load(d159/"analysis/late_half_difference.float16.npy",mmap_mode="r")
    pairs=core.rows(d159/"analysis/late_half_difference_index.jsonl")
    fresh=[r for r in pairs if r["partition"]=="fresh"]
    if len(fresh)!=256:raise RuntimeError(len(fresh))
    qi=8
    for panel in ("natural_lexical","isomorphic_nonce"):
        ids=[i for i,r in enumerate(fresh) if r["panel"]==panel]
        actual=np.asarray([target[fresh[i]["pair_index"],qi] for i in ids],np.float32).mean(0)
        predicted=np.asarray(pred[ids,qi],np.float32).mean(0)
        for ri,role in enumerate(ROLES):
            add(rows,"C160","recipient_prediction",f"C160 predicted {panel} {role} q32",predicted[ri],panel=panel,role=role,checkpoint=32)
            add(rows,"C160","actual_response",f"C160 actual {panel} {role} q32",actual[ri],panel=panel,role=role,checkpoint=32)

def c161_rows(rows):
    d=DIRS[161]; raw=np.load(d/"raw/q24_relation_to_q25_six_role_response.float16.npy",mmap_mode="r")
    strength=np.zeros(DIM,np.float32)
    for start in range(0,DIM,32):
        block=np.asarray(raw[:,start:start+32],np.float32)
        strength[start:start+len(block[0])]=np.sqrt(np.mean(block*block,axis=(0,2,3)))
    add(rows,"C161","outgoing_rms", "C161 q24 relation source-coordinate outgoing RMS",strength,checkpoint=24,role="relation")
    top=json.loads((d/"analysis/top_coordinate_edges.json").read_text(encoding="utf-8"))[:16]
    for item in top:
        source=int(item["source_coordinate"]); response=np.asarray(raw[:,source],np.float32).mean(0)
        for ri,role in enumerate(ROLES):
            add(rows,"C161","source_target_response",f"C161 source {source} -> {role} q25",response[ri],source_coordinate=source,target_role=role,checkpoint=25)

def c162_rows(rows):
    d=DIRS[162]; field=np.load(d/"analysis/unit_term_fields.float16.npy",mmap_mode="r")
    index=core.rows(d/"analysis/term_index.jsonl"); chosen=[0,1,2,3,4,5,11,12,14,15,17,19]
    qi=8
    for ti in chosen:
        value=np.asarray(field[6:8,ti,qi],np.float32).mean(0)
        for ri in (2,4,5):
            add(rows,"C162","linguistic_term",f"C162 {index[ti]['name']} {ROLES[ri]} q32",value[ri],term=index[ti]["name"],order=index[ti]["order"],role=ROLES[ri],checkpoint=32)

def c165_rows(rows):
    d=DIRS[165]; p=core.load(d/"protocol/preregistration.json")
    if "qwen3" not in p["eligible_models"]:return
    raw=np.load(d/"raw/qwen3_role_states.float16.npy",mmap_mode="r")
    for case in (0,1):
        for qi,q in ((0,"embedding"),(4,"final")):
            for ri,role in enumerate(("source_record","relation_record","target_record","query_source","boundary")):
                add(rows,"C165","representative_role_state",f"C165 Qwen case {case} {role} {q}",raw[case,qi,ri],case_index=case,role=role,checkpoint=q)

def build():
    rows=[];c159_rows(rows);c160_rows(rows);c161_rows(rows);c162_rows(rows);c165_rows(rows)
    matrix=np.asarray([r["values"] for r in rows],np.float32)
    score=np.mean(np.abs(matrix),axis=0);default=np.argsort(-score)[:64].astype(int).tolist()
    summaries={f"C{c}":core.load(d/"analysis/final.json") for c,d in DIRS.items()}
    asset={"schema":"c157_c166_local_field_heatmap.v1","result_type":"local_field_coordinate_heatmap","phase":PHASE,"campaign":"C157-C166","model":"Qwen3-4B plus coordinate-free cross-model summary","title":"Local Prediction, Coordinate Transmission and Linguistic Program Field","dimensions":list(range(DIM)),"default_coordinates":default,"rows":rows,"summaries":summaries,"c161":core.load(DIRS[161]/"analysis/transmission.json"),"c164":core.load(DIRS[164]/"analysis/summary.json"),"c165":core.load(DIRS[165]/"analysis/summary.json"),"coordinate_semantics":"Every value column is a Qwen3-4B embedding or HiddenState activation coordinate; it is not a model weight, standalone neuron, or cross-model coordinate.","claim_boundary":"C160 closes recipient-only local prediction, C161 shows a reproducible but mostly generic local transport skeleton, C162 shows reusable fixed-operation fields, C163 fails relation-selective call, and C165 cross-model topology was not tested because no common interface qualified."}
    OUT.joinpath("analysis").mkdir(parents=True,exist_ok=True); FRONTEND.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(asset,separators=(",",":"),ensure_ascii=True)
    (OUT/"analysis/heatmap.json").write_text(text,encoding="utf-8");FRONTEND.write_text(text,encoding="utf-8")
    report={"phase":PHASE,"campaign":CAMPAIGN,"status":"asset_built","row_count":len(rows),"datasets":{name:sum(r["dataset"]==name for r in rows) for name in sorted({r["dataset"] for r in rows})},"default_coordinates":default,"asset_bytes":FRONTEND.stat().st_size,"asset_sha256":sha(FRONTEND),"next_authorization":"independent_audit_and_memo"}
    core.save(OUT/"analysis/synthesis.json",report)
    checks={"rows":len(rows)>=150,"dimensions":matrix.shape[1]==DIM,"finite":bool(np.isfinite(matrix).all()),"embedding":any(r.get("state_kind")=="embedding" for r in rows),"hidden":any(r.get("state_kind")=="hidden_state" for r in rows),"asset_equal":sha(FRONTEND)==sha(OUT/"analysis/heatmap.json")}
    core.save(OUT/"audit/internal_build_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())});print(json.dumps(report,indent=2))

def close():
    r=core.load(OUT/"analysis/synthesis.json");checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"build":core.load(OUT/"audit/internal_build_audit.json")["all_checks_passed"]}
    final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed","checks":checks,"all_checks_passed":all(checks.values()),"headline":r,"next_authorization":"new campaign only after reviewing C157-C166"};core.save(OUT/"analysis/final.json",final);print(json.dumps(final,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("contract","build","close"));a=p.parse_args();{"contract":contract,"build":build,"close":close}[a.command]()
if __name__=="__main__":main()
