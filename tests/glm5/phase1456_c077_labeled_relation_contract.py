#!/usr/bin/env python3
"""Phase1456: preregister C077 labeled-relation full-field calibration."""
from __future__ import annotations
import json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as c072
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer

PHASE, CAMPAIGN = 1456, "C077"; PARENT = TESTS / "result/phase1455_c076_behavior_gate_closure"; OUT = TESTS / "result/phase1456_c077_labeled_relation_contract"
FAMILIES = {
    "Apex": ("Marshall", "Marvin", "Maurice", "Miles", "Nikki", "Otto"),
    "Domain": ("Quentin", "Rafael", "Reese", "Reid", "Ricardo", "Rodney"),
    "Granite": ("Roman", "Rosa", "Sally", "Saul", "Sergio", "Sonia"),
    "Jade": ("Stacy", "Stefan", "Tobias", "Trent", "Vera", "Vernon"),
    "Portal": ("Zelda", "Olive", "Pearl", "Rita", "Simone", "Alberta"),
    "Relay": ("Beth", "Cara", "Cyrus", "Dante", "Desmond", "Elliott"),
}
RELATIONS = {
    "join": {"record": "joined", "query": "join", "label": "affiliation"},
    "support": {"record": "supported", "query": "support", "label": "assistance"},
    "visit": {"record": "visited", "query": "visit", "label": "travel"},
    "contact": {"record": "contacted", "query": "contact", "label": "communication"},
    "select": {"record": "selected", "query": "select", "label": "choice"},
    "praise": {"record": "praised", "query": "praise", "label": "admiration"},
}
IDS = tuple(RELATIONS); ORDER = tuple(FAMILIES); PARTITIONS = {"response_discovery": range(0,2), "confirmation": range(2,4), "lockbox": range(4,6)}
SYSTEM = "Compare only the two explicit relation labels. Answer yes when the labels are identical and no otherwise. The person, verb form, and organization are context. Output exactly yes or no."
SURFACES = {
    "a_labeled": "First relation label: {record_label}. First clause: {record_target} {record_relation} {record_object}. Second relation label: {query_label}. Second clause: {query_target} may {query_relation} {query_object}. Are the two relation labels identical? Answer only yes or no.",
    "b_labeled": "Relation label one is {record_label}. Clause one says {record_target} {record_relation} {record_object}. Relation label two is {query_label}. Clause two says {query_target} can {query_relation} {query_object}. Do the two relation labels match exactly? Reply only yes or no.",
}
ROLES = ("record_label","record_target","record_relation","record_object","query_label","query_target","query_relation","query_object"); ROLE_SLOTS = ROLES + ("boundary",)
CELLS = tuple(f"{e}{o}{r}" for e in (1,0) for o in (1,0) for r in (1,0))
def partition(i): return next(k for k,v in PARTITIONS.items() if i in v)

def active_cases():
    rows=[]
    for fi,family in enumerate(ORDER):
        other_family=ORDER[(fi+1)%6]
        for index in range(6):
            rt,ot=FAMILIES[family][index],FAMILIES[other_family][index]
            for ri,rid in enumerate(IDS):
                oid=IDS[(ri+1+index%2)%6]
                for surface,template in SURFACES.items():
                    for em in (1,0):
                        for om in (1,0):
                            for rm in (1,0):
                                qid=rid if rm else oid
                                row={"case_id":f"c077-a-{len(rows):04d}","partition":partition(index),"family":family,"other_family":other_family,"index":index,"surface":surface,"cell":f"{em}{om}{rm}",
                                     "record_label":RELATIONS[rid]["label"],"record_target":rt,"record_relation":RELATIONS[rid]["record"],"record_relation_id":rid,"record_object":family,
                                     "query_label":RELATIONS[qid]["label"],"query_target":rt if em else ot,"query_relation":RELATIONS[qid]["query"],"query_relation_id":qid,"query_object":family if om else other_family,
                                     "entity_match":bool(em),"object_match":bool(om),"relation_match":bool(rm),"truth":bool(rm),"candidates":["yes","no"],"gold_position":0 if rm else 1}
                                row["prompt"]=template.format(**row); rows.append(row)
    return rows

def compile_rows(tok, rows):
    out=[]
    for row in rows:
        ids=core.chat_ids(tok,SYSTEM,row["prompt"]); spans={r:c072.all_spans(tok,ids,row[r]) for r in ROLES}
        if not all(spans.values()): raise RuntimeError((row["case_id"],spans))
        pos={r:spans[r][0] for r in ROLES[:4]}; pos.update({r:spans[r][-1] for r in ROLES[4:]}); pos["boundary"]=[len(ids)-1]
        out.append({**row,"prompt_ids":ids,"role_positions":pos,"candidate_ids":[list(map(int,tok.encode(" "+v,add_special_tokens=False))) for v in row["candidates"]]})
    return out

def composition_sets(active):
    by={(r["family"],r["index"],r["record_relation_id"],r["surface"],r["cell"]):r for r in active}; out=[]
    for family in ORDER:
        for index in range(6):
            for rid in IDS:
                row={"set_id":f"c077-compose-{len(out):04d}","partition":partition(index),"family":family,"index":index,"record_relation_id":rid}
                for s in SURFACES:
                    for c in CELLS: row[f"{s}_{c}"]=by[(family,index,rid,s,c)]["case_id"]
                out.append(row)
    return out

def ba(t,p): return c072.balanced_accuracy(t,p)
def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1456 exists")
    parent=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json")
    if parent["authorization"]!="preregister_c077_labeled_relation_full_field_calibration" or not pa["all_checks_passed"]: raise RuntimeError("C076 closure missing")
    tok=tokenizer(); active=active_cases(); compiled=compile_rows(tok,active); composition=composition_sets(active); old=c072.old_material_words()
    labels=set(FAMILIES); members={x for v in FAMILIES.values() for x in v}; relation_words={x for v in RELATIONS.values() for x in v.values()}; truths=[r["truth"] for r in active]
    lengths={s:{len(r["prompt_ids"]) for r in compiled if r["surface"]==s} for s in SURFACES}; signatures={s:{tuple((role,tuple(r["role_positions"][role])) for role in ROLE_SLOTS) for r in compiled if r["surface"]==s} for s in SURFACES}
    zero={"always_yes":ba(truths,[True]*len(active)),"always_no":ba(truths,[False]*len(active)),"surface":ba(truths,[r["surface"]=="a_labeled" for r in active]),"entity":ba(truths,[r["entity_match"] for r in active]),"object":ba(truths,[r["object_match"] for r in active]),"entity_object":ba(truths,[r["entity_match"] and r["object_match"] for r in active]),"label_identity":ba(truths,[r["record_label"]==r["query_label"] for r in active])}
    checks={"parent":pa["all_checks_passed"],"fresh_labels":len(labels)==6 and not ({x.lower() for x in labels}&old),"fresh_members":len(members)==36 and not ({x.lower() for x in members}&old),"singletons":all(len(tok.encode(" "+x,add_special_tokens=False))==1 for x in labels|members|relation_words),
            "active":len(active)==3456 and Counter(r["surface"] for r in active)=={s:1728 for s in SURFACES},"truth":Counter(truths)=={True:1728,False:1728},"semantic":all(r["truth"]==(r["record_label"]==r["query_label"]) for r in active),"nuisance":all(Counter(r[k] for r in active)=={True:1728,False:1728} for k in ("entity_match","object_match","relation_match")),
            "composition":len(composition)==216 and Counter(r["partition"] for r in composition)=={k:72 for k in PARTITIONS},"compiled":len(compiled)==3456,"same_shape":all(len(v)==1 for v in lengths.values()),"stable_roles":all(len(v)==1 for v in signatures.values()),"role_singletons":all(all(len(r["role_positions"][role])==1 for role in ROLE_SLOTS) and len({r["role_positions"][role][0] for role in ROLE_SLOTS})==9 for r in compiled),"naturalness":all(r["prompt"].count("?")==1 and r["prompt"].endswith("yes or no.") for r in active),"zero":all(v==0.5 for k,v in zero.items() if k!="label_identity") and zero["label_identity"]==1.0,"hidden_not_accessed":True}
    if not all(checks.values()): raise RuntimeError({k:v for k,v in checks.items() if not v})
    core.save(OUT/"material/frozen_concept_graph.json",{"schema":"c077.labeled_relation.v1","families":FAMILIES,"relations":RELATIONS,"partitions":{k:list(v) for k,v in PARTITIONS.items()},"surfaces":SURFACES,"concepts":[{"word":w,"family":f,"index":i,"partition":partition(i)} for f,vs in FAMILIES.items() for i,w in enumerate(vs)]}); core.write_rows(OUT/"material/active_cases.jsonl",active); core.write_rows(OUT/"material/composition_sets.jsonl",composition); core.write_rows(OUT/"compiled/qwen3_active.jsonl",compiled)
    pre={"phase":PHASE,"campaign":CAMPAIGN,"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"zero_models":zero,"semantic_scope":"known-truth explicit relation-label equality paired with natural verbs and balanced entity/object nuisances","naturalness_scope":"machine-audited controlled English; no human blind review"}; core.save(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json",pre)
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"schema":"c077.labeled_relation_full_field.v1","model":"qwen3-bfloat16-cuda-no-quantization","expected_hidden_state_count":37,"research_object":"known-truth labeled relation carrier from embeddings through all role-aligned Hidden States","allowed_observables":["input token embeddings","full-dimensional hidden states at every model state","yes/no logits"],"forbidden":["attention","MLP","parameters","gradients","PCA","t-SNE","UMAP","learned probe","coordinate pruning before raw capture","post-holdout changes"],"roles":list(ROLES),"role_slots":list(ROLE_SLOTS),"relations":RELATIONS,"surfaces":list(SURFACES),"cells":list(CELLS),"partitions":list(PARTITIONS),
              "material":{"active_count":3456,"composition_count":216,"surface_lengths":{k:next(iter(v)) for k,v in lengths.items()},"active_sha256":core.sha(OUT/"material/active_cases.jsonl"),"composition_sha256":core.sha(OUT/"material/composition_sets.jsonl"),"human_naturalness_lock":False},
              "behavior":{"global_surface_balanced_accuracy_min":0.98,"family_relation_surface_accuracy_min":0.95,"family_relation_surface_balanced_accuracy_min":0.95,"partition_min":0.9,"truth_min":0.9,"cell_min":0.9,"all_composition_sets_required":True,"same_batch_repeat_max_abs_diff":1e-6},
              "discovery_capture":{"partition":"response_discovery","expected_case_count":1152,"state_count":37,"role_slot_count":9,"dtype":"float16","raw_format":"numpy memmap N x state x role_slot x hidden_dimension plus JSON index","no_pooling":True,"no_coordinate_selection":True,"no_holdout_access":True},
              "discovery_description":{"effects":["relation_label","entity_nuisance","object_nuisance"],"operation":"full-factorial paired first differences at every coordinate/layer/role/relation/surface","allowed_summaries":["L2 norm","mean vector","direction consistency","cross-surface cosine","coordinate sign consistency"],"candidate_freeze_before_holdout":True,"label_scope_only":True},
              "holdout_validation":{"partitions":["confirmation","lockbox"],"candidate_source":"frozen discovery manifest only","branch_failure":"closes candidate only"},"stop_rule":"behavior first; discovery only raw capture; freeze candidates before holdout; candidate failures do not stop other branches","claim_boundary":{"allowed":"known-truth labeled relation carrier regularities in one controlled Qwen3 task","forbidden":["unlabeled natural relation mechanism","semantic neurons from discovery alone","necessity or natural use","relative encoding proven","cross-model law","new mathematics"]},"branching":{"phase1457":"behavior","phase1458":"discovery capture","phase1459":"description freeze","phase1460":"holdout validation","phase1461":"closure"},"created_at_utc":datetime.now(timezone.utc).isoformat()}
    protocol["contract_sha256"]=core.digest(protocol); protocol["authorization"]="run_phase1457_c077_behavior"; core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_gates_passed":True,"contract_sha256":protocol["contract_sha256"],"authorization":protocol["authorization"]}); print(json.dumps({"preaudit":pre,"protocol":protocol},indent=2))
if __name__=="__main__": main()
