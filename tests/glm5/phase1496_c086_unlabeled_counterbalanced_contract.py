#!/usr/bin/env python3
"""Phase1496: freeze C086 unlabeled, answer-counterbalanced observation campaign."""
from __future__ import annotations
import json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; PARENT=RESULT/"phase1495_c085_major_stage_closure"; C085=RESULT/"phase1489_c085_prospective_layered_contract"; OUT=RESULT/"phase1496_c086_unlabeled_counterbalanced_contract"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1435_c072_permutation_spectrum_contract as c072
import phase1456_c077_labeled_relation_contract as relation_source
import phase1489_c085_prospective_layered_contract as c085
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer
PHASE,CAMPAIGN=1496,"C086"; FAMILIES=c085.FAMILIES; ORDER=tuple(FAMILIES); RELATIONS=relation_source.RELATIONS; IDS=tuple(RELATIONS); PARTITIONS=relation_source.PARTITIONS
SYSTEM="Compare the kind of action relation expressed by the two clauses. Ignore whether the person and organization match. Follow the item-specific answer code exactly. Output exactly yes or no."
SURFACES={
 "a_unlabeled":"First clause: {record_target} {record_relation} {record_object}. Second clause: {query_target} may {query_relation} {query_object}. Do the clauses express the same kind of action relation? For this item, use {same_answer} for same and {different_answer} for different. Answer only yes or no.",
 "b_unlabeled":"Recorded fact: {record_target} {record_relation} {record_object}. Possible fact: {query_target} can {query_relation} {query_object}. Is the action relation the same in both facts? In this item, write {same_answer} when it is the same and {different_answer} when it differs. Reply only yes or no.",
}
CODEBOOKS={"standard":{"same":"yes","different":"no","sign":1},"reversed":{"same":"no","different":"yes","sign":-1}}
ROLES=("record_target","record_relation","record_object","query_target","query_relation","query_object"); ROLE_SLOTS=ROLES+("boundary",); CELLS=tuple(f"{e}{o}{r}" for e in (1,0) for o in (1,0) for r in (1,0))
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
                    for codebook,code in CODEBOOKS.items():
                        for em in (1,0):
                            for om in (1,0):
                                for rm in (1,0):
                                    qid=rid if rm else oid; output_yes=bool(rm) if code["sign"]==1 else not bool(rm)
                                    row={"case_id":f"c086-a-{len(rows):05d}","partition":partition(index),"family":family,"other_family":other_family,"index":index,"surface":surface,"codebook":codebook,"code_sign":code["sign"],"cell":f"{em}{om}{rm}","record_target":rt,"record_relation":RELATIONS[rid]["record"],"record_relation_id":rid,"record_object":family,"query_target":rt if em else ot,"query_relation":RELATIONS[qid]["query"],"query_relation_id":qid,"query_object":family if om else other_family,"entity_match":bool(em),"object_match":bool(om),"relation_match":bool(rm),"semantic_truth":bool(rm),"same_answer":code["same"],"different_answer":code["different"],"output_yes":output_yes,"candidates":["yes","no"],"gold_position":0 if output_yes else 1}
                                    row["prompt"]=template.format(**row); rows.append(row)
    return rows
def compile_rows(tok,rows):
    result=[]
    for row in rows:
        ids=core.chat_ids(tok,SYSTEM,row["prompt"]); spans={role:c072.all_spans(tok,ids,row[role]) for role in ROLES}
        if not all(spans.values()): raise RuntimeError((row["case_id"],spans))
        positions={role:(spans[role][0] if role.startswith("record") else spans[role][-1]) for role in ROLES}; positions["boundary"]=[len(ids)-1]
        result.append({**row,"prompt_ids":ids,"role_positions":positions,"candidate_ids":[list(map(int,tok.encode(" "+v,add_special_tokens=False))) for v in row["candidates"]]})
    return result
def composition_sets(rows):
    by={(r["family"],r["index"],r["record_relation_id"],r["surface"],r["codebook"],r["cell"]):r for r in rows}; result=[]
    for family in ORDER:
        for index in range(6):
            for rid in IDS:
                group={"set_id":f"c086-compose-{len(result):04d}","partition":partition(index),"family":family,"index":index,"record_relation_id":rid}
                for surface in SURFACES:
                    for codebook in CODEBOOKS:
                        for cell in CELLS: group[f"{surface}_{codebook}_{cell}"]=by[(family,index,rid,surface,codebook,cell)]["case_id"]
                result.append(group)
    return result
def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1496 exists")
    parent=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json")
    if parent["authorization"]!="preregister_c086_label_carrier_withdrawal_layered_observation" or not pa["all_checks_passed"]: raise RuntimeError("Phase1495 authorization missing")
    tok=tokenizer(); rows=active_cases(); compiled=compile_rows(tok,rows); sets=composition_sets(rows); lengths={(s,c):{len(r["prompt_ids"]) for r in compiled if r["surface"]==s and r["codebook"]==c} for s in SURFACES for c in CODEBOOKS}; signatures={(s,c):{tuple((role,tuple(r["role_positions"][role])) for role in ROLE_SLOTS) for r in compiled if r["surface"]==s and r["codebook"]==c} for s in SURFACES for c in CODEBOOKS}; labels={v["label"] for v in RELATIONS.values()}; prompts="\n".join(r["prompt"].lower() for r in rows)
    semantic=[r["semantic_truth"] for r in rows]; output=[r["output_yes"] for r in rows]
    zero={"always_yes":relation_source.ba(output,[True]*len(rows)),"always_no":relation_source.ba(output,[False]*len(rows)),"relation_only":relation_source.ba(output,[r["relation_match"] for r in rows]),"code_only":relation_source.ba(output,[r["code_sign"]==1 for r in rows]),"entity":relation_source.ba(output,[r["entity_match"] for r in rows]),"object":relation_source.ba(output,[r["object_match"] for r in rows]),"relation_x_code_oracle":relation_source.ba(output,[r["relation_match"]==(r["code_sign"]==1) for r in rows])}
    checks={"parent":pa["all_checks_passed"],"active":len(rows)==6912,"composition":len(sets)==216,"surface_balance":Counter(r["surface"] for r in rows)=={s:3456 for s in SURFACES},"code_balance":Counter(r["codebook"] for r in rows)=={c:3456 for c in CODEBOOKS},"semantic_balance":Counter(semantic)=={True:3456,False:3456},"output_balance":Counter(output)=={True:3456,False:3456},"semantic_definition":all(r["semantic_truth"]==(r["record_relation_id"]==r["query_relation_id"]) for r in rows),"counterbalance":all(r["output_yes"]==(r["relation_match"]==(r["code_sign"]==1)) for r in rows),"labels_absent":all(label.lower() not in prompts for label in labels),"singletons":all(len(tok.encode(" "+v,add_special_tokens=False))==1 for v in set(FAMILIES)|{x for vs in FAMILIES.values() for x in vs}|{x for rel in RELATIONS.values() for x in (rel["record"],rel["query"])}|{"yes","no"}),"same_shape":all(len(v)==1 for v in lengths.values()),"stable_roles":all(len(v)==1 for v in signatures.values()),"role_singletons":all(all(len(r["role_positions"][role])==1 for role in ROLE_SLOTS) for r in compiled),"zero_models":all(v==.5 for k,v in zero.items() if k!="relation_x_code_oracle") and zero["relation_x_code_oracle"]==1.0,"machine_naturalness":all(r["prompt"].count("?")==1 and r["prompt"].endswith("yes or no.") for r in rows),"hidden_not_accessed":True}
    if not all(checks.values()): raise RuntimeError({k:v for k,v in checks.items() if not v})
    core.write_rows(OUT/"material/active_cases.jsonl",rows); core.write_rows(OUT/"compiled/qwen3_active.jsonl",compiled); core.write_rows(OUT/"material/composition_sets.jsonl",sets); examples=[rows[i] for i in (0,8,2,10)]; core.write_rows(OUT/"material/frozen_test_examples.jsonl",examples); core.save(OUT/"material/frozen_concept_graph.json",{"schema":"c086.unlabeled_counterbalanced.v1","families":FAMILIES,"relations":RELATIONS,"surfaces":SURFACES,"codebooks":CODEBOOKS,"partitions":{k:list(v) for k,v in PARTITIONS.items()}})
    pre={"phase":PHASE,"campaign":CAMPAIGN,"checks":checks,"zero_models":zero,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"semantic_scope":"same-versus-different natural action relation under counterbalanced yes/no code","naturalness_scope":"machine-audited controlled English; no independent human blind review","hidden_state_accessed":False}; core.save(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json",pre)
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"schema":"c086.unlabeled_counterbalanced_layered_observation.v1","model":"qwen3-bfloat16-cuda-no-quantization","research_object":"separate unlabeled lexical relation content from answer-code polarity in the embedding-to-all-Hidden-State field","paired_source":"C085 entities, relations and nuisances; explicit abstract relation labels removed","roles":list(ROLE_SLOTS),"relations":list(IDS),"surfaces":list(SURFACES),"codebooks":CODEBOOKS,"cells":list(CELLS),"partitions":list(PARTITIONS),"factors":["relation","entity","object","code"],"allowed_observables":["input embeddings","all full-dimensional Hidden States","yes/no logits"],"forbidden":["attention","MLP","parameters","gradients","PCA","TDA","learned probes","post-discovery threshold changes","post-unblind material changes"],"material":{"active_count":6912,"composition_count":216,"cases_per_composition":32,"active_sha256":core.sha(OUT/"material/active_cases.jsonl"),"compiled_sha256":core.sha(OUT/"compiled/qwen3_active.jsonl"),"composition_sha256":core.sha(OUT/"material/composition_sets.jsonl"),"human_naturalness_lock":False},"behavior_strata":{"success":"32/32 correct","mixed":"1-31/32 correct","failed":"0/32 correct","role":"stratification, not campaign stop"},"capture":{"scope":"all 6912 cases","state_count":37,"role_slot_count":7,"hidden_dimension":2560,"dtype":"float16","no_pooling":True,"no_coordinate_selection":True},"observation":{"full_factorial_effect_count":15,"key_effects":["relation","code","relation_code"],"conditional_relation_formula":"D_standard=C_R+0.5*C_RP; D_reversed=C_R-0.5*C_RP","coefficient_energy_formula":"rho_content=||C_R/2||^2/(||C_R/2||^2+||C_RP/4||^2)","discovery_partition":"response_discovery","validation_partitions":["confirmation","lockbox"],"discovery_first":True},"route":["phase1497 behavior strata","phase1498 all-case field capture","phase1499 four-factor atlas","phase1500 discovery observation and freeze","phase1501 dual-holdout validation","phase1502 behavior-stratum and C085 paired diagnostics","phase1503 closure"],"stop_rule":"only integrity, nonfinite execution, or contract mutation stops the campaign; all behavior strata and missing strata continue under typed evidence","claim_boundary":{"allowed":"Qwen3 controlled unlabeled lexical relation/content-versus-code observations","forbidden":["natural relation mechanism in general","causal necessity or sufficiency","semantic neurons","cross-model law","new mathematics"]},"created_at_utc":datetime.now(timezone.utc).isoformat()}
    protocol["contract_sha256"]=core.digest(protocol); protocol["authorization"]="run_phase1497_c086_behavior_stratification"; core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"analysis/uploaded_analysis_adjudication.json",{"retain":["P084 was a genuine fresh-material prospective replication","mixed averages do not identify per-case failures","explicit labels and yes/no polarity remain the dominant boundaries","six P084 metrics are correlated views of one field"],"correct":["decision field is a task-scoped name, not a natural semantic mechanism","low-dimensional manifold, fiber bundle and new mathematics are unsupported","neuron-level structure was not measured","C086 must orthogonalize semantic match and output code, not merely delete label words"]}); core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"status":"unlabeled_counterbalanced_contract_frozen","contract_sha256":protocol["contract_sha256"],"authorization":protocol["authorization"]}); print(json.dumps({"preaudit":pre,"contract_sha256":protocol["contract_sha256"],"authorization":protocol["authorization"]},indent=2))
if __name__=="__main__": main()
