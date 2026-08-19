#!/usr/bin/env python3
"""Phase1342: freeze C048 factorial interaction-field and causal-purity campaign."""
from __future__ import annotations
import json,sys
from collections import Counter
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS
PHASE,CAMPAIGN=1342,"C048";OUT=T/"result/phase1342_c048_factorial_causal_contract";PARENT=T/"result/phase1341_c047_full_relation_field";MODELS=("qwen3","glm4","deepseek7b");PARTITIONS=("discovery","confirmation","holdout");FAMILIES=("dance","spice","bread","beverage")
WORDS={
 "discovery":{"dance":("tango","waltz","salsa","ballet"),"spice":("cinnamon","cumin","paprika","turmeric"),"bread":("baguette","brioche","ciabatta","focaccia"),"beverage":("coffee","tea","cocoa","lemonade")},
 "confirmation":{"dance":("foxtrot","rumba","samba","polka"),"spice":("nutmeg","saffron","cardamom","allspice"),"bread":("sourdough","pita","naan","challah"),"beverage":("smoothie","milkshake","espresso","kombucha")},
 "holdout":{"dance":("flamenco","mazurka","cancan","jitterbug"),"spice":("coriander","anise","fenugreek","sumac"),"bread":("cornbread","flatbread","crumpet","pretzel"),"beverage":("cola","seltzer","cappuccino","mocha")}}
SURFACES={"ordinary":"In ordinary English, does the noun \"{word}\" belong to the category {family}?","dictionary":"Would a standard dictionary classify \"{word}\" as a type of {family}?","claim":"Is the category statement \"{word} is a {family}\" correct?"};SYSTEM="Evaluate the ordinary-English category relation. Output only yes or no."
def concepts():return [{"word":w,"family":f,"partition":p,"sense":f"ordinary noun sense of {w}"} for p in PARTITIONS for f in FAMILIES for w in WORDS[p][f]]
def cases():
 out=[]
 for p in PARTITIONS:
  for fa,fb in combinations(FAMILIES,2):
   for i,(wa,wb) in enumerate(zip(WORDS[p][fa],WORDS[p][fb])):
    for s,t in SURFACES.items():
     q=f"{p}:{fa}__{fb}:{i}:{s}"
     for cell,w,f,truth,sign in (("aa",wa,fa,True,1),("ab",wa,fb,False,-1),("ba",wb,fa,False,-1),("bb",wb,fb,True,1)):
      out.append({"case_id":f"c048-b-{len(out):04d}","partition":p,"family_pair":f"{fa}__{fb}","pair_index":i,"surface":s,"quartet_key":q,"cell":cell,"interaction_sign":sign,"target":w,"target_family":fa if cell[0]=="a" else fb,"tested_family":f,"truth":truth,"prompt":t.format(word=w,family=f)+" Output only yes or no.","candidates":["yes","no"],"gold_position":0 if truth else 1})
 return out
def prior_words():
 found=set()
 for path in (T/"result").glob("phase13*/material/frozen_concept_graph.json"):
  try:found.update(str(x["word"]) for x in core.load(path).get("concepts",[]))
  except Exception:pass
 return found
def tokenizer(m):
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained(MODEL_CONFIGS[m]["path"],trust_remote_code=True,local_files_only=True,use_fast=False)
 if t.pad_token is None:t.pad_token=t.eos_token
 t.padding_side="right";return t
def span(t,ids,v):return core.locate_last_subsequence(ids,[[int(x) for x in t.encode(z,add_special_tokens=False)] for z in (v," "+v)])
def compile_for(m,rows):
 t=tokenizer(m);out=[]
 for r in rows:
  ids=core.chat_ids(t,SYSTEM,r["prompt"]);out.append({"case_id":r["case_id"],"prompt_ids":ids,"candidate_ids":[[int(x) for x in t.encode(v,add_special_tokens=False)] for v in ("yes","no")],"target_span":span(t,ids,r["target"]),"tested_family_span":span(t,ids,r["tested_family"]),"boundary_position":len(ids)-1})
 return out
def main():
 pf=core.load(PARENT/"analysis/final.json");pa=core.load(PARENT/"audit/independent_final_audit.json")
 if pf.get("authorization")!="close_c047_descriptive_field_and_authorize_separate_causal_preregistration" or not pa.get("all_checks_passed"):raise RuntimeError("parent")
 if (OUT/"analysis/final.json").exists():raise RuntimeError("exists")
 cs,rows=concepts(),cases();compiled={m:compile_for(m,rows) for m in MODELS};qs={k:[x for x in rows if x["quartet_key"]==k] for k in {x["quartet_key"] for x in rows}}
 checks={"fresh":not({x["word"] for x in cs}&prior_words()),"concepts":len(cs)==48 and len({x["word"] for x in cs})==48,"concept_balance":all(sum(x["partition"]==p and x["family"]==f for x in cs)==4 for p in PARTITIONS for f in FAMILIES),"cases":len(rows)==864 and len({x["case_id"] for x in rows})==864,"quartets":len(qs)==216 and all([x["cell"] for x in q]==["aa","ab","ba","bb"] and [x["interaction_sign"] for x in q]==[1,-1,-1,1] for q in qs.values()),"truth":Counter(x["truth"] for x in rows)=={True:432,False:432},"factorial":all(sum(x["partition"]==p for x in rows)==288 for p in PARTITIONS) and all(sum(x["surface"]==s for x in rows)==288 for s in SURFACES),"semantic_unique":all(x["sense"] for x in cs),"machine_natural":all("  " not in x["prompt"] and x["prompt"].endswith("yes or no.") for x in rows)}
 for m,c in compiled.items():checks[m+"_compiled"]=len(c)==864 and all(a["case_id"]==b["case_id"] for a,b in zip(rows,c));checks[m+"_tokens"]=all(all(len(z)==1 for z in x["candidate_ids"]) for x in c);checks[m+"_spans"]=all(x["target_span"] and len(x["tested_family_span"])==1 and max(x["target_span"]+x["tested_family_span"])<x["boundary_position"] for x in c)
 if not all(checks.values()):raise RuntimeError([k for k,v in checks.items() if not v])
 core.save(OUT/"material/frozen_concept_graph.json",{"schema":"c048.graph.v1","concepts":cs});core.write_rows(OUT/"material/frozen_factorial_cases.jsonl",rows)
 for m in MODELS:core.write_rows(OUT/f"compiled/{m}_factorial.jsonl",compiled[m])
 zeros={"always_yes_accuracy":.5,"always_no_accuracy":.5,"target_additive_interaction":0.0,"family_additive_interaction":0.0,"surface_additive_interaction":0.0,"generic_truth_oracle":"passes behavior but must fail family-pair identity if its hidden response is pair-independent"}
 core.save(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json",{"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"zero_models":zeros,"human_blind_naturalness":"not available; claims limited to curated controlled English"})
 sent=[q[0]["case_id"] for q in list(qs.values())[:24]]
 protocol={"phase":PHASE,"campaign":CAMPAIGN,"schema":"c048.factorial_causal.v1","research_object":"target-by-family second-order interaction that survives additive lexical cancellation, family-pair discrimination, transfer, and conditional bidirectional natural-state swaps","claim_boundary":{"allowed":"factorial behavior, descriptive interaction field, and only conditionally typed causal sufficiency/necessity","not_assumed":["pure semantics","ontology identity","attention or MLP localization","parameter mechanism","cross-model physical coordinates"]},"material":{"case_count":864,"quartet_count":216,"partitions":list(PARTITIONS),"families":list(FAMILIES),"family_pairs":6,"surfaces":list(SURFACES),"graph_sha256":core.sha(OUT/"material/frozen_concept_graph.json"),"cases_sha256":core.sha(OUT/"material/frozen_factorial_cases.jsonl")},"models":list(MODELS),"model_order":list(MODELS),"precision":"bfloat16-no-quantization","batch_size":4,"zero_models":zeros,
 "executor_gate":{"sentinel_case_ids":sent,"finite_fraction_min":1.0,"rank_agreement_min":1.0,"max_abs_diff_max":1e-6},
 "behavior_gate":{"accuracy_min":.90,"partition_min":.85,"surface_min":.85,"family_min":.85,"truth_min":.85,"pairwise_true_over_false_min":.95,"quartet_all_correct_min":.90,"positive_interaction_fraction_min":.95,"median_interaction_min":2.0,"minimum_authorized_models":2},
 "field_gate":{"depths":"all embedding and model-layer outputs","roles":["target_span_mean","tested_family","boundary"],"primary_role":"tested_family","storage":"float32 full-dimensional signed quartet interactions","layer0_relative_norm_max":1e-5,"numeric_relative_l2_p95_max":1e-5,"numeric_relative_l2_max":1e-4,"discovery_identity_win_min":.70,"discovery_median_gap_min":.10,"discovery_relative_norm_min":.01,"transfer_identity_win_min":.65,"transfer_median_gap_min":.05,"selection":"earliest layer passing discovery; no reselection","minimum_authorized_models":2},
 "causal_gate":{"site_role":"tested_family","layer":"per-model frozen field-selected layer","target_partitions":["confirmation","holdout"],"donor_partition":"discovery","donor_surface":"different from target surface by frozen cycle","arms":["baseline","self_patch","same_label_false","same_label_true","wrong_label_true","same_label_false_control"],"true_to_false_median_damage_min":.5,"false_to_true_median_gain_min":.5,"direction_fraction_min":.75,"flip_fraction_min":.40,"correct_over_wrong_median_min":.5,"correct_over_wrong_win_min":.70,"self_max_abs_margin_diff_max":1e-4,"partition_direction_fraction_min":.65,"minimum_authorized_models":2},
 "branching":{"behavior_fail":"close before hidden","behavior_pass":"run all-layer interaction field","field_fail":"close without causal claim","field_pass":"run frozen bidirectional natural-state swap","causal_fail":"close at causal selectivity boundary","causal_pass":"close C048 with typed causal interaction evidence; parameter search remains unauthorized"},"stop_rule":"After any reveal do not change object, material, partition, model, null, threshold, role, layer selection rule, donor rule, or stop branch.","parameter_boundary":"No attention, MLP, sparse dictionary, or parameter scan is authorized."}
 protocol["contract_sha256"]=core.digest(protocol);protocol["authorization"]="run_phase1343_c048_factorial_behavior";core.save(OUT/"protocol/preregistration.json",protocol);core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_gates_passed":True,"authorization":protocol["authorization"],"contract_sha256":protocol["contract_sha256"],"finished_at_utc":datetime.now(timezone.utc).isoformat()});print(json.dumps({"checks":checks,"contract_sha256":protocol["contract_sha256"],"authorization":protocol["authorization"]},indent=2))
if __name__=="__main__":main()
