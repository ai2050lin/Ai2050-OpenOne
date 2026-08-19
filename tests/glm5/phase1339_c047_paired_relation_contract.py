#!/usr/bin/env python3
"""Phase1339: freeze C047 paired noun-family relation differential contract."""
from __future__ import annotations

import json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS

PHASE, CAMPAIGN = 1339, "C047"
OUT = T / "result/phase1339_c047_paired_relation_contract"
PARENT = T / "result/phase1338_c046_deconfounded_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "confirmation", "holdout")
FAMILIES = ("insect", "dessert", "fabric", "tree")
LABELS = {"insect":"insect", "dessert":"dessert", "fabric":"fabric", "tree":"tree"}
WORDS = {
 "discovery":{"insect":("ant","bee","beetle","butterfly"),"dessert":("cake","pudding","brownie","doughnut"),"fabric":("denim","velvet","satin","linen"),"tree":("oak","pine","maple","willow")},
 "confirmation":{"insect":("mosquito","dragonfly","grasshopper","termite"),"dessert":("cheesecake","sorbet","custard","tiramisu"),"fabric":("tweed","chiffon","corduroy","flannel"),"tree":("cedar","birch","cypress","redwood")},
 "holdout":{"insect":("ladybird","cicada","firefly","mantis"),"dessert":("macaron","baklava","fruitcake","gelato"),"fabric":("muslin","cashmere","burlap","felt"),"tree":("sycamore","baobab","sequoia","mahogany")},
}
SURFACES = {
 "noun_class":"Does the noun \"{word}\" ordinarily name a kind of {label}?",
 "common_sense":"In ordinary English, is \"{word}\" correctly classified as a {label}?",
 "category_claim":"Is the category claim \"{word} is a {label}\" correct?",
}
SYSTEM = "Evaluate the ordinary-English category relation. Output only yes or no."

def concepts():
 return [{"word":w,"family":f,"label":LABELS[f],"partition":p,"sense":f"ordinary common noun {w}"}
         for p in PARTITIONS for f in FAMILIES for w in WORDS[p][f]]

def material():
 rows=[]
 for c in concepts():
  for s,t in SURFACES.items():
   for f in FAMILIES:
    truth=f==c["family"]
    rows.append({"case_id":f"c047-b-{len(rows):04d}","partition":c["partition"],"surface":s,
      "target":c["word"],"target_family":c["family"],"tested_family":f,"tested_label":LABELS[f],
      "truth":truth,"quartet_key":f'{c["partition"]}:{c["word"]}:{s}',
      "prompt":t.format(word=c["word"],label=LABELS[f])+" Output only yes or no.",
      "candidates":["yes","no"],"gold_position":0 if truth else 1,"gold_value":"yes" if truth else "no"})
 return rows

def tokenizer(name):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL_CONFIGS[name]["path"],trust_remote_code=True,local_files_only=True,use_fast=False)
 if tok.pad_token is None: tok.pad_token=tok.eos_token
 tok.padding_side="right"; return tok

def span(tok,ids,value):
 return core.locate_last_subsequence(ids,[[int(x) for x in tok.encode(v,add_special_tokens=False)] for v in (value," "+value)])

def compile_for(name, rows):
 tok=tokenizer(name); out=[]
 for r in rows:
  ids=core.chat_ids(tok,SYSTEM,r["prompt"])
  out.append({"case_id":r["case_id"],"prompt_ids":ids,
   "candidate_ids":[[int(x) for x in tok.encode(v,add_special_tokens=False)] for v in r["candidates"]],
   "target_span":span(tok,ids,r["target"]),"tested_family_span":span(tok,ids,r["tested_label"]),
   "boundary_position":len(ids)-1})
 return out

def prior_words():
 found=set()
 for path in (T/"result").glob("phase13*/material/frozen_concept_graph.json"):
  try: found.update(str(x["word"]) for x in core.load(path).get("concepts",[]))
  except Exception: pass
 return found

def main():
 pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json")
 if pf.get("authorization")!="close_c046_behavior" or not pa.get("all_checks_passed"): raise RuntimeError("unaudited parent")
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("formal contract exists")
 cs,rows=concepts(),material(); compiled={m:compile_for(m,rows) for m in MODELS}
 quartets={k:[r for r in rows if r["quartet_key"]==k] for k in {r["quartet_key"] for r in rows}}
 checks={"fresh":not({c["word"] for c in cs}&prior_words()),"concepts":len(cs)==48 and len({c["word"] for c in cs})==48,
  "balance":all(sum(c["partition"]==p and c["family"]==f for c in cs)==4 for p in PARTITIONS for f in FAMILIES),
  "cases":len(rows)==576 and len({r["case_id"] for r in rows})==576,
  "truth":Counter(r["truth"] for r in rows)=={True:144,False:432},
  "quartets":len(quartets)==144 and all(len(q)==4 and sum(r["truth"] for r in q)==1 for q in quartets.values()),
  "natural":all("  " not in r["prompt"] and r["prompt"].endswith("yes or no.") for r in rows)}
 for m,cr in compiled.items():
  checks[m+"_compiled"]=len(cr)==576 and all(a["case_id"]==b["case_id"] for a,b in zip(rows,cr))
  checks[m+"_tokens"]=all(all(len(x)==1 for x in r["candidate_ids"]) for r in cr)
  checks[m+"_spans"]=all(r["target_span"] and r["tested_family_span"] and max(r["target_span"]+r["tested_family_span"])<r["boundary_position"] for r in cr)
 if not all(checks.values()): raise RuntimeError([k for k,v in checks.items() if not v])
 core.save(OUT/"material/frozen_concept_graph.json",{"schema":"c047.graph.v1","concepts":cs})
 core.write_rows(OUT/"material/frozen_behavior_cases.jsonl",rows)
 for m in MODELS: core.write_rows(OUT/f"compiled/{m}_behavior.jsonl",compiled[m])
 zeros={"always_yes_accuracy":.25,"always_no_accuracy":.75,"target_only_pair_gap":0.0,"family_only_expected_quartet_rank":.25,"surface_only_pair_gap":0.0}
 core.save(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json",{"checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"zero_models":zeros,"human_blind_naturalness":"not available; claims limited to curated controlled English"})
 sent=[r["case_id"] for r in rows if r["surface"]=="noun_class" and r["truth"]]
 protocol={"phase":PHASE,"campaign":CAMPAIGN,"schema":"c047.paired_relation.v1",
  "research_object":"standard-interface paired noun-family relation differential indexed by target, surface, and tested family",
  "claim_boundary":{"allowed":"behavioral relation ranking and, conditionally, descriptive full-dimensional role-aligned response field","not_assumed":["embedding is a semantic fingerprint","hidden states are a path","sparse basis","causal use","cross-model coordinate identity"]},
  "material":{"case_count":576,"partitions":list(PARTITIONS),"families":list(FAMILIES),"surfaces":list(SURFACES),"graph_sha256":core.sha(OUT/"material/frozen_concept_graph.json"),"behavior_sha256":core.sha(OUT/"material/frozen_behavior_cases.jsonl")},
  "models":list(MODELS),"model_order":list(MODELS),"precision":"bfloat16-no-quantization","batch_size":8,
  "zero_models":zeros,"executor_gate":{"case_ids":sent,"finite_fraction_min":1.0,"rank_agreement_min":1.0,"max_abs_diff_max":1e-6},
  "behavior_gate":{"accuracy_min":.90,"partition_min":.85,"surface_min":.85,"target_family_min":.85,"truth_cell_min":.85,"pairwise_gap_win_min":.95,"quartet_rank_min":.90,"median_relation_gap_min":1.0,"minimum_authorized_models":2},
  "hidden_gate":{"normalized_depths":[0,.25,.5,.75,1],"roles":["target_span_mean","tested_family_span_mean","boundary"],"storage":"selected complete vectors serialized float32 without fitted projection","numeric_relative_l2_p95_max":1e-5,"numeric_relative_l2_max":1e-4,"cross_surface_identity_win_min":.60,"permutation_gap_min":.10,"minimum_authorized_models":2},
  "branching":{"behavior_fail":"close C047 without hidden capture","behavior_pass":"run Phase1341 full-dimensional field; causal work remains separately unauthorized","hidden_fail":"close descriptive field","hidden_pass":"authorize only a separately frozen causal campaign"},
  "stop_rule":"After reveal do not change objects, material, thresholds, models, or nulls; execute only the frozen branch and close when it fails.",
  "parameter_boundary":"No natural-model parameter claim or intervention is authorized in C047."}
 protocol["contract_sha256"]=core.digest(protocol); protocol["authorization"]="run_phase1340_c047_paired_behavior"
 core.save(OUT/"protocol/preregistration.json",protocol)
 core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_gates_passed":True,"authorization":protocol["authorization"],"contract_sha256":protocol["contract_sha256"],"finished_at_utc":datetime.now(timezone.utc).isoformat()})
 print(json.dumps({"checks":checks,"contract_sha256":protocol["contract_sha256"],"authorization":protocol["authorization"]},indent=2))

if __name__=="__main__": main()
