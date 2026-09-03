#!/usr/bin/env python3
"""Phase1565: freeze the independent English WordNet C097-B relation arm."""
from __future__ import annotations
import json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; PARENT=RESULT/"phase1564_c097_targeted_residual_adjudication"; C089=RESULT/"phase1521_c089_natural_relation_observation_contract"; OUT=RESULT/"phase1565_c097_wordnet_independent_contract"
DATA=ROOT/"tests/gpt5/result/phase602_three_track_semantics/source/WordNet-3.0/dict/data.noun"; INDEX=DATA.with_name("index.noun")
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core; import phase1521_c089_natural_relation_observation_contract as c089
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer
FAMILIES=("similarity","class_inclusion","whole_part"); MAP={"synonym":"similarity","kind_of":"class_inclusion","part_of":"whole_part"}; PARTITIONS=("response_discovery","confirmation","lockbox"); SURFACES=("prequery","postquery"); ROLES=("source_word","target_word","relation_anchor","boundary")
SYSTEM='Judge the stated lexical relation between two English nouns. "First" and "second" refer to written order. Answer exactly yes or no.'
ANCHORS={"similarity":"the two nouns can name the same kind of thing with the same or nearly the same meaning","class_inclusion":"the first noun names a kind of the second noun","whole_part":"the first noun names a part of the second noun"}
def pairs_from_c089():
 groups=core.rows(C089/"material/relation_composition_sets.jsonl"); pairs=[]
 for group in groups:
  family=MAP[group["family"]]
  for member in ("pair_a","pair_b"):
   row=group[member]; pairs.append({"pair_id":f"c097b-{family}-{len(pairs):03d}","source":row["source"],"target":row["target"],"source_offset":row["source_offset"],"target_offset":row["target_offset"],"family":family,"wordnet_family":group["family"],"partition":group["partition"],"source_set_id":group["set_id"]})
 return sorted(pairs,key=lambda r:(PARTITIONS.index(r["partition"]),FAMILIES.index(r["family"]),r["source_set_id"],r["pair_id"]))
def prompt(pair,q,surface):
 nouns=f'First noun: "{pair["source"]}". Second noun: "{pair["target"]}".'; claim=ANCHORS[q]
 return f"Claim to test: {claim}. {nouns} Is the claim true? Reply with exactly yes or no." if surface=="prequery" else f"{nouns} Determine whether this claim is true: {claim}. Reply with exactly yes or no."
def build_cases(pairs):
 cases=[]
 for pair in pairs:
  for si,surface in enumerate(SURFACES):
   for qi,q in enumerate(FAMILIES):
    truth=pair["family"]==q; candidates=["yes","no"] if (len(cases)+si+qi)%2==0 else ["no","yes"]
    cases.append({"case_id":f"c097b-{len(cases):04d}","pair_id":pair["pair_id"],"pair_family":pair["family"],"query_family":q,"partition":pair["partition"],"surface":surface,"source":pair["source"],"target":pair["target"],"relation_anchor":ANCHORS[q],"truth":truth,"gold_label":"yes" if truth else "no","candidates":candidates,"gold_position":candidates.index("yes" if truth else "no"),"prompt":prompt(pair,q,surface)})
 return cases
def compile_cases(tok,cases):
 compiled=[]
 for case in cases:
  ids=core.chat_ids(tok,SYSTEM,case["prompt"]); positions={}
  for role,value in (("source_word",case["source"]),("target_word",case["target"]),("relation_anchor",case["relation_anchor"])):
   spans=c089.all_spans(tok,ids,value)
   if len(spans)!=1: raise RuntimeError((case["case_id"],role,value,spans))
   positions[role]=spans[0]
  positions["boundary"]=[len(ids)-1]; compiled.append({**case,"prompt_ids":ids,"role_positions":positions,"candidate_ids":[[int(t) for t in tok.encode(v,add_special_tokens=False)] for v in case["candidates"]]})
 return compiled
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1565 exists")
 pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json"); c089a=core.load(C089/"audit/independent_final_audit.json")
 if pf["authorization"]!="run_phase1565_c097_wordnet_independent_contract" or not pa["all_checks_passed"] or not c089a["all_checks_passed"]: raise RuntimeError("authorization/source missing")
 pairs=pairs_from_c089(); cases=build_cases(pairs); compiled=compile_cases(tokenizer(),cases); labels=[r["gold_label"] for r in cases]; words=[w for r in pairs for w in (r["source"],r["target"])]; counts=Counter((r["partition"],r["family"]) for r in pairs)
 zeros={"always_yes":c089.balanced_accuracy(labels,["yes"]*len(cases)),"always_no":c089.balanced_accuracy(labels,["no"]*len(cases)),"candidate_first":c089.balanced_accuracy(labels,[r["candidates"][0] for r in cases]),"pair_identity":c089.balanced_accuracy(labels,c089.majority_predictions(cases,"pair_id")),"query_identity":c089.balanced_accuracy(labels,c089.majority_predictions(cases,"query_family")),"surface_identity":c089.balanced_accuracy(labels,c089.majority_predictions(cases,"surface"))}
 checks={"parent":True,"source_audited_before_C097":True,"pairs":len(pairs)==90,"cases":len(cases)==540,"balance":len(counts)==9 and set(counts.values())=={10},"global_lexical_uniqueness":len(words)==len(set(words))==180,"wordnet_offsets":all(r["source_offset"] and r["target_offset"] for r in pairs),"zero_models":all(v==.5 for v in zeros.values()),"single_token_outputs":all(all(len(x)==1 for x in r["candidate_ids"]) for r in compiled),"roles":all(all(r["role_positions"][role] for role in ROLES) for r in compiled),"fixed_shape":max(len(r["prompt_ids"]) for r in compiled)<256,"lexicographer_curated":True,"independent_blind_naturalness_missing":True,"model_not_loaded":True}
 if not all(checks.values()): raise RuntimeError(checks)
 core.write_rows(OUT/"material/frozen_wordnet_pairs.jsonl",pairs); core.write_rows(OUT/"material/active_cases.jsonl",cases); core.write_rows(OUT/"compiled/qwen3_active.jsonl",compiled)
 protocol={"phase":1565,"campaign":"C097-B","schema":"c097b.independent_wordnet_triadic_field.v1","object":"independent-source and English-surface replication of identifiable contrast mean plus residual geometry","model":"Qwen3-4B local BF16 CUDA no quantization","system":SYSTEM,"families":list(FAMILIES),"partitions":list(PARTITIONS),"surfaces":list(SURFACES),"roles":list(ROLES),"material":{"source":"WordNet 3.0 noun relations pre-frozen in C089","pairs":90,"cases":540,"data_sha256":core.sha(DATA),"index_sha256":core.sha(INDEX),"source_selection_sha256":core.sha(C089/"material/relation_composition_sets.jsonl"),"pairs_sha256":core.sha(OUT/"material/frozen_wordnet_pairs.jsonl"),"zero_chinese_lexical_overlap_by_script":"different writing systems; not an unseen-training claim","naturalness":"lexicographer-curated monosemous lexical relations; no new independent blind rater"},"numeric_gate":{"repeat_hidden_max_abs":1e-6,"repeat_logit_max_abs":1e-6,"postquery_word_causal_max_abs":1e-6,"finite":True},"behavior_policy":"report query-family behavior continuously; failure adds M_BEHAVIOR but does not erase numeric observation","frozen_decisions":{"B1":"median focus-panel exact common-energy fraction >=0.50","B2":"minimum cross-partition G_C cosine >=0.50","B3":"median English-to-Chinese C096 G_C cosine >=0.50","B4":"every discovery-reference top64 restricted cosine and sign agreement exceeds its 1000-permutation 99th percentile"},"uncertainty":"2000 synchronized rank bootstraps per partition/surface/state preserve dependence among the three shared-cell contrasts","random_support":"coordinate-permutation null for discovery-reference top64","claim_boundary":{"allowed":"Qwen3 cross-source/language-surface contrast geometry with behavior and naturalness missingness typed","forbidden":["universal semantic comparator","semantic neurons","causal mechanism","training-unseen proof","new mathematics"]},"forbidden":["attention","MLP","parameters","gradients","PCA","learned probe","threshold mutation"],"created_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1566_c097_wordnet_capture"}; protocol["contract_sha256"]=core.digest(protocol)
 core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json",{"phase":1565,"checks":checks,"zero_models":zeros,"missingness":["M_HUMAN_NATURALNESS","M_TRAINING_UNSEEN"],"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}); core.save(OUT/"analysis/final.json",{"phase":1565,"campaign":"C097-B","status":"independent_wordnet_contract_frozen","authorization":protocol["authorization"]}); print(json.dumps({"checks":checks,"protocol":protocol},ensure_ascii=False,indent=2))
if __name__=="__main__": main()

