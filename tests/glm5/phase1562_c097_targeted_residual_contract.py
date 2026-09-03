#!/usr/bin/env python3
"""Phase1562: freeze the remaining-pair targeted postquery concrete residual arm."""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"
PARENT=RESULT/"phase1561_c097_analysis_adjudication_and_campaign_contract"; C091=RESULT/"phase1536_c091_human_validated_chinese_relation_contract"; C096=RESULT/"phase1557_c096_fresh_human_relation_field_contract"; OUT=RESULT/"phase1562_c097_targeted_residual_contract"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core; import phase1536_c091_human_validated_chinese_relation_contract as c091
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer
SEED="C097-A-targeted-postquery-concrete-v1"; FAMILIES=("similarity","class_inclusion"); PARTITIONS=("targeted_discovery","targeted_confirmation")
def digest(x): return hashlib.sha256((SEED+":"+core.canonical(x)).encode()).hexdigest()
def select_pairs():
    used={w for p in (C091/"material/frozen_pairs.jsonl",C096/"material/frozen_fresh_pairs.jsonl") for r in core.rows(p) for w in (r["source"],r["target"])}
    selected=[]; fresh=set(); frames=c091.source_frames()
    for family in ("class_inclusion","similarity"):
        key,relation=c091.DATASET_RELATIONS[family]; frame=frames[key]; candidates=[]
        for _,row in frame[(frame["relation"]==relation)&(frame["concreteness"]=="concrete")].iterrows():
            source,target=[v.strip() for v in row["word_pair"].split(":")]
            item={"family":family,"dataset_relation":relation,"source":source,"target":target,"concreteness":"concrete","type_ratings":int(row["type_ratings"])}
            if source!=target and source not in used and target not in used: candidates.append(item)
        chosen=[]
        for item in sorted(candidates,key=digest):
            if item["source"] in fresh or item["target"] in fresh: continue
            chosen.append(item); fresh.update((item["source"],item["target"]))
            if len(chosen)==14: break
        if len(chosen)!=14: raise RuntimeError((family,len(chosen)))
        for rank,item in enumerate(sorted(chosen,key=digest)):
            selected.append({**item,"pair_id":f"c097a-{FAMILIES.index(family)}-{rank:02d}","partition":PARTITIONS[rank//7],"partition_rank":rank%7})
    return sorted(selected,key=lambda r:(PARTITIONS.index(r["partition"]),FAMILIES.index(r["family"]),r["partition_rank"]))
def build_cases(pairs):
    cases=[]
    for pair in pairs:
        for qi,q in enumerate(FAMILIES):
            truth=pair["family"]==q; candidates=["是","否"] if (pair["partition_rank"]+qi)%2==0 else ["否","是"]
            case={"case_id":f"c097a-{len(cases):03d}","pair_id":pair["pair_id"],"pair_family":pair["family"],"query_family":q,"partition":pair["partition"],"partition_rank":pair["partition_rank"],"concreteness":"concrete","surface":"postquery","source":pair["source"],"target":pair["target"],"relation_anchor":c091.ANCHORS[q],"truth":truth,"gold_label":"是" if truth else "否","candidates":candidates}
            case["gold_position"]=candidates.index(case["gold_label"]); case["prompt"]=c091.prompt_for(pair,q,"postquery"); cases.append(case)
    return cases
def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1562 exists")
    pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json")
    if pf["authorization"]!="run_phase1562_c097_targeted_residual_contract" or not pa["all_checks_passed"]: raise RuntimeError("Phase1561 authorization missing")
    pairs=select_pairs(); cases=build_cases(pairs); compiled=c091.compile_cases(tokenizer(),cases)
    labels=[r["gold_label"] for r in cases]; words=[w for r in pairs for w in (r["source"],r["target"])]
    old={w for p in (C091/"material/frozen_pairs.jsonl",C096/"material/frozen_fresh_pairs.jsonl") for r in core.rows(p) for w in (r["source"],r["target"])}
    zeros={"always_yes":c091.balanced_accuracy(labels,["是"]*len(cases)),"always_no":c091.balanced_accuracy(labels,["否"]*len(cases)),"candidate_first":c091.balanced_accuracy(labels,[r["candidates"][0] for r in cases]),"pair_identity":c091.balanced_accuracy(labels,c091.majority_predictions(cases,"pair_id")),"query_identity":c091.balanced_accuracy(labels,c091.majority_predictions(cases,"query_family"))}
    counts=Counter((r["partition"],r["family"]) for r in pairs); maxlen=max(len(r["prompt_ids"]) for r in compiled)
    checks={"parent":True,"pairs":len(pairs)==28,"cases":len(cases)==56,"balanced":len(counts)==4 and set(counts.values())=={7},"fresh":not(set(words)&old),"unique":len(words)==len(set(words))==56,"concrete":all(r["concreteness"]=="concrete" for r in pairs),"postquery":all(r["surface"]=="postquery" for r in cases),"zero_models":all(v==0.5 for v in zeros.values()),"single_token_outputs":all(all(len(x)==1 for x in r["candidate_ids"]) for r in compiled),"roles":all(all(r["role_positions"][role] for role in c091.ROLES) for r in compiled),"fixed_shape":maxlen<256,"human_validated_source":True,"model_not_loaded":True}
    if not all(checks.values()): raise RuntimeError(checks)
    core.write_rows(OUT/"material/frozen_pairs.jsonl",pairs); core.write_rows(OUT/"material/active_cases.jsonl",cases); core.write_rows(OUT/"compiled/qwen3_active.jsonl",compiled)
    protocol={"phase":1562,"campaign":"C097-A","schema":"c097a.targeted_postquery_concrete_residual.v1","object":"whether the C096 0.738 postquery-concrete similarity/class floor reflects a stable conditional reversal or small-cell fragility","model":"Qwen3-4B local BF16 CUDA no quantization","families":list(FAMILIES),"partitions":list(PARTITIONS),"surface":"postquery","pairs":28,"cases":56,"pairs_per_family_partition":7,"source":"remaining unused 2026 human-validated Chinese pairs","material_hash":core.sha(OUT/"material/frozen_pairs.jsonl"),"compiled_hash":core.sha(OUT/"compiled/qwen3_active.jsonl"),"numeric_gate":{"repeat_hidden_max_abs":1e-6,"repeat_logit_max_abs":1e-6,"postquery_word_causal_max_abs":1e-6,"finite":True},"frozen_decisions":{"A1":"new pooled centroid cosine to each C091 and C096 pooled centroid >=0.75","A2":"fixed-seed pair-bootstrap 2.5% cosine to each old pooled centroid >=0.50","A3":"two seven-quartet split centroids cosine >=0.75"},"interpretation":{"A1_A2_pass":"finite-sample fragility is favored over a stable reversal; not proof of sampling-only origin","A1_fail":"persistent postquery-concrete residual boundary","A3_fail":"support remains split-sensitive even if pooled alignment passes"},"claim_boundary":"targeted task-scoped residual stability only; no semantic or causal mechanism","forbidden":["attention","MLP","parameters","PCA","learned probe","threshold mutation"],"created_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1563_c097_targeted_residual_capture"}
    protocol["contract_sha256"]=core.digest(protocol); core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"audit/pre_model_audit.json",{"phase":1562,"checks":checks,"zero_models":zeros,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}); core.save(OUT/"analysis/final.json",{"phase":1562,"campaign":"C097-A","status":"targeted_residual_contract_frozen","authorization":protocol["authorization"]}); print(json.dumps({"checks":checks,"protocol":protocol},ensure_ascii=False,indent=2))
if __name__=="__main__": main()

