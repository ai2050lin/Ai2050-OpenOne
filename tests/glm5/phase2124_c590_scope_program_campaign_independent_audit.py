#!/usr/bin/env python3
"""Independent artifact audit and next-route adjudication for C571-C589."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/"tests/glm5/result"
MAIN=ROOT/"tests/glm5/phase2105_c571_c589_scope_program_algebra_campaign.py"
WORKER=ROOT/"tests/glm5/phase2120_c586_cross_model_scope_program_worker.py"
VISUAL=ROOT/"frontend/public/vis_data/research_kernel/c589_scope_program_algebra_atlas.json"
OUT=RESULT/"phase2124_c590_scope_program_campaign_independent_audit"

PHASES={campaign:(2105+campaign-571,slug) for campaign,slug in (
    (571,"evidence_audit_and_scope_program_master_contract"),(572,"language_program_ontology_and_large_material_freeze"),(573,"compiler_semantic_balance_naturalness_and_qwen_behavior"),(574,"qwen_qualified_all_token_all_coordinate_capture"),(575,"full_field_observation_and_coordinate_response_atlas"),(576,"fixed_query_atomic_response_forward_prediction"),(577,"complete_voice_scope_factorial_decomposition"),(578,"translation_language_layout_factorial_decomposition"),(579,"behavior_qualified_path_depth_response"),(580,"discourse_voice_and_path_paraphrase_composition"),(581,"conditional_full_coordinate_system_identification"),(582,"bidirectional_response_equivalence_graph"),(583,"future_response_signature_and_predictive_state_quotient"),(584,"causal_eligibility_without_route_wide_stop"),(585,"qualified_local_state_guidance_or_registered_na"),(586,"sequential_cross_model_functional_topology"),(587,"nested_attitude_event_flagship"),(588,"recursive_knowledge_graph_flagship"),(589,"parameter_visualization_cleanup_and_campaign_synthesis"))}


def load(path:Path):return json.loads(path.read_text(encoding="utf-8"))
def save(path:Path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def phase_out(c:int)->Path:
    phase,slug=PHASES[c];return RESULT/f"phase{phase}_c{c}_{slug}"


def main()->None:
    checks={}
    finals={}
    current_main_hash=sha(MAIN)
    producer_hashes={}
    for campaign in PHASES:
        base=phase_out(campaign);final_path=base/"analysis/final.json";protocol=base/"protocol/preregistration.json";pre=base/"audit/internal_checks.json";post=base/"audit/internal_checks_post.json"
        checks[f"c{campaign}_final_exists"]=final_path.exists();checks[f"c{campaign}_protocol_exists"]=protocol.exists();checks[f"c{campaign}_precheck_exists"]=pre.exists();checks[f"c{campaign}_postcheck_exists"]=post.exists()
        if final_path.exists():
            value=load(final_path);finals[campaign]=value;checks[f"c{campaign}_closed"]=value.get("status")=="closed" and value.get("all_checks_passed") is True;checks[f"c{campaign}_phase"]=value.get("phase")==PHASES[campaign][0]
        if protocol.exists():
            recorded=load(protocol).get("producer_sha256","")
            producer_hashes[str(campaign)]=recorded
            checks[f"c{campaign}_producer_hash_recorded"]=bool(re.fullmatch(r"[0-9a-f]{64}",recorded))
    checks["phase_continuity"]=[PHASES[c][0] for c in PHASES]==list(range(2105,2124))
    checks["scripts_compile_sources_exist"]=MAIN.exists() and WORKER.exists()
    checks["visual_exists"]=VISUAL.exists() and VISUAL.stat().st_size>0
    if VISUAL.exists():
        atlas=load(VISUAL);checks["visual_schema"]=atlas.get("schema")=="ai2050.scope_program_algebra_atlas.v1";checks["visual_coordinates"]=atlas.get("coordinates")==2560;checks["visual_panels"]=set(atlas.get("panels",{}))=={"atomic","voice_scope","translation_layout","composition","nested_attitude","recursive_graph"}
    capture=phase_out(574)/"raw/qwen3_role_mean_states.float16.npy";shards=phase_out(574)/"raw/qwen3_full_token_shards";index=phase_out(574)/"raw/hidden_index.jsonl"
    checks["qwen_raw_retained"]=capture.exists() and capture.stat().st_size>0 and shards.exists() and any(shards.glob("*.npy"))
    checks["qwen_index_retained"]=index.exists() and index.stat().st_size>0
    synthesis=finals.get(589,{}).get("headline",{});candidate_count=len(synthesis.get("atomic_candidates",[]));composition=any(v.get("candidate",False) for v in synthesis.get("composition",{}).values());dynamic=synthesis.get("system_identification",{}).get("coordinate_affine",0)+synthesis.get("system_identification",{}).get("state_guarded_bilinear",0)>0
    causal=bool(synthesis.get("causal_families",[]));cross=bool(synthesis.get("cross_model_candidates",[]))
    same_exact_goal=bool(candidate_count and (composition or dynamic))
    route={
        "same_exact_goal":same_exact_goal,
        "completed_object":"scope-factorized response-family discovery on the frozen C571 material and model contract",
        "next_if_same":"fresh lexical and construction lockbox for the surviving response/composition/dynamic candidates, followed by causal specificity",
        "next_if_different":"return to material redesign for qualified fixed-query operations; do not reinterpret failed routes as missing mechanisms",
        "candidate_count":candidate_count,"composition_candidate":composition,"dynamic_candidate":dynamic,"causal_candidate":causal,"within_model_cross_candidates":cross,
        "foundational_math_authorized":False,
        "strict_boundary":"An internal artifact audit cannot independently replicate the scientific result. New foundational mathematics remains unauthorized without fresh prediction, composition, causal and cross-model closure.",
    }
    recovery_amendment={
        "status":"post_crash_storage_recovery_amendment",
        "historical_producer_hashes":producer_hashes,
        "current_main_sha256":current_main_hash,
        "changed_before_current_hash":[int(c) for c,h in producer_hashes.items() if h!=current_main_hash],
        "scientific_contract_change":False,
        "implementation_change":"C574 replaced an invalid 93.4 GB monolithic zero-fill with bounded 32-row shards after the process crashed before the first attributable sample/index commit.",
        "evidence_boundary":"This amendment is retrospective and does not convert the storage repair into a preregistered change. The mismatch remains an explicit provenance limitation.",
    }
    checks["post_crash_hash_lineage_explicit"]=set(recovery_amendment["changed_before_current_hash"])=={571,572,573,574}
    checks["stable_hash_from_c575_onward"]=all(producer_hashes.get(str(c))==current_main_hash for c in range(575,590))
    all_passed=all(checks.values());OUT.mkdir(parents=True,exist_ok=True)
    value={"phase":2124,"campaign":"C590","status":"closed","timestamp_utc":datetime.now(timezone.utc).isoformat(),"all_checks_passed":all_passed,"headline":{"status":"independent_artifact_audit_and_route_adjudication_closed","checks_passed":sum(checks.values()),"checks_total":len(checks),"route":route},"next_authorization":"C591_fresh_lockbox" if same_exact_goal else "new_separate_campaign_freeze"}
    save(OUT/"protocol/preregistration.json",{"phase":2124,"campaign":"C590","object":"independent artifact audit and exact-goal route adjudication","main_sha256":sha(MAIN),"worker_sha256":sha(WORKER)})
    save(OUT/"audit/checks.json",checks);save(OUT/"audit/post_crash_storage_recovery_amendment.json",recovery_amendment);save(OUT/"analysis/final.json",value);save(OUT/"analysis/route_adjudication.json",route)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    if not all_passed:raise SystemExit(1)


if __name__=="__main__":main()
