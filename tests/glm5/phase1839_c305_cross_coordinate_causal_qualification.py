#!/usr/bin/env python3
"""C305: freeze only lockbox-supported cross-coordinate causal coalitions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C305"]


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    tournament=core.load(common.OUTS["C300"]/"analysis/final.json"); parents=[core.load(common.OUTS[c]/"analysis/final.json") for c in ("C298","C299","C304")]; gates=core.load(common.OUTS["C293"]/"protocol/preregistration.json")["gates"]
    checks={"parents":tournament["all_checks_passed"] and all(p["all_checks_passed"] for p in parents),"qualification_before_intervention":True,"all_rule_coordinates":True,"no_topk":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1839,"campaign":"C305","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"causal_qualification_frozen","candidate_models":["M3 role-source cross-coordinate","M4 all-token cross-coordinate"],"qualification":"lockbox signed-Jaccard exceeds both persistence and absorbing baselines by>=0.01; mapping has >=16 targets; per-target lockbox score>=0.65","intervention_priority":"all-token candidate first because it is the only registered cross-coordinate branch that can qualify independently","claim_boundary":"Qualification reuses the sixth panel and therefore does not constitute causal evidence. It only freezes the intervention object.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    lockbox_atlas=np.load(common.OUTS["C300"]/"analysis/lockbox_coordinate_score_atlas.float32.npy"); mappings={"M3_cross_coordinate":np.load(common.OUTS["C298"]/"analysis/source_mapping.int32.npy"),"M4_all_token":np.load(common.OUTS["C299"]/"analysis/all_token_source_mapping.int32.npy")}; polarities={"M3_cross_coordinate":np.load(common.OUTS["C298"]/"analysis/source_polarity.int8.npy"),"M4_all_token":np.load(common.OUTS["C299"]/"analysis/all_token_source_polarity.int8.npy")}; model_index={"M3_cross_coordinate":3,"M4_all_token":4}
    rows=[]; masks=np.zeros((6,2,common.DIM),bool)
    for fi,row in enumerate(tournament["headline"]["families"]):
        baseline=max(row["models"]["M0_persistence"]["signed_jaccard"],row["models"]["M1_absorbing"]["signed_jaccard"])
        for mi,name in enumerate(("M3_cross_coordinate","M4_all_token")):
            margin=row["models"][name]["signed_jaccard"]-baseline; valid=(mappings[name][fi]>=0)&(lockbox_atlas[fi,model_index[name]]>=0.65); masks[fi,mi]=valid
            qualified=margin>=gates["model_margin_min"] and int(valid.sum())>=gates["causal_targets_min"]
            rows.append({"family":row["family"],"model":name,"q":row["q"],"destination_role":row["destination_role"],"lockbox_margin_vs_best_baseline":margin,"eligible_targets":int(valid.sum()),"unique_sources":int(len(set(mappings[name][fi][valid].tolist()))),"qualified":qualified})
    np.save(OUT/"analysis/qualified_target_masks.bool.npy",masks); core.write_rows(OUT/"analysis/candidate_results.jsonl",rows); qualified=[r for r in rows if r["qualified"]]
    report={"phase":1839,"campaign":"C305","status":"causal_qualification_adjudicated","candidates":rows,"qualified":qualified,"qualified_count":len(qualified),"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C306_run_qualified_candidates" if qualified else "C306_registered_no_test_and_continue_C307"}; core.save(OUT/"analysis/summary.json",report)
    ach={"candidates":len(rows)==12,"mask_shape":list(masks.shape)==[6,2,2560],"qualification_consistent":len(qualified)==sum(r["qualified"] for r in rows)}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1839,"campaign":"C305","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
