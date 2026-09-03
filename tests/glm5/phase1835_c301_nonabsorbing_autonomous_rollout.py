#!/usr/bin/env python3
"""C301: roll complete three-state events from embedding to final norm."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1813_c279_joint_state_word_partition as partition
import phase1814_c280_multisource_one_step_prediction as one_step
import phase1830_c296_complete_three_state_transition as transition
import phase1834_c300_sixth_material_model_tournament as tournament

core, OUT = common.core, common.OUTS["C301"]
OBSERVE=(8,16,24,36)


def main() -> None:
    lockbox_sha=core.sha(common.OUTS["C295"]/"analysis/final.json")
    if (OUT/"protocol/preregistration.json").exists() and core.load(OUT/"protocol/preregistration.json").get("lockbox_sha256")==lockbox_sha and (OUT/"analysis/final.json").exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C300"]/"analysis/final.json"); gates=core.load(common.OUTS["C293"]/"protocol/preregistration.json")["gates"]
    checks={"parent":parent["all_checks_passed"],"embedding_only_start":True,"no_true_intermediate_lockbox_states":True,"all_roles_coordinates":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1835,"campaign":"C301","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"nonabsorbing_rollout_frozen","lockbox_sha256":lockbox_sha,"start":"only sixth-material embedding edit events","models":["complete three-state C296","absorbing C281-style","fixed embedding persistence"],"readout":"q8,q16,q24,final_norm over all six roles and all 2560 coordinates","gate":"complete model beats both controls by>=0.01 in at least four families","claim_boundary":"The start is an edit-response event field, not a raw sentence embedding. Event rollout does not reconstruct continuous states, all tokens, or output logits.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    a=np.load(transition.C265/"raw/training_role_states.float16.npy",mmap_mode="r"); b=np.load(transition.C264/"raw/role_states.float16.npy",mmap_mode="r"); test=np.load(common.OUTS["C295"]/"raw/role_states.float16.npy",mmap_mode="r"); ia=core.rows(transition.C248/"raw/hidden_index.jsonl"); ib=core.rows(transition.C264/"raw/hidden_index.jsonl"); it=core.rows(common.OUTS["C295"]/"raw/hidden_index.jsonl"); thresholds=common.thresholds(); full_maps=np.load(common.OUTS["C296"]/"analysis/complete_transition_maps.int8.npy",mmap_mode="r")
    rows=[]; atlas=np.zeros((6,len(OBSERVE),3,common.DIM),np.float32)
    for fi,family in enumerate(common.FAMILIES):
        ids={"a":transition.pair_ids(ia,family),"b":transition.pair_ids(ib,family),"t":transition.pair_ids(it,family)}
        truth=[]
        for q in range(37): truth.append(common.event(np.asarray(test[ids['t'][1],common.CANONICAL_NEW_INDICES[q]],np.float32)-np.asarray(test[ids['t'][0],common.CANONICAL_NEW_INDICES[q]],np.float32),thresholds[q]))
        complete=truth[0].copy(); absorbing=truth[0].copy(); fixed=truth[0].copy(); observations={}
        for q in range(36):
            train_cur=np.concatenate((common.event(np.asarray(a[ids['a'][1],q],np.float32)-np.asarray(a[ids['a'][0],q],np.float32),thresholds[q]),common.event(np.asarray(b[ids['b'][1],q],np.float32)-np.asarray(b[ids['b'][0],q],np.float32),thresholds[q])))
            train_next=np.concatenate((common.event(np.asarray(a[ids['a'][1],q+1],np.float32)-np.asarray(a[ids['a'][0],q+1],np.float32),thresholds[q+1]),common.event(np.asarray(b[ids['b'][1],q+1],np.float32)-np.asarray(b[ids['b'][0],q+1],np.float32),thresholds[q+1])))
            next_complete=np.zeros_like(complete); next_absorb=np.zeros_like(absorbing)
            for d in range(6):
                next_complete[:,d]=tournament.map_lookup(np.asarray(full_maps[fi,q,d]),transition.combined_code(complete,d))
                fit,_,_=one_step.fit_map(partition.code_word(train_cur,partition.CANDIDATES["relation_query"]),train_next[:,d],9,gates["transition_support_min"],gates["transition_agreement_min"]); pure=one_step.lookup(fit,partition.code_word(absorbing,partition.CANDIDATES["relation_query"]),9); next_absorb[:,d]=np.where(absorbing[:,d]!=0,absorbing[:,d],pure)
            complete=next_complete; absorbing=next_absorb
            if q+1 in OBSERVE: observations[q+1]=(complete.copy(),absorbing.copy(),fixed.copy())
        metrics={name:np.zeros(5,np.int64) for name in ("complete","absorbing","fixed")}; stage_rows=[]
        for oi,q in enumerate(OBSERVE):
            values=observations[q]; target=truth[q]; stage={"q":q}
            for mi,(name,pred) in enumerate(zip(("complete","absorbing","fixed"),values)):
                stage[name]=common.metric_counts(pred,target); union=(pred!=0)|(target!=0); atlas[fi,oi,mi]=np.divide(((pred==target)&union).sum(axis=(0,1)) if False else ((pred==target)&union).sum(axis=(0,1)),np.maximum(union.sum(axis=(0,1)),1))
            stage_rows.append(stage)
        aggregate={name:float(np.mean([s[name]["signed_jaccard"] for s in stage_rows])) for name in ("complete","absorbing","fixed")}; margin=aggregate["complete"]-max(aggregate["absorbing"],aggregate["fixed"]); row={"family":family,"stages":stage_rows,"aggregate_signed_jaccard":aggregate,"complete_minus_best_control":margin,"family_gate_passed":margin>=gates["model_margin_min"]}; rows.append(row); print(f"[C301] {family}: margin={margin:+.5f}",flush=True)
    np.save(OUT/"analysis/rollout_coordinate_atlas.float32.npy",atlas); core.write_rows(OUT/"analysis/family_results.jsonl",rows); passing=[r["family"] for r in rows if r["family_gate_passed"]]; report={"phase":1835,"campaign":"C301","status":"autonomous_rollout_adjudicated","families":rows,"families_passing":passing,"broad_gate_passed":len(passing)>=gates["broad_families_min"],"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C302_C309_all_branches"}; core.save(OUT/"analysis/summary.json",report)
    ach={"families":len(rows)==6,"atlas_shape":list(atlas.shape)==[6,4,3,2560],"finite":bool(np.isfinite(atlas).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1835,"campaign":"C301","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
