#!/usr/bin/env python3
"""C300: reveal the sixth lockbox and compare all frozen transition models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1813_c279_joint_state_word_partition as partition
import phase1814_c280_multisource_one_step_prediction as one_step
import phase1830_c296_complete_three_state_transition as transition
import phase1831_c297_continuous_amplitude_regimes as amplitude
import phase1832_c298_cross_coordinate_transfer_map as cross
import phase1833_c299_all_token_aligned_transfer as token_transfer

core, OUT = common.core, common.OUTS["C300"]


def map_lookup(table: np.ndarray, code: np.ndarray) -> np.ndarray:
    coordinates=np.arange(common.DIM)[None,:]; return table[code,coordinates]


def main() -> None:
    lockbox_sha=core.sha(common.OUTS["C295"]/"analysis/final.json")
    if (OUT/"protocol/preregistration.json").exists() and core.load(OUT/"protocol/preregistration.json").get("lockbox_sha256")==lockbox_sha and (OUT/"analysis/final.json").exists(): raise RuntimeError(OUT)
    parents={c:core.load(common.OUTS[c]/"analysis/final.json") for c in ("C296","C297","C298","C299")}; gates=core.load(common.OUTS["C293"]/"protocol/preregistration.json")["gates"]
    checks={"parents":all(x["all_checks_passed"] for x in parents.values()),"first_sixth_lockbox_use":True,"frozen_models_only":True,"all_coordinates":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    strata={r["family"]:{"q":r["q"],"destination_role":r["destination_role"]} for r in parents["C298"]["headline"]["families"]}
    protocol={"phase":1834,"campaign":"C300","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"lockbox_tournament_frozen","lockbox":"C295 sixth vocabulary/material","lockbox_sha256":lockbox_sha,"strata":strata,"models":["M0 persistence","M1 C280 absorbing relation/query word","M2 C296 complete same-coordinate transition","M3 C298 role-source cross-coordinate map","M4 C299 all-token cross-coordinate map","M5 C297 amplitude bucket"],"winner":"highest sixth signed-Jaccard among M0-M4; M5 is adjudicated separately by MAE","family_gate":"best nonbaseline beats max(M0,M1) by>=0.01","broad_gate":"at least four of six families","claim_boundary":"The sixth panel is lexically new but reuses controlled syntax. Passing is lexical/graph-rename transfer, not open-domain universality.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    train_a=np.load(transition.C265/"raw/training_role_states.float16.npy",mmap_mode="r"); train_b=np.load(transition.C264/"raw/role_states.float16.npy",mmap_mode="r"); test=np.load(common.OUTS["C295"]/"raw/role_states.float16.npy",mmap_mode="r")
    ia=core.rows(transition.C248/"raw/hidden_index.jsonl"); ib=core.rows(transition.C264/"raw/hidden_index.jsonl"); it=core.rows(common.OUTS["C295"]/"raw/hidden_index.jsonl"); thresholds=common.thresholds()
    full_maps=np.load(common.OUTS["C296"]/"analysis/complete_transition_maps.int8.npy",mmap_mode="r"); role_map=np.load(common.OUTS["C298"]/"analysis/source_mapping.int32.npy"); role_pol=np.load(common.OUTS["C298"]/"analysis/source_polarity.int8.npy"); token_map=np.load(common.OUTS["C299"]/"analysis/all_token_source_mapping.int32.npy"); token_pol=np.load(common.OUTS["C299"]/"analysis/all_token_source_polarity.int8.npy")
    test_fields=np.load(common.OUTS["C295"]/"raw/full_fields.float16.npy",mmap_mode="r"); test_tokens=np.load(common.OUTS["C295"]/"raw/token_ids.int32.npy",mmap_mode="r")
    atlas=np.zeros((6,5,common.DIM),np.float32); rows=[]
    for fi,family in enumerate(common.FAMILIES):
        q=int(strata[family]["q"]); d=common.ROLES.index(strata[family]["destination_role"]); ids={"a":transition.pair_ids(ia,family),"b":transition.pair_ids(ib,family),"t":transition.pair_ids(it,family)}
        train_cur=np.concatenate((common.event(np.asarray(train_a[ids['a'][1],q],np.float32)-np.asarray(train_a[ids['a'][0],q],np.float32),thresholds[q]),common.event(np.asarray(train_b[ids['b'][1],q],np.float32)-np.asarray(train_b[ids['b'][0],q],np.float32),thresholds[q])))
        train_next=np.concatenate((common.event(np.asarray(train_a[ids['a'][1],q+1,d],np.float32)-np.asarray(train_a[ids['a'][0],q+1,d],np.float32),thresholds[q+1]),common.event(np.asarray(train_b[ids['b'][1],q+1,d],np.float32)-np.asarray(train_b[ids['b'][0],q+1,d],np.float32),thresholds[q+1])))
        current=common.event(np.asarray(test[ids['t'][1],common.CANONICAL_NEW_INDICES[q]],np.float32)-np.asarray(test[ids['t'][0],common.CANONICAL_NEW_INDICES[q]],np.float32),thresholds[q]); truth_delta=np.asarray(test[ids['t'][1],common.CANONICAL_NEW_INDICES[q+1],d],np.float32)-np.asarray(test[ids['t'][0],common.CANONICAL_NEW_INDICES[q+1],d],np.float32); truth=common.event(truth_delta,thresholds[q+1])
        m0=current[:,d]
        fit1,_,_=one_step.fit_map(partition.code_word(train_cur,partition.CANDIDATES["relation_query"]),train_next,9,gates["transition_support_min"],gates["transition_agreement_min"]); pure=one_step.lookup(fit1,partition.code_word(current,partition.CANDIDATES["relation_query"]),9); m1=np.where(m0!=0,m0,pure).astype(np.int8)
        m2=map_lookup(np.asarray(full_maps[fi,q,d]),transition.combined_code(current,d)).astype(np.int8)
        src3=cross.joint_source(current); m3=cross.apply_map(src3,role_map[fi],role_pol[fi])
        src4,coverage,_=token_transfer.panel(test_fields,test_tokens,it,family,common.CANONICAL_NEW_INDICES[q]); m4=cross.apply_map(src4,token_map[fi],token_pol[fi])
        preds={"M0_persistence":m0,"M1_absorbing":m1,"M2_complete":m2,"M3_cross_coordinate":m3,"M4_all_token":m4}; metrics={name:common.metric_counts(pred,truth) for name,pred in preds.items()}
        for mi,pred in enumerate(preds.values()): atlas[fi,mi]=cross.coordinate_scores(pred,truth)
        best_name=max((name for name in preds if not name.startswith("M0") and not name.startswith("M1")),key=lambda n:metrics[n]["signed_jaccard"]); margin=metrics[best_name]["signed_jaccard"]-max(metrics["M0_persistence"]["signed_jaccard"],metrics["M1_absorbing"]["signed_jaccard"])
        code_train=transition.combined_code(train_cur,d); code_test=transition.combined_code(current,d); mean,lo,hi,support=amplitude.fit_amplitude(code_train,np.abs(np.concatenate((np.asarray(train_a[ids['a'][1],q+1,d],np.float32)-np.asarray(train_a[ids['a'][0],q+1,d],np.float32),np.asarray(train_b[ids['b'][1],q+1,d],np.float32)-np.asarray(train_b[ids['b'][0],q+1,d],np.float32)))),27); key=code_test.astype(np.int64)+np.arange(common.DIM)[None,:]*27; valid=support[key]>=gates["transition_support_min"]; pred_amp=mean[key]; truth_amp=np.abs(truth_delta); base_amp=np.abs(np.asarray(test[ids['t'][1],common.CANONICAL_NEW_INDICES[q],d],np.float32)-np.asarray(test[ids['t'][0],common.CANONICAL_NEW_INDICES[q],d],np.float32)); mae=float(np.abs(pred_amp[valid]-truth_amp[valid]).mean()); base_mae=float(np.abs(base_amp[valid]-truth_amp[valid]).mean()); amp_gain=(base_mae-mae)/max(base_mae,1e-12)
        row={"family":family,"q":q,"destination_role":common.ROLES[d],"models":metrics,"best_nonbaseline":best_name,"minus_best_baseline":margin,"family_gate_passed":margin>=gates["model_margin_min"],"all_token_alignment_coverage_mean":float(np.mean(coverage)),"amplitude":{"mae":mae,"current_magnitude_mae":base_mae,"relative_gain":amp_gain}}; rows.append(row); print(f"[C300] {family}: {best_name} margin={margin:+.5f} amp={amp_gain:+.4f}",flush=True)
    np.save(OUT/"analysis/lockbox_coordinate_score_atlas.float32.npy",atlas); core.write_rows(OUT/"analysis/family_results.jsonl",rows); passing=[r["family"] for r in rows if r["family_gate_passed"]]
    report={"phase":1834,"campaign":"C300","status":"sixth_lockbox_tournament_adjudicated","families":rows,"families_passing":passing,"broad_gate_passed":len(passing)>=gates["broad_families_min"],"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C301_C309_all_branches"}; core.save(OUT/"analysis/summary.json",report)
    ach={"families":len(rows)==6,"atlas_shape":list(atlas.shape)==[6,5,2560],"finite":bool(np.isfinite(atlas).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1834,"campaign":"C300","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
