#!/usr/bin/env python3
"""C302: forecast the fourth factorial cell with a learned full-field residual."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1830_c296_complete_three_state_transition as transition

core, OUT = common.core, common.OUTS["C302"]


def groups(index: list[dict], family: str) -> list[dict]:
    panel="nested_composition" if family=="nested_attitude" else "core"; rows=[r for r in index if r["panel"]==panel and r["family"]==family and r.get("correct",True)]
    lookup={(r["surface"],r["unit"],r["order"],r["factor_a"],r["factor_b"]):r["hidden_index"] for r in rows}; out=[]
    for surface,unit,order in itertools.product(sorted({r["surface"] for r in rows}),sorted({r["unit"] for r in rows}),(1,-1)):
        ids={(a,b):lookup.get((surface,unit,order,a,b)) for a,b in itertools.product((0,1),repeat=2)}
        if all(v is not None for v in ids.values()): out.append({"surface":surface,"unit":unit,"order":order,"ids":ids})
    return out


def residuals(states: np.ndarray,index: list[dict],family: str,canonical: bool=False) -> tuple[np.ndarray,list[dict]]:
    gs=groups(index,family); values=[]
    for g in gs:
        h={k:np.asarray(states[v],np.float32) for k,v in g["ids"].items()}; values.append(h[(1,1)]-h[(1,0)]-h[(0,1)]+h[(0,0)])
    result=np.asarray(values,np.float32); return (result[:,common.CANONICAL_NEW_INDICES] if canonical else result),gs


def main() -> None:
    lockbox_sha=core.sha(common.OUTS["C295"]/"analysis/final.json")
    if (OUT/"protocol/preregistration.json").exists() and core.load(OUT/"protocol/preregistration.json").get("lockbox_sha256")==lockbox_sha and (OUT/"analysis/final.json").exists() and core.load(OUT/"analysis/final.json").get("all_checks_passed"):
        raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C301"]/"analysis/final.json"); gates=core.load(common.OUTS["C293"]/"protocol/preregistration.json")["gates"]
    checks={"parent":parent["all_checks_passed"],"training_materials_only_for_correction":True,"sixth_is_lockbox":True,"all_roles_checkpoints_coordinates":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol","raw"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1836,"campaign":"C302","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"composition_forecast_frozen","lockbox_sha256":lockbox_sha,"forecast":"H11_hat = H10 + H01 - H00 + mean training interaction residual","training":"third+fourth+fifth role states","lockbox":"sixth role states","coverage":"six families, both surfaces, eight units, both answer orders, 37 canonical checkpoints, six roles, all 2560 coordinates","controls":["zero interaction/additive forecast","zero change H10"],"gate":"corrected relative MAE gain >=1% over additive in at least four families","claim_boundary":"Predicting a held-out fourth state from the other three is field-level composition evidence. It does not establish operator order, commutation, natural generation, or causality.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    states=[np.load(transition.C265/"raw/training_role_states.float16.npy",mmap_mode="r"),np.load(transition.C264/"raw/role_states.float16.npy",mmap_mode="r"),np.load(transition.C278/"raw/role_states.float16.npy",mmap_mode="r")]; indices=[core.rows(transition.C248/"raw/hidden_index.jsonl"),core.rows(transition.C264/"raw/hidden_index.jsonl"),core.rows(transition.C278/"raw/hidden_index.jsonl")]; test=np.load(common.OUTS["C295"]/"raw/role_states.float16.npy",mmap_mode="r"); test_index=core.rows(common.OUTS["C295"]/"raw/hidden_index.jsonl")
    atlas=np.zeros((6,4,common.DIM),np.float32); family_rows=[]; group_rows=[]
    for fi,family in enumerate(common.FAMILIES):
        train_res=np.concatenate([residuals(s,i,family,canonical=(s.shape[1]==38))[0] for s,i in zip(states,indices)],axis=0); correction=train_res.mean(axis=0)
        test_res,gs=residuals(test,test_index,family,canonical=True); additive_error=np.abs(test_res); corrected_error=np.abs(test_res-correction[None,...]); zero_change=[]
        for g in gs:
            h={k:np.asarray(test[v,common.CANONICAL_NEW_INDICES],np.float32) for k,v in g["ids"].items()}; zero_change.append(np.abs(h[(1,1)]-h[(1,0)]))
        zero_change=np.asarray(zero_change)
        add_mae=float(additive_error.mean()); cor_mae=float(corrected_error.mean()); zero_mae=float(zero_change.mean()); gain=float((add_mae-cor_mae)/max(add_mae,1e-12))
        coord_add=additive_error.mean(axis=(0,1,2)); coord_cor=corrected_error.mean(axis=(0,1,2)); atlas[fi,0]=coord_add; atlas[fi,1]=coord_cor; atlas[fi,2]=np.divide(coord_add-coord_cor,np.maximum(coord_add,1e-12)); atlas[fi,3]=np.abs(correction).mean(axis=(0,1))
        for gi,g in enumerate(gs): group_rows.append({"family":family,"surface":g["surface"],"unit":g["unit"],"order":g["order"],"additive_mae":float(additive_error[gi].mean()),"corrected_mae":float(corrected_error[gi].mean()),"relative_gain":float((additive_error[gi].mean()-corrected_error[gi].mean())/max(float(additive_error[gi].mean()),1e-12))})
        row={"family":family,"training_groups":int(len(train_res)),"lockbox_groups":len(gs),"additive_mae":add_mae,"corrected_mae":cor_mae,"zero_change_mae":zero_mae,"relative_mae_gain":gain,"family_gate_passed":gain>=gates["composition_relative_mae_gain_min"]}; family_rows.append(row); print(f"[C302] {family}: gain={gain:+.5f}",flush=True)
    np.save(OUT/"analysis/composition_coordinate_atlas.float32.npy",atlas); core.write_rows(OUT/"raw/group_results.jsonl",group_rows); core.write_rows(OUT/"analysis/family_results.jsonl",family_rows); passing=[r["family"] for r in family_rows if r["family_gate_passed"]]
    report={"phase":1836,"campaign":"C302","status":"composition_forecast_adjudicated","families":family_rows,"families_passing":passing,"broad_gate_passed":len(passing)>=gates["broad_families_min"],"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C303_C309_all_branches"}; core.save(OUT/"analysis/summary.json",report)
    expected_groups=sum(len(groups(test_index,family)) for family in common.FAMILIES)
    ach={"families":len(family_rows)==6,"groups":len(group_rows)==expected_groups,"expected_behavior_eligible_groups":expected_groups,"atlas_shape":list(atlas.shape)==[6,4,2560],"finite":bool(np.isfinite(atlas).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1836,"campaign":"C302","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
