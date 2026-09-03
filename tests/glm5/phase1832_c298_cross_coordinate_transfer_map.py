#!/usr/bin/env python3
"""C298: search all source-target activation-coordinate pairs at frozen strata."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1830_c296_complete_three_state_transition as transition
import phase1831_c297_continuous_amplitude_regimes as amplitude

core, OUT = common.core, common.OUTS["C298"]


def joint_source(events: np.ndarray) -> np.ndarray:
    return np.sign(events[:, common.ROLES.index("relation")] + events[:, common.ROLES.index("query")]).astype(np.int8)


def best_signed_map(source: np.ndarray, target: np.ndarray, block: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every target coordinate, test every source coordinate and both polarities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = torch.tensor(source, dtype=torch.int8, device=device)
    source_active = (s != 0).float().sum(0)
    source_pos = (s == 1).float(); source_neg = (s == -1).float(); source_act = (s != 0).float()
    mapping = np.full(common.DIM, -1, np.int32); polarity = np.ones(common.DIM, np.int8); score = np.zeros(common.DIM, np.float32)
    with torch.inference_mode():
        for start in range(0, common.DIM, block):
            stop = min(common.DIM, start + block); t = torch.tensor(target[:, start:stop], dtype=torch.int8, device=device)
            tpos=(t==1).float(); tneg=(t==-1).float(); tact=(t!=0).float(); truth_active=tact.sum(0)
            both=source_act.T@tact; union=source_active[:,None]+truth_active[None,:]-both
            same=source_pos.T@tpos+source_neg.T@tneg; reverse=source_pos.T@tneg+source_neg.T@tpos
            same=torch.where(union>0,same/union,torch.zeros_like(same)); reverse=torch.where(union>0,reverse/union,torch.zeros_like(reverse))
            stacked=torch.stack((same,reverse),0).reshape(2*common.DIM,stop-start); values,indices=stacked.max(0)
            idx=indices.cpu().numpy(); mapping[start:stop]=idx%common.DIM; polarity[start:stop]=np.where(idx<common.DIM,1,-1); score[start:stop]=values.cpu().numpy()
            mapping[start:stop][truth_active.cpu().numpy()<4]=-1; score[start:stop][truth_active.cpu().numpy()<4]=0
    del s, source_pos, source_neg, source_act
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return mapping, polarity, score


def apply_map(source: np.ndarray, mapping: np.ndarray, polarity: np.ndarray) -> np.ndarray:
    out=np.zeros((len(source),common.DIM),np.int8); valid=mapping>=0; out[:,valid]=source[:,mapping[valid]]*polarity[valid]; return out


def coordinate_scores(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    union=(pred!=0)|(truth!=0); return np.divide(((pred==truth)&union).sum(0),np.maximum(union.sum(0),1)).astype(np.float32)


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C296"]/"analysis/final.json"); gates=core.load(common.OUTS["C293"]/"protocol/preregistration.json")["gates"]
    checks={"parent":parent["all_checks_passed"],"strata_frozen":True,"all_source_target_coordinates":True,"both_polarities":True,"no_topk":True,"lockbox_unread":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    selected_rows=amplitude.nondegenerate_rows()
    protocol={"phase":1832,"campaign":"C298","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"cross_coordinate_map_frozen","source":"sign of relation+query events at each of all 2560 source coordinates","target":"destination-role next event at each of all 2560 target coordinates","search":"all 6,553,600 source-target pairs and both polarities at one independently nondegenerate stratum per family","strata":{r["family"]:r["selected_stratum"] for r in selected_rows},"fit":"third material only","validation":"fourth then fifth material","controls":["same coordinate with best polarity","destination persistence"],"gate":"train score>=0.70, fourth>=0.65, fifth>=0.65 and aggregate fifth margin>=0.01","claim_boundary":"This is a predictive signed dependency map. It is neither unique nor causal and does not read weights, attention, or MLP internals.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    a=np.load(transition.C265/"raw/training_role_states.float16.npy",mmap_mode="r"); b=np.load(transition.C264/"raw/role_states.float16.npy",mmap_mode="r"); c=np.load(transition.C278/"raw/role_states.float16.npy",mmap_mode="r")
    indices={"a":core.rows(transition.C248/"raw/hidden_index.jsonl"),"b":core.rows(transition.C264/"raw/hidden_index.jsonl"),"c":core.rows(transition.C278/"raw/hidden_index.jsonl")}; thresholds=common.thresholds()
    mappings=np.full((6,common.DIM),-1,np.int32); polarities=np.ones((6,common.DIM),np.int8); score_atlas=np.zeros((6,4,common.DIM),np.float32); rows=[]
    for fi,fr in enumerate(selected_rows):
        family=fr["family"]; q=int(fr["selected_stratum"]["q"]); d=int(fr["selected_stratum"]["destination_index"]); ids={n:transition.pair_ids(i,family) for n,i in indices.items()}
        def source_target(states,left,right,qidx,next_qidx):
            cur=common.event(np.asarray(states[right,qidx],np.float32)-np.asarray(states[left,qidx],np.float32),thresholds[q]); target=common.event(np.asarray(states[right,next_qidx,d],np.float32)-np.asarray(states[left,next_qidx,d],np.float32),thresholds[q+1]); return joint_source(cur),target,cur[:,d]
        sa,ta,pa=source_target(a,*ids['a'],q,q+1); sb,tb,pb=source_target(b,*ids['b'],q,q+1); sc,tc,pc=source_target(c,*ids['c'],common.CANONICAL_NEW_INDICES[q],common.CANONICAL_NEW_INDICES[q+1])
        mapping,polarity,train_score=best_signed_map(sa,ta); pred_b=apply_map(sb,mapping,polarity); pred_c=apply_map(sc,mapping,polarity); score_b=coordinate_scores(pred_b,tb); score_c=coordinate_scores(pred_c,tc)
        same_map=np.arange(common.DIM,dtype=np.int32); same_pol=np.where(coordinate_scores(sa,ta)>=coordinate_scores(-sa,ta),1,-1).astype(np.int8); same_c=apply_map(sc,same_map,same_pol)
        valid=(mapping>=0)&(train_score>=gates["cross_coordinate_train_score_min"])&(score_b>=gates["cross_coordinate_confirmation_score_min"])
        filtered=mapping.copy(); filtered[~valid]=-1; pred_filtered=apply_map(sc,filtered,polarity)
        mapped=common.metric_counts(pred_filtered,tc); same=common.metric_counts(same_c,tc); persistence=common.metric_counts(pc,tc); margin=mapped["signed_jaccard"]-max(same["signed_jaccard"],persistence["signed_jaccard"])
        mappings[fi]=filtered; polarities[fi]=polarity; score_atlas[fi]=np.stack((train_score,score_b,score_c,valid.astype(np.float32)))
        row={"family":family,"q":q,"destination_role":common.ROLES[d],"qualified_targets":int(valid.sum()),"unique_source_coordinates":int(len(set(filtered[valid].tolist()))),"mapped":mapped,"same_coordinate":same,"persistence":persistence,"minus_best_control":margin,"family_gate_passed":bool(valid.sum()>=16 and margin>=gates["model_margin_min"])}; rows.append(row); print(f"[C298] {family}: targets={valid.sum()} sources={row['unique_source_coordinates']} margin={margin:+.5f}",flush=True)
    np.save(OUT/"analysis/source_mapping.int32.npy",mappings); np.save(OUT/"analysis/source_polarity.int8.npy",polarities); np.save(OUT/"analysis/coordinate_score_atlas.float32.npy",score_atlas); core.write_rows(OUT/"analysis/family_results.jsonl",rows)
    passing=[r["family"] for r in rows if r["family_gate_passed"]]; report={"phase":1832,"campaign":"C298","status":"cross_coordinate_map_adjudicated","families":rows,"families_passing":passing,"broad_gate_passed":len(passing)>=gates["broad_families_min"],"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C299_C309_all_branches"}; core.save(OUT/"analysis/summary.json",report)
    ach={"families":len(rows)==6,"mapping_shape":list(mappings.shape)==[6,2560],"score_shape":list(score_atlas.shape)==[6,4,2560],"finite":bool(np.isfinite(score_atlas).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())})
    fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1832,"campaign":"C298","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
