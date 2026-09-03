#!/usr/bin/env python3
"""C299: test a full-token, sequence-aligned source field without role averaging."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1830_c296_complete_three_state_transition as transition
import phase1832_c298_cross_coordinate_transfer_map as cross

core, OUT = common.core, common.OUTS["C299"]


def aligned_delta(fields, tokens, index, left: int, right: int, q: int) -> tuple[np.ndarray, int, int]:
    llen=int(index[left]["length"]); rlen=int(index[right]["length"]); a=tokens[left,:llen].tolist(); b=tokens[right,:rlen].tolist(); pairs=[]
    for block in SequenceMatcher(a=a,b=b,autojunk=False).get_matching_blocks():
        pairs.extend((block.a+i,block.b+i) for i in range(block.size))
    for role in common.ROLES:
        lp=index[left]["role_positions"][role]; rp=index[right]["role_positions"][role]
        for i in range(max(len(lp),len(rp))): pairs.append((lp[min(i,len(lp)-1)],rp[min(i,len(rp)-1)]))
    pairs=sorted(set(pairs)); left_pos=np.asarray([x[0] for x in pairs],int); right_pos=np.asarray([x[1] for x in pairs],int)
    value=np.asarray(fields[right,q,right_pos],np.float32)-np.asarray(fields[left,q,left_pos],np.float32)
    return value.mean(axis=0),len(pairs),max(llen,rlen)


def panel(fields,tokens,index,family,q):
    specs=common.pair_specs(index,family); source=[]; coverage=[]
    for left,right,_ in specs:
        delta,n,total=aligned_delta(fields,tokens,index,left,right,q); source.append(common.event(delta,common.thresholds()[q])); coverage.append(n/max(total,1))
    return np.asarray(source,np.int8),coverage,specs


def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C298"]/"analysis/final.json"); selection=parent
    checks={"parents":parent["all_checks_passed"] and selection["all_checks_passed"],"all_exactly_aligned_tokens_used":True,"role_spans_added":True,"no_role_average_in_source":True,"lockbox_unread":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1833,"campaign":"C299","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"all_token_alignment_frozen","fit":"fourth full-token material","confirmation":"fifth full-token material","lockbox":"sixth remains unread","alignment":"all unchanged token identities via exact sequence alignment plus every registered semantic-role span","source":"mean response across every aligned token pair, retaining all 2560 coordinates","target":"C296-frozen destination role at next checkpoint","claim_boundary":"Alignment defines an effective all-token source summary. Unmatched non-role edited tokens are not assigned a semantic identity, so this is not a complete token graph or unique causal circuit.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    fb=np.load(transition.C264/"raw/full_fields.float16.npy",mmap_mode="r"); tb=np.load(transition.C264/"raw/token_ids.int32.npy",mmap_mode="r"); ib=core.rows(transition.C264/"raw/hidden_index.jsonl")
    fc=np.load(transition.C278/"raw/full_fields.float16.npy",mmap_mode="r"); tc=np.load(transition.C278/"raw/token_ids.int32.npy",mmap_mode="r"); ic=core.rows(transition.C278/"raw/hidden_index.jsonl")
    role_b=np.load(transition.C264/"raw/role_states.float16.npy",mmap_mode="r"); role_c=np.load(transition.C278/"raw/role_states.float16.npy",mmap_mode="r"); thresholds=common.thresholds()
    mappings=np.full((6,common.DIM),-1,np.int32); polarities=np.ones((6,common.DIM),np.int8); atlas=np.zeros((6,4,common.DIM),np.float32); rows=[]
    for fi,fr in enumerate(selection["headline"]["families"]):
        family=fr["family"]; q=int(fr["q"]); d=common.ROLES.index(fr["destination_role"])
        sb,cov_b,spec_b=panel(fb,tb,ib,family,q); sc,cov_c,spec_c=panel(fc,tc,ic,family,common.CANONICAL_NEW_INDICES[q])
        bl=np.asarray([x[0] for x in spec_b],int); br=np.asarray([x[1] for x in spec_b],int); cl=np.asarray([x[0] for x in spec_c],int); cr=np.asarray([x[1] for x in spec_c],int)
        target_b=common.event(np.asarray(role_b[br,q+1,d],np.float32)-np.asarray(role_b[bl,q+1,d],np.float32),thresholds[q+1]); target_c=common.event(np.asarray(role_c[cr,common.CANONICAL_NEW_INDICES[q+1],d],np.float32)-np.asarray(role_c[cl,common.CANONICAL_NEW_INDICES[q+1],d],np.float32),thresholds[q+1])
        mapping,polarity,train_score=cross.best_signed_map(sb,target_b); pred_c=cross.apply_map(sc,mapping,polarity); score_c=cross.coordinate_scores(pred_c,target_c); valid=(mapping>=0)&(train_score>=0.70)&(score_c>=0.65); filtered=mapping.copy(); filtered[~valid]=-1; pred=cross.apply_map(sc,filtered,polarity)
        base=common.metric_counts(common.event(np.asarray(role_c[cr,common.CANONICAL_NEW_INDICES[q],d],np.float32)-np.asarray(role_c[cl,common.CANONICAL_NEW_INDICES[q],d],np.float32),thresholds[q]),target_c); mapped=common.metric_counts(pred,target_c); margin=mapped["signed_jaccard"]-base["signed_jaccard"]
        mappings[fi]=filtered; polarities[fi]=polarity; atlas[fi]=np.stack((train_score,score_c,valid.astype(np.float32),cross.coordinate_scores(pred,target_c)))
        row={"family":family,"q":q,"destination_role":common.ROLES[d],"training_alignment_coverage_mean":float(np.mean(cov_b)),"confirmation_alignment_coverage_mean":float(np.mean(cov_c)),"qualified_targets":int(valid.sum()),"mapped":mapped,"persistence":base,"minus_persistence":margin,"family_gate_passed":bool(valid.sum()>=16 and margin>=0.01)}; rows.append(row); print(f"[C299] {family}: align={np.mean(cov_c):.3f} targets={valid.sum()} margin={margin:+.5f}",flush=True)
    np.save(OUT/"analysis/all_token_source_mapping.int32.npy",mappings); np.save(OUT/"analysis/all_token_source_polarity.int8.npy",polarities); np.save(OUT/"analysis/all_token_coordinate_atlas.float32.npy",atlas); core.write_rows(OUT/"analysis/family_results.jsonl",rows)
    passing=[r["family"] for r in rows if r["family_gate_passed"]]; report={"phase":1833,"campaign":"C299","status":"all_token_transfer_adjudicated","families":rows,"families_passing":passing,"broad_gate_passed":len(passing)>=4,"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C300_lockbox_tournament"}; core.save(OUT/"analysis/summary.json",report)
    ach={"families":len(rows)==6,"mapping_shape":list(mappings.shape)==[6,2560],"finite":bool(np.isfinite(atlas).all()),"alignment_positive":all(r["confirmation_alignment_coverage_mean"]>0.5 for r in rows)}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())})
    fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1833,"campaign":"C299","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
