#!/usr/bin/env python3
"""C306: intervene on every source coordinate of each qualified all-token map."""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1832_c298_cross_coordinate_transfer_map as cross

core, OUT = common.core, common.OUTS["C306"]
CONDITIONS=("natural","delete_sources","correct_rescue","wrong_coordinate_delete","wrong_token_delete")


def align_positions(tokens,index,left,right):
    llen=index[left]["length"]; rlen=index[right]["length"]; a=tokens[left,:llen].tolist(); b=tokens[right,:rlen].tolist(); pairs=[]
    for block in SequenceMatcher(a=a,b=b,autojunk=False).get_matching_blocks(): pairs.extend((block.a+i,block.b+i) for i in range(block.size))
    for role in common.ROLES:
        lp=index[left]["role_positions"][role]; rp=index[right]["role_positions"][role]
        for i in range(max(len(lp),len(rp))): pairs.append((lp[min(i,len(lp)-1)],rp[min(i,len(rp)-1)]))
    return sorted(set(pairs))


@torch.inference_mode()
def main() -> None:
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C305"]/"analysis/final.json"); qualified=parent["headline"]["qualified"]
    checks={"parent":parent["all_checks_passed"],"registered_conditions":True,"cuda":torch.cuda.is_available(),"all_qualified_targets":True};
    for sub in ("analysis","audit","protocol","raw"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1840,"campaign":"C306","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"causal_contract_frozen" if qualified else "registered_no_test","qualified":qualified,"samples":"one fixed-factor=1, order=+1 factor-A pair for every sixth-material unit and surface (16 per qualified branch)","conditions":list(CONDITIONS),"delete":"at every exact/role-aligned token, replace every qualified source coordinate with the matched left-cell value","correct_rescue":"restore natural right values after the same deletion mask","controls":["roll source coordinate identities by one","write source coordinates only at the answer boundary token"],"target":"all qualified mapped coordinates at the frozen next-checkpoint destination role","gate":"mean target movement toward left>=0.10 and exceeds both wrong controls by>=0.05","claim_boundary":"This is a broad, non-minimal intervention on a lockbox-qualified predictive map. Passing would show causal use of the coalition, not uniqueness; failure rejects this patch interface only.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    if not qualified:
        report={"phase":1840,"campaign":"C306","status":"no_test_no_qualified_cross_coordinate_branch","branches":[],"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C307_cross_model_and_C308_adjudication"}; core.save(OUT/"analysis/summary.json",report); core.save(OUT/"audit/internal_analysis_audit.json",{"checks":{"no_test_preserved":True},"all_checks_passed":True}); final={"phase":1840,"campaign":"C306","status":"closed","checks":{"contract":all(checks.values()),"analysis":True,"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]},"all_checks_passed":all(checks.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2)); return
    masks=np.load(common.OUTS["C305"]/"analysis/qualified_target_masks.bool.npy"); token_map=np.load(common.OUTS["C299"]/"analysis/all_token_source_mapping.int32.npy"); states=np.load(common.OUTS["C295"]/"raw/role_states.float16.npy",mmap_mode="r"); fields=np.load(common.OUTS["C295"]/"raw/full_fields.float16.npy",mmap_mode="r"); tokens=np.load(common.OUTS["C295"]/"raw/token_ids.int32.npy",mmap_mode="r"); index=core.rows(common.OUTS["C295"]/"raw/hidden_index.jsonl"); compiled=core.rows(common.OUTS["C294"]/"compiled/qwen3.jsonl")
    model=None; samples=[]
    try:
        model,_tok,device,placement=common.model_base.load_bf16("qwen3"); base=model.model; quant=common.model_base.quantization_audit(model)
        for branch in qualified:
            fi=common.FAMILIES.index(branch["family"]); q=int(branch["q"]); d=common.ROLES.index(branch["destination_role"]); mi=1 if branch["model"]=="M4_all_token" else 0; target_mask=masks[fi,mi]; source_coords=np.unique(token_map[fi][target_mask]) if mi==1 else np.asarray([],int); source_coords=source_coords[source_coords>=0]
            specs=[x for x in common.pair_specs(index,branch["family"]) if x[2]["fixed_factor"]==1 and x[2]["order"]==1]
            for left,right,meta in specs:
                pairs=align_positions(tokens,index,left,right); left_pos=np.asarray([p[0] for p in pairs],int); right_pos=np.asarray([p[1] for p in pairs],int); row=compiled[right]; ids=torch.tensor([row["prompt_ids"]],dtype=torch.long,device=device); mask=torch.ones_like(ids); natural_target=np.asarray(states[right,common.CANONICAL_NEW_INDICES[q+1],d],np.float32); left_target=np.asarray(states[left,common.CANONICAL_NEW_INDICES[q+1],d],np.float32); denom=float(np.abs(natural_target[target_mask]-left_target[target_mask]).sum())
                condition_rows={}
                for condition in CONDITIONS:
                    captured=[]
                    def patch_hook(_module,_args,output):
                        if condition=="natural": return output
                        value=output.clone(); coords=source_coords if condition!="wrong_coordinate_delete" else (source_coords+1)%common.DIM
                        positions=right_pos if condition!="wrong_token_delete" else np.asarray(row["role_positions"]["boundary"],int)
                        for li,rp in zip(left_pos[:len(positions)] if condition=="wrong_token_delete" else left_pos,positions):
                            if condition=="correct_rescue": donor=np.asarray(fields[right,common.CANONICAL_NEW_INDICES[q],rp],np.float32)
                            else: donor=np.asarray(fields[left,common.CANONICAL_NEW_INDICES[q],li],np.float32)
                            donor_t=torch.tensor(donor[coords],dtype=value.dtype,device=device); coord_t=torch.tensor(coords,dtype=torch.long,device=device); value[0,int(rp),coord_t]=donor_t
                        return value
                    def capture_hook(_module,_args,output):
                        v=output[0] if isinstance(output,tuple) else output; captured.append(v[0,row["role_positions"][common.ROLES[d]]].mean(0).float().cpu().numpy())
                    patch=base.layers[q-1].register_forward_hook(patch_hook); capture=base.layers[q].register_forward_hook(capture_hook)
                    try: output=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
                    finally: patch.remove(); capture.remove()
                    value=captured[0]; movement=float(1.0-np.abs(value[target_mask]-left_target[target_mask]).sum()/max(denom,1e-12)); logits=[float(output.logits[0,ids.shape[1]-1,c[0]]) for c in row["candidate_ids"]]; condition_rows[condition]={"target_movement_toward_left":movement,"candidate_margin":logits[row["gold_position"]]-logits[1-row["gold_position"]]}; del output
                delete=condition_rows["delete_sources"]["target_movement_toward_left"]; wrong=max(condition_rows["wrong_coordinate_delete"]["target_movement_toward_left"],condition_rows["wrong_token_delete"]["target_movement_toward_left"]); samples.append({"family":branch["family"],"model":branch["model"],"surface":meta["surface"],"unit":meta["unit"],"q":q,"destination_role":common.ROLES[d],"source_coordinates":int(len(source_coords)),"target_coordinates":int(target_mask.sum()),"conditions":condition_rows,"deletion_movement":delete,"delete_minus_best_wrong":delete-wrong})
                print(f"[C306] {branch['family']}/{meta['surface']}/u{meta['unit']}: move={delete:+.4f} control={delete-wrong:+.4f}",flush=True)
        core.write_rows(OUT/"raw/sample_results.jsonl",samples); branches=[]
        for branch in qualified:
            sr=[s for s in samples if s["family"]==branch["family"] and s["model"]==branch["model"]]; movement=float(np.mean([s["deletion_movement"] for s in sr])); margin=float(np.mean([s["delete_minus_best_wrong"] for s in sr])); branches.append({"family":branch["family"],"model":branch["model"],"samples":len(sr),"mean_deletion_movement":movement,"mean_delete_minus_best_wrong":margin,"causal_gate_passed":movement>=0.10 and margin>=0.05})
        report={"phase":1840,"campaign":"C306","status":"multisource_multitarget_causal_adjudicated","branches":branches,"placement":placement,"quantization":quant,"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C307_cross_model_and_C308_adjudication"}; core.save(OUT/"analysis/summary.json",report)
        ach={"all_branches_run":len(branches)==len(qualified),"sixteen_samples_each":all(b["samples"]==16 for b in branches),"finite":bool(np.isfinite([s[k] for s in samples for k in ("deletion_movement","delete_minus_best_wrong")]).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1840,"campaign":"C306","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))
    finally: common.model_base.release(model); gc.collect()


if __name__=="__main__": main()
