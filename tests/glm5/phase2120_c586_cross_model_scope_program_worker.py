#!/usr/bin/env python3
"""Isolated GLM4/DeepSeek worker for C586.

Models are run in separate processes.  Hidden states are captured only after
the model-specific behavior screen passes.  Attention and MLP internals are
never read.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
TESTS=ROOT/"tests/glm5"
sys.path.insert(0,str(TESTS))

import phase1797_c263_c272_state_operator_common as compiler
import phase2105_c571_c589_scope_program_algebra_campaign as campaign


FAMILIES=("discourse_permutation","fact_voice_fixed_query","evidence_paraphrase","clause_packaging","path_depth","translation_language")
UNITS=(0,1,2,3,14,15,16,17)


def save(path:Path,value)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def write_rows(path:Path,rows)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+"\n")


def metric(prediction:np.ndarray,truth:np.ndarray)->dict:
    p=np.asarray(prediction,np.float64).reshape(-1);y=np.asarray(truth,np.float64).reshape(-1);e=p-y;den=math.sqrt(float(np.mean(y*y)))+1e-12
    return {"nrmse":math.sqrt(float(np.mean(e*e)))/den,"cosine":float(np.dot(p,y)/(np.linalg.norm(p)*np.linalg.norm(y)+1e-12)),"sign_agreement":float(np.mean(np.sign(p)==np.sign(y)))}


def scaled_like(control:np.ndarray,reference:np.ndarray)->np.ndarray:
    return control*(float(np.linalg.norm(reference))/(float(np.linalg.norm(control))+1e-12))


def role_bundle(states:np.ndarray,row:dict,q:int)->np.ndarray:
    values=[]
    for role in campaign.ROLES:
        points=[int(v) for v in row["role_positions"][role]];values.append(np.asarray(states[row["hidden_index"],q,points],np.float32).mean(axis=0))
    return np.stack(values)


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--model",choices=("glm4","deepseek7b"),required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    out=args.output.parent.parent/"raw"/args.model;out.mkdir(parents=True,exist_ok=True)
    all_rows=campaign.read_rows(campaign.material_path());rows=[]
    for row in all_rows:
        if row["panel"]!="atomic" or row["family"] not in FAMILIES or row["surface"]!="record" or row["unit"] not in UNITS:continue
        if row["operation_domain"] not in campaign.ATOMIC_SPECS[row["family"]][:2]:continue
        rows.append(row)
    model=None;hooks=[];captured=[];states=None
    try:
        model,tokenizer,device,placement=campaign.parent.previous.model_base().load_bf16(args.model)
        compiled=compiler.compile_qwen(tokenizer,rows);write_rows(out/"compiled.jsonl",compiled)
        pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id);behavior=[]
        for start in range(0,len(compiled),8):
            batch=compiled[start:start+8];width=max(len(r["prompt_ids"]) for r in batch);ids=torch.full((len(batch),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
            for i,row in enumerate(batch):seq=row["prompt_ids"];ids[i,:len(seq)]=torch.tensor(seq,device=device);mask[i,:len(seq)]=1
            pos=mask.long().cumsum(-1)-1;pos.masked_fill_(mask==0,0)
            with torch.inference_mode():logits=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True).logits
            for i,row in enumerate(batch):
                length=len(row["prompt_ids"]);scores=[float(logits[i,length-1,c[0]]) for c in row["candidate_ids"]];pred=int(scores[1]>scores[0]);behavior.append({"case_id":row["case_id"],"correct":pred==row["gold_position"],"prediction":pred,"scores":scores})
        write_rows(out/"behavior.jsonl",behavior);accuracy=float(np.mean([r["correct"] for r in behavior]))
        if accuracy<campaign.BEHAVIOR_SLICE_GATE:
            result={"status":"behavior_unqualified","model":args.model,"rows":len(rows),"behavior_accuracy":accuracy,"hiddenstate_ran":False,"functional_candidate":False,"placement":placement}
            save(args.output,result);raise SystemExit(2)

        base=model.model;layers=list(base.layers);checkpoints=len(layers)+2;dim=int(model.get_input_embeddings().weight.shape[1]);width=max(len(r["prompt_ids"]) for r in compiled);n=len(compiled)
        states=np.lib.format.open_memmap(out/"full_token_states.float16.npy",mode="w+",dtype=np.float16,shape=(n,checkpoints,width,dim))
        def hook(_module,_args,output):captured.append(output[0] if isinstance(output,tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook));hooks.extend(layer.register_forward_hook(hook) for layer in layers);hooks.append(base.norm.register_forward_hook(hook));index=[]
        for start in range(0,n,2):
            batch=compiled[start:start+2];ids=torch.full((len(batch),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);lengths=[]
            for i,row in enumerate(batch):seq=row["prompt_ids"];lengths.append(len(seq));ids[i,:len(seq)]=torch.tensor(seq,device=device);mask[i,:len(seq)]=1
            pos=mask.long().cumsum(-1)-1;pos.masked_fill_(mask==0,0);captured.clear()
            with torch.inference_mode():model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
            if len(captured)!=checkpoints:raise RuntimeError((len(captured),checkpoints))
            for q,state in enumerate(captured):
                for i,length in enumerate(lengths):states[start+i,q,:length]=state[i,:length].float().cpu().numpy().astype(np.float16)
            for i,row in enumerate(batch):
                index.append({"hidden_index":start+i,"case_id":row["case_id"],"family":row["family"],"operation_domain":row["operation_domain"],"unit":row["unit"],"variant":row["variant"],"partition":row["partition"],"role_positions":row["role_positions"]})
            states.flush()
            if start%32==0 or start+len(batch)==n:print(f"[{args.model}] field {start+len(batch)}/{n}",flush=True)
        write_rows(out/"hidden_index.jsonl",index)
        pairs=defaultdict(dict)
        for row in index:pairs[(row["family"],row["operation_domain"],row["unit"])][row["variant"]]=row
        qpoints=sorted(set((0,round((checkpoints-1)*.25),round((checkpoints-1)*.5),round((checkpoints-1)*.75),checkpoints-1)))
        prototypes={};metrics={};gates={};representatives={}
        for family in FAMILIES:
            for domain in campaign.ATOMIC_SPECS[family][:2]:
                train=[v for (f,d,u),v in pairs.items() if f==family and d==domain and u in (0,1,2,3) and set(v)=={0,1}]
                test=[v for (f,d,u),v in pairs.items() if f==family and d==domain and u in (14,15,16,17) and set(v)=={0,1}]
                for q in qpoints:
                    if not train or not test:continue
                    tr=np.stack([role_bundle(states,v[1],q)-role_bundle(states,v[0],q) for v in train]);te=np.stack([role_bundle(states,v[1],q)-role_bundle(states,v[0],q) for v in test]);proto=tr.mean(axis=0);prototypes[(family,domain,q)]=proto
                    wrong_family=next(name for name in FAMILIES if name!=family);wrong_candidates=[p for (f,d,qq),p in prototypes.items() if f==wrong_family and qq==q]
                    wrong=scaled_like(wrong_candidates[0] if wrong_candidates else proto[::-1].copy(),proto)
                    value={"correct":metric(np.broadcast_to(proto,te.shape),te),"zero":metric(np.zeros_like(te),te),"wrong":metric(np.broadcast_to(wrong,te.shape),te),"samples":len(test)};key=f"{family}|{domain}|q{q}";metrics[key]=value;gates[key]=value["correct"]["nrmse"]<=value["zero"]["nrmse"]-campaign.CONTROL_MARGIN and value["correct"]["nrmse"]<=value["wrong"]["nrmse"]-campaign.CONTROL_MARGIN;representatives[key]=proto.tolist()
        family_summary={family:{"passed":int(sum(v for k,v in gates.items() if k.startswith(family+"|"))),"total":sum(k.startswith(family+"|") for k in gates)} for family in FAMILIES}
        for value in family_summary.values():value["pass_rate"]=value["passed"]/max(value["total"],1)
        functional=any(value["pass_rate"]>=campaign.PREDICTION_GATE for value in family_summary.values())
        result={"status":"closed","model":args.model,"rows":n,"behavior_accuracy":accuracy,"hiddenstate_ran":True,"checkpoints":checkpoints,"coordinates":dim,"qpoints":qpoints,"shape":list(states.shape),"raw_path":str((out/"full_token_states.float16.npy").relative_to(ROOT)),"raw_bytes":(out/"full_token_states.float16.npy").stat().st_size,"family_summary":family_summary,"metrics":metrics,"gates":gates,"representative_full_coordinates":representatives,"functional_candidate":functional,"placement":placement,"strict_interpretation":"within-model response topology only; coordinates are not compared across models"}
        save(args.output,result)
    finally:
        for handle in hooks:handle.remove()
        if states is not None:states.flush();del states
        campaign.parent.previous.model_base().release_bf16(model);gc.collect()


if __name__=="__main__":main()
