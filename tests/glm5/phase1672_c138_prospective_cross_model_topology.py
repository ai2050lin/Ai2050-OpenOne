#!/usr/bin/env python3
"""C138 route E: fresh prospective cross-model role topology."""
from __future__ import annotations
import gc,itertools,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1672_c138_prospective_cross_model_topology";C137=R/"phase1671_c137_type_graph_response_ecology";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
from model_utils import MODEL_CONFIGS
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
PHASE,CAMPAIGN=1672,"C138";MODELS=("qwen3","glm4","deepseek7b");DEPTHS=("direct","two_hop");ROLES=("source","bridge","target","boundary");WIDTH,BATCH=224,4;SYL=("zaf","yud","xir","wep","voq","utn","sim","ral","qek","poj","nuv","mox","lir","keg","jaf","huz")
def now():return datetime.now(timezone.utc).isoformat()
def cosine(a,b):
 d=float(np.linalg.norm(a)*np.linalg.norm(b));return 0.0 if d<=1e-12 else float(np.dot(a.ravel(),b.ravel())/d)
def values(i):return tuple(f"Fresh{SYL[i%16]}{SYL[(i*7+5)%16]}{i:02d}{s}" for s in "abcd")
def material():
 units=[];cases=[]
 for i in range(32):
  v=values(i);p="discovery" if i<16 else "confirmation";u={"unit_id":f"c138-{i:02d}","partition":p,"values":list(v)};units.append(u)
  for depth,truth,surface in itertools.product(DEPTHS,(1,-1),(1,-1)):
   if depth=="direct":edges=[(0,2),(1,3)] if truth==1 else [(2,0),(1,3)]
   else:edges=[(0,1),(1,2),(3,1)] if truth==1 else [(1,0),(1,2),(3,1)]
   sep="; " if surface==1 else " | ";label="Directed links" if surface==1 else "Route entries";text=sep.join(f"{v[a]} -> {v[b]}" for a,b in edges);prompt=f"A path must follow arrow direction. Nodes: {', '.join(v)}. {label}: {text}. Is there a directed path from {v[0]} to {v[2]}? Reply exactly yes or no."
   cases.append({**u,"case_id":f"c138-{len(cases):04d}","depth":depth,"truth_factor":truth,"surface_factor":surface,"truth":truth==1,"gold_position":0 if truth==1 else 1,"prompt":prompt,"role_values":{"source":v[0],"bridge":v[1],"target":v[2]}})
 return units,cases
def tokenizer(name):
 from transformers import AutoTokenizer
 return AutoTokenizer.from_pretrained(MODEL_CONFIGS[name]["path"],trust_remote_code=True,local_files_only=True,use_fast=False)
def compile_rows(tok,cases):
 cand=[[int(x) for x in tok.encode(" "+v,add_special_tokens=False)] for v in ("yes","no")];out=[]
 if any(len(x)!=1 for x in cand):raise RuntimeError(cand)
 for r in cases:
  ids=core.chat_ids(tok,"Use only the directed links. Answer only yes or no.",r["prompt"]);pos={}
  for role,val in r["role_values"].items():
   spans=graph_base.name_spans(tok,ids,val)
   if not spans:raise RuntimeError((r["case_id"],role,val))
   pos[role]=spans[-1] if role in {"source","target"} else spans[0]
  pos["boundary"]=[len(ids)-1];out.append({**r,"prompt_ids":ids,"candidate_ids":cand,"role_positions":pos})
 return out
def contract():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C137/"audit/independent_closure_audit.json");u,c=material();compiled={m:compile_rows(tokenizer(m),c) for m in MODELS};zero={"yes":np.mean([r["truth"] for r in c]),"no":np.mean([not r["truth"] for r in c]),"surface":np.mean([(r["surface_factor"]==1)==r["truth"] for r in c])};checks={"auth":parent["all_checks_passed"] and parent["authorization"]=="start_route_E_C138","units":len(u)==32,"cases":len(c)==256,"unique":len({r["prompt"] for r in c})==256,"zero":all(v==.5 for v in zero.values()),"compiled":all(len(v)==256 for v in compiled.values()),"roles":all(set(r["role_positions"])==set(ROLES) for v in compiled.values() for r in v),"width":max(len(r["prompt_ids"]) for v in compiled.values() for r in v)<WIDTH}
 if not all(checks.values()):raise RuntimeError(checks)
 core.write_rows(OUT/"material/units.jsonl",u);core.write_rows(OUT/"material/cases.jsonl",c)
 for m,v in compiled.items():core.write_rows(OUT/f"compiled/{m}.jsonl",v)
 paths={"c137":C137/"audit/independent_closure_audit.json"};p={"phase":PHASE,"campaign":CAMPAIGN,"status":"route_E_cross_model_contract_frozen","object":"fresh direct/two-hop path truth response across three frozen models","models":list(MODELS),"units":32,"cases":256,"roles":list(ROLES),"behavior_gate":{"global_min":.90,"partition_min":.85,"truth_min":.85,"surface_min":.85,"depth_min":.85},"confirmation_gate":{"cosine_min":.80,"relative_peak_depth_difference_max":.20},"comparison":["relative checkpoint depth","winning role topology","within-model normalized role trajectory"],"forbidden":"same physical coordinate identity across models; PCA/SVD; attention/MLP/weight inspection","model_policy":"strictly sequential load, behavior qualification before HiddenState capture","claim_boundary":"controlled route response topology, not a cross-model coordinate alignment or universal language mechanism","source_paths":{k:str(v) for k,v in paths.items()},"source_hashes":{k:core.sha(v) for k,v in paths.items()},"producer_sha256":core.sha(Path(__file__)),"authorization":"run_models_sequentially"};core.save(OUT/"protocol/preregistration.json",p);core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":p["authorization"]});print(json.dumps({"checks":checks,"zero":zero,"max_width":{m:max(len(r["prompt_ids"]) for r in v) for m,v in compiled.items()}},indent=2,default=float))
def acc(x):return float(np.mean([r["correct"] for r in x]))
@torch.inference_mode()
def behavior(name):
 rows=core.rows(OUT/f"compiled/{name}.jsonl");path=OUT/f"raw/{name}_candidate_logits.float32.npy";path.parent.mkdir(parents=True,exist_ok=True);raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(256,2));res=[];model=None;repeat=0.
 try:
  model,tok,dev,place=load_bf16(name);quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  def run(b):
   ids,mask,pos,lens=fixed_base.fixed_batch(b,pad,dev,WIDTH);o=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True);s=np.asarray([[float(o.logits[i,lens[i]-1,c[0]]) for c in r["candidate_ids"]] for i,r in enumerate(b)],np.float32);return s,o,ids,mask,pos
  for st in range(0,256,BATCH):
   b=rows[st:st+BATCH];s,o,ids,mask,pos=run(b);raw[st:st+len(b)]=s
   for i,r in enumerate(b):pred=int(s[i,1]>s[i,0]);res.append({"row_index":st+i,"case_id":r["case_id"],"unit_id":r["unit_id"],"partition":r["partition"],"depth":r["depth"],"truth_factor":r["truth_factor"],"surface_factor":r["surface_factor"],"prediction":pred,"gold_position":r["gold_position"],"correct":pred==r["gold_position"]})
   del o,ids,mask,pos
  raw.flush();s,o,ids,mask,pos=run(rows[:BATCH]);repeat=float(np.max(np.abs(s-np.asarray(raw[:BATCH]))))
 finally:
  raw.flush()
  if model is not None:release_bf16(model)
  gc.collect();torch.cuda.empty_cache()
 core.write_rows(OUT/f"raw/{name}_behavior_index.jsonl",res);summary={"global":acc(res),"partition":{k:acc([r for r in res if r["partition"]==k]) for k in ("discovery","confirmation")},"truth":{str(k):acc([r for r in res if r["truth_factor"]==k]) for k in (1,-1)},"surface":{str(k):acc([r for r in res if r["surface_factor"]==k]) for k in (1,-1)},"depth":{k:acc([r for r in res if r["depth"]==k]) for k in DEPTHS}};g=core.load(OUT/"protocol/preregistration.json")["behavior_gate"];gate=summary["global"]>=g["global_min"] and min(summary["partition"].values())>=g["partition_min"] and min(summary["truth"].values())>=g["truth_min"] and min(summary["surface"].values())>=g["surface_min"] and min(summary["depth"].values())>=g["depth_min"];checks={"rows":len(res)==256,"finite":bool(np.isfinite(raw).all()),"repeat":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};r={"model":name,"status":"behavior_qualified" if gate else "behavior_failed","summary":summary,"checks":checks,"gate_passed":gate,"runtime":place,"authorization":f"capture_{name}" if gate else f"close_{name}_continue"};core.save(OUT/f"analysis/{name}_behavior.json",r);core.save(OUT/f"audit/{name}_behavior_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":gate,"authorization":r["authorization"]});print(json.dumps({k:v for k,v in r.items() if k!="runtime"},indent=2))
def base_parts(model):
 base=model.model
 return base.embed_tokens,base.layers,base.norm
def tensor_output(x):return x[0] if isinstance(x,tuple) else x
@torch.inference_mode()
def capture(name):
 b=core.load(OUT/f"analysis/{name}_behavior.json")
 if not b["gate_passed"]:raise RuntimeError("behavior failed")
 rows=core.rows(OUT/f"compiled/{name}.jsonl");model=None;repeat=0
 try:
  model,tok,dev,place=load_bf16(name);quant=quantization_audit(model);embed,layers,norm=base_parts(model);states=len(layers)+2;dim=int(model.config.hidden_size);path=OUT/f"raw/{name}_role_field.bf16.npy";raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint16,shape=(256,4,states,dim));pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  def run(batch):
   cap={};hs=[embed.register_forward_hook(lambda _m,_a,o:cap.__setitem__("e",tensor_output(o).detach()))];hs += [layer.register_forward_hook(lambda _m,_a,o,j=i:cap.__setitem__(f"b{j}",tensor_output(o).detach())) for i,layer in enumerate(layers)];hs.append(norm.register_forward_hook(lambda _m,_a,o:cap.__setitem__("n",tensor_output(o).detach())))
   try:ids,mask,pos,lens=fixed_base.fixed_batch(batch,pad,dev,WIDTH);out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
   finally:
    for h in hs:h.remove()
   return [cap["e"],*[cap[f"b{i}"] for i in range(len(layers))],cap["n"]],out,ids,mask,pos
  for st in range(0,256,BATCH):
   batch=rows[st:st+BATCH];ts,o,ids,mask,pos=run(batch)
   for i,r in enumerate(batch):
    for ri,role in enumerate(ROLES):
     for q,t in enumerate(ts):raw[st+i,ri,q]=t[i,r["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
   if (st//BATCH+1)%16==0:raw.flush();print(f"[C138 {name}] {st+len(batch)}/256",flush=True)
   del ts,o,ids,mask,pos
  raw.flush();ts,o,ids,mask,pos=run(rows[:BATCH])
  for i,r in enumerate(rows[:BATCH]):
   for ri,role in enumerate(ROLES):
    for q,t in enumerate(ts):
     bits=t[i,r["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy();repeat=max(repeat,int(np.max(np.abs(bits.astype(np.int64)-raw[i,ri,q].astype(np.int64)))))
 finally:
  if 'raw' in locals():raw.flush()
  if model is not None:release_bf16(model)
  gc.collect();torch.cuda.empty_cache()
 checks={"shape":list(raw.shape)==[256,4,states,dim],"finite":bool(np.isfinite(raw[:2].view(np.uint16)).all()),"repeat":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};r={"model":name,"status":"capture_complete","states":states,"layers":states-2,"dimension":dim,"shape":list(raw.shape),"sha256":core.sha(path),"checks":checks,"authorization":f"analyze_{name}"};core.save(OUT/f"analysis/{name}_capture.json",r);core.save(OUT/f"audit/{name}_capture_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":r["authorization"]});print(json.dumps(r,indent=2))
def decode(x):return (np.asarray(x).view(np.uint16).astype(np.uint32)<<16).view(np.float32)
def analyze(name):
 cap=core.load(OUT/f"analysis/{name}_capture.json");raw=np.load(OUT/f"raw/{name}_role_field.bf16.npy",mmap_mode="r");rows=core.rows(OUT/f"compiled/{name}.jsonl");units=core.rows(OUT/"material/units.jsonl");field=np.zeros((32,2,4,cap["states"],cap["dimension"]),np.float32);look={u["unit_id"]:i for i,u in enumerate(units)}
 for i,r in enumerate(rows):field[look[r["unit_id"]],DEPTHS.index(r["depth"])]+=float(r["truth_factor"])/4*decode(raw[i])
 result={};profiles={}
 for di,depth in enumerate(DEPTHS):
  disc=field[:16,di];left=disc[:8].mean(0);right=disc[8:].mean(0);cand=[]
  for ri,role in enumerate(ROLES):
   for q in range(cap["states"]):cand.append((max(cosine(left[ri,q],right[ri,q]),0)*min(np.linalg.norm(left[ri,q]),np.linalg.norm(right[ri,q])),ri,q))
  _,ri,q=max(cand);dv=disc[:,ri,q].mean(0);cv=field[16:,di,ri,q].mean(0);co=cosine(dv,cv);profile=np.linalg.norm(field[:,di].mean(0),axis=-1);profile=profile/np.maximum(profile.max(axis=1,keepdims=True),1e-12);profiles[depth]=profile.tolist();result[depth]={"role":ROLES[ri],"checkpoint_index":q,"relative_depth":q/(cap["states"]-1),"confirmation_cosine":co,"discovery_peak_relative_depth":int(np.argmax(np.linalg.norm(disc[:,ri].mean(0),axis=-1)))/(cap["states"]-1),"passed":co>=core.load(OUT/"protocol/preregistration.json")["confirmation_gate"]["cosine_min"]}
 r={"model":name,"status":"model_topology_adjudicated","result":result,"normalized_role_profiles":profiles,"model_gate_passed":all(x["passed"] for x in result.values()),"authorization":"synthesize_after_all_models"};core.save(OUT/f"analysis/{name}_topology.json",r);core.save(OUT/f"audit/{name}_topology_audit.json",{"checks":{"depths":len(result)==2,"profiles":len(profiles)==2,"finite":bool(np.isfinite(field).all())},"all_checks_passed":True,"scientific_gate_passed":r["model_gate_passed"],"authorization":r["authorization"]});print(json.dumps({k:v for k,v in r.items() if k!="normalized_role_profiles"},indent=2))
def synthesize():
 reports={};missing={}
 for m in MODELS:
  b=core.load(OUT/f"analysis/{m}_behavior.json")
  if b["gate_passed"]:reports[m]=core.load(OUT/f"analysis/{m}_topology.json")
  else:missing[m]="behavior_failed"
 pairs={}
 for i,a in enumerate(reports):
  for b in list(reports)[i+1:]:
   vals=[]
   for depth in DEPTHS:
    pa=np.asarray(reports[a]["normalized_role_profiles"][depth]);pb=np.asarray(reports[b]["normalized_role_profiles"][depth]);x=np.linspace(0,1,pa.shape[1]);y=np.linspace(0,1,pb.shape[1]);grid=np.linspace(0,1,101)
    for ri,role in enumerate(ROLES):vals.append(cosine(np.interp(grid,x,pa[ri]),np.interp(grid,y,pb[ri])))
   pairs[f"{a}__{b}"]={"median_normalized_profile_cosine":float(np.median(vals)),"min":float(np.min(vals)),"comparisons":len(vals)}
 r={"phase":PHASE,"campaign":CAMPAIGN,"status":"route_E_synthesized","qualified_models":list(reports),"missing_models":missing,"model_results":{m:{"result":x["result"],"model_gate_passed":x["model_gate_passed"]} for m,x in reports.items()},"cross_model_profile":pairs,"claim_boundary":"relative depth and normalized role topology only; no coordinate identity","authorization":"close_c138"};core.save(OUT/"analysis/cross_model_synthesis.json",r);print(json.dumps(r,indent=2))
def close():
 s=core.load(OUT/"analysis/cross_model_synthesis.json");checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"three_behaviors":all((OUT/f"audit/{m}_behavior_audit.json").exists() for m in MODELS),"qualified_have_capture":all((OUT/f"audit/{m}_capture_audit.json").exists() and (OUT/f"audit/{m}_topology_audit.json").exists() for m in s["qualified_models"]),"synthesis":len(s["qualified_models"])+len(s["missing_models"])==3};cl={"status":"route_E_closed","headline":s,"claim_boundary":s["claim_boundary"],"next_authorization":"phase1673_C139_campaign_synthesis_and_causal_adjudication"};core.save(OUT/"analysis/closure.json",cl);core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":len(s["qualified_models"])>=2,"authorization":"independent_audit_then_C139"});print(json.dumps(cl,indent=2))
def main():
 mode=sys.argv[1]
 if mode=="contract":contract()
 elif mode in {"behavior","capture","analyze"}:globals()[mode](sys.argv[2])
 elif mode=="synthesize":synthesize()
 elif mode=="close":close()
 else:raise SystemExit(mode)
if __name__=="__main__":main()
