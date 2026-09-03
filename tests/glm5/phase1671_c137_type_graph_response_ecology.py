#!/usr/bin/env python3
"""C137 route D: artificial type graph response ecology plus natural panel."""
from __future__ import annotations
import gc,itertools,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";RESULT=TESTS/"result";OUT=RESULT/"phase1671_c137_type_graph_response_ecology";C136=RESULT/"phase1670_c136_chinese_pattern_composition_field";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1661_c127_typed_transition_language_family as c127
PHASE,CAMPAIGN=1671,"C137";ROUTES=("direct","two_hop","three_hop","shortcut","edge_type","query_reversal","irrelevant_dense");ROLES=("source","mid1","mid2","target","boundary");CHECKPOINTS=c127.CHECKPOINTS;DIM,WIDTH,BATCH=2560,232,8
SYL=("baf","cud","dix","fom","gup","hez","jol","kav","lum","neq","piv","qor","rax","syt","tuv","wim")
NATURAL=(("apple","fruit","food","object"),("salmon","fish","animal","organism"),("rose","flower","plant","organism"),("chair","furniture","artifact","object"),("hammer","tool","artifact","object"),("sparrow","bird","animal","organism"),("oak","tree","plant","organism"),("violin","instrument","artifact","object"))
def now():return datetime.now(timezone.utc).isoformat()
def cosine(a,b):
 d=float(np.linalg.norm(a)*np.linalg.norm(b));return 0.0 if d<=1e-12 else float(np.dot(a.ravel(),b.ravel())/d)
def vals(i):return tuple(f"Type{SYL[i%16]}{SYL[(i*11+3)%16]}{i:02d}{s}" for s in "abcde")
def edges(route,truth):
 a,b,c,d,e=range(5)
 if route=="direct":return [(a,d,"is_a"),(b,c,"is_a"),(e,c,"near")] if truth==1 else [(d,a,"is_a"),(b,c,"is_a"),(e,c,"near")]
 if route=="two_hop":return [(a,b,"is_a"),(b,d,"is_a"),(e,c,"near")] if truth==1 else [(b,a,"is_a"),(b,d,"is_a"),(e,c,"near")]
 if route=="three_hop":return [(a,b,"is_a"),(b,c,"is_a"),(c,d,"is_a")] if truth==1 else [(a,b,"is_a"),(c,b,"is_a"),(c,d,"is_a")]
 if route=="shortcut":return [(a,b,"is_a"),(b,c,"is_a"),(c,d,"is_a"),(a,d,"is_a")] if truth==1 else [(a,b,"is_a"),(c,b,"is_a"),(d,c,"is_a"),(d,a,"is_a")]
 if route=="edge_type":return [(a,b,"is_a"),(b,d,"is_a"),(c,e,"near")] if truth==1 else [(a,b,"is_a"),(b,d,"stored_in"),(c,e,"near")]
 if route=="query_reversal":return [(d,b,"is_a"),(b,a,"is_a"),(c,e,"near")] if truth==1 else [(a,b,"is_a"),(b,d,"is_a"),(c,e,"near")]
 if route=="irrelevant_dense":return [(a,b,"is_a"),(b,d,"is_a"),(c,e,"near"),(e,c,"stored_in"),(c,b,"near")] if truth==1 else [(a,b,"is_a"),(d,b,"is_a"),(c,e,"near"),(e,c,"stored_in"),(c,b,"near")]
 raise KeyError(route)
def reachable(es,start,target):
 seen={start};front=[start]
 while front:
  x=front.pop()
  for l,r,k in es:
   if k=="is_a" and l==x and r not in seen:seen.add(r);front.append(r)
 return target in seen
def render(v,es,route,surface):
 labels={"is_a":"is a kind of","near":"is near","stored_in":"is stored in"};sep="; " if surface==1 else " | ";statements=sep.join(f"{v[l]} {labels[k]} {v[r]}" for l,r,k in es);start,target=(3,0) if route=="query_reversal" else (0,3);return f"Rule: 'is a kind of' is transitive; other relation types do not count. Nodes: {', '.join(v)}. Facts: {statements}. Query: Is {v[start]} a kind of {v[target]}? Reply exactly yes or no.",start,target
def material():
 units=[];cases=[]
 for i in range(24):
  v=vals(i);part="discovery" if i<12 else "confirmation";unit={"unit_id":f"c137-{i:02d}","partition":part,"values":list(v),"panel":"artificial"};units.append(unit)
  for route,truth,surface in itertools.product(ROUTES,(1,-1),(1,-1)):
   es=edges(route,truth);prompt,start,target=render(v,es,route,surface);cases.append({**unit,"case_id":f"c137-{len(cases):04d}","route":route,"truth_factor":truth,"surface_factor":surface,"edges":es,"query_start":start,"query_target":target,"truth":truth==1,"gold_position":0 if truth==1 else 1,"prompt":prompt,"role_values":{"source":v[start],"mid1":v[1],"mid2":v[2],"target":v[target]}})
 natural=[]
 for i,chain in enumerate(NATURAL):
  for route,truth in itertools.product(("direct","two_hop","three_hop","query_reversal"),(1,-1)):
   a,b,c,d=chain
   if route=="direct":facts=[(a,d)] if truth==1 else [(d,a)];start,target=a,d
   elif route=="two_hop":facts=[(a,b),(b,d)] if truth==1 else [(b,a),(b,d)];start,target=a,d
   elif route=="three_hop":facts=[(a,b),(b,c),(c,d)] if truth==1 else [(a,b),(c,b),(c,d)];start,target=a,d
   else:facts=[(d,b),(b,a)] if truth==1 else [(a,b),(b,d)];start,target=d,a
   text="; ".join(f"{l} is a kind of {r}" for l,r in facts);prompt=f"Nodes: {', '.join(chain)}. Use only these classification facts and transitivity: {text}. Is {start} a kind of {target}? Reply exactly yes or no."
   natural.append({"unit_id":f"natural-{i:02d}","partition":"external","values":list(chain),"panel":"natural","case_id":f"c137-natural-{len(natural):03d}","route":route,"truth_factor":truth,"surface_factor":1,"truth":truth==1,"gold_position":0 if truth==1 else 1,"prompt":prompt,"role_values":{"source":start,"mid1":b,"mid2":c,"target":target}})
 return units,cases,natural
def compile_rows(tok,rows):
 cand=[[int(x) for x in tok.encode(" "+v,add_special_tokens=False)] for v in ("yes","no")];out=[]
 for row in rows:
  ids=core.chat_ids(tok,"Use only the stated type relations. Answer only yes or no.",row["prompt"]);pos={}
  for role,value in row["role_values"].items():
   spans=graph_base.name_spans(tok,ids,value)
   if not spans:raise RuntimeError((row["case_id"],role,value))
   pos[role]=spans[-1] if role in {"source","target"} else spans[0]
  pos["boundary"]=[len(ids)-1];out.append({**row,"prompt_ids":ids,"candidate_ids":cand,"role_positions":pos})
 return out
def contract():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C136/"audit/independent_closure_audit.json");u,c,n=material();tok=graph_base.tokenizer();cc=compile_rows(tok,c);nn=compile_rows(tok,n);zero={"yes":np.mean([r["truth"] for r in c]),"no":np.mean([not r["truth"] for r in c]),"surface":np.mean([(r["surface_factor"]==1)==r["truth"] for r in c])};checks={"auth":parent["all_checks_passed"] and parent["authorization"]=="start_route_D_C137","units":len(u)==24,"artificial":len(c)==672,"natural":len(n)==64,"unique":len({r["prompt"] for r in c+n})==736,"zero":all(x==.5 for x in zero.values()),"truth":all(reachable(r["edges"],r["query_start"],r["query_target"])==r["truth"] for r in c),"roles":all(set(r["role_positions"])==set(ROLES) for r in cc+nn),"width":max(len(r["prompt_ids"]) for r in cc+nn)<WIDTH}
 if not all(checks.values()):raise RuntimeError(checks)
 core.write_rows(OUT/"material/artificial_units.jsonl",u);core.write_rows(OUT/"material/artificial_cases.jsonl",c);core.write_rows(OUT/"material/natural_panel.jsonl",n);core.write_rows(OUT/"compiled/qwen3_artificial.jsonl",cc);core.write_rows(OUT/"compiled/qwen3_natural.jsonl",nn)
 paths={"c136":C136/"audit/independent_closure_audit.json"};p={"phase":PHASE,"campaign":CAMPAIGN,"status":"route_D_type_graph_contract_frozen","object":"artificial is-a response ecology with natural classification external panel","model":"Qwen3-4B BF16 CUDA","routes":list(ROUTES),"roles":list(ROLES),"behavior_gate":{"artificial_global_min":.90,"partition_min":.85,"truth_min":.85,"route_min":.80},"confirmation_gate":{"route_cosine_min":.80,"top256_overlap_min":.35,"passing_routes_min":6},"natural_panel":"external descriptive panel; never used to select or qualify the artificial nominee","claim_boundary":"type-graph response ecology, not a lexical apple vector or natural-world ontology proof","source_paths":{k:str(v) for k,v in paths.items()},"source_hashes":{k:core.sha(v) for k,v in paths.items()},"producer_sha256":core.sha(Path(__file__)),"authorization":"run_c137_behavior"};core.save(OUT/"protocol/preregistration.json",p);core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":p["authorization"]});print(json.dumps({"checks":checks,"zero":zero},indent=2,default=float))
def acc(x):return float(np.mean([r["correct"] for r in x]))
@torch.inference_mode()
def behavior():
 p=core.load(OUT/"protocol/preregistration.json");art=core.rows(OUT/"compiled/qwen3_artificial.jsonl");nat=core.rows(OUT/"compiled/qwen3_natural.jsonl");rows=art+nat;path=OUT/"raw/qwen3_behavior_logits.float32.npy";path.parent.mkdir(parents=True,exist_ok=True);raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(rows),2));res=[];model=None;repeat=0.
 try:
  model,tok,dev,place=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  def run(batch):
   ids,mask,pos,lens=fixed_base.fixed_batch(batch,pad,dev,WIDTH);o=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True);h=torch.stack([o.last_hidden_state[i,l-1] for i,l in enumerate(lens)]);z=model.lm_head(h).float();s=np.asarray([[float(z[i,c[0]]) for c in r["candidate_ids"]] for i,r in enumerate(batch)],np.float32);return s,o,ids,mask,pos
  for st in range(0,len(rows),BATCH):
   b=rows[st:st+BATCH];s,o,ids,mask,pos=run(b);raw[st:st+len(b)]=s
   for i,r in enumerate(b):pred=int(s[i,1]>s[i,0]);res.append({"row_index":st+i,"case_id":r["case_id"],"unit_id":r["unit_id"],"panel":r["panel"],"partition":r["partition"],"route":r["route"],"truth_factor":r["truth_factor"],"gold_position":r["gold_position"],"prediction":pred,"correct":pred==r["gold_position"]})
   del o,ids,mask,pos
  raw.flush();s,o,ids,mask,pos=run(rows[:BATCH]);repeat=float(np.max(np.abs(s-np.asarray(raw[:BATCH]))))
 finally:
  raw.flush()
  if model is not None:release_bf16(model)
  gc.collect();torch.cuda.empty_cache()
 core.write_rows(OUT/"raw/qwen3_behavior_index.jsonl",res);ar=[r for r in res if r["panel"]=="artificial"];nr=[r for r in res if r["panel"]=="natural"];summary={"artificial_global":acc(ar),"partition":{k:acc([r for r in ar if r["partition"]==k]) for k in ("discovery","confirmation")},"truth":{str(k):acc([r for r in ar if r["truth_factor"]==k]) for k in (1,-1)},"route":{k:acc([r for r in ar if r["route"]==k]) for k in ROUTES},"natural_external":acc(nr),"natural_route":{k:acc([r for r in nr if r["route"]==k]) for k in ("direct","two_hop","three_hop","query_reversal")}};g=p["behavior_gate"];gate=summary["artificial_global"]>=g["artificial_global_min"] and min(summary["partition"].values())>=g["partition_min"] and min(summary["truth"].values())>=g["truth_min"] and min(summary["route"].values())>=g["route_min"];checks={"rows":len(res)==736,"finite":bool(np.isfinite(raw).all()),"repeat":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};report={"phase":PHASE,"campaign":CAMPAIGN,"status":"behavior_qualified" if gate else "behavior_failed","summary":summary,"checks":checks,"gate_passed":gate,"authorization":"capture_c137_ecology" if gate else "close_c137_continue_E"};core.save(OUT/"analysis/behavior.json",report);core.save(OUT/"audit/internal_behavior_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":gate,"authorization":report["authorization"]});print(json.dumps(report,indent=2))
def tensor_output(x):return x[0] if isinstance(x,tuple) else x
@torch.inference_mode()
def capture():
 if core.load(OUT/"analysis/behavior.json")["authorization"]!="capture_c137_ecology":raise RuntimeError("unauthorized")
 rows=core.rows(OUT/"compiled/qwen3_artificial.jsonl")+core.rows(OUT/"compiled/qwen3_natural.jsonl");path=OUT/"raw/qwen3_type_ecology.bf16.npy";raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint16,shape=(736,5,38,DIM));model=None;repeat=0
 try:
  model,tok,dev,place=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  def run(b):
   cap={};hs=[model.model.embed_tokens.register_forward_hook(lambda _m,_a,o:cap.__setitem__("e",tensor_output(o).detach()))];hs += [layer.register_forward_hook(lambda _m,_a,o,j=i:cap.__setitem__(f"b{j}",tensor_output(o).detach())) for i,layer in enumerate(model.model.layers)];hs.append(model.model.norm.register_forward_hook(lambda _m,_a,o:cap.__setitem__("n",tensor_output(o).detach())))
   try:ids,mask,pos,lens=fixed_base.fixed_batch(b,pad,dev,WIDTH);out=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
   finally:
    for h in hs:h.remove()
   return [cap["e"],*[cap[f"b{i}"] for i in range(36)],cap["n"]],out,ids,mask,pos
  for st in range(0,736,BATCH):
   b=rows[st:st+BATCH];ts,o,ids,mask,pos=run(b)
   for i,r in enumerate(b):
    for ri,role in enumerate(ROLES):
     for q,t in enumerate(ts):raw[st+i,ri,q]=t[i,r["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
   if (st//BATCH+1)%24==0:raw.flush();print(f"[C137] {st+len(b)}/736",flush=True)
   del ts,o,ids,mask,pos
  raw.flush();ts,o,ids,mask,pos=run(rows[:BATCH])
  for i,r in enumerate(rows[:BATCH]):
   for ri,role in enumerate(ROLES):
    for q,t in enumerate(ts):
     bits=t[i,r["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy();repeat=max(repeat,int(np.max(np.abs(bits.astype(np.int64)-raw[i,ri,q].astype(np.int64)))))
 finally:
  raw.flush()
  if model is not None:release_bf16(model)
  gc.collect();torch.cuda.empty_cache()
 checks={"shape":list(raw.shape)==[736,5,38,DIM],"finite":bool(np.isfinite(c127.decode(raw[:2])).all()),"repeat":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};r={"status":"capture_complete","checks":checks,"shape":list(raw.shape),"sha256":core.sha(path),"authorization":"discover_c137_ecology"};core.save(OUT/"analysis/capture.json",r);core.save(OUT/"audit/internal_capture_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":r["authorization"]});print(json.dumps(r,indent=2))
def artificial_fields(part):
 rows=core.rows(OUT/"compiled/qwen3_artificial.jsonl");units=core.rows(OUT/"material/artificial_units.jsonl");sel=[u for u in units if u["partition"]==part];look={u["unit_id"]:i for i,u in enumerate(sel)};raw=np.load(OUT/"raw/qwen3_type_ecology.bf16.npy",mmap_mode="r");f=np.zeros((len(sel),7,5,38,DIM),np.float32)
 for i,r in enumerate(rows):
  if r["partition"]==part:f[look[r["unit_id"]],ROUTES.index(r["route"])]+=float(r["truth_factor"])/4*c127.decode(raw[i])
 return f
def top(v):return set(np.argpartition(np.abs(v),-256)[-256:].tolist())
def discover():
 f=artificial_fields("discovery");np.save(OUT/"analysis/discovery_ecology.float32.npy",f);left=f[:6].mean(0);right=f[6:].mean(0);nom={}
 for ti,route in enumerate(ROUTES):
  cs=[]
  for ri,role in enumerate(ROLES):
   for q in range(38):cs.append((max(cosine(left[ti,ri,q],right[ti,ri,q]),0)*min(np.linalg.norm(left[ti,ri,q]),np.linalg.norm(right[ti,ri,q])),ri,q))
  _,ri,q=max(cs);v=f[:,ti,ri,q].mean(0);path=OUT/f"protocol/{route}.float32.npy";path.parent.mkdir(parents=True,exist_ok=True);np.save(path,v);nom[route]={"role":ROLES[ri],"role_index":ri,"checkpoint":CHECKPOINTS[q],"checkpoint_index":q,"support":sorted(top(v)),"sha256":core.sha(path)}
 freeze={"status":"type_ecology_nominees_frozen","nominees":nom,"confirmation_and_natural_unread":True,"authorization":"validate_c137"};core.save(OUT/"protocol/frozen_ecology.json",freeze);checks={"shape":list(f.shape)==[12,7,5,38,DIM],"finite":bool(np.isfinite(f).all()),"routes":len(nom)==7};core.save(OUT/"audit/internal_discovery_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":freeze["authorization"]});print(json.dumps(freeze,indent=2))
def validate():
 p=core.load(OUT/"protocol/preregistration.json");z=core.load(OUT/"protocol/frozen_ecology.json");f=artificial_fields("confirmation");raw=np.load(OUT/"raw/qwen3_type_ecology.bf16.npy",mmap_mode="r");natrows=core.rows(OUT/"compiled/qwen3_natural.jsonl");results={};natural={}
 for ti,route in enumerate(ROUTES):
  n=z["nominees"][route];d=np.load(OUT/f"protocol/{route}.float32.npy");c=f[:,ti,n["role_index"],n["checkpoint_index"]].mean(0);co=cosine(d,c);ov=len(set(n["support"])&top(c))/256;results[route]={"cosine":co,"top256_overlap":ov,"passed":co>=p["confirmation_gate"]["route_cosine_min"] and ov>=p["confirmation_gate"]["top256_overlap_min"]}
  subset=[(672+i,r) for i,r in enumerate(natrows) if r["route"]==route]
  if subset:
   vec=np.zeros((5,38,DIM),np.float32)
   for idx,r in subset:vec+=float(r["truth_factor"])/len(subset)*c127.decode(raw[idx])
   natural[route]={"cosine_to_artificial":cosine(d,vec[n["role_index"],n["checkpoint_index"]]),"role":n["role"],"checkpoint":n["checkpoint"]}
 gate=sum(r["passed"] for r in results.values())>=p["confirmation_gate"]["passing_routes_min"];report={"status":"route_D_adjudicated","artificial_confirmation":results,"passing_routes":sum(r["passed"] for r in results.values()),"natural_external":natural,"prediction_gate_passed":gate,"authorization":"close_c137_continue_E"};core.save(OUT/"analysis/confirmation.json",report);checks={"shape":list(f.shape)==[12,7,5,38,DIM],"routes":len(results)==7,"natural":len(natural)==4};core.save(OUT/"audit/internal_confirmation_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":gate,"authorization":report["authorization"]});print(json.dumps(report,indent=2))
def close():
 b=core.load(OUT/"analysis/behavior.json");c=core.load(OUT/"analysis/confirmation.json") if b["gate_passed"] else None;cl={"status":"route_D_closed" if b["gate_passed"] else "route_D_behavior_failed","headline":{"behavior":b["summary"],"internal":c},"claim_boundary":"response ecology, not an apple vector, natural ontology proof, or unique causal graph","problems":["explicit metalinguistic rules","natural panel inherits explicit facts","Qwen3 only","late truth/output field possible"],"next_authorization":"continue route E"};core.save(OUT/"analysis/closure.json",cl);checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"behavior":core.load(OUT/"audit/internal_behavior_audit.json")["all_checks_passed"],"branch":(not b["gate_passed"]) or core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"]};core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":bool(c and c["prediction_gate_passed"]),"authorization":"independent_audit_then_route_E"});print(json.dumps(cl,indent=2))
def main():{"contract":contract,"behavior":behavior,"capture":capture,"discover":discover,"validate":validate,"close":close}[sys.argv[1]]()
if __name__=="__main__":main()
