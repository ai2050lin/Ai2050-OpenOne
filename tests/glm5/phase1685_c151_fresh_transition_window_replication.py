#!/usr/bin/env python3
"""C151: fresh-lexicon prospective replication of the C150 transition window."""
from __future__ import annotations
import gc,itertools,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1685_c151_fresh_transition_window_replication";C143=R/"phase1677_c143_transition_model_competition";C150=R/"phase1684_c150_predictable_transition_window_atlas";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1677_c143_transition_model_competition as c143
import phase1661_c127_typed_transition_language_family as c127
PHASE,CAMPAIGN=1685,"C151";WINDOW=tuple(range(24,34));STATES=tuple(range(24,35));ARMS=c141.ARMS;ROLES=c141.ROLES;DIM=2560;WIDTH=224;BATCH=4
def now():return datetime.now(timezone.utc).isoformat()
def material():
 units=[];cases=[]
 for ai,arm in enumerate(ARMS):
  for local in range(4):
   unit=40+ai*4+local;units.append({"unit_id":f"c151-{arm}-{local:02d}","source_unit":unit,"arm":arm})
   for f1,f2,f3,surface,code in itertools.product((1,-1),repeat=5):
    row=c141.make_case(arm,unit,f1,f2,f3,surface,code);row.update({"case_id":f"c151-{len(cases):05d}","unit_id":f"c151-{arm}-{local:02d}","source_unit":unit,"partition":"fresh_prospective"});cases.append(row)
 return units,cases
def contract():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C150/"audit/independent_closure_audit.json");units,cases=material();compiled=c141.compile_rows(graph_base.tokenizer(),cases);cells={}
 for r in cases:
  k=(r["arm"],r["unit_id"],r["factors"]["f1"],r["factors"]["f2"],r["factors"]["f3"],r["surface_factor"],r["codebook_factor"]);cells[k]=cells.get(k,0)+1
 checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="memo_and_fresh_prospective_window_replication","units":len(units)==20,"cases":len(cases)==640,"unique":len({r["prompt"] for r in cases})==640,"cells":len(cells)==640 and set(cells.values())=={1},"balance":sum(r["gold_position"]==0 for r in cases)==320,"roles":all(set(r["role_positions"])==set(ROLES) for r in compiled),"width":max(len(r["prompt_ids"]) for r in compiled)<WIDTH}
 if not all(checks.values()):raise RuntimeError(checks)
 OUT.mkdir(parents=True);core.write_rows(OUT/"material/units.jsonl",units);core.write_rows(OUT/"material/cases.jsonl",cases);core.write_rows(OUT/"compiled/qwen3.jsonl",compiled);p={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"fresh_window_contract_frozen","execution_model":"Qwen3-4B BF16 CUDA nonquantized","cases":640,"window_transitions":list(WINDOW),"saved_states":list(STATES),"predictor_model":"linear_kernel","lambda":.01,"training_asset":str(C143/"analysis/discovery_primary_trajectories.float32.npy"),"gate":core.load(C150/"protocol/frozen_window.json")["observation_checks"],"behavior_policy":"descriptive; errors retained and do not stop capture","claim_boundary":"prospective local effective transition replication, not semantic transport or a unique circuit","source_hashes":{"C150":core.sha(C150/"protocol/frozen_window.json"),"training":core.sha(C143/"analysis/discovery_primary_trajectories.float32.npy")},"producer_sha256":core.sha(Path(__file__)),"authorization":"run_C151_qwen"};core.save(OUT/"protocol/preregistration.json",p);core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":p["authorization"]});print(json.dumps({"checks":checks,"max_width":max(len(r["prompt_ids"]) for r in compiled)},indent=2))
def tensor(x):return x[0] if isinstance(x,tuple) else x
@torch.inference_mode()
def run():
 p=core.load(OUT/"protocol/preregistration.json");rows=core.rows(OUT/"compiled/qwen3.jsonl");path=OUT/"raw/qwen3_window_role_field.bf16.npy";path.parent.mkdir(parents=True,exist_ok=True);raw=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint16,shape=(640,6,11,DIM));logits=np.lib.format.open_memmap(OUT/"raw/qwen3_candidate_logits.float32.npy",mode="w+",dtype=np.float32,shape=(640,2));result=[];model=None;repeat=0
 try:
  model,tok,dev,place=load_bf16("qwen3");quant=quantization_audit(model);base=model.model;embed,layers,norm=base.embed_tokens,base.layers,base.norm;pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  def batch_run(batch):
   cap={};hooks=[embed.register_forward_hook(lambda _m,_a,o:cap.__setitem__(0,tensor(o).detach()))]+[layer.register_forward_hook(lambda _m,_a,o,j=i+1:cap.__setitem__(j,tensor(o).detach())) for i,layer in enumerate(layers)]+[norm.register_forward_hook(lambda _m,_a,o:cap.__setitem__(37,tensor(o).detach()))]
   try:ids,mask,pos,lens=fixed_base.fixed_batch(batch,pad,dev,WIDTH);out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
   finally:
    for h in hooks:h.remove()
   return cap,out,ids,mask,pos,lens
  for st in range(0,640,BATCH):
   batch=rows[st:st+BATCH];cap,o,ids,mask,pos,lens=batch_run(batch);scores=np.asarray([[float(o.logits[i,lens[i]-1,c[0]]) for c in r["candidate_ids"]] for i,r in enumerate(batch)],np.float32);logits[st:st+len(batch)]=scores
   for i,r in enumerate(batch):
    pred=int(scores[i,1]>scores[i,0]);result.append({"row_index":st+i,"case_id":r["case_id"],"unit_id":r["unit_id"],"arm":r["arm"],"factors":r["factors"],"surface_factor":r["surface_factor"],"codebook_factor":r["codebook_factor"],"gold_position":r["gold_position"],"prediction":pred,"correct":pred==r["gold_position"]})
    for ri,role in enumerate(ROLES):
     for sj,q in enumerate(STATES):raw[st+i,ri,sj]=cap[q][i,r["role_positions"][role]].mean(0).contiguous().view(torch.uint16).cpu().numpy()
   if (st//BATCH+1)%40==0:raw.flush();logits.flush();print(f"[C151] {st+len(batch)}/640",flush=True)
   del cap,o,ids,mask,pos
  raw.flush();logits.flush();cap,o,ids,mask,pos,lens=batch_run(rows[:BATCH]);check=np.asarray([[float(o.logits[i,lens[i]-1,c[0]]) for c in r["candidate_ids"]] for i,r in enumerate(rows[:BATCH])],np.float32);repeat=float(np.max(np.abs(check-np.asarray(logits[:BATCH]))))
 finally:
  raw.flush();logits.flush()
  if model is not None:release_bf16(model)
  gc.collect();torch.cuda.empty_cache()
 core.write_rows(OUT/"raw/qwen3_behavior_index.jsonl",result);behavior={"global":float(np.mean([r["correct"] for r in result])),"arm":{a:float(np.mean([r["correct"] for r in result if r["arm"]==a])) for a in ARMS}};checks={"rows":len(result)==640,"shape":list(raw.shape)==[640,6,11,DIM],"finite":bool(np.isfinite(logits).all()),"repeat":repeat==0,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]};report={"phase":PHASE,"campaign":CAMPAIGN,"status":"fresh_window_capture_complete","behavior":behavior,"checks":checks,"repeat_logits_max_abs":repeat,"role_sha256":core.sha(path),"authorization":"analyze_C151"};core.save(OUT/"analysis/capture.json",report);core.save(OUT/"audit/internal_capture_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":report["authorization"]});print(json.dumps(report,indent=2))
def analyze():
 p=core.load(OUT/"protocol/preregistration.json");rows=core.rows(OUT/"compiled/qwen3.jsonl");raw=np.load(OUT/"raw/qwen3_window_role_field.bf16.npy",mmap_mode="r");d=np.load(C143/"analysis/discovery_primary_trajectories.float32.npy",mmap_mode="r");keys=[]
 for arm in ARMS:
  for unit in range(4):
   for surface in (1,-1):
    for code in (1,-1):keys.append({"arm":arm,"unit":unit,"surface":surface,"code":code})
 look={(k["arm"],k["unit"],k["surface"],k["code"]):i for i,k in enumerate(keys)};traj=np.zeros((80,11,6,DIM),np.float32)
 for i,r in enumerate(rows):
  unit=int(r["unit_id"].rsplit("-",1)[1]);traj[look[(r["arm"],unit,r["surface_factor"],r["codebook_factor"])]]+=float(r["factors"]["f1"])*c127.decode(raw[i]).transpose(1,0,2)/8
 np.save(OUT/"analysis/fresh_f1_window_trajectories.float32.npy",traj);core.write_rows(OUT/"analysis/fresh_trajectory_index.jsonl",keys);trs=[];coord=[]
 for j,q in enumerate(WINDOW):
  xd,yd=c143.xy(d,q);x=traj[:,j].reshape(80,-1);y=(traj[:,j+1]-traj[:,j]).reshape(80,-1);pred=c143.fit_predict("linear_kernel",xd,yd,x,.01);target=c143.metrics(pred,y);wr=c143.metrics(np.roll(pred.reshape(80,6,DIM),1,axis=1).reshape(80,-1),y);wc=c143.metrics(np.roll(pred.reshape(80,6,DIM),1,axis=2).reshape(80,-1),y);arm={a:c143.metrics(pred[np.asarray([i for i,k in enumerate(keys) if k["arm"]==a])],y[np.asarray([i for i,k in enumerate(keys) if k["arm"]==a])]) for a in ARMS};trs.append({"transition":q,"target":target,"wrong_role":wr,"wrong_coordinate":wc,"arm":arm});coord += [{"dataset":"C151","kind":"predicted_increment","role":"boundary","transition_index":q,"checkpoint":f"{q}->{q+1}","values":pred.reshape(80,6,DIM)[:,5].mean(0).astype(np.float32).tolist()},{"dataset":"C151","kind":"target_increment","role":"boundary","transition_index":q,"checkpoint":f"{q}->{q+1}","values":y.reshape(80,6,DIM)[:,5].mean(0).astype(np.float32).tolist()}]
 current=traj[:,0].reshape(80,-1).copy();roll=[]
 for j,q in enumerate(WINDOW):
  xd,yd=c143.xy(d,q);current+=c143.fit_predict("linear_kernel",xd,yd,current,.01);target=traj[:,j+1].reshape(80,-1);identity=traj[:,0].reshape(80,-1);m=c143.metrics(current,target);im=c143.metrics(identity,target);roll.append({"q":q,"rollout":m,"identity":im,"ratio":m["relative_error"]/max(im["relative_error"],1e-12)})
 cfg=p["gate"];per={a:float(np.median([r["arm"][a]["cosine"] for r in trs])) for a in ARMS};mr=float(np.median([r["wrong_role"]["relative_error"]-r["target"]["relative_error"] for r in trs]));mc=float(np.median([r["wrong_coordinate"]["relative_error"]-r["target"]["relative_error"] for r in trs]));g={"each_transition":all(r["target"]["relative_error"]<=cfg["confirmation_each_error_max"] and r["target"]["cosine"]>=cfg["confirmation_each_cosine_min"] for r in trs),"arm_breadth":all(v>=cfg["per_arm_median_cosine_min"] for v in per.values()),"wrong_role":mr>=cfg["wrong_control_median_margin_min"],"wrong_coordinate":mc>=cfg["wrong_control_median_margin_min"],"rollout":roll[-1]["ratio"]<=cfg["local_rollout_final_ratio_vs_identity_max"]};report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"fresh_window_replication_adjudicated","transition_rows":trs,"per_arm_median_cosine":per,"control_margins":{"wrong_role":mr,"wrong_coordinate":mc},"rollout_rows":roll,"gates":g,"prospective_window_gate_passed":all(g.values()),"claim_boundary":p["claim_boundary"],"authorization":"close_C151"};core.save(OUT/"analysis/confirmation.json",report);core.save(OUT/"analysis/coordinate_rows.json",coord);checks={"shape":list(traj.shape)==[80,11,6,DIM],"rows":len(trs)==10,"coords":len(coord)==20 and all(len(x["values"])==DIM for x in coord),"finite":bool(np.isfinite(traj).all())};core.save(OUT/"audit/internal_analysis_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_gate_passed":all(g.values()),"authorization":report["authorization"]});print(json.dumps({"behavior":core.load(OUT/"analysis/capture.json")["behavior"],"metrics":[{"q":r["transition"],**r["target"]} for r in trs],"per_arm":per,"margins":report["control_margins"],"final_rollout_ratio":roll[-1]["ratio"],"gates":g},indent=2))
def close():
 r=core.load(OUT/"analysis/confirmation.json");coord=core.load(OUT/"analysis/coordinate_rows.json");payload=core.load(PUBLIC);payload["c151_fresh_transition_window"]={"capture":core.load(OUT/"analysis/capture.json"),"confirmation":r,"coordinate_rows":coord};payload.update({"phase":PHASE,"campaign":"C109-C151","title":"Role-State Atlas + Fresh Transition Window Replication","created_at_utc":now()});canonical=OUT/"analysis/c109_c151_atlas.json";core.save(canonical,payload);shutil.copyfile(canonical,PUBLIC);checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"capture":core.load(OUT/"audit/internal_capture_audit.json")["all_checks_passed"],"analysis":core.load(OUT/"audit/internal_analysis_audit.json")["all_checks_passed"],"asset":core.sha(canonical)==core.sha(PUBLIC)};cl={"phase":PHASE,"campaign":CAMPAIGN,"status":"fresh_transition_window_closed","headline":{"gate_passed":r["prospective_window_gate_passed"],"behavior":core.load(OUT/"analysis/capture.json")["behavior"],"per_arm":r["per_arm_median_cosine"],"final_rollout_ratio":r["rollout_rows"][-1]["ratio"]},"claim_boundary":r["claim_boundary"],"next_authorization":"local causal contract only if gate passed; otherwise continue observational transition-object search"};core.save(OUT/"analysis/closure.json",cl);core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"asset_sha256":core.sha(PUBLIC),"authorization":"independent_final_and_memo"});print(json.dumps(cl,indent=2))
def main():
 modes={"contract":contract,"run":run,"analyze":analyze,"close":close}
 if len(sys.argv)!=2 or sys.argv[1] not in modes:raise SystemExit("contract|run|analyze|close")
 modes[sys.argv[1]]()
if __name__=="__main__":main()
