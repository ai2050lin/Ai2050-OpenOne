#!/usr/bin/env python3
"""C139 synthesis, causal authorization audit, and parameter-level heatmap export."""
from __future__ import annotations
import json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1673_c139_campaign_synthesis_and_heatmap";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
C133=R/"phase1667_c133_multiroute_campaign_contract";C134=R/"phase1668_c134_route_a_directed_composition";C135=R/"phase1669_c135_all_token_coordinate_transmission";C136=R/"phase1670_c136_chinese_pattern_composition_field";C137=R/"phase1671_c137_type_graph_response_ecology";C138=R/"phase1672_c138_prospective_cross_model_topology"
CHECKPOINTS=c127.CHECKPOINTS
def now():return datetime.now(timezone.utc).isoformat()
def c135_rows():
 gain=np.load(C135/"protocol/frozen_diagonal_gain.float32.npy",mmap_mode="r");freeze=core.load(C135/"protocol/frozen_transmission.json");grows=[]
 for si,length in enumerate(freeze["length_strata"]):
  for q in (0,8,16,24,32,35,36):
   norms=np.linalg.norm(gain[si,q,:length],axis=1);token=int(np.argmax(norms));grows.append({"dataset":"C135","kind":"diagonal_gain","length_stratum":length,"token_position":token,"checkpoint_index":q,"checkpoint":f"{CHECKPOINTS[q]}->{CHECKPOINTS[q+1]}","values":np.asarray(gain[si,q,token],np.float32).tolist()})
 field=np.load(C135/"raw/qwen3_all_token_all_checkpoint.bf16.npy",mmap_mode="r");index=core.rows(C135/"raw/all_token_field_index.jsonl");anchors=core.rows(C135/"material/anchors.jsonl");idx=index[0];row=anchors[0];raw=[]
 for role in ("left_fact","boundary"):
  local=row["role_positions"][role]
  for q in (0,8,16,24,32,36,37):
   values=c127.decode(field[q,idx["token_offset_start"]+np.asarray(local)]).mean(0);raw.append({"dataset":"C135","kind":"raw_state","case_id":row["case_id"],"role":role,"checkpoint_index":q,"checkpoint":CHECKPOINTS[q],"token_positions":local,"values":values.tolist()})
 return grows,raw
def c136_rows():
 freeze=core.load(C136/"protocol/frozen_patterns.json");disc=np.load(C136/"analysis/discovery_pattern_fields.float32.npy",mmap_mode="r");conf=np.load(C136/"analysis/confirmation_pattern_fields.float32.npy",mmap_mode="r");tasks=list(freeze["nominees"]);response=[]
 for ti,task in enumerate(tasks):
  n=freeze["nominees"][task]
  for partition,field in (("discovery",disc),("confirmation",conf)):
   response.append({"dataset":"C136","kind":"truth_response","task":task,"partition":partition,"role":n["role"],"checkpoint_index":n["checkpoint_index"],"checkpoint":n["checkpoint"],"values":np.asarray(field[:,ti,n["role_index"],n["checkpoint_index"]].mean(0),np.float32).tolist()})
 rawbits=np.load(C136/"raw/qwen3_pattern_role_field.bf16.npy",mmap_mode="r");compiled=core.rows(C136/"compiled/qwen3.jsonl");raw=[]
 for ri,role in enumerate(("experiencer","agent","action","patient","boundary")):
  for q in (0,16,24,32,36,37):raw.append({"dataset":"C136","kind":"raw_state","case_id":compiled[0]["case_id"],"role":role,"checkpoint_index":q,"checkpoint":CHECKPOINTS[q],"token_positions":compiled[0]["role_positions"][role],"values":c127.decode(rawbits[0,ri,q]).tolist()})
 return response,raw
def c138_rows():
 raw=np.load(C138/"raw/qwen3_role_field.bf16.npy",mmap_mode="r");rows=core.rows(C138/"compiled/qwen3.jsonl");units=core.rows(C138/"material/units.jsonl");look={u["unit_id"]:i for i,u in enumerate(units)};field=np.zeros((32,2,4,38,2560),np.float32)
 for i,r in enumerate(rows):field[look[r["unit_id"]],("direct","two_hop").index(r["depth"])]+=float(r["truth_factor"])/4*c127.decode(raw[i])
 out=[]
 for di,depth in enumerate(("direct","two_hop")):
  for partition,part in (("discovery",field[:16]),("confirmation",field[16:])):
   for q in (0,16,24,32,36,37):out.append({"dataset":"C138","kind":"truth_response","depth":depth,"partition":partition,"role":"boundary","checkpoint_index":q,"checkpoint":CHECKPOINTS[q],"values":np.asarray(part[:,di,3,q].mean(0),np.float32).tolist()})
 return out
def main():
 if OUT.exists():raise RuntimeError(OUT)
 OUT.mkdir(parents=True);(OUT/"analysis").mkdir();(OUT/"audit").mkdir();(OUT/"visualization").mkdir()
 routes={
  "A":{"behavior":core.load(C134/"analysis/behavior.json")["gate_passed"],"discovery_prediction":False,"new_trajectory_prediction":False,"wrong_route_control":False},
  "B":{"behavior":True,"discovery_prediction":True,"new_trajectory_prediction":core.load(C135/"analysis/confirmation.json")["prediction_gate_passed"],"wrong_route_control":core.load(C135/"analysis/confirmation.json")["gates"]["wrong_token"] and core.load(C135/"analysis/confirmation.json")["gates"]["wrong_coordinate"]},
  "C":{"behavior":core.load(C136/"analysis/behavior.json")["gate_passed"],"discovery_prediction":True,"new_trajectory_prediction":False,"wrong_route_control":False},
  "D":{"behavior":core.load(C137/"analysis/behavior.json")["gate_passed"],"discovery_prediction":False,"new_trajectory_prediction":False,"wrong_route_control":False},
  "E":{"behavior":len(core.load(C138/"analysis/cross_model_synthesis.json")["qualified_models"])>=2,"discovery_prediction":True,"new_trajectory_prediction":False,"wrong_route_control":False},
 }
 for values in routes.values():values["causal_authorized"]=all(values.values())
 summary={"phase":1673,"campaign":"C139","created_at_utc":now(),"status":"five_route_campaign_closed","causal_gate_by_route":routes,"causal_intervention_run":False,"causal_reason":"no route satisfies all four frozen C133 prerequisites","route_results":{"A":"ordinary explicit paths strong; balanced edge factorial behavior failed","B":"all-token full-coordinate field captured; diagonal transmission failed confirmation","C":"eight Chinese pattern response graph replicated; dominated by late truth/output field","D":"short type paths strong; three-hop and edge-type behavior failed","E":"Qwen3 replicated; GLM4 and DeepSeek7B behavior interfaces failed"},"theory_update":"Stable observations favor a typed, context-conditioned, distributed response field with common late output preparation and role-specific exceptions; no unique transmission law or causal circuit is established.","claim_boundary":"observation campaign closure; no complete language encoding solution, causal circuit, semantic neuron set, cross-model invariant, or new mathematics","next_authorization":"new campaign should deep-mine cross-coordinate/high-order prediction and output-code separation before any intervention"};core.save(OUT/"analysis/campaign_synthesis.json",summary)
 gains,c135raw=c135_rows();c136resp,c136raw=c136_rows();c138resp=c138_rows();payload=core.load(PUBLIC);payload["c133_c139_observation_batch"]={"campaign_summary":summary,"c135":{"confirmation":core.load(C135/"analysis/confirmation.json"),"freeze":core.load(C135/"protocol/frozen_transmission.json"),"gain_rows":gains,"representative_raw_rows":c135raw},"c136":{"behavior":core.load(C136/"analysis/behavior.json")["summary"],"confirmation":core.load(C136/"analysis/confirmation.json"),"response_rows":c136resp,"representative_raw_rows":c136raw},"c138":{"synthesis":core.load(C138/"analysis/cross_model_synthesis.json"),"qwen_response_rows":c138resp}}
 payload.update({"phase":1673,"campaign":"C109-C139","title":"Role-State Atlas + Full-Token Transmission + Pattern Response Fields","created_at_utc":now(),"claim_boundary":"C135-C139 add all-actual-token BF16 fields, diagonal transmission controls, Chinese pattern responses, and prospective Qwen route responses. Every displayed column is a physical activation coordinate at embedding or a typed HiddenState checkpoint, not a weight parameter. C135 diagonal prediction failed; C136 is dominated by late truth/output preparation; C138 has Qwen3 internal data only; no attention/MLP mechanism, unique circuit, causal closure, cross-model coordinate identity, or new mathematics is claimed."})
 canonical=OUT/"visualization/c109_c139_observation_atlas.json";core.save(canonical,payload);shutil.copyfile(canonical,PUBLIC)
 checks={"route_count":len(routes)==5,"causal_closed":not any(v["causal_authorized"] for v in routes.values()),"rows":len(gains)==14 and len(c135raw)==14 and len(c136resp)==16 and len(c136raw)==30 and len(c138resp)==24,"full_coordinates":all(len(r["values"])==2560 for r in gains+c135raw+c136resp+c136raw+c138resp),"asset_match":core.sha(canonical)==core.sha(PUBLIC),"embedding_present":any(r["checkpoint_index"]==0 for r in c135raw+c136raw+c138resp),"hidden_present":any(r["checkpoint_index"]==36 for r in c135raw+c136raw+c138resp)};audit={"phase":1673,"campaign":"C139","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"scientific_causal_gate_passed":False,"asset_sha256":core.sha(PUBLIC),"authorization":"independent_audit_and_memo"};core.save(OUT/"audit/internal_closure_audit.json",audit);print(json.dumps({"summary":summary,"audit":audit,"asset_bytes":PUBLIC.stat().st_size},indent=2))
if __name__=="__main__":main()
