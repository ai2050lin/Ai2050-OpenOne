#!/usr/bin/env python3
"""C150: retrospective atlas of discovery-defined predictable transition windows."""
from __future__ import annotations
import json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1684_c150_predictable_transition_window_atlas";C143=R/"phase1677_c143_transition_model_competition";C149=R/"phase1683_c149_arm_specific_transition_system_identification";PUBLIC=ROOT/"frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1677_c143_transition_model_competition as c143
PHASE,CAMPAIGN=1684,"C150";ARMS=c141.ARMS;ROLE="boundary";RI=c141.ROLES.index(ROLE)
def now():return datetime.now(timezone.utc).isoformat()
def longest(values):
 best=[];cur=[]
 for q in range(37):
  if q in values:cur.append(q)
  else:
   if len(cur)>len(best):best=cur
   cur=[]
 if len(cur)>len(best):best=cur
 return best
def discover():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C149/"audit/independent_closure_audit.json");source=core.load(C143/"protocol/frozen_model.json")["inner_discovery_rows"]["linear_kernel"];eligible=[i for i,r in enumerate(source) if r["relative_error"]<=.75 and r["cosine"]>=.70];window=longest(set(eligible));OUT.mkdir(parents=True);(OUT/"protocol").mkdir();(OUT/"analysis").mkdir();(OUT/"audit").mkdir()
 freeze={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"retrospective_discovery_window_frozen","model":"linear_kernel","lambda":.01,"selection":{"relative_error_max":.75,"cosine_min":.70,"rule":"longest contiguous discovery segment, earliest on tie"},"eligible_transitions":eligible,"window":window,"window_from_checkpoint":window[0] if window else None,"window_to_checkpoint":window[-1]+1 if window else None,"observation_checks":{"confirmation_each_error_max":.80,"confirmation_each_cosine_min":.65,"per_arm_median_cosine_min":.50,"wrong_control_median_margin_min":.05,"local_rollout_final_ratio_vs_identity_max":.90},"epistemic_status":"retrospective because C143 confirmation was previously opened; cannot serve as a new prospective confirmation","source_hashes":{"C143_freeze":core.sha(C143/"protocol/frozen_model.json"),"confirmation":core.sha(C143/"analysis/confirmation_primary_trajectories.float32.npy")},"claim_boundary":"local effective transition window, not semantic transport, a unique circuit, or causal authorization","producer_sha256":core.sha(Path(__file__)),"authorization":"observe_C150_confirmation_window"}
 core.save(OUT/"protocol/frozen_window.json",freeze);checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="memo_and_next_stage_assessment","window":len(window)>=2,"contiguous":all(b==a+1 for a,b in zip(window,window[1:])),"discovery_only_rule":all(source[q]["relative_error"]<=.75 and source[q]["cosine"]>=.70 for q in window),"retrospective":freeze["epistemic_status"].startswith("retrospective")};core.save(OUT/"audit/internal_discovery_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":freeze["authorization"]});print(json.dumps({"checks":checks,"eligible":eligible,"window":window},indent=2))
def observe():
 freeze=core.load(OUT/"protocol/frozen_window.json");d=np.load(C143/"analysis/discovery_primary_trajectories.float32.npy",mmap_mode="r");c=np.load(C143/"analysis/confirmation_primary_trajectories.float32.npy",mmap_mode="r");dk=core.rows(C143/"analysis/discovery_sample_index.jsonl");ck=core.rows(C143/"analysis/confirmation_sample_index.jsonl");window=freeze["window"];rows=[];coord=[];pred_cache={}
 for q in window:
  xd,yd=c143.xy(d,q);xc,yc=c143.xy(c,q);p=c143.fit_predict("linear_kernel",xd,yd,xc,.01);pred_cache[q]=p;target=c143.metrics(p,yc);wr=c143.metrics(np.roll(p.reshape(80,6,2560),1,axis=1).reshape(80,-1),yc);wc=c143.metrics(np.roll(p.reshape(80,6,2560),1,axis=2).reshape(80,-1),yc);arm={}
  for name in ARMS:
   idx=np.asarray([i for i,k in enumerate(ck) if k["arm"]==name]);arm[name]=c143.metrics(p[idx],yc[idx])
  rows.append({"transition":q,"target":target,"wrong_role":wr,"wrong_coordinate":wc,"arm":arm})
  pv=p.reshape(80,6,2560)[:,RI].mean(0);tv=yc.reshape(80,6,2560)[:,RI].mean(0)
  coord += [{"dataset":"C150","kind":"predicted_increment","role":ROLE,"transition_index":q,"checkpoint":f"{q}->{q+1}","values":pv.astype(np.float32).tolist()},{"dataset":"C150","kind":"target_increment","role":ROLE,"transition_index":q,"checkpoint":f"{q}->{q+1}","values":tv.astype(np.float32).tolist()}]
 start=window[0];current=np.asarray(c[:,start]).reshape(80,-1).copy();roll=[]
 for q in window:
  xd,yd=c143.xy(d,q);delta=c143.fit_predict("linear_kernel",xd,yd,current,.01);current+=delta;target=np.asarray(c[:,q+1]).reshape(80,-1);identity=np.asarray(c[:,start]).reshape(80,-1);m=c143.metrics(current,target);im=c143.metrics(identity,target);roll.append({"transition":q,"rollout":m,"identity":im,"ratio":m["relative_error"]/max(im["relative_error"],1e-12)})
 checks_cfg=freeze["observation_checks"];per_arm={a:float(np.median([r["arm"][a]["cosine"] for r in rows])) for a in ARMS};margins={"wrong_role":float(np.median([r["wrong_role"]["relative_error"]-r["target"]["relative_error"] for r in rows])),"wrong_coordinate":float(np.median([r["wrong_coordinate"]["relative_error"]-r["target"]["relative_error"] for r in rows]))};gates={"each_transition":all(r["target"]["relative_error"]<=checks_cfg["confirmation_each_error_max"] and r["target"]["cosine"]>=checks_cfg["confirmation_each_cosine_min"] for r in rows),"arm_breadth":all(v>=checks_cfg["per_arm_median_cosine_min"] for v in per_arm.values()),"wrong_role":margins["wrong_role"]>=checks_cfg["wrong_control_median_margin_min"],"wrong_coordinate":margins["wrong_coordinate"]>=checks_cfg["wrong_control_median_margin_min"],"local_rollout":roll[-1]["ratio"]<=checks_cfg["local_rollout_final_ratio_vs_identity_max"]}
 report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"retrospective_transition_window_observed","window":window,"transition_rows":rows,"per_arm_median_cosine":per_arm,"control_margins":margins,"rollout_rows":roll,"observation_checks":gates,"window_observation_consistent":all(gates.values()),"epistemic_status":freeze["epistemic_status"],"claim_boundary":freeze["claim_boundary"],"authorization":"close_C150"};core.save(OUT/"analysis/window_observation.json",report);core.save(OUT/"analysis/coordinate_rows.json",coord);checks={"window_rows":len(rows)==len(window),"coordinate_rows":len(coord)==2*len(window),"full_coordinates":all(len(x["values"])==2560 for x in coord),"arms":all(set(x["arm"])==set(ARMS) for x in rows),"rollout":len(roll)==len(window),"finite":all(np.isfinite(x["target"]["relative_error"]) for x in rows)};core.save(OUT/"audit/internal_observation_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_status":"retrospective-consistent" if all(gates.values()) else "retrospective-mixed","authorization":report["authorization"]});print(json.dumps({"checks":checks,"window":window,"transition_metrics":[{"q":r["transition"],**r["target"]} for r in rows],"per_arm":per_arm,"margins":margins,"final_rollout_ratio":roll[-1]["ratio"],"observation_checks":gates},indent=2))
def close():
 r=core.load(OUT/"analysis/window_observation.json");coord=core.load(OUT/"analysis/coordinate_rows.json");payload=core.load(PUBLIC);payload["c149_c150_transition_window"]={"c149":core.load(C149/"analysis/closure.json"),"c150":r,"coordinate_rows":coord};payload.update({"phase":PHASE,"campaign":"C109-C150","title":"Role-State Atlas + Multi-Family Predictable Transition Window","created_at_utc":now(),"claim_boundary":payload["claim_boundary"]+" C150 adds a retrospective discovery-defined local transition window; it does not alter earlier failed prospective gates or authorize causality."});canonical=OUT/"analysis/c109_c150_transition_window_atlas.json";core.save(canonical,payload);shutil.copyfile(canonical,PUBLIC)
 checks={"discovery":core.load(OUT/"audit/internal_discovery_audit.json")["all_checks_passed"],"observation":core.load(OUT/"audit/internal_observation_audit.json")["all_checks_passed"],"asset":core.sha(canonical)==core.sha(PUBLIC),"rows":len(coord)==2*len(r["window"]),"retrospective":"retrospective" in r["epistemic_status"]};cl={"phase":PHASE,"campaign":CAMPAIGN,"status":"predictable_window_atlas_closed","headline":{"window":r["window"],"consistent":r["window_observation_consistent"],"per_arm_median_cosine":r["per_arm_median_cosine"],"final_rollout_ratio":r["rollout_rows"][-1]["ratio"]},"claim_boundary":r["claim_boundary"],"next_authorization":"prospective fresh lexical replication of the frozen local window; causal remains closed"};core.save(OUT/"analysis/closure.json",cl);core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"asset_sha256":core.sha(PUBLIC),"authorization":"independent_final_and_memo"});print(json.dumps(cl,indent=2))
def main():
 modes={"discover":discover,"observe":observe,"close":close}
 if len(sys.argv)!=2 or sys.argv[1] not in modes:raise SystemExit("discover|observe|close")
 modes[sys.argv[1]]()
if __name__=="__main__":main()
