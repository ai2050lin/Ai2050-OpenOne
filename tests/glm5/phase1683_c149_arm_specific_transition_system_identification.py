#!/usr/bin/env python3
"""C149: arm-specific versus cross-arm transition system identification."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];T=ROOT/"tests/glm5";R=T/"result";OUT=R/"phase1683_c149_arm_specific_transition_system_identification";C143=R/"phase1677_c143_transition_model_competition";C148=R/"phase1682_c148_campaign_synthesis_heatmap_and_closure";sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core
import phase1675_c141_multifamily_full_coordinate_atlas as c141
import phase1677_c143_transition_model_competition as c143
PHASE,CAMPAIGN=1683,"C149";ARMS=c141.ARMS;MODELS=c143.MODELS;LAMBDAS=c143.LAMBDAS;CAL=c143.CALIBRATION_TRANSITIONS
def now():return datetime.now(timezone.utc).isoformat()
def candidates():
 out=[]
 for name in MODELS:
  for lam in (LAMBDAS if name in {"diagonal_ridge","linear_kernel","quadratic_kernel"} else (None,)):out.append((name,lam))
 return out
def indices(keys,arm=None,unit=None,exclude_arm=None):
 return np.asarray([i for i,k in enumerate(keys) if (arm is None or k["arm"]==arm) and (unit is None or k["unit"]==unit) and (exclude_arm is None or k["arm"]!=exclude_arm)],int)
def score_candidate(traj,train,test,name,lam,transitions=CAL):
 vals=[]
 for q in transitions:
  x,y=c143.xy(traj,q);p=c143.fit_predict(name,x[train],y[train],x[test],lam);vals.append(c143.metrics(p,y[test]))
 return {"mean_relative_error":float(np.mean([v["relative_error"] for v in vals])),"median_cosine":float(np.median([v["cosine"] for v in vals]))}
def discover():
 if OUT.exists():raise RuntimeError(OUT)
 parent=core.load(C148/"audit/independent_closure_audit.json");traj=np.load(C143/"analysis/discovery_primary_trajectories.float32.npy",mmap_mode="r");keys=core.rows(C143/"analysis/discovery_sample_index.jsonl");OUT.mkdir(parents=True);(OUT/"protocol").mkdir();(OUT/"analysis").mkdir();(OUT/"audit").mkdir()
 arm_selection={}
 for arm in ARMS:
  rows=[];units=sorted({k["unit"] for k in keys if k["arm"]==arm})
  for name,lam in candidates():
   fold=[]
   for unit in units:
    test=indices(keys,arm=arm,unit=unit);train=np.asarray([i for i,k in enumerate(keys) if k["arm"]==arm and k["unit"]!=unit],int);fold.append(score_candidate(traj,train,test,name,lam))
   rows.append({"model":name,"lambda":lam,"mean_relative_error":float(np.mean([x["mean_relative_error"] for x in fold])),"median_fold_cosine":float(np.median([x["median_cosine"] for x in fold]))})
  winner=min(rows,key=lambda x:x["mean_relative_error"]);arm_selection[arm]={"winner":winner,"candidates":rows}
 universal=[]
 for name,lam in candidates():
  folds=[]
  for arm in ARMS:
   test=indices(keys,arm=arm);train=indices(keys,exclude_arm=arm);folds.append(score_candidate(traj,train,test,name,lam))
  universal.append({"model":name,"lambda":lam,"mean_relative_error":float(np.mean([x["mean_relative_error"] for x in folds])),"median_fold_cosine":float(np.median([x["median_cosine"] for x in folds]))})
 uw=min(universal,key=lambda x:x["mean_relative_error"])
 freeze={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"nested_discovery_models_frozen","object":"f1 response residual increment","arm_specific":arm_selection,"universal":{"winner":uw,"candidates":universal},"selection":"arm: leave-one-unit-out; universal: leave-one-arm-out; calibration transitions only","confirmation_gate":{"relative_error_ratio_vs_zero_max":.90,"cosine_min":.50,"wrong_arm_error_margin_min":.05,"rollout_ratio_vs_identity_max":.95},"confirmation_unread":True,"source_hashes":{"discovery":core.sha(C143/"analysis/discovery_primary_trajectories.float32.npy"),"confirmation_sealed":core.sha(C143/"analysis/confirmation_primary_trajectories.float32.npy")},"claim_boundary":"bounded effective response prediction; arm-specific success is not a semantic operator or unique coordinate graph","producer_sha256":core.sha(Path(__file__)),"authorization":"validate_C149"}
 core.save(OUT/"protocol/frozen_models.json",freeze);checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="memo_and_next_observation_campaign","shape":list(traj.shape)==[80,38,6,2560],"arms":len(arm_selection)==5,"candidate_count":all(len(v["candidates"])==16 for v in arm_selection.values()) and len(universal)==16,"finite":all(np.isfinite(v["winner"]["mean_relative_error"]) for v in arm_selection.values()),"sealed":len(freeze["source_hashes"]["confirmation_sealed"])==64};core.save(OUT/"audit/internal_discovery_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":freeze["authorization"]});print(json.dumps({"checks":checks,"arm_winners":{a:v["winner"] for a,v in arm_selection.items()},"universal_winner":uw},indent=2))
def evaluate_model(discovery,confirmation,train,test,name,lam):
 rows=[];current=np.asarray(confirmation[test,0]).reshape(len(test),-1).copy();roll=[]
 for q in range(37):
  xd,yd=c143.xy(discovery,q);xc,yc=c143.xy(confirmation,q);p=c143.fit_predict(name,xd[train],yd[train],xc[test],lam);rows.append(c143.metrics(p,yc[test]));delta=c143.fit_predict(name,xd[train],yd[train],current,lam);current+=delta;roll.append(c143.metrics(current,np.asarray(confirmation[test,q+1]).reshape(len(test),-1)))
 return rows,roll
def med(rows,key):return float(np.median([x[key] for x in rows]))
def validate():
 freeze=core.load(OUT/"protocol/frozen_models.json");d=np.load(C143/"analysis/discovery_primary_trajectories.float32.npy",mmap_mode="r");c=np.load(C143/"analysis/confirmation_primary_trajectories.float32.npy",mmap_mode="r");dk=core.rows(C143/"analysis/discovery_sample_index.jsonl");ck=core.rows(C143/"analysis/confirmation_sample_index.jsonl")
 if core.sha(C143/"analysis/confirmation_primary_trajectories.float32.npy")!=freeze["source_hashes"]["confirmation_sealed"]:raise RuntimeError("confirmation drift")
 results={};gate=freeze["confirmation_gate"]
 for ai,arm in enumerate(ARMS):
  w=freeze["arm_specific"][arm]["winner"];train=indices(dk,arm=arm);test=indices(ck,arm=arm);rows,roll=evaluate_model(d,c,train,test,w["model"],w["lambda"])
  wrong_arm=ARMS[(ai+1)%len(ARMS)];wrong_train=indices(dk,arm=wrong_arm);wrong=[]
  for q in range(37):
   xd,yd=c143.xy(d,q);xc,yc=c143.xy(c,q);wrong.append(c143.metrics(c143.fit_predict(w["model"],xd[wrong_train],yd[wrong_train],xc[test],w["lambda"]),yc[test]))
  err=med(rows,"relative_error");zero=1.0;cos=med(rows,"cosine");wrong_err=med(wrong,"relative_error");roll_err=med(roll,"relative_error");identity=float(np.median([np.linalg.norm(c[test,0].reshape(len(test),-1)-c[test,q+1].reshape(len(test),-1))/max(np.linalg.norm(c[test,q+1]),1e-12) for q in range(37)]));derived={"median_relative_error":err,"median_cosine":cos,"relative_error_ratio_vs_zero":err/zero,"wrong_arm_error":wrong_err,"wrong_arm_error_margin":wrong_err-err,"rollout_median_relative_error":roll_err,"rollout_ratio_vs_identity":roll_err/max(identity,1e-12)};g={"error":derived["relative_error_ratio_vs_zero"]<=gate["relative_error_ratio_vs_zero_max"],"cosine":cos>=gate["cosine_min"],"wrong_arm":derived["wrong_arm_error_margin"]>=gate["wrong_arm_error_margin_min"],"rollout":derived["rollout_ratio_vs_identity"]<=gate["rollout_ratio_vs_identity_max"]};results[arm]={"winner":w,"derived":derived,"gates":g,"passed":all(g.values()),"transition_rows":rows}
 uw=freeze["universal"]["winner"];all_idx=np.arange(80);urows,uroll=evaluate_model(d,c,all_idx,all_idx,uw["model"],uw["lambda"]);universal={"winner":uw,"median_relative_error":med(urows,"relative_error"),"median_cosine":med(urows,"cosine"),"rollout_median_relative_error":med(uroll,"relative_error"),"transition_rows":urows}
 report={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":now(),"status":"arm_specific_transition_confirmation_adjudicated","arm_results":results,"passing_arms":[a for a,v in results.items() if v["passed"]],"universal":universal,"claim_boundary":freeze["claim_boundary"],"authorization":"close_C149"};core.save(OUT/"analysis/confirmation.json",report);checks={"arms":len(results)==5,"rows":all(len(v["transition_rows"])==37 for v in results.values()),"universal_rows":len(urows)==37,"finite":all(np.isfinite(v["derived"]["median_relative_error"]) for v in results.values()),"freeze":all(results[a]["winner"]==freeze["arm_specific"][a]["winner"] for a in ARMS)};core.save(OUT/"audit/internal_confirmation_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"scientific_passing_arms":report["passing_arms"],"authorization":report["authorization"]});print(json.dumps({"checks":checks,"passing_arms":report["passing_arms"],"arm_results":{a:{"winner":v["winner"],"derived":v["derived"],"gates":v["gates"]} for a,v in results.items()},"universal":{k:v for k,v in universal.items() if k!="transition_rows"}},indent=2))
def close():
 r=core.load(OUT/"analysis/confirmation.json");checks={"discovery":core.load(OUT/"audit/internal_discovery_audit.json")["all_checks_passed"],"confirmation":core.load(OUT/"audit/internal_confirmation_audit.json")["all_checks_passed"],"typed":len(r["arm_results"])==5};cl={"phase":PHASE,"campaign":CAMPAIGN,"status":"arm_specific_transition_campaign_closed","headline":{"passing_arms":r["passing_arms"],"universal":{k:v for k,v in r["universal"].items() if k!="transition_rows"}},"theory_update":"separates arm-specific predictability from cross-arm universality under nested lexical-unit selection","claim_boundary":r["claim_boundary"],"next_authorization":"independent audit and memo; causal authorization remains arm-specific and requires every frozen gate"};core.save(OUT/"analysis/closure.json",cl);core.save(OUT/"audit/internal_closure_audit.json",{"checks":checks,"all_checks_passed":all(checks.values()),"authorization":"independent_final_and_memo"});print(json.dumps(cl,indent=2))
def main():
 modes={"discover":discover,"validate":validate,"close":close}
 if len(sys.argv)!=2 or sys.argv[1] not in modes:raise SystemExit("discover|validate|close")
 modes[sys.argv[1]]()
if __name__=="__main__":main()
