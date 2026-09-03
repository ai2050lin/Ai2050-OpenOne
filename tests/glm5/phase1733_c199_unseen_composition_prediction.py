#!/usr/bin/env python3
"""C199: predict held-out composite-program trajectories from atomic natural programs."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; OUT=RESULT/"phase1733_c199_unseen_composition_prediction"; C197=RESULT/"phase1731_c197_structure_model_tournament"; C198=RESULT/"phase1732_c198_broad_natural_program_trajectory"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core

PHASE,CAMPAIGN=1733,"C199"; DIM=2560
ATOMIC=("agent_patient","possession","comparison","causation","translation")
COMPOSITE=("attitude_event","contrast","negation","type_chain")
MODELS=("identity","graph_role_coordinate_gain","atomic_global_gain","atomic_role_gain","atomic_coordinate_gain","atomic_role_coordinate_gain")


def contract():
    if OUT.exists(): raise RuntimeError(OUT)
    parent=core.load(C198/"audit/independent_final_audit.json"); index=core.rows(C198/"raw/hidden_index.jsonl")
    fit=[r["anchor_index"] for r in index if r["program"] in ATOMIC and r["partition"]=="discovery" and r["behavior_correct"]]
    reveal=[r["anchor_index"] for r in index if r["program"] in COMPOSITE and r["partition"]=="fresh" and r["behavior_correct"]]
    checks={"authorization":parent["all_checks_passed"] and parent["authorization"]=="C199_unseen_composition_holdout_prediction","fit":len(fit)==20,"reveal":len(reveal)==8,"disjoint_programs":not set(ATOMIC)&set(COMPOSITE),"all_composites":{index[i]["program"] for i in reveal}==set(COMPOSITE)}
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True); core.save(OUT/"protocol/splits.json",{"fit":fit,"reveal":reveal})
    protocol={"phase":PHASE,"campaign":CAMPAIGN,"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"unseen_composition_prediction_frozen","fit_programs":list(ATOMIC),"held_out_composite_programs":list(COMPOSITE),"fit_partition":"discovery units only","reveal_partition":"fresh units only","models":list(MODELS),"fit":"zero-intercept gains clipped [-4,4]","primary_gate":{"aggregate_identity_nrmse_improvement_min":0.03,"aggregate_nrmse_max":0.75,"programs_better_than_identity_min":3},"semantic_composition_boundary":"This tests whether a checkpoint transform learned on atomic programs predicts held-out composite-program trajectories. It does not test W(composite)=W(b) o W(a), because the material does not isolate semantic operators with a common state domain.","forbidden":["attention","MLP","weights","PCA","fitting on composite programs","claiming a semantic composition law"],"producer_sha256":core.sha(Path(__file__)),"authorization":"C200_natural_deletion_and_typed_rescue"}; core.save(OUT/"protocol/preregistration.json",protocol); core.save(OUT/"audit/internal_contract_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"checks":checks,"models":list(MODELS)},indent=2))


def safe(num,den): return np.clip(np.divide(num,den,out=np.zeros_like(num,dtype=np.float64),where=den>1e-12),-4,4).astype(np.float32)


def metric(pred,truth):
    e=np.square(pred-truth,dtype=np.float64).sum(); t=np.square(truth,dtype=np.float64).sum(); w=np.minimum(np.abs(pred),np.abs(truth)).astype(np.float64); return {"nrmse":float(np.sqrt(e/max(t,1e-30))),"weighted_sign_agreement":float((w*(np.signbit(pred)==np.signbit(truth))).sum()/max(w.sum(),1e-30))}


def build():
    raw=np.load(C198/"raw/natural_signed_trajectory.float16.npy",mmap_mode="r"); index=core.rows(C198/"raw/hidden_index.jsonl"); splits=core.load(OUT/"protocol/splits.json")
    num=np.zeros((6,DIM),np.float64); den=np.zeros_like(num)
    for i in splits["fit"]:
        u=np.asarray(raw[i,:,0],dtype=np.float32); v=np.asarray(raw[i,:,1],dtype=np.float32); num+=(u*v).sum(axis=0,dtype=np.float64); den+=np.square(u,dtype=np.float64).sum(axis=0)
    gains={"identity":np.array(1,np.float32),"graph_role_coordinate_gain":np.load(C197/"analysis/operators/role_coordinate_gain.float32.npy"),"atomic_global_gain":safe(np.array(num.sum()),np.array(den.sum())),"atomic_role_gain":safe(num.sum(axis=1),den.sum(axis=1)),"atomic_coordinate_gain":safe(num.sum(axis=0),den.sum(axis=0)),"atomic_role_coordinate_gain":safe(num,den)}
    (OUT/"analysis/operators").mkdir(parents=True,exist_ok=True)
    for name,value in gains.items(): np.save(OUT/f"analysis/operators/{name}.float32.npy",value)
    predictions={}
    for name,gain in gains.items():
        rows=[]
        for program in COMPOSITE:
            selected=[i for i in splits["reveal"] if index[i]["program"]==program]; u=np.asarray(raw[selected,:,0],dtype=np.float32); v=np.asarray(raw[selected,:,1],dtype=np.float32)
            if name in ("identity","atomic_global_gain"): pred=u*gain
            elif name in ("atomic_role_gain",): pred=u*gain[None,None,:,None]
            elif name in ("atomic_coordinate_gain",): pred=u*gain[None,None,None,:]
            else: pred=u*gain[None,None,:,:]
            rows.append((program,metric(pred,v),len(selected)))
        total_u=np.asarray(raw[splits["reveal"],:,0],dtype=np.float32); total_v=np.asarray(raw[splits["reveal"],:,1],dtype=np.float32)
        if name in ("identity","atomic_global_gain"): total_pred=total_u*gain
        elif name=="atomic_role_gain": total_pred=total_u*gain[None,None,:,None]
        elif name=="atomic_coordinate_gain": total_pred=total_u*gain[None,None,None,:]
        else: total_pred=total_u*gain[None,None,:,:]
        predictions[name]={"aggregate":metric(total_pred,total_v),"by_program":{program:value for program,value,_ in rows}}
    ranking=sorted(MODELS,key=lambda m:(predictions[m]["aggregate"]["nrmse"],MODELS.index(m))); winner=ranking[0]; identity=predictions["identity"]["aggregate"]["nrmse"]; improvement=identity-predictions[winner]["aggregate"]["nrmse"]; better=sum(predictions[winner]["by_program"][p]["nrmse"]<predictions["identity"]["by_program"][p]["nrmse"] for p in COMPOSITE); gate=core.load(OUT/"protocol/preregistration.json")["primary_gate"]; passed=improvement>=gate["aggregate_identity_nrmse_improvement_min"] and predictions[winner]["aggregate"]["nrmse"]<=gate["aggregate_nrmse_max"] and better>=gate["programs_better_than_identity_min"]
    report={"phase":PHASE,"campaign":CAMPAIGN,"status":"unseen_composition_prediction_analyzed","predictions":predictions,"ranking":ranking,"winner":winner,"identity_improvement":improvement,"programs_better_than_identity":better,"primary_gate_passed":passed,"semantic_composition_tested":False,"interpretation":"A successful checkpoint-transform holdout would show predictive reuse across program complexity, but not a semantic composition law.","next_authorization":"C200_natural_deletion_and_typed_rescue"}; core.save(OUT/"analysis/composition_prediction.json",report)
    checks={"models":set(predictions)==set(MODELS),"programs":all(set(v["by_program"])==set(COMPOSITE) for v in predictions.values()),"finite":bool(np.isfinite([[predictions[m]["aggregate"]["nrmse"],predictions[m]["aggregate"]["weighted_sign_agreement"]] for m in MODELS]).all())}; core.save(OUT/"audit/internal_build_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())}); print(json.dumps({"winner":winner,"identity_improvement":improvement,"better_programs":better,"passed":passed,"predictions":predictions,"checks":checks},indent=2))


def close():
    protocol=core.load(OUT/"protocol/preregistration.json"); report=core.load(OUT/"analysis/composition_prediction.json"); checks={"contract":core.load(OUT/"audit/internal_contract_audit.json")["all_checks_passed"],"build":core.load(OUT/"audit/internal_build_audit.json")["all_checks_passed"],"hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":PHASE,"campaign":CAMPAIGN,"status":"closed","checks":checks,"all_checks_passed":all(checks.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("contract","build","close")); a=p.parse_args(); {"contract":contract,"build":build,"close":close}[a.command]()
if __name__=="__main__": main()
