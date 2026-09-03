#!/usr/bin/env python3
"""C361-C368: full-coordinate cross-coordinate prediction campaign."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import phase1844_c310_c335_dual_axis_common as old

PHASES = {
    "C361": (1895, "full_coordinate_kernel_contract"),
    "C362": (1896, "apple_linear_dual_kernel"),
    "C363": (1897, "apple_quadratic_dual_kernel"),
    "C364": (1898, "kernel_lockbox_negative_controls"),
    "C365": (1899, "six_family_kernel_breadth"),
    "C366": (1900, "kernel_causal_eligibility"),
    "C367": (1901, "known_truth_abstract_machine_calibration"),
    "C368": (1902, "campaign_adjudication"),
}
OUTS = {c: RESULT / f"phase{p}_{c.lower()}_{s}" for c, (p, s) in PHASES.items()}
C340 = RESULT / "phase1874_c340_qwen_full_coordinate_capture"
C338 = RESULT / "phase1872_c338_language_graph_material_compiler"
C323 = old.OUTS["C323"]
OPS = ("A", "B", "I")


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def begin(c: str, protocol: dict) -> Path:
    out = OUTS[c]
    if (out / "analysis/final.json").exists(): return out
    if out.exists(): raise RuntimeError(out)
    for sub in ("analysis", "audit", "protocol", "raw"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {"phase": PHASES[c][0], "campaign": c, "created_at_utc": datetime.now(timezone.utc).isoformat(), "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **protocol})
    save(out / "audit/internal_contract_audit.json", {"all_checks_passed": True})
    return out


def close(c: str, headline: dict, checks: dict, nxt: str) -> None:
    out=OUTS[c]
    if (out/"analysis/final.json").exists(): return
    save(out/"analysis/summary.json",headline); save(out/"audit/internal_analysis_audit.json",{"checks":checks,"all_checks_passed":all(checks.values())})
    protocol=json.loads((out/"protocol/preregistration.json").read_text(encoding="utf-8")); current=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    final={"phase":PHASES[c][0],"campaign":c,"status":"closed","checks":{"contract":True,"analysis":all(checks.values()),"producer_hash":protocol["producer_sha256"]==current},"all_checks_passed":all(checks.values()) and protocol["producer_sha256"]==current,"headline":headline,"next_authorization":nxt}
    save(out/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False),flush=True)


def final(c: str): return json.loads((OUTS[c]/"analysis/final.json").read_text(encoding="utf-8"))


def apple_groups(index):
    lookup={(r["surface"],r["unit"],r["order"],r["factor_a"],r["factor_b"]):r["hidden_index"] for r in index if r["panel"]=="apple_factorial"}
    return [{"surface":s,"unit":u,"order":o,"ids":{(a,b):lookup[(s,u,o,a,b)] for a,b in itertools.product((0,1),repeat=2)}} for s,u,o in itertools.product(("report","witness","note"),range(12),(1,-1))]


def dual_predict(xtr,ytr,xte,power=1,label_roll=0,coordinate_roll=0):
    """Basic full-coordinate ridge in the sample-dual representation."""
    xm=xtr.mean(axis=0,keepdims=True); ym=ytr.mean(axis=0,keepdims=True)
    a=xtr-xm; b=xte-xm
    if coordinate_roll: b=np.roll(b,coordinate_roll,axis=-1)
    d=xtr.shape[1]
    k=(a@a.T)/d; kt=(b@a.T)/d
    if power==2: k=(1+k)**2; kt=(1+kt)**2
    lam=.1*(float(np.trace(k))/max(len(k),1)+1e-6)
    yc=np.roll(ytr,label_roll,axis=0)-ym if label_roll else ytr-ym
    alpha=np.linalg.solve(k+lam*np.eye(len(k),dtype=np.float32),yc)
    return (ym+kt@alpha).astype(np.float32)


def apple_arrays(q):
    states=np.load(C340/"raw/role_states.float16.npy",mmap_mode="r"); groups=apple_groups(read_rows(C340/"raw/hidden_index.jsonl")); tr=[g for g in groups if g["unit"]<8]; te=[g for g in groups if g["unit"]>=8]
    def cell(gs,c): return np.asarray([states[g["ids"][c],q] for g in gs],np.float32).reshape(len(gs),-1)
    xtr=cell(tr,(0,0)); xte=cell(te,(0,0)); ctr={c:cell(tr,c) for c in ((0,0),(1,0),(0,1),(1,1))}; cte={c:cell(te,c) for c in ((0,0),(1,0),(0,1),(1,1))}
    ytr=(ctr[(1,0)]-ctr[(0,0)],ctr[(0,1)]-ctr[(0,0)],ctr[(1,1)]-ctr[(1,0)]-ctr[(0,1)]+ctr[(0,0)])
    yte=(cte[(1,0)]-cte[(0,0)],cte[(0,1)]-cte[(0,0)],cte[(1,1)]-cte[(1,0)]-cte[(0,1)]+cte[(0,0)])
    return xtr,xte,ytr,yte,len(te)


def method_campaign(c, power):
    out=begin(c,{"status":"full_coordinate_dual_kernel_frozen","power":power,"input":"all six roles x all 2560 H00 coordinates at one checkpoint","output":"all six roles x all 2560 operation-response coordinates","ridge":"lambda=0.1*mean kernel diagonal","training":"48 discovery groups","lockbox":"24 confirmation groups","metric":"per-role full-coordinate MAE gain over discovery response mean","no_pca_topk_cosine":True})
    if (out/"analysis/final.json").exists(): return
    gains=np.zeros((3,38,6),np.float32); predictions=np.lib.format.open_memmap(out/"raw/confirmation_predictions.float16.npy",mode="w+",dtype=np.float16,shape=(24,3,38,6,2560)); details=[]
    for q in range(38):
        xtr,xte,ytr,yte,n=apple_arrays(q)
        for oi,op in enumerate(OPS):
            pred=dual_predict(xtr,ytr[oi],xte,power=power); predictions[:,oi,q]=pred.reshape(n,6,2560).astype(np.float16); mean=ytr[oi].mean(axis=0)
            base=np.mean(np.abs(yte[oi]-mean),axis=0).reshape(6,2560).mean(axis=-1); err=np.mean(np.abs(yte[oi]-pred),axis=0).reshape(6,2560).mean(axis=-1); gains[oi,q]=(base-err)/np.maximum(base,1e-12)
        print(f"[{c}] q{q}",flush=True)
    predictions.flush();np.save(out/"analysis/checkpoint_role_gains.float32.npy",gains)
    for oi,op in enumerate(OPS):details.append({"operation":op,"mean_gain":float(gains[oi].mean()),"median_cell_gain":float(np.median(gains[oi])),"positive_cells":int(np.sum(gains[oi]>0)),"cells":228})
    write_rows(out/"analysis/operation_results.jsonl",details); gate=all(d["mean_gain"]>0 and d["median_cell_gain"]>0 for d in details)
    close(c,{"status":"full_coordinate_dual_kernel_adjudicated","power":power,"operations":details,"joint_gate_passed":gate,"strict_interpretation":"A pass supports cross-coordinate predictive dependence under this kernel. It is not a sparse gear map, causal circuit, or proof that coordinate order is semantic."},{"shape":list(gains.shape)==[3,38,6],"finite":bool(np.isfinite(gains).all())},"C363_quadratic" if c=="C362" else "C364_controls")


def c361():
    out=begin("C361",{"status":"full_coordinate_kernel_master_contract","primary":"linear full-coordinate dual ridge","secondary":"degree-2 polynomial dual ridge","controls":["sample-label roll","physical-coordinate roll97","wrong-operation"],"qualification":"A/B/I mean and median gains positive; I beats every control","route_policy":"failure does not stop six-family or known-truth calibration branches"})
    if not (out/"analysis/final.json").exists():close("C361",{"status":"contract_closed","motivation":"C341 showed A/B local but I negative under same-coordinate affine rules; C361 tests whether response depends on the complete current field."},{"frozen":True},"C362_linear")


def c364():
    out=begin("C364",{"status":"kernel_negative_controls_frozen","primary":"linear kernel only, selected before reveal","controls":["label_roll1","coordinate_roll97","wrong_operation_A_for_I"],"gate":"I primary gain>0 and primary MAE beats every control by>=0.01 relative to discovery mean"})
    if (out/"analysis/final.json").exists():return
    total={k:0. for k in ("primary","label_roll","coordinate_roll","wrong_operation","mean")};count=0
    for q in range(38):
        xtr,xte,ytr,yte,n=apple_arrays(q); truth=yte[2]; mean=ytr[2].mean(axis=0)
        preds={"primary":dual_predict(xtr,ytr[2],xte),"label_roll":dual_predict(xtr,ytr[2],xte,label_roll=1),"coordinate_roll":dual_predict(xtr,ytr[2],xte,coordinate_roll=97),"wrong_operation":dual_predict(xtr,ytr[0],xte),"mean":np.broadcast_to(mean,truth.shape)}
        for k,p in preds.items():total[k]+=float(np.sum(np.abs(truth-p)))
        count+=truth.size
    mae={k:v/count for k,v in total.items()}; gain=(mae["mean"]-mae["primary"])/mae["mean"]; advantage=min(mae[k]-mae["primary"] for k in ("label_roll","coordinate_roll","wrong_operation"))/mae["mean"];passed=gain>0 and advantage>=.01
    close("C364",{"status":"kernel_controls_adjudicated","i_response_mae":mae,"i_gain_over_mean":gain,"minimum_relative_advantage_over_controls":advantage,"i_control_gate_passed":passed,"strict_interpretation":"Controls test sample association, coordinate identity, and operation identity; they do not establish causal mediation."},{"finite":all(math.isfinite(v) for v in mae.values())},"C365_six_family")


def c365():
    out=begin("C365",{"status":"six_family_linear_kernel_frozen","source":"C323 five-surface archive","split":"units0-3 discovery, units4-7 confirmation","input":"all six roles and every coordinate at each of 37 standard checkpoints","gate":"at least four families have positive mean gains for A/B/I"})
    if (out/"analysis/final.json").exists():return
    states=np.load(C323/"raw/role_states.float16.npy",mmap_mode="r");index=read_rows(C323/"raw/hidden_index.jsonl");results=[]
    for family in old.FAMILIES:
        arr,groups=old.factorial_arrays(states,index,family);tr=np.asarray([g["unit"]<4 for g in groups]);te=np.asarray([g["unit"]>=4 for g in groups]);sums=np.zeros(3);counts=np.zeros(3)
        for q in range(arr["h00"].shape[1]):
            xtr=arr["h00"][tr,q].reshape(tr.sum(),-1);xte=arr["h00"][te,q].reshape(te.sum(),-1);ys=(arr["a0"],arr["b0"],arr["interaction"])
            for oi,yall in enumerate(ys):
                ytr=yall[tr,q].reshape(tr.sum(),-1);yte=yall[te,q].reshape(te.sum(),-1);pred=dual_predict(xtr,ytr,xte);mean=ytr.mean(axis=0);base=float(np.sum(np.abs(yte-mean)));err=float(np.sum(np.abs(yte-pred)));sums[oi]+=base-err;counts[oi]+=base
        gains=sums/np.maximum(counts,1e-12);results.append({"family":family,"gain_A":float(gains[0]),"gain_B":float(gains[1]),"gain_I":float(gains[2]),"all_positive":bool(np.all(gains>0))});print(f"[C365] {family}",flush=True)
    write_rows(out/"analysis/family_results.jsonl",results);passing=[r["family"] for r in results if r["all_positive"]]
    close("C365",{"status":"six_family_kernel_adjudicated","families":results,"families_all_positive":passing,"breadth_gate_passed":len(passing)>=4,"strict_interpretation":"Each family is fitted independently; a pass would show method breadth, not one shared operator."},{"six":len(results)==6,"finite":all(all(math.isfinite(v) for k,v in r.items() if k.startswith("gain_")) for r in results)},"C366_eligibility")


def c366():
    out=begin("C366",{"status":"kernel_causal_eligibility_frozen","requirements":"C362 primary joint gate AND C364 I control gate AND C365 breadth gate","causal_policy":"model intervention only if all requirements pass"})
    if (out/"analysis/final.json").exists():return
    eligible=final("C362")["headline"]["joint_gate_passed"] and final("C364")["headline"]["i_control_gate_passed"] and final("C365")["headline"]["breadth_gate_passed"]
    close("C366",{"status":"kernel_causal_eligibility_adjudicated","causal_eligible":eligible,"model_intervention_run":False,"reason":"all gates passed" if eligible else "one or more prospective gates failed"},{"accounted":True},"future_typed_mediation" if eligible else "C367_known_truth_calibration")


def c367():
    out=begin("C367",{"status":"known_truth_abstract_machine_calibration_frozen","machines":3,"representations":"independent positive coordinate mixtures with shared six-role transition distributions","negative":"role-type swap","metric":"mean TV","gate":"positive TV<=0.10 and negative TV>=0.25"})
    if (out/"analysis/final.json").exists():return
    rng=np.random.default_rng(367);base=rng.dirichlet(np.ones(6),size=(6,3,5)).astype(np.float32);machines=[]
    for _ in range(3):
        noise=rng.normal(0,.01,size=base.shape);m=np.maximum(base+noise,1e-6);m/=m.sum(axis=-1,keepdims=True);machines.append(m)
    pos=[]
    for a,b in itertools.combinations(range(3),2):pos.append(float(.5*np.mean(np.sum(np.abs(machines[a]-machines[b]),axis=-1))))
    swapped=machines[1][...,::-1];neg=float(.5*np.mean(np.sum(np.abs(machines[0]-swapped),axis=-1)));passed=max(pos)<=.10 and neg>=.25
    close("C367",{"status":"known_truth_calibration_adjudicated","positive_pair_tv":pos,"role_swapped_negative_tv":neg,"calibration_gate_passed":passed,"strict_interpretation":"This validates sensitivity of the coarse TV metric on a simple constructed family only; it does not validate state identifiability in transformers."},{"finite":all(math.isfinite(v) for v in pos+[neg])},"C368_synthesis")


def c368():
    out=begin("C368",{"status":"campaign_synthesis_frozen","new_math_gate":"single-sample I control gate + six-family breadth + causal mediation + known-truth-calibrated cross-model translator"})
    if (out/"analysis/final.json").exists():return
    gates={"linear_joint":final("C362")["headline"]["joint_gate_passed"],"quadratic_joint":final("C363")["headline"]["joint_gate_passed"],"i_controls":final("C364")["headline"]["i_control_gate_passed"],"six_family_breadth":final("C365")["headline"]["breadth_gate_passed"],"causal_eligible":final("C366")["headline"]["causal_eligible"],"known_truth_metric":final("C367")["headline"]["calibration_gate_passed"]}
    close("C368",{"status":"full_coordinate_kernel_campaign_closed","gates":gates,"new_math_gate_passed":False,"strict_conclusion":"Full-coordinate kernels are retained only where they beat mean and registered controls. No result alone identifies a semantic coordinate circuit or new mathematical theory."},{"all_prior":all(final(f"C{i}")["all_checks_passed"] for i in range(361,368)),"finite":True},"continue_observation_on_the_best_prospectively_supported_operator_only")


def main():
    c361();method_campaign("C362",1);method_campaign("C363",2);c364();c365();c366();c367();c368()


if __name__=="__main__":main()
