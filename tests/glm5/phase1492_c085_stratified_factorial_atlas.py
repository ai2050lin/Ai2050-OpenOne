#!/usr/bin/env python3
"""Phase1492: build success/mixed/failed factorial atlases from the frozen C085 field."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; C=RESULT/"phase1489_c085_prospective_layered_contract"; B=RESULT/"phase1490_c085_behavior_stratification"; CAP=RESULT/"phase1491_c085_all_case_field_capture"; OUT=RESULT/"phase1492_c085_stratified_factorial_atlas"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
EFFECTS=["relation","entity","object","relation_entity","relation_object","entity_object","relation_entity_object"]
ORDERS={"relation":1,"entity":1,"object":1,"relation_entity":2,"relation_object":2,"entity_object":2,"relation_entity_object":3}
STRATA=["success","mixed","failed"]
def signs(row):
    r=1 if row["relation_match"] else -1; e=1 if row["entity_match"] else -1; o=1 if row["object_match"] else -1
    return {"relation":r,"entity":e,"object":o,"relation_entity":r*e,"relation_object":r*o,"entity_object":e*o,"relation_entity_object":r*e*o}
def cosine(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b)); return float(np.dot(a.astype(np.float64,copy=False),b.astype(np.float64,copy=False))/d) if d>1e-12 else 0.0
def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1492 exists")
    pf=core.load(CAP/"analysis/final.json"); pa=core.load(CAP/"audit/independent_final_audit.json"); protocol=core.load(C/"protocol/preregistration.json"); meta=core.load(CAP/"analysis/capture_metadata.json")
    if pf["authorization"]!="run_phase1492_c085_stratified_factorial_atlas" or not pa["all_checks_passed"] or core.sha(CAP/"raw/all_role_field.float16.npy")!=meta["raw_sha256"]: raise RuntimeError("Phase1491 authorization/integrity missing")
    field=np.load(CAP/"raw/all_role_field.float16.npy",mmap_mode="r"); index=core.rows(CAP/"raw/all_role_field_index.jsonl"); lookup={(r["family"],r["index"],r["record_relation_id"],r["surface"],r["cell"]):r for r in index}; groups=core.rows(B/"material/stratified_composition_sets.jsonl"); relations=protocol["relations"]; splits=protocol["partitions"]; surfaces=protocol["surfaces"]; cells=protocol["cells"]
    OUT.joinpath("atlas").mkdir(parents=True,exist_ok=True); success_path=OUT/"atlas/success_factorial_contrast_mean.float32.npy"; stratum_path=OUT/"atlas/stratum_relation_effect.float32.npy"
    success=np.lib.format.open_memmap(success_path,mode="w+",dtype=np.float32,shape=(7,6,3,2,37,9,2560)); success[:]=0
    stratum=np.lib.format.open_memmap(stratum_path,mode="w+",dtype=np.float32,shape=(3,6,3,2,37,9,2560)); stratum[:]=0; counts=np.zeros((3,6,3,2),dtype=np.int32); finite=True
    by_panel={(st,rel,sp):[g for g in groups if g["stratum"]==st and g["record_relation_id"]==rel and g["partition"]==sp] for st in STRATA for rel in relations for sp in splits}
    for si,st in enumerate(STRATA):
        for ri,rel in enumerate(relations):
            for pi,sp in enumerate(splits):
                selected=by_panel[(st,rel,sp)]
                for ui,surface in enumerate(surfaces):
                    counts[si,ri,pi,ui]=len(selected)
                    if not selected: continue
                    total=np.zeros((7 if st=="success" else 1,37,9,2560),dtype=np.float64)
                    for g in selected:
                        rows=[lookup[(g["family"],g["index"],rel,surface,cell)] for cell in cells]; block=np.asarray(field[[r["row_index"] for r in rows]],dtype=np.float32)
                        effects=EFFECTS if st=="success" else ["relation"]
                        weight=np.asarray([[signs(r)[eff]*(2**ORDERS[eff])/8.0 for r in rows] for eff in effects],dtype=np.float32); contrast=np.tensordot(weight,block,axes=(1,0)); finite=finite and bool(np.isfinite(contrast).all()); total+=contrast
                    mean=(total/len(selected)).astype(np.float32); stratum[si,ri,pi,ui]=mean[0]
                    if st=="success": success[:,ri,pi,ui]=mean
    success.flush(); stratum.flush(); del success,stratum; np.save(OUT/"atlas/stratum_sample_counts.int32.npy",counts)
    sf=np.load(success_path,mmap_mode="r"); sr=np.load(stratum_path,mmap_mode="r"); boundary=protocol["roles"].index("boundary"); summary_rows=[]
    for si,st in enumerate(STRATA):
        for ri,rel in enumerate(relations):
            valid=[(pi,ui) for pi in range(3) for ui in range(2) if counts[si,ri,pi,ui]>0]
            vectors=[np.asarray(sr[si,ri,pi,ui,35,boundary],dtype=np.float32) for pi,ui in valid]
            summary_rows.append({"stratum":st,"relation":rel,"panel_count":len(valid),"set_count":int(sum(counts[si,ri,pi,0] for pi in range(3))),"state35_boundary_norm_mean":float(np.mean([np.linalg.norm(v) for v in vectors])) if vectors else None,"panel_pairwise_cosine_mean":float(np.mean([cosine(vectors[i],vectors[j]) for i in range(len(vectors)) for j in range(i+1,len(vectors))])) if len(vectors)>1 else None})
    core.write_rows(OUT/"analysis/stratum_relation_summary.jsonl",summary_rows)
    valid=counts>0; checks={"finite":finite and bool(np.isfinite(sf).all()) and bool(np.isfinite(sr[valid]).all()),"success_shape":list(sf.shape)==[7,6,3,2,37,9,2560],"stratum_shape":list(sr.shape)==[3,6,3,2,37,9,2560],"count_consistency":all(counts[si,ri,pi,0]==counts[si,ri,pi,1] for si in range(3) for ri in range(6) for pi in range(3)),"success_panels":bool(np.all(counts[0]>=protocol["p084"]["minimum_sets_per_relation_split_surface"])),"mixed_present":bool(np.any(counts[1]>0)),"failed_registered_missing":bool(np.all(counts[2]==0)),"success_relation_identity":float(np.max(np.abs(np.asarray(sf[0])-np.asarray(sr[0]))))==0.0}
    if not all(checks.values()): raise RuntimeError(checks)
    summary={"phase":1492,"campaign":"C085","axis_orders":{"success_factorial":["effect","relation","split","surface","state","role","coordinate"],"stratum_relation":["stratum","relation","split","surface","state","role","coordinate"]},"effects":EFFECTS,"strata":STRATA,"counts":counts.tolist(),"checks":checks,"missingness":{"failed":"M2: no 0/16 composition set was observed; no failed field is imputed"},"files":{"success_factorial":{"bytes":success_path.stat().st_size,"sha256":core.sha(success_path)},"stratum_relation":{"bytes":stratum_path.stat().st_size,"sha256":core.sha(stratum_path)},"counts":{"sha256":core.sha(OUT/"atlas/stratum_sample_counts.int32.npy")}},"interpretation_boundary":"success atlas can test P084; mixed and failed strata are diagnostic only","finished_at_utc":datetime.now(timezone.utc).isoformat()}
    core.save(OUT/"analysis/stratified_atlas_summary.json",summary); core.save(OUT/"analysis/final.json",{"phase":1492,"campaign":"C085","status":"stratified_factorial_atlas_complete","authorization":"run_phase1493_c085_prospective_p084_adjudication"}); print(json.dumps({k:v for k,v in summary.items() if k!="counts"},indent=2))
if __name__=="__main__": main()
