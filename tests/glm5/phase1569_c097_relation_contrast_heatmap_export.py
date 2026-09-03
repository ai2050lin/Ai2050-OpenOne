#!/usr/bin/env python3
"""Phase1569: export the audited C097 relation-contrast heatmap asset."""
from __future__ import annotations
import json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; PARENT=RESULT/"phase1568_c097_major_stage_closure"; SOURCE=RESULT/"phase1567_c097_identifiable_common_residual_atlas"; OUT=RESULT/"phase1569_c097_relation_contrast_heatmap_export"; CLIENT=ROOT/"frontend/public/vis_data/research_kernel/c097_relation_contrast_heatmap.json"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
from phase1561_c097_common import top_indices
PARTITIONS=("response_discovery","confirmation","lockbox"); SURFACES=("prequery","postquery"); FAMILY_PAIRS=("similarity-class","similarity-whole","class-whole")
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1569 exists")
 pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json"); closure=core.load(PARENT/"analysis/c097_major_stage_closure.json")
 if pf["authorization"]!="run_phase1569_c097_relation_contrast_heatmap_export" or not pa["all_checks_passed"] or not closure["visualization"]["important"]: raise RuntimeError("visualization not authorized")
 common=np.load(SOURCE/"raw/c097b_common_contrast_field.float32.npy",mmap_mode="r"); residual=np.load(SOURCE/"raw/c097b_residual_contrast_field.float32.npy",mmap_mode="r"); atlas=core.rows(SOURCE/"analysis/common_residual_atlas.jsonl")
 reference=np.mean(np.abs(np.asarray(common[:,:,31:33,3],np.float64)),axis=(0,1,2)); dimensions=top_indices(reference,64).tolist(); scale=float(np.max(np.abs(common[:,:,:,3,dimensions])))
 rows=[]
 for pi,partition in enumerate(PARTITIONS):
  for si,surface in enumerate(SURFACES):
   for state in range(37): rows.append({"partition":partition,"surface":surface,"state":state,"values":[float(v) for v in common[pi,si,state,3,dimensions]],"normalized":[float(v/scale) for v in common[pi,si,state,3,dimensions]]})
 residual_rows=[]
 for pi,partition in enumerate(PARTITIONS):
  for si,surface in enumerate(SURFACES):
   for fi,name in enumerate(FAMILY_PAIRS): residual_rows.append({"partition":partition,"surface":surface,"family_pair":name,"state_norms":[float(np.linalg.norm(residual[pi,si,fi,state,3])) for state in range(37)]})
 asset={"schema":"relation_contrast_heatmap.v1","result_type":"relation_contrast_heatmap","phase":1569,"source_phase":1567,"campaign":"C097","model":"Qwen3-4B","title":"C097 Identifiable Relation Contrast Field","dimensions":dimensions,"scale":scale,"common_rows":rows,"residual_rows":residual_rows,"component_rows":[r for r in atlas if r["role"]=="boundary"],"evidence":{"grade":"E3-OBS-cross-source-diagnostic","behavior_scope":{"similarity":"qualified","class_inclusion":"M_BEHAVIOR","whole_part":"M_BEHAVIOR"},"boundary":"Contrast mean is not purified semantics or causal mechanism."},"created_at_utc":datetime.now(timezone.utc).isoformat()}
 output=OUT/"visualization/c097_relation_contrast_heatmap.json"; core.save(output,asset); CLIENT.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(output,CLIENT)
 checks={"schema":asset["schema"]=="relation_contrast_heatmap.v1","dimensions":len(dimensions)==64 and len(set(dimensions))==64,"rows":len(rows)==222,"residual_rows":len(residual_rows)==18,"finite":bool(np.isfinite(np.asarray([r["normalized"] for r in rows])).all()),"client_copy":core.sha(output)==core.sha(CLIENT)}
 if not all(checks.values()): raise RuntimeError(checks)
 report={"phase":1569,"campaign":"C097","status":"audited_relation_contrast_heatmap_exported","checks":checks,"asset":{"path":str(output.relative_to(ROOT)),"sha256":core.sha(output),"client_path":str(CLIENT.relative_to(ROOT)),"bytes":output.stat().st_size},"finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"freeze_C098_observation_first_graph_contract"}; core.save(OUT/"analysis/visualization_export.json",report); core.save(OUT/"analysis/final.json",{"phase":1569,"campaign":"C097","status":report["status"],"authorization":report["authorization"]}); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

