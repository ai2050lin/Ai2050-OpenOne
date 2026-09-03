#!/usr/bin/env python3
"""Phase1566: one-load BF16 CUDA capture for independent WordNet C097-B."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; CONTRACT=RESULT/"phase1565_c097_wordnet_independent_contract"; OUT=RESULT/"phase1566_c097_wordnet_capture"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
from phase1561_c097_common import balanced_accuracy
from phase1563_c097_targeted_residual_capture import run_batch
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1566 exists")
 pf=core.load(CONTRACT/"analysis/final.json"); pa=core.load(CONTRACT/"audit/independent_final_audit.json"); protocol=core.load(CONTRACT/"protocol/preregistration.json")
 if pf["authorization"]!="run_phase1566_c097_wordnet_capture" or not pa["all_checks_passed"]: raise RuntimeError("Phase1565 authorization missing")
 rows=core.rows(CONTRACT/"compiled/qwen3_active.jsonl"); pairs=core.rows(CONTRACT/"material/frozen_wordnet_pairs.jsonl"); lookup={r["case_id"]:i for i,r in enumerate(rows)}; fixed=max(len(r["prompt_ids"]) for r in rows)
 field_path=OUT/"raw/c097b_all_role_field.float32.npy"; field_path.parent.mkdir(parents=True,exist_ok=True); field=np.lib.format.open_memmap(field_path,mode="w+",dtype=np.float32,shape=(540,37,4,2560)); scores={}; repeats=[]; model=None
 try:
  model,tok,device,placement=load_bf16("qwen3"); quant=quantization_audit(model); pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  for bi,pair in enumerate(pairs):
   batch=[r for r in rows if r["pair_id"]==pair["pair_id"]]; batch.sort(key=lambda r:(protocol["surfaces"].index(r["surface"]),protocol["families"].index(r["query_family"]))); pooled,values=run_batch(model,batch,pad,device,fixed)
   for i,row in enumerate(batch): field[lookup[row["case_id"]]]=pooled[i]; scores[row["case_id"]]=values[i].tolist()
   if bi<3: repeats.append((batch,pooled,values))
   if (bi+1)%15==0: print(f"[phase1566] {bi+1}/{len(pairs)} pair batches",flush=True)
  field.flush(); rh=rl=0.0
  for batch,pooled,values in repeats:
   ah,al=run_batch(model,batch,pad,device,fixed); rh=max(rh,float(np.max(np.abs(pooled-ah)))); rl=max(rl,float(np.max(np.abs(values-al))))
 finally:
  if model is not None: release_bf16(model)
 del field; field=np.load(field_path,mmap_mode="r"); index=[]; behavior=[]
 for i,row in enumerate(rows):
  values=scores[row["case_id"]]; prediction=row["candidates"][int(values[1]>values[0])]; item={**{k:row[k] for k in ("case_id","pair_id","pair_family","query_family","partition","surface","truth")},"gold_label":row["gold_label"],"prediction":prediction,"correct":prediction==row["gold_label"],"candidate_logits":values}; behavior.append(item); index.append({"row_index":i,**{k:row[k] for k in ("case_id","pair_id","pair_family","query_family","partition","surface")}})
 causal=0.0
 for pair in pairs:
  ids=[lookup[r["case_id"]] for r in rows if r["pair_id"]==pair["pair_id"] and r["surface"]=="postquery"]
  for other in ids[1:]: causal=max(causal,float(np.max(np.abs(field[ids[0],:,:2].astype(np.float64)-field[other,:,:2].astype(np.float64)))))
 core.write_rows(OUT/"raw/c097b_field_index.jsonl",index); core.write_rows(OUT/"raw/c097b_behavior_logits.jsonl",behavior)
 family_behavior={}
 for family in protocol["families"]:
  subset=[r for r in behavior if r["query_family"]==family]; family_behavior[family]={"balanced_accuracy":balanced_accuracy(subset),"accuracy":float(np.mean([r["correct"] for r in subset])),"true_recall":float(np.mean([r["correct"] for r in subset if r["truth"]])),"false_recall":float(np.mean([r["correct"] for r in subset if not r["truth"]]))}
 three=[]
 for pair in pairs:
  for surface in protocol["surfaces"]:
   subset=[r for r in rows if r["pair_id"]==pair["pair_id"] and r["surface"]==surface]; margins={r["query_family"]:scores[r["case_id"]][r["candidates"].index("yes")]-scores[r["case_id"]][r["candidates"].index("no")] for r in subset}; pred=max(margins,key=margins.get); three.append({"pair_id":pair["pair_id"],"surface":surface,"gold":pair["family"],"prediction":pred,"correct":pred==pair["family"],"margins":margins})
 core.write_rows(OUT/"analysis/c097b_three_way_selection.jsonl",three)
 checks={"shape":list(field.shape)==[540,37,4,2560],"finite":bool(np.isfinite(field[::11]).all()),"repeat_hidden":rh<=1e-6,"repeat_logits":rl<=1e-6,"postquery_causal":causal<=1e-6,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
 if not all(checks.values()): raise RuntimeError(checks)
 report={"phase":1566,"campaign":"C097-B","status":"independent_wordnet_numeric_gate_pass","field_shape":list(field.shape),"repeat_hidden_max_abs":rh,"repeat_logit_max_abs":rl,"postquery_word_causal_max_abs":causal,"family_behavior":family_behavior,"three_way_accuracy":float(np.mean([r["correct"] for r in three])),"runtime":{"placement":placement,"quantization":quant},"checks":checks,"files":{"field":{"path":str(field_path.relative_to(ROOT)),"sha256":core.sha(field_path),"bytes":field_path.stat().st_size},"index":{"sha256":core.sha(OUT/"raw/c097b_field_index.jsonl")},"behavior":{"sha256":core.sha(OUT/"raw/c097b_behavior_logits.jsonl")}},"finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1567_c097_identifiable_common_residual_atlas"}; core.save(OUT/"analysis/capture_summary.json",report); core.save(OUT/"analysis/final.json",{"phase":1566,"campaign":"C097-B","status":report["status"],"authorization":report["authorization"]}); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

