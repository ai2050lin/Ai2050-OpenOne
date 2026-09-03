#!/usr/bin/env python3
"""Phase1563: unified BF16 CUDA capture for the C097-A targeted arm."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; CONTRACT=RESULT/"phase1562_c097_targeted_residual_contract"; OUT=RESULT/"phase1563_c097_targeted_residual_capture"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
from phase1537_c091_behavior_only_qualification import make_fixed_batch
from phase1561_c097_common import ROLES,balanced_accuracy
@torch.inference_mode()
def run_batch(model,rows,pad,device,fixed_length):
 ids,mask,pos,lengths=make_fixed_batch(rows,pad,device,fixed_length); out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
 pooled=np.empty((len(rows),len(out.hidden_states),len(ROLES),model.config.hidden_size),np.float32); logits=np.empty((len(rows),2),np.float32)
 for state,hidden in enumerate(out.hidden_states):
  for i,row in enumerate(rows):
   for ri,role in enumerate(ROLES): pooled[i,state,ri]=hidden[i,torch.tensor(row["role_positions"][role],device=device)].float().mean(0).cpu().numpy()
 for i,row in enumerate(rows):
  for j,candidate in enumerate(row["candidate_ids"]): logits[i,j]=float(out.logits[i,lengths[i]-1,candidate[0]].float().cpu())
 return pooled,logits
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1563 exists")
 parent=core.load(CONTRACT/"analysis/final.json"); audit=core.load(CONTRACT/"audit/independent_final_audit.json"); protocol=core.load(CONTRACT/"protocol/preregistration.json")
 if parent["authorization"]!="run_phase1563_c097_targeted_residual_capture" or not audit["all_checks_passed"]: raise RuntimeError("Phase1562 authorization missing")
 rows=core.rows(CONTRACT/"compiled/qwen3_active.jsonl"); pairs=core.rows(CONTRACT/"material/frozen_pairs.jsonl"); lookup={r["case_id"]:i for i,r in enumerate(rows)}; fixed=max(len(r["prompt_ids"]) for r in rows)
 field_path=OUT/"raw/c097a_all_role_field.float32.npy"; field_path.parent.mkdir(parents=True,exist_ok=True); field=np.lib.format.open_memmap(field_path,mode="w+",dtype=np.float32,shape=(56,37,4,2560)); scores={}; repeats=[]; model=None
 try:
  model,tok,device,placement=load_bf16("qwen3"); quant=quantization_audit(model); pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  for bi,pair in enumerate(pairs):
   batch=sorted([r for r in rows if r["pair_id"]==pair["pair_id"]],key=lambda r:protocol["families"].index(r["query_family"])); pooled,values=run_batch(model,batch,pad,device,fixed)
   for i,row in enumerate(batch): field[lookup[row["case_id"]]]=pooled[i]; scores[row["case_id"]]=values[i].tolist()
   if bi<3: repeats.append((batch,pooled,values))
  field.flush(); rh=rl=0.0
  for batch,pooled,values in repeats:
   ah,al=run_batch(model,batch,pad,device,fixed); rh=max(rh,float(np.max(np.abs(pooled-ah)))); rl=max(rl,float(np.max(np.abs(values-al))))
 finally:
  if model is not None: release_bf16(model)
 del field; field=np.load(field_path,mmap_mode="r"); index=[]; behavior=[]
 for i,row in enumerate(rows):
  values=scores[row["case_id"]]; pred=row["candidates"][int(values[1]>values[0])]; item={**{k:row[k] for k in ("case_id","pair_id","pair_family","query_family","partition","partition_rank","truth")},"gold_label":row["gold_label"],"prediction":pred,"correct":pred==row["gold_label"],"candidate_logits":values}; behavior.append(item); index.append({"row_index":i,**{k:row[k] for k in ("case_id","pair_id","pair_family","query_family","partition","partition_rank","surface","concreteness")}})
 causal=0.0
 bypair={p["pair_id"]:[lookup[r["case_id"]] for r in rows if r["pair_id"]==p["pair_id"]] for p in pairs}
 for ids in bypair.values(): causal=max(causal,float(np.max(np.abs(field[ids[0],:,:2].astype(np.float64)-field[ids[1],:,:2].astype(np.float64)))))
 core.write_rows(OUT/"raw/c097a_field_index.jsonl",index); core.write_rows(OUT/"raw/c097a_behavior_logits.jsonl",behavior)
 family_behavior={f:{"balanced_accuracy":balanced_accuracy([r for r in behavior if r["pair_family"]==f]),"accuracy":float(np.mean([r["correct"] for r in behavior if r["pair_family"]==f]))} for f in protocol["families"]}
 checks={"shape":list(field.shape)==[56,37,4,2560],"finite":bool(np.isfinite(field).all()),"repeat_hidden":rh<=1e-6,"repeat_logits":rl<=1e-6,"postquery_causal":causal<=1e-6,"bf16":quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
 if not all(checks.values()): raise RuntimeError(checks)
 report={"phase":1563,"campaign":"C097-A","status":"numeric_gate_pass","field_shape":list(field.shape),"repeat_hidden_max_abs":rh,"repeat_logit_max_abs":rl,"postquery_word_causal_max_abs":causal,"family_behavior":family_behavior,"runtime":{"placement":placement,"quantization":quant},"checks":checks,"files":{"field":{"path":str(field_path.relative_to(ROOT)),"sha256":core.sha(field_path),"bytes":field_path.stat().st_size},"index":{"sha256":core.sha(OUT/"raw/c097a_field_index.jsonl")},"behavior":{"sha256":core.sha(OUT/"raw/c097a_behavior_logits.jsonl")}},"finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"run_phase1564_c097_targeted_residual_adjudication"}
 core.save(OUT/"analysis/capture_summary.json",report); core.save(OUT/"analysis/final.json",{"phase":1563,"campaign":"C097-A","status":report["status"],"authorization":report["authorization"]}); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

