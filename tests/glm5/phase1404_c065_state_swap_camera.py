#!/usr/bin/env python3
"""Phase1404: calibrate C065 whole-state natural donor swap camera."""
from __future__ import annotations
import json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS));import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16
import phase1392_c062_full_field_camera as old
PHASE,CAMPAIGN=1404,"C065";CONTRACT=TESTS/"result/phase1403_c065_active_only_natural_state_contract";MATERIAL=TESTS/"result/phase1400_c064_fixed_answer_factorial_contract";OUT=TESTS/"result/phase1404_c065_state_swap_camera"

def known_truth(n):
 rows=[]
 for seed in range(n):
  g=torch.Generator().manual_seed(650000+seed);x=torch.randn(7,32,generator=g);same=x.clone();same[2]=x[2];same[5]=x[5];role=x.clone();role[3]=x[1];expected=x.clone();expected[3]=x[1]
  rows.append({"seed":seed,"zero_exact":torch.equal(x+torch.zeros_like(x),x),"self_reset_exact":torch.equal(same,x),"role_swap_exact":torch.equal(role,expected),"shape_exact":list(x.shape)==[7,32]})
 return rows

@torch.inference_mode()
def qwen_camera(cases,compiled):
 model=None
 try:
  model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);records=[]
  for case in cases:
   row=compiled[case["recipient"]];rows=[row,row,row,row];ids,mask,pos,offs=old.make_batch(rows,pad,device);handles=[]
   def zero_hook(_m,args):
    original=args[0];value=original.clone();points=old.points(row,offs[2],"record_family");value[2,points]=original[2,points]+torch.zeros_like(original[2,points]);return (value,)+args[1:]
   def reset_hook(_m,args):
    original=args[0];value=original.clone();value[3]=original[0];return (value,)+args[1:]
   handles.append(model.model.layers[3].register_forward_pre_hook(zero_hook));handles.append(model.model.layers[15].register_forward_pre_hook(reset_hook));handles.append(model.model.layers[27].register_forward_pre_hook(reset_hook))
   try:out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
   finally:
    for h in handles:h.remove()
   le=[];se=[]
   for j in (1,2,3):
    le.append(float((out.logits[0,-1].float()-out.logits[j,-1].float()).abs().max()))
    for hs in out.hidden_states:
     left=hs[0,offs[0]:].float();right=hs[j,offs[j]:].float();se.append(float(torch.linalg.vector_norm(left-right)/(torch.linalg.vector_norm(left)+1e-12)))
   records.append({"set_id":case["set_id"],"family":case["family"],"surface":case["surface"],"output_max_abs_diff":max(le),"all_state_relative_l2_max":max(se),"state_count":len(out.hidden_states),"position_count":len(row["prompt_ids"])})
   del out,ids,mask,pos
  return records,{"placement":placement,"quantization":quant}
 finally:
  if model is not None:release_bf16(model)

def main():
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1404 exists")
 f=core.load(CONTRACT/"analysis/final.json");a=core.load(CONTRACT/"audit/independent_final_audit.json");p=core.load(CONTRACT/"protocol/preregistration.json")
 if f["authorization"]!="run_phase1404_c065_state_swap_camera" or not a["all_checks_passed"]:raise RuntimeError("not authorized")
 selected=core.rows(CONTRACT/"material/eligible_factor_sets.jsonl");cases=[r for r in selected if r["partition"]=="response_discovery"]
 if len(cases)!=p["camera"]["qwen_cases"]:raise RuntimeError("camera count")
 compiled={r["case_id"]:r for r in core.rows(MATERIAL/"compiled/qwen3_active.jsonl")};kt=known_truth(p["camera"]["known_truth_systems"]);qr,runtime=qwen_camera(cases,compiled)
 core.write_rows(OUT/"raw/known_truth_systems.jsonl",kt);core.write_rows(OUT/"raw/qwen_identity_camera.jsonl",qr)
 checks={"known_zero":all(r["zero_exact"] for r in kt),"known_self":all(r["self_reset_exact"] for r in kt),"known_role":all(r["role_swap_exact"] for r in kt),"known_shape":all(r["shape_exact"] for r in kt),"qwen_count":len(qr)==18,"states":all(r["state_count"]==37 for r in qr),"logits":max(r["output_max_abs_diff"] for r in qr)<=p["camera"]["logit_identity_max_abs_diff"],"state_identity":max(r["all_state_relative_l2_max"] for r in qr)<=p["camera"]["all_state_identity_relative_l2_max"],"finite":all(math.isfinite(r["all_state_relative_l2_max"]) for r in qr),"bf16":runtime["quantization"]["has_bf16_parameters"],"not_quantized":not runtime["quantization"]["has_quantized_modules"]}
 summary={"phase":PHASE,"campaign":CAMPAIGN,"known_truth_count":len(kt),"qwen_case_count":len(qr),"qwen_output_max_abs_diff":max(r["output_max_abs_diff"] for r in qr),"qwen_all_state_relative_l2_max":max(r["all_state_relative_l2_max"] for r in qr),"checks":checks,"camera_qualified":all(checks.values()),"runtime":runtime,"finished_at_utc":datetime.now(timezone.utc).isoformat()};core.save(OUT/"analysis/camera_summary.json",summary);auth="run_phase1405_c065_natural_discovery_field" if summary["camera_qualified"] else "close_c065_at_camera_gate";core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"camera_qualified":summary["camera_qualified"],"authorization":auth});print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
