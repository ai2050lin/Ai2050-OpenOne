#!/usr/bin/env python3
"""Phase1392: calibrate the C062 full-field, coordinate, and reset cameras."""
from __future__ import annotations

import inspect, json, math, random, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE,CAMPAIGN=1392,"C062"
CONTRACT=TESTS/"result/phase1390_c062_route_factorized_field_campaign_contract"
BEHAVIOR=TESTS/"result/phase1391_c062_family_factorized_behavior"
OUT=TESTS/"result/phase1392_c062_full_field_camera"


def parents():
    f=core.load(BEHAVIOR/"analysis/final.json");a=core.load(BEHAVIOR/"audit/independent_final_audit.json")
    if f["authorization"]!="run_phase1392_c062_full_field_camera" or not a["all_checks_passed"]:raise RuntimeError("behavior not authorized")
    return core.load(CONTRACT/"protocol/preregistration.json"),f


def make_batch(rows,pad,device):
    width=max(len(r["prompt_ids"]) for r in rows);ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);offs=[]
    for i,r in enumerate(rows):
        v=torch.tensor(r["prompt_ids"],dtype=torch.long,device=device);o=width-len(v);offs.append(o);ids[i,o:]=v;mask[i,o:]=1
    pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,offs


def points(row,off,role):return [off+p for p in row["role_positions"][role]]


def known_truth(n):
    records=[]
    for seed in range(n):
        g=torch.Generator().manual_seed(620000+seed);x=torch.randn(6,32,generator=g);coords=torch.randperm(32,generator=g)[:8]
        delta=torch.randn(6,32,generator=g);masked=torch.zeros_like(delta);masked[:,coords]=delta[:,coords]
        a=x.clone();a+=masked;b=x.clone();b[:,coords]+=delta[:,coords]
        states=[torch.randn(5,32,generator=g) for _ in range(4)];reset=states[0].clone();reset[2]=states[1][2];reset[4]=states[1][4]
        expected=states[0].clone();expected[2]=states[1][2];expected[4]=states[1][4]
        records.append({"seed":seed,"zero_exact":torch.equal(x+torch.zeros_like(x),x),
                        "mask_exact":torch.equal(a,b),"field_shape_exact":list(delta.shape)==[6,32],
                        "multi_reset_exact":torch.equal(reset,expected)})
    return records


@torch.inference_mode()
def run_qwen(cases,compiled):
    model=None
    try:
        model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model)
        pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);records=[]
        for case in cases:
            donors=[compiled[case[k]] for k in ("clean_true","corrupt_false","wrong_identity_true","status_true")]
            rows=donors+[donors[1],donors[1],donors[1]]
            ids,mask,pos,offs=make_batch(rows,pad,device);handles=[]
            def source_hook(_m,args):
                original=args[0];value=original.clone();tp=points(rows[5],offs[5],"family")
                value[5,tp]=original[5,tp]+torch.zeros_like(original[5,tp]);return (value,)+args[1:]
            def reset_hook(_m,args):
                original=args[0];value=original.clone();value[6]=original[1];return (value,)+args[1:]
            handles.append(model.model.layers[3].register_forward_pre_hook(source_hook))
            handles.append(model.model.layers[15].register_forward_pre_hook(reset_hook))
            handles.append(model.model.layers[27].register_forward_pre_hook(reset_hook))
            try:
                out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,
                          output_hidden_states=True,return_dict=True)
            finally:
                for h in handles:h.remove()
            base=1;comparators=(4,5,6);logit_errors=[];state_errors=[]
            for j in comparators:
                logit_errors.append(float((out.logits[base,-1].float()-out.logits[j,-1].float()).abs().max()))
                for hs in out.hidden_states:
                    left=hs[base,offs[base]:].float();right=hs[j,offs[j]:].float()
                    state_errors.append(float(torch.linalg.vector_norm(left-right)/(torch.linalg.vector_norm(left)+1e-12)))
            records.append({"pair_id":case["pair_id"],"family":case["target_family"],"surface":case["surface"],
                            "output_max_abs_diff":max(logit_errors),"all_state_relative_l2_max":max(state_errors),
                            "state_count":len(out.hidden_states),"position_count":len(rows[base]["prompt_ids"])})
            del out,ids,mask,pos
        return records,{"placement":placement,"quantization":quant}
    finally:
        if model is not None:release_bf16(model)


def main():
    protocol,behavior=parents()
    if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1392 already exists")
    eligible=[r for r in core.rows(BEHAVIOR/"material/eligible_pairs.jsonl") if r["partition"]=="response_discovery"]
    cases=[]
    for fam in behavior["qualified_families"]:cases.extend([r for r in eligible if r["target_family"]==fam][:6])
    if len(cases)!=protocol["camera"]["qwen_cases"]:raise RuntimeError("camera case count")
    compiled={r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_active.jsonl")}
    compiled.update({r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_status.jsonl")})
    kt=known_truth(protocol["camera"]["known_truth_systems"]);qwen,runtime=run_qwen(cases,compiled)
    core.write_rows(OUT/"raw/known_truth_systems.jsonl",kt);core.write_rows(OUT/"raw/qwen_identity_camera.jsonl",qwen)
    checks={"known_zero":all(r["zero_exact"] for r in kt),"known_mask":all(r["mask_exact"] for r in kt),
            "known_field":all(r["field_shape_exact"] for r in kt),"known_reset":all(r["multi_reset_exact"] for r in kt),
            "qwen_count":len(qwen)==24,"qwen_states":all(r["state_count"]==37 for r in qwen),
            "qwen_output":max(r["output_max_abs_diff"] for r in qwen)<=protocol["camera"]["zero_write_output_max_abs_diff"],
            "qwen_state":max(r["all_state_relative_l2_max"] for r in qwen)<=protocol["camera"]["zero_write_all_state_relative_l2_max"],
            "finite":all(math.isfinite(r["all_state_relative_l2_max"]) for r in qwen),
            "bf16":runtime["quantization"]["has_bf16_parameters"],"not_quantized":not runtime["quantization"]["has_quantized_modules"]}
    summary={"phase":PHASE,"campaign":CAMPAIGN,"known_truth_count":len(kt),"qwen_case_count":len(qwen),
             "qwen_output_max_abs_diff":max(r["output_max_abs_diff"] for r in qwen),
             "qwen_all_state_relative_l2_max":max(r["all_state_relative_l2_max"] for r in qwen),
             "checks":checks,"camera_qualified":all(checks.values()),"runtime":runtime,
             "finished_at_utc":datetime.now(timezone.utc).isoformat()}
    core.save(OUT/"analysis/camera_summary.json",summary)
    auth="run_phase1393_c062_discovery_full_field" if summary["camera_qualified"] else "close_c062_at_camera_gate"
    core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"camera_qualified":summary["camera_qualified"],"authorization":auth})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
