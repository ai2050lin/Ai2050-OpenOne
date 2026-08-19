#!/usr/bin/env python3
"""Phase1393: discovery-only all-layer/all-position C062 response field."""
from __future__ import annotations

import json, math, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16

PHASE,CAMPAIGN=1393,"C062"
CONTRACT=TESTS/"result/phase1390_c062_route_factorized_field_campaign_contract"
BEHAVIOR=TESTS/"result/phase1391_c062_family_factorized_behavior"
CAMERA=TESTS/"result/phase1392_c062_full_field_camera"
OUT=TESTS/"result/phase1393_c062_discovery_full_field"


def parents():
 f=core.load(CAMERA/"analysis/final.json");a=core.load(CAMERA/"audit/independent_final_audit.json")
 if f["authorization"]!="run_phase1393_c062_discovery_full_field" or not a["all_checks_passed"]:raise RuntimeError("camera not authorized")
 return core.load(CONTRACT/"protocol/preregistration.json"),core.load(BEHAVIOR/"analysis/final.json")


def make_batch(rows,pad,device):
 width=max(len(r["prompt_ids"]) for r in rows);ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);offs=[]
 for i,r in enumerate(rows):
  v=torch.tensor(r["prompt_ids"],dtype=torch.long,device=device);o=width-len(v);offs.append(o);ids[i,o:]=v;mask[i,o:]=1
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,offs


def points(row,off,role):return [off+p for p in row["role_positions"][role]]
def scaled(v,n):
 norm=torch.linalg.vector_norm(v)
 if float(norm)<=1e-12:raise RuntimeError("zero control")
 return v*(n/norm)
def margin(out,i,row):
 z=out.logits[i,-1].float();return float(z[row["candidate_ids"][0][0]]-z[row["candidate_ids"][1][0]])
def order(score):return sorted(range(score.numel()),key=lambda i:(-float(score[i]),i))
def cosine(a,b):
 den=float(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))
 return float(torch.dot(a.flatten(),b.flatten())/den) if den>1e-12 else 0.0


@torch.inference_mode()
def main():
 protocol,behavior=parents()
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1393 already exists")
 cases=[r for r in core.rows(BEHAVIOR/"material/eligible_pairs.jsonl") if r["partition"]=="response_discovery"]
 if len(cases)!=18*len(behavior["qualified_families"]):raise RuntimeError("discovery count")
 compiled={r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_active.jsonl")}
 compiled.update({r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_status.jsonl")})
 layouts=core.load(CONTRACT/"protocol/surface_layouts.json");obs=protocol["observation"]
 model=None
 try:
  model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
  source_vectors=[];source_meta=[];case_records=[];aggregate=defaultdict(lambda:{"selectivity":[],"strength":[],"wins":[],"correct_norm":[],"alignment":[]})
  raw_path=OUT/"raw/full_field_events.jsonl";raw_path.parent.mkdir(parents=True,exist_ok=True)
  with raw_path.open("w",encoding="utf-8",newline="\n") as raw:
   for ci,case in enumerate(cases):
    donors=[compiled[case[k]] for k in ("clean_true","corrupt_false","wrong_identity_true","status_true")]
    rows=donors+[donors[1],donors[1],donors[1],donors[1]];ids,mask,pos,offs=make_batch(rows,pad,device);holder={"norm_error":0.0}
    def hook(_m,args):
     original=args[0];value=original.clone();role=protocol["source"]["role"]
     dv=[original[i,points(rows[i],offs[i],role)].float() for i in range(4)];correct=dv[0]-dv[1];n=torch.linalg.vector_norm(correct)
     directions=[torch.zeros_like(correct),correct,scaled(dv[2]-dv[1],n),scaled(dv[3]-dv[1],n)]
     holder["source"]=correct.detach().cpu();holder["norm_error"]=max(abs(float(torch.linalg.vector_norm(d)/(n+1e-12))-1.0) for d in directions[1:])
     for local,d in enumerate(directions):
      ti=4+local;tp=points(rows[ti],offs[ti],role);value[ti,tp]=original[ti,tp]+d.to(original.dtype)
     return (value,)+args[1:]
    h=model.model.layers[protocol["source"]["layer"]].register_forward_pre_hook(hook)
    try:out=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
    finally:h.remove()
    source_vectors.append(holder["source"].flatten());source_meta.append({k:case[k] for k in ("pair_id","target_family","surface","partition")})
    margins={"clean":margin(out,0,donors[0]),"corrupt":margin(out,1,donors[1]),"wrong_identity":margin(out,2,donors[2]),
             "status":margin(out,3,donors[3]),"self":margin(out,4,rows[4]),"correct":margin(out,5,rows[5]),
             "wrong":margin(out,6,rows[6]),"status_direction":margin(out,7,rows[7])}
    effects={a:margins[a]-margins["self"] for a in ("correct","wrong","status_direction")}
    case_records.append({"pair_id":case["pair_id"],"family":case["target_family"],"surface":case["surface"],
                         "margins":margins,"effects":effects,"whole_effect":margins["clean"]-margins["corrupt"],
                         "source_norm_error":holder["norm_error"]})
    base_off=offs[4];length=len(rows[4]["prompt_ids"])
    role_by_pos={p:role for role,ps in rows[4]["role_positions"].items() for p in ps}
    for s,hs in enumerate(out.hidden_states):
     clean=hs[0,offs[0]:].float();corrupt=hs[1,offs[1]:].float();selfv=hs[4,base_off:].float()
     cv=hs[5,base_off:].float();wv=hs[6,base_off:].float();sv=hs[7,base_off:].float()
     if any(v.shape[0]!=length for v in (clean,corrupt,selfv,cv,wv,sv)):raise RuntimeError("layout mismatch")
     for p in range(length):
      dc=cv[p]-selfv[p];dw=wv[p]-selfv[p];ds=sv[p]-selfv[p];natural=clean[p]-corrupt[p]
      cn=float(torch.linalg.vector_norm(dc));ctrl=max(float(torch.linalg.vector_norm(dw)),float(torch.linalg.vector_norm(ds)))
      sel=(cn-ctrl)/(cn+ctrl+1e-12);strength=cn/(float(torch.linalg.vector_norm(selfv[p]))+1e-12);align=cosine(dc,natural)
      rec={"pair_id":case["pair_id"],"family":case["target_family"],"surface":case["surface"],"state_index":s,
           "position":p,"role":role_by_pos.get(p,"other"),"correct_norm":cn,"control_norm":ctrl,"selectivity":sel,
           "strength_ratio":strength,"correct_wins":cn>ctrl,"natural_alignment":align}
      raw.write(core.canonical(rec)+"\n");a=aggregate[(case["surface"],s,p)];a["selectivity"].append(sel);a["strength"].append(strength);a["wins"].append(cn>ctrl);a["correct_norm"].append(cn);a["alignment"].append(align)
    del out,ids,mask,pos
    if (ci+1)%12==0:print(json.dumps({"full_field_cases":ci+1,"total":len(cases)}),flush=True)
  vectors=torch.stack(source_vectors);payload={"vectors":vectors.to(torch.float32),"metadata":source_meta}
  pt=OUT/"raw/discovery_family3_differences.pt";torch.save(payload,pt)
  rankings={"source_sha256":core.sha(pt),"global":order(vectors.abs().mean(0)),"families":{},"selection_scope":"C062 response_discovery only"}
  for fam in behavior["qualified_families"]:
   idx=[i for i,m in enumerate(source_meta) if m["target_family"]==fam];rankings["families"][fam]=order(vectors[idx].abs().mean(0))
  core.save(OUT/"protocol/discovery_rankings.json",rankings)
  aggregate_rows=[];candidates=defaultdict(lambda:defaultdict(list))
  for (surface,s,p),a in sorted(aggregate.items()):
   item={"surface":surface,"state_index":s,"position":p,"role":next((r for r,ps in layouts[surface]["role_positions"].items() if p in ps),"other"),
         "count":len(a["selectivity"]),"selectivity_median":statistics.median(a["selectivity"]),
         "strength_ratio_median":statistics.median(a["strength"]),"win_fraction":sum(a["wins"])/len(a["wins"]),
         "correct_norm_median":statistics.median(a["correct_norm"]),"natural_alignment_median":statistics.median(a["alignment"])}
   item["candidate_qualified"]=(item["strength_ratio_median"]>=obs["event_strength_ratio_min"] and
      item["selectivity_median"]>=obs["event_selectivity_median_min"] and item["win_fraction"]>=obs["event_selectivity_win_min"])
   aggregate_rows.append(item)
   if item["candidate_qualified"]:
    for wi,(lo,hi) in enumerate(obs["stage_windows"]):
     if lo<=s<=hi:candidates[surface][str(wi)].append(item)
  core.write_rows(OUT/"analysis/full_field_aggregate.jsonl",aggregate_rows)
  selected={};bundles={}
  for surface in protocol["material"]["surfaces"]:
   stage={};flat=[]
   for wi in range(len(obs["stage_windows"])):
    ordered=sorted(candidates[surface][str(wi)],key=lambda r:(-r["selectivity_median"],-r["strength_ratio_median"],r["state_index"],r["position"]))
    stage[str(wi)]=[{k:r[k] for k in ("surface","state_index","position","role","selectivity_median","strength_ratio_median","win_fraction")}
                    for r in ordered[:obs["top_events_per_stage_surface"]]];flat.extend(stage[str(wi)])
   selected[surface]=stage;flat_order=sorted(flat,key=lambda r:(-r["selectivity_median"],-r["strength_ratio_median"],r["state_index"],r["position"]))
   bundles[surface]={"top1":flat_order[:1],"stage_top1":[v[0] for v in stage.values() if v],
                     "stage_top2":[x for v in stage.values() for x in v],
                     "query_reference":[{"surface":surface,"state_index":15,"role":"query","position":layouts[surface]["role_positions"]["query"][0]}],
                     "boundary_reference":[{"surface":surface,"state_index":27,"role":"boundary","position":layouts[surface]["role_positions"]["boundary"][0]}]}
  core.save(OUT/"protocol/discovery_event_candidates.json",{"selected":selected,"bundles":bundles,"selection_scope":"response_discovery only"})
  core.write_rows(OUT/"raw/case_response.jsonl",case_records)
  correct=[r["effects"]["correct"] for r in case_records];adv=[r["effects"]["correct"]-max(r["effects"]["wrong"],r["effects"]["status_direction"]) for r in case_records]
  summary={"phase":PHASE,"campaign":CAMPAIGN,"case_count":len(cases),"event_record_count":sum(len(a["selectivity"]) for a in aggregate.values()),
           "aggregate_event_count":len(aggregate_rows),"qualified_candidate_count":sum(r["candidate_qualified"] for r in aggregate_rows),
           "selected_event_count":sum(len(v) for s in selected.values() for v in s.values()),
           "endpoint":{"correct_gain_median":statistics.median(correct),"advantage_median":statistics.median(adv),
                       "win_fraction":sum(v>0 for v in adv)/len(adv)},
           "rankings_sha256":core.sha(OUT/"protocol/discovery_rankings.json"),"candidates_sha256":core.sha(OUT/"protocol/discovery_event_candidates.json"),
           "raw_field_sha256":core.sha(raw_path),"source_sha256":core.sha(pt),
           "checks":{"discovery_only":all(r["partition"]=="response_discovery" for r in cases),"case_count":len(cases)==72,
                     "all_states":all(r["state_index"] in range(37) for r in aggregate_rows),"all_positions":len(aggregate_rows)>0,
                     "finite":all(math.isfinite(r[k]) for r in aggregate_rows for k in ("selectivity_median","strength_ratio_median","natural_alignment_median")),
                     "rankings_complete":all(len(v)==2560 for v in rankings["families"].values()),
                     "norm_matched":max(r["source_norm_error"] for r in case_records)<=1e-5},
           "runtime":{"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}}
  core.save(OUT/"analysis/full_field_summary.json",summary)
  auth="run_phase1394_c062_coordinate_curves_and_phase1395_event_mediation"
  core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_checks_passed":all(summary["checks"].values()),"authorization":auth})
  print(json.dumps(summary,indent=2))
 finally:
  if model is not None:release_bf16(model)


if __name__=="__main__":main()
