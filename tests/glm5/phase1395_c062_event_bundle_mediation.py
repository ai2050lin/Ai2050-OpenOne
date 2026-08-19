#!/usr/bin/env python3
"""Phase1395: frozen discovery-event bundle mediation on C062 holdouts."""
from __future__ import annotations

import inspect,json,math,statistics,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16

PHASE,CAMPAIGN=1395,"C062"
CONTRACT=TESTS/"result/phase1390_c062_route_factorized_field_campaign_contract"
BEHAVIOR=TESTS/"result/phase1391_c062_family_factorized_behavior"
FIELD=TESTS/"result/phase1393_c062_discovery_full_field"
COORD=TESTS/"result/phase1394_c062_coordinate_curves"
OUT=TESTS/"result/phase1395_c062_event_bundle_mediation"
BUNDLES=("top1","stage_top1","stage_top2","query_reference","boundary_reference")
ARMS=("corrupt","clean","wrong")


def parents():
 f=core.load(COORD/"analysis/final.json");a=core.load(COORD/"audit/independent_final_audit.json")
 if f["authorization"]!="run_phase1395_c062_event_bundle_mediation_then_close" or not a["all_checks_passed"]:raise RuntimeError("coordinate phase not authorized")
 return core.load(CONTRACT/"protocol/preregistration.json")
def make_batch(rows,pad,device):
 width=max(len(r["prompt_ids"]) for r in rows);ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);offs=[]
 for i,r in enumerate(rows):
  v=torch.tensor(r["prompt_ids"],dtype=torch.long,device=device);o=width-len(v);offs.append(o);ids[i,o:]=v;mask[i,o:]=1
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,offs
def points(r,o,role):return [o+p for p in r["role_positions"][role]]
def margin(out,i,row):
 z=out.logits[i,-1].float();return float(z[row["candidate_ids"][0][0]]-z[row["candidate_ids"][1][0]])


def metrics(rows,gate):
 rescue=[r["rescue"] for r in rows];blocks=[(r["rescue"]-r["reset_corrupt"])/(r["rescue"]+1e-12) for r in rows]
 clean=[abs(r["rescue"]-r["reset_clean"])/(abs(r["rescue"])+1e-12) for r in rows]
 wrong=[(r["rescue"]-r["reset_wrong"])/(r["rescue"]+1e-12) for r in rows]
 d={"count":len(rows),"upstream_rescue_median":statistics.median(rescue),"block_fraction_median":statistics.median(blocks),
    "block_positive_fraction":sum(v>0 for v in blocks)/len(blocks),"clean_control_loss_fraction_median":statistics.median(clean),
    "wrong_block_fraction_median":statistics.median(wrong)}
 d["qualified"]=(d["upstream_rescue_median"]>=gate["upstream_rescue_median_min"] and d["block_fraction_median"]>=gate["block_fraction_median_min"] and
                  d["block_positive_fraction"]>=gate["block_positive_fraction_min"] and d["clean_control_loss_fraction_median"]<=gate["clean_checkpoint_control_loss_fraction_max"])
 return d


@torch.inference_mode()
def main():
 protocol=parents()
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1395 already exists")
 gate=protocol["mediation"];candidate=core.load(FIELD/"protocol/discovery_event_candidates.json")
 cases=[r for r in core.rows(BEHAVIOR/"material/eligible_pairs.jsonl") if r["partition"] in gate["evaluation_partitions"]]
 compiled={r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_active.jsonl")};compiled.update({r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_status.jsonl")})
 core.save(OUT/"protocol/execution_manifest.json",{"phase":PHASE,"candidate_sha256":core.sha(FIELD/"protocol/discovery_event_candidates.json"),
           "bundles":list(BUNDLES),"arms":list(ARMS),"case_ids":[r["pair_id"] for r in cases],"post_reveal_changes_forbidden":True})
 model=None
 try:
  model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);supports="logits_to_keep" in inspect.signature(model.forward).parameters
  records=[]
  for ci,case in enumerate(cases):
   donors=[compiled[case[k]] for k in ("clean_true","corrupt_false","wrong_identity_true","status_true")]
   rows=list(donors)+[donors[1]]
   target_map={}
   for bundle in BUNDLES:
    for arm in ARMS:target_map[(bundle,arm)]=len(rows);rows.append(donors[1])
   ids,mask,pos,offs=make_batch(rows,pad,device);handles=[];surface_bundles=candidate["bundles"][case["surface"]]
   def source_hook(_m,args):
    original=args[0];value=original.clone();role=protocol["source"]["role"];clean=original[0,points(rows[0],offs[0],role)].float();corrupt=original[1,points(rows[1],offs[1],role)].float();d=clean-corrupt
    for ti in range(4,len(rows)):
     tp=points(rows[ti],offs[ti],role);value[ti,tp]=original[ti,tp]+d.to(original.dtype)
    return (value,)+args[1:]
   handles.append(model.model.layers[protocol["source"]["layer"]].register_forward_pre_hook(source_hook))
   events_by_state=defaultdict(list)
   for bundle in BUNDLES:
    for event in surface_bundles[bundle]:events_by_state[event["state_index"]].append((bundle,event["position"]))
   for state,events in sorted(events_by_state.items()):
    if not 4<=state<=35:raise RuntimeError(f"unhookable state {state}")
    def checkpoint_hook(_m,args,events=tuple(events)):
     original=args[0];value=original.clone();donor_for={"corrupt":1,"clean":0,"wrong":2}
     for bundle,p in events:
      for arm in ARMS:
       ti=target_map[(bundle,arm)];di=donor_for[arm];value[ti,offs[ti]+p]=original[di,offs[di]+p]
     return (value,)+args[1:]
    handles.append(model.model.layers[state].register_forward_pre_hook(checkpoint_hook))
   try:
    kw={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"return_dict":True}
    if supports:kw["logits_to_keep"]=1
    out=model(**kw)
   finally:
    for h in handles:h.remove()
   corrupt=margin(out,1,donors[1]);rescue=margin(out,4,rows[4])-corrupt
   for bundle in BUNDLES:
    vals={arm:margin(out,target_map[(bundle,arm)],rows[target_map[(bundle,arm)]])-corrupt for arm in ARMS}
    records.append({"pair_id":case["pair_id"],"partition":case["partition"],"surface":case["surface"],"family":case["target_family"],
                    "bundle":bundle,"event_count":len(surface_bundles[bundle]),"rescue":rescue,"reset_corrupt":vals["corrupt"],
                    "reset_clean":vals["clean"],"reset_wrong":vals["wrong"]})
   del out,ids,mask,pos
   if (ci+1)%12==0:print(json.dumps({"mediation_cases":ci+1,"total":len(cases)}),flush=True)
  core.write_rows(OUT/"raw/event_bundle_mediation.jsonl",records)
  summary_metrics={}
  for bundle in BUNDLES:
   br=[r for r in records if r["bundle"]==bundle]
   splits={p:metrics(br if p=="pooled" else [r for r in br if r["partition"]==p],gate) for p in ("pooled","confirmation","lockbox")}
   fams={fam:{p:metrics([r for r in br if r["family"]==fam and (p=="pooled" or r["partition"]==p)],gate) for p in ("pooled","confirmation","lockbox")}
         for fam in sorted({r["family"] for r in br})}
   summary_metrics[bundle]={"splits":splits,"families":fams,"qualified_all_holdouts":all(splits[p]["qualified"] for p in ("confirmation","lockbox")),
                            "family_hit_count":sum(all(v[p]["qualified"] for p in ("confirmation","lockbox")) for v in fams.values())}
  qualified=[b for b,v in summary_metrics.items() if v["qualified_all_holdouts"]]
  summary={"phase":PHASE,"campaign":CAMPAIGN,"case_count":len(cases),"record_count":len(records),"metrics":summary_metrics,
           "qualified_bundles":qualified,"top1_qualified":"top1" in qualified,"stage_top1_qualified":"stage_top1" in qualified,
           "stage_top2_qualified":"stage_top2" in qualified,"query_reference_qualified":"query_reference" in qualified,
           "boundary_reference_qualified":"boundary_reference" in qualified,
           "checks":{"case_count":len(cases)==144,"record_count":len(records)==144*len(BUNDLES),
                     "holdout_only":set(r["partition"] for r in records)=={"confirmation","lockbox"},
                     "finite":all(math.isfinite(r[k]) for r in records for k in ("rescue","reset_corrupt","reset_clean","reset_wrong")),
                     "candidate_hash":core.sha(FIELD/"protocol/discovery_event_candidates.json")==core.load(OUT/"protocol/execution_manifest.json")["candidate_sha256"]},
           "runtime":{"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}}
  core.save(OUT/"analysis/event_mediation_summary.json",summary)
  core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_checks_passed":all(summary["checks"].values()),
            "qualified_bundles":qualified,"authorization":"run_phase1396_c062_campaign_closure"})
  print(json.dumps({k:v for k,v in summary.items() if k!="metrics"},indent=2))
 finally:
  if model is not None:release_bf16(model)


if __name__=="__main__":main()
