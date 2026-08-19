#!/usr/bin/env python3
"""Phase1394: frozen coordinate transfer and rediscovery curves for C062."""
from __future__ import annotations

import hashlib, inspect, json, math, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/"tests/glm5";sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16,quantization_audit,release_bf16

PHASE,CAMPAIGN=1394,"C062"
CONTRACT=TESTS/"result/phase1390_c062_route_factorized_field_campaign_contract"
BEHAVIOR=TESTS/"result/phase1391_c062_family_factorized_behavior"
FIELD=TESTS/"result/phase1393_c062_discovery_full_field"
OUT=TESTS/"result/phase1394_c062_coordinate_curves"
C060=TESTS/"result/phase1384_c060_fixed_dynamic_coalitions/protocol/discovery_rankings.json"
DONOR_KEYS=("clean_true","corrupt_false","wrong_identity_true","status_true")
SPECS=(("sufficiency","self"),("sufficiency","correct"),("sufficiency","wrong"),("sufficiency","status"),("sufficiency","random"),
       ("reverse","self"),("reverse","correct"),("reverse","status"),("reverse","random"))
CHUNK=8


def parents():
 f=core.load(FIELD/"analysis/final.json");a=core.load(FIELD/"audit/independent_final_audit.json")
 if f["authorization"]!="run_phase1394_c062_coordinate_curves_and_phase1395_event_mediation" or not a["all_checks_passed"]:raise RuntimeError("field not authorized")
 return core.load(CONTRACT/"protocol/preregistration.json"),core.load(BEHAVIOR/"analysis/final.json")
def stable_seed(*parts):return int(hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:15],16)%(2**63-1)
def make_batch(rows,pad,device):
 width=max(len(r["prompt_ids"]) for r in rows);ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);offs=[]
 for i,r in enumerate(rows):
  v=torch.tensor(r["prompt_ids"],dtype=torch.long,device=device);o=width-len(v);offs.append(o);ids[i,o:]=v;mask[i,o:]=1
 pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0);return ids,mask,pos,offs
def points(r,o,role):return [o+p for p in r["role_positions"][role]]
def margin(out,i,row):
 z=out.logits[i,-1].float();return float(z[row["candidate_ids"][0][0]]-z[row["candidate_ids"][1][0]])
def scaled(v,n):
 x=torch.linalg.vector_norm(v)
 if float(x)<=1e-12:raise RuntimeError("zero control")
 return v*(n/x)


def coords(group,case,full,c060,c062,device):
 k=group["size"];rule=group["rule"]
 if rule=="c060_family_fixed":values=c060["families"][case["target_family"]][:k]
 elif rule=="c060_global_fixed":values=c060["global"][:k]
 elif rule=="c062_family_discovery":values=c062["families"][case["target_family"]][:k]
 elif rule=="c062_global_discovery":values=c062["global"][:k]
 elif rule=="per_example_top_abs":values=torch.argsort(full.flatten().abs(),descending=True,stable=True)[:k].cpu().tolist()
 elif rule=="per_example_bottom_abs":values=torch.argsort(full.flatten().abs(),descending=False,stable=True)[:k].cpu().tolist()
 elif rule=="deterministic_random":
  g=torch.Generator(device="cpu");g.manual_seed(stable_seed(case["pair_id"],k,6201394));values=torch.randperm(full.numel(),generator=g)[:k].tolist()
 else:raise RuntimeError(rule)
 return torch.tensor(values,dtype=torch.long,device=device)


def metric(rows,gate):
 if not rows:return None
 sc=[r["suff"]["correct"] for r in rows];sa=[r["suff"]["correct"]-max(r["suff"][x] for x in ("wrong","status","random")) for r in rows]
 sw=[v>0 for v in sa];fr=[r["suff"]["correct"]/r["whole_effect"] for r in rows if abs(r["whole_effect"])>1e-12]
 rd=[r["reverse"]["correct"] for r in rows];ra=[r["reverse"]["correct"]-r["reverse"]["status"] for r in rows]
 d={"count":len(rows),"suff_gain_median":statistics.median(sc),"suff_advantage_median":statistics.median(sa),
    "suff_win_fraction":sum(sw)/len(sw),"whole_effect_fraction_median":statistics.median(fr),
    "reverse_damage_median":statistics.median(rd),"reverse_over_status_median":statistics.median(ra),
    "reverse_over_status_win_fraction":sum(v>0 for v in ra)/len(ra),"self_max_abs_diff":max(r["self_max_abs_diff"] for r in rows),
    "norm_error_max":max(r["norm_error_max"] for r in rows)}
 d["sufficiency_qualified"]=(d["suff_gain_median"]>=gate["suff_gain_median_min"] and d["suff_advantage_median"]>=gate["suff_advantage_median_min"] and
  d["suff_win_fraction"]>=gate["suff_win_min"] and d["whole_effect_fraction_median"]>=gate["whole_effect_fraction_median_min"] and d["self_max_abs_diff"]<=gate["self_max_abs_diff"])
 d["reverse_qualified"]=(d["reverse_damage_median"]>=gate["reverse_damage_median_min"] and d["reverse_over_status_median"]>=gate["reverse_over_status_median_min"] and
  d["reverse_over_status_win_fraction"]>=gate["reverse_over_status_win_min"] and d["self_max_abs_diff"]<=gate["self_max_abs_diff"])
 return d


@torch.inference_mode()
def main():
 protocol,behavior=parents()
 if (OUT/"analysis/final.json").exists():raise RuntimeError("Phase1394 already exists")
 gate=protocol["coordinates"];c060=core.load(C060);c062=core.load(FIELD/"protocol/discovery_rankings.json")
 if core.sha(C060)!=gate["c060_rankings_sha256"]:raise RuntimeError("C060 rankings changed")
 groups=[]
 for rule in gate["routes"]:
  for k in gate["sizes"]:groups.append({"group_id":f"{rule}@{k}","rule":rule,"size":k})
 core.save(OUT/"protocol/execution_manifest.json",{"phase":PHASE,"groups":groups,"specs":[list(v) for v in SPECS],"chunk":CHUNK,
           "case_partitions":gate["evaluation_partitions"],"post_reveal_changes_forbidden":True})
 cases=[r for r in core.rows(BEHAVIOR/"material/eligible_pairs.jsonl") if r["partition"] in gate["evaluation_partitions"]]
 compiled={r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_active.jsonl")};compiled.update({r["case_id"]:r for r in core.rows(CONTRACT/"compiled/qwen3_status.jsonl")})
 model=None
 try:
  model,tok,device,placement=load_bf16("qwen3");quant=quantization_audit(model);pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id);supports="logits_to_keep" in inspect.signature(model.forward).parameters
  records=[]
  for ci,case in enumerate(cases):
   available=[g for g in groups if g["rule"]!="c060_family_fixed" or case["target_family"] in c060["families"]]
   donors=[compiled[case[k]] for k in DONOR_KEYS]
   for start in range(0,len(available),CHUNK):
    chunk=available[start:start+CHUNK];rows=list(donors)
    for _ in chunk:
     for mode,_arm in SPECS:rows.append(donors[1 if mode=="sufficiency" else 0])
    ids,mask,pos,offs=make_batch(rows,pad,device);errors=[0.0]*(len(chunk)*len(SPECS))
    def hook(_m,args):
     original=args[0];value=original.clone();role=protocol["source"]["role"]
     dv=[original[i,points(rows[i],offs[i],role)].float() for i in range(4)];full_for_mask=dv[0]-dv[1]
     cc={g["group_id"]:coords(g,case,full_for_mask,c060,c062,original.device) for g in chunk}
     for gi,g in enumerate(chunk):
      ix=cc[g["group_id"]]
      for si,(mode,arm) in enumerate(SPECS):
       local=gi*len(SPECS)+si;ti=4+local;origin,goal=((1,0) if mode=="sufficiency" else (0,1));full=dv[goal]-dv[origin]
       correct=torch.zeros_like(full);correct[...,ix]=full[...,ix];n=torch.linalg.vector_norm(correct)
       if arm=="self":direction=torch.zeros_like(correct)
       elif arm=="correct":direction=correct
       elif arm in ("wrong","status"):
        di=2 if arm=="wrong" else 3;raw=torch.zeros_like(correct);raw[...,ix]=(dv[di]-dv[origin])[...,ix];direction=scaled(raw,n)
       else:
        gen=torch.Generator(device=original.device);gen.manual_seed(stable_seed(case["pair_id"],g["group_id"],mode,6201394));raw=torch.zeros_like(correct)
        raw[...,ix]=torch.randn((correct.shape[0],ix.numel()),generator=gen,device=original.device,dtype=torch.float32);direction=scaled(raw,n)
       if arm!="self":errors[local]=abs(float(torch.linalg.vector_norm(direction)/(n+1e-12))-1.0)
       tp=points(rows[ti],offs[ti],role);value[ti,tp]=original[ti,tp]+direction.to(original.dtype)
     return (value,)+args[1:]
    h=model.model.layers[protocol["source"]["layer"]].register_forward_pre_hook(hook)
    try:
     kw={"input_ids":ids,"attention_mask":mask,"position_ids":pos,"use_cache":False,"return_dict":True}
     if supports:kw["logits_to_keep"]=1
     out=model(**kw)
    finally:h.remove()
    cm=margin(out,0,donors[0]);xm=margin(out,1,donors[1]);whole=cm-xm
    for gi,g in enumerate(chunk):
     eff={"sufficiency":{},"reverse":{}};selfs=[];errs=[]
     for si,(mode,arm) in enumerate(SPECS):
      local=gi*len(SPECS)+si;ti=4+local;m=margin(out,ti,rows[ti]);e=m-xm if mode=="sufficiency" else cm-m
      eff[mode][arm]=e;errs.append(errors[local]);
      if arm=="self":selfs.append(abs(e))
     records.append({"pair_id":case["pair_id"],"partition":case["partition"],"surface":case["surface"],"family":case["target_family"],"panel":case["panel"],
                     **g,"whole_effect":whole,"suff":eff["sufficiency"],"reverse":eff["reverse"],"self_max_abs_diff":max(selfs),"norm_error_max":max(errs)})
    del out,ids,mask,pos
   if (ci+1)%12==0:print(json.dumps({"coordinate_cases":ci+1,"total":len(cases)}),flush=True)
  core.write_rows(OUT/"raw/coordinate_curves.jsonl",records)
  metrics={}
  for g in groups:
   gr=[r for r in records if r["group_id"]==g["group_id"]]
   if not gr:continue
   splits={p:metric(gr if p=="pooled" else [r for r in gr if r["partition"]==p],gate) for p in ("pooled","confirmation","lockbox")}
   families={fam:{p:metric([r for r in gr if r["family"]==fam and (p=="pooled" or r["partition"]==p)],gate)
                  for p in ("pooled","confirmation","lockbox")} for fam in sorted({r["family"] for r in gr})}
   metrics[g["group_id"]]={"group":g,"splits":splits,"families":families,
      "suff_all_holdouts":all(splits[p]["sufficiency_qualified"] for p in ("confirmation","lockbox")),
      "reverse_all_holdouts":all(splits[p]["reverse_qualified"] for p in ("confirmation","lockbox")),
      "family_suff_hit_count":sum(all(v[p]["sufficiency_qualified"] for p in ("confirmation","lockbox")) for v in families.values()),
      "family_reverse_hit_count":sum(all(v[p]["reverse_qualified"] for p in ("confirmation","lockbox")) for v in families.values())}
  overlap={}
  for fam in behavior["qualified_families"]:
   if fam not in c060["families"]:continue
   overlap[fam]={}
   for k in gate["sizes"]:
    inter=len(set(c060["families"][fam][:k])&set(c062["families"][fam][:k]));overlap[fam][str(k)]={"intersection":inter,"intersection_fraction":inter/k,"chance_fraction":k/2560,
       "enrichment":inter/k-k/2560}
  first_qualified={}
  for route in gate["routes"]:
   hits=[k for k in gate["sizes"] if f"{route}@{k}" in metrics and metrics[f"{route}@{k}"]["suff_all_holdouts"]]
   first_qualified[route]=min(hits) if hits else None
  primary_transfer=metrics.get(gate["same_family_transfer_primary"]);primary_rediscovery=metrics.get(gate["rediscovery_primary"])
  summary={"phase":PHASE,"campaign":CAMPAIGN,"case_count":len(cases),"record_count":len(records),"metrics":metrics,"overlap":overlap,
           "first_tested_qualified_size":first_qualified,
           "primary":{"c060_family_transfer_suff":bool(primary_transfer and primary_transfer["suff_all_holdouts"]),
                      "c060_family_transfer_reverse":bool(primary_transfer and primary_transfer["reverse_all_holdouts"]),
                      "c062_family_rediscovery_suff":bool(primary_rediscovery and primary_rediscovery["suff_all_holdouts"]),
                      "c062_family_rediscovery_reverse":bool(primary_rediscovery and primary_rediscovery["reverse_all_holdouts"]),
                      "rediscovery_family_suff_hit_count":primary_rediscovery["family_suff_hit_count"] if primary_rediscovery else 0},
           "checks":{"case_count":len(cases)==144,"partitions":set(r["partition"] for r in records)=={"confirmation","lockbox"},
                     "finite":all(math.isfinite(v) for r in records for v in list(r["suff"].values())+list(r["reverse"].values())),
                     "self":max(r["self_max_abs_diff"] for r in records)<=gate["self_max_abs_diff"],"norm":max(r["norm_error_max"] for r in records)<=1e-5},
           "runtime":{"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}}
  core.save(OUT/"analysis/coordinate_summary.json",summary)
  core.save(OUT/"analysis/final.json",{"phase":PHASE,"campaign":CAMPAIGN,"all_checks_passed":all(summary["checks"].values()),
            "primary":summary["primary"],"authorization":"run_phase1395_c062_event_bundle_mediation_then_close"})
  print(json.dumps({k:v for k,v in summary.items() if k not in {"metrics"}},indent=2))
 finally:
  if model is not None:release_bf16(model)


if __name__=="__main__":main()
