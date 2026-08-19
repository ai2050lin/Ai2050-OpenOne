#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[2];T=R/"tests/glm5";P=T/"result/phase1339_c047_paired_relation_contract";B=T/"result/phase1340_c047_paired_behavior";O=T/"result/phase1341_c047_full_relation_field";M=("qwen3","glm4")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def pre():
 x=load(O/"protocol/execution_manifest.json");c={"behavior":load(B/"audit/independent_final_audit.json").get("authorization")=="run_phase1341_c047_full_relation_field","models":x["model_order"]==list(M),"storage":x["primary_storage"].startswith("float32 complete"),"no_results":not any((O/f"analysis/{m}_summary.json").exists() for m in M)};res={"stage":"pre","checks":c,"all_checks_passed":all(c.values())};(O/"audit").mkdir(parents=True,exist_ok=True);(O/"audit/independent_preaudit.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res,indent=2));raise SystemExit(0 if all(c.values()) else 1)
def post():
 pr=load(P/"protocol/preregistration.json");fin=load(O/"analysis/final.json");c={};q=[]
 for m in M:
  s=load(O/f"analysis/{m}_summary.json");v=torch.load(O/f"raw/{m}_field.pt",map_location="cpu",weights_only=True);shape=v["vectors"].shape;c[m+"_shape"]=shape[0]==576 and shape[1:3]==(5,3);g=pr["hidden_gate"];eligible=[s["metrics"][f"d{d}:tested_family"] for d in (1,2,3,4)];qual=s["numeric"]["relative_l2_p95"]<=g["numeric_relative_l2_p95_max"] and s["numeric"]["relative_l2_max"]<=g["numeric_relative_l2_max"] and sum(x["identity_win"]>=g["cross_surface_identity_win_min"] and x["median_gap"]>=g["permutation_gap_min"] for x in eligible)>=2;c[m+"_qualified"]=s["qualified"]==qual
  if qual:q.append(m)
 passed=len(q)>=2;auth="close_c047_descriptive_field_and_authorize_separate_causal_preregistration" if passed else "close_c047_descriptive_field";c["final"]=fin["qualified_models"]==q and fin["authorization"]==auth and fin["causal_claim"] is False;res={"stage":"post","checks":c,"independently_qualified_models":q,"all_checks_passed":all(c.values()),"authorization":auth if all(c.values()) else "deny_next_campaign"};(O/"audit/independent_final_audit.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res,indent=2));raise SystemExit(0 if all(c.values()) else 1)
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--stage",choices=("pre","post"),required=True);x=a.parse_args();pre() if x.stage=="pre" else post()
