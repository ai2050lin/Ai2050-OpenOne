#!/usr/bin/env python3
import argparse,json,math
from collections import defaultdict
from pathlib import Path
from statistics import median
R=Path(__file__).resolve().parents[2];T=R/"tests/glm5";P=T/"result/phase1339_c047_paired_relation_contract";O=T/"result/phase1340_c047_paired_behavior";M=("qwen3","glm4","deepseek7b")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]
def pre():
 man=load(O/"protocol/execution_manifest.json");c={"parent":load(P/"audit/independent_final_audit.json").get("all_checks_passed"),"order":man["model_order"]==list(M),"precision":man["precision"]=="bfloat16-no-quantization","no_results":not any((O/f"analysis/{m}_summary.json").exists() for m in M)};res={"stage":"pre","checks":c,"passed":sum(c.values()),"total":len(c),"all_checks_passed":all(c.values())};(O/"audit").mkdir(parents=True,exist_ok=True);(O/"audit/independent_preaudit.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res,indent=2));raise SystemExit(0 if all(c.values()) else 1)
def post():
 pr=load(P/"protocol/preregistration.json");fin=load(O/"analysis/final.json");c={};q=[]
 for m in M:
  s=load(O/f"analysis/{m}_summary.json");r=rows(O/f"raw/{m}_behavior.jsonl");groups=defaultdict(list)
  for x in r:groups[x["quartet_key"]].append(x)
  gaps=[];quart=[]
  for z in groups.values():
   t=next(x for x in z if x["truth"]);g=[t["semantic_margin"]-x["semantic_margin"] for x in z if not x["truth"]];gaps+=g;quart.append(all(v>0 for v in g))
  met=s["behavior_metrics"];g=pr["behavior_gate"];qual=(len(r)==576 and met["accuracy"]>=g["accuracy_min"] and min(met["partition"].values())>=g["partition_min"] and min(met["surface"].values())>=g["surface_min"] and min(met["family"].values())>=g["target_family_min"] and min(met["truth"].values())>=g["truth_cell_min"] and sum(v>0 for v in gaps)/len(gaps)>=g["pairwise_gap_win_min"] and sum(quart)/len(quart)>=g["quartet_rank_min"] and median(gaps)>=g["median_relation_gap_min"] and s["executor"]["qualified"])
  if qual:q.append(m)
  c[m+"_count"]=len(r)==576;c[m+"_qualified"]=s["qualified"]==qual
 auth="run_phase1341_c047_full_relation_field" if len(q)>=2 else "close_c047_behavior";c["final"]=fin["qualified_models"]==q and fin["authorization"]==auth;res={"stage":"post","checks":c,"independently_qualified_models":q,"passed":sum(c.values()),"total":len(c),"all_checks_passed":all(c.values()),"authorization":auth if all(c.values()) else "deny_phase1341"};(O/"audit/independent_final_audit.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res,indent=2));raise SystemExit(0 if all(c.values()) else 1)
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--stage",choices=("pre","post"),required=True);x=a.parse_args();pre() if x.stage=="pre" else post()
