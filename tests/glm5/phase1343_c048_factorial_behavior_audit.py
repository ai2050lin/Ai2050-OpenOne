#!/usr/bin/env python3
import argparse,json
from collections import defaultdict
from pathlib import Path
from statistics import median
R=Path(__file__).resolve().parents[2];T=R/"tests/glm5";P=T/"result/phase1342_c048_factorial_causal_contract";O=T/"result/phase1343_c048_factorial_behavior";M=("qwen3","glm4","deepseek7b")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]
def pre():
 x=load(O/"protocol/execution_manifest.json");c={"parent":load(P/"audit/independent_final_audit.json").get("all_checks_passed"),"order":x["model_order"]==list(M),"batch":x["batch_size"]==4,"no_results":not any((O/f"analysis/{m}_summary.json").exists() for m in M)};z={"stage":"pre","checks":c,"all_checks_passed":all(c.values())};(O/"audit").mkdir(parents=True,exist_ok=True);(O/"audit/independent_preaudit.json").write_text(json.dumps(z,indent=2)+"\n");print(json.dumps(z,indent=2));raise SystemExit(0 if all(c.values()) else 1)
def post():
 pr=load(P/"protocol/preregistration.json");fin=load(O/"analysis/final.json");c={};q=[]
 for m in M:
  s=load(O/f"analysis/{m}_summary.json");r=rows(O/f"raw/{m}_behavior.jsonl");g=pr["behavior_gate"];met=s["behavior_metrics"];qual=len(r)==864 and s["executor"]["qualified"] and met["accuracy"]>=g["accuracy_min"] and min(met["partition"].values())>=g["partition_min"] and min(met["surface"].values())>=g["surface_min"] and min(met["family"].values())>=g["family_min"] and min(met["truth"].values())>=g["truth_min"] and met["pairwise_true_over_false"]>=g["pairwise_true_over_false_min"] and met["quartet_all_correct"]>=g["quartet_all_correct_min"] and met["positive_interaction_fraction"]>=g["positive_interaction_fraction_min"] and met["median_interaction"]>=g["median_interaction_min"];c[m+"_count"]=len(r)==864;c[m+"_qualified"]=s["qualified"]==qual
  if qual:q.append(m)
 auth="run_phase1344_c048_interaction_field" if len(q)>=2 else "close_c048_behavior";c["final"]=fin["qualified_models"]==q and fin["authorization"]==auth;z={"stage":"post","checks":c,"independently_qualified_models":q,"all_checks_passed":all(c.values()),"authorization":auth if all(c.values()) else "deny_phase1344"};(O/"audit/independent_final_audit.json").write_text(json.dumps(z,indent=2)+"\n");print(json.dumps(z,indent=2));raise SystemExit(0 if all(c.values()) else 1)
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--stage",choices=("pre","post"),required=True);x=a.parse_args();pre() if x.stage=="pre" else post()
