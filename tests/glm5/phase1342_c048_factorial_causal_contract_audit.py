#!/usr/bin/env python3
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[2];T=R/"tests/glm5";O=T/"result/phase1342_c048_factorial_causal_contract";P=T/"result/phase1341_c047_full_relation_field";M=("qwen3","glm4","deepseek7b")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def main():
 pr=load(O/"protocol/preregistration.json");fin=load(O/"analysis/final.json");mat=rows(O/"material/frozen_factorial_cases.jsonl");audit=load(O/"audit/pre_model_semantic_naturalness_zero_model_audit.json");base={k:v for k,v in pr.items() if k not in ("contract_sha256","authorization")};q=defaultdict(list)
 for x in mat:q[x["quartet_key"]].append(x)
 c={"parent":load(P/"analysis/final.json").get("authorization")=="close_c047_descriptive_field_and_authorize_separate_causal_preregistration" and load(P/"audit/independent_final_audit.json").get("all_checks_passed"),"hash":digest(base)==pr["contract_sha256"],"final":fin.get("authorization")=="run_phase1343_c048_factorial_behavior","count":len(mat)==864,"truth":Counter(x["truth"] for x in mat)=={True:432,False:432},"quartets":len(q)==216 and all([x["cell"] for x in z]==["aa","ab","ba","bb"] for z in q.values()),"material_audit":audit.get("all_checks_passed"),"gating":pr["behavior_gate"]["minimum_authorized_models"]==2 and pr["field_gate"]["minimum_authorized_models"]==2 and pr["causal_gate"]["minimum_authorized_models"]==2,"stop":"do not change" in pr["stop_rule"].lower()}
 for m in M:
  x=rows(O/f"compiled/{m}_factorial.jsonl");c[m]=len(x)==864 and all(a["case_id"]==b["case_id"] for a,b in zip(mat,x)) and all(len(z["tested_family_span"])==1 for z in x)
 res={"phase":1342,"campaign":"C048","checks":c,"passed":sum(c.values()),"total":len(c),"all_checks_passed":all(c.values()),"authorization":"run_phase1343_c048_factorial_behavior" if all(c.values()) else "deny_phase1343"};(O/"audit/independent_final_audit.json").write_text(json.dumps(res,indent=2)+"\n",encoding="utf-8");print(json.dumps(res,indent=2));raise SystemExit(0 if all(c.values()) else 1)
if __name__=="__main__":main()
