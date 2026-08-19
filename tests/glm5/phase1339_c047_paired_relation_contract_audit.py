#!/usr/bin/env python3
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[2]; T=R/"tests/glm5"; O=T/"result/phase1339_c047_paired_relation_contract"; P=T/"result/phase1338_c046_deconfounded_behavior"
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]
def digest(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def main():
 pr=load(O/"protocol/preregistration.json"); fin=load(O/"analysis/final.json"); mat=rows(O/"material/frozen_behavior_cases.jsonl"); audit=load(O/"audit/pre_model_semantic_naturalness_zero_model_audit.json")
 base={k:v for k,v in pr.items() if k not in ("contract_sha256","authorization")}; q=defaultdict(list)
 for r in mat:q[r["quartet_key"]].append(r)
 c={"parent":load(P/"analysis/final.json").get("authorization")=="close_c046_behavior" and load(P/"audit/independent_final_audit.json").get("all_checks_passed"),"hash":digest(base)==pr["contract_sha256"],"final":fin.get("authorization")=="run_phase1340_c047_paired_behavior","count":len(mat)==576,"truth":Counter(r["truth"] for r in mat)=={True:144,False:432},"quartets":len(q)==144 and all(len(v)==4 and sum(r["truth"] for r in v)==1 for v in q.values()),"audit":audit.get("all_checks_passed"),"branch":pr["behavior_gate"]["minimum_authorized_models"]==2 and "do not change" in pr["stop_rule"].lower()}
 for m in pr["models"]:
  x=rows(O/f"compiled/{m}_behavior.jsonl"); c[m]=len(x)==576 and all(a["case_id"]==b["case_id"] for a,b in zip(mat,x)) and all(all(len(z)==1 for z in r["candidate_ids"]) for r in x)
 result={"phase":1339,"campaign":"C047","checks":c,"passed":sum(c.values()),"total":len(c),"all_checks_passed":all(c.values()),"authorization":"run_phase1340_c047_paired_behavior" if all(c.values()) else "deny_phase1340"}
 (O/"audit/independent_final_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); raise SystemExit(0 if all(c.values()) else 1)
if __name__=="__main__":main()
