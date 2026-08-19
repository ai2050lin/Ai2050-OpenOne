#!/usr/bin/env python3
"""Independent artifact audit for Phase1351/C052."""
from __future__ import annotations
import json, py_compile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1351_c052_qwen_pair_probe_contract"

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def rows(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    final, protocol = load(OUT/"analysis/final.json"), load(OUT/"protocol/preregistration.json")
    pre, data, compiled = load(OUT/"audit/pre_model_material_audit.json"), rows(OUT/"material/frozen_cases.jsonl"), rows(OUT/"compiled/qwen3_cases.jsonl")
    groups=defaultdict(list)
    for r in data: groups[r["quartet_key"]].append(r)
    checks={
      "authorization": final.get("authorization")=="run_phase1352_c052_qwen_behavior",
      "contract": final.get("contract_sha256")==protocol.get("contract_sha256"),
      "preaudit": pre.get("all_checks_passed") and pre.get("passed")==pre.get("total"),
      "qwen_only": protocol.get("model")=="qwen3",
      "counts": len(data)==4608 and len(compiled)==4608,
      "panels": Counter(r["panel"] for r in data)=={"core_membership":1536,"role_bound_lexical":1536,"explicit_status":1536},
      "quartets": len(groups)==1152 and all(len(v)==4 for v in groups.values()),
      "partitions": set(r["partition"] for r in data)=={"prototype_discovery","clock_selection","confirmation","holdout"},
      "spans": all(r["target_span"] and r["tested_family_span"] for r in compiled),
      "probe_frozen": protocol["probe_gate"]["probe"]=="full-dimensional cosine nearest centroid",
      "no_reduction": protocol["probe_gate"]["no_dimension_reduction"] and protocol["probe_gate"]["no_probe_hyperparameter_search"],
      "claim_boundary": "readability only" in protocol["claim_boundary"],
      "stop": "do not change" in protocol["stop_rule"],
      "compiled_script": True,
    }
    try: py_compile.compile(str(TESTS/"phase1351_c052_qwen_pair_probe_contract.py"),doraise=True)
    except Exception: checks["compiled_script"]=False
    result={"phase":1351,"campaign":"C052","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values())}
    (OUT/"audit").mkdir(parents=True,exist_ok=True)
    (OUT/"audit/independent_final_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)

if __name__=="__main__": main()
