#!/usr/bin/env python3
"""Independent audit for Phase1335 C045 contract."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; T=ROOT/"tests/glm5"; sys.path.insert(0,str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
OUT=T/"result/phase1335_c045_standard_executor_contract"; PARENT=T/"result/phase1334_c044_numeric_factorial"
MODELS=("qwen3","glm4","deepseek7b")


def main():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"analysis/final.json")
    pa=core.load(PARENT/"analysis/final.json"); ma=core.load(OUT/"audit/pre_model_semantic_naturalness_zero_model_audit.json")
    graph=core.load(OUT/"material/frozen_concept_graph.json")["concepts"]
    behavior=core.rows(OUT/"material/frozen_behavior_cases.jsonl"); contexts=core.rows(OUT/"material/frozen_context_cases.jsonl")
    b=[x for x in behavior if x["interface"]=="binary"]; c=[x for x in behavior if x["interface"]=="choice"]
    g=[x for x in behavior if x["interface"]=="generation"]
    frozen={k:v for k,v in p.items() if k not in {"contract_sha256","authorization"}}
    old=set()
    for path in (T/"result/phase1329_c042_relational_ecology_contract/material/frozen_concept_graph.json",
                 T/"result/phase1331_c043_native_relational_contract/material/frozen_concept_graph.json",
                 T/"result/phase1333_c044_relational_measurement_contract/material/frozen_concept_graph.json"):
        old.update(x["word"] for x in core.load(path)["concepts"])
    checks={"parent":pa["authorization"]=="close_c044_numeric_factorial",
            "contract_hash":core.digest(frozen)==p["contract_sha256"],
            "sources":core.sha(T/"phase1335_c045_standard_executor_contract.py")==p["script_sha256"] and
                      core.sha(Path(__file__).resolve())==p["auditor_sha256"] and
                      core.sha(T/"phase1333_c044_relational_measurement_contract.py")==p["compiler_sha256"],
            "hashes":core.sha(OUT/"material/frozen_concept_graph.json")==p["material"]["graph_sha256"] and
                     core.sha(OUT/"material/frozen_behavior_cases.jsonl")==p["material"]["behavior_sha256"] and
                     core.sha(OUT/"material/frozen_context_cases.jsonl")==p["material"]["context_sha256"],
            "fresh":not ({x["word"] for x in graph}&old),"counts":len(graph)==48 and len(b)==288 and len(c)==96 and len(g)==96 and len(contexts)==144,
            "binary":Counter(x["gold_value"] for x in b)=={"yes":144,"no":144},
            "choice":Counter(x["gold_position"] for x in c)=={0:24,1:24,2:24,3:24},
            "generation":Counter(x["gold_value"] for x in g)=={"fish":24,"flower":24,"footwear":24,"kitchen utensil":24},
            "material_audit":ma["all_checks_passed"],"executor":p["standard_executor"]["batch_size"]==8 and p["standard_executor"]["cross_shape_status"]=="engineering_diagnostic_only",
            "behavior_before_hidden":p["executor_gate"]["failure"].endswith("before behavior and hidden states"),
            "parameter_boundary":p["parameter_boundary"].startswith("No natural-model parameter"),
            "authorization":f["authorization"]==p["authorization"]=="run_phase1336_c045_standard_behavior"}
    for model in MODELS:
        cb=core.rows(OUT/f"compiled/{model}_behavior.jsonl"); ch=core.rows(OUT/f"compiled/{model}_context.jsonl")
        checks[f"{model}_compiled"]=len(cb)==480 and len(ch)==144
        checks[f"{model}_binary_tokens"]=all(all(len(y)==1 for y in x["candidate_ids"]) for x in cb if x["interface"]=="binary")
        checks[f"{model}_spans"]=all(x["target_span"] and max(x["target_span"])<x["boundary_position"] for x in ch)
    result={"phase":1335,"campaign":"C045","checks":checks,"passed":sum(checks.values()),"total":len(checks),
            "all_checks_passed":all(checks.values()),"authorization":p["authorization"] if all(checks.values()) else "none"}
    core.save(OUT/"audit/independent_final_audit.json",result);print(json.dumps(result,indent=2))
    if not result["all_checks_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
