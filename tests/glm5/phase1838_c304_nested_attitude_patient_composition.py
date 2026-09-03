#!/usr/bin/env python3
"""C304: isolate attitude wrapping x patient specialization in the lockbox."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C304"]


def main() -> None:
    parent_sha=core.sha(common.OUTS["C302"]/"analysis/final.json")
    if (OUT/"protocol/preregistration.json").exists() and core.load(OUT/"protocol/preregistration.json").get("parent_sha256")==parent_sha and (OUT/"analysis/final.json").exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C303"]/"analysis/final.json"); rows=core.rows(common.OUTS["C302"]/"raw/group_results.jsonl"); nested=[r for r in rows if r["family"]=="nested_attitude"]
    compiled=core.rows(common.OUTS["C294"]/"compiled/qwen3.jsonl"); examples=[]
    for a,b in ((0,0),(1,0),(0,1),(1,1)):
        item=next(r for r in compiled if r["family"]=="nested_attitude" and r["surface"]=="dossier" and r["unit"]==0 and r["order"]==1 and r["factor_a"]==a and r["factor_b"]==b); examples.append({"factor_a":a,"factor_b":b,"prompt_core":item["prompt_core"],"answer":item["correct_answer"]})
    checks={"parent":parent["all_checks_passed"],"groups":len(nested)==32,"actual_examples_only":len(examples)==4};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1838,"campaign":"C304","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"nested_composition_adjudication_frozen","parent_sha256":parent_sha,"factor_a":"reported that vs was pleased to learn that (attitude wrapping)","factor_b":"parent-class value vs concrete object (patient specialization)","test":"training-material mean interaction residual predicts sixth-material H11 from H00,H10,H01","gate":"overall gain>=1%, both surfaces positive, at least six of eight units positive","claim_boundary":"These are the actually run reported/pleased templates, not a fabricated literal 'I like eating apples' test. Static factorial states do not reveal execution order or commutation.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    surface={s:float(np.mean([r["relative_gain"] for r in nested if r["surface"]==s])) for s in common.SURFACES}; unit={u:float(np.mean([r["relative_gain"] for r in nested if r["unit"]==u])) for u in range(8)}; overall=float(np.mean([r["relative_gain"] for r in nested])); positive=sum(v>0 for v in unit.values()); passed=overall>=0.01 and all(v>0 for v in surface.values()) and positive>=6
    report={"phase":1838,"campaign":"C304","status":"nested_composition_adjudicated","actual_test_examples":examples,"overall_relative_gain":overall,"surface_mean_gain":surface,"unit_mean_gain":unit,"positive_units":positive,"field_composition_gate_passed":passed,"operator_order_status":"no_test_static_factorial_grid","strict_interpretation":protocol["claim_boundary"],"next_authorization":"C305_causal_qualification_and_C307_cross_model_regardless"}; core.save(OUT/"analysis/summary.json",report)
    ach={"examples":len(examples)==4,"finite":bool(np.isfinite([overall,*surface.values(),*unit.values()]).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1838,"campaign":"C304","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
