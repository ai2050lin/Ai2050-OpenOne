#!/usr/bin/env python3
"""Independent audits for C122."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; OUT = TESTS / "result/phase1653_c122_multi_interface_comparison_calibration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1653_c122_multi_interface_common as c122

def save(name, phase, checks, authorization):
    report = {"phase": phase, "campaign": "C122", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "authorization": authorization}
    if not report["all_checks_passed"]: raise RuntimeError(report)
    core.save(OUT / f"audit/{name}.json", report); print(json.dumps(report, indent=2))

def contract():
    p=core.load(OUT/"protocol/preregistration.json"); i=core.load(OUT/"audit/internal_contract_audit.json"); units=core.rows(OUT/"material/units.jsonl"); cases=core.rows(OUT/"material/cases.jsonl"); compiled=core.rows(OUT/"compiled/qwen3.jsonl"); cells=Counter((r["partition"],r["dimension"],r["truth_factor"],r["gap_factor"],r["surface_factor"],r["interface"]) for r in cases)
    checks={"internal":i["all_checks_passed"],"producer":p["producer_sha256"]==core.sha(TESTS/"phase1653_c122_multi_interface_common.py"),"digest":p["material_digest"]==core.digest([*units,*cases]),"counts":(len(units),len(cases),len(compiled))==(24,3456,3456),"factorial":len(cells)==432 and all(v==8 for v in cells.values()),"interfaces":tuple(p["interfaces"])==c122.INTERFACES,"balance":all(sum(r["truth_factor"] for r in cases if r["partition"]==part and r["interface"]==interface)==0 for part in c122.PARTITIONS for interface in c122.INTERFACES),"candidates":all(len(c)==1 for r in compiled for c in r["candidate_ids"]),"behavior_first":"no HiddenState archive" in p["behavior_first"],"authorization":p["authorization"]=="execute_phase1654_c122_all_interface_behavior_capture"}
    save("independent_contract_audit",1653,checks,p["authorization"])

def capture():
    r=core.load(OUT/"analysis/capture_summary.json"); logits=np.load(OUT/"raw/qwen3_all_interface_logits.float32.npy",mmap_mode="r"); rows=core.rows(OUT/"raw/qwen3_all_interface_behavior.jsonl")
    checks={"contract":core.load(OUT/"audit/independent_contract_audit.json")["all_checks_passed"],"logits":list(logits.shape)==[3456,2] and bool(np.isfinite(logits).all()) and core.sha(OUT/"raw/qwen3_all_interface_logits.float32.npy")==r["logits_sha256"],"index":len(rows)==3456 and core.sha(OUT/"raw/qwen3_all_interface_behavior.jsonl")==r["index_sha256"],"interfaces":Counter(x["interface"] for x in rows)=={name:576 for name in c122.INTERFACES},"repeat":r["repeat_logits_max_abs"]==0,"bf16":r["runtime"]["quantization"]["has_bf16_parameters"] and not r["runtime"]["quantization"]["has_quantized_modules"],"no_hidden":not (OUT/"raw/qwen3_role_subtoken_all_states.uint16.npy").exists(),"authorization":r["authorization"]=="execute_phase1655_c122_discovery_interface_selection"}
    save("independent_capture_audit",1654,checks,r["authorization"])

def selection():
    p=core.load(OUT/"protocol/preregistration.json"); f=core.load(OUT/"protocol/frozen_interface_selection.json"); rows=[r for r in core.rows(OUT/"raw/qwen3_all_interface_behavior.jsonl") if r["partition"]=="discovery"]
    table=[]
    for interface in c122.INTERFACES:
        local=[r for r in rows if r["interface"]==interface]; s={"interface":interface,"n":len(local),"overall":c122.acc(local),"by_dimension":{name:c122.acc([r for r in local if r["dimension"]==name]) for name in c122.DIMENSIONS},"by_truth":{str(v):c122.acc([r for r in local if r["truth_factor"]==v]) for v in (1,-1)},"by_gap":{str(v):c122.acc([r for r in local if r["gap_factor"]==v]) for v in (1,-1)}}; s["minimum_slice"]=min(*s["by_dimension"].values(),*s["by_truth"].values(),*s["by_gap"].values()); s["eligible"]=min(s["by_dimension"].values())>=p["selection_rule"]["eligible_each_dimension_min"] and min(s["by_truth"].values())>=p["selection_rule"]["eligible_each_truth_min"] and min(s["by_gap"].values())>=p["selection_rule"]["eligible_each_gap_min"]; table.append(s)
    eligible=[r for r in table if r["eligible"]]; winner=None if not eligible else sorted(eligible,key=lambda r:(-r["minimum_slice"],-r["overall"],c122.INTERFACES.index(r["interface"])))[0]
    checks={"capture":core.load(OUT/"audit/independent_capture_audit.json")["all_checks_passed"],"discovery_only":len(rows)==1152 and f["read_partition"]=="discovery","table":f["table"]==table,"winner":f["winner"]==winner,"hash":f["source_index_sha256"]==core.sha(OUT/"raw/qwen3_all_interface_behavior.jsonl"),"authorization":f["authorization"]==("close_C122_no_eligible_interface" if winner is None else "execute_phase1656_c122_frozen_winner_holdout_validation")}
    save("independent_selection_audit",1655,checks,f["authorization"])

def validate():
    f=core.load(OUT/"protocol/frozen_interface_selection.json"); r=core.load(OUT/"analysis/holdout_validation.json"); rows=[x for x in core.rows(OUT/"raw/qwen3_all_interface_behavior.jsonl") if x["interface"]==f["winner"]["interface"] and x["partition"]!="discovery"]
    checks={"selection":core.load(OUT/"audit/independent_selection_audit.json")["all_checks_passed"],"winner":r["winner"]==f["winner"]["interface"],"rows":len(rows)==384 and all(s["n"]==192 for s in r["summaries"]),"partitions":{s["partition"] for s in r["summaries"]}=={"confirmation","lockbox"},"passed":r["passed"]==all(r["checks"].values()),"authorization":r["authorization"]==("freeze_C123_all_coordinate_capture_on_C122_winner" if r["passed"] else "close_comparison_interface_campaign")}
    save("independent_holdout_audit",1656,checks,r["authorization"])

STAGES={"contract":contract,"capture":capture,"selection":selection,"validate":validate}
if __name__=="__main__":STAGES[sys.argv[1]]()
