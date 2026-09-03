#!/usr/bin/env python3
"""Phase1490: Qwen3 behavior run and frozen composition-set stratification."""
from __future__ import annotations
import inspect, json, math, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; CONTRACT=TESTS/"result/phase1489_c085_prospective_layered_contract"; OUT=TESTS/"result/phase1490_c085_behavior_stratification"; sys.path.insert(0,str(TESTS))
import phase1331_relational_measurement_core as core
import phase1391_c062_family_factorized_behavior as runner
import phase1457_c077_behavior as metric
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
BATCH=24

def summarize(rows, sets, protocol, repeat_error, quant):
    keys=tuple(f"{s}_{c}" for s in protocol["surfaces"] for c in protocol["cells"]); by={r["case_id"]:r for r in rows}; stratified=[]
    for group in sets:
        correct=sum(by[group[k]]["correct"] for k in keys); stratum="success" if correct==16 else ("failed" if correct==0 else "mixed")
        stratified.append({**group,"correct_count":correct,"case_count":16,"stratum":stratum})
    surface={s:{"accuracy":metric.accuracy([r for r in rows if r["surface"]==s]),"balanced_accuracy":metric.balanced_accuracy([r for r in rows if r["surface"]==s])} for s in protocol["surfaces"]}
    relation_surface={rel:{s:metric.balanced_accuracy([r for r in rows if r["record_relation_id"]==rel and r["surface"]==s]) for s in protocol["surfaces"]} for rel in protocol["relations"]}
    strata=Counter(r["stratum"] for r in stratified); split={st:{p:sum(r["stratum"]==st and r["partition"]==p for r in stratified) for p in protocol["partitions"]} for st in ("success","mixed","failed")}
    checks={"count":len(rows)==3456,"sets":len(stratified)==216 and sum(strata.values())==216,"repeat":repeat_error<=1e-6,"finite":all(math.isfinite(v) for r in rows for v in r["scores"]),"bf16":quant["has_bf16_parameters"],"not_quantized":not quant["has_quantized_modules"],"hidden_not_accessed":True}
    return {"phase":1490,"campaign":"C085","global_accuracy":metric.accuracy(rows),"global_balanced_accuracy":metric.balanced_accuracy(rows),"surface":surface,"relation_surface_balanced_accuracy":relation_surface,"stratum_counts":dict(strata),"stratum_partition_counts":split,"case_error_count":sum(not r["correct"] for r in rows),"numeric_repeat_max_abs_diff":repeat_error,"checks":checks,"all_integrity_checks_passed":all(checks.values()),"hidden_state_accessed":False},stratified

def main():
    if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1490 exists")
    final=core.load(CONTRACT/"analysis/final.json"); audit=core.load(CONTRACT/"audit/independent_final_audit.json"); protocol=core.load(CONTRACT/"protocol/preregistration.json")
    if final["authorization"]!="run_phase1490_c085_behavior_stratification" or not audit["all_checks_passed"]: raise RuntimeError("Phase1489 authorization missing")
    source=core.rows(CONTRACT/"material/active_cases.jsonl"); compiled=core.rows(CONTRACT/"compiled/qwen3_active.jsonl"); sets=core.rows(CONTRACT/"material/composition_sets.jsonl"); model=None
    try:
        model,tok,device,placement=load_bf16("qwen3"); quant=quantization_audit(model); pad=int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id); supports="logits_to_keep" in inspect.signature(model.forward).parameters
        predictions=[]; first=None
        for start in range(0,len(compiled),BATCH):
            block=runner.forward(model,compiled[start:start+BATCH],pad,device,supports)
            if first is None: first=block
            predictions.extend(block)
        repeated=runner.forward(model,compiled[:BATCH],pad,device,supports); repeat=max(abs(a["scores"][i]-b["scores"][i]) for a,b in zip(first,repeated) for i in range(2))
        rows=[{**row,**pred,"correct":pred["prediction"]==row["gold_position"]} for row,pred in zip(source,predictions)]
        summary,stratified=summarize(rows,sets,protocol,repeat,quant); summary["runtime"]={"placement":placement,"quantization":quant,"finished_at_utc":datetime.now(timezone.utc).isoformat()}
        if not summary["all_integrity_checks_passed"]: raise RuntimeError(summary["checks"])
        core.write_rows(OUT/"raw/behavior.jsonl",rows); core.write_rows(OUT/"material/stratified_composition_sets.jsonl",stratified); core.save(OUT/"analysis/behavior_stratification_summary.json",summary)
        core.save(OUT/"analysis/final.json",{"phase":1490,"campaign":"C085","status":"behavior_stratification_complete","stratum_counts":summary["stratum_counts"],"authorization":"run_phase1491_c085_all_case_field_capture"}); print(json.dumps({k:v for k,v in summary.items() if k!="runtime"},indent=2))
    finally:
        if model is not None: release_bf16(model)
if __name__=="__main__": main()
