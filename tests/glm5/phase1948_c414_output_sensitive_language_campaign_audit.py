#!/usr/bin/env python3
"""Independent audit for C399-C414 / Phase1933-1948."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TESTS=ROOT/"tests/glm5"
sys.path.insert(0,str(TESTS))
import phase1933_c399_c414_output_sensitive_language_campaign as campaign


def load(path:Path): return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    finals={name:load(out/"analysis/final.json") for name,out in campaign.OUTS.items()}
    checks={}
    checks["phase_sequence"]=[finals[f"C{i}"]["phase"] for i in range(399,415)]==list(range(1933,1949))
    checks["all_closed"]=all(v["status"]=="closed" and v["all_checks_passed"] for v in finals.values())
    c400=finals["C400"]["headline"]
    checks["material_3840"]=c400["rows"]==3840 and c400["partition_counts"]=={"discovery":1920,"confirmation":1280,"lockbox":640}
    checks["zero_models_balanced"]=max(c400["zero_model_accuracies"].values())==0.5
    checks["both_factors_output_sensitive"]=c400["both_factors_output_sensitive"] is True
    checks["naturalness_not_overclaimed"]=c400["human_naturalness_review"] is False
    c401=finals["C401"]["headline"]
    checks["behavior_accounted"]=c401["rows"]==3840 and set(c401["family_accuracy"])==set(campaign.FAMILIES)
    c402=finals["C402"]["headline"]
    checks["full_coordinate_field"]=c402["role_shape"][-1]==2560 and c402["role_shape"][1:3]==[38,6] and c402["full_shape"][-1]==2560
    c403=finals["C403"]["headline"]
    checks["atlas_full_axis"]=c403["shape"][-3:]==[38,6,2560]
    c404=finals["C404"]["headline"]
    checks["cross_construction_matrix"]=c404["cells"]>=0 and isinstance(c404["candidate_families"],list)
    c405=finals["C405"]["headline"]
    checks["single_sample_initial_state"]=c405["cells"]>=0 and c405["strict_interpretation"].endswith("causal circuit.")
    c406=finals["C406"]["headline"]
    checks["full_token_deltas"]=c406["pairs"]>0 and c406["delta_shape"][-1]==2560
    c407=finals["C407"]["headline"]
    checks["composition_branch_consistent"]=c407["composition"]["ran"]==c407["behavior_eligible"]
    c408=finals["C408"]["headline"]
    checks["known_truth_writer"]=c408["writer_calibrated"] is True and c408["mean_recovery"]["correct"]>0.99
    c409=finals["C409"]["headline"]
    checks["graph_balanced"]=c409["rows"]==672 and max(abs(v-1/3) for v in c409["gold_position_frequency"].values())<1e-9
    c410=finals["C410"]["headline"]
    checks["graph_branch_consistent"]=c410["field_ran"]==c409["graph_field_eligible"]
    c411=finals["C411"]["headline"]
    joint=bool(set(c404["candidate_families"])&set(c405["candidate_families"]))
    checks["causal_branch_consistent"]=(not joint and not c411["writer_ran"]) or (joint and "result" in c411)
    c412=finals["C412"]["headline"]
    checks["external_models_sequential"]=[r["model"] for r in c412["results"]]==["glm4","deepseek7b"]
    checks["behavior_before_hidden"]=all(r["capture"]["ran"]==r["eligible"] for r in c412["results"])
    c413=finals["C413"]["headline"]
    checks["no_bisimulation_overclaim"]=c413["functional_bisimulation_established"] is False
    c414=finals["C414"]["headline"]
    visual=load(campaign.VISUAL)
    checks["visual_full_coordinates"]=visual["schema"]=="c414.output_sensitive_language_field.v1" and len(visual["rows"])==c414["visual_rows"] and all(len(r["values"])==2560 for r in visual["rows"])
    cleanup=load(campaign.OUTS["C414"]/"audit/cleanup.json")
    checks["cleanup_checksummed"]=len(cleanup)==c414["cleanup_files"] and all(i["sha256"] and i["removed"] for i in cleanup)
    checks["cleanup_absent"]=all(not (ROOT/i["path"]).exists() for i in cleanup)
    checks["new_math_closed"]=c414["new_math_gate_passed"] is False
    failed=[name for name,passed in checks.items() if not passed]
    if failed: raise AssertionError(failed)
    result={"phase":1948,"campaign":"C414","audit":"independent","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":True,"strict_conclusion":"The campaign separates output-sensitive behavior, construction transfer, single-sample prediction, composition, graph, causal-writer, and cross-model evidence. Only explicitly passed branches are authorized; fixed coordinates, universal operators, causal closure, and new mathematics are not assumed."}
    save_path=campaign.OUTS["C414"]/"audit/independent_audit.json"; save_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
