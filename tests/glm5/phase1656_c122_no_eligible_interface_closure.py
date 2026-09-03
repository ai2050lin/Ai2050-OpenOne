#!/usr/bin/env python3
"""Close C122 and the C120-C122 comparison qualification stage."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; OUT=TESTS/"result/phase1653_c122_multi_interface_comparison_calibration"; C120=TESTS/"result/phase1647_c120_controlled_comparison_observation_campaign"; C121=TESTS/"result/phase1650_c121_structured_comparison_qualification"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
if __name__=="__main__":
    freeze=core.load(OUT/"protocol/frozen_interface_selection.json"); audit=core.load(OUT/"audit/independent_selection_audit.json")
    if not audit["all_checks_passed"] or freeze["winner"] is not None: raise RuntimeError("C122 no-winner closure mismatch")
    ranked=sorted(freeze["table"],key=lambda row:(-row["minimum_slice"],-row["overall"]))
    closure={
        "phase":1656,"campaign":"C122","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"no_discovery_interface_eligible_comparison_stage_closed",
        "headline":{"best_interface":ranked[0],"all_interfaces":[{"interface":row["interface"],"overall":row["overall"],"minimum_slice":row["minimum_slice"],"eligible":row["eligible"]} for row in freeze["table"]]},
        "strict_conclusion":"Among six preregistered candidate-output interfaces, none qualified on discovery. The strongest true/false interface reached 0.848958 overall but missed the frozen per-dimension gate. Confirmation and lockbox were not revealed for selection or adjudication, and no HiddenState mechanism test was authorized.",
        "large_stage_chain":{
            "C120":core.load(C120/"analysis/closure.json")["strict_conclusion"],
            "C121":core.load(C121/"analysis/closure.json")["strict_conclusion"],
            "C122":"multi-interface discovery selection found no eligible executor",
        },
        "new_puzzles":{"K314-BOUNDARY":"six balanced candidate-output interfaces all fail the frozen discovery eligibility rule; true/false is the strongest but remains dimension-conditioned"},
        "theory_update":"No comparison response field is added to RDC. The large stage identifies an output-qualified-object bottleneck before internal observation, not an absence of comparison encoding.",
        "problems":["one Qwen3","controlled integer tables","candidate-logit scoring","no free-generation behavior audit","discovery interfaces share related wording","no human naturalness","confirmation and lockbox intentionally remain sealed","no HiddenState analysis or heatmap"],
        "claim_boundary":"behavioral interface calibration only; no claim about embedding/HiddenState fields, coordinates, weights, attention/MLP, semantic comparators, cross-model invariance, topology or new mathematics",
        "next_authorization":"end_C120_C122_comparison_stage; next campaign must change the research object or independently preregister a free-generation behavior instrument rather than tune another candidate interface",
    }
    core.save(OUT/"analysis/closure.json",closure)
    checks={"selection_audit":audit["all_checks_passed"],"no_winner":freeze["winner"] is None,"six":len(freeze["table"])==6,"all_failed":not any(row["eligible"] for row in freeze["table"]),"no_holdout":not (OUT/"analysis/holdout_validation.json").exists(),"no_hidden":not (OUT/"raw/qwen3_role_subtoken_all_states.uint16.npy").exists(),"authorization":closure["next_authorization"].startswith("end_C120_C122")}
    report={"phase":1656,"campaign":"C122","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_checks_passed":all(checks.values()),"authorization":closure["next_authorization"]}
    if not report["all_checks_passed"]: raise RuntimeError(report)
    core.save(OUT/"audit/internal_closure_audit.json",report); print(json.dumps({"closure":closure,"audit":report},indent=2))
