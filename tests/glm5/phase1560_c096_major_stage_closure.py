#!/usr/bin/env python3
"""Phase1560: close C096 with scoped prospective evidence and a targeted next stage."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P1556 = RESULT / "phase1556_c095_joint_synthesis_and_major_stage_closure"
P1557 = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
P1558 = RESULT / "phase1558_c096_unified_behavior_and_all_state_capture"
P1559 = RESULT / "phase1559_c096_fresh_prediction_atlas_and_adjudication"
OUT = RESULT / "phase1560_c096_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1560 exists")
    finals = [core.load(path / "analysis/final.json") for path in (P1557, P1558, P1559)]
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1557, P1558, P1559)]
    c095 = core.load(P1556 / "analysis/c095_major_stage_synthesis.json")
    capture = core.load(P1558 / "analysis/c096_capture_and_behavior_summary.json")
    reveal = core.load(P1559 / "analysis/c096_prediction_adjudication.json")
    checks = {
        "sequence": [row["phase"] for row in finals] == [1557, 1558, 1559],
        "all_audited": all(row["all_checks_passed"] for row in audits),
        "authorization": finals[-1]["authorization"] == "run_phase1560_c096_major_stage_closure",
        "fresh_material": core.load(P1557 / "protocol/preregistration.json")["material"]["lexical_overlap_with_c091"] == 0,
        "numeric_gate": all(capture["checks"].values()),
        "prediction_accounting": reveal["passed_predictions"] == 4 and reveal["total_predictions"] == 5 and not reveal["all_predictions_passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    k269 = {
        "id": "K269",
        "grade": "E3-OBS-prospective-scoped",
        "name": "fresh lexical replication of the common-plus-conditional late-boundary coordinate field",
        "statement": "On 90 previously unused human-validated Chinese relation pairs with zero lexical overlap with C091, four of five frozen predictions repeated: the prequery common field, top-64 energy scale, raw-coordinate sign/restricted-cosine stability, and stronger postquery conditioning. The universal all-condition cross-partition floor failed locally for postquery concrete similarity-class.",
        "positive": {
            "prequery_min_triadic_cosine": reveal["predictions"]["P096_1_prequery_common_field"]["observed"],
            "top64_median_energy": reveal["predictions"]["P096_3_top64_energy"]["observed"],
            "top64_min_sign_and_cosine": reveal["predictions"]["P096_4_coordinate_stability"]["observed"],
            "pre_minus_post_gap": reveal["predictions"]["P096_5_order_conditioning"]["observed"]["gap"],
            "cross_material_median_full_cosine": reveal["cross_material_c091_to_c096"]["median_full_cosine"],
        },
        "negative_boundary": {
            "failed_prediction": "P096_2_cross_partition",
            "observed": reveal["predictions"]["P096_2_cross_partition"]["observed"],
            "threshold": 0.75,
            "localized_object": "postquery x concrete x similarity-class x state32 boundary",
            "cross_material_minimum_full_cosine": reveal["cross_material_c091_to_c096"]["minimum_full_cosine"],
        },
        "scope": "task-scoped Qwen3 observation; whole-part behavior-qualified, similarity/class diagnostic",
        "forbidden_upgrade": ["semantic neuron group", "causal code", "cross-model invariant", "universal language relation field", "new mathematics"],
    }
    closure = {
        "phase": 1560,
        "campaign": "C096",
        "status": "major_stage_complete_with_scoped_prospective_result",
        "major_answer": {
            "supported": "A reusable late-boundary diagonal-match component and a stable signed distributed coordinate scale repeat on fresh lexical material.",
            "refuted": "A single uniform cross-partition cosine floor across every order, concreteness, and family-pair condition.",
            "best_current_object": "common boundary field plus condition-indexed residual, not a fixed relation direction or fixed neuron list",
        },
        "puzzle_updates": {
            "K267": "preserved but remains whole-part-scoped and output-confounded",
            "K268": {**c095["adjudication"]["K268"], "status_after_c096": "prospectively supported in four registered dimensions, not fully confirmed because P096-2 failed"},
            "K269": k269,
        },
        "unified_theory": {
            "name": "conditional output field closure theory",
            "organizing_principle": "reuse-difference-conditioning (RDC)",
            "mechanism_formula": "H = mu + P_pair + Q_query + G_boundary(order,concreteness) + R_pair,query(order,concreteness) + epsilon",
            "coordinate_formula": "v=sum_j v_j e_j; stable signs and response geometry coexist with condition-dependent top-k membership",
            "global_graph": "embedding/word-pair field -> query-conditioned reusable match component plus relation/order residual -> distributed signed late-boundary field -> output competition",
            "math_status": "Not closed. Existing basic algebra and conditional dynamical notation explain the current observations; no new conserved object, composition law, or theorem yet requires new mathematics.",
        },
        "hard_limits": [
            "same Qwen3-4B model and same controlled prompt family",
            "C091 and C096 use disjoint words but the same published source database",
            "similarity and class-inclusion remain M_BEHAVIOR under this interface",
            "truth, answer token, and task termination remain coupled",
            "no coordinate intervention, necessity, sufficiency, or rescue",
            "one preregistered universal cross-partition floor failed",
        ],
        "next_stage": {
            "campaign": "C097",
            "priority": "targeted residual breadth before causal intervention",
            "route_A": "Use remaining unused human-validated pairs to enlarge postquery concrete similarity-class, retaining all full-coordinate matrices and testing whether the 0.738 floor is sampling variation or a stable conditional residual.",
            "route_B": "Acquire a genuinely independent relation-material source; same-database lexical holdout is not external validity.",
            "route_C_after_A_or_B": "Only after the common component survives another independent material axis, preregister raw-coordinate dose/necessity/rescue on the behavior-qualified whole-part route with matched norm and output-margin controls.",
            "forbidden": "Do not launch an unregistered causal patch now: all five C096 predictions did not pass and output identity remains entangled.",
        },
        "checks": checks,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "preregister_C097_targeted_postquery_residual_and_independent_material_stage",
    }
    core.save(OUT / "analysis/c096_major_stage_closure.json", closure)
    core.save(OUT / "protocol/c097_requirements.json", closure["next_stage"])
    core.save(OUT / "analysis/final.json", {"phase": 1560, "campaign": "C096", "status": closure["status"], "authorization": closure["authorization"]})
    print(json.dumps(closure, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
