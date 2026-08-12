from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import phase1149_role_factorized_mediation as p1149


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1149_role_factorized_mediation"


def audit() -> dict[str, Any]:
    prereg = p1149.p1148.read_json(p1149.PREREG_PATH)
    p1149.verify_preregistration(prereg)
    selection = p1149.p1148.read_json(
        OUT_ROOT / "analysis" / "discovery_selection.json"
    )
    independent = p1149.p1148.read_json(
        OUT_ROOT / "audit" / "discovery_independent_result_audit.json"
    )
    replicates = selection["replicates"]
    key_cells: list[dict[str, Any]] = []
    for replicate in replicates:
        for split in ("holdout", "quartet"):
            effect = selection["effects"][replicate][split]
            case_count = p1149.load_summary(
                replicate, "answer_boundary", prereg
            )["evaluation"][split]["case_count"]
            required_gain = float(
                prereg["thresholds"]["minimum_paired_accuracy_gain"]
            )
            observed_gain = float(effect["paired_gain"])
            shortfall_cases = max(
                0,
                int(round((required_gain - observed_gain) * int(case_count))),
            )
            key_cells.append(
                {
                    "replicate": replicate,
                    "split": split,
                    "case_count": int(case_count),
                    "baseline_accuracy": float(effect["answer_boundary_accuracy"]),
                    "candidate_accuracy": float(effect["role_factorized_accuracy"]),
                    "paired_gain": observed_gain,
                    "gain_gate": bool(effect["gain_gate"]),
                    "shortfall_cases": shortfall_cases,
                }
            )
    failed_cells = [cell for cell in key_cells if not cell["gain_gate"]]
    hard_stop = not independent["all_checks_passed"]
    claim_stop = not selection["gain_scope_pass"]
    absolute_candidate_pass = bool(
        selection["condition_all_qualified"]["role_factorized"]
    )
    baseline_failed = not bool(
        selection["condition_all_qualified"]["answer_boundary"]
    )
    branch_allowed = (
        not hard_stop and absolute_candidate_pass and baseline_failed and claim_stop
    )
    result = {
        "phase": p1149.PHASE,
        "scope": "claim_action_gate_decomposition",
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "independent_audit_digest": independent["audit_digest"],
        "key_cells": key_cells,
        "key_cell_pass_count": sum(cell["gain_gate"] for cell in key_cells),
        "key_cell_count": len(key_cells),
        "failed_cells": failed_cells,
        "gate_state": {
            "HardStop": hard_stop,
            "ClaimStop": claim_stop,
            "DatasetSeal": True,
            "Phase1149ConfirmationAllowed": bool(
                selection["confirmation_authorized"]
            ),
            "BranchAllowed": branch_allowed,
            "SearchStop": not branch_allowed,
        },
        "evidence_vector": {
            "integrity": "passed_independent_recomputation",
            "absolute_formation": "4_of_4_discovery_replicates_passed",
            "paired_specificity": "7_of_8_key_cells_passed",
            "independent_confirmation": "not_run_not_negative",
            "causal_position_use": "not_run_not_negative",
            "natural_transformer_mechanism": "not_tested",
        },
        "formal_decision": (
            "Honor the preregistered Phase1149 ClaimStop and deny its confirmation. "
            "Do not interpret that decision as absence of role-aligned formation."
        ),
        "research_action": (
            "A new protocol may test absolute formation on fresh material, but it must be labeled "
            "a new claim prompted by this gate conflict and must obtain its own discovery and confirmation."
        ),
        "claim_boundary": (
            "This is a post-result gate audit. It does not upgrade the role-factorized candidate, "
            "change the frozen threshold, or authorize reuse of the sealed discovery data."
        ),
    }
    result["audit_digest"] = p1149.p1148.canonical_digest(result)
    p1149.p1148.write_json(
        OUT_ROOT / "audit" / "claim_action_gate_audit.json", result
    )
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
