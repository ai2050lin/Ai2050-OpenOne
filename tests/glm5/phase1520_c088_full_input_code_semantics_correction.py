#!/usr/bin/env python3
"""Phase1520: correct C088 semantics audit by including the actual system message."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1512_c088_cross_root_semantic_code_factorial_contract"
PRIOR_AUDIT = RESULT / "phase1519_c088_response_code_semantics_audit"
OUT = RESULT / "phase1520_c088_full_input_code_semantics_correction"
sys.path.insert(0, str(TESTS))

import phase1512_c088_cross_root_semantic_code_factorial_contract as c088
import phase1331_relational_measurement_core as core
from phase1373_c058_dose_distance_group_campaign_contract import tokenizer


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1520 exists")
    prior = core.load(PRIOR_AUDIT / "analysis/response_code_semantics_audit.json")
    prior_independent = core.load(PRIOR_AUDIT / "audit/independent_final_audit.json")
    cases = core.rows(CONTRACT / "material/active_cases.jsonl")
    compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    tok = tokenizer()
    system = " ".join(c088.SYSTEM.lower().split())
    required_clauses = {
        "standard_same_yes": "standard code means same -> yes" in system,
        "standard_different_no": "different -> no" in system,
        "reversed_same_no": "reversed code means same -> no" in system,
        "reversed_different_yes": "different -> yes" in system,
    }
    case_lookup = {row["case_id"]: row for row in cases}
    exact_recompile = 0
    full_input_mapping = 0
    decoded_sample = None
    for row in compiled:
        case = case_lookup[row["case_id"]]
        rebuilt = core.chat_ids(tok, c088.SYSTEM, case["prompt"])
        exact_recompile += rebuilt == row["prompt_ids"]
        decoded = tok.decode(row["prompt_ids"], skip_special_tokens=False).lower()
        mapping_present = (
            "standard code means same -> yes and different -> no" in decoded
            and "reversed code means same -> no and different -> yes" in decoded
        )
        full_input_mapping += mapping_present
        if decoded_sample is None:
            decoded_sample = decoded[:1200]
    correction = {
        "phase": 1520,
        "campaign": "C088",
        "audit_type": "full-model-input semantic-contract correction",
        "case_count": len(cases),
        "compiled_count": len(compiled),
        "system_mapping_clauses": required_clauses,
        "exact_chat_recompile_count": exact_recompile,
        "full_input_mapping_definition_count": full_input_mapping,
        "user_prompt_only_mapping_definition_count": prior["explicit_mapping_definition_count"],
        "finding": (
            "Phase1519 audited only the user prompt field. The actual model input prepended a system message "
            "that explicitly and uniquely defined both response-code mappings in every case."
        ),
        "claim_restoration": {
            "restore": [
                "C088 validly manipulates a defined standard versus reversed answer-code contract",
                "different-reversed accuracy 0 is evidence of failure under an explicitly supplied mapping",
                "the factorial semantic-by-code effects are identified at the task-input level",
            ],
            "retain_boundaries": [
                "the code main effect may mix lexical label, instruction processing, and rule execution",
                "the late semantic main effect is not a pure semantic vector or localized circuit",
                "K265 remains descriptive Hidden-State evidence rather than causal closure",
            ],
            "k265_title": "cross-root code-marginalized late semantic-match-associated full-state response field",
            "evidence": "E3-HS-descriptive",
        },
        "phase1519_status": "superseded_due_to_incomplete_input_scope",
        "prior_audit_passed_its_own_incomplete_checklist": prior_independent["all_checks_passed"],
        "decoded_full_input_sample": decoded_sample,
        "c089_requirement": "audit the complete model input, including system and chat template, before interpreting any task variable",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "counts": len(cases) == len(compiled) == 1984,
        "system_complete": all(required_clauses.values()),
        "exact_recompile": exact_recompile == 1984,
        "mapping_in_every_full_input": full_input_mapping == 1984,
        "prior_scope_was_partial": prior["explicit_mapping_definition_count"] == 0,
        "claim_restored_with_boundaries": len(correction["claim_restoration"]["restore"]) == 3 and len(correction["claim_restoration"]["retain_boundaries"]) == 3,
        "no_model_run": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "analysis/full_input_code_semantics_correction.json", correction)
    final = {
        "phase": 1520,
        "campaign": "C088",
        "status": "phase1519_superseded_full_input_semantics_restored",
        "checks": checks,
        "authorization": "preregister_c089_natural_relation_full_state_observation_atlas",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(correction, indent=2))


if __name__ == "__main__":
    main()
