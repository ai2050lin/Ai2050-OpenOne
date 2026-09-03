#!/usr/bin/env python3
"""Phase1519: audit whether C088 prompts define the response-code semantics."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1512_c088_cross_root_semantic_code_factorial_contract"
CLOSURE = RESULT / "phase1518_c088_major_stage_closure"
OUT = RESULT / "phase1519_c088_response_code_semantics_audit"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def mapping_defined(prompt, codebook):
    text = " ".join(prompt.lower().split())
    if codebook == "standard":
        patterns = (
            r"same[^.]{0,80}(?:means|=|answer)[^.]{0,30}yes[^.]{0,80}different[^.]{0,80}(?:means|=|answer)[^.]{0,30}no",
            r"yes[^.]{0,80}(?:means|=)[^.]{0,30}same[^.]{0,80}no[^.]{0,80}(?:means|=)[^.]{0,30}different",
        )
    else:
        patterns = (
            r"same[^.]{0,80}(?:means|=|answer)[^.]{0,30}no[^.]{0,80}different[^.]{0,80}(?:means|=|answer)[^.]{0,30}yes",
            r"no[^.]{0,80}(?:means|=)[^.]{0,30}same[^.]{0,80}yes[^.]{0,80}(?:means|=)[^.]{0,30}different",
        )
    return any(re.search(pattern, text) for pattern in patterns)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1519 exists")
    closure = core.load(CLOSURE / "analysis/final.json")
    closure_audit = core.load(CLOSURE / "audit/independent_final_audit.json")
    cases = core.rows(CONTRACT / "material/active_cases.jsonl")
    standard = [row for row in cases if row["codebook"] == "standard"]
    reversed_rows = [row for row in cases if row["codebook"] == "reversed"]
    explicit = [row for row in cases if mapping_defined(row["prompt"], row["codebook"])]
    code_label_present = [row for row in cases if row["codebook"] in row["prompt"].lower()]
    audit = {
        "phase": 1519,
        "campaign": "C088",
        "audit_type": "post-closure zero-model semantic-contract audit",
        "case_count": len(cases),
        "standard_count": len(standard),
        "reversed_count": len(reversed_rows),
        "code_label_present_count": len(code_label_present),
        "explicit_mapping_definition_count": len(explicit),
        "mapping_identifiable_from_each_prompt": len(explicit) == len(cases),
        "finding": (
            "C088 manipulates the lexical labels standard/reversed and assigns gold answers externally, "
            "but the prompts do not define either label's yes/no mapping."
        ),
        "claim_correction": {
            "retain": (
                "a pre-registered, replicated late full-state response associated with the same/different material factor "
                "under balanced code-label perturbations"
            ),
            "withdraw": [
                "behavioral failure to follow an explicitly taught reversed code",
                "identification of a semantic-by-operational-answer-code mechanism",
                "interpretation of the code main effect as code execution rather than lexical/protocol-label response",
            ],
            "k265_revised_title": "cross-root code-label-marginalized late same/different-associated response field",
            "evidence": "E3-HS-descriptive with contract-semantic limitation",
        },
        "c089_requirement": (
            "Any future answer-code manipulation must state the mapping explicitly in every prompt or teach it with "
            "balanced demonstrations; natural-task observation should remain the primary branch."
        ),
        "parent_closure_authorization": closure["authorization"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "closure_audited": closure_audit["all_checks_passed"],
        "counts": len(cases) == 1984 and len(standard) == 992 and len(reversed_rows) == 992,
        "labels_present": len(code_label_present) == 1984,
        "mapping_absent": len(explicit) == 0,
        "scope_corrected": len(audit["claim_correction"]["withdraw"]) == 3,
        "no_model_run": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "analysis/response_code_semantics_audit.json", audit)
    final = {
        "phase": 1519,
        "campaign": "C088",
        "status": "claim_scope_corrected_after_semantic_contract_audit",
        "checks": checks,
        "authorization": "preregister_c089_natural_relation_full_state_observation_atlas",
        "auto_continue": False,
        "reason": "C088 is closed with corrected scope; C089 requires a new frozen natural-task contract",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
