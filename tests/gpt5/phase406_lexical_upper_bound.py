#!/usr/bin/env python3
"""Compute a deliberately optimistic lexical upper bound for Phase406.

This is not a replacement gate.  Any exact target occurrence anywhere in the
12-token response is credited, even if it may be incidental.  It tests whether
the formal stop decision depends on the conservative semantic parser.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase406_conditioned_sequence_analysis import (  # noqa: E402
    GROUP_REQUIRED,
    group_audit,
)
from phase406_conditioned_sequence_protocol import (  # noqa: E402
    FAMILIES,
    MODELS,
    OUT,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_target_present(row: dict) -> bool:
    target = row["target_semantic_label_private"]
    aliases = row.get("semantic_aliases_private", {}).get(target, [target])
    present = any(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            row["generated_text_clean_private"],
            re.IGNORECASE,
        )
        for alias in aliases
    )
    return bool(present) and row["all_generated_step_logits_valid"]


def optimistic(row: dict) -> dict:
    result = dict(row)
    if exact_target_present(row):
        result["semantic_label_private"] = row["target_semantic_label_private"]
        result["short_sequence_semantic_correct"] = True
        result["semantic_answer_complete"] = True
    return result


def main() -> None:
    model_family_rows = []
    all_formal_rows = []
    all_optimistic_rows = []
    for model in MODELS:
        source = OUT / "analysis/discovery/private" / model / "semantic_rows.jsonl"
        formal_rows = read_jsonl(source)
        rows = [optimistic(row) for row in formal_rows]
        all_formal_rows.extend(formal_rows)
        all_optimistic_rows.extend(rows)
        for family in FAMILIES:
            selected = [row for row in rows if row["family_id"] == family]
            group_ids = sorted(
                {row["anonymous_parallel_group_id"] for row in selected}
            )
            audits = [
                group_audit(
                    [
                        row
                        for row in selected
                        if row["anonymous_parallel_group_id"] == group_id
                    ],
                    family,
                )
                for group_id in group_ids
            ]
            pass_count = sum(audit["group_pass"] for audit in audits)
            model_family_rows.append(
                {
                    "model": model,
                    "family_id": family,
                    "case_count": len(selected),
                    "optimistic_exact_target_present_count": sum(
                        exact_target_present(row) for row in selected
                    ),
                    "optimistic_sequence_correct_count": sum(
                        row["short_sequence_semantic_correct"] for row in selected
                    ),
                    "group_pass_count": pass_count,
                    "required_group_pass_count": GROUP_REQUIRED["discovery"],
                    "model_family_upper_bound_pass": pass_count
                    >= GROUP_REQUIRED["discovery"],
                }
            )

    crossmodel = []
    for family in FAMILIES:
        selected = [row for row in model_family_rows if row["family_id"] == family]
        if len(selected) == len(MODELS) and all(
            row["model_family_upper_bound_pass"] for row in selected
        ):
            crossmodel.append(family)

    payload = {
        "schema_version": "80.4.0",
        "phase_id": "Phase406-LexicalUpperBoundAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": "credit_any_exact_target_occurrence_anywhere_in_12_tokens",
        "is_formal_gate": False,
        "is_deliberately_optimistic": True,
        "case_count": len(all_formal_rows),
        "formal_sequence_correct_count": sum(
            row["short_sequence_semantic_correct"] for row in all_formal_rows
        ),
        "exact_target_or_alias_present_count": sum(
            exact_target_present(row) for row in all_formal_rows
        ),
        "optimistic_sequence_correct_count": sum(
            row["short_sequence_semantic_correct"] for row in all_optimistic_rows
        ),
        "newly_credited_case_count": sum(
            optimistic_row["short_sequence_semantic_correct"]
            and not formal_row["short_sequence_semantic_correct"]
            for formal_row, optimistic_row in zip(
                all_formal_rows, all_optimistic_rows
            )
        ),
        "model_family_rows": model_family_rows,
        "crossmodel_upper_bound_candidate_families": crossmodel,
        "claim_boundary": {
            "exact_target_occurrence_is_semantic_correctness": False,
            "upper_bound_can_promote_candidate": False,
            "upper_bound_can_only_test_stop_robustness": True,
        },
    }
    path = OUT / "phase406_lexical_upper_bound_audit.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
