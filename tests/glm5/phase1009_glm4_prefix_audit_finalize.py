#!/usr/bin/env python3
"""Finalize GLM4 prefix audit on a common behavior-qualified denominator."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1009_crossfamily_heldout_causal_replication import summarize
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    OUT_ROOT,
    PHASE,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODEL = "glm4"
SURFACES = ("answer", "result", "choice")
OPERATIONS = ("F", "Q")


def main() -> None:
    root = OUT_ROOT / "causal_replication" / "glm4_prefix_audit"
    behavior = read_jsonl(root / "behavior_rows.jsonl")
    causal = read_jsonl(root / "causal_rows.jsonl")
    common_rows = []
    common_cells = []
    common_units_by_cell = {}
    for family in FAMILIES:
        for operation in OPERATIONS:
            hits: dict[
                str,
                dict[str, dict[str, bool]],
            ] = {
                surface: defaultdict(dict) for surface in SURFACES
            }
            for row in behavior:
                if (
                    row["family"] == family
                    and row["operation"] == operation
                ):
                    hits[row["surface"]][row["unit_id"]][row["state"]] = (
                        bool(row["candidate_panel_hit"])
                    )
            all_units = set.intersection(*[
                set(hits[surface]) for surface in SURFACES
            ])
            common_units = sorted(
                unit_id
                for unit_id in all_units
                if all(
                    hits[surface][unit_id].get("base", False)
                    and hits[surface][unit_id].get(operation, False)
                    for surface in SURFACES
                )
            )
            common_units_by_cell[f"{family}:{operation}"] = common_units
            for surface in SURFACES:
                rows = [
                    row for row in causal
                    if row["output_surface"] == surface
                    and row["family"] == family
                    and row["operation"] == operation
                    and row["unit_id"] in common_units
                ]
                for row in rows:
                    row = dict(row)
                    row["common_surface_qualification"] = True
                    common_rows.append(row)
                if len(rows) < 8:
                    common_cells.append({
                        "surface": surface,
                        "family": family,
                        "operation": operation,
                        "n": len(rows),
                        "status": "underfilled",
                        "localized_directional_contribution": False,
                    })
                    continue
                summary = summarize(MODEL, family, operation, rows)
                summary.update({
                    "surface": surface,
                    "status": "complete",
                    "common_surface_qualification": True,
                    "posthoc_specificity_audit": True,
                })
                common_cells.append(summary)
    original_behavior = read_jsonl(
        OUT_ROOT / "behavior" / MODEL / "rows.jsonl"
    )
    original_hits = {
        (row["unit_id"], row["state"]): bool(row["semantic_gate"])
        for row in original_behavior
        if row["split"] == "confirmation"
    }
    answer_rows = [
        row for row in behavior if row["surface"] == "answer"
    ]
    batch_mismatch = [
        row for row in answer_rows
        if original_hits.get((row["unit_id"], row["state"]))
        != bool(row["candidate_panel_hit"])
    ]
    complete = [row for row in common_cells if row["status"] == "complete"]
    result = {
        "schema_version": "phase1009_output_prefix_common_audit.v1",
        "phase": PHASE,
        "model": MODEL,
        "surfaces": list(SURFACES),
        "common_units_per_family_operation": {
            key: len(value)
            for key, value in common_units_by_cell.items()
        },
        "cell_count": len(common_cells),
        "complete_cell_count": len(complete),
        "positive_cell_count": int(sum(
            row["localized_directional_contribution"]
            for row in complete
        )),
        "all_complete_cells_positive": bool(
            complete
            and all(
                row["localized_directional_contribution"]
                for row in complete
            )
        ),
        "original_vs_rebatched_answer_state_count": len(answer_rows),
        "original_vs_rebatched_answer_hit_mismatch_count": len(
            batch_mismatch
        ),
        "original_vs_rebatched_answer_hit_mismatch_rate": (
            len(batch_mismatch) / max(len(answer_rows), 1)
        ),
        "cell_summaries": common_cells,
        "interpretation": (
            "The common denominator removes behavior-selection differences "
            "between prefixes. Any remaining effect is insensitive to the "
            "literal Answer/Result/Choice prefix among behavior-valid cases. "
            "All outputs still use person names, and the comparison with the "
            "initial run also exposes batch/quantization sensitivity."
        ),
        "posthoc_specificity_audit": True,
    }
    write_jsonl(root / "common_causal_rows.jsonl", common_rows)
    write_jsonl(root / "common_cell_summaries.jsonl", common_cells)
    write_json(root / "common_denominator_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
