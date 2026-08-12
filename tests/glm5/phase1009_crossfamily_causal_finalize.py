#!/usr/bin/env python3
"""Aggregate independent Phase1009 cross-family causal replications."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    OUT_ROOT,
    PHASE,
    read_json,
    write_json,
)


MODELS = ("qwen3", "glm4")


def main() -> None:
    model_summaries = {
        model: read_json(
            OUT_ROOT / "causal_replication" / model / "summary.json"
        )
        for model in MODELS
    }
    cells = [
        cell
        for summary in model_summaries.values()
        for cell in summary["cell_summaries"]
    ]
    positive = [
        cell
        for cell in cells
        if cell["localized_directional_contribution"]
    ]
    family_model_support = {
        family: sorted({
            cell["model"]
            for cell in positive
            if cell["family"] == family
        })
        for family in FAMILIES
    }
    operation_model_support = {
        operation: sorted({
            cell["model"]
            for cell in positive
            if cell["operation"] == operation
        })
        for operation in ("F", "Q")
    }
    result = {
        "schema_version": "phase1009_crossfamily_causal_final.v1",
        "phase": PHASE,
        "source_selection": (
            "Phase1008 binding-task discovery only; no Phase1009 response "
            "or confirmation data selected heads"
        ),
        "evaluation": (
            "Phase1009 confirmation names and templates across comparison, "
            "negation, and semantic-role families"
        ),
        "cell_count": len(cells),
        "positive_cell_count": len(positive),
        "positive_cell_ids": [
            f"{cell['model']}:{cell['family']}:{cell['operation']}"
            for cell in positive
        ],
        "family_model_support": family_model_support,
        "operation_model_support": operation_model_support,
        "models_with_cross_family_local_replication": [
            model
            for model, summary in model_summaries.items()
            if summary["cross_family_local_replication"]
        ],
        "all_no_op_audits_pass": all(
            summary["no_op_audit_pass"]
            for summary in model_summaries.values()
        ),
        "strongest_supported_statement": (
            "A positive cell shows that a head group selected on the "
            "Phase1008 binding task makes a local directional contribution "
            "to a different held-out language family. Even multi-family "
            "replication does not establish a shared transport path, a "
            "necessary or sufficient mechanism, natural rollout closure, "
            "or a language formula."
        ),
        "model_summaries": model_summaries,
    }
    write_json(OUT_ROOT / "causal_replication" / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
