#!/usr/bin/env python3
"""Finalize Phase1033 and compare it with the Phase1032 discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1032_finalize as base
import phase1033_alliance_replication_protocol as protocol


def composition_models(
    aggregate: dict[str, Any],
    readout: str,
    condition: str,
) -> set[str]:
    return set(
        aggregate["cross_model_evidence"][readout][
            "composition_restoration"
        ][condition]["models"]
    )


def main() -> None:
    base.protocol = protocol
    base.main()

    current_path = protocol.OUT_ROOT / "aggregate.json"
    discovery_path = (
        ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1032_span_aware_alliance"
        / "aggregate.json"
    )
    current = protocol.read_json(current_path)
    discovery = protocol.read_json(discovery_path)

    composition = {}
    for readout in ("within_template", "next_token"):
        composition[readout] = {}
        for condition in (
            "query_nonce_q",
            "query_clause_q",
            "query_nonce_bq",
            "query_clause_bq",
        ):
            first = composition_models(
                discovery, readout, condition
            )
            second = composition_models(
                current, readout, condition
            )
            repeated = sorted(first & second)
            composition[readout][condition] = {
                "phase1032_models": sorted(first),
                "phase1033_models": sorted(second),
                "same_models_in_both_phases": repeated,
                "independently_repeated_at_least_two_models": (
                    len(repeated) >= 2
                ),
            }

    selected = {}
    for readout in ("within_template", "next_token"):
        first = set(
            discovery["cross_model_evidence"][readout][
                "selected_source"
            ]["models"]
        )
        second = set(
            current["cross_model_evidence"][readout][
                "selected_source"
            ]["models"]
        )
        repeated = sorted(first & second)
        selected[readout] = {
            "phase1032_models": sorted(first),
            "phase1033_models": sorted(second),
            "same_models_in_both_phases": repeated,
            "independently_repeated_at_least_two_models": (
                len(repeated) >= 2
            ),
        }

    alliance_repeated = any(
        row["independently_repeated_at_least_two_models"]
        for row in composition["within_template"].values()
    )
    selected_repeated = selected["within_template"][
        "independently_repeated_at_least_two_models"
    ]
    if alliance_repeated:
        route = (
            "The local source/query state alliance independently repeated. "
            "Do not declare output closure. Next map the post-query "
            "integration suffix across depth, then decompose the repeated "
            "alliance into attention, MLP, and residual contributions."
        )
    elif selected_repeated:
        route = (
            "Conditional source selection independently repeated but the "
            "alliance did not. Downgrade Phase1032 composition and map the "
            "post-query integration suffix before further coalition tests."
        )
    else:
        route = (
            "Neither candidate independently repeated. Preserve the raw "
            "atlas, abandon this frozen alliance, and return to global "
            "component-resolved response mapping."
        )

    current["replication_assessment"] = {
        "source_discovery_phase": 1032,
        "replication_phase": 1033,
        "selected_source": selected,
        "composition": composition,
        "selected_source_independently_repeated": selected_repeated,
        "alliance_independently_repeated": alliance_repeated,
        "automatic_next_route": route,
        "claim_limit": (
            "Independent repetition establishes only a local sufficient "
            "state coalition in the artificial two-binding task. It does "
            "not establish a natural component path or next-token closure."
        ),
    }
    protocol.write_json(current_path, current)
    manifest = base.artifact_manifest()
    protocol.write_json(
        protocol.OUT_ROOT / "final_audit.json",
        {
            "schema_version": "phase1033_final_audit.v1",
            "phase": protocol.PHASE,
            "checks": current["checks"],
            "all_checks_passed": current["all_checks_passed"],
            "replication_assessment": current[
                "replication_assessment"
            ],
            "artifact_manifest": {
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
            },
        },
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "replication_assessment": current[
            "replication_assessment"
        ],
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
