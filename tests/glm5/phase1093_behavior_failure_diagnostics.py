#!/usr/bin/env python3
"""Summarize Phase1093 behavior-gate failures without changing frozen gates."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1093_independent_relation_protocol as protocol


def main() -> None:
    source = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {}
    for model_name, model_row in source["models"].items():
        attributes = {}
        for attribute, attribute_row in model_row["attributes"].items():
            surfaces = {}
            surface_passes = {
                surface: bool(surface_row["passed"])
                for surface, surface_row in attribute_row["surfaces"].items()
            }
            for surface, surface_row in attribute_row["surfaces"].items():
                failed_pairs = {}
                weak_world_counts = Counter()
                all_worlds = set(protocol.BASE_WORLDS)
                for pair_name, pair_row in surface_row["pairs"].items():
                    if pair_row["passed"]:
                        continue
                    missing_by_panel = {}
                    for panel in protocol.PANELS:
                        passing = set(pair_row["passing_worlds_by_panel"][panel])
                        missing = sorted(all_worlds - passing)
                        missing_by_panel[panel] = missing
                        weak_world_counts.update(missing)
                    failed_pairs[pair_name] = {
                        "missing_worlds_by_panel": missing_by_panel,
                    }
                surfaces[surface] = {
                    "passed": bool(surface_row["passed"]),
                    "passing_pair_count": int(surface_row["passing_pair_count"]),
                    "passing_pairs": list(surface_row["passing_pairs"]),
                    "failed_pairs": failed_pairs,
                    "weak_world_counts": dict(sorted(weak_world_counts.items())),
                    "generation_passed": bool(surface_row["generation_passed"]),
                    "interface_specific_failure": (
                        not surface_passes[surface]
                        and any(
                            other_passed
                            for other_surface, other_passed in surface_passes.items()
                            if other_surface != surface
                        )
                    ),
                }
            attributes[attribute] = {
                "passed": bool(attribute_row["passed"]),
                "surfaces": surfaces,
            }
        models[model_name] = {
            "model_authorized": bool(model_row["model_authorized"]),
            "candidate_finite_fraction": float(
                model_row["candidate_finite_fraction"]
            ),
            "passing_attributes": list(model_row["passing_attributes"]),
            "attributes": attributes,
        }

    output = {
        "schema_version": "phase1093_behavior_failure_diagnostics.v1",
        "phase": 1093,
        "protocol_digest": source["protocol_digest"],
        "frozen_behavior_summary_digest": source["summary_digest"],
        "interpretation": (
            "Post hoc localization only. It cannot change behavior authorization or "
            "upgrade any hidden-state result."
        ),
        "models": models,
    }
    output["diagnostic_digest"] = protocol.digest(output)
    destination = (
        protocol.OUT_ROOT / "analysis" / "behavior_failure_diagnostics.json"
    )
    protocol.write_json(destination, output)

    compact = {
        model_name: {
            attribute: {
                surface: {
                    "passed": row["passed"],
                    "passing_pairs": row["passing_pair_count"],
                    "interface_specific_failure": row[
                        "interface_specific_failure"
                    ],
                }
                for surface, row in attribute_row["surfaces"].items()
            }
            for attribute, attribute_row in model_row["attributes"].items()
        }
        for model_name, model_row in models.items()
    }
    print(
        {
            "phase": 1093,
            "compact": compact,
            "diagnostic_digest": output["diagnostic_digest"],
        }
    )


if __name__ == "__main__":
    main()
