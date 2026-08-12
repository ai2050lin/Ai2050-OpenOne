#!/usr/bin/env python3
"""Collect the Phase1099 sampled full-network relation-family atlas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1098_relative_relation_geometry_scan as shared
import phase1099_relation_family_atlas_protocol as protocol


DEPTH_FRACTIONS = tuple(value / 10.0 for value in range(11))
COMPONENTS = ("residual", "attention_output", "mlp_output")
_FULL_EVENT_DEFINITIONS = shared.event_definitions


def sampled_event_definitions(layer_count: int) -> list[dict]:
    events = _FULL_EVENT_DEFINITIONS(layer_count)
    selected = []
    seen = set()
    for component in COMPONENTS:
        candidates = [row for row in events if row["component"] == component]
        for fraction in DEPTH_FRACTIONS:
            row = min(candidates, key=lambda value: (abs(float(value["relative_depth"]) - fraction), int(value["depth"])))
            key = (str(row["component"]), int(row["depth"]))
            if key in seen:
                continue
            seen.add(key)
            copied = dict(row)
            copied["event_index"] = len(selected)
            selected.append(copied)
    return selected


def run(model_name: str) -> None:
    shared.protocol = protocol
    shared.shared.protocol = protocol
    shared.event_definitions = sampled_event_definitions
    shared.run(model_name)
    path = protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
    summary = protocol.read_json(path)
    summary["schema_version"] = "phase1099_model_family_atlas_summary.v1"
    summary["event_sampling"] = {
        "components": list(COMPONENTS),
        "requested_relative_depths": list(DEPTH_FRACTIONS),
        "deduplicated_event_count": summary["event_count"],
    }
    summary.pop("summary_digest", None)
    summary["summary_digest"] = protocol.digest(summary)
    protocol.write_json(path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
