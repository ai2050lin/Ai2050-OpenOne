#!/usr/bin/env python3
"""Freeze discovery-selected events and held-out units for BF16 audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
OUT_ROOT = SOURCE_ROOT / "precision_protocol"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
OUTPUT_MODES = ("entity", "property", "binary")
MAX_EVENTS_PER_MODEL = 8
UNITS_PER_PANEL = 8


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def discovery_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    discovery = row["splits"]["discovery"]
    return (
        -int(discovery["directional_panel_count"]),
        -int(discovery["specificity_panel_count"]),
        -float(
            discovery["cross_panel_direction_consistency"]
            if discovery["cross_panel_direction_consistency"] is not None
            else -1.0
        ),
        str(row["event_id"]),
    )


def select_units(
    units: list[dict[str, Any]],
    family: str,
    output_mode: str,
) -> list[dict[str, Any]]:
    panel = [
        row for row in units
        if row["family"] == family
        and row["output_mode"] == output_mode
        and row["split"] == "confirmation"
    ]
    templates = sorted({int(row["template"]) for row in panel})
    pools = sorted({int(row["name_pool"]) for row in panel})
    if len(templates) < 2 or len(pools) < 2:
        raise RuntimeError(
            f"underfilled confirmation panel {family}/{output_mode}"
        )
    selected = []
    for world_index in range(4):
        for template, pool in (
            (templates[0], pools[0]),
            (templates[1], pools[1]),
        ):
            matches = [
                row for row in panel
                if int(row["world_index"]) == world_index
                and int(row["template"]) == template
                and int(row["name_pool"]) == pool
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"precision cell drift {family}/{output_mode}/"
                    f"w{world_index}/t{template}/p{pool}"
                )
            selected.append(matches[0])
    if len(selected) != UNITS_PER_PANEL:
        raise RuntimeError("precision panel unit count drift")
    return selected


def main() -> None:
    source_protocol = read_json(
        SOURCE_ROOT / "protocol" / "protocol.json"
    )
    analysis = read_json(SOURCE_ROOT / "analysis" / "summary.json")
    candidates = read_jsonl(
        SOURCE_ROOT
        / "analysis"
        / "control_specific_shared_events.jsonl"
    )
    model_summaries = []
    digest_payload = {
        "phase": 1014,
        "source_protocol_digest": source_protocol[
            "preregistration_digest"
        ],
        "source_analysis_schema": analysis["schema_version"],
        "selection_uses": (
            "discovery-only control-specific shared-direction events"
        ),
        "confirmation_sampling": (
            "per panel and world: template0/pool0 plus "
            "template1/pool1 in confirmation split"
        ),
        "maximum_events_per_model": MAX_EVENTS_PER_MODEL,
        "units_per_panel": UNITS_PER_PANEL,
    }
    for model in MODELS:
        model_candidates = sorted(
            [row for row in candidates if row["model"] == model],
            key=discovery_rank,
        )[:MAX_EVENTS_PER_MODEL]
        selections = []
        for rank, row in enumerate(model_candidates, 1):
            discovery = row["splits"]["discovery"]
            selections.append({
                "schema_version": (
                    "phase1014_bf16_event_selection.v1"
                ),
                "phase": 1014,
                "model": model,
                "selection_rank": rank,
                "operation": row["operation"],
                "event_index": int(row["event_index"]),
                "event_id": row["event_id"],
                "component": row["component"],
                "depth": int(row["depth"]),
                "head": row["head"],
                "discovery_directional_panel_count": int(
                    discovery["directional_panel_count"]
                ),
                "discovery_specificity_panel_count": int(
                    discovery["specificity_panel_count"]
                ),
                "discovery_cross_panel_direction_consistency": (
                    discovery["cross_panel_direction_consistency"]
                ),
                "selection_used_confirmation": False,
                "claim": (
                    "BF16 audit target frozen from 8-bit discovery "
                    "evidence only"
                ),
            })
        units = read_jsonl(
            SOURCE_ROOT / "protocol" / model / "units.jsonl"
        )
        selected_units = []
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                selected_units.extend(
                    select_units(units, family, output_mode)
                )
        write_jsonl(OUT_ROOT / model / "events.jsonl", selections)
        write_jsonl(OUT_ROOT / model / "units.jsonl", selected_units)
        model_summaries.append({
            "model": model,
            "selected_event_count": len(selections),
            "confirmation_unit_count": len(selected_units),
            "confirmation_panel_count": len(FAMILIES)
            * len(OUTPUT_MODES),
            "world_counts": {
                str(world): sum(
                    int(row["world_index"]) == world
                    for row in selected_units
                )
                for world in range(4)
            },
            "selection_used_confirmation": False,
        })
        digest_payload[model] = {
            "events": selections,
            "unit_ids": [row["unit_id"] for row in selected_units],
        }
    protocol_digest = canonical_digest(digest_payload)
    result = {
        "schema_version": "phase1014_bf16_precision_protocol.v1",
        "phase": 1014,
        "source_protocol_digest": source_protocol[
            "preregistration_digest"
        ],
        "precision_protocol_digest": protocol_digest,
        "maximum_events_per_model": MAX_EVENTS_PER_MODEL,
        "units_per_panel": UNITS_PER_PANEL,
        "state_forward_mode": "singleton_bfloat16",
        "selection_used_confirmation": False,
        "model_summaries": model_summaries,
        "claim_limit": (
            "precision audit protocol only; no causal intervention"
        ),
    }
    write_json(OUT_ROOT / "protocol.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
