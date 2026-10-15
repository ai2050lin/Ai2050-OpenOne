#!/usr/bin/env python3
"""Validate and publish the canonical language graph test-target registry."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "ai2050_research_os" / "registry"
SOURCE = REGISTRY / "language_graph_targets.json"
FAMILIES = REGISTRY / "language_families.json"
DESTINATION = ROOT / "frontend" / "public" / "research_data" / "current" / "language_graph_targets.v1.json"


def main() -> None:
    catalog = json.loads(SOURCE.read_text(encoding="utf-8"))
    families = json.loads(FAMILIES.read_text(encoding="utf-8"))["families"]
    family_ids = {row["id"] for row in families}
    target_ids = [row["id"] for row in catalog["targets"]]
    if catalog.get("schema_version") != "language-graph-targets.v1":
        raise ValueError("unexpected language graph target schema_version")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("duplicate language graph target id")
    for target in catalog["targets"]:
        if target["family_id"] not in family_ids:
            raise ValueError(f"unknown language family: {target['family_id']}")
        if target["domain_id"] not in {row["id"] for row in catalog["domains"]}:
            raise ValueError(f"unknown language graph domain: {target['domain_id']}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"published {len(target_ids)} language graph test targets to {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
