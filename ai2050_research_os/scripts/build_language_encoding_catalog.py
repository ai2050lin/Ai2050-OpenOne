#!/usr/bin/env python3
"""Build the manifest-driven language-encoding client catalog."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OS_ROOT = ROOT / "ai2050_research_os"
REGISTRY = OS_ROOT / "registry"
DESTINATION = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"


def read(name: str):
    return json.loads((REGISTRY / name).read_text(encoding="utf-8"))


def main() -> None:
    families = read("language_families.json")["families"]
    datasets = read("field_datasets.json")["datasets"]
    relations = read("response_relations.json")["relations"]
    visualizations = read("visualization_specs.json")["specs"]
    ids = [row["id"] for row in datasets]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate dataset id")
    for dataset in datasets:
        source = ROOT / "frontend/public" / dataset["source_path"].lstrip("/")
        if not source.exists():
            raise FileNotFoundError(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != dataset["source_schema"]:
            raise RuntimeError((dataset["id"], payload.get("schema"), dataset["source_schema"]))
        if payload.get("rows") and any(len(row.get("values", [])) != dataset["coordinate_count"] for row in payload["rows"]):
            raise RuntimeError((dataset["id"], "coordinate width"))
        dataset["row_count"] = len(payload.get("rows", []))
    catalog = {
        "schema": "language-encoding-catalog.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "families": families,
        "datasets": datasets,
        "relations": relations,
        "visualizations": visualizations,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"built {DESTINATION.relative_to(ROOT)}: {len(datasets)} datasets, {sum(d['row_count'] for d in datasets)} rows")


if __name__ == "__main__":
    main()
