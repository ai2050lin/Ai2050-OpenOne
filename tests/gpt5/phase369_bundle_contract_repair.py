#!/usr/bin/env python3
"""Repair the missing anonymous group field in already computed Phase369 bundles."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase365_dynamic_flow_instrumentation import validate_blind_bundle  # noqa: E402


BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
COLLECTION = BASE / "raw_collection"
BUNDLES = BASE / "dynamic_bundle_extraction"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    repaired = 0
    valid = 0
    model_rows = []
    for model in MODELS:
        manifest = read_json(COLLECTION / "models" / model / "manifest.json")
        group_by_case = {
            row["blind_case_id"]: row["anonymous_group_id"] for row in manifest["case_rows"]
        }
        model_repaired = 0
        model_valid = 0
        for path in sorted((BUNDLES / "blind_bundles" / model).glob("*.json")):
            bundle = read_json(path)
            case_id = bundle["anonymous_case_id"]
            expected_group = group_by_case[case_id]
            if "anonymous_group_id" not in bundle:
                bundle["anonymous_group_id"] = expected_group
                write_json(path, bundle)
                repaired += 1
                model_repaired += 1
            elif bundle["anonymous_group_id"] != expected_group:
                raise RuntimeError(f"Group mismatch in {path}")
            errors = validate_blind_bundle(bundle)
            if errors:
                raise RuntimeError(f"Bundle remains invalid after repair: {path}: {errors[:3]}")
            valid += 1
            model_valid += 1
        model_rows.append({
            "model": model,
            "repaired_bundle_count": model_repaired,
            "valid_bundle_count": model_valid,
        })
    if repaired != 336 or valid != 336:
        raise RuntimeError(f"Unexpected repair denominator: repaired={repaired} valid={valid}")
    summary = {
        "schema_version": "46.0.0",
        "phase_id": "Phase369",
        "created_at": now(),
        "repair": "add_missing_anonymous_group_id_from_raw_collection_manifest",
        "raw_vectors_recomputed": False,
        "events_or_edges_modified": False,
        "repaired_bundle_count": repaired,
        "valid_bundle_count": valid,
        "models": model_rows,
    }
    write_json(BUNDLES / "phase369_bundle_contract_repair_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
