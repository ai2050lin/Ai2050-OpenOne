#!/usr/bin/env python3
"""Verify every derived role-edge reference used by the 288 frozen bundles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation"
BUNDLE_ROOT = PHASE_ROOT / "dynamic_bundle_extraction"
OUT = BUNDLE_ROOT / "phase366_derived_artifact_integrity_summary.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    references: dict[str, str] = {}
    conflicting_digest_count = 0
    bundle_count = 0
    for model in MODELS:
        for bundle_path in sorted((BUNDLE_ROOT / "blind_bundles" / model).glob("*.json")):
            bundle = read_json(bundle_path)
            bundle_count += 1
            for event in bundle["events"]:
                reference = event["vector_ref"]
                relative = reference["relative_path"]
                if not relative.startswith("dynamic_bundle_extraction/private/role_edges/"):
                    continue
                existing = references.get(relative)
                if existing is not None and existing != reference["sha256"]:
                    conflicting_digest_count += 1
                references[relative] = reference["sha256"]

    status_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    bad_rows = []
    for index, (relative, expected_digest) in enumerate(sorted(references.items()), 1):
        path = PHASE_ROOT / relative
        model = Path(relative).parts[3]
        model_counts[model] += 1
        if not path.is_file():
            status = "missing"
        elif path.stat().st_size == 0:
            status = "zero_byte"
        else:
            try:
                torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                status = "unreadable"
            else:
                status = "valid" if sha256_file(path) == expected_digest else "hash_mismatch"
        status_counts[status] += 1
        if status != "valid":
            bad_rows.append({"relative_path": relative, "status": status})
        if index % 2048 == 0 or index == len(references):
            print(f"integrity {index}/{len(references)} bad={len(bad_rows)}", flush=True)

    summary = {
        "schema_version": "43.3.0",
        "phase_id": "Phase366-C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "bundle_count": bundle_count,
            "unique_derived_file_reference_count": len(references),
            "model_file_counts": dict(sorted(model_counts.items())),
        },
        "results": {
            "status_counts": dict(sorted(status_counts.items())),
            "conflicting_expected_digest_count": conflicting_digest_count,
            "bad_file_count": len(bad_rows),
            "all_referenced_derived_files_valid": (
                status_counts == Counter({"valid": len(references)})
                and conflicting_digest_count == 0
            ),
        },
        "bad_rows": bad_rows,
        "claim_boundary": {
            "derived_artifact_integrity_verified": len(bad_rows) == 0,
            "raw_collection_integrity_source": "phase366_full_collection_summary",
            "scientific_motif_scoring_executed": False,
        },
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["results"]["all_referenced_derived_files_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
