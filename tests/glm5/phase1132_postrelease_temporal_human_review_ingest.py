#!/usr/bin/env python3
"""Merge two independent blind reviews into the frozen Phase 1132 package.

This utility validates human-supplied judgments. It never generates judgments,
loads a model, or starts a model test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    REPO_ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision6_temporal_relation_binding_overprovisioned"
)
FROZEN_PACKAGE = RESULT_ROOT / "material/candidate_package_unreviewed.jsonl"
DEFAULT_OUTPUT = RESULT_ROOT / "material/candidate_package_reviewed.jsonl"
REVIEW_RESULT_ROOT = RESULT_ROOT / "review"
REQUIRED_JUDGMENTS = (
    "gold_answer_correct",
    "candidate_unique",
    "matched_null_globally_false",
    "matched_null_locally_plausible",
    "natural_language_acceptable",
)
MIN_ITEMS_PER_SPLIT = 128

sys.path.insert(0, str(REPO_ROOT / "tests/glm5"))
from phase1131_material_readiness_and_claim_scope_audit import audit_package  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_review(
    path: Path, expected_ids: set[str]
) -> tuple[str, dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    item_ids = [str(row.get("item_id", "")) for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"{path}: duplicate item_id")
    if set(item_ids) != expected_ids:
        missing = len(expected_ids - set(item_ids))
        extra = len(set(item_ids) - expected_ids)
        raise ValueError(f"{path}: incomplete coverage (missing={missing}, extra={extra})")

    reviewer_ids = {str(row.get("reviewer_id", "")).strip() for row in rows}
    reviewer_ids.discard("")
    if len(reviewer_ids) != 1:
        raise ValueError(f"{path}: exactly one stable nonempty reviewer_id is required")
    reviewer_id = next(iter(reviewer_ids))

    for row in rows:
        if row.get("annotation_blinded_to_model_outputs") is not True:
            raise ValueError(f"{path}: blind-to-model-output attestation must be true")
        for field in REQUIRED_JUDGMENTS:
            if not isinstance(row.get(field), bool):
                raise ValueError(f"{path}: {field} must be a JSON boolean")
    return reviewer_id, {str(row["item_id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    package_rows = read_jsonl(FROZEN_PACKAGE)
    package_by_id = {str(row["item_id"]): row for row in package_rows}
    if len(package_rows) != len(package_by_id):
        raise ValueError("frozen package contains duplicate item_id values")

    reviewer_a, review_a = validate_review(args.review_a, set(package_by_id))
    reviewer_b, review_b = validate_review(args.review_b, set(package_by_id))
    if reviewer_a == reviewer_b:
        raise ValueError("the two review manifests must use distinct reviewer_id values")

    accepted: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    for item_id, source in package_by_id.items():
        reviews = (review_a[item_id], review_b[item_id])
        failed = [
            field
            for field in REQUIRED_JUDGMENTS
            if not all(review[field] for review in reviews)
        ]
        if failed:
            rejection_reasons.update(failed)
            continue
        row = dict(source)
        row.update(
            {
                "annotator_ids": sorted((reviewer_a, reviewer_b)),
                "annotation_blinded_to_model_outputs": True,
                "candidate_uniqueness_confirmed": True,
                "matched_null_globally_false_confirmed": True,
                "matched_null_locally_plausible_confirmed": True,
                "machine_validation_only": False,
                "human_review_manifest_sha256": {
                    reviewer_a: file_digest(args.review_a),
                    reviewer_b: file_digest(args.review_b),
                },
            }
        )
        accepted.append(row)

    write_jsonl(args.output, accepted)
    contract_audit = audit_package(args.output)
    split_counts = Counter(str(row["split"]) for row in accepted)
    volume_survived = all(
        split_counts[split] >= MIN_ITEMS_PER_SPLIT
        for split in ("discovery", "confirmation", "natural_use")
    )
    model_test_authorized = bool(volume_survived and contract_audit["material_ready"])
    summary = {
        "phase": 1132,
        "analysis_type": "external_human_review_ingest",
        "frozen_package": str(FROZEN_PACKAGE.relative_to(REPO_ROOT)),
        "frozen_package_sha256": file_digest(FROZEN_PACKAGE),
        "reviewer_ids": sorted((reviewer_a, reviewer_b)),
        "review_manifest_sha256": {
            reviewer_a: file_digest(args.review_a),
            reviewer_b: file_digest(args.review_b),
        },
        "input_count": len(package_rows),
        "accepted_count": len(accepted),
        "rejected_count": len(package_rows) - len(accepted),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "minimum_split_volume_survived": volume_survived,
        "phase1131_contract_audit": contract_audit,
        "model_test_authorized": model_test_authorized,
        "model_execution": False,
        "next_action": (
            "Run a separately frozen Phase 1133 FP16 behavior protocol."
            if model_test_authorized
            else "Do not test models; repair material supply without changing judgments."
        ),
    }
    write_json(REVIEW_RESULT_ROOT / "review_ingest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
