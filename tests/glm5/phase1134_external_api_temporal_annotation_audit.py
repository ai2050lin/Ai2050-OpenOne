"""Independent, offline audit for Phase1134 external-model annotations."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/glm5/result/phase1132_postrelease_temporal_material"
    / "revision6_temporal_relation_binding_overprovisioned"
    / "material/candidate_package_unreviewed.jsonl"
)
RESULT = ROOT / "tests/glm5/result/phase1134_external_api_temporal_annotation"
PROTOCOL = RESULT / "protocol/protocol.json"
DEEPSEEK = RESULT / "reviews/deepseek_machine_review.jsonl"
CLAUDE = RESULT / "reviews/claude_machine_review.jsonl"
CONSENSUS = RESULT / "analysis/external_machine_consensus_package.jsonl"
DISAGREEMENTS = RESULT / "analysis/machine_disagreements.jsonl"
SUMMARY = RESULT / "analysis/summary.json"
OUTPUT = RESULT / "audit/independent_result_audit.json"
PROMPT_VERSION = "phase1134_temporal_material_review.v4"
REQUIRED = (
    "gold_answer_correct",
    "candidate_unique",
    "matched_null_globally_false",
    "matched_null_locally_plausible",
    "natural_language_acceptable",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = read_jsonl(SOURCE)
    deepseek = read_jsonl(DEEPSEEK)
    claude = read_jsonl(CLAUDE)
    consensus = read_jsonl(CONSENSUS)
    disagreements = read_jsonl(DISAGREEMENTS)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    source_ids = [str(row["item_id"]) for row in source]
    source_set = set(source_ids)
    reviews = {"deepseek": deepseek, "claude": claude}
    indexed = {
        name: {str(row["item_id"]): row for row in rows}
        for name, rows in reviews.items()
    }
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("source_count_493", len(source) == 493, len(source))
    check("source_ids_unique", len(source_set) == len(source_ids), len(source_set))
    check(
        "protocol_source_hash",
        protocol.get("source_sha256") == sha256(SOURCE),
        protocol.get("source_sha256"),
    )
    check(
        "protocol_prompt_v4",
        protocol.get("prompt_version") == PROMPT_VERSION,
        protocol.get("prompt_version"),
    )
    check(
        "protocol_nonhuman_scope",
        protocol.get("phase1132_human_ingest_eligible") is False,
        protocol.get("evidence_scope"),
    )

    for name, rows in reviews.items():
        ids = [str(row["item_id"]) for row in rows]
        check(f"{name}_count_493", len(rows) == 493, len(rows))
        check(f"{name}_coverage", set(ids) == source_set, len(set(ids)))
        check(f"{name}_ids_unique", len(ids) == len(set(ids)), len(set(ids)))
        check(
            f"{name}_required_booleans",
            all(
                isinstance(row.get(field), bool)
                for row in rows
                for field in REQUIRED
            ),
            len(rows),
        )
        check(
            f"{name}_prompt_v4",
            all(row.get("prompt_version") == PROMPT_VERSION for row in rows),
            len(rows),
        )
        check(
            f"{name}_cannot_enter_human_ingest",
            all(
                row.get("reviewer_type") == "external_model"
                and row.get("human_reviewer") is False
                and row.get("eligible_for_phase1132_human_ingest") is False
                and row.get("annotation_blinded_to_model_outputs") is False
                for row in rows
            ),
            len(rows),
        )

    expected_joint = {
        item_id
        for item_id in source_ids
        if all(
            indexed[name][item_id][field]
            for name in reviews
            for field in REQUIRED
        )
    }
    expected_disagreement = {
        item_id
        for item_id in source_ids
        if any(
            indexed["deepseek"][item_id][field]
            != indexed["claude"][item_id][field]
            for field in REQUIRED
        )
    }
    consensus_ids = {str(row["item_id"]) for row in consensus}
    disagreement_ids = {str(row["item_id"]) for row in disagreements}
    check("joint_consensus_count_491", len(expected_joint) == 491, len(expected_joint))
    check("consensus_package_exact", consensus_ids == expected_joint, len(consensus_ids))
    check(
        "consensus_stays_machine_only",
        all(
            row.get("machine_validation_only") is True
            and row.get("external_machine_review", {}).get("human_reviewer") is False
            and row.get("external_machine_review", {}).get(
                "eligible_for_phase1132_human_ingest"
            )
            is False
            for row in consensus
        ),
        len(consensus),
    )
    check("disagreement_count_2", len(expected_disagreement) == 2, len(expected_disagreement))
    check(
        "disagreement_queue_exact",
        disagreement_ids == expected_disagreement,
        sorted(disagreement_ids),
    )
    check(
        "summary_counts_match",
        summary.get("comparison", {}).get("jointly_accepted_count") == 491
        and summary.get("comparison", {}).get("disagreement_count") == 2
        and summary.get("human_review_gate_satisfied") is False,
        summary.get("comparison"),
    )

    raw_root = RESULT / "raw" / PROMPT_VERSION / "batch_20"
    for name in reviews:
        batch_files = sorted((raw_root / name).glob("batch_*.json"))
        raw_ids: list[str] = []
        raw_valid = True
        for path in batch_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_ids.extend(str(item_id) for item_id in payload.get("item_ids", []))
            raw_valid = raw_valid and (
                payload.get("prompt_version") == PROMPT_VERSION
                and payload.get("source_sha256") == sha256(SOURCE)
                and payload.get("api_key_recorded") is False
            )
        check(f"{name}_raw_batch_count_25", len(batch_files) == 25, len(batch_files))
        check(f"{name}_raw_batches_valid", raw_valid, len(batch_files))
        check(f"{name}_raw_partition_exact", raw_ids == source_ids, len(raw_ids))

    artifact_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (PROTOCOL, DEEPSEEK, CLAUDE, CONSENSUS, DISAGREEMENTS, SUMMARY)
    }
    result = {
        "schema_version": "phase1134_external_annotation_independent_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "artifact_sha256": artifact_hashes,
        "claim_scope": "external-model review only; human gate remains unsatisfied",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
