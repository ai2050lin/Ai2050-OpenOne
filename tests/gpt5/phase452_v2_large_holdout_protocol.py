#!/usr/bin/env python3
"""Phase452 v2 large independent holdout protocol for GLM4 knowledge retest.

No model run, no CUDA, no physical trace. This freezes a larger v2 holdout
using a split that Phase446 behavior did not evaluate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE446_SAMPLES = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_static_contract" / "phase446_samples.jsonl"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase452_v2_large_holdout_protocol"
PROTOCOL_PATH = OUT_DIR / "phase452_v2_large_holdout_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase452_v2_glm4_knowledge_large_holdout_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase452_v2_large_holdout_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase452_v2_large_holdout_manifest.json"

SOURCE_SPLIT = "physical_window_freeze"
PAIRS = 96


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_facts(facts: str) -> list[str]:
    return [part.strip() for part in facts.split(". ") if part.strip()]


def claim_text(query_text: str) -> str:
    return query_text.removeprefix("Queried statement: ").strip()


def instruction(noun: str = "claim") -> str:
    return f"Reply with A if the {noun} is true, and B if it is false."


def v2_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    facts = row["facts_text"]
    claim = claim_text(row["query_text"])
    fact_lines = " ".join(f"Fact {idx + 1}: {fact.rstrip('.') }." for idx, fact in enumerate(split_facts(facts)))
    answer = row["canonical_answer"]
    semantic_hash = row["semantic_hash"]
    return [
        {
            "transform": "v2_lexical_frame",
            "text": f"Records: {facts} Claim: {claim} {instruction()}",
            "expected_label": answer,
            "semantic_hash": semantic_hash,
        },
        {
            "transform": "v2_boundary_bullets",
            "text": f"{fact_lines} Claim: {claim} {instruction()}",
            "expected_label": answer,
            "semantic_hash": semantic_hash,
        },
        {
            "transform": "v2_local_order_control",
            "text": f"Evidence: {facts} Claim to evaluate: {claim} {instruction()}",
            "expected_label": answer,
            "semantic_hash": semantic_hash,
        },
        {
            "transform": "v2_query_claim_sync",
            "text": f"{facts} Claim: {claim} {instruction()}",
            "expected_label": answer,
            "semantic_hash": semantic_hash,
        },
        {
            "transform": "v2_strong_statement_control",
            "text": f"Use only these facts: {facts} The statement is: {claim} Reply with A if the statement is true, and B if it is false.",
            "expected_label": answer,
            "semantic_hash": semantic_hash,
        },
    ]


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["ability"] == "knowledge_network"
        and row["task"] == "relation_truth_judgment"
        and row["split"] == SOURCE_SPLIT
        and row["pair_index"] < PAIRS
    ]
    return sorted(selected, key=lambda row: (row["pair_index"], row["pair_role"]))


def build_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append({
            "sample_id": stable_hash("phase452", row["sample_id"]),
            "source_sample_id": row["sample_id"],
            "source_pair_id": row["pair_id"],
            "pair_index": row["pair_index"],
            "pair_role": row["pair_role"],
            "model_target": "glm4",
            "ability": row["ability"],
            "task": row["task"],
            "split": "phase452_v2_large_independent_holdout",
            "canonical_answer": row["canonical_answer"],
            "truth_value": row["truth_value"],
            "logic_form": row["logic_form"],
            "semantic_hash": row["semantic_hash"],
            "surface_variants": v2_variants(row),
        })
    return out


def audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    label_counts = {"A": 0, "B": 0}
    role_counts = {"base": 0, "counterfactual": 0}
    transform_counts: dict[str, int] = {}
    for sample in samples:
        label_counts[sample["canonical_answer"]] += 1
        role_counts[sample["pair_role"]] += 1
        for variant in sample["surface_variants"]:
            text = variant["text"].lower()
            transform_counts[variant["transform"]] = transform_counts.get(variant["transform"], 0) + 1
            if "queried statement" in text:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "old_query_marker"})
            if "claim" in text and "if the queried statement is true" in text:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "instruction_marker_mismatch"})
            if text.find("claim") >= 0 and text.find("evidence:") >= 0 and text.find("claim") < text.find("evidence:"):
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "query_before_evidence"})
            if variant["expected_label"] != sample["canonical_answer"] or variant["semantic_hash"] != sample["semantic_hash"]:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "semantic_certificate_mismatch"})
    return {
        "schema_version": "phase452_v2_large_holdout_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": SOURCE_SPLIT,
        "sample_count": len(samples),
        "pair_count": len({sample["source_pair_id"] for sample in samples}),
        "variant_count": sum(len(sample["surface_variants"]) for sample in samples),
        "label_counts": label_counts,
        "role_counts": role_counts,
        "transform_counts": transform_counts,
        "failure_count": len(failures),
        "pass": not failures and label_counts["A"] == label_counts["B"] and role_counts["base"] == role_counts["counterfactual"],
        "failures": failures[:20],
        "authorization": {
            "model_rerun_authorized": not failures,
            "physical_trace_authorized": False,
            "scope": "glm4_knowledge_network_large_independent_behavior_retest_only",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples(select_rows(load_jsonl(PHASE446_SAMPLES)))
    protocol = {
        "schema_version": "phase452_v2_large_holdout_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "large_independent_v2_holdout_frozen_no_model_run",
        "target_model": "glm4",
        "target_task": "knowledge_network/relation_truth_judgment",
        "source_split": SOURCE_SPLIT,
        "pairs": PAIRS,
        "samples": len(samples),
        "variants_per_sample": 5,
        "strict_qualification_claimed": False,
        "physical_trace_authorized": False,
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    audit_report = audit(samples)
    AUDIT_PATH.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "phase452_v2_large_holdout_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_model_run",
        "artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [PROTOCOL_PATH, SAMPLES_PATH, AUDIT_PATH]
        },
    }
    manifest["joint_sha256"] = hashlib.sha256(json.dumps(manifest["artifacts"], sort_keys=True).encode("utf-8")).hexdigest()
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
