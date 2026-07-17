#!/usr/bin/env python3
"""Phase455 independent core-track protocol and static baseline audit.

No model run, no CUDA, no physical trace. This stage uses a new generator,
new vocabulary, pre-registered core/stress tracks, and static baselines before
any GLM4 replication.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase455_independent_core_protocol"
PROTOCOL_PATH = OUT_DIR / "phase455_independent_core_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase455_independent_core_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase455_independent_core_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase455_independent_core_manifest.json"

PAIRS = 96
Z_TWO_SIDED_95 = 1.96

ITEMS = [
    "arven", "belto", "cerin", "dasko", "elmir", "faron", "giren", "hasto",
    "ivren", "jasko", "kelto", "lorin", "mavik", "nerys", "olven", "praxu",
]
CLASSES = ["sable", "tundra", "vireo", "wexel", "yarden", "zephyr", "brindle", "corven"]
SIGNALS = ["amber", "cobalt", "dorsal", "ember", "frost", "garnet", "helium", "ivory"]
LEDGERS = ["ledger", "catalog", "register", "index", "roster", "docket"]

CORE_TRANSFORMS = [
    "core_catalog_frame",
    "core_numbered_records",
    "core_evidence_claim",
    "core_question_sync",
]

STRESS_TRANSFORMS = [
    "stress_claim_first",
    "stress_compact_semicolon",
]


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_bounds(k: int, n: int, z: float = Z_TWO_SIDED_95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def pick(pool: list[str], idx: int, offset: int = 0) -> str:
    return pool[(idx + offset) % len(pool)]


def label(is_true: bool) -> str:
    return "A" if is_true else "B"


def answer_instruction(noun: str = "claim") -> str:
    return f"Reply with A if the {noun} is true, and B if it is false."


def make_pair(pair_index: int) -> list[dict[str, Any]]:
    code = f"p455_{pair_index:03d}"
    item_a = f"{pick(ITEMS, pair_index)}_{code}_left"
    item_b = f"{pick(ITEMS, pair_index, 7)}_{code}_right"
    class_a = f"{pick(CLASSES, pair_index)}_{code}_class_a"
    class_b = f"{pick(CLASSES, pair_index, 3)}_{code}_class_b"
    signal_a = f"{pick(SIGNALS, pair_index)}_{code}_signal_a"
    signal_b = f"{pick(SIGNALS, pair_index, 5)}_{code}_signal_b"
    ledger = pick(LEDGERS, pair_index)
    use_right = pair_index % 2 == 1
    item = item_b if use_right else item_a
    true_signal = signal_b if use_right else signal_a
    false_signal = signal_a if use_right else signal_b
    facts = [
        f"{ledger.capitalize()} entry one says {item_a} is filed under {class_a}.",
        f"{ledger.capitalize()} entry two says every {class_a} item carries marker {signal_a}.",
        f"{ledger.capitalize()} entry three says {item_b} is filed under {class_b}.",
        f"{ledger.capitalize()} entry four says every {class_b} item carries marker {signal_b}.",
    ]
    rows = []
    for role, is_true in (("base", True), ("counterfactual", False)):
        query_signal = true_signal if is_true else false_signal
        claim = f"{item} carries marker {query_signal}."
        logic = {
            "phase": "phase455",
            "task": "independent_marker_truth",
            "pair_index": pair_index,
            "role": role,
            "item": item,
            "query_signal": query_signal,
            "truth_value": is_true,
            "facts": facts,
        }
        semantic_hash = stable_hash(json.dumps(logic, sort_keys=True), n=20)
        sample_id = stable_hash("phase455", pair_index, role)
        pair_id = stable_hash("phase455", pair_index, "pair")
        rows.append({
            "sample_id": sample_id,
            "source_pair_id": pair_id,
            "pair_index": pair_index,
            "pair_role": role,
            "model_target": "glm4",
            "ability": "knowledge_network",
            "task": "independent_marker_truth",
            "split": "phase455_independent_core_replicate",
            "canonical_answer": label(is_true),
            "truth_value": is_true,
            "logic_form": logic,
            "facts": facts,
            "claim": claim,
            "role_nodes": {
                "target_item": item,
                "left_item": item_a,
                "right_item": item_b,
                "left_signal": signal_a,
                "right_signal": signal_b,
                "query_signal": query_signal,
                "ledger": ledger,
            },
            "semantic_hash": semantic_hash,
            "core_transforms": CORE_TRANSFORMS,
            "stress_transforms": STRESS_TRANSFORMS,
            "surface_variants": surface_variants(facts, claim, label(is_true), semantic_hash),
        })
    return rows


def surface_variants(facts: list[str], claim: str, answer: str, semantic_hash: str) -> list[dict[str, str]]:
    fact_text = " ".join(facts)
    numbered = " ".join(f"Record {idx + 1}: {fact}" for idx, fact in enumerate(facts))
    variants = [
        {
            "track": "core",
            "transform": "core_catalog_frame",
            "text": f"Catalog facts: {fact_text} Claim: {claim} {answer_instruction()}",
        },
        {
            "track": "core",
            "transform": "core_numbered_records",
            "text": f"{numbered} Claim: {claim} {answer_instruction()}",
        },
        {
            "track": "core",
            "transform": "core_evidence_claim",
            "text": f"Evidence: {fact_text} Claim to check: {claim} {answer_instruction()}",
        },
        {
            "track": "core",
            "transform": "core_question_sync",
            "text": f"{fact_text} The claim is: {claim} {answer_instruction()}",
        },
        {
            "track": "stress",
            "transform": "stress_claim_first",
            "text": f"Claim: {claim} Evidence follows: {fact_text} {answer_instruction()}",
        },
        {
            "track": "stress",
            "transform": "stress_compact_semicolon",
            "text": f"{'; '.join(facts)} Claim: {claim} {answer_instruction()}",
        },
    ]
    return [
        {
            **variant,
            "expected_label": answer,
            "semantic_hash": semantic_hash,
        }
        for variant in variants
    ]


def build_samples() -> list[dict[str, Any]]:
    rows = []
    for pair_index in range(PAIRS):
        rows.extend(make_pair(pair_index))
    return rows


def baseline_predictions(sample: dict[str, Any], variant: dict[str, str]) -> dict[str, str]:
    text = variant["text"]
    nodes = sample["role_nodes"]
    claim = sample["claim"]
    return {
        "constant_A": "A",
        "constant_B": "B",
        "target_is_right_item": "A" if nodes["target_item"] == nodes["right_item"] else "B",
        "query_signal_is_right_signal": "A" if nodes["query_signal"] == nodes["right_signal"] else "B",
        "claim_mentions_left_signal": "A" if nodes["left_signal"] in claim else "B",
        "claim_mentions_right_signal": "A" if nodes["right_signal"] in claim else "B",
        "claim_mentions_first_fact_item": "A" if nodes["left_item"] in claim else "B",
        "claim_mentions_third_fact_item": "A" if nodes["right_item"] in claim else "B",
        "template_has_numbered_records": "A" if "Record 1:" in text else "B",
        "template_has_claim_first": "A" if text.startswith("Claim:") else "B",
        "text_length_even": "A" if len(text) % 2 == 0 else "B",
        "pair_index_even": "A" if sample["pair_index"] % 2 == 0 else "B",
    }


def static_baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    failing = []
    for track_name, transforms in (("core", CORE_TRANSFORMS), ("stress", STRESS_TRANSFORMS), ("all", CORE_TRANSFORMS + STRESS_TRANSFORMS)):
        rows = [
            (sample, variant)
            for sample in samples
            for variant in sample["surface_variants"]
            if variant["transform"] in transforms
        ]
        reports[track_name] = {}
        for baseline in baseline_predictions(rows[0][0], rows[0][1]):
            correct = sum(
                1 for sample, variant in rows
                if baseline_predictions(sample, variant)[baseline] == sample["canonical_answer"]
            )
            n = len(rows)
            acc = correct / n
            reports[track_name][baseline] = {"n": n, "correct": correct, "accuracy": acc}
            if acc > 0.55:
                failing.append({"track": track_name, "baseline": baseline, "accuracy": acc})
    return {"pass": not failing, "reports": reports, "failing_baselines": failing}


def track_certificate_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    counts = Counter()
    for sample in samples:
        labels = [variant["expected_label"] for variant in sample["surface_variants"]]
        if any(label != sample["canonical_answer"] for label in labels):
            failures.append({"sample_id": sample["sample_id"], "reason": "label_mismatch"})
        if any(variant["semantic_hash"] != sample["semantic_hash"] for variant in sample["surface_variants"]):
            failures.append({"sample_id": sample["sample_id"], "reason": "semantic_hash_mismatch"})
        for variant in sample["surface_variants"]:
            counts[variant["track"]] += 1
            text = variant["text"].lower()
            if "queried statement" in text:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "old_query_marker"})
            if variant["track"] == "core" and text.startswith("claim:"):
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "core_claim_first"})
    return {"pass": not failures, "track_variant_counts": dict(counts), "failure_count": len(failures), "failures": failures[:20]}


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(sample["canonical_answer"] for sample in samples)
    role_counts = Counter(sample["pair_role"] for sample in samples)
    target_side = Counter("right" if sample["role_nodes"]["target_item"] == sample["role_nodes"]["right_item"] else "left" for sample in samples)
    query_side = Counter("right" if sample["role_nodes"]["query_signal"] == sample["role_nodes"]["right_signal"] else "left" for sample in samples)
    pass_flag = (
        label_counts["A"] == label_counts["B"]
        and role_counts["base"] == role_counts["counterfactual"]
        and target_side["left"] == target_side["right"]
        and query_side["left"] == query_side["right"]
    )
    return {
        "pass": pass_flag,
        "label_counts": dict(label_counts),
        "role_counts": dict(role_counts),
        "target_side_counts": dict(target_side),
        "query_signal_side_counts": dict(query_side),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    protocol = {
        "schema_version": "phase455_independent_core_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pre_registered_independent_core_protocol_no_model_run",
        "target_model": "glm4",
        "target_task": "knowledge_network/independent_marker_truth",
        "pairs": PAIRS,
        "samples": len(samples),
        "core_transforms": CORE_TRANSFORMS,
        "stress_transforms": STRESS_TRANSFORMS,
        "pre_registered_tracks": True,
        "uses_phase446_generator": False,
        "physical_trace_authorized": False,
    }
    baseline = static_baseline_audit(samples)
    certs = track_certificate_audit(samples)
    balance = balance_audit(samples)
    all_pass = baseline["pass"] and certs["pass"] and balance["pass"]
    audit = {
        "schema_version": "phase455_independent_core_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if all_pass else "static_fail_no_model_run",
        "balance": balance,
        "semantic_and_track_certificates": certs,
        "static_baselines": baseline,
        "authorization": {
            "glm4_behavior_replicate_authorized": all_pass,
            "physical_trace_authorized": False,
            "scope": "glm4_independent_core_behavior_replicate_only",
        },
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [PROTOCOL_PATH, SAMPLES_PATH, AUDIT_PATH]
    }
    manifest = {
        "schema_version": "phase455_independent_core_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_model_run",
        "artifacts": artifacts,
        "joint_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
