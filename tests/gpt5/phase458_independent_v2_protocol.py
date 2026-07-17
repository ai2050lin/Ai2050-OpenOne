#!/usr/bin/env python3
"""Phase458 independent generator v2 protocol.

No model run, no CUDA, no physical trace. This removes left/right coupling by
using four rotating positions and pre-registering core/stress tracks.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase458_independent_v2_protocol"
PROTOCOL_PATH = OUT_DIR / "phase458_independent_v2_protocol.json"
SAMPLES_PATH = OUT_DIR / "phase458_independent_v2_samples.jsonl"
AUDIT_PATH = OUT_DIR / "phase458_independent_v2_static_audit.json"
MANIFEST_PATH = OUT_DIR / "phase458_independent_v2_manifest.json"

PAIRS = 96
N_POS = 4
ENTITIES = ["avel", "brin", "coru", "delt", "eska", "fenn", "galo", "hexi", "irun", "jora", "kemi", "luto"]
KINDS = ["noral", "oswin", "pelar", "quint", "riven", "sovar", "taren", "ulvic", "vasko", "wyren"]
MARKERS = ["m01", "m13", "m27", "m34", "m42", "m58", "m69", "m75", "m86", "m90", "m104", "m118"]

CORE_TRANSFORMS = [
    "core_table_frame",
    "core_record_lines",
    "core_evidence_then_claim",
    "core_claim_reference_sync",
]
STRESS_TRANSFORMS = [
    "stress_claim_first",
    "stress_dense_records",
]


def stable_hash(*parts: object, n: int = 16) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label(is_true: bool) -> str:
    return "A" if is_true else "B"


def instruction(noun: str = "claim") -> str:
    return f"Reply with A if the {noun} is true, and B if it is false."


def make_pair(pair_index: int) -> list[dict[str, Any]]:
    code = f"p458_{pair_index:03d}"
    target_pos = pair_index % N_POS
    false_pos = (pair_index + 1 + (pair_index // N_POS) % (N_POS - 1)) % N_POS
    entities = [f"{ENTITIES[(pair_index + pos * 2) % len(ENTITIES)]}_{code}_{stable_hash('entity', pair_index, pos, n=4)}" for pos in range(N_POS)]
    kinds = [f"{KINDS[(pair_index + pos * 3) % len(KINDS)]}_{code}_{stable_hash('kind', pair_index, pos, n=4)}" for pos in range(N_POS)]
    markers = [f"{MARKERS[(pair_index + pos * 5) % len(MARKERS)]}_{code}_{stable_hash('marker', pair_index, pos, n=4)}" for pos in range(N_POS)]
    facts = []
    for pos in range(N_POS):
        facts.append(f"Cell {pos + 1} states that {entities[pos]} belongs to group {kinds[pos]}.")
        facts.append(f"Cell {pos + 1} rule says each {kinds[pos]} member has marker {markers[pos]}.")

    rows = []
    for role, is_true in (("base", True), ("counterfactual", False)):
        query_marker = markers[target_pos if is_true else false_pos]
        claim = f"{entities[target_pos]} has marker {query_marker}."
        logic = {
            "phase": "phase458",
            "pair_index": pair_index,
            "role": role,
            "target_position": target_pos,
            "query_marker_position": target_pos if is_true else false_pos,
            "truth_value": is_true,
        }
        semantic_hash = stable_hash(json.dumps(logic, sort_keys=True), n=20)
        rows.append({
            "sample_id": stable_hash("phase458", pair_index, role),
            "source_pair_id": stable_hash("phase458", pair_index, "pair"),
            "pair_index": pair_index,
            "pair_role": role,
            "model_target": "glm4",
            "ability": "knowledge_network",
            "task": "independent_v2_marker_truth",
            "split": "phase458_independent_v2_replicate",
            "canonical_answer": label(is_true),
            "truth_value": is_true,
            "logic_form": logic,
            "facts": facts,
            "claim": claim,
            "role_nodes": {
                "target_position": target_pos,
                "query_marker_position": target_pos if is_true else false_pos,
                "target_entity": entities[target_pos],
                "query_marker": query_marker,
            },
            "semantic_hash": semantic_hash,
            "core_transforms": CORE_TRANSFORMS,
            "stress_transforms": STRESS_TRANSFORMS,
            "surface_variants": variants(facts, claim, label(is_true), semantic_hash),
        })
    return rows


def variants(facts: list[str], claim: str, answer: str, semantic_hash: str) -> list[dict[str, str]]:
    fact_text = " ".join(facts)
    record_lines = " ".join(f"Record {idx + 1}: {fact}" for idx, fact in enumerate(facts))
    dense = " ; ".join(fact.rstrip(".") for fact in facts)
    raw = [
        ("core", "core_table_frame", f"Table facts: {fact_text} Claim: {claim} {instruction()}"),
        ("core", "core_record_lines", f"{record_lines} Claim: {claim} {instruction()}"),
        ("core", "core_evidence_then_claim", f"Evidence: {fact_text} Claim to evaluate: {claim} {instruction()}"),
        ("core", "core_claim_reference_sync", f"{fact_text} The claim is: {claim} {instruction()}"),
        ("stress", "stress_claim_first", f"Claim: {claim} Evidence follows: {fact_text} {instruction()}"),
        ("stress", "stress_dense_records", f"{dense}. Claim: {claim} {instruction()}"),
    ]
    return [
        {"track": track, "transform": transform, "text": text, "expected_label": answer, "semantic_hash": semantic_hash}
        for track, transform, text in raw
    ]


def build_samples() -> list[dict[str, Any]]:
    rows = []
    for pair_index in range(PAIRS):
        rows.extend(make_pair(pair_index))
    return rows


def baseline_predictions(sample: dict[str, Any], variant: dict[str, str]) -> dict[str, str]:
    nodes = sample["role_nodes"]
    text = variant["text"]
    return {
        "constant_A": "A",
        "constant_B": "B",
        "target_pos_0_or_1": "A" if nodes["target_position"] in (0, 1) else "B",
        "target_pos_even": "A" if nodes["target_position"] % 2 == 0 else "B",
        "query_marker_pos_even": "A" if nodes["query_marker_position"] % 2 == 0 else "B",
        "record_lines_template": "A" if "Record 1:" in text else "B",
        "claim_first_template": "A" if text.startswith("Claim:") else "B",
        "dense_template": "A" if " ; " in text else "B",
        "pair_index_even": "A" if sample["pair_index"] % 2 == 0 else "B",
        "text_length_even": "A" if len(text) % 2 == 0 else "B",
    }


def static_baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    failing = []
    for track_name, transforms in (("core", CORE_TRANSFORMS), ("stress", STRESS_TRANSFORMS), ("all", CORE_TRANSFORMS + STRESS_TRANSFORMS)):
        rows = [(s, v) for s in samples for v in s["surface_variants"] if v["transform"] in transforms]
        reports[track_name] = {}
        for baseline in baseline_predictions(rows[0][0], rows[0][1]):
            correct = sum(1 for sample, variant in rows if baseline_predictions(sample, variant)[baseline] == sample["canonical_answer"])
            n = len(rows)
            acc = correct / n
            reports[track_name][baseline] = {"n": n, "correct": correct, "accuracy": acc}
            if acc > 0.55:
                failing.append({"track": track_name, "baseline": baseline, "accuracy": acc})
    return {"pass": not failing, "reports": reports, "failing_baselines": failing}


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(s["canonical_answer"] for s in samples)
    role_counts = Counter(s["pair_role"] for s in samples)
    target_positions = Counter(s["role_nodes"]["target_position"] for s in samples)
    query_positions = Counter(s["role_nodes"]["query_marker_position"] for s in samples)
    pass_flag = (
        label_counts["A"] == label_counts["B"]
        and role_counts["base"] == role_counts["counterfactual"]
        and len(set(target_positions.values())) == 1
        and len(set(query_positions.values())) == 1
    )
    return {
        "pass": pass_flag,
        "label_counts": dict(label_counts),
        "role_counts": dict(role_counts),
        "target_positions": dict(target_positions),
        "query_marker_positions": dict(query_positions),
    }


def certificate_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    track_counts = Counter()
    for sample in samples:
        for variant in sample["surface_variants"]:
            track_counts[variant["track"]] += 1
            if variant["expected_label"] != sample["canonical_answer"] or variant["semantic_hash"] != sample["semantic_hash"]:
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "certificate_mismatch"})
            if variant["track"] == "core" and variant["text"].startswith("Claim:"):
                failures.append({"sample_id": sample["sample_id"], "transform": variant["transform"], "reason": "core_claim_first"})
    return {"pass": not failures, "track_counts": dict(track_counts), "failure_count": len(failures), "failures": failures[:20]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    protocol = {
        "schema_version": "phase458_independent_v2_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pre_registered_independent_v2_no_model_run",
        "target_model": "glm4",
        "target_task": "knowledge_network/independent_v2_marker_truth",
        "pairs": PAIRS,
        "samples": len(samples),
        "core_transforms": CORE_TRANSFORMS,
        "stress_transforms": STRESS_TRANSFORMS,
        "uses_phase446_generator": False,
        "uses_left_right_binary_frame": False,
        "physical_trace_authorized": False,
    }
    baseline = static_baseline_audit(samples)
    balance = balance_audit(samples)
    certs = certificate_audit(samples)
    all_pass = baseline["pass"] and balance["pass"] and certs["pass"]
    audit = {
        "schema_version": "phase458_independent_v2_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if all_pass else "static_fail_no_model_run",
        "balance": balance,
        "certificates": certs,
        "static_baselines": baseline,
        "authorization": {
            "glm4_behavior_replicate_authorized": all_pass,
            "physical_trace_authorized": False,
            "scope": "glm4_independent_v2_core_behavior_replicate_only",
        },
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(SAMPLES_PATH, samples)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {str(path.relative_to(ROOT)): sha256_file(path) for path in [PROTOCOL_PATH, SAMPLES_PATH, AUDIT_PATH]}
    manifest = {
        "schema_version": "phase458_independent_v2_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_no_model_run",
        "artifacts": artifacts,
        "joint_sha256": hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(AUDIT_PATH)


if __name__ == "__main__":
    main()
