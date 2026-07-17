#!/usr/bin/env python3
"""Phase461 joint-balance and generator-structure audit.

No model run, no CUDA, no physical trace. Compares the successful Phase452
generator with the failed independent Phase455/458 generators.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE452_SAMPLES = ROOT / "tests" / "gpt5" / "result" / "phase452_v2_large_holdout_protocol" / "phase452_v2_glm4_knowledge_large_holdout_samples.jsonl"
PHASE453_SUMMARY = ROOT / "tests" / "gpt5" / "result" / "phase453_glm4_v2_large_holdout_behavior" / "phase453_glm4_v2_large_holdout_summary.json"
PHASE455_SAMPLES = ROOT / "tests" / "gpt5" / "result" / "phase455_independent_core_protocol" / "phase455_independent_core_samples.jsonl"
PHASE456_SUMMARY = ROOT / "tests" / "gpt5" / "result" / "phase456_glm4_independent_core_behavior" / "phase456_glm4_independent_core_summary.json"
PHASE458_SAMPLES = ROOT / "tests" / "gpt5" / "result" / "phase458_independent_v2_protocol" / "phase458_independent_v2_samples.jsonl"
PHASE460_READOUT = ROOT / "tests" / "gpt5" / "result" / "phase460_independent_v2_readout" / "phase460_independent_v2_readout.json"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase461_generator_structure_audit"
OUT_PATH = OUT_DIR / "phase461_generator_structure_audit.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def text_metrics(text: str) -> dict[str, Any]:
    return {
        "chars": len(text),
        "tokens_approx": len(text.split()),
        "sentence_marks": sum(text.count(mark) for mark in ".?!"),
        "semicolon_count": text.count(";"),
        "record_marker_count": len(re.findall(r"\b(Fact|Record|Cell)\b", text)),
        "claim_first": text.startswith("Claim:"),
    }


def phase452_position(sample: dict[str, Any]) -> dict[str, Any]:
    roles = sample["logic_form"]["roles"]
    entity = roles["entity"]
    query = roles["query_property"]
    first = roles["first_candidate"]
    last = roles["last_candidate"]
    target_position = 0 if entity.endswith("_u") else 1
    query_position = 0 if query == first else 1 if query == last else "unknown"
    return {
        "target_position": target_position,
        "query_marker_position": query_position,
        "target_position_family": "binary",
    }


def phase455_position(sample: dict[str, Any]) -> dict[str, Any]:
    nodes = sample["role_nodes"]
    target_position = 1 if nodes["target_item"] == nodes["right_item"] else 0
    query_position = 1 if nodes["query_signal"] == nodes["right_signal"] else 0
    return {
        "target_position": target_position,
        "query_marker_position": query_position,
        "target_position_family": "binary",
    }


def phase458_position(sample: dict[str, Any]) -> dict[str, Any]:
    nodes = sample["role_nodes"]
    return {
        "target_position": nodes["target_position"],
        "query_marker_position": nodes["query_marker_position"],
        "target_position_family": "four_position",
    }


def sample_facts(sample: dict[str, Any]) -> list[str]:
    if "facts" in sample:
        return list(sample["facts"])
    variants = sample["surface_variants"]
    text = variants[0]["text"]
    claim_index = text.find("Claim:")
    facts_text = text[:claim_index] if claim_index >= 0 else text
    return [part.strip() for part in facts_text.split(".") if part.strip()]


def sample_claim(sample: dict[str, Any]) -> str:
    if "claim" in sample:
        return sample["claim"]
    roles = sample["logic_form"]["roles"]
    return f"{roles['entity']} has trait {roles['query_property']}."


def normalize_sample(generator: str, sample: dict[str, Any]) -> dict[str, Any]:
    if generator == "phase452_success":
        pos = phase452_position(sample)
    elif generator == "phase455_independent_v1":
        pos = phase455_position(sample)
    elif generator == "phase458_independent_v2":
        pos = phase458_position(sample)
    else:
        raise ValueError(generator)
    facts = sample_facts(sample)
    variants = sample["surface_variants"]
    variant_metrics = [text_metrics(variant["text"]) for variant in variants]
    return {
        "generator": generator,
        "sample_id": sample["sample_id"],
        "pair_id": sample["source_pair_id"],
        "pair_index": sample["pair_index"],
        "label": sample["canonical_answer"],
        "role": sample["pair_role"],
        **pos,
        "fact_count": len(facts),
        "claim_chars": len(sample_claim(sample)),
        "variant_count": len(variants),
        "avg_variant_chars": mean(item["chars"] for item in variant_metrics),
        "avg_variant_tokens": mean(item["tokens_approx"] for item in variant_metrics),
        "avg_record_markers": mean(item["record_marker_count"] for item in variant_metrics),
        "has_stress_claim_first": any(item["claim_first"] for item in variant_metrics),
        "core_transform_names": tuple(
            variant["transform"] for variant in variants
            if variant.get("track", "core") == "core" or variant["transform"].startswith("v2_")
        ),
    }


def joint_counts(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, int]:
    counts = Counter(tuple(row[field] for field in fields) for row in rows)
    return {" | ".join(map(str, key)): value for key, value in sorted(counts.items())}


def structural_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(field: str) -> float:
        return mean(row[field] for row in rows)
    return {
        "n_samples": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "roles": dict(Counter(row["role"] for row in rows)),
        "target_positions": dict(Counter(row["target_position"] for row in rows)),
        "query_marker_positions": dict(Counter(row["query_marker_position"] for row in rows)),
        "fact_count_values": dict(Counter(row["fact_count"] for row in rows)),
        "variant_count_values": dict(Counter(row["variant_count"] for row in rows)),
        "avg_claim_chars": avg("claim_chars"),
        "avg_variant_chars": avg("avg_variant_chars"),
        "avg_variant_tokens": avg("avg_variant_tokens"),
        "avg_record_markers": avg("avg_record_markers"),
        "label_role_joint": joint_counts(rows, ["label", "role"]),
        "label_target_joint": joint_counts(rows, ["label", "target_position"]),
        "role_target_joint": joint_counts(rows, ["role", "target_position"]),
        "label_role_target_joint": joint_counts(rows, ["label", "role", "target_position"]),
        "label_role_target_query_joint": joint_counts(rows, ["label", "role", "target_position", "query_marker_position"]),
    }


def performance_summary() -> dict[str, Any]:
    phase453 = load_json(PHASE453_SUMMARY)
    phase456 = load_json(PHASE456_SUMMARY)
    phase460 = load_json(PHASE460_READOUT)
    return {
        "phase452_success_generator": {
            "overall": phase453["overall"],
            "orbit": phase453["orbit"],
            "gate_readout": phase453["gate_readout"],
        },
        "phase455_independent_v1": {
            "overall": phase456["overall"],
            "core_track": phase456["core_track"],
            "stress_track": phase456["stress_track"],
        },
        "phase458_independent_v2": {
            "overall": phase460["overall"],
            "core_track": phase460["core_track"],
            "stress_track": phase460["stress_track"],
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "phase452_success_generator": [
            normalize_sample("phase452_success", sample)
            for sample in load_jsonl(PHASE452_SAMPLES)
        ],
        "phase455_independent_v1": [
            normalize_sample("phase455_independent_v1", sample)
            for sample in load_jsonl(PHASE455_SAMPLES)
        ],
        "phase458_independent_v2": [
            normalize_sample("phase458_independent_v2", sample)
            for sample in load_jsonl(PHASE458_SAMPLES)
        ],
    }
    summaries = {name: structural_summary(rows) for name, rows in data.items()}
    out = {
        "schema_version": "phase461_generator_structure_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda_no_physical_trace",
        "generator_structural_summaries": summaries,
        "performance_summaries": performance_summary(),
        "interpretation": {
            "main_result": "Phase454 remains a within-generator window; generator-independent S3-core is not confirmed.",
            "major_confound": "All audited pair designs bind base to A and counterfactual to B, so counterfactual consistency is confounded with label direction and role identity.",
            "success_failure_structure_gap": [
                "phase452 has two entities and four facts; phase458 has four entities and eight facts.",
                "phase458 prompts are much longer and contain more record/cell markers.",
                "phase458 tests four-position distractor structure, not merely generator style.",
            ],
            "next_step": "pre-register a bridge protocol that keeps Phase452 difficulty constant while breaking base/A and counterfactual/B binding.",
        },
        "authorization": {
            "physical_trace_authorized": False,
            "model_run_authorized": False,
            "next_step": "phase462_bridge_protocol_static_only",
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
