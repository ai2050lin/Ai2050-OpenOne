#!/usr/bin/env python3
"""Compute the frozen Phase1122 lexical nulls and compare them with Phase1121."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1121_wordnet_adjective_double_orthogonal_finalize as source_finalize
import phase1121_wordnet_adjective_double_orthogonal_protocol as source
import phase1122_adjective_lexical_coherence_protocol as protocol


def cosine_counts(left: Counter[str], right: Counter[str], idf: dict[str, float] | None = None) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    weights = idf or {}
    dot = sum(left[key] * right[key] * (weights.get(key, 1.0) ** 2) for key in keys)
    left_norm = math.sqrt(sum((left[key] * weights.get(key, 1.0)) ** 2 for key in keys))
    right_norm = math.sqrt(sum((right[key] * weights.get(key, 1.0)) ** 2 for key in keys))
    return dot / (left_norm * right_norm) if left_norm > 0.0 and right_norm > 0.0 else 0.0


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def char_trigrams(text: str) -> Counter[str]:
    normalized = " ".join(protocol.word_tokens(text))
    padded = f"  {normalized}  "
    return Counter(padded[index:index + 3] for index in range(max(len(padded) - 2, 0)))


def build_idf(documents: list[list[str]]) -> dict[str, float]:
    document_count = len(documents)
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update(set(document))
    return {term: math.log((1.0 + document_count) / (1.0 + count)) + 1.0 for term, count in frequency.items()}


def score(metric: str, sentence: str, ablated: str, definition: str, idf: dict[str, float]) -> float:
    if metric == "target_ablated_raw_unigram_jaccard":
        return jaccard(protocol.word_tokens(ablated), protocol.word_tokens(definition))
    if metric == "target_ablated_content_unigram_jaccard":
        return jaccard(protocol.content_tokens(ablated), protocol.content_tokens(definition))
    if metric == "target_ablated_content_tfidf_cosine":
        return cosine_counts(Counter(protocol.content_tokens(ablated)), Counter(protocol.content_tokens(definition)), idf)
    if metric == "target_ablated_character_trigram_cosine":
        return cosine_counts(char_trigrams(ablated), char_trigrams(definition))
    if metric == "full_content_tfidf_cosine":
        return cosine_counts(Counter(protocol.content_tokens(sentence)), Counter(protocol.content_tokens(definition)), idf)
    raise KeyError(metric)


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def grouped_rate(rows: list[dict[str, Any]], field: str, key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {name: rate(panel, key) for name, panel in sorted(grouped.items())}


def lexical_interactions(material: dict[str, Any], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = material["rows"]
    documents: list[list[str]] = []
    for row in rows:
        documents.append(protocol.content_tokens(row["target_ablated_sentence"]))
        documents.extend(protocol.content_tokens(definition) for definition in row["definitions"])
    idf = build_idf(documents)
    metrics = [protocol.PRIMARY_METRIC, *protocol.SECONDARY_METRICS]
    by_cell = {(row["concept_id"], row["surface"], int(row["context_sense"])): row for row in rows}
    selected_by_id = {row["concept_id"]: row for row in selected}
    output: list[dict[str, Any]] = []
    for concept in selected:
        for surface in source.SURFACES:
            cells: dict[tuple[int, int, str], float] = {}
            for context_sense in source.SENSES:
                material_row = by_cell[(concept["concept_id"], surface, context_sense)]
                for definition_sense in source.DEFINITION_SENSES:
                    definition = concept["definitions"][definition_sense]
                    for metric in metrics:
                        cells[(context_sense, definition_sense, metric)] = score(
                            metric,
                            material_row["sentence"],
                            material_row["target_ablated_sentence"],
                            definition,
                            idf,
                        )
            metric_values: dict[str, float] = {}
            for metric in metrics:
                metric_values[metric] = 0.5 * (
                    (cells[(0, 0, metric)] - cells[(0, 1, metric)])
                    - (cells[(1, 0, metric)] - cells[(1, 1, metric)])
                )

            deranged = selected_by_id[concept["deranged_control_concept_id"]]
            deranged_values: dict[str, float] = {}
            for metric in metrics:
                deranged_cells: dict[tuple[int, int], float] = {}
                for context_sense in source.SENSES:
                    material_row = by_cell[(concept["concept_id"], surface, context_sense)]
                    for definition_sense in source.DEFINITION_SENSES:
                        deranged_cells[(context_sense, definition_sense)] = score(
                            metric,
                            material_row["sentence"],
                            material_row["target_ablated_sentence"],
                            deranged["definitions"][definition_sense],
                            idf,
                        )
                deranged_values[metric] = 0.5 * (
                    (deranged_cells[(0, 0)] - deranged_cells[(0, 1)])
                    - (deranged_cells[(1, 0)] - deranged_cells[(1, 1)])
                )
            output.append({
                "concept_id": concept["concept_id"],
                "deranged_control_concept_id": concept["deranged_control_concept_id"],
                "split": concept["split"],
                "surface": surface,
                "metric_interactions": metric_values,
                "deranged_metric_interactions": deranged_values,
            })
    return output


def model_metrics(model_name: str, lexical: list[dict[str, Any]]) -> dict[str, Any]:
    detail = source.read_jsonl(source.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl")
    interactions = source_finalize.interaction_rows(detail)
    lexical_by_cell = {(row["concept_id"], row["surface"]): row for row in lexical}
    joined: list[dict[str, Any]] = []
    for row in interactions:
        null = lexical_by_cell[(row["concept_id"], row["surface"])]
        primary = float(null["metric_interactions"][protocol.PRIMARY_METRIC])
        joined.append({
            "concept_id": row["concept_id"],
            "split": row["split"],
            "surface": row["surface"],
            "template": row["template"],
            "model_interaction": row["interaction"],
            "model_hit": row["direction_hit"],
            "primary_null_interaction": primary,
            "primary_null_hit": primary > 0.0,
            "primary_adversarial": primary <= 0.0,
        })
    adversarial = [row for row in joined if row["primary_adversarial"]]
    model_direction = rate(joined, "model_hit")
    primary_direction = rate(joined, "primary_null_hit")
    return {
        "model": model_name,
        "interaction_count": len(joined),
        "model_direction_accuracy": model_direction,
        "primary_null_direction_accuracy_on_model_cells": primary_direction,
        "behavior_advantage_over_primary": model_direction - primary_direction,
        "primary_adversarial_interaction_count": len(adversarial),
        "primary_adversarial_direction_accuracy": rate(adversarial, "model_hit"),
        "primary_adversarial_by_split": grouped_rate(adversarial, "split", "model_hit"),
        "primary_adversarial_by_surface": grouped_rate(adversarial, "surface", "model_hit"),
        "joined_digest": protocol.digest(joined),
    }


def compute() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    material = protocol.read_json(protocol.OUT_ROOT / "protocol" / "material.json")
    source_prereg = source.read_json(source.OUT_ROOT / "protocol" / "preregistration.json")
    source_final = source.read_json(source.OUT_ROOT / "analysis" / "final_summary.json")
    selected = source.read_json(source.OUT_ROOT / "protocol" / "selected_concepts.json")["selected"]
    if not audit["all_checks_passed"] or prereg["source_protocol_digest"] != source_prereg["protocol_digest"]:
        raise RuntimeError("Phase1122 source or protocol audit failed")
    if prereg["source_final_digest"] != source_final["final_digest"]:
        raise RuntimeError("Phase1121 final result changed after Phase1122 freeze")

    lexical = lexical_interactions(material, selected)
    metrics = [protocol.PRIMARY_METRIC, *protocol.SECONDARY_METRICS]
    null_summary: dict[str, Any] = {}
    for metric in metrics:
        panel = [
            {
                "hit": row["metric_interactions"][metric] > 0.0,
                "deranged_hit": row["deranged_metric_interactions"][metric] > 0.0,
                "split": row["split"],
                "surface": row["surface"],
            }
            for row in lexical
        ]
        null_summary[metric] = {
            "cell_count": len(panel),
            "direction_rate": rate(panel, "hit"),
            "deranged_direction_rate": rate(panel, "deranged_hit"),
            "same_minus_deranged_direction_rate": rate(panel, "hit") - rate(panel, "deranged_hit"),
            "by_split": grouped_rate(panel, "split", "hit"),
            "by_surface": grouped_rate(panel, "surface", "hit"),
        }

    models = {model: model_metrics(model, lexical) for model in protocol.MODELS}
    thresholds = prereg["thresholds"]
    primary_rate = null_summary[protocol.PRIMARY_METRIC]["direction_rate"]
    max_secondary = max(null_summary[metric]["direction_rate"] for metric in protocol.SECONDARY_METRICS)
    qualified_reference_models = [
        model for model in protocol.REFERENCE_MODELS
        if models[model]["behavior_advantage_over_primary"] >= thresholds["minimum_behavior_advantage_over_primary"]
        and models[model]["primary_adversarial_direction_accuracy"] >= thresholds["minimum_primary_adversarial_direction_accuracy"]
        and models[model]["primary_adversarial_interaction_count"] >= thresholds["minimum_primary_adversarial_interaction_count"]
    ]
    gates = {
        "source_and_protocol_integrity": True,
        "primary_null_below_ceiling": primary_rate <= thresholds["maximum_primary_null_direction_rate"],
        "secondary_nulls_below_ceiling": max_secondary <= thresholds["maximum_any_secondary_null_direction_rate"],
        "reference_behavior_exceeds_primary": len(qualified_reference_models) >= thresholds["minimum_qualified_reference_models"],
        "scope_limit_preserved": True,
    }
    core = {
        "schema_version": "phase1122_lexical_coherence_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "source_protocol_digest": prereg["source_protocol_digest"],
        "source_final_digest": prereg["source_final_digest"],
        "lexical_interactions_digest": protocol.digest(lexical),
        "null_summary": null_summary,
        "models": models,
        "qualified_reference_models": qualified_reference_models,
        "gates": gates,
        "lexical_null_audit_passed": all(gates.values()),
        "interpretation": (
            "The frozen token-overlap null family does not reproduce K57, including model cells where the primary null points the wrong way. "
            "K57 remains behavior-level evidence only; static embedding, hidden, natural-use, and causal claims remain untested."
            if all(gates.values()) else
            "At least one frozen lexical null can account for too much of K57. K57 must not authorize hidden semantic interpretation without redesign."
        ),
    }
    final = dict(core)
    final["final_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "lexical_interactions.json", {"rows": lexical, "digest": protocol.digest(lexical)})
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    return final


def main() -> None:
    print(json.dumps(compute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
