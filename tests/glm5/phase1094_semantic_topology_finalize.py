#!/usr/bin/env python3
"""Finalize Phase1094 semantic-alias versus topology orthogonalization.

All formal tests were preregistered in the protocol.  The analysis first asks
whether exact-word-disjoint synonym occurrences recreate graph nodes, then asks
whether a semantic scramble follows the words' actual concepts rather than the
researcher's nominal slots.  Physical hotspots remain descriptive.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_finalize as common
import phase1094_semantic_topology_protocol as protocol


EPSILON = 1e-12
PERMUTATIONS = np.asarray(list(itertools.permutations(range(8))), dtype=np.int16)


@contextlib.contextmanager
def using_protocol(value) -> Iterator[None]:
    previous = common.protocol
    common.protocol = value
    try:
        yield
    finally:
        common.protocol = previous


def load_models() -> dict[str, dict[str, Any]]:
    with using_protocol(protocol):
        return {name: common.load_model(name) for name in protocol.MODELS}


def unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > EPSILON else np.zeros_like(values)


def relation_from_gram(gram: np.ndarray) -> np.ndarray:
    upper = gram[np.triu_indices(gram.shape[0], k=1)]
    return unit(upper - upper.mean())


def relation(values: np.ndarray) -> np.ndarray:
    return relation_from_gram(values @ values.T)


def operation_names(attribute: str, topology: str, coherence: str) -> tuple[str, ...]:
    return protocol.operation_names(attribute, topology, coherence)


def profile(
    data: dict[str, Any],
    operation: str,
    surface: str,
    split: str,
    field: str,
    replicate: int,
    *,
    role: str = "answer_boundary",
    event_index: int | None = None,
    worlds: tuple[str, ...] | None = None,
) -> np.ndarray:
    values = []
    for world in (worlds or protocol.BASE_WORLDS):
        family = f"{operation}__{world}@{surface}"
        fi = protocol.FAMILIES.index(family)
        si = protocol.SPLITS.index(split)
        ri = protocol.CAPTURE_ROLES.index(role)
        di = protocol.SIGNED_FIELDS.index(field)
        current = data["direction_mean"][fi, si, :, ri, di, :, :, replicate, :]
        if event_index is not None:
            current = current[event_index:event_index + 1]
        current = current.mean(axis=(1, 2)).reshape(-1)
        values.append(current)
    return np.mean(np.stack(values), axis=0)


def bank(
    data: dict[str, Any],
    attribute: str,
    topology: str,
    coherence: str,
    surface: str,
    split: str,
    field: str,
    replicate: int,
    *,
    role: str = "answer_boundary",
    event_index: int | None = None,
    worlds: tuple[str, ...] | None = None,
) -> np.ndarray:
    values = np.stack([
        profile(
            data, operation, surface, split, field, replicate,
            role=role, event_index=event_index, worlds=worlds,
        )
        for operation in operation_names(attribute, topology, coherence)
    ])
    return common.row_normalize(values, centered=True)


def incidence_bank(topology: str, coherence: str, *, semantic: bool) -> np.ndarray:
    pairs = protocol.incidence_pairs(topology, coherence, semantic=semantic)
    values = np.zeros((8, 8), dtype=np.float64)
    for row, (left, right) in enumerate(pairs):
        values[row, left] = -1.0
        values[row, right] = 1.0
    return common.row_normalize(values, centered=True)


REFERENCES = {
    (topology, coherence, semantic): incidence_bank(
        topology, coherence, semantic=semantic
    )
    for topology in protocol.TOPOLOGIES
    for coherence in protocol.COHERENCES
    for semantic in (False, True)
}


def permutation_relations(reference: np.ndarray) -> np.ndarray:
    gram = reference @ reference.T
    return np.stack([
        relation_from_gram(gram[np.ix_(indices, indices)])
        for indices in (np.asarray(value, dtype=np.int64) for value in PERMUTATIONS)
    ])


REFERENCE_PERMUTATIONS = {
    key: permutation_relations(value) for key, value in REFERENCES.items()
}


def graph_fit(values: np.ndarray, key: tuple[str, str, bool], *, exact: bool = True) -> dict[str, Any]:
    observed = relation(values)
    target = relation(REFERENCES[key])
    score = float(np.dot(observed, target))
    result = {"cosine": score}
    if exact:
        scores = REFERENCE_PERMUTATIONS[key] @ observed
        result.update({
            "exact_upper_tail_p": float(np.mean(scores >= score - 1e-12)),
            "permutation_count": int(scores.shape[0]),
        })
    return result


def identity_record(source: np.ndarray, target: np.ndarray, labels: tuple[str, ...]) -> dict[str, Any]:
    return common.exact_assignment(source, target, labels)


def identity_pass(content: dict[str, Any], null: dict[str, Any]) -> tuple[bool, float]:
    advantage = float(content["identity_mean_score"] - null["identity_mean_score"])
    threshold = protocol.EVIDENCE_THRESHOLDS
    passed = (
        int(content["top1_correct"]) >= int(threshold["minimum_edge_top1"])
        and float(content["exact_upper_tail_p"]) <= float(threshold["permutation_p_max"])
        and int(content["top1_correct"]) > int(null["top1_correct"])
        and advantage >= float(threshold["minimum_content_identity_advantage"])
    )
    return passed, advantage


def hidden_audit(models: dict[str, dict[str, Any]], behavior: dict[str, Any]) -> dict[str, Any]:
    threshold = protocol.EVIDENCE_THRESHOLDS
    by_model = {}
    passing = []
    for model_name, data in models.items():
        summary = data["summary"]
        projection_rows = summary["projection_audit"]["replicates"]
        checks = {
            "precision_fp16_no_quantization": (
                summary["precision"]["has_fp16_parameters"]
                and not summary["precision"]["has_bf16_parameters"]
                and not summary["precision"]["has_quantized_modules"]
            ),
            "hidden_finite": float(summary["hidden_finite_fraction_lower_bound"])
            >= float(threshold["minimum_hidden_finite_fraction"]),
            "projection_median": all(
                float(row["median_abs_norm_error"])
                <= float(threshold["maximum_projection_median_abs_norm_error"])
                for row in projection_rows
            ),
            "projection_p95": all(
                float(row["p95_abs_norm_error"])
                <= float(threshold["maximum_projection_p95_abs_norm_error"])
                for row in projection_rows
            ),
            "pre_query_zero": float(summary["pre_query_global_max_abs"])
            <= float(threshold["pre_query_tolerance"]),
            "behavior_authorized": model_name in behavior["authorized_models"],
        }
        passed = all(checks.values())
        if passed:
            passing.append(model_name)
        by_model[model_name] = {
            "checks": checks,
            "passed": passed,
            "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
            "pre_query_global_max_abs": summary["pre_query_global_max_abs"],
            "projection_audit": summary["projection_audit"],
            "atlas_sha256": data["npz_sha256"],
        }
    return {
        "models": by_model,
        "passing_models": passing,
        "passed": len(passing) >= int(threshold["minimum_required_models"]),
    }


def edge_identity_analysis(models: dict[str, dict[str, Any]], behavior: dict[str, Any]) -> dict[str, Any]:
    by_model = {}
    passing_models = []
    for model_name, data in models.items():
        rows = []
        for topology in protocol.TOPOLOGIES:
            labels = operation_names(protocol.PRIMARY_ATTRIBUTE, topology, "coherent")
            for surface in protocol.SURFACES:
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    content = identity_record(
                        bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, "discovery", "content", replicate),
                        bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, "confirmation", "content", replicate),
                        labels,
                    )
                    null = identity_record(
                        bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, "discovery", "field_null", replicate),
                        bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, "confirmation", "field_null", replicate),
                        labels,
                    )
                    passed, advantage = identity_pass(content, null)
                    rows.append({
                        "topology": topology,
                        "surface": surface,
                        "replicate": replicate,
                        "content": content,
                        "field_null": null,
                        "content_over_null_advantage": advantage,
                        "passed": passed,
                    })
        formal = model_name in behavior["authorized_models"]
        passed_count = sum(int(row["passed"]) for row in rows)
        passed = formal and passed_count >= int(protocol.EVIDENCE_THRESHOLDS["minimum_required_cells"])
        if passed:
            passing_models.append(model_name)
        by_model[model_name] = {
            "formal": formal,
            "passing_count": passed_count,
            "row_count": len(rows),
            "rows": rows,
            "passed": passed,
        }
    return {
        "models": by_model,
        "passing_models": passing_models,
        "passed": len(passing_models) >= int(protocol.EVIDENCE_THRESHOLDS["minimum_required_models"]),
    }


def semantic_topology_analysis(models: dict[str, dict[str, Any]], behavior: dict[str, Any]) -> dict[str, Any]:
    threshold = protocol.EVIDENCE_THRESHOLDS
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for topology in protocol.TOPOLOGIES:
                nominal_key = (topology, "coherent", False)
                scrambled_slot_key = (topology, "scrambled", False)
                scrambled_semantic_key = (topology, "scrambled", True)
                for surface in protocol.SURFACES:
                    for split in protocol.SPLITS:
                        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                            coherent = bank(data, attribute, topology, "coherent", surface, split, "content", replicate)
                            scrambled = bank(data, attribute, topology, "scrambled", surface, split, "content", replicate)
                            coherent_null = bank(data, attribute, topology, "coherent", surface, split, "field_null", replicate)
                            scrambled_null = bank(data, attribute, topology, "scrambled", surface, split, "field_null", replicate)
                            coherent_fit = graph_fit(coherent, nominal_key)
                            coherent_null_fit = graph_fit(coherent_null, nominal_key)
                            scrambled_slot_fit = graph_fit(scrambled, scrambled_slot_key)
                            scrambled_semantic_fit = graph_fit(scrambled, scrambled_semantic_key)
                            scrambled_null_semantic_fit = graph_fit(scrambled_null, scrambled_semantic_key)
                            alias_advantage = float(coherent_fit["cosine"] - scrambled_slot_fit["cosine"])
                            coherent_null_advantage = float(coherent_fit["cosine"] - coherent_null_fit["cosine"])
                            semantic_over_slot = float(scrambled_semantic_fit["cosine"] - scrambled_slot_fit["cosine"])
                            semantic_over_null = float(scrambled_semantic_fit["cosine"] - scrambled_null_semantic_fit["cosine"])
                            p5_passed = (
                                coherent_fit["cosine"] >= float(threshold["minimum_incidence_fit"])
                                and coherent_null_advantage >= float(threshold["minimum_content_over_null_fit_advantage"])
                                and alias_advantage >= float(threshold["minimum_coherent_over_scrambled_advantage"])
                                and coherent_fit["exact_upper_tail_p"] <= float(threshold["permutation_p_max"])
                            )
                            p6_passed = (
                                scrambled_semantic_fit["cosine"] >= float(threshold["minimum_incidence_fit"])
                                and semantic_over_slot >= float(threshold["minimum_true_over_slot_scrambled_advantage"])
                                and semantic_over_null >= float(threshold["minimum_content_over_null_fit_advantage"])
                                and scrambled_semantic_fit["exact_upper_tail_p"] <= float(threshold["permutation_p_max"])
                            )
                            rows.append({
                                "topology": topology,
                                "surface": surface,
                                "split": split,
                                "replicate": replicate,
                                "coherent_slot_fit": coherent_fit,
                                "coherent_null_slot_fit": coherent_null_fit,
                                "scrambled_slot_fit": scrambled_slot_fit,
                                "scrambled_semantic_fit": scrambled_semantic_fit,
                                "scrambled_null_semantic_fit": scrambled_null_semantic_fit,
                                "coherent_over_scrambled_slot_advantage": alias_advantage,
                                "coherent_content_over_null_advantage": coherent_null_advantage,
                                "scrambled_semantic_over_slot_advantage": semantic_over_slot,
                                "scrambled_semantic_over_null_advantage": semantic_over_null,
                                "P5_passed": p5_passed,
                                "P6_passed": p6_passed,
                            })
            formal = model_name in behavior["authorized_models"]
            topology_summary = {}
            for topology in protocol.TOPOLOGIES:
                current = [row for row in rows if row["topology"] == topology]
                surface_summary = {}
                for surface in protocol.SURFACES:
                    local = [row for row in current if row["surface"] == surface]
                    surface_summary[surface] = {
                        "P5_passing": sum(int(row["P5_passed"]) for row in local),
                        "P6_passing": sum(int(row["P6_passed"]) for row in local),
                        "row_count": len(local),
                        "P5_passed": sum(int(row["P5_passed"]) for row in local) >= 3,
                        "P6_passed": sum(int(row["P6_passed"]) for row in local) >= 3,
                    }
                p5_count = sum(int(row["P5_passed"]) for row in current)
                p6_count = sum(int(row["P6_passed"]) for row in current)
                topology_summary[topology] = {
                    "P5_passing": p5_count,
                    "P6_passing": p6_count,
                    "row_count": len(current),
                    "surface_summary": surface_summary,
                    "P5_passed": p5_count >= 6 and all(v["P5_passed"] for v in surface_summary.values()),
                    "P6_passed": p6_count >= 6 and all(v["P6_passed"] for v in surface_summary.values()),
                    "mean_coherent_fit": float(np.mean([row["coherent_slot_fit"]["cosine"] for row in current])),
                    "mean_scrambled_slot_fit": float(np.mean([row["scrambled_slot_fit"]["cosine"] for row in current])),
                    "mean_scrambled_semantic_fit": float(np.mean([row["scrambled_semantic_fit"]["cosine"] for row in current])),
                    "mean_alias_advantage": float(np.mean([row["coherent_over_scrambled_slot_advantage"] for row in current])),
                    "mean_semantic_over_slot": float(np.mean([row["scrambled_semantic_over_slot_advantage"] for row in current])),
                }
            p5_passed = formal and all(value["P5_passed"] for value in topology_summary.values())
            p6_passed = formal and all(value["P6_passed"] for value in topology_summary.values())
            attributes[attribute] = {
                "formal": formal,
                "topologies": topology_summary,
                "rows": rows,
                "P5_passed": p5_passed,
                "P6_passed": p6_passed,
            }
        by_model[model_name] = {"attributes": attributes}
    return {"models": by_model}


def cross_model_analysis(
    models: dict[str, dict[str, Any]],
    behavior: dict[str, Any],
) -> dict[str, Any]:
    threshold = protocol.EVIDENCE_THRESHOLDS
    rows = []
    authorized = tuple(behavior["authorized_models"])
    for source_name in authorized:
        for target_name in authorized:
            if source_name == target_name:
                continue
            source = models[source_name]
            target = models[target_name]
            pair_rows = []
            for topology in protocol.TOPOLOGIES:
                for surface in protocol.SURFACES:
                    for split in protocol.SPLITS:
                        for source_replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                            for target_replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                                content_source = bank(source, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, split, "content", source_replicate)
                                content_target = bank(target, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, split, "content", target_replicate)
                                null_source = bank(source, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, split, "field_null", source_replicate)
                                null_target = bank(target, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, split, "field_null", target_replicate)
                                content = float(np.dot(relation(content_source), relation(content_target)))
                                null = float(np.dot(relation(null_source), relation(null_target)))
                                advantage = content - null
                                pair_rows.append({
                                    "topology": topology,
                                    "surface": surface,
                                    "split": split,
                                    "source_replicate": source_replicate,
                                    "target_replicate": target_replicate,
                                    "content_gram_cosine": content,
                                    "field_null_gram_cosine": null,
                                    "content_over_null_advantage": advantage,
                                    "passed": content >= 0.50 and advantage >= 0.10,
                                })
            passing = sum(int(row["passed"]) for row in pair_rows)
            rows.append({
                "source_model": source_name,
                "target_model": target_name,
                "passing_count": passing,
                "row_count": len(pair_rows),
                "mean_content_gram_cosine": float(np.mean([row["content_gram_cosine"] for row in pair_rows])),
                "mean_content_over_null_advantage": float(np.mean([row["content_over_null_advantage"] for row in pair_rows])),
                "rows": pair_rows,
                "passed": passing >= math.ceil(0.75 * len(pair_rows)),
            })
    passing_pairs = [f"{row['source_model']}->{row['target_model']}" for row in rows if row["passed"]]
    return {
        "authorized_models": list(authorized),
        "directed_pairs": rows,
        "passing_directed_pairs": passing_pairs,
        "passed": len(passing_pairs) >= 2 and len(authorized) >= int(threshold["minimum_required_models"]),
    }


def residual_relation(values: np.ndarray, topology: str) -> np.ndarray:
    observed = relation(values)
    incidence = relation(REFERENCES[(topology, "coherent", True)])
    return unit(observed - float(np.dot(observed, incidence)) * incidence)


def residual_family_analysis(models: dict[str, dict[str, Any]], behavior: dict[str, Any]) -> dict[str, Any]:
    by_model = {}
    passing_models = []
    for model_name, data in models.items():
        rows = []
        for topology in protocol.TOPOLOGIES:
            for surface in protocol.SURFACES:
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    content_source = []
                    content_target = []
                    null_source = []
                    null_target = []
                    for attribute in protocol.ATTRIBUTES:
                        content_source.append(residual_relation(bank(data, attribute, topology, "coherent", surface, "discovery", "content", replicate), topology))
                        content_target.append(residual_relation(bank(data, attribute, topology, "coherent", surface, "confirmation", "content", replicate), topology))
                        null_source.append(residual_relation(bank(data, attribute, topology, "coherent", surface, "discovery", "field_null", replicate), topology))
                        null_target.append(residual_relation(bank(data, attribute, topology, "coherent", surface, "confirmation", "field_null", replicate), topology))
                    content_matrix = np.stack(content_source) @ np.stack(content_target).T
                    null_matrix = np.stack(null_source) @ np.stack(null_target).T
                    content_advantage = float(np.diag(content_matrix).mean() - content_matrix[[0, 1], [1, 0]].mean())
                    null_advantage = float(np.diag(null_matrix).mean() - null_matrix[[0, 1], [1, 0]].mean())
                    over_null = content_advantage - null_advantage
                    rows.append({
                        "topology": topology,
                        "surface": surface,
                        "replicate": replicate,
                        "content_similarity_matrix": content_matrix.tolist(),
                        "field_null_similarity_matrix": null_matrix.tolist(),
                        "content_family_advantage": content_advantage,
                        "field_null_family_advantage": null_advantage,
                        "content_over_null_advantage": over_null,
                        "passed": content_advantage >= 0.10 and over_null >= float(protocol.EVIDENCE_THRESHOLDS["minimum_residual_family_advantage"]),
                    })
        formal = model_name in behavior["authorized_models"]
        passing = sum(int(row["passed"]) for row in rows)
        passed = formal and passing >= int(protocol.EVIDENCE_THRESHOLDS["minimum_required_cells"])
        if passed:
            passing_models.append(model_name)
        by_model[model_name] = {
            "formal": formal,
            "passing_count": passing,
            "row_count": len(rows),
            "rows": rows,
            "passed": passed,
        }
    return {
        "models": by_model,
        "passing_models": passing_models,
        "passed": len(passing_models) >= int(protocol.EVIDENCE_THRESHOLDS["minimum_required_models"]),
    }


def physical_map(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model_name, data in models.items():
        events = data["summary"]["events"]
        rows = []
        for event_index, event in enumerate(events):
            for role in protocol.CAPTURE_ROLES:
                alias_scores = []
                true_slot_scores = []
                null_scores = []
                for topology in protocol.TOPOLOGIES:
                    nominal_key = (topology, "coherent", False)
                    semantic_key = (topology, "scrambled", True)
                    slot_key = (topology, "scrambled", False)
                    for surface in protocol.SURFACES:
                        for split in protocol.SPLITS:
                            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                                coherent = bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, split, "content", replicate, role=role, event_index=event_index)
                                scrambled = bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "scrambled", surface, split, "content", replicate, role=role, event_index=event_index)
                                coherent_null = bank(data, protocol.PRIMARY_ATTRIBUTE, topology, "coherent", surface, split, "field_null", replicate, role=role, event_index=event_index)
                                coherent_fit = graph_fit(coherent, nominal_key, exact=False)["cosine"]
                                alias_scores.append(coherent_fit - graph_fit(scrambled, slot_key, exact=False)["cosine"])
                                true_slot_scores.append(graph_fit(scrambled, semantic_key, exact=False)["cosine"] - graph_fit(scrambled, slot_key, exact=False)["cosine"])
                                null_scores.append(coherent_fit - graph_fit(coherent_null, nominal_key, exact=False)["cosine"])
                rows.append({
                    "event_index": event_index,
                    "event": event,
                    "role": role,
                    "mean_coherent_over_scrambled_slot": float(np.mean(alias_scores)),
                    "mean_scrambled_semantic_over_slot": float(np.mean(true_slot_scores)),
                    "mean_coherent_content_over_null": float(np.mean(null_scores)),
                    "combined_descriptive_score": float(np.mean(alias_scores) + np.mean(true_slot_scores) + np.mean(null_scores)),
                })
        ranked = sorted(rows, key=lambda row: row["combined_descriptive_score"], reverse=True)
        by_model[model_name] = {
            "top20": ranked[:20],
            "event_role_count": len(rows),
            "promoted": False,
        }
    return {
        "models": by_model,
        "interpretation": "Post-analysis ranking only. Promotion requires all semantic gates and independent replication.",
    }


def compact_semantic(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        model: {
            attribute: {
                "P5_passed": row["P5_passed"],
                "P6_passed": row["P6_passed"],
                "topologies": {
                    topology: {
                        key: value[key]
                        for key in (
                            "P5_passing", "P6_passing", "row_count", "P5_passed", "P6_passed",
                            "mean_coherent_fit", "mean_scrambled_slot_fit", "mean_scrambled_semantic_fit",
                            "mean_alias_advantage", "mean_semantic_over_slot",
                        )
                    }
                    for topology, value in row["topologies"].items()
                },
            }
            for attribute, row in model_row["attributes"].items()
        }
        for model, model_row in analysis["models"].items()
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    if not behavior["hidden_scan_authorized"]:
        raise RuntimeError("Phase1094 hidden scan was not authorized")
    models = load_models()
    audits = hidden_audit(models, behavior)
    identity = edge_identity_analysis(models, behavior)
    semantic = semantic_topology_analysis(models, behavior)
    cross_model = cross_model_analysis(models, behavior)
    residual = residual_family_analysis(models, behavior)
    physical = physical_map(models)

    authorized = set(behavior["authorized_models"])
    p5_models = [
        model for model in authorized
        if semantic["models"][model]["attributes"][protocol.PRIMARY_ATTRIBUTE]["P5_passed"]
    ]
    p6_models = [
        model for model in authorized
        if semantic["models"][model]["attributes"][protocol.PRIMARY_ATTRIBUTE]["P6_passed"]
    ]
    p7_models = sorted(set(p5_models) & set(p6_models))
    p8_models = [
        model for model in authorized
        if semantic["models"][model]["attributes"][protocol.SECONDARY_ATTRIBUTE]["P5_passed"]
        and semantic["models"][model]["attributes"][protocol.SECONDARY_ATTRIBUTE]["P6_passed"]
    ]
    minimum_models = int(protocol.EVIDENCE_THRESHOLDS["minimum_required_models"])
    predictions = {
        "P1": {"passed": bool(protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")["all_checks_passed"])},
        "P2": {"passed": bool(behavior["hidden_scan_authorized"]), "authorized_models": sorted(authorized)},
        "P3": {"passed": audits["passed"], "passing_models": audits["passing_models"]},
        "P4": {"passed": identity["passed"], "passing_models": identity["passing_models"]},
        "P5": {"passed": len(p5_models) >= minimum_models, "passing_models": p5_models},
        "P6": {"passed": len(p6_models) >= minimum_models, "passing_models": p6_models},
        "P7": {"passed": len(p7_models) >= minimum_models, "passing_models": p7_models},
        "P8": {"passed": len(p8_models) >= minimum_models, "passing_models": p8_models},
        "P9": {"passed": cross_model["passed"], "passing_directed_pairs": cross_model["passing_directed_pairs"]},
        "P10": {"passed": residual["passed"], "passing_models": residual["passing_models"]},
    }
    predictions["P11"] = {
        "passed": all(predictions[f"P{index}"]["passed"] for index in range(1, 11)),
        "criterion": "all discovery gates pass before physical promotion",
    }
    semantic_candidate = predictions["P5"]["passed"] or predictions["P6"]["passed"]
    decision = (
        "continue_phase1095_independent_semantic_replication_without_causal_localization"
        if semantic_candidate
        else "retain_generic_directed_binding_skeleton_and_redesign_semantic_alias_control"
    )
    result = {
        "schema_version": "phase1094_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_digest": behavior["summary_digest"],
        "predictions": predictions,
        "hidden_audit": audits,
        "edge_identity": identity,
        "semantic_topology": semantic,
        "semantic_topology_compact": compact_semantic(semantic),
        "cross_model": cross_model,
        "incidence_residual_family": residual,
        "physical_map": physical,
        "decision": decision,
        "automatic_next_required": True,
        "automatic_next_reason": (
            "A semantic candidate requires independent replication."
            if semantic_candidate
            else "The semantic/topology ambiguity remains and requires a different natural control."
        ),
        "causal_authorized": False,
        "theory_status": {
            "generic_directed_binding_skeleton": "retained from Phase1093",
            "relative_synonym_node_reuse": "supported only if P5 or P6 passes in at least two models",
            "complete_concept_code": "not established",
            "new_mathematics_required": False,
        },
        "hard_limits": [
            "Researcher-defined synonyms and color shades are imperfect semantic equivalences.",
            "Subword overlap remains possible despite exact alias-string disjointness.",
            "Graph fits describe relation geometry and do not identify a causal circuit.",
            "Field-null retains a strong task route and can share incidence structure.",
            "Only two semantic families and controlled binary judgments are tested.",
            "No result here closes rare words, punctuation, translation, contrast, grammar, or a global knowledge network.",
        ],
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print({
        "phase": protocol.PHASE,
        "predictions": {key: value["passed"] for key, value in predictions.items()},
        "semantic_compact": result["semantic_topology_compact"],
        "decision": decision,
        "summary_digest": result["summary_digest"],
    })


if __name__ == "__main__":
    main()
