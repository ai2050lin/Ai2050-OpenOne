#!/usr/bin/env python3
"""Finalize the frozen Phase1123 terminal residual geometry test."""

from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1123_adjective_terminal_hidden_protocol as protocol


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm) or left_norm <= 1.0e-8 or right_norm <= 1.0e-8:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None


def load_artifact(model_name: str, prereg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    root = protocol.OUT_ROOT / "hidden" / model_name
    summary = protocol.read_json(root / "summary.json")
    cases = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    if summary["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError(f"protocol digest mismatch for {model_name}")
    if protocol.digest(cases) != prereg["case_digests"][model_name]:
        raise RuntimeError(f"case digest mismatch for {model_name}")
    artifact_path = protocol.OUT_ROOT / summary["artifact"]
    if protocol.file_sha256(artifact_path) != summary["artifact_sha256"]:
        raise RuntimeError(f"artifact digest mismatch for {model_name}")
    with np.load(artifact_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    spec = prereg["model_specs"][model_name]
    expected = {
        "case_indices": (prereg["case_count_per_model"],),
        "projected_states": (
            prereg["case_count_per_model"],
            spec["hidden_state_count"],
            len(prereg["roles"]),
            prereg["projection_dimension"],
        ),
        "output_z": (prereg["case_count_per_model"],),
        "source_z": (prereg["case_count_per_model"],),
    }
    if set(arrays) != set(expected):
        raise RuntimeError(f"unexpected artifact arrays for {model_name}")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"artifact shape mismatch for {model_name}/{name}: {arrays[name].shape}")
    if not np.array_equal(arrays["case_indices"], np.arange(prereg["case_count_per_model"], dtype=np.int32)):
        raise RuntimeError(f"case order mismatch for {model_name}")
    return summary, arrays, cases


def factorial_rows(arrays: dict[str, np.ndarray], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[row["interaction_id"]].append(row)
    states = arrays["projected_states"].astype(np.float32)
    output: list[dict[str, Any]] = []
    for interaction_id, panel in sorted(grouped.items()):
        by_cell = {(int(row["context_sense"]), int(row["definition_sense"])): row for row in panel}
        if set(by_cell) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise RuntimeError(f"malformed factorial panel: {interaction_id}")
        h00 = states[int(by_cell[(0, 0)]["case_index"])]
        h01 = states[int(by_cell[(0, 1)]["case_index"])]
        h10 = states[int(by_cell[(1, 0)]["case_index"])]
        h11 = states[int(by_cell[(1, 1)]["case_index"])]
        first = panel[0]
        output.append({
            "interaction_id": interaction_id,
            "concept_id": first["concept_id"],
            "control_concept_id": first["deranged_control_concept_id"],
            "split": first["split"],
            "template": int(first["template"]),
            "surface": first["surface"],
            "C": 0.5 * ((h00 + h01) - (h10 + h11)),
            "D": 0.5 * ((h00 + h10) - (h01 + h11)),
            "I": 0.5 * ((h00 + h11) - (h01 + h10)),
        })
    if len(output) != 288:
        raise RuntimeError(f"unexpected factorial row count: {len(output)}")
    return output


def cosine_panel(left_rows: list[np.ndarray], right_rows: list[np.ndarray]) -> list[float]:
    values: list[float] = []
    for left, right in zip(left_rows, right_rows):
        value = cosine(left, right)
        if value is not None:
            values.append(value)
    return values


def shared_energy(vectors: list[np.ndarray]) -> float | None:
    units = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        if math.isfinite(norm) and norm > 1.0e-8:
            units.append(vector / norm)
    if not units:
        return None
    mean = np.mean(np.stack(units), axis=0)
    return float(np.dot(mean, mean))


def panel_metrics(
    factors: list[dict[str, Any]],
    panel_name: str,
    layer_index: int,
    role_index: int,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    panel_spec = prereg["panels"][panel_name]
    selected = [
        row for row in factors
        if row["split"] == panel_spec["split"] and row["template"] in panel_spec["templates"]
    ]
    lookup = {(row["concept_id"], row["template"], row["surface"]): row for row in selected}
    if len(selected) != 32:
        raise RuntimeError(f"unexpected panel size for {panel_name}: {len(selected)}")

    cd_same_left: list[np.ndarray] = []
    cd_same_right: list[np.ndarray] = []
    cd_null_right: list[np.ndarray] = []
    for row in selected:
        control = lookup[(row["control_concept_id"], row["template"], row["surface"])]
        cd_same_left.append(row["C"][layer_index, role_index])
        cd_same_right.append(row["D"][layer_index, role_index])
        cd_null_right.append(control["D"][layer_index, role_index])
    cd_same = cosine_panel(cd_same_left, cd_same_right)
    cd_null = cosine_panel(cd_same_left, cd_null_right)

    surface_same_left: list[np.ndarray] = []
    surface_same_right: list[np.ndarray] = []
    surface_null_right: list[np.ndarray] = []
    concepts = sorted({row["concept_id"] for row in selected})
    for concept_id in concepts:
        for template in panel_spec["templates"]:
            base = lookup[(concept_id, template, "base")]
            synonym = lookup[(concept_id, template, "synonym")]
            control = lookup[(base["control_concept_id"], template, "synonym")]
            surface_same_left.append(base["C"][layer_index, role_index])
            surface_same_right.append(synonym["C"][layer_index, role_index])
            surface_null_right.append(control["C"][layer_index, role_index])
    surface_same = cosine_panel(surface_same_left, surface_same_right)
    surface_null = cosine_panel(surface_same_left, surface_null_right)

    template_same_left: list[np.ndarray] = []
    template_same_right: list[np.ndarray] = []
    template_null_right: list[np.ndarray] = []
    template0, template1 = panel_spec["templates"]
    for concept_id in concepts:
        for surface in ("base", "synonym"):
            left = lookup[(concept_id, template0, surface)]
            right = lookup[(concept_id, template1, surface)]
            control = lookup[(left["control_concept_id"], template1, surface)]
            template_same_left.append(left["C"][layer_index, role_index])
            template_same_right.append(right["C"][layer_index, role_index])
            template_null_right.append(control["C"][layer_index, role_index])
    template_same = cosine_panel(template_same_left, template_same_right)
    template_null = cosine_panel(template_same_left, template_null_right)

    cd_same_median, cd_null_median = median(cd_same), median(cd_null)
    surface_same_median, surface_null_median = median(surface_same), median(surface_null)
    template_same_median, template_null_median = median(template_same), median(template_null)
    cd_advantage = cd_same_median - cd_null_median if cd_same_median is not None and cd_null_median is not None else None
    surface_advantage = surface_same_median - surface_null_median if surface_same_median is not None and surface_null_median is not None else None
    template_advantage = template_same_median - template_null_median if template_same_median is not None and template_null_median is not None else None
    advantages = [cd_advantage, surface_advantage, template_advantage]
    semantic_score = min(advantages) if all(value is not None for value in advantages) else None

    all_c = [row["C"][layer_index, role_index] for row in selected]
    all_d = [row["D"][layer_index, role_index] for row in selected]
    all_i = [row["I"][layer_index, role_index] for row in selected]
    return {
        "panel": panel_name,
        "factorial_row_count": len(selected),
        "context_definition": {
            "same_count": len(cd_same),
            "null_count": len(cd_null),
            "median_same_cosine": cd_same_median,
            "median_deranged_cosine": cd_null_median,
            "advantage": cd_advantage,
        },
        "cross_surface_context": {
            "same_count": len(surface_same),
            "null_count": len(surface_null),
            "median_same_cosine": surface_same_median,
            "median_deranged_cosine": surface_null_median,
            "advantage": surface_advantage,
        },
        "cross_template_context": {
            "same_count": len(template_same),
            "null_count": len(template_null),
            "median_same_cosine": template_same_median,
            "median_deranged_cosine": template_null_median,
            "advantage": template_advantage,
        },
        "semantic_score": semantic_score,
        "median_norms": {
            "C": median([float(np.linalg.norm(vector)) for vector in all_c]),
            "D": median([float(np.linalg.norm(vector)) for vector in all_d]),
            "I": median([float(np.linalg.norm(vector)) for vector in all_i]),
        },
        "signed_shared_energy": {
            "C": shared_energy(all_c),
            "D": shared_energy(all_d),
            "I": shared_energy(all_i),
        },
    }


def select_layer(layer_metrics: dict[str, Any], eligible: list[int]) -> dict[str, Any]:
    candidates = []
    for layer_index in eligible:
        metric = layer_metrics[str(layer_index)]["discovery"]["answer_boundary"]
        score = metric["semantic_score"] if metric["semantic_score"] is not None else -math.inf
        candidates.append({"layer_index": layer_index, "normalized_depth": layer_metrics[str(layer_index)]["normalized_depth"], "score": score})
    selected = sorted(candidates, key=lambda row: (-row["score"], row["layer_index"]))[0]
    return {"selected": selected, "candidates": candidates}


def confirmation_gate(selected: dict[str, Any], embedding: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    branches = (
        selected["context_definition"],
        selected["cross_surface_context"],
        selected["cross_template_context"],
    )
    same_pass = all(branch["median_same_cosine"] is not None and branch["median_same_cosine"] >= thresholds["minimum_same_cosine"] for branch in branches)
    advantage_pass = all(branch["advantage"] is not None and branch["advantage"] >= thresholds["minimum_matched_advantage"] for branch in branches)
    score = selected["semantic_score"]
    embedding_score = embedding["semantic_score"]
    gain = score - embedding_score if score is not None and embedding_score is not None else None
    score_pass = score is not None and score >= thresholds["minimum_semantic_score"]
    gain_pass = gain is not None and gain >= thresholds["minimum_gain_over_embedding"]
    return {
        "same_cosines_pass": same_pass,
        "matched_advantages_pass": advantage_pass,
        "semantic_score": score,
        "embedding_semantic_score": embedding_score,
        "gain_over_embedding": gain,
        "semantic_score_pass": score_pass,
        "gain_over_embedding_pass": gain_pass,
        "passed": same_pass and advantage_pass and score_pass and gain_pass,
    }


def model_result(
    model_name: str,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    factors: list[dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    hidden_count = prereg["model_specs"][model_name]["hidden_state_count"]
    role_lookup = {role: index for index, role in enumerate(prereg["roles"])}
    layer_metrics: dict[str, Any] = {}
    for layer_index in range(hidden_count):
        layer_metrics[str(layer_index)] = {
            "normalized_depth": layer_index / max(hidden_count - 1, 1),
            **{
                panel_name: {
                    role: panel_metrics(factors, panel_name, layer_index, role_index, prereg)
                    for role, role_index in role_lookup.items()
                }
                for panel_name in prereg["panels"]
            },
        }
    selection = select_layer(layer_metrics, prereg["model_specs"][model_name]["eligible_hidden_state_indices"])
    selected_layer = int(selection["selected"]["layer_index"])
    gates = {}
    for panel_name in ("independent_confirmation", "heldout"):
        selected_metric = layer_metrics[str(selected_layer)][panel_name]["answer_boundary"]
        embedding_metric = layer_metrics["0"][panel_name]["answer_boundary"]
        gates[panel_name] = confirmation_gate(selected_metric, embedding_metric, prereg["thresholds"])

    context_role = role_lookup["context_end"]
    answer_role = role_lookup["answer_boundary"]
    context_d = []
    context_i = []
    answer_d = []
    answer_i = []
    for layer_index in range(hidden_count):
        for row in factors:
            context_d.append(float(np.linalg.norm(row["D"][layer_index, context_role])))
            context_i.append(float(np.linalg.norm(row["I"][layer_index, context_role])))
            answer_d.append(float(np.linalg.norm(row["D"][layer_index, answer_role])))
            answer_i.append(float(np.linalg.norm(row["I"][layer_index, answer_role])))
    if not all(math.isfinite(value) for value in [*context_d, *context_i, *answer_d, *answer_i]):
        d_leak_ratio = None
        i_leak_ratio = None
    else:
        d_leak_ratio = max(context_d) / max(max(answer_d), 1.0e-12)
        i_leak_ratio = max(context_i) / max(max(answer_i), 1.0e-12)
    thresholds = prereg["thresholds"]
    instrument = {
        "finite_fraction": summary["finite_fraction"],
        "finite_pass": summary["finite_fraction"] >= thresholds["minimum_finite_fraction"],
        "maximum_behavior_z_reproduction_error": summary["maximum_behavior_z_reproduction_error"],
        "behavior_reproduction_pass": summary["maximum_behavior_z_reproduction_error"] <= thresholds["maximum_behavior_z_reproduction_error"],
        "context_end_D_leak_ratio": d_leak_ratio,
        "context_end_I_leak_ratio": i_leak_ratio,
        "causal_role_pass": (
            d_leak_ratio is not None
            and i_leak_ratio is not None
            and max(d_leak_ratio, i_leak_ratio) <= thresholds["maximum_context_end_definition_leak_ratio"]
        ),
    }
    qualified = all(instrument[key] for key in ("finite_pass", "behavior_reproduction_pass", "causal_role_pass")) and all(gate["passed"] for gate in gates.values())
    return {
        "model": model_name,
        "instrument": instrument,
        "selection": selection,
        "selected_layer": selected_layer,
        "selected_normalized_depth": selection["selected"]["normalized_depth"],
        "confirmation_gates": gates,
        "selected_role_metrics": {
            panel_name: {
                role: layer_metrics[str(selected_layer)][panel_name][role]
                for role in prereg["roles"]
            }
            for panel_name in prereg["panels"]
        },
        "qualified": qualified,
        "summary_digest": summary["summary_digest"],
        "layer_metrics": layer_metrics,
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if math.isfinite(norm) and norm > 1.0e-8 else np.zeros_like(vector)


def concept_gram(
    factors: list[dict[str, Any]],
    panel_name: str,
    layer_index: int,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    spec = prereg["panels"][panel_name]
    rows = [row for row in factors if row["split"] == spec["split"] and row["template"] in spec["templates"]]
    concepts = sorted({row["concept_id"] for row in rows})
    vectors_c: list[np.ndarray] = []
    vectors_d: list[np.ndarray] = []
    control_map: dict[str, str] = {}
    answer_role = prereg["roles"].index("answer_boundary")
    for concept_id in concepts:
        panel = [row for row in rows if row["concept_id"] == concept_id]
        vectors_c.append(normalize(np.mean(np.stack([row["C"][layer_index, answer_role] for row in panel]), axis=0)))
        vectors_d.append(normalize(np.mean(np.stack([row["D"][layer_index, answer_role] for row in panel]), axis=0)))
        control_map[concept_id] = panel[0]["control_concept_id"]
    vectors = np.stack([*vectors_c, *vectors_d])
    gram = vectors @ vectors.T
    index = {concept_id: slot for slot, concept_id in enumerate(concepts)}
    permutation = [index[control_map[concept_id]] for concept_id in concepts]
    permuted = vectors[np.array([*permutation, *(slot + len(concepts) for slot in permutation)], dtype=np.int64)]
    null_gram = permuted @ permuted.T
    return {"concepts": concepts, "gram": gram, "null_gram": null_gram}


def centered_upper(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    vector = matrix[indices].astype(np.float64)
    return vector - np.mean(vector)


def cross_model_metrics(
    factors_by_model: dict[str, list[dict[str, Any]]],
    model_results: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    thresholds = prereg["thresholds"]
    for left, right in itertools.combinations(prereg["models"], 2):
        panel_results = {}
        pair_pass = True
        for panel_name in ("independent_confirmation", "heldout"):
            left_gram = concept_gram(factors_by_model[left], panel_name, model_results[left]["selected_layer"], prereg)
            right_gram = concept_gram(factors_by_model[right], panel_name, model_results[right]["selected_layer"], prereg)
            if left_gram["concepts"] != right_gram["concepts"]:
                raise RuntimeError("cross-model concept ordering mismatch")
            left_vector = centered_upper(left_gram["gram"])
            right_vector = centered_upper(right_gram["gram"])
            right_null_vector = centered_upper(right_gram["null_gram"])
            true_cosine = cosine(left_vector, right_vector)
            null_cosine = cosine(left_vector, right_null_vector)
            advantage = true_cosine - null_cosine if true_cosine is not None and null_cosine is not None else None
            passed = (
                true_cosine is not None
                and advantage is not None
                and true_cosine >= thresholds["minimum_cross_model_gram_cosine"]
                and advantage >= thresholds["minimum_cross_model_gram_advantage"]
            )
            panel_results[panel_name] = {
                "true_gram_cosine": true_cosine,
                "deranged_gram_cosine": null_cosine,
                "advantage": advantage,
                "passed": passed,
            }
            pair_pass = pair_pass and passed
        output[f"{left}__{right}"] = {"models": [left, right], "panels": panel_results, "passed": pair_pass}
    return output


def compact_model_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "layer_metrics"}


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1123 protocol audit failed")

    summaries: dict[str, Any] = {}
    factors_by_model: dict[str, list[dict[str, Any]]] = {}
    model_results: dict[str, dict[str, Any]] = {}
    layer_payload: dict[str, Any] = {}
    for model_name in prereg["models"]:
        summary, arrays, cases = load_artifact(model_name, prereg)
        factors = factorial_rows(arrays, cases)
        result = model_result(model_name, summary, arrays, factors, prereg)
        summaries[model_name] = summary
        factors_by_model[model_name] = factors
        model_results[model_name] = result
        layer_payload[model_name] = result["layer_metrics"]

    cross_model = cross_model_metrics(factors_by_model, model_results, prereg)
    qualified_models = [model for model, result in model_results.items() if result["qualified"]]
    qualified_pairs = [
        name for name, result in cross_model.items()
        if result["passed"] and all(model_results[model]["qualified"] for model in result["models"])
    ]
    thresholds = prereg["thresholds"]
    instrument_qualified_models = [
        model for model, result in model_results.items()
        if result["instrument"]["finite_pass"]
        and result["instrument"]["behavior_reproduction_pass"]
        and result["instrument"]["causal_role_pass"]
    ]
    instrument_passed = len(instrument_qualified_models) >= thresholds["minimum_qualified_models"]
    terminal_hidden_passed = instrument_passed and len(qualified_models) >= thresholds["minimum_qualified_models"]
    cross_model_passed = len(qualified_pairs) >= thresholds["minimum_qualified_cross_model_pairs"]
    component_discovery_authorized = terminal_hidden_passed and cross_model_passed
    predictions = {
        "P1_source_and_protocol_integrity": "pass",
        "P2_instrument_and_causal_order": "pass" if instrument_passed else "fail",
        "P3_two_model_internal_confirmation": "pass" if terminal_hidden_passed else "fail",
        "P4_gain_over_embedding": "pass" if terminal_hidden_passed else "fail",
        "P5_cross_model_gram": "pass" if cross_model_passed else "fail",
        "P6_component_discovery_authorized": "pass" if component_discovery_authorized else "fail",
    }
    core = {
        "schema_version": "phase1123_adjective_terminal_hidden_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "models": {name: compact_model_result(result) for name, result in model_results.items()},
        "cross_model": cross_model,
        "qualified_models": qualified_models,
        "instrument_qualified_models": instrument_qualified_models,
        "qualified_cross_model_pairs": qualified_pairs,
        "instrument_passed": instrument_passed,
        "terminal_hidden_passed": terminal_hidden_passed,
        "cross_model_passed": cross_model_passed,
        "component_discovery_authorized": component_discovery_authorized,
        "predictions": predictions,
        "summary_digests": {name: summary["summary_digest"] for name, summary in summaries.items()},
        "interpretation": {
            "positive_limit": "A pass identifies a terminal residual relation between truth-balanced context-sense and definition-sense fields that repeats across surfaces, templates, concepts, and at least two models above embedding and deranged controls.",
            "negative_limit": "Failure constrains this residual projection, answer-boundary role, panels, and thresholds; it does not show that contextual semantics or a different dynamic representation is absent.",
            "not_claimed": [
                "training formation",
                "a shared physical coordinate across models",
                "attention, MLP, head, or neuron implementation",
                "natural continuation use",
                "causal necessity or sufficiency",
                "global language-code closure",
            ],
        },
        "automatic_continuation": {
            "separate_component_discovery_protocol_authorized": component_discovery_authorized,
            "run_component_or_causal_in_phase1123": False,
            "reason": "joint internal and cross-model gates passed" if component_discovery_authorized else "one or more frozen internal or cross-model gates failed",
        },
    }
    final = dict(core)
    final["final_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "layer_metrics.json", {"phase": protocol.PHASE, "metrics": layer_payload})
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    return final


def main() -> None:
    result = finalize()
    print(json.dumps({
        "phase": result["phase"],
        "models": result["models"],
        "cross_model": result["cross_model"],
        "qualified_models": result["qualified_models"],
        "qualified_cross_model_pairs": result["qualified_cross_model_pairs"],
        "predictions": result["predictions"],
        "automatic_continuation": result["automatic_continuation"],
        "final_digest": result["final_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
