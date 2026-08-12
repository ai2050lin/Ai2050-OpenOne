from __future__ import annotations

import gc
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


PHASE = 1124
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1124_adjective_cross_role_relational_geometry"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"
PERMUTATIONS = np.asarray(list(itertools.permutations(range(8))), dtype=np.int16)
TRIANGLE = np.triu_indices(8, k=1)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_preregistration(prereg: dict[str, Any]) -> None:
    expected = prereg["protocol_digest"]
    body = dict(prereg)
    del body["protocol_digest"]
    if canonical_digest(body) != expected:
        raise RuntimeError("Phase1124 protocol digest mismatch")
    for model, specs in prereg["source_files"].items():
        for name, spec in specs.items():
            path = ROOT / spec["path"]
            if not path.is_file() or file_sha256(path) != spec["sha256"]:
                raise RuntimeError(f"Phase1124 source file mismatch: {model}/{name}")


def safe_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector.astype(np.float64)))


def normalize_rows(matrix: np.ndarray) -> np.ndarray | None:
    matrix = matrix.astype(np.float64)
    if not np.isfinite(matrix).all():
        return None
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        return None
    return matrix / norms


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def factorial_rows(states: np.ndarray, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        groups.setdefault(row["interaction_id"], []).append(row)
    output: list[dict[str, Any]] = []
    for interaction_id, rows in groups.items():
        if len(rows) != 4:
            raise RuntimeError(f"Malformed factorial panel: {interaction_id}")
        grid = {(int(row["context_sense"]), int(row["definition_sense"])): states[int(row["case_index"])] for row in rows}
        if set(grid) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise RuntimeError(f"Incomplete factorial panel: {interaction_id}")
        h00, h01, h10, h11 = grid[(0, 0)], grid[(0, 1)], grid[(1, 0)], grid[(1, 1)]
        first = rows[0]
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
        raise RuntimeError(f"Unexpected factorial row count: {len(output)}")
    return output


def exact_permutation_scores(gram_c: np.ndarray, gram_d: np.ndarray) -> np.ndarray:
    c = gram_c[TRIANGLE].astype(np.float64)
    c = c - np.mean(c)
    c_norm = np.linalg.norm(c)
    if c_norm <= 1e-12:
        return np.full(len(PERMUTATIONS), np.nan, dtype=np.float64)
    rows = PERMUTATIONS[:, TRIANGLE[0]]
    cols = PERMUTATIONS[:, TRIANGLE[1]]
    d = gram_d[rows, cols].astype(np.float64)
    d = d - np.mean(d, axis=1, keepdims=True)
    d_norms = np.linalg.norm(d, axis=1)
    scores = np.full(len(PERMUTATIONS), np.nan, dtype=np.float64)
    valid = d_norms > 1e-12
    scores[valid] = (d[valid] @ c) / (d_norms[valid] * c_norm)
    return scores


def cell_metric(
    factors: list[dict[str, Any]],
    panel: dict[str, Any],
    layer_index: int,
    context_role: int,
    definition_role: int,
    template: int,
    surface: str,
    exact: bool,
) -> dict[str, Any]:
    rows = [
        row
        for row in factors
        if row["split"] == panel["split"] and row["template"] == template and row["surface"] == surface
    ]
    concepts = sorted({row["concept_id"] for row in rows})
    if len(rows) != 8 or len(concepts) != 8:
        raise RuntimeError(f"Unexpected cell size for template={template}, surface={surface}")
    lookup = {row["concept_id"]: row for row in rows}
    c_matrix = normalize_rows(np.stack([lookup[concept]["C"][layer_index, context_role] for concept in concepts]))
    d_matrix = normalize_rows(np.stack([lookup[concept]["D"][layer_index, definition_role] for concept in concepts]))
    if c_matrix is None or d_matrix is None:
        return {
            "concepts": concepts,
            "finite": False,
            "same_gram_cosine": None,
            "fixed_deranged_cosine": None,
            "fixed_derangement_advantage": None,
            "exact_permutation_percentile": None,
            "exact_null_median": None,
            "exact_null_maximum": None,
            "gram_c_vector": None,
            "gram_d_vector": None,
            "fixed_deranged_gram_d_vector": None,
        }

    gram_c = c_matrix @ c_matrix.T
    gram_d = d_matrix @ d_matrix.T
    c_vector = gram_c[TRIANGLE]
    d_vector = gram_d[TRIANGLE]
    same = centered_cosine(c_vector, d_vector)

    concept_index = {concept: index for index, concept in enumerate(concepts)}
    fixed_permutation = np.asarray(
        [concept_index[lookup[concept]["control_concept_id"]] for concept in concepts], dtype=np.int64
    )
    deranged_gram_d = gram_d[np.ix_(fixed_permutation, fixed_permutation)]
    deranged_vector = deranged_gram_d[TRIANGLE]
    deranged = centered_cosine(c_vector, deranged_vector)
    advantage = same - deranged if same is not None and deranged is not None else None

    percentile = None
    null_median = None
    null_maximum = None
    if exact:
        scores = exact_permutation_scores(gram_c, gram_d)
        finite_scores = scores[np.isfinite(scores)]
        if same is not None and finite_scores.size:
            percentile = float(np.mean(finite_scores <= same + 1e-12))
            null_median = float(np.median(finite_scores))
            null_maximum = float(np.max(finite_scores))

    return {
        "concepts": concepts,
        "finite": True,
        "same_gram_cosine": same,
        "fixed_deranged_cosine": deranged,
        "fixed_derangement_advantage": advantage,
        "exact_permutation_percentile": percentile,
        "exact_null_median": null_median,
        "exact_null_maximum": null_maximum,
        "gram_c_vector": [float(value) for value in c_vector],
        "gram_d_vector": [float(value) for value in d_vector],
        "fixed_deranged_gram_d_vector": [float(value) for value in deranged_vector],
    }


def panel_metric(
    factors: list[dict[str, Any]],
    panel_name: str,
    layer_index: int,
    context_role: int,
    definition_role: int,
    prereg: dict[str, Any],
    exact: bool,
) -> dict[str, Any]:
    panel = prereg["panels"][panel_name]
    cells: dict[str, Any] = {}
    for template in panel["templates"]:
        for surface in prereg["surfaces"]:
            key = f"template{template}.{surface}"
            cells[key] = cell_metric(
                factors, panel, layer_index, context_role, definition_role, int(template), surface, exact
            )
    scores = []
    for cell in cells.values():
        same = cell["same_gram_cosine"]
        advantage = cell["fixed_derangement_advantage"]
        scores.append(min(same, advantage) if same is not None and advantage is not None else float("-inf"))
    return {
        "layer_index": layer_index,
        "cells": cells,
        "selection_score": float(min(scores)),
        "median_same_gram_cosine": float(np.median([
            cell["same_gram_cosine"] for cell in cells.values() if cell["same_gram_cosine"] is not None
        ])),
        "median_fixed_derangement_advantage": float(np.median([
            cell["fixed_derangement_advantage"]
            for cell in cells.values()
            if cell["fixed_derangement_advantage"] is not None
        ])),
    }


def panel_gate(selected: dict[str, Any], baseline: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    cell_gates: dict[str, Any] = {}
    for key, cell in selected["cells"].items():
        same = cell["same_gram_cosine"]
        advantage = cell["fixed_derangement_advantage"]
        percentile = cell["exact_permutation_percentile"]
        cell_gates[key] = {
            "same_cosine_pass": same is not None and same >= thresholds["minimum_same_gram_cosine"],
            "derangement_advantage_pass": advantage is not None
            and advantage >= thresholds["minimum_fixed_derangement_advantage"],
            "exact_percentile_pass": percentile is not None
            and percentile >= thresholds["minimum_exact_permutation_percentile"],
        }
        cell_gates[key]["passed"] = all(cell_gates[key].values())
    gain = selected["selection_score"] - baseline["selection_score"]
    gain_pass = np.isfinite(gain) and gain >= thresholds["minimum_gain_over_embedding"]
    return {
        "cell_gates": cell_gates,
        "all_cells_pass": all(gate["passed"] for gate in cell_gates.values()),
        "embedding_selection_score": baseline["selection_score"],
        "gain_over_embedding": float(gain) if np.isfinite(gain) else None,
        "gain_over_embedding_pass": bool(gain_pass),
        "passed": all(gate["passed"] for gate in cell_gates.values()) and bool(gain_pass),
    }


def instrument_gate(
    factors: list[dict[str, Any]],
    states: np.ndarray,
    hidden_summary: dict[str, Any],
    selected_layer: int,
    role_lookup: dict[str, int],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    context_role = role_lookup["context_end"]
    definition_role = role_lookup["definition_end"]
    answer_role = role_lookup["answer_boundary"]
    d_context = [safe_norm(row["D"][selected_layer, context_role]) for row in factors]
    d_definition = [safe_norm(row["D"][selected_layer, definition_role]) for row in factors]
    i_context = [safe_norm(row["I"][selected_layer, context_role]) for row in factors]
    i_answer = [safe_norm(row["I"][selected_layer, answer_role]) for row in factors]
    d_ratio = float(np.median(d_context) / max(np.median(d_definition), 1e-12))
    i_ratio = float(np.median(i_context) / max(np.median(i_answer), 1e-12))
    selected_finite = bool(np.isfinite(states[:, selected_layer, :, :]).all())
    behavior_error = float(hidden_summary["maximum_behavior_z_reproduction_error"])
    result = {
        "selected_layer_values_finite": selected_finite,
        "behavior_z_reproduction_error": behavior_error,
        "behavior_reproduction_pass": behavior_error <= thresholds["maximum_behavior_z_reproduction_error"],
        "context_end_definition_leak_ratio": d_ratio,
        "context_end_interaction_leak_ratio": i_ratio,
        "causal_role_pass": max(d_ratio, i_ratio) <= thresholds["maximum_context_end_definition_leak_ratio"],
    }
    result["passed"] = (
        result["selected_layer_values_finite"]
        and result["behavior_reproduction_pass"]
        and result["causal_role_pass"]
    )
    return result


def cross_model_cell(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role_key in ("gram_c_vector", "gram_d_vector"):
        left_vector = left[role_key]
        right_vector = right[role_key]
        null_vector = right["fixed_deranged_gram_d_vector"] if role_key == "gram_d_vector" else None
        same = centered_cosine(np.asarray(left_vector), np.asarray(right_vector)) if left_vector and right_vector else None
        if role_key == "gram_d_vector":
            null = centered_cosine(np.asarray(left_vector), np.asarray(null_vector)) if left_vector and null_vector else None
        else:
            # The same fixed concept derangement is encoded by the target model's D control vector.
            # Reuse its label permutation on C by reconstructing from concept order outside this helper is unnecessary
            # for qualification; C is reported descriptively and D carries the registered null advantage.
            null = None
        output[role_key] = {"same_cosine": same, "fixed_deranged_cosine": null}
    d_same = output["gram_d_vector"]["same_cosine"]
    d_null = output["gram_d_vector"]["fixed_deranged_cosine"]
    output["d_derangement_advantage"] = d_same - d_null if d_same is not None and d_null is not None else None
    return output


def main() -> None:
    prereg = read_json(PREREG_PATH)
    verify_preregistration(prereg)
    thresholds = prereg["thresholds"]
    role_lookup = {role: index for index, role in enumerate(("context_end", "definition_end", "answer_boundary"))}

    model_results: dict[str, Any] = {}
    selected_metrics: dict[str, Any] = {}
    for model in prereg["models"]:
        specs = prereg["source_files"][model]
        cases = read_jsonl(ROOT / specs["cases"]["path"])
        hidden_summary = read_json(ROOT / specs["hidden_summary"]["path"])
        with np.load(ROOT / specs["hidden"]["path"], allow_pickle=False) as artifact:
            case_indices = artifact["case_indices"].astype(np.int64)
            if not np.array_equal(case_indices, np.arange(len(cases), dtype=np.int64)):
                raise RuntimeError(f"Case index mismatch for {model}")
            states = artifact["projected_states"].astype(np.float32)

        factors = factorial_rows(states, cases)
        layer_metrics: dict[str, Any] = {}
        for layer_index in prereg["eligible_hidden_state_indices"][model]:
            layer_metrics[str(layer_index)] = panel_metric(
                factors,
                "discovery",
                int(layer_index),
                role_lookup["context_end"],
                role_lookup["definition_end"],
                prereg,
                exact=False,
            )
        selected_layer = min(
            prereg["eligible_hidden_state_indices"][model],
            key=lambda layer: (-layer_metrics[str(layer)]["selection_score"], int(layer)),
        )

        panels: dict[str, Any] = {}
        gates: dict[str, Any] = {}
        for panel_name in prereg["panels"]:
            selected = panel_metric(
                factors,
                panel_name,
                int(selected_layer),
                role_lookup["context_end"],
                role_lookup["definition_end"],
                prereg,
                exact=True,
            )
            baseline = panel_metric(
                factors,
                panel_name,
                0,
                role_lookup["context_end"],
                role_lookup["definition_end"],
                prereg,
                exact=False,
            )
            panels[panel_name] = {"selected": selected, "embedding_baseline": baseline}
            if panel_name != "discovery":
                gates[panel_name] = panel_gate(selected, baseline, thresholds)

        instrument = instrument_gate(factors, states, hidden_summary, int(selected_layer), role_lookup, thresholds)
        qualified = instrument["passed"] and all(gate["passed"] for gate in gates.values())
        result = {
            "model": model,
            "selected_layer": int(selected_layer),
            "relative_depth": float(selected_layer / max(prereg["eligible_hidden_state_indices"][model])),
            "discovery_selection_score": layer_metrics[str(selected_layer)]["selection_score"],
            "instrument": instrument,
            "panels": panels,
            "confirmation_gates": gates,
            "qualified": bool(qualified),
        }
        model_results[model] = result
        selected_metrics[model] = panels
        write_json(OUT_ROOT / "analysis" / f"layer_metrics.{model}.json", {
            "schema_version": "phase1124_adjective_cross_role_layer_metrics.v1",
            "phase": PHASE,
            "model": model,
            "protocol_digest": prereg["protocol_digest"],
            "selected_layer": int(selected_layer),
            "discovery_layer_metrics": layer_metrics,
            "result": result,
        })
        del states, factors, cases
        gc.collect()

    cross_model: dict[str, Any] = {}
    qualified_pair_count = 0
    for left_index, left_model in enumerate(prereg["models"]):
        for right_model in prereg["models"][left_index + 1 :]:
            pair_key = f"{left_model}__{right_model}"
            pair_panels: dict[str, Any] = {}
            pair_pass = model_results[left_model]["qualified"] and model_results[right_model]["qualified"]
            for panel_name in ("independent_confirmation", "heldout"):
                cells: dict[str, Any] = {}
                for cell_key, left_cell in selected_metrics[left_model][panel_name]["selected"]["cells"].items():
                    right_cell = selected_metrics[right_model][panel_name]["selected"]["cells"][cell_key]
                    cells[cell_key] = cross_model_cell(left_cell, right_cell)
                cell_passes = []
                for cell in cells.values():
                    c_same = cell["gram_c_vector"]["same_cosine"]
                    d_same = cell["gram_d_vector"]["same_cosine"]
                    d_advantage = cell["d_derangement_advantage"]
                    cell_passes.append(
                        c_same is not None
                        and d_same is not None
                        and d_advantage is not None
                        and c_same >= thresholds["minimum_cross_model_gram_cosine"]
                        and d_same >= thresholds["minimum_cross_model_gram_cosine"]
                        and d_advantage >= thresholds["minimum_cross_model_derangement_advantage"]
                    )
                panel_pass = all(cell_passes)
                pair_panels[panel_name] = {"cells": cells, "passed": panel_pass}
                pair_pass = pair_pass and panel_pass
            cross_model[pair_key] = {
                "models_independently_qualified": model_results[left_model]["qualified"]
                and model_results[right_model]["qualified"],
                "panels": pair_panels,
                "passed": bool(pair_pass),
            }
            qualified_pair_count += int(pair_pass)

    qualified_models = [model for model, result in model_results.items() if result["qualified"]]
    instrument_models = [model for model, result in model_results.items() if result["instrument"]["passed"]]
    p1 = len(instrument_models) >= thresholds["minimum_qualified_models"]
    p2 = len(qualified_models) >= thresholds["minimum_qualified_models"]
    p3 = len([
        model
        for model, result in model_results.items()
        if all(gate["gain_over_embedding_pass"] for gate in result["confirmation_gates"].values())
    ]) >= thresholds["minimum_qualified_models"]
    p4 = qualified_pair_count >= thresholds["minimum_qualified_cross_model_pairs"]
    authorization = p1 and p2 and p3 and p4

    final_summary: dict[str, Any] = {
        "schema_version": "phase1124_adjective_cross_role_relational_geometry_final.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1123_final_digest": prereg["source_phase1123_final_digest"],
        "model_results": model_results,
        "instrument_qualified_models": instrument_models,
        "qualified_models": qualified_models,
        "cross_model": cross_model,
        "qualified_cross_model_pair_count": qualified_pair_count,
        "predictions": {
            "P1_instrument_and_causal_role": "pass" if p1 else "fail",
            "P2_two_model_double_confirmation": "pass" if p2 else "fail",
            "P3_gain_over_embedding": "pass" if p3 else "fail",
            "P4_cross_model_relational_geometry": "pass" if p4 else "fail",
            "P5_new_data_nomination_authorized": "pass" if authorization else "fail",
        },
        "new_data_confirmation_authorized": authorization,
        "component_or_causal_work_authorized": False,
        "evidence_level": "E2 at most: frozen post-source analysis using Phase1123 stored data.",
        "interpretation": {
            "positive_limit": (
                "A pass would nominate a rotation-invariant context-to-definition relation geometry that survives "
                "surface, template, concept, embedding, permutation, and model controls for new-data confirmation."
            ),
            "negative_limit": (
                "A failure constrains a shared concept-Gram implementation class. It does not exclude a learned "
                "non-orthogonal map, nonlinear decoder, attention-mediated comparison, sparse features, or causality."
            ),
            "probe_note": (
                "A global sense-0/sense-1 probe is not identified because sense labels are concept-local; a per-concept "
                "unmodified linear probe largely collapses to the C-dot-D direction test already constrained by K59."
            ),
        },
        "auto_continue": {
            "value": 0,
            "reason": (
                "Phase1124 cannot authorize components or causality. A positive result requires new independent data; "
                "a negative result requires a separately frozen instrument-positive training or dynamic-routing axis."
            ),
        },
    }
    final_summary["final_digest"] = canonical_digest(final_summary)
    write_json(OUT_ROOT / "analysis" / "final_summary.json", final_summary)

    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final_summary["final_digest"],
        "instrument_qualified_models": instrument_models,
        "qualified_models": qualified_models,
        "qualified_cross_model_pair_count": qualified_pair_count,
        "predictions": final_summary["predictions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
