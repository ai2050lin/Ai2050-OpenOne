from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


PHASE = 1124
ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1123_adjective_terminal_hidden_external_validity"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1124_adjective_cross_role_relational_geometry"
MODELS = ("qwen3", "glm4", "deepseek7b")
PANELS = {
    "discovery": {"split": "discovery", "templates": [0, 1]},
    "independent_confirmation": {"split": "independent_confirmation", "templates": [2, 3]},
    "heldout": {"split": "heldout", "templates": [4, 5]},
}
SURFACES = ("base", "synonym")
SOURCE_ROLE = "context_end"
TARGET_ROLE = "definition_end"

THRESHOLDS = {
    "maximum_behavior_z_reproduction_error": 0.05,
    "maximum_context_end_definition_leak_ratio": 0.02,
    "minimum_same_gram_cosine": 0.30,
    "minimum_fixed_derangement_advantage": 0.10,
    "minimum_exact_permutation_percentile": 0.95,
    "minimum_gain_over_embedding": 0.05,
    "minimum_qualified_models": 2,
    "minimum_cross_model_gram_cosine": 0.20,
    "minimum_cross_model_derangement_advantage": 0.05,
    "minimum_qualified_cross_model_pairs": 1,
}


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    source_prereg_path = SOURCE_ROOT / "protocol" / "preregistration.json"
    source_final_path = SOURCE_ROOT / "analysis" / "final_summary.json"
    source_prereg = read_json(source_prereg_path)
    source_final = read_json(source_final_path)

    if source_prereg["protocol_digest"] != "33327af930e37392c6c94fd0ce7d0d66029301c9a423706568aa166ee16d5519":
        raise RuntimeError("Unexpected Phase1123 protocol digest")
    if source_final["final_digest"] != "d37806a2f8b4bd4b4c3aba305d4cea50f3bdb7349f9bcdb5b9be33a7b2b156ff":
        raise RuntimeError("Unexpected Phase1123 final digest")

    source_files: dict[str, dict[str, Any]] = {}
    panel_audits: dict[str, Any] = {}
    for model in MODELS:
        case_path = SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"
        hidden_path = SOURCE_ROOT / "hidden" / model / "hidden_detail.npz"
        hidden_summary_path = SOURCE_ROOT / "hidden" / model / "summary.json"
        cases = read_jsonl(case_path)
        if len(cases) != source_prereg["case_count_per_model"]:
            raise RuntimeError(f"Unexpected case count for {model}")

        panel_rows: dict[str, Any] = {}
        for panel_name, panel in PANELS.items():
            rows = [
                row
                for row in cases
                if row["split"] == panel["split"] and int(row["template"]) in panel["templates"]
            ]
            concepts = sorted({row["concept_id"] for row in rows})
            controls = {row["concept_id"]: row["deranged_control_concept_id"] for row in rows}
            panel_rows[panel_name] = {
                "case_count": len(rows),
                "concept_count": len(concepts),
                "interaction_count": len({row["interaction_id"] for row in rows}),
                "templates": sorted({int(row["template"]) for row in rows}),
                "surfaces": sorted({row["surface"] for row in rows}),
                "control_is_derangement": (
                    set(controls) == set(concepts)
                    and set(controls.values()) == set(concepts)
                    and all(concept != controls[concept] for concept in concepts)
                ),
            }
        panel_audits[model] = panel_rows
        source_files[model] = {
            "cases": {
                "path": str(case_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": file_sha256(case_path),
            },
            "hidden": {
                "path": str(hidden_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": file_sha256(hidden_path),
            },
            "hidden_summary": {
                "path": str(hidden_summary_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": file_sha256(hidden_summary_path),
            },
        }

    audit_checks = {
        "source_protocol_digest_frozen": source_prereg["protocol_digest"]
        == "33327af930e37392c6c94fd0ce7d0d66029301c9a423706568aa166ee16d5519",
        "source_final_digest_frozen": source_final["final_digest"]
        == "d37806a2f8b4bd4b4c3aba305d4cea50f3bdb7349f9bcdb5b9be33a7b2b156ff",
        "all_panels_have_eight_concepts": all(
            panel["concept_count"] == 8
            for model in panel_audits.values()
            for panel in model.values()
        ),
        "all_panels_have_four_cells_per_concept": all(
            panel["interaction_count"] == 32
            for model in panel_audits.values()
            for panel in model.values()
        ),
        "all_control_maps_are_derangements": all(
            panel["control_is_derangement"]
            for model in panel_audits.values()
            for panel in model.values()
        ),
        "exact_permutation_count_is_40320": len(list(itertools.permutations(range(8)))) == 40320,
        "source_roles_available": SOURCE_ROLE in source_prereg["roles"] and TARGET_ROLE in source_prereg["roles"],
        "source_projection_is_fp32": source_prereg["projected_state_storage_dtype"] == "float32",
    }
    if not all(audit_checks.values()):
        raise RuntimeError(f"Phase1124 protocol audit failed: {audit_checks}")

    preregistration: dict[str, Any] = {
        "schema_version": "phase1124_adjective_cross_role_relational_geometry_preregistration.v1",
        "phase": PHASE,
        "source_phase1123_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1123_final_digest": source_final["final_digest"],
        "source_files": source_files,
        "models": list(MODELS),
        "precision_of_source_states": "fp16 model forward; fixed Rademacher projection and artifact storage in fp32",
        "projection_dimension": int(source_prereg["projection_dimension"]),
        "roles": {"context_field": SOURCE_ROLE, "definition_field": TARGET_ROLE},
        "panels": PANELS,
        "surfaces": list(SURFACES),
        "eligible_hidden_state_indices": {
            model: source_prereg["model_specs"][model]["eligible_hidden_state_indices"] for model in MODELS
        },
        "primary_object": (
            "For every concept/template/surface, form the truth-balanced context field C at context_end "
            "and definition field D at definition_end. Compare their centered off-diagonal concept Gram "
            "vectors. This is invariant to a shared orthogonal rotation between role coordinates."
        ),
        "factorial_fields": source_prereg["factorial_fields"],
        "cell_structure": "Each panel is evaluated in four separate template-by-surface cells; pooling cannot rescue a failed cell.",
        "layer_selection": (
            "Discovery only: select the eligible layer maximizing the minimum, over four cells, of "
            "min(same-role-geometry cosine, same-minus-fixed-derangement advantage). Ties choose the earlier layer."
        ),
        "confirmation": (
            "At the frozen layer, every one of four cells must pass absolute Gram cosine, fixed derangement "
            "advantage, and exact 8! permutation percentile in both independent_confirmation and heldout."
        ),
        "embedding_baseline": (
            "Hidden-state index 0 at context_end and definition_end, evaluated by the identical four-cell rule. "
            "It is a role-local terminal-token embedding baseline, not a pooled semantic embedding."
        ),
        "exact_permutation_null": {
            "concept_count": 8,
            "permutation_count": 40320,
            "statistic": "centered off-diagonal Gram cosine after jointly permuting definition-role concept labels",
        },
        "instrument_gate": (
            "Source hashes and logit reproduction must pass; all selected eligible values must be finite; "
            "the inherited 0.02 context-end definition/interactions leak ratio is not relaxed."
        ),
        "cross_model_object": (
            "Only independently qualified models are compared, using centered off-diagonal C and D Gram vectors "
            "at their separately frozen layers in every confirmation cell."
        ),
        "thresholds": THRESHOLDS,
        "predictions": {
            "P1": "Source integrity, eligible-state finiteness, behavior reproduction, and causal-role audit pass.",
            "P2": "At least two models pass all four cells in both independent confirmation panels.",
            "P3": "The frozen relational-geometry score exceeds the role-local embedding baseline in both panels.",
            "P4": "At least one independently qualified model pair preserves both C and D relation geometry above deranged controls in both panels.",
            "P5": "Only joint P1-P4 success nominates a relation-geometry candidate for new-data confirmation; no component or causal work is authorized here.",
        },
        "evidence_level": (
            "At most E2 because the hypothesis was formulated after Phase1123 and reuses its stored data, "
            "even though all Phase1124 metrics and thresholds are frozen before reading them."
        ),
        "scope_limit": (
            "A pass would identify rotation-invariant relation geometry, not a universal sense decoder, dynamic use, "
            "a component, causality, non-WordNet generalization, or training formation. A failure constrains only "
            "this concept-Gram implementation class and cannot exclude nonlinear or distributed dynamic matching."
        ),
        "forbidden_actions": [
            "fit a probe, CCA, Procrustes map, SAE, or any learned alignment in Phase1124",
            "change layers, roles, cells, thresholds, nulls, or panel membership after metrics are read",
            "pool cells before applying the four-cell gate",
            "drop a failing model, panel, surface, template, or concept",
            "upgrade a reused-data result to E3",
            "run component, head, neuron, patch, ablation, restoration, or training in Phase1124",
        ],
    }
    preregistration["protocol_digest"] = canonical_digest(preregistration)

    audit = {
        "schema_version": "phase1124_adjective_cross_role_relational_geometry_protocol_audit.v1",
        "phase": PHASE,
        "checks": audit_checks,
        "panel_audits": panel_audits,
        "protocol_digest": preregistration["protocol_digest"],
    }
    audit["audit_digest"] = canonical_digest(audit)

    write_json(OUT_ROOT / "protocol" / "preregistration.json", preregistration)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "all_checks_passed": all(audit_checks.values()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
