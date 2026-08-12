#!/usr/bin/env python3
"""Freeze the Phase1123 terminal hidden external-validity protocol."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS
import phase1121_wordnet_adjective_double_orthogonal_protocol as source
import phase1122_adjective_lexical_coherence_protocol as lexical_protocol


PHASE = 1123
PROTOCOL_REVISION = 3
MODELS = tuple(source.REFERENCE_MODELS)
ROLES = ("context_end", "definition_end", "answer_boundary")
PROJECTION_DIM = 256
BATCH_SIZES = {"qwen3": 12, "glm4": 4, "deepseek7b": 4}
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1123_adjective_terminal_hidden_external_validity"
PANELS = {
    "discovery": {"split": "discovery", "templates": [0, 1]},
    "independent_confirmation": {"split": "independent_confirmation", "templates": [2, 3]},
    "heldout": {"split": "heldout", "templates": [4, 5]},
}


THRESHOLDS = {
    "minimum_finite_fraction": 0.999,
    "maximum_behavior_z_reproduction_error": 0.05,
    "maximum_context_end_definition_leak_ratio": 0.02,
    "minimum_same_cosine": 0.10,
    "minimum_matched_advantage": 0.05,
    "minimum_semantic_score": 0.05,
    "minimum_gain_over_embedding": 0.03,
    "minimum_qualified_models": 2,
    "minimum_cross_model_gram_cosine": 0.20,
    "minimum_cross_model_gram_advantage": 0.05,
    "minimum_qualified_cross_model_pairs": 1,
}


PREDICTIONS = {
    "P1": "Phase1121 behavior and Phase1122 lexical-null audits remain intact; all token roles and projections pass before hidden output is read.",
    "P2": "Output z values reproduce Phase1121 and definition/interactions remain causally absent at context_end.",
    "P3": "At least two models independently confirm context-to-definition, cross-surface, and cross-template geometry above deranged controls in both confirmation panels.",
    "P4": "The selected semantic score exceeds the embedding-state baseline in both confirmation panels.",
    "P5": "At least one model pair preserves the confirmed concept-relation Gram geometry above a fixed deranged-label null in both panels.",
    "P6": "Only joint P1-P5 success authorizes a separately frozen component-discovery protocol; this phase performs no component selection or intervention.",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def tokenizer_for_model(model_name: str):
    return source.tokenizer_for_phase(model_name)


def prefix_role_index(tokenizer: Any, rendered_prompt: str, input_ids: list[int], fragment: str) -> tuple[int, str]:
    if rendered_prompt.count(fragment) != 1:
        raise RuntimeError(f"role fragment is not unique: {fragment!r}")
    start = rendered_prompt.index(fragment)
    end = start + len(fragment)
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    encoded_ids = [int(value) for value in encoded["input_ids"]]
    if encoded_ids != input_ids:
        raise RuntimeError("offset tokenization does not reproduce the frozen input ids")
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    overlapping = [
        index for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not overlapping:
        raise RuntimeError(f"no token overlaps role fragment: {fragment!r}")
    role_index = max(overlapping)
    return role_index, digest(input_ids[: role_index + 1])


def projection_for(model_name: str, hidden_size: int) -> tuple[np.ndarray, int]:
    seed = int(hashlib.sha256(f"phase1123-projection-{model_name}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(hidden_size, PROJECTION_DIM), dtype=np.int8)
    matrix = (signs.astype(np.float32) * 2.0 - 1.0) / math.sqrt(PROJECTION_DIM)
    return matrix, seed


def main() -> None:
    source_prereg = read_json(source.OUT_ROOT / "protocol" / "preregistration.json")
    source_audit = read_json(source.OUT_ROOT / "protocol" / "audit.json")
    source_final = read_json(source.OUT_ROOT / "analysis" / "final_summary.json")
    lexical_final = read_json(lexical_protocol.OUT_ROOT / "analysis" / "final_summary.json")
    lexical_audit = read_json(lexical_protocol.OUT_ROOT / "audit" / "result_audit.json")
    if not source_audit["all_checks_passed"] or not lexical_final["lexical_null_audit_passed"] or not lexical_audit["all_checks_passed"]:
        raise RuntimeError("Phase1121/1122 source authorization failed")
    if not all(source_final["models"][model]["qualified"] for model in MODELS):
        raise RuntimeError("not every frozen chat model passed Phase1121")

    model_specs: dict[str, Any] = {}
    case_digests: dict[str, str] = {}
    projection_specs: dict[str, Any] = {}
    model_audits: dict[str, Any] = {}
    for model_name in MODELS:
        config = AutoConfig.from_pretrained(MODEL_CONFIGS[model_name]["path"], trust_remote_code=True, local_files_only=True)
        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        hidden_state_count = num_layers + 1
        tokenizer = tokenizer_for_model(model_name)
        source_cases = read_jsonl(source.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
        if digest(source_cases) != source_prereg["case_digests"][model_name]:
            raise RuntimeError(f"Phase1121 case digest mismatch for {model_name}")

        cases: list[dict[str, Any]] = []
        role_checks: list[bool] = []
        for row in source_cases:
            context_index, context_prefix_digest = prefix_role_index(
                tokenizer, row["rendered_prompt"], row["input_ids"], row["sentence"]
            )
            definition_index, definition_prefix_digest = prefix_role_index(
                tokenizer, row["rendered_prompt"], row["input_ids"], row["definition"]
            )
            role_indices = {
                "context_end": context_index,
                "definition_end": definition_index,
                "answer_boundary": len(row["input_ids"]) - 1,
            }
            role_checks.extend([
                0 <= context_index < definition_index < len(row["input_ids"]) - 1,
                role_indices["answer_boundary"] == len(row["input_ids"]) - 1,
            ])
            cases.append({
                **row,
                "schema_version": "phase1123_adjective_terminal_hidden_case.v1",
                "source_schema_version": row["schema_version"],
                "source_record_id": row["record_id"],
                "role_indices": role_indices,
                "role_prefix_digests": {
                    "context_end": context_prefix_digest,
                    "definition_end": definition_prefix_digest,
                },
            })
        case_digests[model_name] = digest(cases)
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl", cases)

        projection, seed = projection_for(model_name, hidden_size)
        projection_path = OUT_ROOT / "protocol" / f"projection.{model_name}.npy"
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(projection_path, projection, allow_pickle=False)
        projection_specs[model_name] = {
            "path": str(projection_path.relative_to(OUT_ROOT)).replace("\\", "/"),
            "sha256": file_sha256(projection_path),
            "seed": seed,
            "shape": list(projection.shape),
            "distribution": "Rademacher entries scaled by 1/sqrt(projection_dimension)",
        }
        eligible = list(range(1, num_layers))
        model_specs[model_name] = {
            "path": MODEL_CONFIGS[model_name]["path"],
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "hidden_state_count": hidden_state_count,
            "eligible_hidden_state_indices": eligible,
        }
        model_audits[model_name] = {
            "case_count": len(cases),
            "case_index_bijection": sorted(int(row["case_index"]) for row in cases) == list(range(len(cases))),
            "all_role_indices_ordered": all(role_checks),
            "all_three_roles": all(set(row["role_indices"]) == set(ROLES) for row in cases),
            "factorial_cell_count": len({row["interaction_id"] for row in cases}) == 288,
            "projection_shape": projection.shape == (hidden_size, PROJECTION_DIM),
            "projection_finite": bool(np.isfinite(projection).all()),
            "eligible_layer_count": len(eligible) == max(num_layers - 1, 0),
        }

    prereg_core = {
        "schema_version": "phase1123_adjective_terminal_hidden_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "precision": "fp16",
        "quantization": "none",
        "source_phase1121_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1121_final_digest": source_final["final_digest"],
        "source_phase1122_final_digest": lexical_final["final_digest"],
        "case_digests": case_digests,
        "case_count_per_model": source_prereg["case_count_per_model"],
        "roles": list(ROLES),
        "projection_dimension": PROJECTION_DIM,
        "projected_state_storage_dtype": "float32",
        "batch_sizes": BATCH_SIZES,
        "projection_specs": projection_specs,
        "model_specs": model_specs,
        "panels": PANELS,
        "thresholds": THRESHOLDS,
        "predictions": PREDICTIONS,
        "factorial_fields": {
            "context_C": "0.5*((h00+h01)-(h10+h11)); definition and truth balanced",
            "definition_D": "0.5*((h00+h10)-(h01+h11)); context and truth balanced",
            "interaction_I": "0.5*((h00+h11)-(h01+h10)); true-minus-false comparison field",
        },
        "primary_role": "answer_boundary",
        "layer_selection": "per model, maximize the minimum of C-D, base-synonym C, and cross-template C matched-control advantages on discovery only",
        "embedding_baseline": "hidden_state_index 0 evaluated independently on each confirmation panel",
        "cross_model_object": "centered off-diagonal Gram geometry of concept-averaged C and D fields at each model's frozen selected layer",
        "model_outputs_read_during_protocol": False,
        "scope_limit": "Terminal residual geometry only. This protocol does not identify training formation, a component, use, necessity, sufficiency, or an abstract universal coordinate.",
        "forbidden_actions": [
            "change cases, roles, projections, panels, thresholds, or layer selection after hidden output is read",
            "select layers from either confirmation panel",
            "use the truth/interactions field I as the semantic qualification object",
            "drop a failing model, panel, template, surface, or concept",
            "run component, head, neuron, patch, ablation, or restoration analysis in Phase1123",
        ],
        "revision_note": "Revision 3 retains the Revision 2 source batch sizes and stores every projected state in FP32 after Revision 2 exposed DS7B FP16 artifact overflow before any factorial hidden metric was computed. Revision 2 had repaired Revision 1's batch-size logit reproduction failure. No threshold, case, role, projection, panel, or scientific metric changed in either instrument-only revision.",
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)

    checks = {
        "phase1121_protocol_passed": source_audit["all_checks_passed"],
        "phase1121_all_chat_models_qualified": all(source_final["models"][model]["qualified"] for model in MODELS),
        "phase1122_lexical_null_audit_passed": lexical_final["lexical_null_audit_passed"],
        "phase1122_result_audit_passed": lexical_audit["all_checks_passed"],
        "model_count_three": len(MODELS) == 3,
        "case_counts_1152": all(audit["case_count"] == 1152 for audit in model_audits.values()),
        "all_model_protocol_audits": all(all(audit.values()) for audit in model_audits.values()),
        "panel_splits_disjoint": len({spec["split"] for spec in PANELS.values()}) == len(PANELS),
        "panel_templates_disjoint": len({template for spec in PANELS.values() for template in spec["templates"]}) == 6,
        "protocol_digest": digest(prereg_core) == prereg["protocol_digest"],
    }
    audit_core = {
        "schema_version": "phase1123_adjective_terminal_hidden_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "model_audits": model_audits,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1123 protocol audit failed")
    print(json.dumps({"preregistration": prereg, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
