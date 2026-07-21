#!/usr/bin/env python3
"""Register the retrospective Phase584 prompt-boundary calibration audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase581_typed_category_protocol as source
import phase583_prompt_boundary_observer as observer
import phase583_prompt_boundary_protocol as prompt_boundary


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase584"
MODELS = prompt_boundary.MODELS
RELATIONS = prompt_boundary.RELATIONS
CALIBRATION_SPLIT = "behavior_discovery"
EVALUATION_SPLITS = ("behavior_confirmation", "heldout_objects")
REPEATS = prompt_boundary.NOOP_REPEATS
MIN_OVERALL_ACCURACY = 0.90
MIN_PER_CATEGORY_ACCURACY = 0.90
MIN_STABLE_SURFACES_PER_OBJECT = prompt_boundary.MIN_STABLE_SURFACES_PER_OBJECT
MIN_QUALIFIED_BY_RELATION_CATEGORY = source.MIN_QUALIFIED_BY_RELATION_CATEGORY
MAX_REPEAT_LOGIT_DELTA = prompt_boundary.MAX_REPEAT_LOGIT_DELTA

OUT_DIR = ROOT / "tests/gpt5/result/phase584_boundary_calibration"
PROTOCOL_PATH = OUT_DIR / "phase584_frozen_analysis_contract.json"
DECISION_PATH = OUT_DIR / "phase584_boundary_calibration_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def register() -> dict[str, Any]:
    source_decision = json.loads(
        (prompt_boundary.OUT_DIR / "phase583_prompt_boundary_decision.json").read_text(
            encoding="utf-8"
        )
    )
    if source_decision["sealed_split_read"]:
        raise RuntimeError("Phase584 requires an unsealed Phase583 source")

    source_artifacts: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        paths = observer.paths(model)
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        rows_sha256 = sha256_file(paths["rows"])
        if summary["rows_sha256"] != rows_sha256:
            raise RuntimeError(f"Phase584 {model} source rows drift")
        source_artifacts[model] = {
            "rows_path": str(paths["rows"].relative_to(ROOT)),
            "rows_sha256": rows_sha256,
            "summary_path": str(paths["summary"].relative_to(ROOT)),
            "summary_sha256": sha256_file(paths["summary"]),
        }

    payload = {
        "schema_version": "phase584_boundary_calibration_contract.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Retrospective binary boundary calibration of prompt-boundary category logits",
        "source_phase": prompt_boundary.PHASE,
        "source_protocol_sha256": sha256_file(prompt_boundary.PROTOCOL_PATH),
        "source_decision_sha256": sha256_file(
            prompt_boundary.OUT_DIR / "phase583_prompt_boundary_decision.json"
        ),
        "source_artifacts": source_artifacts,
        "models": list(MODELS),
        "relations": list(RELATIONS),
        "relation_categories_in_fixed_axis_order": {
            relation: list(source.RELATION_CATEGORIES[relation])
            for relation in RELATIONS
        },
        "calibration_split": CALIBRATION_SPLIT,
        "evaluation_splits": list(EVALUATION_SPLITS),
        "repeats": list(REPEATS),
        "calibration_rule": {
            "axis": "first category boundary logit minus second category boundary logit",
            "orientation": "larger discovery class mean is assigned to its observed class",
            "threshold": "midpoint of the two discovery class means",
            "prediction": "class selected by oriented side of the frozen threshold",
            "fitted_parameters_per_model_relation": ["orientation", "threshold"],
        },
        "diagnostic_gate": {
            "minimum_overall_accuracy_each_evaluation_split": MIN_OVERALL_ACCURACY,
            "minimum_accuracy_each_category_each_evaluation_split": MIN_PER_CATEGORY_ACCURACY,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_by_relation_category_each_split": MIN_QUALIFIED_BY_RELATION_CATEGORY,
            "maximum_repeat_logit_delta": MAX_REPEAT_LOGIT_DELTA,
            "both_evaluation_splits_must_pass": True,
        },
        "evidence_policy": {
            "retrospective_open_data_diagnostic": True,
            "registered_after_open_data_exploration": True,
            "not_independent_confirmatory_evidence": True,
            "not_natural_generation": True,
            "not_internal_localization": True,
            "not_causal_evidence": True,
            "prompt_trace_authorized": False,
            "causal_intervention_authorized": False,
            "sealed_split_read": False,
            "strict_mechanism_closure_claimed": False,
        },
    }
    write_json(PROTOCOL_PATH, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
