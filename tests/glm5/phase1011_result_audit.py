#!/usr/bin/env python3
"""Audit Phase1011 protocol, behavior, scans, summaries, and interventions."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1011_native_semantic_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_MODES,
    PAIR_OPERATIONS,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
)


OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    require(int(protocol["phase"]) == PHASE, "phase drift")
    require(
        int(protocol["protocol_revision"]) == PROTOCOL_REVISION,
        "protocol revision drift",
    )
    require(
        protocol["output_contract"]["explicit_response_map"] is False,
        "explicit response map enabled",
    )
    model_rows = []
    for model in MODELS:
        cases_path = OUT_ROOT / "protocol" / model / "cases.jsonl"
        units_path = OUT_ROOT / "protocol" / model / "units.jsonl"
        cases = read_jsonl(cases_path)
        units = read_jsonl(units_path)
        require(len(cases) == 3456, f"{model}: case count drift")
        require(len(units) == 432, f"{model}: unit count drift")
        require(
            all(
                case["explicit_response_map_present"] is False
                for case in cases
            ),
            f"{model}: explicit map case found",
        )
        require(
            all(len(case["answer_token_ids"]) == 1 for case in cases),
            f"{model}: answer token width drift",
        )
        require(
            len({case["record_id"] for case in cases}) == len(cases),
            f"{model}: duplicate record id",
        )
        require(
            len({unit["unit_id"] for unit in units}) == len(units),
            f"{model}: duplicate unit id",
        )
        model_rows.append({
            "model": model,
            "case_count": len(cases),
            "unit_count": len(units),
            "cases_sha256": sha256(cases_path),
            "units_sha256": sha256(units_path),
        })
    return {
        "protocol_digest": protocol["preregistration_digest"],
        "models": model_rows,
    }


def audit_behavior(protocol_digest: str) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        root = OUT_ROOT / "behavior" / model
        summary = read_json(root / "summary.json")
        behavior = read_jsonl(root / "rows.jsonl")
        pairs = read_jsonl(root / "pair_qualification.jsonl")
        require(
            summary["protocol_digest"] == protocol_digest,
            f"{model}: behavior digest drift",
        )
        require(
            len(behavior) == 3456,
            f"{model}: behavior count drift",
        )
        require(
            len(pairs) == 432 * len(PAIR_OPERATIONS),
            f"{model}: pair count drift",
        )
        recomputed_panel = float(np.mean([
            row["semantic_panel_hit"] for row in behavior
        ]))
        recomputed_rollout = float(np.mean([
            row["rollout_first_word_hit"] for row in behavior
        ]))
        require(
            abs(
                recomputed_panel
                - float(summary["overall_semantic_panel_rate"])
            ) < 1e-12,
            f"{model}: panel rate drift",
        )
        require(
            abs(
                recomputed_rollout
                - float(summary["overall_rollout_first_word_rate"])
            ) < 1e-12,
            f"{model}: rollout rate drift",
        )
        rows.append({
            "model": model,
            "behavior_count": len(behavior),
            "pair_count": len(pairs),
            "semantic_panel_rate": recomputed_panel,
            "natural_rollout_first_word_rate": recomputed_rollout,
            "full_vocabulary_exact_controlled_token_rate": float(
                summary["overall_full_vocab_rate"]
            ),
        })
    return rows


def audit_scans(protocol_digest: str) -> list[dict[str, Any]]:
    results = []
    for model in MODELS:
        root = OUT_ROOT / "scan" / model
        summary = read_json(root / "summary.json")
        require(
            summary["protocol_digest"] == protocol_digest,
            f"{model}: scan digest drift",
        )
        require(summary["unit_count"] == 432, f"{model}: scan units")
        require(
            summary["raw_hidden_tensors_persisted"] == 0,
            f"{model}: hidden tensors persisted",
        )
        scalar_count = 0
        event_count = 0
        identity_maximum = 0.0
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel = root / family / output_mode
                events = read_jsonl(panel / "events.jsonl")
                units = read_jsonl(panel / "units.jsonl")
                panel_summary = read_json(panel / "summary.json")
                scalar = np.load(panel / "response_scalars.npz")
                direction = np.load(
                    panel / "direction_consistency.npz"
                )
                values = scalar["normalized_magnitude"]
                require(
                    values.shape == (
                        48,
                        len(ANALYSIS_OPERATIONS),
                        len(events),
                    ),
                    f"{model}/{family}/{output_mode}: scalar shape",
                )
                require(
                    len(units) == 48,
                    f"{model}/{family}/{output_mode}: unit count",
                )
                require(
                    np.all(np.isfinite(values)),
                    f"{model}/{family}/{output_mode}: nonfinite scalar",
                )
                identity = values[:, OP_INDEX["I"], :]
                maximum = float(np.max(np.abs(identity)))
                require(
                    maximum == 0.0,
                    f"{model}/{family}/{output_mode}: identity {maximum}",
                )
                require(
                    direction["direction_consistency"].shape == (
                        2,
                        len(ANALYSIS_OPERATIONS),
                        2,
                        len(events),
                    ),
                    f"{model}/{family}/{output_mode}: direction shape",
                )
                counts = direction["direction_count"]
                require(
                    np.all(counts[:, OP_INDEX["I"], :, :] == 0),
                    f"{model}/{family}/{output_mode}: identity directions",
                )
                scalar_count += int(values.size)
                event_count += len(events)
                identity_maximum = max(identity_maximum, maximum)
                require(
                    int(panel_summary["scalar_measurement_count"])
                    == int(values.size),
                    f"{model}/{family}/{output_mode}: count drift",
                )
                scalar.close()
                direction.close()
        require(
            scalar_count == int(summary["scalar_measurement_count"]),
            f"{model}: model scalar count drift",
        )
        results.append({
            "model": model,
            "unit_count": summary["unit_count"],
            "event_count_across_panels": event_count,
            "scalar_measurement_count": scalar_count,
            "identity_maximum": identity_maximum,
            "raw_hidden_tensors_persisted": 0,
        })
    forbidden = [
        path
        for path in (OUT_ROOT / "scan").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".safetensors"}
    ]
    require(not forbidden, f"raw tensor files found: {forbidden[:3]}")
    return results


def audit_final(protocol_digest: str) -> dict[str, Any]:
    root = OUT_ROOT / "final"
    summary = read_json(root / "summary.json")
    require(
        summary["protocol_digest"] == protocol_digest,
        "final digest drift",
    )
    motifs = read_jsonl(root / "repeated_events.jsonl")
    contours = read_jsonl(root / "response_contours.jsonl")
    sensitivity = read_jsonl(root / "threshold_sensitivity.jsonl")
    require(
        len(motifs) == summary["canonical_repeated_event_count"],
        "motif count drift",
    )
    require(
        len(contours) == summary["response_contour_count"],
        "contour count drift",
    )
    motif_ids = {row["event_id"] for row in motifs}
    require(
        all(
            event_id in motif_ids
            for row in contours
            for event_id in row["event_ids"]
        ),
        "contour references missing motif",
    )
    grouped: dict[tuple, dict[tuple[float, float], int]] = defaultdict(dict)
    for row in sensitivity:
        key = (
            row["model"],
            row["family"],
            row["output_mode"],
            row["qualification_axis"],
            row["operation"],
        )
        grouped[key][(
            float(row["direction_threshold"]),
            float(row["prevalence_threshold"]),
        )] = int(row["candidate_count"])
    for key, grid in grouped.items():
        for (direction, prevalence), count in grid.items():
            for (other_direction, other_prevalence), other_count in grid.items():
                if (
                    other_direction >= direction
                    and other_prevalence >= prevalence
                ):
                    require(
                        other_count <= count,
                        f"threshold monotonicity failed {key}",
                    )
    return {
        "canonical_repeated_event_count": len(motifs),
        "response_contour_count": len(contours),
        "threshold_cell_count": len(sensitivity),
        "prompt_repeated_event_count": int(sum(
            row["stage"] == "prompt" for row in motifs
        )),
        "after_answer_repeated_event_count": int(sum(
            row["stage"] == "after_answer" for row in motifs
        )),
        "threshold_monotonicity_pass": True,
    }


def audit_causal() -> list[dict[str, Any]]:
    results = []
    for model in ("qwen3", "glm4"):
        root = OUT_ROOT / "causal_frozen_heads" / model
        if not (root / "summary.json").exists():
            continue
        summary = read_json(root / "summary.json")
        units = read_jsonl(root / "units.jsonl")
        cells = read_jsonl(root / "cell_summaries.jsonl")
        require(
            summary["selection_used_phase1011_data"] is False,
            f"{model}: selection leakage",
        )
        require(
            len(units) == summary["unit_operation_count"],
            f"{model}: causal unit count drift",
        )
        require(
            len(cells) == summary["cell_count"],
            f"{model}: causal cell count drift",
        )
        maximum = max(
            row["noop_max_logit_error"] for row in units
        )
        require(maximum <= 1e-5, f"{model}: no-op error")
        results.append({
            "model": model,
            "unit_operation_count": len(units),
            "cell_count": len(cells),
            "descriptive_positive_cell_count": int(
                summary["descriptive_positive_cell_count"]
            ),
            "maximum_noop_logit_error": maximum,
            "selection_leakage": False,
        })
    return results


def audit_padding() -> dict[str, Any]:
    rows = []
    for model in MODELS:
        summary = read_json(
            OUT_ROOT / "padding_audit" / model / "summary.json"
        )
        require(
            summary["schema_version"]
            == "phase1011_padding_equivalence.v2",
            f"{model}: padding audit revision drift",
        )
        mixed = summary["mixed_batch_vs_same_shape_homogeneous"]
        for field in (
            "candidate_panel_prediction_agreement_rate",
            "median_hidden_relative_error",
            "median_response_relative_error",
            "median_response_direction_cosine",
        ):
            require(
                math.isfinite(float(mixed[field])),
                f"{model}: nonfinite padding metric {field}",
            )
        rows.append({
            "model": model,
            "candidate_panel_prediction_agreement_rate": mixed[
                "candidate_panel_prediction_agreement_rate"
            ],
            "median_hidden_relative_error": mixed[
                "median_hidden_relative_error"
            ],
            "median_response_relative_error": mixed[
                "median_response_relative_error"
            ],
            "median_response_direction_cosine": mixed[
                "median_response_direction_cosine"
            ],
            "minimum_response_direction_cosine": mixed[
                "minimum_response_direction_cosine"
            ],
        })
    return {
        "models": rows,
        "mixed_batch_numeric_hard_flaw_detected": bool(any(
            row["candidate_panel_prediction_agreement_rate"] < 1.0
            or row["median_response_direction_cosine"] < 0.95
            for row in rows
        )),
        "consequence": (
            "Phase1011 mixed-batch direction maps are exploratory "
            "candidate maps; Phase1013 uses singleton state forwards"
        ),
    }


def audit_bf16() -> dict[str, Any]:
    summary = read_json(
        OUT_ROOT / "precision_bf16" / "glm4" / "summary.json"
    )
    require(
        summary["entry_frozen_from_8bit_before_bf16"] is True,
        "BF16 entry was not frozen",
    )
    require(
        summary["no_op_audit_pass"] is True,
        "BF16 no-op failed",
    )
    require(
        float(summary["maximum_noop_logit_error"]) <= 1e-5,
        "BF16 no-op error",
    )
    return {
        "entered_positive_axis_count": int(
            summary["entered_positive_axis_count"]
        ),
        "confirmed_positive_axis_count": int(
            summary["bf16_confirmed_positive_axis_count"]
        ),
        "confirmation_rate": float(
            summary["bf16_confirmation_rate"]
        ),
        "underpowered_axis_count": int(sum(
            row["eight_bit_positive_axis"]
            and not row["minimum_n_met"]
            for row in summary["cell_summaries"]
        )),
        "maximum_noop_logit_error": float(
            summary["maximum_noop_logit_error"]
        ),
    }


def main() -> None:
    protocol = audit_protocol()
    behavior = audit_behavior(protocol["protocol_digest"])
    scans = audit_scans(protocol["protocol_digest"])
    final = audit_final(protocol["protocol_digest"])
    causal = audit_causal()
    padding = audit_padding()
    bf16 = audit_bf16()
    summary = {
        "schema_version": "phase1011_result_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "status": "PASS",
        "protocol": protocol,
        "behavior": behavior,
        "scans": scans,
        "final": final,
        "causal": causal,
        "padding": padding,
        "bf16": bf16,
        "claim_limits": [
            "audit verifies data integrity, not language-mechanism truth",
            "behavior, response repetition, local intervention, and rollout "
            "remain separate evidence axes",
        ],
    }
    output_root = OUT_ROOT / "audit"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
