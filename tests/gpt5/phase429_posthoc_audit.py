#!/usr/bin/env python3
"""Audit Phase429 evidence without changing its frozen protocol or gates."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase429_typed_route"
VIS = ROOT / "frontend/public/vis_data/phase429_architecture_path"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
PHASE_ID = "Phase429-PosthocEvidenceAudit"
SCHEMA_VERSION = "phase429_posthoc_audit.v1"

METRIC_FIELDS = (
    "residual_pre_rms",
    "query_projection_rms",
    "source_key_projection_rms",
    "source_value_projection_rms",
    "attention_write_rms",
    "mlp_write_rms",
    "residual_post_rms",
    "transition_rms",
    "reconstruction_relative_error",
    "attention_mlp_cosine",
    "transition_attention_cosine",
    "transition_mlp_cosine",
    "target_first_token_margin",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    return round(float(value), 10)


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"trace_row_count": len(values)}
    for field in METRIC_FIELDS:
        numbers = [float(row[field]) for row in values]
        result[f"{field}_mean"] = clean(statistics.fmean(numbers))
        result[f"{field}_median"] = clean(statistics.median(numbers))
    result["attention_mlp_negative_fraction"] = clean(
        sum(float(row["attention_mlp_cosine"]) < 0 for row in values) / len(values)
    )
    return result


def selected_observer_metrics(
    model: str,
    interface: str,
    observer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        row
        for row in observer_rows
        if row["model"] == model and row["interface"] == interface
    ]
    return {
        f"{row['block_id']}::{row['split']}": {
            "teacher_correct": row["metrics"]["teacher_correct"],
            "target_first": row["metrics"]["target_first"],
            "opposite_first": row["metrics"]["opposite_first"],
            "interface_valid": row["metrics"]["interface_valid"],
            "stop": row["metrics"]["stop"],
            "censoring": row["metrics"]["censoring"],
        }
        for row in selected
    }


def correct_visual_evidence_metadata() -> None:
    manifest_path = VIS / "manifest.json"
    manifest = read_json(manifest_path)
    for item in manifest["items"]:
        item["evidence_scope"] = (
            "architecture observation at a registered query token; diagnostic readout "
            "is non-terminal, non-predictive, non-causal and non-neuronal"
        )
        payload_path = VIS / item["filename"]
        payload = read_json(payload_path)
        payload["evidence_scope"] = item["evidence_scope"]
        payload["graph"]["meta"].update(
            {
                "readout_position_contract": "last token of registered question span",
                "readout_is_autoregressive_terminal": False,
                "terminal_prediction_interpretation_valid": False,
                "prediction_gate_pass": False,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "single_neuron": False,
                "causal": False,
            }
        )
        write_json(payload_path, payload)
    manifest["generated_at"] = now()
    write_json(manifest_path, manifest)

    registry = read_json(REGISTRY)
    for source in registry["sources"]:
        if source["id"] == "gpt5_phase429_architecture_path":
            source["description"] = (
                "仅显示内容门授权块在注册查询词元处的残差、查询、来源键值、注意力和"
                "多层感知机写入；诊断读出不是提示词末端预测，不含头、通道或神经元扫描。"
            )
            source["evidence_scope"] = (
                "注册查询词元处的架构级物理观察；非末端预测、非因果、非神经元闭合"
            )
            break
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def main() -> None:
    protocol = read_json(OUT / "phase429_protocol.json")
    physical_protocol = read_json(OUT / "phase429_physical_protocol.json")
    interface_freeze = read_json(OUT / "phase429_interface_freeze.json")
    observer_rows = read_jsonl(OUT / "phase429_observer_summaries.jsonl")
    candidates = read_jsonl(OUT / "phase429_candidate_audits.jsonl")
    behavior_gate = read_json(OUT / "phase429_open_behavior_gate.json")
    physical_gate = read_json(OUT / "phase429_open_physical_gate.json")
    trace_rows = read_jsonl(
        OUT / "physical/open/qwen3/phase429_physical_rows.jsonl"
    )

    if behavior_gate["sealed_rows_read"] or physical_gate["sealed_rows_read"]:
        raise RuntimeError("Phase429 posthoc audit may not read sealed rows")
    if physical_gate["head_channel_neuron_scan"] or physical_gate["causal_tested"]:
        raise RuntimeError("Phase429 physical evidence boundary was widened")

    observer = {}
    for model, frozen in interface_freeze["models"].items():
        observer[model] = {
            "selected_interface": frozen["selected_interface"],
            "behavior_authorized": frozen["behavior_authorized"],
            "selection_reused_behavior_data": frozen[
                "selection_reused_behavior_data"
            ],
            "calibration_gate_pass": all(
                gate["gate_pass"] for gate in frozen["calibration_gates"].values()
            ),
            "holdout_gate_pass": all(
                gate["gate_pass"] for gate in frozen["holdout_gates"].values()
            ),
            "selected_interface_metrics": selected_observer_metrics(
                model, frozen["selected_interface"], observer_rows
            ),
        }

    candidate_rows = []
    failed_gate_counts: Counter[str] = Counter()
    for audit in candidates:
        route_failures = {}
        for route, route_audit in audit["route_gates"].items():
            split_failures = {}
            for split, paired in route_audit["splits"].items():
                candidate_failed = [
                    gate_name
                    for gate_name, gate in paired["candidate"]["typed_gates"].items()
                    if not gate["gate_pass"]
                ]
                control_failed = [
                    gate_name
                    for gate_name, gate in paired["control"]["typed_gates"].items()
                    if not gate["gate_pass"]
                ]
                failed_gate_counts.update(f"candidate::{name}" for name in candidate_failed)
                failed_gate_counts.update(f"control::{name}" for name in control_failed)
                split_failures[split] = {
                    "candidate_failed_typed_gates": candidate_failed,
                    "control_failed_typed_gates": control_failed,
                    "paired_content_pass": paired["paired_content_pass"],
                }
            route_failures[route] = split_failures
        candidate_rows.append(
            {
                "model": audit["model"],
                "block_id": audit["block_id"],
                "contract_variant": audit["contract_variant"],
                "dual_route_content_qualified": audit[
                    "dual_route_content_qualified"
                ],
                "specificity_qualified": audit["specificity_qualified"],
                "complete_generation_qualified": audit[
                    "complete_generation_qualified"
                ],
                "physical_content_authorized": audit[
                    "physical_content_authorized"
                ],
                "specificity": audit["specificity"],
                "route_failures": route_failures,
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[
            (
                row["model"],
                row["block_id"],
                row["contract_variant"],
                row["split"],
                row["route_mode"],
                int(row["layer"]),
            )
        ].append(row)
    layer_rows = []
    for key, values in sorted(grouped.items()):
        model, block_id, contract_variant, split, route_mode, layer = key
        layer_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "model": model,
                "block_id": block_id,
                "contract_variant": contract_variant,
                "split": split,
                "route_mode": route_mode,
                "layer": layer,
                "layer_fraction": clean(layer / 35),
                **summarize(values),
                "descriptive_only": True,
                "causal": False,
                "single_neuron": False,
            }
        )
    write_jsonl(OUT / "phase429_physical_layer_summaries.jsonl", layer_rows)

    selected_layers = [
        row
        for row in layer_rows
        if row["block_id"] == "language_action_dual_route_candidate"
        and row["split"] == "behavior_holdout"
        and row["route_mode"] == "consistent"
    ]
    peak_fields = (
        "query_projection_rms_mean",
        "source_key_projection_rms_mean",
        "source_value_projection_rms_mean",
        "attention_write_rms_mean",
        "mlp_write_rms_mean",
        "transition_rms_mean",
    )
    peaks = {
        field: max(
            ({"layer": row["layer"], "value": row[field]} for row in selected_layers),
            key=lambda item: item["value"],
        )
        for field in peak_fields
    }
    bands = {"early": range(0, 12), "middle": range(12, 24), "late": range(24, 36)}
    cancellation_bands = {}
    for name, layers in bands.items():
        values = [row for row in selected_layers if row["layer"] in layers]
        cancellation_bands[name] = {
            "attention_mlp_cosine_mean": clean(
                statistics.fmean(row["attention_mlp_cosine_mean"] for row in values)
            ),
            "attention_mlp_negative_fraction_mean": clean(
                statistics.fmean(
                    row["attention_mlp_negative_fraction"] for row in values
                )
            ),
        }

    prompt_sample = read_jsonl(OUT / "phase429_physical_conditions_open.jsonl")[0][
        "rendered_prompt"
    ]
    question_marker = "Question: Which item is selected?"
    output_marker = "Output exactly the selected item and then stop."
    query_precedes_output_instruction = (
        prompt_sample.find(question_marker) >= 0
        and prompt_sample.find(output_marker) > prompt_sample.find(question_marker)
    )
    if not query_precedes_output_instruction:
        raise RuntimeError("Could not verify the Phase429 query-position audit")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "frozen_protocol_changed": False,
        "threshold_prompt_window_or_sample_changed": False,
        "registered_denominators": {
            "observer_formal_condition_count": protocol["validation"][
                "observer_formal_condition_count"
            ],
            "behavior_formal_group_count": protocol["validation"][
                "behavior_formal_group_count"
            ],
            "behavior_open_group_count": protocol["validation"][
                "behavior_open_group_count"
            ],
            "behavior_sealed_group_count": protocol["validation"][
                "behavior_sealed_group_count"
            ],
            "executed_behavior_condition_count": 30720,
            "physical_condition_count": physical_protocol["condition_count"],
            "physical_trace_row_count": len(trace_rows),
            "physical_layer_count": 36,
        },
        "observer_interface_audit": observer,
        "candidate_audits": candidate_rows,
        "failed_typed_gate_counts": dict(sorted(failed_gate_counts.items())),
        "authorized_candidate_count": sum(
            row["physical_content_authorized"] for row in candidates
        ),
        "cross_model_content_candidate_count": 0,
        "physical_audit": {
            "reconstruction": physical_gate["reconstruction"],
            "reconstruction_gate_pass": physical_gate[
                "reconstruction_gate_pass"
            ],
            "registered_numeric_prediction_result": physical_gate["prediction"],
            "registered_numeric_prediction_gate_pass": physical_gate[
                "prediction_gate_pass"
            ],
            "readout_position": physical_protocol["record_contract"][
                "query_position"
            ],
            "query_precedes_output_instruction": query_precedes_output_instruction,
            "readout_is_true_autoregressive_terminal": False,
            "terminal_prediction_interpretation_valid": False,
            "interpretation": (
                "The 0/96 result rejects the registered query-token pseudo-readout. "
                "It is not evidence that a true prompt-terminal state or transport path is absent."
            ),
            "selected_candidate_holdout_consistent_peak_layers": peaks,
            "selected_candidate_holdout_consistent_cancellation_bands": cancellation_bands,
            "sealed_rows_read": False,
            "sealed_unlock": False,
            "head_channel_neuron_scan": False,
            "intervention": False,
            "causal_tested": False,
        },
        "evidence_boundary": {
            "strict_mechanism_closure": "0/72",
            "overall_scientific_progress_percent": 21,
            "progress_interval_percent": [18, 24],
            "physical_graph_level": "architecture observation",
            "predictive_graph_level": "not qualified",
            "causal_graph_level": "not tested",
            "neuron_graph_level": "not tested",
        },
        "stage_decision": {
            "phase429_complete": True,
            "sealed_stage_authorized": False,
            "automatic_same_stage_continuation": False,
            "next_phase_required": True,
            "reason": (
                "The current stage hit its registered stop. The next test must freeze "
                "a different position-time denominator before reading any sealed group."
            ),
        },
    }
    write_json(OUT / "phase429_posthoc_audit.json", audit)

    global_summary = read_json(OUT / "phase429_global_summary.json")
    global_summary.update(
        {
            "prediction_readout_position": "registered_query_token",
            "prediction_terminal_interpretation_valid": False,
            "sealed_tested": False,
            "sealed_unlock": False,
            "causal_tested": False,
            "strict_mechanism_closure": "0/72",
            "overall_scientific_progress_percent": 21,
            "progress_interval_percent": [18, 24],
            "conclusion": (
                "Architecture conservation passed. The registered query-token "
                "pseudo-readout failed its numeric gate and is not a terminal "
                "autoregressive prediction. Sealed, causal, head, channel and neuron "
                "stages remain closed."
            ),
        }
    )
    write_json(OUT / "phase429_global_summary.json", global_summary)
    correct_visual_evidence_metadata()
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
