#!/usr/bin/env python3
"""Phase493 read-only stage audit and fixed relation-state atlas export.

This script consolidates the frozen Phase487-492 evidence. It deliberately
keeps behavior qualification, observational physical prediction, output
binding, and causal closure as separate ledgers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests" / "gpt5" / "result"
OUT_DIR = RESULT_ROOT / "phase493_relation_state_stage_audit"
OUT_PATH = OUT_DIR / "phase493_relation_state_stage_audit.json"
ATLAS_DIR = ROOT / "frontend" / "public" / "vis_data" / "phase492_relation_state_atlas"

CORRECTED_PHASE486 = (
    RESULT_ROOT
    / "phase487_dual_observer_native_core_protocol"
    / "phase487_phase486_corrected_audit.json"
)
BEHAVIOR_PATH = (
    RESULT_ROOT
    / "phase489_three_channel_behavior_analysis"
    / "phase489_three_channel_behavior_analysis.json"
)
CONTAMINATION_PATH = (
    RESULT_ROOT
    / "phase491_geometry_contamination_audit"
    / "phase491_geometry_contamination_audit.json"
)
PHYSICAL_DIR = RESULT_ROOT / "phase492_independent_relation_prediction"

MODELS = ("qwen3", "glm4", "deepseek7b")
PHYSICAL_MODELS = ("qwen3", "glm4")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱GLM4）",
    "deepseek7b": "DS7B（深度求索7B）",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def concise_track(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_candidate": track["semantic_candidate"],
        "label_candidate": track["label_candidate"],
        "strict_event": track["strict_event"],
        "unrecoverable": track["unrecoverable"],
    }


def model_behavior_summary(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "row_count": payload["row_count"],
        "native_semantic_intersection": payload["native_semantic_intersection"],
        "native_label_intersection": payload["native_label_intersection"],
        "identity": concise_track(payload["tracks"]["identity"]),
        "native_plain_candidate": concise_track(
            payload["tracks"]["native_plain_candidate"]
        ),
        "formal_light_stress_semantic": payload["tracks"]["formal_light_stress"][
            "semantic_candidate"
        ],
        "order_stress_semantic": payload["tracks"]["order_stress_claim_first"][
            "semantic_candidate"
        ],
        "gates": payload["gates"],
    }


def physical_summary(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "status": payload["status"],
        "gate_pass": payload["gate_pass"],
        "window": payload["frozen_window"],
        "geometry": payload["independent_geometry"],
        "truth_prediction": payload["independent_truth_prediction"],
        "cuda_used": payload["cuda_used"],
        "sealed_split_read": payload["sealed_split_read"],
        "raw_hidden_states_saved": payload["raw_hidden_states_saved"],
    }


def behavior_node(model: str, summary: dict[str, Any]) -> dict[str, Any]:
    intersection = summary["native_semantic_intersection"]
    passed = summary["gates"]["relation_semantic_pass"]
    return {
        "id": f"phase492:{model}:behavior:relation_semantics",
        "label": f"{MODEL_LABELS[model]} / 关系语义行为门",
        "type": "relation_semantic_behavior_gate",
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "relation_truth_state_candidate",
        "layer": -1,
        "relative_depth": 0.0,
        "position_role": "direct_true_false_output_boundary",
        "position": [-12.0, 0.0, 0.0],
        "score": intersection["rate"],
        "count": intersection["count"],
        "n": intersection["n"],
        "lcb95": intersection["lcb95"],
        "gate_pass": passed,
        "color": "#22c55e" if passed else "#dc2626",
        "size": 0.8,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "frozen_behavior_qualification",
        "show_label": True,
    }


def failed_interface_nodes(model: str, summary: dict[str, Any], y: float) -> list[dict[str, Any]]:
    label = summary["native_label_intersection"]
    identity_strict = summary["identity"]["strict_event"]
    plain_strict = summary["native_plain_candidate"]["strict_event"]
    strict_count = identity_strict["count"] + plain_strict["count"]
    strict_n = identity_strict["n"] + plain_strict["n"]
    return [
        {
            "id": f"phase492:{model}:interface:label_binding",
            "label": f"{MODEL_LABELS[model]} / 标签绑定未通过",
            "type": "unclosed_interface_gate",
            "model": model,
            "layer": -1,
            "relative_depth": 1.0,
            "position_role": "mapped_label_boundary",
            "position": [12.0, y, -4.0],
            "score": label["rate"],
            "count": label["count"],
            "n": label["n"],
            "gate_pass": False,
            "color": "#f97316",
            "size": 0.66,
            "physical": False,
            "observer": True,
            "predictive": False,
            "causal": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "behavior_failure_boundary",
            "show_label": True,
        },
        {
            "id": f"phase492:{model}:interface:strict_output",
            "label": f"{MODEL_LABELS[model]} / 严格输出未通过",
            "type": "unclosed_interface_gate",
            "model": model,
            "layer": -1,
            "relative_depth": 1.0,
            "position_role": "exact_single_label_event",
            "position": [18.0, y, 4.0],
            "score": rate(strict_count, strict_n),
            "count": strict_count,
            "n": strict_n,
            "gate_pass": False,
            "color": "#dc2626",
            "size": 0.66,
            "physical": False,
            "observer": True,
            "predictive": False,
            "causal": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "behavior_failure_boundary",
            "show_label": True,
        },
    ]


def physical_payload(
    model: str,
    behavior: dict[str, Any],
    physical: dict[str, Any],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    window = physical["window"]
    prediction = physical["truth_prediction"]["overall"]
    y = round(window["normalized_depth"] * 100.0, 4)
    nodes = [behavior_node(model, behavior)]
    nodes.append({
        "id": f"phase492:{model}:artifact:claim_embedding",
        "label": f"{MODEL_LABELS[model]} / L0 词元伪迹（已排除）",
        "type": "invalidated_lexical_artifact",
        "model": model,
        "layer": 0,
        "relative_depth": 0.0,
        "position_role": "claim_end",
        "position": [0.0, 0.0, -8.0],
        "score": artifact["q_native"],
        "direction_coherence": artifact["relation_direction_coherence"],
        "classification": artifact["classification"],
        "invalid_relation_evidence": True,
        "color": "#64748b",
        "size": 0.55,
        "physical": True,
        "observer": True,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "invalidated_lexical_artifact",
        "show_label": True,
    })
    nodes.append({
        "id": f"phase492:{model}:late_relation_state",
        "label": f"{MODEL_LABELS[model]} / L{window['layer_with_embedding']} 晚层关系状态",
        "type": "observational_relation_state_window",
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "late_terminal_relation_state_candidate",
        "layer": window["layer_with_embedding"],
        "relative_depth": window["normalized_depth"],
        "position_role": "prompt_end",
        "position": [0.0, y, 0.0],
        "score": physical["geometry"]["q_native"],
        "q_native": physical["geometry"]["q_native"],
        "direction_coherence": physical["geometry"]["relation_direction_coherence"],
        "family_q": {
            key: value["q_native"]
            for key, value in physical["geometry"]["by_family"].items()
        },
        "gate_pass": physical["gate_pass"],
        "color": "#10b981",
        "size": 1.0,
        "physical": True,
        "observer": True,
        "predictive": True,
        "causal": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "independent_open_observational_prediction",
        "show_label": True,
    })
    nodes.append({
        "id": f"phase492:{model}:independent_prediction",
        "label": f"{MODEL_LABELS[model]} / 独立预测 {prediction['count']}/{prediction['n']}",
        "type": "independent_observational_prediction_gate",
        "model": model,
        "layer": window["layer_with_embedding"],
        "relative_depth": window["normalized_depth"],
        "position_role": "physical_prediction_split",
        "position": [8.0, y, 0.0],
        "score": prediction["rate"],
        "count": prediction["count"],
        "n": prediction["n"],
        "lcb95": prediction["lcb95"],
        "gate_pass": physical["gate_pass"],
        "color": "#22c55e",
        "size": 0.88,
        "physical": False,
        "observer": True,
        "predictive": True,
        "causal": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "independent_open_prediction",
        "show_label": True,
    })
    nodes.extend(failed_interface_nodes(model, behavior, y))
    edges = [
        {
            "id": f"phase492:{model}:behavior_to_window",
            "source": f"phase492:{model}:behavior:relation_semantics",
            "target": f"phase492:{model}:late_relation_state",
            "type": "measurement_authorization",
            "label": "行为合格后允许物理观察",
            "score": behavior["native_semantic_intersection"]["rate"],
            "compute_edge": False,
            "predictive": False,
            "causal": False,
            "evidence_level": "protocol_gate",
        },
        {
            "id": f"phase492:{model}:window_to_prediction",
            "source": f"phase492:{model}:late_relation_state",
            "target": f"phase492:{model}:independent_prediction",
            "type": "observational_prediction",
            "label": "冻结方向的独立开放预测",
            "score": prediction["rate"],
            "compute_edge": False,
            "predictive": True,
            "causal": False,
            "evidence_level": "independent_open_prediction",
        },
    ]
    return {
        "schema_version": "phase492_relation_state_atlas_graph.v1",
        "phase_id": "Phase487-493-RelationStateAudit",
        "model": model,
        "title": f"Phase492 {MODEL_LABELS[model]} 晚层关系状态独立预测图谱",
        "evidence_scope": (
            "independent open observational late prompt-end relation-state prediction; "
            "not causal, sealed, component-localized, neuronal, or mechanism closure"
        ),
        "graph": {
            "meta": {
                "relation_semantic_behavior_pass": True,
                "independent_physical_prediction_pass": True,
                "label_binding_pass": False,
                "strict_output_event_pass": False,
                "sealed_split_read": False,
                "causal": False,
                "single_neuron": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def unqualified_payload(model: str, behavior: dict[str, Any]) -> dict[str, Any]:
    nodes = [behavior_node(model, behavior)]
    nodes.extend(failed_interface_nodes(model, behavior, 18.0))
    nodes.append({
        "id": f"phase492:{model}:physical:not_authorized",
        "label": f"{MODEL_LABELS[model]} / 物理采集未授权",
        "type": "closed_evidence_gate",
        "model": model,
        "layer": -1,
        "relative_depth": 0.5,
        "position_role": "not_collected",
        "position": [0.0, 48.0, 0.0],
        "score": 0.0,
        "gate_pass": False,
        "color": "#64748b",
        "size": 0.72,
        "physical": False,
        "observer": False,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "not_authorized_after_behavior_failure",
        "show_label": True,
    })
    return {
        "schema_version": "phase492_relation_state_atlas_graph.v1",
        "phase_id": "Phase487-493-RelationStateAudit",
        "model": model,
        "title": f"Phase492 {MODEL_LABELS[model]} 关系状态资格失败图谱",
        "evidence_scope": (
            "behavior qualification failed on the frozen relation interface; "
            "physical data were not collected and no route is displayed"
        ),
        "graph": {
            "meta": {
                "relation_semantic_behavior_pass": False,
                "independent_physical_prediction_pass": False,
                "physical_stage_run": False,
                "label_binding_pass": False,
                "strict_output_event_pass": False,
                "sealed_split_read": False,
                "causal": False,
                "single_neuron": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": [],
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)

    corrected = load_json(CORRECTED_PHASE486)
    behavior_raw = load_json(BEHAVIOR_PATH)
    contamination = load_json(CONTAMINATION_PATH)
    physical_raw = {
        model: load_json(PHYSICAL_DIR / f"phase492_{model}_summary.json")
        for model in PHYSICAL_MODELS
    }

    core = corrected["corrected_core_tracks"]
    behavior = {
        model: model_behavior_summary(model, behavior_raw["models"][model])
        for model in MODELS
    }
    physical = {
        model: physical_summary(model, physical_raw[model])
        for model in PHYSICAL_MODELS
    }

    assert core["n"] == 4608
    assert core["semantic"]["correct"] == 4067
    assert behavior["qwen3"]["gates"]["relation_semantic_pass"]
    assert behavior["glm4"]["gates"]["relation_semantic_pass"]
    assert not behavior["deepseek7b"]["gates"]["relation_semantic_pass"]
    assert all(physical[model]["gate_pass"] for model in PHYSICAL_MODELS)
    assert not any(behavior[model]["gates"]["label_binding_pass"] for model in MODELS)
    assert not any(behavior[model]["gates"]["strict_output_event_pass"] for model in MODELS)

    artifacts = {row["model"]: row for row in contamination["lexical_artifacts"]}
    atlas_payloads: dict[str, dict[str, Any]] = {}
    for model in PHYSICAL_MODELS:
        atlas_payloads[model] = physical_payload(
            model, behavior[model], physical[model], artifacts[model]
        )
    atlas_payloads["deepseek7b"] = unqualified_payload(
        "deepseek7b", behavior["deepseek7b"]
    )

    for model, payload in atlas_payloads.items():
        filename = f"phase492_{model}_relation_state.json"
        (ATLAS_DIR / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "phase492_relation_state_atlas_manifest.v1",
        "generated_at": created_at,
        "default_item_id": "phase492_qwen3_relation_state",
        "items": [
            {
                "id": f"phase492_{model}_relation_state",
                "label": f"Phase492 {MODEL_LABELS[model]} 关系状态图谱",
                "filename": f"phase492_{model}_relation_state.json",
                "model": model,
                "phase": 492,
                "evidence_scope": atlas_payloads[model]["evidence_scope"],
            }
            for model in MODELS
        ],
    }
    (ATLAS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregate = {
        "schema_version": "phase493_relation_state_stage_audit.v1",
        "created_at": created_at,
        "status": "stage_complete_with_independent_observational_endpoint",
        "phase486_correction": {
            "legacy_claim_semantic_rate": rate(4196, 4608),
            "corrected_core_semantic": {
                **core["semantic"],
                "rate": rate(core["semantic"]["correct"], core["n"]),
            },
            "corrected_core_strict_exact": {
                **core["strict_exact"],
                "rate": rate(core["strict_exact"]["correct"], core["n"]),
            },
            "duplicate_prompt_finding": (
                "Phase484 identity and core_surface_plain prompts were byte-identical"
            ),
            "verdict": (
                "the three-channel decomposition was directionally correct, but 91.06% "
                "was not a valid frozen target-blind semantic estimate"
            ),
        },
        "fresh_behavior": behavior,
        "independent_open_physical_prediction": physical,
        "invalidated_artifacts": contamination["lexical_artifacts"],
        "evidence_ledger": {
            "relation_semantic_behavior_pass_models": ["qwen3", "glm4"],
            "relation_semantic_behavior_fail_models": ["deepseek7b"],
            "independent_observational_prediction_pass_models": ["qwen3", "glm4"],
            "label_binding_pass_models": [],
            "strict_output_event_pass_models": [],
            "sealed_split_read": False,
            "causal_intervention_run": False,
            "component_localization_run": False,
            "head_channel_neuron_scan_run": False,
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
        },
        "allowed_claims": [
            "Qwen3 and GLM4 solve the frozen direct relation-semantics interface.",
            "A frozen late prompt-end direction independently predicts relation truth on a second open split for Qwen3 and GLM4.",
            "The finding is a model-specific observational endpoint puzzle and narrows a future causal search window.",
        ],
        "forbidden_claims": [
            "Phase486 achieved 91.06% frozen target-blind semantic recovery.",
            "The late direction is the upstream relation-computation mechanism.",
            "Label binding, strict serialization, causal mediation, components, heads, channels, or neurons are closed.",
            "All three small models share a universal coding direction.",
            "One endpoint puzzle closes any of the 72 mechanisms.",
        ],
        "progress": {
            "strict_mechanism_closure": {"closed": 0, "total": 72, "rate": 0.0},
            "project_maturity_point_percent": 25,
            "project_maturity_plausible_range_percent": [24, 26],
            "reason": (
                "one relation-state endpoint now has independent open physical prediction, "
                "while upstream formation, transport, interface binding, causal necessity, "
                "neuron localization, sealed replication, and broad family coverage remain open"
            ),
        },
        "next_stage_decision": {
            "current_stage_complete": True,
            "automatic_same_stage_continuation": False,
            "next_phase": 494,
            "title": "上游形成边界与跨关系族外推门",
            "reason": (
                "the next work changes the scientific object from endpoint observation to "
                "upstream formation and held-out family transport; it requires a new frozen "
                "protocol and must not consume sealed or causal data automatically"
            ),
            "ordered_gates": [
                "add at least four held-out relation families with length and order controls",
                "fit on two families and predict held-out families at source, claim, and prompt roles",
                "localize the earliest stable formation boundary without selecting on the test split",
                "only after cross-family prediction, run residual-path necessity and sufficiency with matched random controls",
                "only after route-level causal passage, authorize component, head, channel, and neuron refinement",
            ],
        },
        "atlas_export": {
            "manifest": str((ATLAS_DIR / "manifest.json").relative_to(ROOT)),
            "models": list(MODELS),
            "shows_failed_gates": True,
            "draws_unverified_compute_path": False,
        },
    }
    OUT_PATH.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT_PATH)
    print(ATLAS_DIR / "manifest.json")


if __name__ == "__main__":
    main()
