#!/usr/bin/env python3
"""Analyze Phase436 semantic-content observers and authorized physical paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase435_natural_relation_analysis as a435  # noqa: E402
from phase436_observer_decomposition_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    CONTRACTS,
    GENERIC_CONTROL,
    INTERFACE_SIMPLICITY,
    INTERFACES,
    MODELS,
    OUT,
    SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    write_json,
)


PHASE_ID = "Phase436-ObserverDecompositionAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase436_observer_decomposition"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODEL_COLORS = {"qwen3": "#22c55e", "glm4": "#0ea5e9", "deepseek7b": "#f97316"}
CONTRACT_LABELS = {
    "field_extract": "字段抽取",
    "natural_qa": "自然问答",
    "relation_rewrite": "关系改写",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase436 non-finite scalar: {value}")
    return round(float(value), 9)


def semantic_content_good(row: dict[str, Any]) -> bool:
    return bool(
        row["teacher_sequence_correct"]
        and row["actual_choice"] == row["semantic_target_source"]
        and row["natural_target_first"]
        and not row["natural_opposite_first"]
        and not row["natural_revision"]
    )


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "condition_count": total,
        "semantic_content": a435.wilson(sum(semantic_content_good(row) for row in rows), total),
        "teacher": a435.wilson(sum(bool(row["teacher_sequence_correct"]) for row in rows), total),
        "registered_value": a435.wilson(sum(row["actual_choice"] != "other" for row in rows), total),
        "exact_format": a435.wilson(sum(bool(row["natural_interface_valid"]) for row in rows), total),
        "exact_target_format": a435.wilson(sum(bool(row["natural_exact_target_contract"]) for row in rows), total),
        "stop": a435.wilson(sum(bool(row["natural_stop_good"]) for row in rows), total),
        "other": a435.wilson(sum(row["actual_choice"] == "other" for row in rows), total),
        "choice_counts": dict(Counter(str(row["actual_choice"]) for row in rows)),
    }


def position_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        position: metrics([row for row in rows if row["target_position"] == position])
        for position in ("first", "second")
    }
    output["semantic_position_gap"] = clean(
        abs(
            float(output["first"]["semantic_content"]["estimate"])
            - float(output["second"]["semantic_content"]["estimate"])
        )
    )
    return output


def behavior_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "behavior/phase435_behavior_rows.jsonl"


def physical_path(stage: str, model: str) -> Path:
    return OUT / stage / model / "physical/phase435_physical_rows.jsonl.gz"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_observer_model(model: str) -> dict[str, Any]:
    rows = read_jsonl(behavior_path("interface", model))
    thresholds = read_json(OUT / "phase436_protocol.json")["interface_semantic_gate"]
    contracts = {}
    for contract in CONTRACTS:
        interfaces = {}
        for interface in INTERFACES:
            selected = [
                row
                for row in rows
                if row["contract"] == contract and row["interface"] == interface
            ]
            aggregate = metrics(selected)
            positions = position_metrics(selected)
            passed = bool(
                all(
                    positions[position]["semantic_content"]["lcb"]
                    >= thresholds["per_position_wilson_lcb"]
                    and positions[position]["teacher"]["lcb"]
                    >= thresholds["teacher_per_position_wilson_lcb"]
                    for position in ("first", "second")
                )
                and positions["semantic_position_gap"] <= thresholds["maximum_position_gap"]
                and aggregate["other"]["ucb"] <= thresholds["other_wilson_ucb"]
            )
            interfaces[interface] = {
                "metrics": aggregate,
                "positions": positions,
                "pass": passed,
                "selection_rank": [
                    min(
                        float(positions[position]["semantic_content"]["lcb"])
                        for position in ("first", "second")
                    ),
                    -float(positions["semantic_position_gap"]),
                    -float(aggregate["other"]["ucb"]),
                    -INTERFACE_SIMPLICITY.index(interface),
                ],
            }
        qualified = [interface for interface in INTERFACE_SIMPLICITY if interfaces[interface]["pass"]]
        selected_interface = (
            qualified[0]
            if qualified
            else max(
                INTERFACES,
                key=lambda interface: tuple(interfaces[interface]["selection_rank"]),
            )
        )
        contracts[contract] = {
            "interfaces": interfaces,
            "qualified_interfaces": qualified,
            "selected_interface": selected_interface,
            "observer_qualified": bool(qualified),
            "selection_unit": "model_x_contract",
        }
    return {
        "model": model,
        "row_count": len(rows),
        "contracts": contracts,
        "semantic_content_excludes_exact_format_and_stop": True,
    }


def analyze_observer_freeze() -> dict[str, Any]:
    freeze()
    input_hashes = {
        model: sha256_file(behavior_path("interface", model)) for model in MODELS
    }
    path = OUT / "phase436_observer_freeze.json"
    if path.exists():
        existing = read_json(path)
        if existing["input_hashes"] != input_hashes:
            raise RuntimeError("Phase436 observer calibration changed after freeze")
        return existing
    models = {model: analyze_observer_model(model) for model in MODELS}
    qualified = [
        {"model": model, "contract": contract, "interface": payload["selected_interface"]}
        for model, model_payload in models.items()
        for contract, payload in model_payload["contracts"].items()
        if payload["observer_qualified"]
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "input_hashes": input_hashes,
        "models": models,
        "qualified_model_contracts": qualified,
        "qualified_count": len(qualified),
        "frozen_before_phase435_behavior_denominator_read": True,
        "format_and_stop_not_used_as_semantic_content": True,
    }
    write_json(path, output)
    return output


def analyze_behavior_model(model: str, observer: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(behavior_path("behavior", model))
    thresholds = read_json(OUT / "phase436_protocol.json")["behavior_gate"]
    contracts = {}
    for contract, observer_payload in observer["models"][model]["contracts"].items():
        if not observer_payload["observer_qualified"]:
            contracts[contract] = {
                "observer_qualified": False,
                "splits": {},
                "behavior_eligible": False,
            }
            continue
        interface = observer_payload["selected_interface"]
        contract_rows = [row for row in rows if row["contract"] == contract]
        if any(row["interface"] != interface for row in contract_rows):
            raise RuntimeError(f"Phase436 interface drift for {model}/{contract}")
        splits = {}
        for split in BEHAVIOR_SPLITS:
            selected = [row for row in contract_rows if row["split"] == split]
            aggregate = metrics(selected)
            positions = position_metrics(selected)
            passed = bool(
                all(
                    positions[position]["semantic_content"]["lcb"]
                    >= thresholds["discovery_and_holdout_per_position_wilson_lcb"]
                    and positions[position]["teacher"]["lcb"]
                    >= thresholds["teacher_per_position_wilson_lcb"]
                    for position in ("first", "second")
                )
                and positions["semantic_position_gap"] <= thresholds["maximum_position_gap"]
                and (
                    split != "behavior_holdout"
                    or aggregate["other"]["ucb"] <= thresholds["holdout_other_wilson_ucb"]
                )
            )
            splits[split] = {"metrics": aggregate, "positions": positions, "pass": passed}
        contracts[contract] = {
            "observer_qualified": True,
            "selected_interface": interface,
            "splits": splits,
            "behavior_eligible": all(value["pass"] for value in splits.values()),
            "format_and_stop_reported_separately": True,
        }
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "row_count": len(rows),
        "contracts": contracts,
    }
    write_json(OUT / f"phase436_{model}_behavior_audit.json", output)
    return output


def analyze_behavior_gate() -> dict[str, Any]:
    observer = read_json(OUT / "phase436_observer_freeze.json")
    behavior = {}
    for model in MODELS:
        qualified = any(
            payload["observer_qualified"]
            for payload in observer["models"][model]["contracts"].values()
        )
        if qualified:
            if not behavior_path("behavior", model).exists():
                raise RuntimeError(f"Phase436 qualified behavior rows missing for {model}")
            behavior[model] = analyze_behavior_model(model, observer)
        else:
            behavior[model] = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": model,
                "row_count": 0,
                "status": "observer_gate_failed_behavior_unread",
                "contracts": {
                    contract: {
                        "observer_qualified": False,
                        "splits": {},
                        "behavior_eligible": False,
                    }
                    for contract in CONTRACTS
                },
            }
            write_json(OUT / f"phase436_{model}_behavior_audit.json", behavior[model])
    eligible = [
        {"model": model, "contract": contract}
        for model in MODELS
        for contract, payload in behavior[model]["contracts"].items()
        if payload["behavior_eligible"]
    ]
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "behavior": behavior,
        "eligible_model_contracts": eligible,
        "eligible_count": len(eligible),
        "physical_unlock": bool(eligible),
        "sealed_rows_read": False,
    }
    write_json(OUT / "phase436_behavior_gate.json", output)
    return output


def analyze_physical_model(model: str, stage: str = "physical") -> dict[str, Any]:
    protocol = read_json(OUT / "phase436_protocol.json")
    thresholds = protocol["physical_numeric_gates"]
    geometry = a435.geometry_ledger(physical_path(stage, model))
    transport = a435.transport_ledger(physical_path(stage, model))
    identity_pass = bool(
        geometry["max_block_reconstruction_relative_error"]
        <= thresholds["component_reconstruction_relative_error_max"]
        and geometry["max_attention_replay_relative_error"]
        <= thresholds["attention_replay_relative_error_max"]
    )
    if stage == "physical":
        contracts = [
            row["contract"]
            for row in read_json(OUT / "phase436_behavior_gate.json")["eligible_model_contracts"]
            if row["model"] == model
        ]
        open_audit = None
    else:
        open_gate = read_json(OUT / "phase436_open_gate.json")
        contracts = [
            row["contract"]
            for row in open_gate["sealed_authorized_model_contracts"]
            if row["model"] == model
        ]
        open_audit = read_json(OUT / f"phase436_{model}_physical_audit.json")
    contract_results = {}
    for contract in contracts:
        result = a435.evaluate_physical_contract(
            contract,
            geometry,
            transport,
            thresholds,
            stage=stage,
            open_freeze=(open_audit["contracts"][contract] if open_audit else None),
            generic_contract=f"{GENERIC_CONTROL}_for_{contract}",
        )
        result["gates"]["G2_component_identity_and_position_registration"] = identity_pass
        contract_results[contract] = result
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "identity": {
            "max_block_reconstruction_relative_error": geometry["max_block_reconstruction_relative_error"],
            "max_attention_replay_relative_error": geometry["max_attention_replay_relative_error"],
            "pass": identity_pass,
        },
        "contracts": contract_results,
        "geometry_ledger": geometry,
        "transport_ledger": transport,
        "source_transport_not_inferred_from_geometry": True,
        "physical": True,
        "observer": True,
        "predictive": any(
            payload["gates"]["G5_frozen_holdout_prediction"]
            for payload in contract_results.values()
        ),
        "causal": False,
        "single_neuron": False,
    }
    suffix = "physical_audit" if stage == "physical" else "sealed_physical_audit"
    write_json(OUT / f"phase436_{model}_{suffix}.json", output)
    return output


def analyze_open() -> dict[str, Any]:
    behavior = read_json(OUT / "phase436_behavior_gate.json")
    models = sorted({row["model"] for row in behavior["eligible_model_contracts"]})
    physical = {
        model: analyze_physical_model(model)
        for model in models
        if physical_path("physical", model).exists()
    }
    audits = []
    authorized = []
    for row in behavior["eligible_model_contracts"]:
        model, contract = row["model"], row["contract"]
        candidate = physical.get(model, {}).get("contracts", {}).get(contract)
        gates = {
            "G0_semantic_observer_by_model_contract": True,
            "G1_natural_content_and_order_balance": True,
            "G2_component_identity_and_position_registration": bool(
                candidate and candidate["gates"]["G2_component_identity_and_position_registration"]
            ),
            "G3_label_blind_order_geometry": bool(candidate and candidate["gates"]["G3_label_blind_order_geometry"]),
            "G4_semantic_source_transport": bool(candidate and candidate["gates"]["G4_semantic_source_transport"]),
            "G5_frozen_holdout_prediction": bool(candidate and candidate["gates"]["G5_frozen_holdout_prediction"]),
            "G6_generic_pairing_specificity": bool(candidate and candidate["gates"]["G6_generic_pairing_specificity"]),
        }
        audit = {"model": model, "contract": contract, "gates": gates, "pass": all(gates.values())}
        audits.append(audit)
        if audit["pass"]:
            authorized.append({"model": model, "contract": contract})
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": "open",
        "physical": physical,
        "model_contract_gates": audits,
        "sealed_authorized_model_contracts": authorized,
        "sealed_unlock": bool(authorized),
        "sealed_rows_read": False,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "phase436_open_gate.json", output)
    return output


def sealed_behavior(model: str, contract: str) -> dict[str, Any]:
    rows = [row for row in read_jsonl(behavior_path("sealed", model)) if row["contract"] == contract]
    aggregate = metrics(rows)
    positions = position_metrics(rows)
    passed = bool(
        all(positions[position]["semantic_content"]["lcb"] >= 0.80 for position in ("first", "second"))
        and positions["semantic_position_gap"] <= 0.05
        and aggregate["other"]["ucb"] <= 0.05
    )
    return {"metrics": aggregate, "positions": positions, "pass": passed}


def analyze_sealed() -> dict[str, Any]:
    open_gate = read_json(OUT / "phase436_open_gate.json")
    if not open_gate["sealed_unlock"]:
        raise RuntimeError("Phase436 sealed analysis is not authorized")
    models = sorted({row["model"] for row in open_gate["sealed_authorized_model_contracts"]})
    physical = {
        model: analyze_physical_model(model, "sealed")
        for model in models
        if physical_path("sealed", model).exists()
    }
    results = []
    for row in open_gate["sealed_authorized_model_contracts"]:
        model, contract = row["model"], row["contract"]
        behavior = sealed_behavior(model, contract)
        candidate = physical.get(model, {}).get("contracts", {}).get(contract)
        physical_pass = bool(candidate and all(candidate["gates"].values()))
        results.append(
            {
                "model": model,
                "contract": contract,
                "sealed_behavior": behavior,
                "sealed_physical_pass": physical_pass,
                "G7_sealed_physical_replication": bool(behavior["pass"] and physical_pass),
            }
        )
    output = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "results": results,
        "sealed_pass": bool(results) and all(row["G7_sealed_physical_replication"] for row in results),
        "sealed_rows_read": True,
        "causal": False,
        "single_neuron": False,
    }
    write_json(OUT / "sealed/phase436_sealed_result.json", output)
    return output


def build_summary() -> dict[str, Any]:
    protocol = read_json(OUT / "phase436_protocol.json")
    observer = read_json(OUT / "phase436_observer_freeze.json")
    behavior_path_value = OUT / "phase436_behavior_gate.json"
    open_path = OUT / "phase436_open_gate.json"
    sealed_path = OUT / "sealed/phase436_sealed_result.json"
    behavior = read_json(behavior_path_value) if behavior_path_value.exists() else None
    open_gate = read_json(open_path) if open_path.exists() else None
    sealed = read_json(sealed_path) if sealed_path.exists() else None
    if not observer["qualified_model_contracts"]:
        status, progress = "semantic_observer_failed_behavior_unread", 21
    elif not behavior or not behavior["eligible_model_contracts"]:
        status, progress = "observer_passed_natural_behavior_failed", 21
    elif not open_gate or not open_gate["sealed_unlock"]:
        status, progress = "open_physical_gates_failed_or_pending", 22
    elif not sealed:
        status, progress = "open_physical_passed_sealed_pending", 22
    elif sealed["sealed_pass"]:
        status, progress = "sealed_observational_path_replication_passed", 24
    else:
        status, progress = "sealed_replication_failed", 22
    summary = {
        "schema_version": "phase436_observer_decomposition_summary.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "status": status,
        "denominator": protocol["denominator_audit"],
        "observer": observer,
        "behavior": behavior,
        "open": open_gate,
        "sealed": sealed,
        "evidence": {
            "semantic_observer": bool(observer["qualified_model_contracts"]),
            "format_and_stop_separate": True,
            "physical": bool(open_gate and open_gate["physical"]),
            "predictive": bool(open_gate and open_gate["sealed_authorized_model_contracts"]),
            "causal": False,
            "single_neuron": False,
            "mechanism_closure": False,
        },
        "closure": {
            "strict_mechanisms": "0/72",
            "overall_scientific_progress_percent": progress,
            "cautious_interval_percent": [max(0, progress - 3), progress + 3],
        },
    }
    write_json(OUT / "phase436_final_summary.json", summary)
    return summary


def publish_visual() -> dict[str, Any]:
    summary = build_summary()
    nodes = []
    edges = []
    for model_index, model in enumerate(MODELS):
        for contract_index, contract in enumerate(CONTRACTS):
            payload = summary["observer"]["models"][model]["contracts"][contract]
            interface = payload["selected_interface"]
            selected = payload["interfaces"][interface]
            score = statistics.mean(
                float(selected["positions"][position]["semantic_content"]["estimate"])
                for position in ("first", "second")
            )
            node_id = f"phase436:{model}:{contract}:observer"
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{model} / {CONTRACT_LABELS[contract]}",
                    "type": "decomposed_semantic_observer",
                    "model": model,
                    "contract": contract,
                    "interface": interface,
                    "layer": -1,
                    "relative_depth": 0.0,
                    "position_role": "first_second_balanced",
                    "position": [float(contract_index * 5), float(model_index * 6), -4.0],
                    "score": clean(score),
                    "format_score": selected["metrics"]["exact_format"]["estimate"],
                    "stop_score": selected["metrics"]["stop"]["estimate"],
                    "color": MODEL_COLORS[model],
                    "size": 1.0 if payload["observer_qualified"] else 0.6,
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "evidence_level": "fresh_semantic_observer_calibration",
                    "show_label": payload["observer_qualified"],
                }
            )
            if summary.get("behavior"):
                behavior = summary["behavior"]["behavior"][model]["contracts"][contract]
                if behavior["splits"]:
                    holdout = behavior["splits"]["behavior_holdout"]["positions"]
                    behavior_id = f"phase436:{model}:{contract}:behavior"
                    nodes.append(
                        {
                            "id": behavior_id,
                            "label": f"{model} / {CONTRACT_LABELS[contract]}留出",
                            "type": "natural_relation_behavior",
                            "model": model,
                            "contract": contract,
                            "layer": -1,
                            "relative_depth": 0.0,
                            "position_role": "behavior_holdout",
                            "position": [float(contract_index * 5 + 2), float(model_index * 6), -1.0],
                            "score": clean(statistics.mean(float(holdout[p]["semantic_content"]["estimate"]) for p in ("first", "second"))),
                            "color": MODEL_COLORS[model],
                            "size": 1.0,
                            "physical": False,
                            "observer": True,
                            "predictive": False,
                            "causal": False,
                            "single_neuron": False,
                            "pipeline_sealed": False,
                            "evidence_level": "independent_behavior_holdout",
                            "show_label": behavior["behavior_eligible"],
                        }
                    )
                    edges.append(
                        {
                            "id": f"{node_id}->{behavior_id}",
                            "source": node_id,
                            "target": behavior_id,
                            "type": "frozen_observer_application",
                            "physical": False,
                            "observer": True,
                            "predictive": False,
                            "causal": False,
                            "single_neuron": False,
                            "evidence_level": "protocol_dependency",
                            "color": "#64748b",
                            "weight": 0.5,
                        }
                    )
    physical_stage_run = bool(summary.get("open") and summary["open"]["physical"])
    evidence_scope = (
        "semantic-content observer with independent behavior and open physical tests; format and stop separated; non-causal"
        if physical_stage_run
        else "fresh semantic-content observer calibration; exact format and stop reported separately; physical and causal claims locked"
    )
    payload = {
        "schema_version": "phase436_observer_decomposition_graph.v1",
        "phase_id": PHASE_ID,
        "title": "Phase436 内容-格式-停止分账观察器",
        "model": "multi_model",
        "evidence_scope": evidence_scope,
        "graph": {
            "meta": {
                "qualified_model_contracts": summary["observer"]["qualified_model_contracts"],
                "physical_stage_run": physical_stage_run,
                "sealed_pass": bool(summary.get("sealed") and summary["sealed"]["sealed_pass"]),
                "causal": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    VIS.mkdir(parents=True, exist_ok=True)
    filename = "phase436_observer_decomposition.json"
    write_json(VIS / filename, payload)
    manifest = {
        "schema_version": "phase436_observer_decomposition_manifest.v1",
        "generated_at": now(),
        "default_item_id": "phase436_observer_decomposition",
        "items": [
            {
                "id": "phase436_observer_decomposition",
                "label": "Phase436 内容-格式-停止分账观察器",
                "filename": filename,
                "model": "multi_model",
                "phase": 436,
                "evidence_scope": evidence_scope,
            }
        ],
    }
    write_json(VIS / "manifest.json", manifest)
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase436_observer_decomposition",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase436 内容-格式-停止分账观察器",
        "description": "按模型与合同冻结语义内容观察器，精确格式和停止单独登记；后续分母按门读取。",
        "manifest_path": "/vis_data/phase436_observer_decomposition/manifest.json",
        "manifest_schema": "phase436_observer_decomposition_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase436_observer_decomposition",
        "models": list(MODELS),
        "evidence_scope": evidence_scope,
        "color": "#14b8a6",
    }
    registry["sources"] = [item for item in registry["sources"] if item["id"] != source["id"]] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    return {"manifest": manifest, "node_count": len(nodes), "edge_count": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("observer", "behavior", "open", "sealed", "summary"), required=True)
    parser.add_argument("--publish-visual", action="store_true")
    args = parser.parse_args()
    if args.stage == "observer":
        output = analyze_observer_freeze()
    elif args.stage == "behavior":
        output = analyze_behavior_gate()
    elif args.stage == "open":
        output = analyze_open()
    elif args.stage == "sealed":
        output = analyze_sealed()
    else:
        output = build_summary()
    if args.publish_visual:
        output = {"analysis": output, "visual": publish_visual()}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
