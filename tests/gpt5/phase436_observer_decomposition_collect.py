#!/usr/bin/env python3
"""Collect Phase436 decomposed observers and authorized downstream traces."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase435_natural_relation_collect as engine  # noqa: E402
import phase435_natural_relation_protocol as p435  # noqa: E402
from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase436_observer_decomposition_protocol import (  # noqa: E402
    CONTRACTS,
    DTYPES,
    GENERIC_CONTROL,
    INTERFACES,
    MODELS,
    OUT,
    P435_OUT,
    PHYSICAL_SPLIT,
    SEALED_SPLIT,
    freeze,
    read_json,
    read_jsonl,
)


PHASE_ID = "Phase436-ObserverDecompositionCollection"
engine.OUT = OUT


def observer_contracts(model: str) -> list[str]:
    path = OUT / "phase436_observer_freeze.json"
    if not path.exists():
        return []
    return sorted(
        contract
        for contract, payload in read_json(path)["models"][model]["contracts"].items()
        if payload["observer_qualified"]
    )


def selected_interface(model: str, contract: str) -> str:
    return str(
        read_json(OUT / "phase436_observer_freeze.json")["models"][model]["contracts"][contract]["selected_interface"]
    )


def behavior_contracts(model: str) -> list[str]:
    path = OUT / "phase436_behavior_gate.json"
    if not path.exists():
        return []
    return sorted(
        row["contract"]
        for row in read_json(path).get("eligible_model_contracts", [])
        if row["model"] == model
    )


def sealed_contracts(model: str) -> list[str]:
    path = OUT / "phase436_open_gate.json"
    if not path.exists() or not read_json(path).get("sealed_unlock"):
        return []
    return sorted(
        row["contract"]
        for row in read_json(path).get("sealed_authorized_model_contracts", [])
        if row["model"] == model
    )


def phase435_group_path(split: str) -> Path:
    if split == SEALED_SPLIT:
        return P435_OUT / "sealed/phase435_groups_sealed.jsonl"
    return P435_OUT / f"phase435_groups_{split}.jsonl"


def factors() -> list[tuple[str, str, str]]:
    return [
        (order, mapping, query_role)
        for order in p435.RECORD_ORDERS
        for mapping in p435.MAPPINGS
        for query_role in p435.QUERY_ROLES
    ]


def materialize_interface(loaded: Any) -> list[dict[str, Any]]:
    rows = []
    for group in read_jsonl(OUT / "phase436_groups_interface_calibration.jsonl"):
        contract = group["contract_variants"][0]
        for interface in INTERFACES:
            row = engine.materialize_condition(
                group,
                contract,
                interface,
                group["baseline_record_order"],
                group["baseline_mapping"],
                group["baseline_query_role"],
                loaded,
            )
            row["phase_id"] = PHASE_ID
            rows.append(row)
    return sorted(rows, key=lambda row: row["condition_id"])


def materialize_behavior(loaded: Any) -> list[dict[str, Any]]:
    rows = []
    for split in p435.BEHAVIOR_SPLITS:
        for group in read_jsonl(phase435_group_path(split)):
            for contract in observer_contracts(loaded.key):
                interface = selected_interface(loaded.key, contract)
                for order, mapping, query_role in factors():
                    row = engine.materialize_condition(
                        group, contract, interface, order, mapping, query_role, loaded
                    )
                    row["phase_id"] = PHASE_ID
                    rows.append(row)
    return sorted(rows, key=lambda row: row["condition_id"])


def generic_control(
    group: dict[str, Any], model: str, contract: str, interface: str, order: str, mapping: str, query_role: str, loaded: Any
) -> dict[str, Any]:
    row = engine.materialize_condition(
        group, GENERIC_CONTROL, interface, order, mapping, query_role, loaded
    )
    reference = f"{GENERIC_CONTROL}_for_{contract}"
    row["contract"] = reference
    row["condition_kind"] = "generic_control"
    row["reference_contract"] = contract
    row["condition_id"] = f"{row['condition_id']}__reference_{contract}"
    row["phase_id"] = PHASE_ID
    return row


def materialize_physical(loaded: Any, stage: str) -> list[dict[str, Any]]:
    contracts = behavior_contracts(loaded.key) if stage == "physical" else sealed_contracts(loaded.key)
    split = PHYSICAL_SPLIT if stage == "physical" else SEALED_SPLIT
    rows = []
    for group in read_jsonl(phase435_group_path(split)):
        for contract in contracts:
            interface = selected_interface(loaded.key, contract)
            for order, mapping, query_role in factors():
                candidate = engine.materialize_condition(
                    group, contract, interface, order, mapping, query_role, loaded
                )
                candidate["phase_id"] = PHASE_ID
                rows.append(candidate)
                rows.append(
                    generic_control(
                        group,
                        loaded.key,
                        contract,
                        interface,
                        order,
                        mapping,
                        query_role,
                        loaded,
                    )
                )
    return sorted(rows, key=lambda row: row["condition_id"])


def collect(model: str, stage: str, mode: str) -> dict[str, Any]:
    freeze()
    if stage == "behavior" and not observer_contracts(model):
        return {"model": model, "stage": stage, "skipped": True, "reason": "no_qualified_semantic_observer"}
    if stage == "physical" and not behavior_contracts(model):
        return {"model": model, "stage": stage, "skipped": True, "reason": "no_behavior_eligible_contract"}
    if stage == "sealed" and not sealed_contracts(model):
        return {"model": model, "stage": stage, "skipped": True, "reason": "sealed_not_authorized"}
    loaded = None
    try:
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != DTYPES[model]:
            raise RuntimeError(f"Execution dtype mismatch: {actual_dtype} != {DTYPES[model]}")
        if stage == "interface":
            rows = materialize_interface(loaded)
        elif stage == "behavior":
            rows = materialize_behavior(loaded)
        else:
            rows = materialize_physical(loaded, stage)
        output = {
            "phase_id": PHASE_ID,
            "model": model,
            "stage": stage,
            "condition_count": len(rows),
            "behavior": None,
            "physical": None,
        }
        if mode in {"behavior", "all"}:
            output["behavior"] = engine.collect_behavior(loaded, model, stage, rows)
        if mode in {"physical", "all"}:
            behavior_complete = OUT / stage / model / "behavior/phase435_behavior_complete.json"
            if not behavior_complete.exists():
                output["behavior"] = engine.collect_behavior(loaded, model, stage, rows)
            output["physical"] = engine.collect_physical(loaded, model, stage, rows)
        return output
    finally:
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("interface", "behavior", "physical", "sealed"), required=True)
    parser.add_argument("--mode", choices=("behavior", "physical", "all"), default="behavior")
    args = parser.parse_args()
    print(json.dumps(collect(args.model, args.stage, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
