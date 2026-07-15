#!/usr/bin/env python3
"""Freeze Phase436 semantic/format/stop observer decomposition."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase435_natural_relation_protocol as p435  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase436_observer_decomposition"
P435_OUT = ROOT / "tests/gpt5/result/phase435_natural_relation"
PHASE_ID = "Phase436-ObserverDecompositionProtocol"
SCHEMA_VERSION = "phase436_observer_decomposition.v1"
INTERFACE_SPLIT = "phase436_interface_calibration"
CALIBRATION_GROUPS_PER_CONTRACT = 192
LEXICAL_OFFSET = 3000

MODELS = p435.MODELS
DTYPES = p435.DTYPES
CONTRACTS = p435.CONTRACTS
INTERFACES = p435.INTERFACES
INTERFACE_SIMPLICITY = p435.INTERFACE_SIMPLICITY
BEHAVIOR_SPLITS = p435.BEHAVIOR_SPLITS
PHYSICAL_SPLIT = p435.PHYSICAL_SPLIT
SEALED_SPLIT = p435.SEALED_SPLIT
GENERIC_CONTROL = p435.GENERIC_CONTROL


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_calibration_groups() -> list[dict[str, Any]]:
    rows = []
    serial = 0
    for contract in CONTRACTS:
        for _ in range(CALIBRATION_GROUPS_PER_CONTRACT):
            row = p435.build_group(
                p435.INTERFACE_SPLIT,
                serial,
                forced_contract=contract,
                lexical_index=LEXICAL_OFFSET + serial,
            )
            row.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "split": INTERFACE_SPLIT,
                    "semantic_group_id": f"phase436__interface__group_{serial:04d}",
                    "paired_group_id": f"phase436__interface__pair_{serial:04d}",
                    "pipeline_sealed": False,
                    "physical_fold": None,
                }
            )
            rows.append(row)
            serial += 1
    return rows


def denominator_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_contract = Counter(row["contract_variants"][0] for row in rows)
    phase435_protocol = read_json(P435_OUT / "phase435_protocol.json")
    calibration_vocab = {
        value.lower()
        for row in rows
        for value in (row["entity_a"], row["entity_b"], row["value_1"], row["value_2"])
    }
    reused_paths = [
        P435_OUT / "phase435_groups_behavior_discovery.jsonl",
        P435_OUT / "phase435_groups_behavior_holdout.jsonl",
        P435_OUT / "phase435_groups_physical_calibration.jsonl",
        P435_OUT / "sealed/phase435_groups_sealed.jsonl",
    ]
    reused_vocab = {
        value.lower()
        for path in reused_paths
        for row in read_jsonl(path)
        for value in (row["entity_a"], row["entity_b"], row["value_1"], row["value_2"])
    }
    factor_balance = {}
    for contract in CONTRACTS:
        selected = [row for row in rows if row["contract_variants"] == [contract]]
        factor_balance[contract] = {
            "relation_family": dict(Counter(row["relation_family"] for row in selected)),
            "record_order": dict(Counter(row["baseline_record_order"] for row in selected)),
            "mapping": dict(Counter(row["baseline_mapping"] for row in selected)),
            "query_role": dict(Counter(row["baseline_query_role"] for row in selected)),
        }
    valid = bool(
        len(rows) == 576
        and by_contract == Counter({contract: 192 for contract in CONTRACTS})
        and not calibration_vocab.intersection(reused_vocab)
        and len({row["semantic_group_id"] for row in rows}) == len(rows)
        and phase435_protocol["denominator_audit"]["valid"]
    )
    return {
        "valid": valid,
        "calibration_group_count": len(rows),
        "calibration_groups_per_contract": dict(by_contract),
        "calibration_conditions_per_model": len(rows) * len(INTERFACES),
        "three_model_calibration_conditions": len(rows) * len(INTERFACES) * len(MODELS),
        "fresh_calibration_vocabulary_disjoint_from_phase435": not calibration_vocab.intersection(reused_vocab),
        "reused_phase435_behavior_and_physical_denominators_remain_frozen": True,
        "phase435_behavior_discovery_and_holdout_were_not_used_for_observer_selection": True,
        "factor_balance": factor_balance,
    }


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase436_observer_decomposition_protocol.py",
        "phase436_observer_decomposition_collect.py",
        "phase436_observer_decomposition_analysis.py",
        "test_phase436_observer_decomposition.py",
    )
    return {
        name: (
            sha256_file(ROOT / "tests/gpt5" / name)
            if (ROOT / "tests/gpt5" / name).exists()
            else None
        )
        for name in names
    }


def freeze() -> dict[str, Any]:
    p435.freeze()
    rows = build_calibration_groups()
    audit = denominator_audit(rows)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "phase436_groups_interface_calibration.jsonl", rows)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models_in_execution_order": list(MODELS),
        "execution_dtypes": DTYPES,
        "contracts": list(CONTRACTS),
        "interfaces": list(INTERFACES),
        "interface_selection_unit": "model_x_contract",
        "observer_axes": ["semantic_content", "exact_format", "stop"],
        "semantic_content_excludes_exact_format_and_stop": True,
        "denominator_audit": audit,
        "calibration_rows_sha256": digest_rows(rows),
        "phase435_reused_rows_sha256": {
            "behavior_discovery": sha256_file(P435_OUT / "phase435_groups_behavior_discovery.jsonl"),
            "behavior_holdout": sha256_file(P435_OUT / "phase435_groups_behavior_holdout.jsonl"),
            "physical_calibration": sha256_file(P435_OUT / "phase435_groups_physical_calibration.jsonl"),
        },
        "phase435_sealed_commitment": read_json(P435_OUT / "phase435_sealed_commitment.json"),
        "interface_semantic_gate": {
            "per_position_wilson_lcb": 0.80,
            "maximum_position_gap": 0.05,
            "other_wilson_ucb": 0.10,
            "teacher_per_position_wilson_lcb": 0.80,
            "format_and_stop_are_reported_but_not_gating": True,
        },
        "behavior_gate": {
            "discovery_and_holdout_per_position_wilson_lcb": 0.80,
            "maximum_position_gap": 0.05,
            "holdout_other_wilson_ucb": 0.05,
            "teacher_per_position_wilson_lcb": 0.80,
            "format_and_stop_are_reported_but_not_gating": True,
        },
        "physical_numeric_gates": read_json(P435_OUT / "phase435_protocol.json")["numeric_gates"],
        "gate_order": [
            "G0_semantic_observer_by_model_contract",
            "G1_natural_content_and_order_balance",
            "G2_component_identity_and_position_registration",
            "G3_label_blind_order_geometry",
            "G4_semantic_source_transport",
            "G5_frozen_holdout_prediction",
            "G6_generic_pairing_specificity",
            "G7_sealed_physical_replication",
        ],
        "source_transport_is_not_inferred_from_geometry": True,
        "causal_and_single_neuron_forbidden": True,
        "implementation_hashes": implementation_hashes(),
    }
    write_json(OUT / "phase436_protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
