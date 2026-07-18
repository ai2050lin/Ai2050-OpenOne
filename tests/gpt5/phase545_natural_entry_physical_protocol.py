#!/usr/bin/env python3
"""Freeze same-contract physical discovery/prediction rows for Phase544 entries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
OUT_DIR = ROOT / "tests/gpt5/result/phase545_natural_entry_physical_path"
MATRIX = SOURCE / "phase544_model_mechanism_qualification.jsonl"
CASES = SOURCE / "phase544_registered_cases.jsonl"
PAIRS_PATH = OUT_DIR / "phase545_registered_physical_pairs.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase545_physical_protocol.json"
AUDIT_PATH = OUT_DIR / "phase545_static_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
PAIR_LIMIT = 24


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def register() -> dict[str, Any]:
    matrix = read_jsonl(MATRIX)
    eligible = {
        (row["model"], row["family_id"], row["mechanism_id"])
        for row in matrix if row["behavior_entry_eligible"]
    }
    cases = [
        row for row in read_jsonl(CASES)
        if row["surface"] == "direct_natural"
        and row["pair_index"] < PAIR_LIMIT
        and (row["model"], row["family_id"], row["mechanism_id"]) in eligible
    ]
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        groups[(
            row["model"], row["family_id"], row["mechanism_id"],
            row["split"], row["pair_index"],
        )].append(row)
    pair_rows = []
    for (model, family, mechanism, split, pair_index), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["world_condition"])
        if len(rows) != 2 or {row["world_condition"] for row in rows} != {"world_a", "world_b"}:
            raise RuntimeError((model, family, mechanism, split, pair_index, len(rows)))
        pair_rows.append({
            "schema_version": "phase545_natural_entry_physical_protocol.v1",
            "phase_id": "Phase545",
            "created_at": now(),
            "physical_pair_id": f"phase545_{model}_{family}_{mechanism}_{split}_{pair_index:03d}",
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "split": split,
            "pair_index": pair_index,
            "world_a_case_id": rows[0]["case_id"],
            "world_b_case_id": rows[1]["case_id"],
            "world_a_target": rows[0]["target"],
            "world_b_target": rows[1]["target"],
            "contrast_kind": "same_contract_counterfactual_world_pair",
            "surface": "direct_natural",
            "stages": ["prompt_end", "after_first_generated_token", "after_third_generated_token"],
            "components": ["layer_input", "attention_output", "mlp_output", "layer_output"],
            "roles": ["source", "query", "current"],
            "observer_only": True,
            "predictive": False,
            "causal": False,
            "single_neuron": False,
            "sealed": False,
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(PAIRS_PATH, pair_rows)
    counts = Counter(row["model"] for row in pair_rows)
    cells = Counter((row["model"], row["family_id"], row["mechanism_id"]) for row in pair_rows)
    audit = {
        "schema_version": "phase545_natural_entry_physical_static_audit.v1",
        "phase_id": "Phase545",
        "created_at": now(),
        "behavior_eligible_cell_count": len(eligible),
        "physical_pair_count": len(pair_rows),
        "physical_pair_count_by_model": dict(counts),
        "pair_count_per_model_mechanism": sorted(set(cells.values())),
        "target_identity_error_count": sum(row["world_a_target"] == row["world_b_target"] for row in pair_rows),
        "sealed_pair_count": sum(row["sealed"] for row in pair_rows),
        "deepseek_gate_skip": not any(row[0] == "deepseek7b" for row in eligible),
        "valid": (
            len(eligible) == 9
            and len(pair_rows) == 432
            and dict(counts) == {"glm4": 240, "qwen3": 192}
            and sorted(set(cells.values())) == [48]
            and all(row["world_a_target"] != row["world_b_target"] for row in pair_rows)
            and not any(row["sealed"] for row in pair_rows)
        ),
    }
    audit["status"] = "static_pass_no_hidden_state_read" if audit["valid"] else "static_fail"
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": "phase545_natural_entry_physical_protocol.v1",
        "phase_id": "Phase545",
        "created_at": now(),
        "title": "Same-contract full-layer multi-position natural-entry physical paths",
        "models_in_required_order": list(MODELS),
        "gate_skipped_models": ["deepseek7b"],
        "pair_limit_per_split_model_mechanism": PAIR_LIMIT,
        "discovery_split": "discovery",
        "independent_prediction_split": "independent_confirmation",
        "frozen_event_selection": {
            "score": "median normalized world-pair delta multiplied by positive direction-support fraction",
            "event_axes": ["stage", "component", "role", "layer"],
            "selection_split": "discovery",
            "selection_updates_after_prediction": False,
        },
        "prediction_gates": {
            "same_event_positive_direction_fraction_min": 0.75,
            "same_event_median_direction_alignment_min": 0.0,
            "peak_layer_relative_error_max": 0.10,
            "component_ledger_relative_error_max": 0.01,
        },
        "claim_boundaries": {
            "observer_event_is_compute_edge": False,
            "observer_event_is_causal": False,
            "world_pair_delta_is_abstract_semantics": False,
            "explicit_source_copy_is_pretrained_knowledge": False,
            "single_neuron_scan_allowed": False,
            "sealed_split_read": False,
        },
        "registered_pairs_path": str(PAIRS_PATH.relative_to(ROOT)),
        "registered_pairs_sha256": sha256_file(PAIRS_PATH),
        "source_behavior_cases_sha256": sha256_file(CASES),
        "source_behavior_matrix_sha256": sha256_file(MATRIX),
        "static_audit": audit,
    }
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    register()
