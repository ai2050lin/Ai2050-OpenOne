#!/usr/bin/env python3
"""Audit generated semantic events without rewriting the frozen Phase518 gate.

Phase519 deliberately used an exact-whole-response parser.  This audit keeps that
result intact and adds a second, target-blind ledger for the first complete
canonical event.  Discovery can authorize an untouched confirmation split; only
confirmation can authorize physical tracing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")
Z = 1.96

STAGES = {
    "discovery": {
        "input_phase": 519,
        "input_dir": ROOT / "tests/gpt5/result/phase519_natural_relation_binding_calibration",
        "expected_relation_rows": 384,
        "output_phase": 520,
        "output_dir": ROOT / "tests/gpt5/result/phase520_behavior_authorization",
        "output_name": "phase520_behavior_authorization.json",
    },
    "confirmation": {
        "input_phase": 521,
        "input_dir": ROOT / "tests/gpt5/result/phase521_natural_relation_binding_confirmation",
        "expected_relation_rows": 768,
        "output_phase": 522,
        "output_dir": ROOT / "tests/gpt5/result/phase522_semantic_event_confirmation",
        "output_name": "phase522_physical_authorization.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * ((p * (1 - p) + Z * Z / (4 * n)) / n) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    n = len(rows)
    count = sum(bool(row[field]) for row in rows)
    lower, upper = wilson(count, n)
    return {
        "n": n,
        "count": count,
        "rate": count / n if n else 0.0,
        "lcb95": lower,
        "ucb95": upper,
    }


def paired_rate(
    rows: list[dict[str, Any]],
    field: str,
    group_field: str,
    expected_count: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(row)
    records = []
    for key, items in groups.items():
        if len(items) != expected_count:
            raise RuntimeError(f"incomplete {group_field} group {key}: {len(items)}")
        records.append({"all_correct": all(bool(item[field]) for item in items)})
    return rate(records, "all_correct")


def parse_first_natural_event(text: str) -> dict[str, Any]:
    """Parse only the first canonical sentence, without access to target truth."""

    normalized = " ".join(text.lstrip().split())
    match = re.match(
        r"^(The statement is supported\.|The statement is contradicted\.)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return {"value": None, "recoverable": False, "event_type": "no_canonical_first_event"}
    value = match.group(1).lower() == "the statement is supported."
    return {
        "value": value,
        "recoverable": True,
        "event_type": "first_supported" if value else "first_contradicted",
    }


def parse_first_mapping_event(text: str) -> dict[str, Any]:
    normalized = " ".join(text.lstrip().split())
    match = re.match(r"^The holding symbol is ([AB01])\.", normalized, flags=re.IGNORECASE)
    return {
        "symbol": match.group(1).upper() if match else None,
        "recoverable": bool(match),
    }


def relation_audit(
    rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    for row in rows:
        parsed = parse_first_natural_event(row["generated_natural_text"])
        row["first_event_value"] = parsed["value"]
        row["first_event_recoverable"] = parsed["recoverable"]
        row["first_event_correct"] = parsed["value"] is not None and parsed["value"] == row["truth_value"]

    first_by_surface = {
        surface: rate([row for row in rows if row["surface"] == surface], "first_event_correct")
        for surface in SURFACES
    }
    candidate_by_surface = {
        surface: rate([row for row in rows if row["surface"] == surface], "candidate_correct")
        for surface in SURFACES
    }
    first_intersection = paired_rate(rows, "first_event_correct", "sample_id", 2)
    first_four_way = paired_rate(rows, "first_event_correct", "source_pair_id", 4)
    candidate_intersection = paired_rate(rows, "candidate_correct", "sample_id", 2)
    candidate_four_way = paired_rate(rows, "candidate_correct", "source_pair_id", 4)
    unrecoverable_rows = [dict(row, first_event_unrecoverable=not row["first_event_recoverable"]) for row in rows]
    first_unrecoverable = rate(unrecoverable_rows, "first_event_unrecoverable")

    gate = contract["gates"]["natural_relation"]
    passed = (
        all(item["lcb95"] >= gate["surface_lcb95_min"] for item in first_by_surface.values())
        and first_intersection["lcb95"] >= gate["surface_intersection_lcb95_min"]
        and first_four_way["lcb95"] >= gate["four_way_lcb95_min"]
        and all(item["lcb95"] >= gate["candidate_surface_lcb95_min"] for item in candidate_by_surface.values())
        and first_unrecoverable["ucb95"] <= gate["unrecoverable_ucb95_max"]
    )
    return {
        "ledger_status": "posthoc_discovery" if rows[0]["split"] == "calibration" else "independent_confirmation",
        "first_event_by_surface": first_by_surface,
        "first_event_surface_intersection": first_intersection,
        "first_event_four_way": first_four_way,
        "first_event_unrecoverable": first_unrecoverable,
        "candidate_by_surface": candidate_by_surface,
        "candidate_surface_intersection": candidate_intersection,
        "candidate_four_way": candidate_four_way,
        "first_event_gate_pass": passed,
    }


def mapping_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "not_run"}
    for row in rows:
        parsed = parse_first_mapping_event(row["generated_mapping_text"])
        row["first_mapping_recoverable"] = parsed["recoverable"]
        row["first_mapping_correct"] = parsed["symbol"] == row["holding_symbol"]
    return {
        "status": "exploratory_only_not_an_authorization_gate",
        "overall": rate(rows, "first_mapping_correct"),
        "by_label_system": {
            label: rate([row for row in rows if row["label_system"] == label], "first_mapping_correct")
            for label in ("mapped_ab", "mapped_01")
        },
    }


def audit_stage(stage: str) -> Path:
    config = STAGES[stage]
    contract = read_json(
        ROOT / "tests/gpt5/result/phase518_world_query_platform_protocol/phase518_frozen_contract.json"
    )
    model_reports: dict[str, Any] = {}
    relation_models: list[str] = []
    binding_models: list[str] = []

    for model in MODELS:
        summary_path = config["input_dir"] / f"phase{config['input_phase']}_{model}_summary.json"
        if not summary_path.exists():
            model_reports[model] = {"status": "missing_summary"}
            continue
        source_summary = read_json(summary_path)
        relation_path = config["input_dir"] / f"phase{config['input_phase']}_{model}_relation_rows.jsonl"
        if source_summary["status"] != "complete" or not relation_path.exists():
            model_reports[model] = {
                "status": source_summary["status"],
                "source_summary_sha256": sha256_file(summary_path),
                "first_event_gate_pass": False,
            }
            continue
        relation_rows = read_jsonl(relation_path)
        if len(relation_rows) != config["expected_relation_rows"]:
            raise RuntimeError(f"unexpected relation row count for {model}: {len(relation_rows)}")
        relation_report = relation_audit(relation_rows, contract)

        mapping_path = config["input_dir"] / f"phase{config['input_phase']}_{model}_mapping_rows.jsonl"
        mapping_rows = read_jsonl(mapping_path) if mapping_path.exists() else []
        strict_relation = source_summary["contract_summaries"].get("R_natural", {})
        strict_binding = source_summary["contract_summaries"].get("B_ledger", {})
        strict_binding_pass = bool(strict_binding.get("gate_pass", False))
        model_reports[model] = {
            "status": "audited",
            "source_summary_sha256": sha256_file(summary_path),
            "source_relation_rows_sha256": sha256_file(relation_path),
            "strict_whole_response_relation_gate_pass": bool(strict_relation.get("gate_pass", False)),
            "strict_binding_gate_pass": strict_binding_pass,
            "relation_first_event": relation_report,
            "mapping_first_event_diagnostic": mapping_diagnostic(mapping_rows),
        }
        if relation_report["first_event_gate_pass"]:
            relation_models.append(model)
        if strict_binding_pass:
            binding_models.append(model)

    config["output_dir"].mkdir(parents=True, exist_ok=True)
    output_path = config["output_dir"] / config["output_name"]
    payload = {
        "schema_version": f"phase{config['output_phase']}_semantic_event_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": "complete",
        "method": {
            "strict_frozen_gate_preserved": True,
            "first_event_parser_target_blind": True,
            "calibration_first_event_ledger_is_posthoc": stage == "discovery",
            "confirmation_split_untouched_when_discovery_parser_defined": True,
            "binding_authorization_uses_only_frozen_strict_gate": True,
            "sealed_split_read": False,
        },
        "relation_models": relation_models,
        "binding_models": binding_models,
        "model_reports": model_reports,
        "physical_authorized": stage == "confirmation" and bool(relation_models or binding_models),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    audit_stage(args.stage)


if __name__ == "__main__":
    main()
