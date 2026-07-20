#!/usr/bin/env python3
"""Freeze Qwen3 Phase571 matched and wrong-target donor assignments."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase571_relation_block_protocol as protocol  # noqa: E402


MODEL = "qwen3"
OUT_DIR = protocol.OUT_DIR
DECISION_PATH = OUT_DIR / "phase571_stage_decision.json"
RESERVE_PATH = OUT_DIR / f"phase571_{MODEL}_causal_reserve_summary.json"
CAUSAL_ROWS_PATH = OUT_DIR / f"phase571_{MODEL}_coarse_block_causal_rows.jsonl.gz"
BLOCK_REGISTRY_PATH = OUT_DIR / "phase571_continuous_block_registry.json"
DONOR_PROTOCOL_PATH = OUT_DIR / "phase571_relation_donor_frozen_protocol.json"
DONOR_REGISTRY_PATH = OUT_DIR / "phase571_relation_donor_registry.json"
CONDITIONS = (
    "baseline",
    "self_entry_restore",
    "matched_correct_answer_entry",
    "matched_correct_answer_exit",
    "matched_correct_query_entry",
    "matched_correct_target_fact_entry",
    "wrong_target_answer_entry",
    "random_matched_answer_entry",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["source_factorial_cell"], row["target"], row["other_relation_target"]


def freeze() -> dict[str, Any]:
    decision = read_json(DECISION_PATH)
    if MODEL not in decision["coarse_causal_gate_passed_models"]:
        raise RuntimeError("Phase571 donor stage is not authorized for Qwen3")
    reserve = read_json(RESERVE_PATH)
    block_registry = read_json(BLOCK_REGISTRY_PATH)
    block = block_registry["selected_block_by_model"][MODEL]
    ids = reserve["selected_case_ids_by_phenotype"]
    labels = {
        **{case_id: "stable_correct" for case_id in ids["stable_correct"]},
        **{
            case_id: "stable_relation_confusion"
            for case_id in ids["stable_relation_confusion"]
        },
    }
    bank = {
        row["case_id"]: {**row, "causal_phenotype": labels[row["case_id"]]}
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["model"] == MODEL and row["case_id"] in labels
    }
    ordered_pairs = list(zip(ids["stable_correct"], ids["stable_relation_confusion"]))
    if len(ordered_pairs) < 128 or len(bank) != len(labels):
        raise RuntimeError("Phase571 donor pair denominator drift")
    correct_rows = [bank[left] for left, _right in ordered_pairs]
    correct_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in correct_rows:
        correct_by_stratum[stratum(row)].append(row)
    correct_index = {row["case_id"]: index for index, row in enumerate(correct_rows)}

    def foreign_same_stratum(receiver: dict[str, Any]) -> dict[str, Any]:
        options = correct_by_stratum[stratum(receiver)]
        if receiver["case_id"] not in {row["case_id"] for row in options}:
            return options[0]
        if len(options) == 1:
            return options[0]
        position = next(
            index for index, row in enumerate(options) if row["case_id"] == receiver["case_id"]
        )
        return options[(position + 1) % len(options)]

    def wrong_target(receiver: dict[str, Any]) -> dict[str, Any]:
        ordered = sorted(correct_rows, key=lambda row: row["case_id"])
        preferred = [
            row for row in ordered
            if row["target"] == receiver["other_relation_target"]
            and row["target"] != receiver["target"]
        ]
        fallback = [row for row in ordered if row["target"] != receiver["target"]]
        candidates = preferred or fallback
        if not candidates:
            raise RuntimeError("Phase571 could not assign a wrong-target donor")
        offset = int(hashlib.sha256(receiver["case_id"].encode("utf-8")).hexdigest()[:8], 16)
        return candidates[offset % len(candidates)]

    entries = []
    for pair_index, (correct_id, confusion_id) in enumerate(ordered_pairs):
        correct = bank[correct_id]
        confusion = bank[confusion_id]
        if stratum(correct) != stratum(confusion):
            raise RuntimeError("Phase571 donor matched-pair stratum drift")
        for receiver in (correct, confusion):
            matched_donor = (
                foreign_same_stratum(receiver)
                if receiver["causal_phenotype"] == "stable_correct"
                else correct
            )
            wrong_donor = wrong_target(receiver)
            entries.append({
                "pair_index": pair_index,
                "receiver_case_id": receiver["case_id"],
                "receiver_phenotype": receiver["causal_phenotype"],
                "matched_correct_donor_case_id": matched_donor["case_id"],
                "wrong_target_donor_case_id": wrong_donor["case_id"],
                "receiver_target": receiver["target"],
                "receiver_other_relation_target": receiver["other_relation_target"],
                "matched_donor_target": matched_donor["target"],
                "wrong_donor_target": wrong_donor["target"],
                "matched_donor_same_factor_and_value_pair": (
                    stratum(receiver) == stratum(matched_donor)
                ),
                "wrong_donor_target_differs": wrong_donor["target"] != receiver["target"],
            })
    if len(entries) != len(ordered_pairs) * 2:
        raise RuntimeError("Phase571 donor registry count drift")
    if any(
        not entry["matched_donor_same_factor_and_value_pair"]
        or not entry["wrong_donor_target_differs"]
        for entry in entries
    ):
        raise RuntimeError("Phase571 donor control identity failed")
    registry = {
        "schema_version": "phase571_relation_donor_registry.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "selected_block": block,
        "candidate_pair_count": len(ordered_pairs),
        "final_pair_count": 128,
        "receiver_count": len(entries),
        "entries": entries,
        "selection_uses_intervention_outcomes": False,
        "sealed_split_read": False,
    }
    write_json(DONOR_REGISTRY_PATH, registry)
    frozen = {
        "schema_version": "phase571_relation_donor_frozen_protocol.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "conditions": list(CONDITIONS),
        "candidate_receiver_count_per_phenotype": len(ordered_pairs),
        "final_receiver_count_per_phenotype": 128,
        "fixed_batch_size": 8,
        "entry_layer": block["start_layer"],
        "exit_layer": block["end_layer"],
        "answer_role": "answer_boundary",
        "query_role": "query_relation",
        "target_fact_role": "target_fact_value",
        "donor_gate": {
            "minimum_confusion_repair": 0.10,
            "minimum_specificity_over_wrong_target": 0.10,
            "minimum_correct_preservation": 0.90,
            "minimum_self_restore_semantic_match": 0.95,
            "minimum_query_or_fact_repair_for_upstream_claim": 0.10,
            "exit_only_repair_is_terminal_content_not_relation_selection": True,
        },
        "registry_sha256": sha256_file(DONOR_REGISTRY_PATH),
        "coarse_causal_rows_sha256": sha256_file(CAUSAL_ROWS_PATH),
        "selection_uses_intervention_outcomes": False,
        "head_channel_parameter_neuron_scan_allowed": False,
        "sealed_split_read": False,
    }
    write_json(DONOR_PROTOCOL_PATH, frozen)
    print(json.dumps({
        "model": MODEL,
        "candidate_pairs": len(ordered_pairs),
        "final_pairs": 128,
        "candidate_receivers": len(entries),
        "entry_layer": block["start_layer"],
        "exit_layer": block["end_layer"],
        "conditions": list(CONDITIONS),
    }, ensure_ascii=False, indent=2))
    return frozen


if __name__ == "__main__":
    freeze()
