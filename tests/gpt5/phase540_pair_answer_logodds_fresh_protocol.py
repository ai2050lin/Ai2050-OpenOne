#!/usr/bin/env python3
"""Freeze independent pair-addressed confirmation splits for Phase539 candidates."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase535_pair_addressed_binding_protocol as base  # noqa: E402


OUT_DIR = ROOT / "tests/gpt5/result/phase540_pair_answer_logodds_fresh_protocol"
PHASE535_CONTRACT = ROOT / "tests/gpt5/result/phase535_pair_addressed_binding_protocol/phase535_frozen_contract.json"
PHASE539_AUTH = ROOT / "tests/gpt5/result/phase539_pair_answer_logodds_observer/phase539_fresh_confirmation_authorization.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
PHASE535_OPEN_SPLITS = ("discovery", "entity_prediction", "relation_prediction")

SPLITS = {
    "fresh_vocabulary_confirmation": {
        "index": 0,
        "group_count": 128,
        "sealed": False,
        "entity_pool": 10,
        "relation_pool": 10,
    },
    "fresh_relation_confirmation": {
        "index": 1,
        "group_count": 128,
        "sealed": False,
        "entity_pool": 11,
        "relation_pool": 11,
    },
    "sealed": {
        "index": 2,
        "group_count": 128,
        "sealed": True,
        "entity_pool": 12,
        "relation_pool": 12,
    },
}

FRESH_ENTITY_POOLS = {
    10: ("Arden", "Bianca", "Cedric", "Delia", "Emmett", "Flora", "Gideon", "Helena", "Isolde", "Kellan", "Leona", "Magnus"),
    11: ("Nadia", "Orson", "Priya", "Quentin", "Rhea", "Silas", "Talia", "Ulric", "Vera", "Wesley", "Yara", "Zoren"),
    12: ("Alina", "Bram", "Celeste", "Devon", "Elara", "Finnian", "Gemma", "Hadrian", "Ilona", "Jarek", "Kyra", "Lysander"),
}

FRESH_RELATION_POOLS = {
    10: ("signals", "notifies"),
    11: ("audits", "tracks"),
    12: ("mirrors", "follows"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_phase_ids(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ("sample_id", "source_group_id", "world_surface_id", "pair_flip_id"):
            row[key] = str(row[key]).replace("phase535:", "phase540:", 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase535 = read_json(PHASE535_CONTRACT)
    phase539 = read_json(PHASE539_AUTH)
    candidates = list(phase539["fresh_confirmation_required_models"])
    if candidates != ["qwen3"]:
        raise RuntimeError(f"unexpected frozen Phase539 candidates: {candidates}")

    base.ENTITY_POOLS.update(FRESH_ENTITY_POOLS)
    base.RELATION_POOLS.update(FRESH_RELATION_POOLS)
    audits: dict[str, Any] = {}
    split_files: dict[str, Any] = {}
    entity_sets: dict[str, set[str]] = {}
    relation_sets: dict[str, set[str]] = {}
    for split, spec in SPLITS.items():
        rows = base.split_rows(split, spec)
        rewrite_phase_ids(rows)
        path = OUT_DIR / f"phase540_{split}.jsonl"
        base.write_jsonl(path, rows)
        audits[split] = base.audit_split(split, rows, spec)
        entity_sets[split] = {name for row in rows for name in row["entity_names"]}
        relation_sets[split] = {str(row["relation_active"]) for row in rows}
        split_files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": base.sha256_file(path),
            "sealed": bool(spec["sealed"]),
            "row_count": len(rows),
            "source_group_count": int(spec["group_count"]),
        }

    prior_entities: set[str] = set()
    prior_relations: set[str] = set()
    for split in PHASE535_OPEN_SPLITS:
        spec = phase535["split_files"][split]
        rows = [json.loads(line) for line in (ROOT / spec["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        prior_entities.update(name for row in rows for name in row["entity_names"])
        prior_relations.update(str(row["relation_active"]) for row in rows)

    fresh_entity_disjoint = all(
        not (entity_sets[left] & entity_sets[right])
        for index, left in enumerate(SPLITS)
        for right in list(SPLITS)[index + 1 :]
    )
    fresh_relation_disjoint = all(
        not (relation_sets[left] & relation_sets[right])
        for index, left in enumerate(SPLITS)
        for right in list(SPLITS)[index + 1 :]
    )
    prior_disjoint = not (prior_entities & set().union(*entity_sets.values())) and not (
        prior_relations & set().union(*relation_sets.values())
    )
    audit_keys = (
        "row_count_pass",
        "sixteen_way_group_pass",
        "world_surface_four_way_pass",
        "pair_status_flip_pass",
        "matched_fact_token_bag_pass",
        "pair_ledger_world_identity_pass",
        "query_section_separated_from_world_prefix_pass",
        "slot_label_balance_pass",
    )
    static_pass = fresh_entity_disjoint and fresh_relation_disjoint and prior_disjoint and all(
        report[key] for report in audits.values() for key in audit_keys
    )

    observer_ledgers: dict[str, Any] = {}
    for model in candidates:
        ledger_path = ROOT / f"tests/gpt5/result/phase539_pair_answer_logodds_observer/phase539_{model}_frozen_discovery_ledger.json"
        observer_ledgers[model] = {
            "path": str(ledger_path.relative_to(ROOT)),
            "sha256": base.sha256_file(ledger_path),
        }

    contract = {
        "schema_version": "phase540_pair_answer_logodds_fresh_protocol.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_fresh_model_execution" if static_pass else "static_fail",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": base.sha256_file(SOURCE),
        "models_in_required_order": list(MODELS),
        "fresh_confirmation_candidate_models": candidates,
        "phase535_contract_path": str(PHASE535_CONTRACT.relative_to(ROOT)),
        "phase535_contract_sha256": base.sha256_file(PHASE535_CONTRACT),
        "phase539_authorization_path": str(PHASE539_AUTH.relative_to(ROOT)),
        "phase539_authorization_sha256": base.sha256_file(PHASE539_AUTH),
        "phase535_prior_splits_read": list(PHASE535_OPEN_SPLITS),
        "phase535_sealed_split_read_by_current_protocol": False,
        "historical_phase535_sealed_compromised_by_initial_phase540_audit": True,
        "frozen_observer_ledgers": observer_ledgers,
        "split_files": split_files,
        "behavior_gate": phase535["behavior_gate"],
        "confirmation_rules": [
            "No threshold, direction, continuation, or gate may be fitted on Phase540 data.",
            "Only Phase539-qualified models may load weights.",
            "Both open confirmation splits must pass; the sealed split remains unread.",
            "A pass authorizes observational physical collection, not a mechanism or causal claim.",
        ],
        "evidence_boundaries": {
            "fresh_vocabulary": True,
            "fresh_relations": True,
            "current_phase540_sealed_read": False,
            "historical_phase535_sealed_read": True,
            "physical_state_collected": False,
            "causal": False,
        },
    }
    contract_path = OUT_DIR / "phase540_frozen_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "phase540_pair_answer_logodds_fresh_static_audit.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if static_pass else "static_fail",
        "splits": audits,
        "fresh_entity_pool_disjoint_pass": fresh_entity_disjoint,
        "fresh_relation_pool_disjoint_pass": fresh_relation_disjoint,
        "phase535_open_vocabulary_disjoint_pass": prior_disjoint,
        "phase535_prior_splits_read": list(PHASE535_OPEN_SPLITS),
        "phase535_sealed_split_read_by_current_protocol": False,
        "historical_phase535_sealed_compromised_by_initial_phase540_audit": True,
        "open_row_count": sum(spec["row_count"] for spec in split_files.values() if not spec["sealed"]),
        "sealed_row_count": split_files["sealed"]["row_count"],
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": base.sha256_file(contract_path),
        "model_run": False,
        "current_phase540_sealed_split_read_by_downstream": False,
    }
    audit_path = OUT_DIR / "phase540_static_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(contract_path)
    print(audit_path)
    if not static_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
