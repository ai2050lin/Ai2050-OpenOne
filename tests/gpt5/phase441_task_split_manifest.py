#!/usr/bin/env python3
"""Generate the Phase441 task, orbit, and split manifest before any CUDA run."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from phase439_natural_stable_entry_protocol import (
    OUT_DIR,
    OUT_PATH as PROTOCOL_PATH,
    build_protocol,
)


OUT_PATH = OUT_DIR / "phase441_task_split_manifest.json"
GROUPS_PER_SPLIT = 64


ABILITY_ROLE_MAPS = {
    "knowledge_network": {
        "semantic_nodes": ["entity", "attribute", "category", "query"],
        "role_map_required_for": ["synonym_rewrite", "order_swap", "distance_change", "query_expression_rewrite"],
        "expected_state": "object_attribute_binding",
    },
    "single_step_reasoning": {
        "semantic_nodes": ["premise_a", "premise_b", "rule", "query"],
        "role_map_required_for": ["order_swap", "distance_change", "label_or_structure_order_change"],
        "expected_state": "premise_rule_transition",
    },
    "syntax_system": {
        "semantic_nodes": ["controller", "slot", "distractor", "boundary"],
        "role_map_required_for": ["active_passive_role_conversion", "boundary_rewrite", "distance_change"],
        "expected_state": "structural_role_binding",
    },
}


def stable_id(parts: list[str]) -> str:
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_manifest() -> dict:
    protocol = build_protocol()
    entries = []
    for ability, tasks in protocol["task_library"].items():
        role_contract = ABILITY_ROLE_MAPS[ability]
        for task in tasks:
            for split in protocol["splits"]:
                for group_index in range(GROUPS_PER_SPLIT):
                    family_id = stable_id([ability, task, split, f"{group_index:03d}"])
                    entries.append(
                        {
                            "sample_family_id": family_id,
                            "ability": ability,
                            "task": task,
                            "split": split,
                            "base_group_index": group_index,
                            "leakage_group_key": f"{ability}/{task}/{split}/{family_id}",
                            "surface_transforms": protocol["surface_transforms"],
                            "interfaces": protocol["interfaces"],
                            "semantic_nodes": role_contract["semantic_nodes"],
                            "role_map_required_for": role_contract["role_map_required_for"],
                            "expected_state": role_contract["expected_state"],
                            "answer_alias_policy": "freeze_aliases_before_behavior_run",
                            "semantic_preservation_proof": "required_before_cuda",
                            "node_mapping_status": "pre_registered_required",
                        }
                    )

    manifest = {
        "schema_version": "phase441_task_split_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "sample_split_manifest_frozen_no_cuda_run",
        "protocol_schema_version": protocol["schema_version"],
        "protocol_path": str(PROTOCOL_PATH.relative_to(Path(__file__).resolve().parents[2])),
        "groups_per_task": GROUPS_PER_SPLIT * len(protocol["splits"]),
        "groups_per_split": GROUPS_PER_SPLIT,
        "total_sample_families": len(entries),
        "split_names": protocol["splits"],
        "ability_role_contracts": ABILITY_ROLE_MAPS,
        "static_contract_checks_required": [
            "lexical_disjoint_across_splits",
            "template_disjoint_across_splits",
            "answer_distribution_balance",
            "position_length_balance",
            "shortcut_baseline_cannot_perfectly_predict",
            "semantic_transform_preserves_answer",
            "node_role_mapping_is_legal",
            "task_difficulty_not_single_token_dependent",
            "multi_token_event_length_matched",
            "all_artifacts_have_frozen_hash",
        ],
        "entries": entries,
    }
    frozen = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(frozen).hexdigest()
    return manifest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL_PATH.exists():
        PROTOCOL_PATH.write_text(json.dumps(build_protocol(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_PATH.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
