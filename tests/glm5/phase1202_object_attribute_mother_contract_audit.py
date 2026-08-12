#!/usr/bin/env python3
"""Independent zero-model audit for the Phase1202 mother-family contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"

CONTRACT_PATH = TEST_ROOT / "result/phase1202_object_attribute_mother_contract/protocol/mother_family_contract.json"
PACKAGE_PATH = TEST_ROOT / "result/phase1202_object_attribute_mother_contract/material/object_attribute_binding.jsonl"
TOKEN_AUDIT_PATH = TEST_ROOT / "result/phase1202_object_attribute_mother_contract/audit/tokenizer_audit.json"
SUMMARY_PATH = TEST_ROOT / "result/phase1202_object_attribute_mother_contract/analysis/readiness_summary.json"
AUDIT_PATH = TEST_ROOT / "result/phase1202_object_attribute_mother_contract/audit/independent_audit.json"
UPSTREAM_FINAL = TEST_ROOT / "result/phase1201_registry_identifiability_abstention/analysis/final.json"

EXPECTED_PHASE = 1202
EXPECTED_UPSTREAM_DIGEST = "0a0c5cee0f0ed305b35d959d5921f66c31474005b9fe04eee99ebcdbfb042b91"
EXPECTED_ROWS = 4608
EXPECTED_SPLITS = {"discovery": 2304, "confirmation": 1152, "unseen_composition": 1152}
EXPECTED_PANELS = {"active", "matched_null", "surface_only", "semantic_neighbor"}
EXPECTED_ATTRIBUTES = {"color", "material", "location", "size", "shape", "status"}
EXPECTED_WORLDS = {f"lexical_world_{index}" for index in range(4)}
EXPECTED_TEMPLATES = {"profile_prose", "compact_ledger"}
REQUIRED_FIELDS = {
    "item_id", "family", "world", "profile_id", "combination_id", "split",
    "panel", "template", "candidate_order", "binding_state", "attribute",
    "neighbor_attribute", "target_value", "entities", "record_order", "candidates",
    "gold_candidate", "gold_position", "assignments", "prompt",
    "prompt_token_multiset_digest", "probe_anchors",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def exact_truth(row: dict[str, Any]) -> tuple[bool, str | None]:
    matches = [
        entity
        for entity in row["entities"]
        if row["assignments"][entity][row["attribute"]] == row["target_value"]
    ]
    return len(matches) == 1, matches[0] if len(matches) == 1 else None


def audit(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(PACKAGE_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    summary = read_json(SUMMARY_PATH)
    upstream = read_json(UPSTREAM_FINAL)

    contract_body = {key: value for key, value in contract.items() if key != "contract_digest"}
    add(checks, "phase", contract.get("phase") == EXPECTED_PHASE)
    add(checks, "contract_digest", digest(contract_body) == contract.get("contract_digest"))
    add(checks, "upstream_digest", upstream.get("final_digest") == EXPECTED_UPSTREAM_DIGEST)
    add(
        checks,
        "upstream_file_hash",
        contract["upstream"]["upstream_hashes"]["phase1201_final_file"] == file_sha256(UPSTREAM_FINAL),
    )
    add(
        checks,
        "source_hashes",
        contract["source_hashes"]
        == {
            "generator": file_sha256(TEST_ROOT / "phase1202_object_attribute_mother_contract.py"),
            "independent_audit": file_sha256(Path(__file__).resolve()),
        },
    )
    add(checks, "zero_model_scope", contract["execution_policy"]["this_phase_model_execution"] is False)
    add(checks, "no_k_item", contract["execution_policy"]["this_phase_new_k_item"] is False)
    add(checks, "row_count", len(rows) == EXPECTED_ROWS, len(rows))
    add(checks, "unique_item_ids", len({row.get("item_id") for row in rows}) == len(rows))
    missing = sorted({field for row in rows for field in REQUIRED_FIELDS if field not in row})
    add(checks, "required_fields", not missing, missing)

    split_counts = Counter(row["split"] for row in rows)
    add(checks, "split_counts", dict(split_counts) == EXPECTED_SPLITS, dict(split_counts))
    add(checks, "panel_levels", {row["panel"] for row in rows} == EXPECTED_PANELS)
    add(checks, "attribute_levels", {row["attribute"] for row in rows} == EXPECTED_ATTRIBUTES)
    add(checks, "world_levels", {row["world"] for row in rows} == EXPECTED_WORLDS)
    add(checks, "template_levels", {row["template"] for row in rows} == EXPECTED_TEMPLATES)
    add(checks, "binding_levels", {row["binding_state"] for row in rows} == {0, 1})
    add(checks, "candidate_order_levels", {row["candidate_order"] for row in rows} == {0, 1, 2})

    combinations_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        combinations_by_split[row["split"]].add(row["combination_id"])
    split_pairs_disjoint = all(
        combinations_by_split[left].isdisjoint(combinations_by_split[right])
        for index, left in enumerate(EXPECTED_SPLITS)
        for right in tuple(EXPECTED_SPLITS)[index + 1 :]
    )
    add(checks, "combination_split_disjoint", split_pairs_disjoint)
    add(
        checks,
        "combination_counts",
        {split: len(values) for split, values in combinations_by_split.items()}
        == {"discovery": 48, "confirmation": 24, "unseen_composition": 24},
        {split: len(values) for split, values in combinations_by_split.items()},
    )

    truth_results = [exact_truth(row) for row in rows]
    add(checks, "unique_semantic_truth", all(valid for valid, _ in truth_results))
    add(
        checks,
        "gold_matches_truth",
        all(truth == row["gold_candidate"] for row, (_, truth) in zip(rows, truth_results)),
    )
    add(checks, "gold_in_candidates", all(row["gold_candidate"] in row["candidates"] for row in rows))
    add(
        checks,
        "gold_position_exact",
        all(row["candidates"][row["gold_position"]] == row["gold_candidate"] for row in rows),
    )

    state_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["combination_id"], row["panel"], row["template"], row["candidate_order"]
        )
        state_groups[key].append(row)
    add(checks, "state_pair_count", len(state_groups) == EXPECTED_ROWS // 2, len(state_groups))
    add(checks, "state_pair_complete", all({row["binding_state"] for row in group} == {0, 1} for group in state_groups.values()))

    active_flip = True
    controls_stable = True
    state_multiset_match = True
    active_swap_exact = True
    matched_anchor_exact = True
    surface_assignments_equal = True
    surface_order_changes = True
    neighbor_swap_exact = True
    for key, group in state_groups.items():
        by_state = {row["binding_state"]: row for row in group}
        if set(by_state) != {0, 1}:
            continue
        left, right = by_state[0], by_state[1]
        panel = key[1]
        state_multiset_match &= left["prompt_token_multiset_digest"] == right["prompt_token_multiset_digest"]
        if panel == "active":
            active_flip &= left["gold_candidate"] != right["gold_candidate"]
            e0, e1 = left["entities"][:2]
            attribute = left["attribute"]
            active_swap_exact &= (
                left["assignments"][e0][attribute] == right["assignments"][e1][attribute]
                and left["assignments"][e1][attribute] == right["assignments"][e0][attribute]
            )
        else:
            controls_stable &= left["gold_candidate"] == right["gold_candidate"]
        if panel == "matched_null":
            matched_anchor_exact &= left["gold_candidate"] == left["entities"][2]
            matched_anchor_exact &= right["gold_candidate"] == right["entities"][2]
        elif panel == "surface_only":
            surface_assignments_equal &= left["assignments"] == right["assignments"]
            surface_order_changes &= left["record_order"] != right["record_order"]
        elif panel == "semantic_neighbor":
            e0, e1 = left["entities"][:2]
            query_attribute = left["attribute"]
            neighbor = left["neighbor_attribute"]
            neighbor_swap_exact &= all(
                left["assignments"][entity][query_attribute]
                == right["assignments"][entity][query_attribute]
                for entity in left["entities"]
            )
            neighbor_swap_exact &= (
                left["assignments"][e0][neighbor] == right["assignments"][e1][neighbor]
                and left["assignments"][e1][neighbor] == right["assignments"][e0][neighbor]
            )
    add(checks, "state_token_multisets_match", state_multiset_match)
    add(checks, "active_answer_flips", active_flip)
    add(checks, "control_answers_stable", controls_stable)
    add(checks, "active_swap_exact", active_swap_exact)
    add(checks, "matched_null_anchor_exact", matched_anchor_exact)
    add(checks, "surface_assignments_equal", surface_assignments_equal)
    add(checks, "surface_order_changes", surface_order_changes)
    add(checks, "semantic_neighbor_swap_exact", neighbor_swap_exact)

    order_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        key = (
            row["combination_id"], row["panel"], row["template"], row["binding_state"]
        )
        order_groups[key].add(row["gold_position"])
    add(checks, "candidate_position_rotation", all(values == {0, 1, 2} for values in order_groups.values()))
    position_counts = Counter(row["gold_position"] for row in rows)
    add(checks, "global_position_balance", len(set(position_counts.values())) == 1, dict(position_counts))

    entity_sets = {
        world: {entity for row in rows if row["world"] == world for entity in row["entities"]}
        for world in EXPECTED_WORLDS
    }
    value_sets = {
        world: {
            value
            for row in rows
            if row["world"] == world
            for entity_values in row["assignments"].values()
            for value in entity_values.values()
        }
        for world in EXPECTED_WORLDS
    }
    world_pairs = [
        (left, right)
        for index, left in enumerate(sorted(EXPECTED_WORLDS))
        for right in sorted(EXPECTED_WORLDS)[index + 1 :]
    ]
    add(checks, "entity_worlds_disjoint", all(entity_sets[l].isdisjoint(entity_sets[r]) for l, r in world_pairs))
    add(checks, "value_worlds_disjoint", all(value_sets[l].isdisjoint(value_sets[r]) for l, r in world_pairs))
    add(
        checks,
        "no_indirect_candidate_aliases",
        all("Candidate A" not in row["prompt"] and "Candidate B" not in row["prompt"] for row in rows),
    )

    add(checks, "tokenizer_audit_pass", token_audit.get("overall_pass") is True)
    add(checks, "tokenizer_no_weights", token_audit.get("model_weights_loaded") is False)
    add(
        checks,
        "all_atoms_single_token",
        all(model["all_atoms_single_token"] for model in token_audit["models"].values()),
    )
    add(
        checks,
        "candidate_sequences_prefix_free",
        all(model["candidate_sequences_unique_and_prefix_free"] for model in token_audit["models"].values()),
    )
    add(
        checks,
        "prompt_length_bound",
        all(model["prompt_within_512_tokens"] for model in token_audit["models"].values()),
    )
    token_body = {key: value for key, value in token_audit.items() if key != "tokenizer_audit_digest"}
    add(checks, "tokenizer_audit_digest", digest(token_body) == token_audit["tokenizer_audit_digest"])

    add(checks, "summary_contract_link", summary["contract_digest"] == contract["contract_digest"])
    add(checks, "summary_package_digest", summary["package_digest"] == digest(rows))
    add(checks, "summary_tokenizer_link", summary["tokenizer_audit_digest"] == token_audit["tokenizer_audit_digest"])
    add(checks, "summary_no_behavior", summary["model_behavior_cases_scored"] == 0)
    add(checks, "summary_no_k", summary["new_k_item"] is None)
    add(checks, "natural_use_not_faked", "this package alone cannot satisfy U" in contract["evidence_gates"]["U_natural_use"])
    add(checks, "hidden_gate_closed", contract["execution_policy"]["hidden_scan_before_behavior_gate"] is False)
    add(checks, "probe_registry_frozen", set(contract["probe_registry"]) == {
        "P1_record_entity_entry", "P2_query_attribute_selector", "P3_record_binding_write",
        "P4_query_value_load", "P5_answer_competition", "P6_matched_rescue",
    })
    add(checks, "identifiability_limits_explicit", all(
        contract["identifiability_policy"][key] == "reserved_not_implemented"
        for key in ("local_equivalence_class_merging", "near_collision_calibration", "sample_level_ood")
    ))

    gate = all(check["pass"] for check in checks)
    output = {
        "phase": EXPECTED_PHASE,
        "kind": "independent_zero_model_mother_family_contract_audit",
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed": {
            "row_count": len(rows),
            "split_counts": dict(split_counts),
            "combination_counts": {split: len(values) for split, values in combinations_by_split.items()},
            "position_counts": dict(position_counts),
            "package_digest": digest(rows),
        },
    }
    output["audit_digest"] = digest(output)
    if write:
        write_json(AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = audit(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
