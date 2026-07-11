#!/usr/bin/env python3
"""Publish heldout-stable Phase332 interface paths without moving prior atlas nodes."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas"
OUTPUT = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
PUBLIC = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"
POSITIVES = {
    ("language_action", "summarize"),
    ("reasoning_constraint", "missing_condition_control"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_json_artifacts(root: Path) -> int:
    changed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "checksums.json":
            continue
        if path.suffix == ".json":
            payload = read_json(path)
            safe = json_safe(payload)
            if safe != payload:
                write_json(path, safe)
                changed += 1
        elif path.suffix == ".jsonl":
            rows = read_jsonl(path)
            safe = json_safe(rows)
            if safe != rows:
                write_jsonl(path, safe)
                changed += 1
    return changed


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["component_type"], int(row["component_layer"]), row["position_role"],
        int(row["component_index"]), int(row["component_start"]), int(row["component_end"]),
    )


def stable_member_keys(validation: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    shared: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    stable: set[tuple[Any, ...]] = set()
    for row in validation:
        member = tuple(json.loads(row["member_identity"]))
        base = (row["model"], row["family_id"], row["mechanism_id"])
        if (row["family_id"], row["mechanism_id"]) not in POSITIVES:
            continue
        if row["set_type"] == "shared_skeleton":
            shared[(*base, member)].append(row)
        elif row.get("heldout_branch_specific"):
            stable.add((*base, "interface_branch", row["owner_interface"], member))
    for (*base, member), rows in shared.items():
        expected = 3 if base[0] == "glm4" else 4
        if len(rows) == expected and min(row["heldout_item_sign_consistency"] for row in rows) >= 0.75:
            stable.add((*base, "shared_skeleton", "shared_all_unique_interfaces", member))
    return stable


def make_node(
    member: dict[str, Any], family_name: str, model_revision: str,
    validation_rows: list[dict[str, Any]], generated_at: str,
) -> dict[str, Any]:
    component_type = member["component_type"]
    unit_kind = "attention_head" if component_type == "attention_head_input" else "mlp_product_group"
    component = "attention" if component_type == "attention_head_input" else "mlp"
    owner = member["interface"]
    member_id = (
        f"p332:{member['model']}:{member['family_id']}:{member['mechanism_id']}:"
        f"{member['set_type']}:{owner}:{member['position_role']}:L{member['component_layer']}:"
        f"{component_type}:{member['component_index']}"
    )
    valid = [
        row for row in validation_rows
        if row["model"] == member["model"]
        and row["family_id"] == member["family_id"]
        and row["mechanism_id"] == member["mechanism_id"]
        and row["set_type"] == member["set_type"]
        and tuple(json.loads(row["member_identity"])) == identity(member)
        and (member["set_type"] == "shared_skeleton" or row["owner_interface"] == owner)
    ]
    heldout_consistency = min((row["heldout_item_sign_consistency"] for row in valid), default=0.0)
    return {
        "schema_version": "neuron_atlas_node.v1",
        "node_id": member_id,
        "node_type": "interface_path_member",
        "family_id": member["family_id"],
        "family_name": family_name,
        "mechanism_id": member["mechanism_id"],
        "relation": member["mechanism_id"],
        "model": member["model"],
        "model_revision": model_revision,
        "layer": int(member["component_layer"]),
        "component": component,
        "unit_kind": unit_kind,
        "unit_index": int(member["component_index"]),
        "component_start": int(member["component_start"]),
        "component_end": int(member["component_end"]),
        "position_role": member["position_role"],
        "natural_activation": float(member["mean_contribution"]),
        "candidate_score": float(member["selection_score"]),
        "display_priority": 1.4 + min(1.0, float(member["selection_score"])),
        "case_count": int(member["discovery_case_count"]),
        "natural_observed": True,
        "phase332_tested": True,
        "phase332_path_role": member["set_type"],
        "phase332_interface": owner,
        "phase332_position_role": member["position_role"],
        "phase332_discovery_item_sign_consistency": float(member["item_sign_consistency"]),
        "phase332_heldout_item_sign_consistency": float(heldout_consistency),
        "phase332_heldout_stable": True,
        "phase332_exchange_causally_effective": False,
        "evidence_level": "L3_interface_path_map_not_causally_closed",
        "evidence_status": "heldout_stable_interface_path_member_not_causally_closed",
        "causal_scope": "component_member_not_single_neuron",
        "single_unit_causal": False,
        "evidence_boundary": (
            "Discovered on four Phase332 objects and retained only after four disjoint heldout objects. "
            "The component is a shared-skeleton or interface-branch member; path exchange and single-neuron "
            "causality did not close."
        ),
        "source_artifacts": [
            "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_member_sets.jsonl",
            "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_member_validation.jsonl",
        ],
        "published_at": generated_at,
    }


def publish(output: Path = OUTPUT, public: Path = PUBLIC) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = read_json(SOURCE / "phase332_global_summary.json")
    members = read_jsonl(SOURCE / "phase332_member_sets.jsonl")
    validation = read_jsonl(SOURCE / "phase332_member_validation.jsonl")
    paths = read_jsonl(SOURCE / "phase332_path_summary.jsonl")
    cross = read_jsonl(SOURCE / "phase332_cross_model_summary.jsonl")
    claims = read_jsonl(SOURCE / "phase332_claim_registry.jsonl")
    stable = stable_member_keys(validation)
    selected_members = []
    for row in members:
        key = (
            row["model"], row["family_id"], row["mechanism_id"],
            row["set_type"], row["interface"], identity(row),
        )
        if key in stable:
            selected_members.append(row)

    index = read_json(output / "neuron_index.json")
    refs = index["partitions"]
    partition_lookup = {(ref["model"], ref["family_id"]): ref for ref in refs}
    new_nodes = []
    for member in selected_members:
        ref = partition_lookup[(member["model"], member["family_id"])]
        partition = read_json(output / ref["path"])
        new_nodes.append(make_node(
            member, partition["family"]["family_name"],
            partition["model_snapshot"]["model_revision"], validation, generated_at,
        ))

    old_nodes = [row for row in read_jsonl(output / "neuron_nodes.jsonl") if not row.get("phase332_tested")]
    write_jsonl(output / "neuron_nodes.jsonl", [*old_nodes, *new_nodes])
    write_jsonl(output / "phase332_interface_path_nodes.jsonl", new_nodes)
    write_jsonl(output / "phase332_path_summary.jsonl", paths)
    write_jsonl(output / "phase332_cross_model_summary.jsonl", cross)
    write_jsonl(output / "phase332_claim_registry.jsonl", claims)
    shutil.copy2(SOURCE / "phase332_global_summary.json", output / "phase332_global_summary.json")
    shutil.copy2(SOURCE / "phase332_report.md", output / "phase332_report.md")

    nodes_by_partition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in new_nodes:
        nodes_by_partition[(node["model"], node["family_id"])].append(node)
    path_by_partition = {(row["model"], row["family_id"]): row for row in paths if (row["family_id"], row["mechanism_id"]) in POSITIVES}
    cross_by_family = {row["family_id"]: row for row in cross}
    updated_partitions = 0
    for ref in refs:
        key = (ref["model"], ref["family_id"])
        if key not in nodes_by_partition:
            continue
        partition_path = output / ref["path"]
        partition = read_json(partition_path)
        partition["nodes"] = [row for row in partition.get("nodes", []) if not row.get("phase332_tested")]
        partition["nodes"].extend(nodes_by_partition[key])
        partition["scope"]["phase"] = 332
        partition["scope"]["source_phases"] = sorted(set([*partition["scope"].get("source_phases", []), 332]))
        local_path = path_by_partition[key]
        global_path = cross_by_family[ref["family_id"]]
        shared_count = local_path["heldout_stable_shared_member_count"]
        branch_count = (
            local_path["heldout_specific_raw_branch_member_count"]
            + local_path["heldout_specific_aligned_branch_member_count"]
        )
        partition["metrics"].update({
            "phase332_interface_path_member_count": len(nodes_by_partition[key]),
            "phase332_stable_shared_member_count": shared_count,
            "phase332_specific_interface_branch_member_count": branch_count,
            "phase332_path_exchange_effective_count": int(global_path["gate"]["path_exchange_effective"]),
            "phase332_full_gate_pass_count": int(global_path["full_gate_pass"]),
            "phase332_single_unit_causal_count": 0,
        })
        partition["path"]["phase332_interface_path"] = {
            "model_path_summary": local_path,
            "cross_model_summary": global_path,
            "display": "heldout_stable_members_only",
        }
        partition["mapping_status"] = "phase332_interface_path_mapped_not_causally_closed"
        partition["evidence_boundary"] = (
            "Phase332 displays only discovery members that remained stable or interface-specific on disjoint "
            "heldout objects. Cross-model path exchange passed 0/2 positive mechanisms; displayed components "
            "are not causal single neurons."
        )
        partition["generated_at"] = generated_at
        write_json(partition_path, partition)
        ref.update({
            "mapping_status": partition["mapping_status"],
            "phase332_interface_path_member_count": len(nodes_by_partition[key]),
            "phase332_stable_shared_member_count": shared_count,
            "phase332_specific_interface_branch_member_count": branch_count,
            "phase332_full_gate_pass_count": int(global_path["full_gate_pass"]),
        })
        updated_partitions += 1

    index["generated_at"] = generated_at
    write_json(output / "neuron_index.json", index)
    manifest = read_json(output / "manifest.json")
    manifest["phase"] = 332
    manifest["generated_at"] = generated_at
    manifest["partitions"] = refs
    manifest["metrics"].update({
        "phase332_registered_interface_case_count": summary["denominator"]["registered_interface_case_count"],
        "phase332_natural_path_row_count": summary["denominator"]["natural_path_row_count"],
        "phase332_natural_unit_row_count": summary["denominator"]["natural_unit_row_count"],
        "phase332_exchange_condition_row_count": summary["denominator"]["exchange_condition_row_count"],
        "phase332_interface_path_member_count": len(new_nodes),
        "phase332_stable_shared_mechanism_count": summary["results"]["cross_model_stable_shared_skeleton_count"],
        "phase332_specific_interface_branch_mechanism_count": summary["results"]["cross_model_specific_interface_branch_count"],
        "phase332_path_exchange_effective_count": summary["results"]["cross_model_path_exchange_effective_count"],
        "phase332_full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        "phase332_incomplete_exchange_cell_count": summary["results"]["incomplete_exchange_cell_count"],
        "single_unit_causal_count": 0,
    })
    manifest["evidence_boundary"] = {
        **manifest.get("evidence_boundary", {}),
        "statement": (
            "Phase332 maps heldout-stable shared skeletons and interface branches for two positive mechanisms "
            "and matched controls. Path exchange passed 0/2; no displayed member is a causal single neuron."
        ),
        "phase332_full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        "language_encoding_mechanism_closed": False,
        "single_unit_intervention_gate_open": False,
    }
    manifest["source_artifacts"] = list(dict.fromkeys([
        *manifest.get("source_artifacts", []),
        "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_report.md",
        "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_global_summary.json",
        "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas/phase332_claim_registry.jsonl",
    ]))
    write_json(output / "manifest.json", manifest)
    sanitized_artifact_count = sanitize_json_artifacts(output)
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            checksums.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path)})
    write_json(output / "checksums.json", {"schema_version": "artifact_checksums.v1", "files": checksums})
    shutil.copytree(output, public, dirs_exist_ok=True)
    return {
        "phase": 332,
        "prior_node_count": len(old_nodes),
        "interface_path_member_count": len(new_nodes),
        "updated_partition_count": updated_partitions,
        "full_gate_pass_count": summary["results"]["full_gate_pass_count"],
        "single_unit_causal_count": 0,
        "sanitized_nonfinite_artifact_count": sanitized_artifact_count,
        "valid": len(old_nodes) == 1985 and len(new_nodes) == 286 and updated_partitions == 6,
    }


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False, indent=2))
