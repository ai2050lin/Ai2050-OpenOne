#!/usr/bin/env python3
"""Freeze pair-addressed world-binding splits before any model execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
OUT_DIR = ROOT / "tests/gpt5/result/phase535_pair_addressed_binding_protocol"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")
CANDIDATE_PAIRS = ((0, 1), (0, 3), (2, 1), (2, 3))

SPLITS = {
    "discovery": {"index": 0, "group_count": 48, "sealed": False, "entity_pool": 0, "relation_pool": 0},
    "entity_prediction": {"index": 1, "group_count": 96, "sealed": False, "entity_pool": 1, "relation_pool": 0},
    "relation_prediction": {"index": 2, "group_count": 96, "sealed": False, "entity_pool": 2, "relation_pool": 1},
    "sealed": {"index": 3, "group_count": 96, "sealed": True, "entity_pool": 3, "relation_pool": 2},
}

ENTITY_POOLS = {
    0: ("Amina", "Blaise", "Cora", "Dorian", "Esme", "Felix", "Greta", "Hugo", "Iris", "Jasper", "Keira", "Lucan"),
    1: ("Mara", "Nevin", "Olive", "Pavel", "Rosa", "Stefan", "Thea", "Uriel", "Vivienne", "Wade", "Xiomara", "Yusuf"),
    2: ("Ada", "Bruno", "Clara", "Damon", "Elena", "Farid", "Gina", "Harold", "Ines", "Jonas", "Kara", "Leon"),
    3: ("Mina", "Nestor", "Odette", "Paolo", "Ruth", "Sami", "Tara", "Umar", "Violet", "Will", "Xenia", "Yanni"),
}

RELATION_POOLS = {
    0: ("guides", "advises"),
    1: ("mentors", "trains"),
    2: ("briefs", "coaches"),
}

NATURAL_SENTENCES = {
    True: "The statement is supported.",
    False: "The statement is contradicted.",
}


def stable_hash(*parts: object, n: int = 20) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def token_counter(lines: list[str]) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z0-9_]+", " ".join(lines).lower()))


def rotate(values: list[int], amount: int) -> list[int]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def choose_names(pool_index: int, group_index: int) -> list[str]:
    pool = ENTITY_POOLS[pool_index]
    start = (group_index * 5) % len(pool)
    names = [pool[(start + offset * 7) % len(pool)] for offset in range(4)]
    if len(set(names)) != 4:
        raise RuntimeError("entity collision")
    return names


def world_edges(world_id: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if world_id == 0:
        return ((0, 1), (2, 3))
    if world_id == 1:
        return ((0, 3), (2, 1))
    raise ValueError(world_id)


def build_prefix(
    surface: str,
    relation: str,
    names: list[str],
    edges: tuple[tuple[int, int], tuple[int, int]],
    fact_order: list[int],
    pair_order: list[int],
) -> dict[str, Any]:
    facts = [f"{names[edges[index][0]]} {relation} {names[edges[index][1]]}." for index in fact_order]
    if surface == "identity":
        opening = "Use only the following closed world."
        rule_header = "Rules:"
        fact_header = "Facts:"
        register_header = "Entity register:"
        pair_header = "Neutral candidate-pair index:"
        pair_note = "The index names pairs only; it does not assert that any indexed pair is linked."
        pair_template = lambda slot, left, right: f"{slot + 1}. {left} ; {right}"
        end_marker = "Pair index complete."
    else:
        opening = "Judge propositions only from this miniature world."
        rule_header = "World constraints:"
        fact_header = "World evidence:"
        register_header = "Neutral entity ledger:"
        pair_header = "Pair-address ledger:"
        pair_note = "A ledger entry is an address, not evidence that the two names are related."
        pair_template = lambda slot, left, right: f"Slot {slot + 1}: {left} alongside {right}"
        end_marker = "Address ledger complete."
    rules = (
        f"The relation expressed by '{relation}' is directed and is not automatically reciprocal.",
        f"Only explicitly listed '{relation}' links hold in this world.",
        "The entity and pair ledgers are neutral indexes and add no facts.",
    )
    rules_block = "\n".join(f"- {item}" for item in rules)
    facts_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(facts))
    register_lines = [f"{index + 1}. {name}" for index, name in enumerate(names)]
    register_block = "\n".join(register_lines)
    pair_lines = []
    for slot, candidate_index in enumerate(pair_order):
        left, right = CANDIDATE_PAIRS[candidate_index]
        pair_lines.append(pair_template(slot, names[left], names[right]))
    pair_block = "\n".join(pair_lines)
    world_prefix = (
        f"{opening}\n{rule_header}\n{rules_block}\n{fact_header}\n{facts_block}\n"
        f"{register_header}\n{register_block}\n{pair_header}\n{pair_note}\n{pair_block}\n{end_marker}"
    )

    fact_start = world_prefix.index(fact_header) + len(fact_header)
    fact_char_ends = []
    cursor = fact_start
    for line in facts:
        line_start = world_prefix.index(line, cursor)
        fact_char_ends.append(line_start + len(line))
        cursor = line_start + len(line)
    pair_start = world_prefix.index(pair_header) + len(pair_header)
    pair_char_ends = {}
    cursor = pair_start
    for slot, line in enumerate(pair_lines):
        line_start = world_prefix.index(line, cursor)
        candidate_index = pair_order[slot]
        _, right = CANDIDATE_PAIRS[candidate_index]
        right_name = names[right]
        right_start = line.rfind(right_name)
        pair_char_ends[f"pair_slot_{slot}_end"] = line_start + right_start + len(right_name)
        cursor = line_start + len(line)
    role_char_ends = {
        "facts_end": max(fact_char_ends),
        "world_end": len(world_prefix),
        **pair_char_ends,
    }
    return {
        "facts": facts,
        "pair_ledger_lines": pair_lines,
        "world_prefix": world_prefix,
        "role_char_ends": role_char_ends,
    }


def add_query(
    prefix: dict[str, Any],
    surface: str,
    relation: str,
    names: list[str],
    candidate_index: int,
) -> dict[str, Any]:
    left, right = CANDIDATE_PAIRS[candidate_index]
    claim = f"{names[left]} {relation} {names[right]}."
    if surface == "identity":
        statement_header = "Statement:"
        instruction = "Return exactly one complete sentence from this list:"
        verdict = "Verdict:"
    else:
        statement_header = "Proposition:"
        instruction = "Reply with exactly one of the following complete sentences:"
        verdict = "Assessment:"
    prompt = (
        prefix["world_prefix"]
        + f"\n\n{statement_header}\n{claim}\n{instruction}\n"
        + f"{NATURAL_SENTENCES[True]}\n{NATURAL_SENTENCES[False]}\n{verdict}"
    )
    return {"claim": claim, "natural_prompt": prompt}


def split_rows(split: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group_index in range(spec["group_count"]):
        names = choose_names(spec["entity_pool"], group_index)
        relation = RELATION_POOLS[spec["relation_pool"]][group_index % 2]
        source_group_id = f"phase535:{split}:{stable_hash(split, group_index, names, relation)}"
        for surface_index, surface in enumerate(SURFACES):
            pair_order = rotate(list(range(4)), (group_index + surface_index) % 4)
            if surface == "natural_paraphrase":
                pair_order = list(reversed(pair_order))
            fact_order = rotate([0, 1], (group_index + surface_index) % 2)
            for world_id in (0, 1):
                edges = world_edges(world_id)
                prefix = build_prefix(surface, relation, names, edges, fact_order, pair_order)
                edge_set = set(edges)
                candidate_slot = {candidate_index: pair_order.index(candidate_index) for candidate_index in range(4)}
                for candidate_index, candidate in enumerate(CANDIDATE_PAIRS):
                    truth_value = candidate in edge_set
                    query = add_query(prefix, surface, relation, names, candidate_index)
                    rows.append({
                        "sample_id": f"{source_group_id}:W{world_id}:{surface}:P{candidate_index}",
                        "source_group_id": source_group_id,
                        "world_surface_id": f"{source_group_id}:W{world_id}:{surface}",
                        "pair_flip_id": f"{source_group_id}:{surface}:P{candidate_index}",
                        "split": split,
                        "sealed": bool(spec["sealed"]),
                        "group_index": group_index,
                        "world_id": world_id,
                        "surface": surface,
                        "relation_active": relation,
                        "entity_names": names,
                        "edges": [list(edge) for edge in edges],
                        "fact_order": fact_order,
                        "pair_order": pair_order,
                        "candidate_pairs": [list(pair) for pair in CANDIDATE_PAIRS],
                        "candidate_index": candidate_index,
                        "candidate_pair": list(candidate),
                        "candidate_slot": candidate_slot[candidate_index],
                        "truth_value": truth_value,
                        **prefix,
                        **query,
                    })
    return rows


def audit_split(split: str, rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    world_surfaces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_flips: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_group_id"]].append(row)
        world_surfaces[row["world_surface_id"]].append(row)
        pair_flips[row["pair_flip_id"]].append(row)

    group_pass = all(
        len(group) == 16
        and {row["world_id"] for row in group} == {0, 1}
        and {row["surface"] for row in group} == set(SURFACES)
        and sum(bool(row["truth_value"]) for row in group) == 8
        for group in groups.values()
    )
    world_surface_pass = all(
        len(group) == 4
        and {row["candidate_index"] for row in group} == set(range(4))
        and sum(bool(row["truth_value"]) for row in group) == 2
        and len({row["world_prefix"] for row in group}) == 1
        for group in world_surfaces.values()
    )
    pair_flip_pass = all(
        len(group) == 2
        and {row["world_id"] for row in group} == {0, 1}
        and {bool(row["truth_value"]) for row in group} == {False, True}
        and len({row["candidate_slot"] for row in group}) == 1
        and len({tuple(row["pair_order"]) for row in group}) == 1
        for group in pair_flips.values()
    )
    token_bag_pass = True
    ledger_identity_pass = True
    for group in groups.values():
        for surface in SURFACES:
            local = [row for row in group if row["surface"] == surface and row["candidate_index"] == 0]
            if len(local) != 2:
                token_bag_pass = False
                ledger_identity_pass = False
                continue
            local.sort(key=lambda row: row["world_id"])
            token_bag_pass &= token_counter(local[0]["facts"]) == token_counter(local[1]["facts"])
            ledger_identity_pass &= local[0]["pair_ledger_lines"] == local[1]["pair_ledger_lines"]
    query_isolation_pass = all(
        row["natural_prompt"].startswith(row["world_prefix"])
        and row["natural_prompt"][len(row["world_prefix"]) :].lstrip().startswith(
            "Statement:" if row["surface"] == "identity" else "Proposition:"
        )
        for row in rows
    )
    slot_labels = Counter((row["candidate_slot"], bool(row["truth_value"])) for row in rows)
    slot_balance_pass = all(
        slot_labels[(slot, True)] == slot_labels[(slot, False)]
        for slot in range(4)
    )
    expected = int(spec["group_count"]) * 16
    return {
        "split": split,
        "sealed": bool(spec["sealed"]),
        "row_count": len(rows),
        "expected_row_count": expected,
        "source_group_count": len(groups),
        "world_surface_count": len(world_surfaces),
        "pair_flip_count": len(pair_flips),
        "row_count_pass": len(rows) == expected,
        "sixteen_way_group_pass": group_pass,
        "world_surface_four_way_pass": world_surface_pass,
        "pair_status_flip_pass": pair_flip_pass,
        "matched_fact_token_bag_pass": token_bag_pass,
        "pair_ledger_world_identity_pass": ledger_identity_pass,
        "query_section_separated_from_world_prefix_pass": query_isolation_pass,
        "slot_label_balance_pass": slot_balance_pass,
        "slot_label_counts": {f"slot_{slot}_{label}": slot_labels[(slot, label)] for slot in range(4) for label in (False, True)},
        "relations": sorted({row["relation_active"] for row in rows}),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audits = {}
    split_files = {}
    entity_sets = {}
    for split, spec in SPLITS.items():
        rows = split_rows(split, spec)
        path = OUT_DIR / f"phase535_{split}.jsonl"
        write_jsonl(path, rows)
        audits[split] = audit_split(split, rows, spec)
        entity_sets[split] = {name for row in rows for name in row["entity_names"]}
        split_files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sealed": bool(spec["sealed"]),
            "row_count": len(rows),
        }

    entity_disjoint = all(
        not (entity_sets[left] & entity_sets[right])
        for index, left in enumerate(SPLITS)
        for right in list(SPLITS)[index + 1 :]
    )
    relation_holdout = (
        set(audits["discovery"]["relations"]) == set(audits["entity_prediction"]["relations"])
        and not (set(audits["discovery"]["relations"]) & set(audits["relation_prediction"]["relations"]))
        and not (set(audits["discovery"]["relations"]) & set(audits["sealed"]["relations"]))
        and not (set(audits["relation_prediction"]["relations"]) & set(audits["sealed"]["relations"]))
    )
    static_pass = entity_disjoint and relation_holdout and all(
        report[key]
        for report in audits.values()
        for key in (
            "row_count_pass",
            "sixteen_way_group_pass",
            "world_surface_four_way_pass",
            "pair_status_flip_pass",
            "matched_fact_token_bag_pass",
            "pair_ledger_world_identity_pass",
            "query_section_separated_from_world_prefix_pass",
            "slot_label_balance_pass",
        )
    )

    contract = {
        "schema_version": "phase535_pair_addressed_binding_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_execution",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "models_in_required_order": list(MODELS),
        "split_files": split_files,
        "world_design": {
            "world_0_edges": [[0, 1], [2, 3]],
            "world_1_edges": [[0, 3], [2, 1]],
            "candidate_pairs": [list(pair) for pair in CANDIDATE_PAIRS],
            "all_nodes_keep_source_or_target_role": True,
            "every_candidate_pair_flips_edge_status": True,
            "pair_address_slot_fixed_across_world_flip": True,
            "pair_ledger_asserts_relation": False,
        },
        "behavior_gate": {
            "overall_lcb95_min": 0.9,
            "surface_lcb95_min": 0.88,
            "world_exact_lcb95_min": 0.78,
            "pair_flip_exact_lcb95_min": 0.82,
            "unrecoverable_ucb95_max": 0.05,
            "all_open_splits_required": True,
        },
        "physical_design": {
            "projection_dimension": 48,
            "projection_seeds": [518031, 518037, 518041],
            "position_roles": ["pair_slot_0_end", "pair_slot_1_end", "pair_slot_2_end", "pair_slot_3_end", "facts_end", "world_end"],
            "group_folds": 4,
            "projection_consensus_required": 2,
            "minimum_contiguous_layers": 4,
            "discovery_gate": {
                "overall_lcb95_min": 0.82,
                "surface_lcb95_min": 0.76,
                "pair_flip_exact_lcb95_min": 0.72,
                "candidate_lcb95_min": 0.72,
                "embedding_gain_min": 0.12,
                "position_gain_min": 0.2,
            },
            "prediction_gate": {
                "overall_lcb95_min": 0.84,
                "surface_lcb95_min": 0.8,
                "pair_flip_exact_lcb95_min": 0.76,
                "candidate_lcb95_min": 0.76,
            },
            "pipeline_permutation_count": 1024,
            "pipeline_permutation_seed": 535901,
        },
        "stop_rules": [
            "Models run in Qwen3, GLM4, DS7B order and release CUDA memory between runs.",
            "Physical collection is forbidden unless all three open behavior splits pass for that model.",
            "Prediction splits remain unread until a durable discovery platform ledger exists.",
            "Pipeline permutations run only after an independently predictive platform exists.",
            "The pair ledger is an observer scaffold, not a spontaneous natural world state.",
            "No component, head, channel, neuron, causal, compute-edge, or sealed claim.",
        ],
        "evidence_boundaries": {
            "pair_address_is_relation_label": False,
            "scaffold_observer_is_natural_state": False,
            "observer_is_model_mechanism": False,
            "causal": False,
            "sealed_read": False,
        },
    }
    contract_path = OUT_DIR / "phase535_frozen_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema_version": "phase535_pair_addressed_binding_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if static_pass else "static_fail",
        "splits": audits,
        "entity_pool_disjoint_pass": entity_disjoint,
        "relation_holdout_design_pass": relation_holdout,
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256_file(contract_path),
        "model_run": False,
        "sealed_split_read_by_downstream": False,
    }
    audit_path = OUT_DIR / "phase535_static_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(contract_path)
    print(audit_path)
    if not static_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
