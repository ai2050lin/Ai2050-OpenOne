#!/usr/bin/env python3
"""Freeze a factorial audit of the Phase526 behavior-contract failure."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
OUT_DIR = ROOT / "tests/gpt5/result/phase528_relation_contract_factorial_protocol"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")
GRAPH_SHAPES = ("two_edge", "four_cycle")
FOIL_TYPES = ("reverse", "disconnected")
QUERY_VOICES = ("active", "passive")
REGISTER_MODES = ("absent", "present")

SPLITS = {
    "calibration": {"index": 0, "instance_count": 16, "sealed": False, "entity_pool": 0, "relation_pool": 0},
    "confirmation": {"index": 1, "instance_count": 32, "sealed": False, "entity_pool": 1, "relation_pool": 2},
    "sealed": {"index": 2, "instance_count": 32, "sealed": True, "entity_pool": 2, "relation_pool": 4},
}

ENTITY_POOLS = {
    0: ("Nolan", "Opal", "Perrin", "Quinn", "Rhea", "Silas", "Talia", "Ulric", "Vera", "Wes", "Xena", "Yorick"),
    1: ("Nadia", "Oren", "Priya", "Ronan", "Selene", "Tobin", "Una", "Vance", "Willa", "Xavier", "Yara", "Zane"),
    2: ("Noel", "Orla", "Pascal", "Rina", "Soren", "Tessa", "Uri", "Viola", "Wyatt", "Ximena", "Yael", "Zelda"),
}

RELATION_POOLS = {
    0: (("consults", "is consulted by"), ("supports", "is supported by")),
    2: (("monitors", "is monitored by"), ("organizes", "is organized by")),
    4: (("notifies", "is notified by"), ("represents", "is represented by")),
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


def choose_names(pool_index: int, instance_index: int) -> list[str]:
    pool = ENTITY_POOLS[pool_index]
    start = (instance_index * 5) % len(pool)
    names = [pool[(start + offset * 7) % len(pool)] for offset in range(4)]
    if len(set(names)) != 4:
        raise RuntimeError("entity collision")
    return names


def rotate(values: list[int], amount: int) -> list[int]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def graph_edges(graph_shape: str, foil_type: str, truth_value: bool) -> list[tuple[int, int]]:
    if graph_shape == "two_edge":
        if truth_value:
            return [(0, 1), (2, 3)]
        if foil_type == "reverse":
            return [(1, 0), (3, 2)]
        return [(0, 2), (1, 3)]
    if graph_shape != "four_cycle":
        raise ValueError(graph_shape)
    if truth_value:
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if foil_type == "reverse":
        return [(0, 3), (3, 2), (2, 1), (1, 0)]
    return [(0, 2), (2, 1), (1, 3), (3, 0)]


def build_prompt(
    surface: str,
    active: str,
    passive: str,
    names: list[str],
    edges: list[tuple[int, int]],
    fact_order: list[int],
    query_voice: str,
    register_mode: str,
    register_order: list[int],
) -> dict[str, Any]:
    ordered_edges = [edges[index] for index in fact_order]
    facts = [f"{names[source]} {active} {names[target]}." for source, target in ordered_edges]
    if surface == "identity":
        opening = "Use only the following closed world."
        rule_header = "Rules:"
        fact_header = "Facts:"
        register_header = "Entity register:"
        statement_header = "Statement:"
        instruction = "Return exactly one complete sentence from this list:"
        verdict = "Verdict:"
    else:
        opening = "Judge the proposition only from this miniature world."
        rule_header = "World constraints:"
        fact_header = "World evidence:"
        register_header = "Neutral entity ledger:"
        statement_header = "Proposition:"
        instruction = "Reply with exactly one of the following complete sentences:"
        verdict = "Assessment:"
    rules = [
        f"The relation expressed by '{active}' is directed and is not automatically reciprocal.",
        f"Only explicitly listed '{active}' links hold in this world.",
    ]
    if register_mode == "present":
        rules.append("The entity register order is arbitrary and does not express a relation.")
    rules_block = "\n".join(f"- {item}" for item in rules)
    facts_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(facts))
    prefix = f"{opening}\n{rule_header}\n{rules_block}\n{fact_header}\n{facts_block}"
    if register_mode == "present":
        register_block = "\n".join(f"{index + 1}. {names[item]}" for index, item in enumerate(register_order))
        prefix += f"\n{register_header}\n{register_block}\nEvidence ledger complete."
    if query_voice == "active":
        claim = f"{names[0]} {active} {names[1]}."
    elif query_voice == "passive":
        claim = f"{names[1]} {passive} {names[0]}."
    else:
        raise ValueError(query_voice)
    prompt = (
        prefix
        + f"\n\n{statement_header}\n{claim}\n{instruction}\n"
        + f"{NATURAL_SENTENCES[True]}\n{NATURAL_SENTENCES[False]}\n{verdict}"
    )
    return {"facts": facts, "world_prefix": prefix, "claim": claim, "natural_prompt": prompt}


def split_rows(split: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    conditions = list(itertools.product(GRAPH_SHAPES, FOIL_TYPES, QUERY_VOICES, REGISTER_MODES))
    for instance_index in range(spec["instance_count"]):
        names = choose_names(spec["entity_pool"], instance_index)
        active, passive = RELATION_POOLS[spec["relation_pool"]][instance_index % 2]
        for graph_shape, foil_type, query_voice, register_mode in conditions:
            condition_id = f"{graph_shape}__{foil_type}__{query_voice}__register_{register_mode}"
            group_id = f"phase528:{split}:{condition_id}:{stable_hash(split, instance_index, names, active)}"
            for truth_value in (True, False):
                edges = graph_edges(graph_shape, foil_type, truth_value)
                for surface_index, surface in enumerate(SURFACES):
                    order = rotate(list(range(len(edges))), (instance_index + surface_index) % len(edges))
                    if surface == "natural_paraphrase":
                        order = list(reversed(order))
                    register_order = rotate([0, 1, 2, 3], (instance_index + surface_index) % 4)
                    if surface == "natural_paraphrase":
                        register_order = list(reversed(register_order))
                    prompt = build_prompt(
                        surface,
                        active,
                        passive,
                        names,
                        edges,
                        order,
                        query_voice,
                        register_mode,
                        register_order,
                    )
                    rows.append({
                        "sample_id": f"{group_id}:{'true' if truth_value else 'false'}:{surface}",
                        "source_group_id": group_id,
                        "split": split,
                        "sealed": bool(spec["sealed"]),
                        "instance_index": instance_index,
                        "condition_id": condition_id,
                        "graph_shape": graph_shape,
                        "foil_type": foil_type,
                        "query_voice": query_voice,
                        "register_mode": register_mode,
                        "surface": surface,
                        "truth_value": truth_value,
                        "relation_active": active,
                        "relation_passive": passive,
                        "entity_names": names,
                        "edges": [list(edge) for edge in edges],
                        "fact_order": order,
                        "register_order": register_order,
                        **prompt,
                    })
    return rows


def audit_split(split: str, rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    condition_count = len(GRAPH_SHAPES) * len(FOIL_TYPES) * len(QUERY_VOICES) * len(REGISTER_MODES)
    expected = int(spec["instance_count"]) * condition_count * 4
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_group_id"]].append(row)
    group_pass = True
    token_pass = True
    logic_pass = True
    for group in groups.values():
        group_pass &= len(group) == 4
        group_pass &= len({row["claim"] for row in group}) == 1
        for surface in SURFACES:
            local = [row for row in group if row["surface"] == surface]
            group_pass &= len(local) == 2
            if len(local) == 2:
                token_pass &= token_counter(local[0]["facts"]) == token_counter(local[1]["facts"])
        for row in group:
            edge_set = {tuple(edge) for edge in row["edges"]}
            logic_pass &= ((0, 1) in edge_set) == bool(row["truth_value"])
            if not row["truth_value"]:
                logic_pass &= ((1, 0) in edge_set) == (row["foil_type"] == "reverse")
            logic_pass &= row["natural_prompt"].startswith(row["world_prefix"])
            logic_pass &= "Verdict:" not in row["world_prefix"]
            logic_pass &= "Assessment:" not in row["world_prefix"]
    condition_counts = Counter(row["condition_id"] for row in rows)
    return {
        "split": split,
        "sealed": bool(spec["sealed"]),
        "row_count": len(rows),
        "expected_row_count": expected,
        "source_group_count": len(groups),
        "condition_count": len(condition_counts),
        "condition_row_count_min": min(condition_counts.values()),
        "condition_row_count_max": max(condition_counts.values()),
        "row_count_pass": len(rows) == expected,
        "four_way_group_pass": group_pass,
        "matched_fact_token_bag_pass": token_pass,
        "query_logic_pass": logic_pass,
        "factor_balance_pass": len(set(condition_counts.values())) == 1,
        "relations": sorted({row["relation_active"] for row in rows}),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_files = {}
    audits = {}
    for split, spec in SPLITS.items():
        rows = split_rows(split, spec)
        path = OUT_DIR / f"phase528_{split}.jsonl"
        write_jsonl(path, rows)
        split_files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sealed": bool(spec["sealed"]),
            "row_count": len(rows),
        }
        audits[split] = audit_split(split, rows, spec)

    contract = {
        "schema_version": "phase528_relation_contract_factorial_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_execution",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "models_in_required_order": list(MODELS),
        "factors": {
            "graph_shape": list(GRAPH_SHAPES),
            "foil_type": list(FOIL_TYPES),
            "query_voice": list(QUERY_VOICES),
            "register_mode": list(REGISTER_MODES),
            "surface": list(SURFACES),
            "truth_value": [True, False],
        },
        "split_files": split_files,
        "condition_gate": {
            "overall_lcb95_min": 0.78,
            "truth_lcb95_min": 0.68,
            "surface_lcb95_min": 0.7,
            "four_way_lcb95_min": 0.6,
            "unrecoverable_ucb95_max": 0.08,
            "calibration_and_confirmation_required": True,
        },
        "selection_rule": (
            "A condition is physically eligible for a later fresh split only when it passes unchanged "
            "in calibration and independent entity/relation confirmation."
        ),
        "evidence_boundaries": {
            "factorial_behavior_is_world_geometry": False,
            "best_condition_is_pre_registered_mechanism": False,
            "causal_intervention": False,
            "sealed_read": False,
        },
        "stop_rules": [
            "No thresholds or factor definitions change after any model output.",
            "No physical analysis uses calibration or confirmation rows from this diagnostic.",
            "No best-cell selection is called a mechanism discovery.",
            "No sealed split is read in Phase528-529.",
        ],
    }
    contract_path = OUT_DIR / "phase528_frozen_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    calibration_relations = set(audits["calibration"]["relations"])
    confirmation_relations = set(audits["confirmation"]["relations"])
    sealed_relations = set(audits["sealed"]["relations"])
    static_pass = (
        all(
            item["row_count_pass"]
            and item["four_way_group_pass"]
            and item["matched_fact_token_bag_pass"]
            and item["query_logic_pass"]
            and item["factor_balance_pass"]
            for item in audits.values()
        )
        and not (calibration_relations & confirmation_relations)
        and not (calibration_relations & sealed_relations)
        and not (confirmation_relations & sealed_relations)
    )
    audit = {
        "schema_version": "phase528_relation_contract_factorial_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if static_pass else "static_fail",
        "splits": audits,
        "relation_vocabulary_disjoint_pass": not (
            (calibration_relations & confirmation_relations)
            or (calibration_relations & sealed_relations)
            or (confirmation_relations & sealed_relations)
        ),
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256_file(contract_path),
        "sealed_split_read_by_downstream": False,
        "model_run": False,
    }
    audit_path = OUT_DIR / "phase528_static_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(contract_path)
    print(audit_path)
    if not static_pass:
        raise SystemExit("Phase528 static audit failed")


if __name__ == "__main__":
    main()
