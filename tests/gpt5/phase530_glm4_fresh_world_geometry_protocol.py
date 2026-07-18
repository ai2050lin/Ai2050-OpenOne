#!/usr/bin/env python3
"""Freeze a fresh GLM4 world-geometry split after Phase529 confirmation."""

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
OUT_DIR = ROOT / "tests/gpt5/result/phase530_glm4_fresh_world_geometry_protocol"
AUTH_PATH = ROOT / "tests/gpt5/result/phase529_relation_contract_factorial_behavior/phase529_factorial_authorization.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")
SELECTED_CONDITION = "two_edge__reverse__active__register_present"

SPLITS = {
    "discovery": {"index": 0, "pair_count": 96, "sealed": False, "entity_pool": 0, "relation_pool": 0},
    "entity_prediction": {"index": 1, "pair_count": 192, "sealed": False, "entity_pool": 1, "relation_pool": 0},
    "relation_prediction": {"index": 2, "pair_count": 192, "sealed": False, "entity_pool": 2, "relation_pool": 1},
    "sealed": {"index": 3, "pair_count": 192, "sealed": True, "entity_pool": 3, "relation_pool": 2},
}

ENTITY_POOLS = {
    0: ("Amina", "Blaise", "Cora", "Dorian", "Esme", "Felix", "Greta", "Hugo", "Iris", "Jasper", "Keira", "Lucan"),
    1: ("Mara", "Nevin", "Olive", "Pavel", "Rosa", "Stefan", "Thea", "Uriel", "Vivienne", "Wade", "Xiomara", "Yusuf"),
    2: ("Ada", "Bruno", "Clara", "Damon", "Elena", "Farid", "Gina", "Harold", "Ines", "Jonas", "Kara", "Leon"),
    3: ("Mina", "Nestor", "Odette", "Paolo", "Ruth", "Sami", "Tara", "Umar", "Violet", "Will", "Xenia", "Yanni"),
}

RELATION_POOLS = {
    0: (("guides", "is guided by"), ("advises", "is advised by")),
    1: (("mentors", "is mentored by"), ("trains", "is trained by")),
    2: (("briefs", "is briefed by"), ("coaches", "is coached by")),
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


def choose_names(pool_index: int, pair_index: int) -> list[str]:
    pool = ENTITY_POOLS[pool_index]
    start = (pair_index * 5) % len(pool)
    names = [pool[(start + offset * 7) % len(pool)] for offset in range(4)]
    if len(set(names)) != 4:
        raise RuntimeError("entity collision")
    return names


def rotate(values: list[int], amount: int) -> list[int]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def build_prompt(
    surface: str,
    active: str,
    names: list[str],
    edges: list[tuple[int, int]],
    fact_order: list[int],
    register_order: list[int],
) -> dict[str, Any]:
    facts = [f"{names[edges[index][0]]} {active} {names[edges[index][1]]}." for index in fact_order]
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
    rules = (
        f"The relation expressed by '{active}' is directed and is not automatically reciprocal.",
        f"Only explicitly listed '{active}' links hold in this world.",
        "The entity register order is arbitrary and does not express a relation.",
    )
    rules_block = "\n".join(f"- {item}" for item in rules)
    facts_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(facts))
    register_lines = [f"{index + 1}. {names[entity]}" for index, entity in enumerate(register_order)]
    register_block = "\n".join(register_lines)
    world_prefix = (
        f"{opening}\n{rule_header}\n{rules_block}\n{fact_header}\n{facts_block}\n"
        f"{register_header}\n{register_block}\nEvidence ledger complete."
    )
    claim = f"{names[0]} {active} {names[1]}."
    prompt = (
        world_prefix
        + f"\n\n{statement_header}\n{claim}\n{instruction}\n"
        + f"{NATURAL_SENTENCES[True]}\n{NATURAL_SENTENCES[False]}\n{verdict}"
    )
    register_start = world_prefix.index(register_header) + len(register_header)
    cursor = register_start
    role_ends = {}
    for slot, line in enumerate(register_lines):
        line_start = world_prefix.index(line, cursor)
        name = names[register_order[slot]]
        role_ends[f"register_slot_{slot}_end"] = line_start + line.index(name) + len(name)
        cursor = line_start + len(line)
    claim_start = prompt.index(claim, len(world_prefix))
    role_ends.update({
        "evidence_end": len(world_prefix),
        "claim_source_end": claim_start + len(names[0]),
        "claim_relation_end": claim_start + claim.index(active) + len(active),
        "claim_target_end": claim_start + claim.rfind(names[1]) + len(names[1]),
        "claim_end": claim_start + len(claim),
        "prompt_end": len(prompt),
    })
    return {
        "facts": facts,
        "world_prefix": world_prefix,
        "claim": claim,
        "natural_prompt": prompt,
        "role_char_ends": role_ends,
    }


def split_rows(split: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair_index in range(spec["pair_count"]):
        names = choose_names(spec["entity_pool"], pair_index)
        active, passive = RELATION_POOLS[spec["relation_pool"]][pair_index % 2]
        pair_id = f"phase530:{split}:{stable_hash(split, pair_index, names, active)}"
        for truth_value, edges in (
            (True, [(0, 1), (2, 3)]),
            (False, [(1, 0), (3, 2)]),
        ):
            for surface_index, surface in enumerate(SURFACES):
                fact_order = rotate([0, 1], (pair_index + surface_index) % 2)
                if surface == "natural_paraphrase":
                    fact_order = list(reversed(fact_order))
                register_order = rotate([0, 1, 2, 3], (pair_index + surface_index) % 4)
                if surface == "natural_paraphrase":
                    register_order = list(reversed(register_order))
                prompt = build_prompt(surface, active, names, edges, fact_order, register_order)
                rows.append({
                    "sample_id": f"{pair_id}:{'true' if truth_value else 'false'}:{surface}",
                    "source_pair_id": pair_id,
                    "split": split,
                    "sealed": bool(spec["sealed"]),
                    "pair_index": pair_index,
                    "surface": surface,
                    "truth_value": truth_value,
                    "selected_condition": SELECTED_CONDITION,
                    "relation_active": active,
                    "relation_passive": passive,
                    "entity_names": names,
                    "edges": [list(edge) for edge in edges],
                    "fact_order": fact_order,
                    "register_order": register_order,
                    **prompt,
                })
    return rows


def audit_split(split: str, rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_pair_id"]].append(row)
    group_pass = True
    token_pass = True
    direction_pass = True
    slot_counts = Counter()
    for group in groups.values():
        group_pass &= len(group) == 4 and len({row["claim"] for row in group}) == 1
        for surface in SURFACES:
            local = [row for row in group if row["surface"] == surface]
            group_pass &= len(local) == 2
            if len(local) == 2:
                token_pass &= token_counter(local[0]["facts"]) == token_counter(local[1]["facts"])
        for row in group:
            edges = {tuple(edge) for edge in row["edges"]}
            direction_pass &= ((0, 1) in edges) == bool(row["truth_value"])
            direction_pass &= ((1, 0) in edges) != bool(row["truth_value"])
            direction_pass &= row["natural_prompt"].startswith(row["world_prefix"])
            for slot, entity in enumerate(row["register_order"]):
                slot_counts[(entity, slot)] += 1
    expected = int(spec["pair_count"]) * 4
    return {
        "split": split,
        "sealed": bool(spec["sealed"]),
        "row_count": len(rows),
        "expected_row_count": expected,
        "source_pair_count": len(groups),
        "row_count_pass": len(rows) == expected,
        "four_way_group_pass": group_pass,
        "matched_fact_token_bag_pass": token_pass,
        "direction_logic_pass": direction_pass,
        "register_slot_balance_min": min(slot_counts.values()),
        "register_slot_balance_max": max(slot_counts.values()),
        "relations": sorted({row["relation_active"] for row in rows}),
    }


def main() -> None:
    authorization = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    confirmed = authorization["confirmed_conditions_by_model"].get("glm4", [])
    if SELECTED_CONDITION not in confirmed:
        raise RuntimeError("selected GLM4 condition was not independently confirmed")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audits = {}
    split_files = {}
    for split, spec in SPLITS.items():
        rows = split_rows(split, spec)
        path = OUT_DIR / f"phase530_{split}.jsonl"
        write_jsonl(path, rows)
        audits[split] = audit_split(split, rows, spec)
        split_files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sealed": bool(spec["sealed"]),
            "row_count": len(rows),
        }

    contract = {
        "schema_version": "phase530_glm4_fresh_world_geometry_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_fresh_behavior_or_physical_execution",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "phase529_authorization_path": str(AUTH_PATH.relative_to(ROOT)),
        "phase529_authorization_sha256": sha256_file(AUTH_PATH),
        "models_in_required_order": list(MODELS),
        "physical_authorized_model": "glm4",
        "selected_condition": SELECTED_CONDITION,
        "selection_reason": (
            "Predeclared priority among confirmed conditions: entity register present, reverse foil, "
            "active voice, then lowest lexical scaffolding."
        ),
        "split_files": split_files,
        "fresh_behavior_gate": {
            "overall_lcb95_min": 0.9,
            "surface_lcb95_min": 0.88,
            "four_way_lcb95_min": 0.82,
            "unrecoverable_ucb95_max": 0.05,
            "all_open_splits_required": True,
        },
        "physical_design": {
            "projection_dimension": 48,
            "projection_seeds": [518031, 518037, 518041],
            "position_roles": [
                "register_slot_0_end",
                "register_slot_1_end",
                "register_slot_2_end",
                "register_slot_3_end",
                "claim_source_end",
                "claim_target_end",
                "claim_end",
            ],
            "pair_feature": ["target_minus_source", "source_hadamard_target"],
            "pair_feature_is_mechanism": False,
            "group_folds": 4,
            "minimum_fold_passes": 3,
            "projection_consensus_required": 2,
            "minimum_contiguous_layers": 4,
            "discovery_gate": {
                "orientation_lcb95_min": 0.82,
                "direction_pair_lcb95_min": 0.8,
                "disconnected_specificity_lcb95_min": 0.75,
                "exact_world_lcb95_min": 0.4,
                "surface_lcb95_min": 0.78,
                "embedding_gain_min": 0.1,
                "position_gain_min": 0.15,
            },
            "prediction_gate": {
                "orientation_lcb95_min": 0.85,
                "direction_pair_lcb95_min": 0.82,
                "disconnected_specificity_lcb95_min": 0.78,
                "exact_world_lcb95_min": 0.45,
                "surface_lcb95_min": 0.82,
            },
            "pipeline_permutation_count": 1024,
            "pipeline_permutation_seed": 530901,
        },
        "stop_rules": [
            "Qwen3 and DS7B receive not-authorized records and no weight load.",
            "GLM4 physical arrays are not collected unless every fresh open behavior split passes.",
            "Prediction splits are not read before the discovery observer ledger is durable.",
            "No query-platform bridge claim because GLM4 lacks a strict Phase524 query platform.",
            "No causal, compute-edge, attention-head, channel, neuron, patching, or sealed claim.",
        ],
        "evidence_boundaries": {
            "neutral_register_is_unprompted_natural_state": False,
            "pair_observer_is_model_mechanism": False,
            "world_geometry_is_query_platform_bridge": False,
            "causal_intervention": False,
            "sealed_read": False,
        },
    }
    contract_path = OUT_DIR / "phase530_frozen_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    relation_sets = [set(item["relations"]) for item in audits.values()]
    relation_pass = all(not (left & right) for index, left in enumerate(relation_sets) for right in relation_sets[index + 1 :])
    # Discovery and entity prediction intentionally share relation words.
    relation_pass = (
        set(audits["discovery"]["relations"]) == set(audits["entity_prediction"]["relations"])
        and not (set(audits["discovery"]["relations"]) & set(audits["relation_prediction"]["relations"]))
        and not (set(audits["discovery"]["relations"]) & set(audits["sealed"]["relations"]))
        and not (set(audits["relation_prediction"]["relations"]) & set(audits["sealed"]["relations"]))
    )
    static_pass = all(
        item["row_count_pass"]
        and item["four_way_group_pass"]
        and item["matched_fact_token_bag_pass"]
        and item["direction_logic_pass"]
        and item["register_slot_balance_min"] == item["register_slot_balance_max"]
        for item in audits.values()
    ) and relation_pass
    audit = {
        "schema_version": "phase530_glm4_fresh_world_geometry_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if static_pass else "static_fail",
        "splits": audits,
        "relation_holdout_design_pass": relation_pass,
        "selected_condition_confirmed_pass": True,
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256_file(contract_path),
        "sealed_split_read_by_downstream": False,
        "model_run": False,
    }
    audit_path = OUT_DIR / "phase530_static_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(contract_path)
    print(audit_path)
    if not static_pass:
        raise SystemExit("Phase530 static audit failed")


if __name__ == "__main__":
    main()
