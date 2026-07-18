#!/usr/bin/env python3
"""Freeze role-normalized world geometry and query-platform bridge contracts."""

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
OUT_DIR = ROOT / "tests/gpt5/result/phase526_role_normalized_world_geometry_protocol"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")

SPLITS = {
    "world_fit": {"index": 0, "pair_count": 96, "sealed": False, "entity_pool": 0, "relation_pool": 0},
    "world_entity_prediction": {
        "index": 1,
        "pair_count": 192,
        "sealed": False,
        "entity_pool": 1,
        "relation_pool": 0,
    },
    "world_relation_prediction": {
        "index": 2,
        "pair_count": 192,
        "sealed": False,
        "entity_pool": 2,
        "relation_pool": 2,
    },
    "bridge_open_prediction": {
        "index": 3,
        "pair_count": 192,
        "sealed": False,
        "entity_pool": 3,
        "relation_pool": 3,
    },
    "sealed": {"index": 4, "pair_count": 192, "sealed": True, "entity_pool": 4, "relation_pool": 4},
}

ENTITY_POOLS = {
    0: ("Nolan", "Opal", "Perrin", "Quinn", "Rhea", "Silas", "Talia", "Ulric", "Vera", "Wes", "Xena", "Yorick"),
    1: ("Nadia", "Oren", "Priya", "Ronan", "Selene", "Tobin", "Una", "Vance", "Willa", "Xavier", "Yara", "Zane"),
    2: ("Noel", "Orla", "Pascal", "Rina", "Soren", "Tessa", "Uri", "Viola", "Wyatt", "Ximena", "Yael", "Zelda"),
    3: ("Nia", "Otis", "Petra", "Ravi", "Sable", "Theo", "Ula", "Viktor", "Wren", "Xander", "Yvette", "Zora"),
    4: ("Niko", "Oona", "Pia", "Rufus", "Sonia", "Tariq", "Ursula", "Val", "Wanda", "Xia", "Yosef", "Zuri"),
}

RELATION_POOLS = {
    0: (("consults", "is consulted by"), ("supports", "is supported by")),
    1: (("coordinates", "is coordinated by"), ("reviews", "is reviewed by")),
    2: (("monitors", "is monitored by"), ("organizes", "is organized by")),
    3: (("licenses", "is licensed by"), ("recommends", "is recommended by")),
    4: (("notifies", "is notified by"), ("represents", "is represented by")),
}

# Three pairs of reverse four-cycles. Each graph has identical in/out degree,
# and the two edge sets inside a pair are disjoint.
CYCLE_PAIRS = (
    ((1, 2, 3, 0), (3, 0, 1, 2)),
    ((1, 3, 0, 2), (2, 0, 3, 1)),
    ((2, 3, 1, 0), (3, 2, 0, 1)),
)

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
    names = [pool[(start + offset * 7) % len(pool)] for offset in range(8)]
    if len(set(names)) != 8:
        raise RuntimeError("entity collision")
    return names


def rotate(values: list[int], amount: int) -> list[int]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def graph_facts(names: list[str], relation: str, mapping: tuple[int, ...], order: list[int]) -> list[str]:
    return [f"{names[source]} {relation} {names[mapping[source]]}." for source in order]


def distractor_facts(names: list[str], count: int) -> list[str]:
    zones = ("amber zone", "cobalt zone", "ivory zone", "jade zone")
    return [f"{names[4 + index]} appears in the {zones[index]}." for index in range(count)]


def build_prompt(
    surface: str,
    relation: str,
    passive: str,
    facts: list[str],
    register_names: list[str],
    claim_source: str,
    claim_target: str,
) -> dict[str, Any]:
    rules = (
        f"The relation expressed by '{relation}' is directed and is not automatically reciprocal.",
        f"Only explicitly listed '{relation}' links hold in this world.",
        "The entity register order is arbitrary and does not express a relation.",
    )
    if surface == "identity":
        rule_header = "Rules:"
        fact_header = "Facts:"
        register_header = "Entity register:"
        opening = "Use only the following closed world."
        statement_header = "Statement:"
        instruction = "Return exactly one complete sentence from this list:"
        verdict = "Verdict:"
    elif surface == "natural_paraphrase":
        rule_header = "World constraints:"
        fact_header = "World evidence:"
        register_header = "Neutral entity ledger:"
        opening = "Judge the proposition only from this miniature world."
        statement_header = "Proposition:"
        instruction = "Reply with exactly one of the following complete sentences:"
        verdict = "Assessment:"
    else:
        raise ValueError(surface)

    rules_block = "\n".join(f"- {item}" for item in rules)
    facts_block = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(facts))
    register_lines = [f"{index + 1}. {name}" for index, name in enumerate(register_names)]
    register_block = "\n".join(register_lines)
    world_prefix = (
        f"{opening}\n{rule_header}\n{rules_block}\n{fact_header}\n{facts_block}\n"
        f"{register_header}\n{register_block}\nEvidence ledger complete."
    )
    claim = f"{claim_target} {passive} {claim_source}."
    prompt = (
        world_prefix
        + f"\n\n{statement_header}\n{claim}\n{instruction}\n"
        + f"{NATURAL_SENTENCES[True]}\n{NATURAL_SENTENCES[False]}\n{verdict}"
    )

    register_start = world_prefix.index(register_header) + len(register_header)
    cursor = register_start
    slot_ends = {}
    for slot, line in enumerate(register_lines):
        line_start = world_prefix.index(line, cursor)
        name = register_names[slot]
        name_start = line_start + line.index(name)
        slot_ends[f"register_slot_{slot}_end"] = name_start + len(name)
        cursor = line_start + len(line)

    claim_start = prompt.index(claim, len(world_prefix))
    target_start = claim_start
    relation_start = claim_start + claim.index(passive)
    source_start = claim_start + claim.rfind(claim_source)
    return {
        "world_prefix": world_prefix,
        "natural_prompt": prompt,
        "claim": claim,
        "role_char_ends": {
            **slot_ends,
            "evidence_end": len(world_prefix),
            "claim_target_end": target_start + len(claim_target),
            "claim_relation_end": relation_start + len(passive),
            "claim_source_end": source_start + len(claim_source),
            "claim_end": claim_start + len(claim),
            "prompt_end": len(prompt),
        },
    }


def edge_matrix(mapping: tuple[int, ...]) -> list[list[int]]:
    return [[int(source != target and mapping[source] == target) for target in range(4)] for source in range(4)]


def split_rows(split: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair_index in range(spec["pair_count"]):
        names = choose_names(spec["entity_pool"], pair_index)
        active, passive = RELATION_POOLS[spec["relation_pool"]][pair_index % 2]
        graph_a, graph_b = CYCLE_PAIRS[(pair_index // 2) % len(CYCLE_PAIRS)]
        query_source_index = (pair_index // 6) % 4
        query_target_index = graph_a[query_source_index]
        pair_id = f"phase526:{split}:W:{stable_hash(split, pair_index, active, names[:4])}"
        distractor_count = 2 if pair_index % 2 == 0 else 4
        extras = distractor_facts(names, distractor_count)
        for graph_label, mapping, truth_value in (("A", graph_a, True), ("B", graph_b, False)):
            for surface_index, surface in enumerate(SURFACES):
                base_order = rotate([0, 1, 2, 3], (pair_index + surface_index) % 4)
                fact_order = base_order if surface == "identity" else list(reversed(base_order))
                register_order = rotate([0, 1, 2, 3], (pair_index + 2 * surface_index) % 4)
                if surface == "natural_paraphrase":
                    register_order = list(reversed(register_order))
                facts = graph_facts(names, active, mapping, fact_order) + extras
                register_names = [names[index] for index in register_order]
                prompt = build_prompt(
                    surface,
                    active,
                    passive,
                    facts,
                    register_names,
                    names[query_source_index],
                    names[query_target_index],
                )
                rows.append({
                    "sample_id": f"{pair_id}:{graph_label}:{surface}",
                    "source_pair_id": pair_id,
                    "split": split,
                    "sealed": bool(spec["sealed"]),
                    "pair_index": pair_index,
                    "surface": surface,
                    "graph_label": graph_label,
                    "truth_value": truth_value,
                    "relation_active": active,
                    "relation_passive": passive,
                    "entity_names": names[:4],
                    "entity_pool": int(spec["entity_pool"]),
                    "relation_pool": int(spec["relation_pool"]),
                    "graph_mapping": list(mapping),
                    "edge_matrix": edge_matrix(mapping),
                    "fact_order": fact_order,
                    "register_order": register_order,
                    "query_source_index": query_source_index,
                    "query_target_index": query_target_index,
                    "facts": facts,
                    **prompt,
                })
    return rows


def audit_split(split: str, rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    expected = int(spec["pair_count"]) * 4
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_pair_id"]].append(row)
    group_pass = True
    token_bag_pass = True
    graph_pass = True
    prefix_pass = True
    slot_counts = Counter()
    for group in groups.values():
        group_pass &= len(group) == 4
        claims = {row["claim"] for row in group}
        group_pass &= len(claims) == 1
        for surface in SURFACES:
            local = [row for row in group if row["surface"] == surface]
            group_pass &= len(local) == 2
            if len(local) == 2:
                token_bag_pass &= token_counter(local[0]["facts"]) == token_counter(local[1]["facts"])
                edges = []
                for row in local:
                    edges.append({(source, target) for source, target in enumerate(row["graph_mapping"])})
                graph_pass &= not (edges[0] & edges[1])
        for row in group:
            mapping = row["graph_mapping"]
            graph_pass &= sorted(mapping) == [0, 1, 2, 3]
            graph_pass &= all(source != target for source, target in enumerate(mapping))
            graph_pass &= row["truth_value"] == (
                mapping[row["query_source_index"]] == row["query_target_index"]
            )
            prefix_pass &= row["natural_prompt"].startswith(row["world_prefix"])
            prefix_pass &= row["claim"] not in row["world_prefix"]
            for slot, entity_index in enumerate(row["register_order"]):
                slot_counts[(entity_index, slot)] += 1

    relations = sorted({row["relation_active"] for row in rows})
    return {
        "split": split,
        "sealed": bool(spec["sealed"]),
        "row_count": len(rows),
        "expected_row_count": expected,
        "source_pair_count": len(groups),
        "row_count_pass": len(rows) == expected,
        "four_way_group_pass": group_pass,
        "matched_fact_token_bag_pass": token_bag_pass,
        "directed_cycle_and_query_pass": graph_pass,
        "query_absent_from_world_prefix_pass": prefix_pass,
        "register_slot_balance_min": min(slot_counts.values()),
        "register_slot_balance_max": max(slot_counts.values()),
        "relations": relations,
        "entity_pool": int(spec["entity_pool"]),
        "relation_pool": int(spec["relation_pool"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[dict[str, Any]]] = {}
    split_files = {}
    audits = {}
    for split, spec in SPLITS.items():
        rows = split_rows(split, spec)
        generated[split] = rows
        path = OUT_DIR / f"phase526_{split}.jsonl"
        write_jsonl(path, rows)
        audits[split] = audit_split(split, rows, spec)
        split_files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sealed": bool(spec["sealed"]),
            "row_count": len(rows),
        }

    fit_relations = set(audits["world_fit"]["relations"])
    entity_relations = set(audits["world_entity_prediction"]["relations"])
    relation_relations = set(audits["world_relation_prediction"]["relations"])
    bridge_relations = set(audits["bridge_open_prediction"]["relations"])
    relation_split_pass = (
        fit_relations == entity_relations
        and not (fit_relations & relation_relations)
        and not (fit_relations & bridge_relations)
        and not (relation_relations & bridge_relations)
    )
    entity_split_pass = len({tuple(ENTITY_POOLS[spec["entity_pool"]]) for spec in SPLITS.values()}) == len(SPLITS)

    contract = {
        "schema_version": "phase526_role_normalized_world_geometry_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_execution",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "models_in_required_order": list(MODELS),
        "surfaces": list(SURFACES),
        "natural_event_sentences": {str(key).lower(): value for key, value in NATURAL_SENTENCES.items()},
        "split_files": split_files,
        "behavior_gate": {
            "overall_lcb95_min": 0.9,
            "surface_lcb95_min": 0.88,
            "four_way_lcb95_min": 0.82,
            "unrecoverable_ucb95_max": 0.05,
            "all_open_splits_required": True,
        },
        "physical_design": {
            "observer_scaffold": "query-free neutral entity register after complete world evidence",
            "projection_dimension": 48,
            "projection_seeds": [518031, 518037, 518041],
            "pair_feature": ["target_minus_source", "source_hadamard_target"],
            "pair_feature_is_mechanism": False,
            "relative_depth_bins": 8,
            "group_folds": 4,
            "minimum_fold_passes": 3,
            "projection_consensus_required": 2,
            "minimum_contiguous_depth_bins": 2,
            "discovery_gate": {
                "balanced_edge_lcb95_min": 0.82,
                "outgoing_top1_lcb95_min": 0.7,
                "directional_lcb95_min": 0.7,
                "exact_graph_lcb95_min": 0.45,
                "embedding_gain_min": 0.1,
                "position_baseline_gain_min": 0.15,
            },
            "prediction_gate": {
                "balanced_edge_lcb95_min": 0.85,
                "outgoing_top1_lcb95_min": 0.75,
                "directional_lcb95_min": 0.75,
                "exact_graph_lcb95_min": 0.5,
                "both_surfaces_required": True,
            },
            "direction_test": "true directed edge score must exceed its reverse edge score",
            "swap_operator_fit_forbidden": True,
            "world_matrix_test": "recover each source node's unique outgoing target; exact graph requires all four",
            "phase524_platform_targets": [
                {"role": "claim_target_end", "legacy_role": "claim_entity_end", "layers": [18, 19, 20, 21, 22]},
                {"role": "claim_end", "legacy_role": "claim_end", "layers": [19, 20, 21, 22, 23, 24]},
            ],
            "pipeline_permutation_count": 1024,
            "pipeline_permutation_seed": 526901,
        },
        "stage_order": [
            "behavior_qualification",
            "world_fit_and_freeze",
            "entity_holdout",
            "relation_word_holdout",
            "world_query_bridge_open_prediction",
            "full_pipeline_permutation",
        ],
        "stop_rules": [
            "No physical run for a model failing any open-split behavior gate.",
            "No bridge claim if world geometry fails entity or relation-word holdout.",
            "No transport claim from temporal order or predictive agreement alone.",
            "No attention-head, channel, neuron, patching, or sealed-split run in this stage.",
            "No result-contingent edits to features, bins, projections, thresholds, or permutations.",
        ],
        "evidence_boundaries": {
            "entity_register_is_natural_unprompted_state": False,
            "pair_observer_is_model_mechanism": False,
            "prediction_is_compute_edge": False,
            "temporal_order_is_transport": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "sealed_read": False,
        },
    }
    contract_path = OUT_DIR / "phase526_frozen_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    static_pass = (
        all(
            item["row_count_pass"]
            and item["four_way_group_pass"]
            and item["matched_fact_token_bag_pass"]
            and item["directed_cycle_and_query_pass"]
            and item["query_absent_from_world_prefix_pass"]
            and item["register_slot_balance_min"] == item["register_slot_balance_max"]
            for item in audits.values()
        )
        and relation_split_pass
        and entity_split_pass
    )
    audit = {
        "schema_version": "phase526_role_normalized_world_geometry_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run" if static_pass else "static_fail",
        "splits": audits,
        "relation_holdout_design_pass": relation_split_pass,
        "entity_pool_disjoint_pass": entity_split_pass,
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": sha256_file(contract_path),
        "sealed_split_read_by_downstream": False,
        "model_run": False,
    }
    audit_path = OUT_DIR / "phase526_static_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(contract_path)
    print(audit_path)
    if not static_pass:
        raise SystemExit("Phase526 static audit failed")


if __name__ == "__main__":
    main()
