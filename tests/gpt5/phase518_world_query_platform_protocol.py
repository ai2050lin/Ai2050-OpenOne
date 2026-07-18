#!/usr/bin/env python3
"""Freeze Phase518 world/query, natural-event, and label-ledger contracts."""

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
OUT_DIR = ROOT / "tests/gpt5/result/phase518_world_query_platform_protocol"
SURFACES = ("identity", "natural_paraphrase")
LABEL_SYSTEMS = ("mapped_ab", "mapped_01")
MAPPING_ORDERS = ("mapping_first", "state_first")
FACT_ORDERS = ("target_first", "distractor_first", "interleaved")
LENGTHS = ("compact", "extended")
MODELS = ("qwen3", "glm4", "deepseek7b")

RELATION_SPLITS = {
    "calibration": {"index": 0, "pair_count": 96, "sealed": False},
    "confirmation": {"index": 1, "pair_count": 192, "sealed": False},
    "platform_discovery": {"index": 2, "pair_count": 96, "sealed": False},
    "platform_prediction": {"index": 3, "pair_count": 192, "sealed": False},
    "sealed": {"index": 4, "pair_count": 192, "sealed": True},
}
BINDING_SPLITS = {
    "calibration": {"index": 0, "repeats": 16, "sealed": False},
    "confirmation": {"index": 1, "repeats": 32, "sealed": False},
    "sealed": {"index": 2, "repeats": 32, "sealed": True},
}

NAME_POOLS = {
    0: ("Nolan", "Opal", "Perrin", "Quinn", "Rhea", "Silas", "Talia", "Ulric", "Vera", "Wes", "Xena", "Yorick"),
    1: ("Nadia", "Oren", "Priya", "Ronan", "Selene", "Tobin", "Una", "Vance", "Willa", "Xavier", "Yara", "Zane"),
    2: ("Noel", "Orla", "Pascal", "Rina", "Soren", "Tessa", "Uri", "Viola", "Wyatt", "Ximena", "Yael", "Zelda"),
    3: ("Nia", "Otis", "Petra", "Ravi", "Sable", "Theo", "Ula", "Viktor", "Wren", "Xander", "Yvette", "Zora"),
    4: ("Niko", "Oona", "Pia", "Rufus", "Sonia", "Tariq", "Ursula", "Val", "Wanda", "Xia", "Yosef", "Zuri"),
}
RELATIONS = {
    0: (("consults", "is consulted by"), ("supports", "is supported by")),
    1: (("coordinates", "is coordinated by"), ("reviews", "is reviewed by")),
    2: (("monitors", "is monitored by"), ("organizes", "is organized by")),
    3: (("licenses", "is licensed by"), ("recommends", "is recommended by")),
    4: (("notifies", "is notified by"), ("represents", "is represented by")),
}

STATE_CUES = {
    0: {
        True: ("The proposition survives the evidence check.", "The record endorses the proposition."),
        False: ("The proposition fails the evidence check.", "The record opposes the proposition."),
    },
    1: {
        True: ("The assessment upholds the proposition.", "The available evidence favors the proposition."),
        False: ("The assessment overturns the proposition.", "The available evidence rejects the proposition."),
    },
    2: {
        True: ("The proposition receives an affirmative assessment.", "The review accepts the proposition."),
        False: ("The proposition receives a negative assessment.", "The review declines the proposition."),
    },
}

NATURAL_SENTENCES = {
    True: "The statement is supported.",
    False: "The statement is contradicted.",
}


def stable_hash(*parts: object, n: int = 20) -> str:
    value = "::".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:n]


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


def parse_natural_event(text: str) -> dict[str, Any]:
    """Parse a generated relation sentence without access to the target truth."""

    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    supported = NATURAL_SENTENCES[True].lower()
    contradicted = NATURAL_SENTENCES[False].lower()
    if lowered == supported:
        return {"semantic_value": True, "recoverable": True, "event_type": "exact_supported"}
    if lowered == contradicted:
        return {"semantic_value": False, "recoverable": True, "event_type": "exact_contradicted"}
    if "supported" in lowered and "contradicted" in lowered:
        return {"semantic_value": None, "recoverable": False, "event_type": "ambiguous_both"}
    return {"semantic_value": None, "recoverable": False, "event_type": "unrecoverable"}


def parse_mapping_event(text: str) -> dict[str, Any]:
    """Parse the canonical mapping-comprehension sentence target-blindly."""

    normalized = " ".join(text.strip().split())
    match = re.fullmatch(r"The holding symbol is ([AB01])\.", normalized, flags=re.IGNORECASE)
    if not match:
        return {"symbol": None, "recoverable": False, "event_type": "unrecoverable"}
    return {
        "symbol": match.group(1).upper(),
        "recoverable": True,
        "event_type": "exact_mapping_sentence",
    }


def parse_free_label(text: str) -> dict[str, Any]:
    normalized = text.strip()
    if re.fullmatch(r"[AB01]", normalized):
        return {"symbol": normalized.upper(), "recoverable": True, "event_type": "strict_label"}
    return {"symbol": None, "recoverable": False, "event_type": "unrecoverable"}


def entity(split_index: int, pair_index: int, slot: int) -> str:
    pool = NAME_POOLS[split_index]
    return pool[(pair_index * 5 + slot) % len(pool)]


def order_facts(target: list[str], distractors: list[str], fact_order: str) -> list[str]:
    if fact_order == "target_first":
        return target + distractors
    if fact_order == "distractor_first":
        return distractors + target
    if fact_order != "interleaved":
        raise ValueError(fact_order)
    rows = []
    for left, right in itertools.zip_longest(target, distractors):
        if left is not None:
            rows.append(left)
        if right is not None:
            rows.append(right)
    return rows


def relation_world(split_index: int, pair_index: int) -> dict[str, Any]:
    names = [entity(split_index, pair_index, slot) for slot in range(8)]
    if len(set(names)) < 8:
        raise RuntimeError("entity collision")
    active, passive = RELATIONS[split_index][pair_index % 2]
    topology_facts = {
        "A": [f"{names[0]} {active} {names[1]}.", f"{names[2]} {active} {names[3]}."],
        "B": [f"{names[0]} {active} {names[2]}.", f"{names[1]} {active} {names[3]}."],
    }
    query_pattern = "A" if (pair_index // 2) % 2 == 0 else "B"
    query_target = names[1] if query_pattern == "A" else names[2]
    claim = f"{query_target} {passive} {names[0]}."
    rules = [
        f"The relation expressed by '{active}' is directed and is not automatically reciprocal.",
        f"Only explicitly listed '{active}' links hold in this world.",
    ]
    return {
        "active_relation": active,
        "passive_relation": passive,
        "names": names,
        "topology_facts": topology_facts,
        "query_pattern": query_pattern,
        "query_target": query_target,
        "claim": claim,
        "rules": rules,
    }


def distractors(world: dict[str, Any], length: str) -> list[str]:
    count = 2 if length == "compact" else 6
    regions = ("amber zone", "cobalt zone", "ivory zone", "jade zone")
    return [
        f"{world['names'][4 + index % 4]} appears in the {regions[index % 4]}."
        for index in range(count)
    ]


def relation_prompt(
    surface: str,
    rules: list[str],
    facts: list[str],
    target_facts: list[str],
    extra_facts: list[str],
    claim: str,
    query_target: str,
    passive_relation: str,
) -> dict[str, Any]:
    rules_block = "\n".join(f"- {item}" for item in rules)
    facts_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(facts))
    if surface == "identity":
        common = (
            "Use only the following closed world.\n"
            f"Rules:\n{rules_block}\nFacts:\n{facts_block}\n"
            f"Statement:\n{claim}\n"
        )
        candidate_prompt = common + "Question: Is the statement true or false?\nAnswer:"
        natural_prompt = (
            common
            + "Return exactly one complete sentence from this list:\n"
            + f"{NATURAL_SENTENCES[True]}\n{NATURAL_SENTENCES[False]}\nVerdict:"
        )
    elif surface == "natural_paraphrase":
        common = (
            "Judge the proposition only from this miniature world.\n"
            f"World rules:\n{rules_block}\nWorld evidence:\n{facts_block}\n"
            f"Proposition:\n{claim}\n"
        )
        candidate_prompt = common + "Decide whether the proposition holds. Answer true or false.\nAnswer:"
        natural_prompt = (
            common
            + "Reply with exactly one of the following complete sentences:\n"
            + f"{NATURAL_SENTENCES[True]}\n{NATURAL_SENTENCES[False]}\nVerdict:"
        )
    else:
        raise ValueError(surface)
    claim_start = natural_prompt.rfind(claim)
    entity_offset = claim.find(query_target)
    relation_offset = claim.find(passive_relation)
    if min(claim_start, entity_offset, relation_offset) < 0:
        raise RuntimeError("missing relation prompt anchor")
    return {
        "surface": surface,
        "candidate_prompt": candidate_prompt,
        "natural_prompt": natural_prompt,
        "role_char_ends": {
            "target_evidence_end": max(natural_prompt.rfind(item) + len(item) for item in target_facts),
            "distractor_evidence_end": max(natural_prompt.rfind(item) + len(item) for item in extra_facts),
            "claim_entity_end": claim_start + entity_offset + len(query_target),
            "claim_relation_end": claim_start + relation_offset + len(passive_relation),
            "claim_end": claim_start + len(claim),
            "prompt_end": len(natural_prompt),
        },
    }


def relation_samples(split: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    split_index = spec["index"]
    rows = []
    for pair_index in range(spec["pair_count"]):
        world = relation_world(split_index, pair_index)
        length = LENGTHS[pair_index % len(LENGTHS)]
        fact_order = FACT_ORDERS[(pair_index // len(LENGTHS)) % len(FACT_ORDERS)]
        extras = distractors(world, length)
        true_topology = world["query_pattern"]
        false_topology = "B" if true_topology == "A" else "A"
        source_pair_id = f"phase518:{split}:R:{stable_hash(split, pair_index, world['claim'])}"
        for truth_value, topology in ((True, true_topology), (False, false_topology)):
            target = world["topology_facts"][topology]
            facts = order_facts(target, extras, fact_order)
            variants = [
                relation_prompt(
                    surface,
                    world["rules"],
                    facts,
                    target,
                    extras,
                    world["claim"],
                    world["query_target"],
                    world["passive_relation"],
                )
                for surface in SURFACES
            ]
            rows.append({
                "sample_id": f"{source_pair_id}:{'true' if truth_value else 'false'}",
                "source_pair_id": source_pair_id,
                "split": split,
                "sealed": spec["sealed"],
                "pair_index": pair_index,
                "truth_value": truth_value,
                "world_topology": topology,
                "query_pattern": world["query_pattern"],
                "length_control": length,
                "fact_order": fact_order,
                "relation_verb": world["active_relation"],
                "claim": world["claim"],
                "rules": world["rules"],
                "facts": facts,
                "target_facts": target,
                "distractors": extras,
                "variants": variants,
            })
    return rows


def mapping_spec(label_system: str, flip: bool, template: int) -> dict[str, str]:
    if label_system == "mapped_ab":
        holding, failing = ("B", "A") if flip else ("A", "B")
    elif label_system == "mapped_01":
        holding, failing = ("1", "0") if flip else ("0", "1")
    else:
        raise ValueError(label_system)
    templates = (
        "Use {h} when the proposition holds and {f} when it does not hold.",
        "The code for a holding proposition is {h}; the code for a failing proposition is {f}.",
        "Assign {h} to a proposition that holds and {f} to one that fails.",
        "A holding proposition receives {h}, whereas a failing proposition receives {f}.",
        "If the assessment accepts the proposition, write {h}; otherwise write {f}.",
        "Symbol {h} denotes that the proposition holds, and symbol {f} denotes that it fails.",
        "Choose {h} for the holding state and {f} for the failing state.",
        "Under this convention, holding maps to {h} and failing maps to {f}.",
        "Encode an upheld proposition as {h} and a rejected proposition as {f}.",
        "The affirmative state uses {h}; the negative state uses {f}.",
        "Record {h} after an accepting verdict and {f} after a rejecting verdict.",
        "For this task, {h} means the proposition holds while {f} means it does not.",
        "Translate a successful evidence check to {h} and an unsuccessful check to {f}.",
        "Return {h} when the record supports the proposition and {f} when it opposes it.",
        "The holding-state marker is {h}; the failing-state marker is {f}.",
        "Use marker {h} for acceptance and marker {f} for rejection.",
    )
    line = templates[template].format(h=holding, f=failing)
    return {"mapping_line": line, "holding": holding, "failing": failing}


def binding_prompts(
    cue: str,
    mapping: dict[str, str],
    order: str,
    surface: str,
    label_system: str,
) -> dict[str, str]:
    if surface == "identity":
        state_line = f"Semantic assessment: {cue}"
        mapping_line = f"Coding rule: {mapping['mapping_line']}"
    else:
        state_line = f"Evaluation note: {cue}"
        mapping_line = f"Output convention: {mapping['mapping_line']}"
    blocks = [mapping_line, state_line] if order == "mapping_first" else [state_line, mapping_line]
    common = "\n".join(blocks)
    candidate_prompt = common + "\nReturn the code for this assessment.\nCode:"
    free_prompt = common + "\nReturn only the single code symbol and nothing else.\nCode:"
    allowed = (
        ("The holding symbol is A.", "The holding symbol is B.")
        if label_system == "mapped_ab"
        else ("The holding symbol is 0.", "The holding symbol is 1.")
    )
    mapping_prompt = (
        f"Coding rule: {mapping['mapping_line']}\n"
        "Identify the symbol that denotes a holding proposition. Reply with exactly one sentence:\n"
        f"{allowed[0]}\n{allowed[1]}\nAnswer:"
    )
    return {
        "candidate_prompt": candidate_prompt,
        "free_prompt": free_prompt,
        "mapping_prompt": mapping_prompt,
    }


def binding_samples(split: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    split_index = spec["index"]
    cue_bank = STATE_CUES[split_index]
    for repeat in range(spec["repeats"]):
        for truth_value, label_system, flip, order in itertools.product(
            (False, True), LABEL_SYSTEMS, (False, True), MAPPING_ORDERS
        ):
            template = repeat % 16
            mapping = mapping_spec(label_system, flip, template)
            expected = mapping["holding"] if truth_value else mapping["failing"]
            reversal_id = f"phase518:{split}:B:{stable_hash(split, repeat, truth_value, label_system, order, template)}"
            variants = []
            for surface_index, surface in enumerate(SURFACES):
                cue = cue_bank[truth_value][surface_index]
                prompts = binding_prompts(cue, mapping, order, surface, label_system)
                variants.append({"surface": surface, "state_cue": cue, **prompts})
            rows.append({
                "sample_id": f"{reversal_id}:{'flip' if flip else 'base'}",
                "mapping_reversal_id": reversal_id,
                "mapping_probe_id": f"phase518:{split}:map:{stable_hash(split, label_system, flip, template)}",
                "split": split,
                "sealed": spec["sealed"],
                "repeat": repeat,
                "truth_value": truth_value,
                "label_system": label_system,
                "mapping_flip": flip,
                "mapping_order": order,
                "mapping_template": template,
                "mapping_line": mapping["mapping_line"],
                "holding_symbol": mapping["holding"],
                "failing_symbol": mapping["failing"],
                "expected_symbol": expected,
                "variants": variants,
            })
    return rows


def relation_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_pair_id"]].append(row)
    fixed_claim = True
    token_match = True
    complete = True
    query_counts = Counter()
    topology_truth = Counter()
    for items in groups.values():
        complete &= len(items) == 2 and {item["truth_value"] for item in items} == {False, True}
        fixed_claim &= len({item["claim"] for item in items}) == 1
        token_match &= len({tuple(sorted(token_counter(item["facts"]).items())) for item in items}) == 1
        query_counts[items[0]["query_pattern"]] += 1
        for item in items:
            topology_truth[(item["world_topology"], item["truth_value"])] += 1
    return {
        "semantic_sample_count": len(rows),
        "variant_row_count": len(rows) * len(SURFACES),
        "source_pair_count": len(groups),
        "complete_truth_pairs": bool(complete),
        "fixed_claim_pair_pass": bool(fixed_claim),
        "fixed_fact_token_multiset_pair_pass": bool(token_match),
        "query_pattern_counts": dict(sorted(query_counts.items())),
        "world_topology_truth_crosstab": {
            f"{topology}|{str(truth).lower()}": topology_truth[(topology, truth)]
            for topology in ("A", "B") for truth in (False, True)
        },
        "world_topology_and_truth_balanced": len(set(topology_truth.values())) == 1,
    }


def binding_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = re.compile(r"(?<![A-Za-z0-9])[AB01](?![A-Za-z0-9])")
    reversal: dict[tuple[str, str], set[bool]] = defaultdict(set)
    cue_clean = True
    for row in rows:
        for variant in row["variants"]:
            cue_clean &= forbidden.search(variant["state_cue"]) is None
            reversal[(row["mapping_reversal_id"], variant["surface"])].add(row["mapping_flip"])
    return {
        "semantic_sample_count": len(rows),
        "variant_row_count": len(rows) * len(SURFACES),
        "state_cues_omit_candidate_symbols": bool(cue_clean),
        "mapping_reversal_groups_complete": all(values == {False, True} for values in reversal.values()),
        "mapping_reversal_group_count": len(reversal),
        "truth_counts": dict(sorted(Counter(str(row["truth_value"]).lower() for row in rows).items())),
        "label_system_counts": dict(sorted(Counter(row["label_system"] for row in rows).items())),
    }


def parser_audit() -> dict[str, Any]:
    fixtures = [
        ("The statement is supported.", True),
        ("The statement is contradicted.", False),
        ("The statement is supported. The statement is contradicted.", None),
        ("I think the statement is supported.", None),
        ("", None),
    ]
    natural_pass = all(parse_natural_event(text)["semantic_value"] is expected for text, expected in fixtures)
    mapping_pass = (
        parse_mapping_event("The holding symbol is A.")["symbol"] == "A"
        and parse_mapping_event("A")["symbol"] is None
        and parse_mapping_event("The holding symbol is A. or B.")["symbol"] is None
    )
    label_pass = parse_free_label("A")["symbol"] == "A" and parse_free_label("A.")["symbol"] is None
    return {
        "natural_target_blind_fixture_pass": natural_pass,
        "mapping_target_blind_fixture_pass": mapping_pass,
        "strict_label_fixture_pass": label_pass,
        "human_blind_review_claimed": False,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_files = {}
    relation_audits = {}
    binding_audits = {}
    relation_rows_by_split = {}
    binding_rows_by_split = {}
    for split, spec in RELATION_SPLITS.items():
        rows = relation_samples(split, spec)
        relation_rows_by_split[split] = rows
        path = OUT_DIR / f"phase518_{split}_relation.jsonl"
        write_jsonl(path, rows)
        split_files[f"{split}_relation"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sealed": spec["sealed"],
        }
        relation_audits[split] = relation_audit(rows)
    for split, spec in BINDING_SPLITS.items():
        rows = binding_samples(split, spec)
        binding_rows_by_split[split] = rows
        path = OUT_DIR / f"phase518_{split}_binding.jsonl"
        write_jsonl(path, rows)
        split_files[f"{split}_binding"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "sealed": spec["sealed"],
        }
        binding_audits[split] = binding_audit(rows)

    relation_ids = {
        split: {row["sample_id"] for row in rows}
        for split, rows in relation_rows_by_split.items()
    }
    binding_ids = {
        split: {row["sample_id"] for row in rows}
        for split, rows in binding_rows_by_split.items()
    }
    contract = {
        "schema_version": "phase518_world_query_platform_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_execution",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "models_in_required_order": list(MODELS),
        "surfaces": list(SURFACES),
        "natural_event_sentences": {str(key).lower(): value for key, value in NATURAL_SENTENCES.items()},
        "split_files": split_files,
        "gates": {
            "natural_relation": {
                "surface_lcb95_min": 0.90,
                "surface_intersection_lcb95_min": 0.90,
                "four_way_lcb95_min": 0.85,
                "candidate_surface_lcb95_min": 0.90,
                "unrecoverable_ucb95_max": 0.05,
            },
            "binding": {
                "mapping_comprehension_lcb95_min": 0.90,
                "candidate_surface_lcb95_min": 0.90,
                "mapping_reversal_lcb95_min": 0.85,
                "strict_free_output_lcb95_min": 0.90,
                "mean_non_candidate_mass_max": 0.05,
            },
        },
        "physical_design": {
            "position_roles": [
                "target_evidence_end", "distractor_evidence_end", "claim_entity_end",
                "claim_relation_end", "claim_end", "prompt_end",
            ],
            "projection_dimension": 48,
            "projection_seeds": [518031, 518037, 518041],
            "group_folds": 4,
            "minimum_fold_passes": 3,
            "role_local_minimum_contiguous_layers": 4,
            "projection_consensus_required": 3,
            "discovery_local_gate": {
                "surface_lcb95_min": 0.80,
                "overall_lcb95_min": 0.80,
                "four_way_lcb95_min": 0.70,
            },
            "prediction_gate": {
                "surface_lcb95_min": 0.90,
                "overall_lcb95_min": 0.90,
                "four_way_lcb95_min": 0.75,
            },
            "pipeline_permutation_count": 128,
            "pipeline_permutation_seed": 518901,
            "permutation_quantile": 0.99,
            "platform_connectivity": "contiguous layers within the same position role only",
            "cross_projection_comparison": "prediction and interval overlap, never direction angle",
        },
        "evidence_boundaries": {
            "canonical_sentence_is_spontaneous_natural_language": False,
            "human_blind_review_claimed": False,
            "distance_correlation_proves_world_state": False,
            "platform_is_compute_transport": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "sealed_read": False,
        },
    }
    contract_path = OUT_DIR / "phase518_frozen_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    static = {
        "schema_version": "phase518_world_query_platform_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run",
        "contract_sha256": sha256_file(contract_path),
        "relation_splits": relation_audits,
        "binding_splits": binding_audits,
        "parser_audit": parser_audit(),
        "relation_split_ids_disjoint": all(
            relation_ids[left].isdisjoint(relation_ids[right])
            for left, right in itertools.combinations(relation_ids, 2)
        ),
        "binding_split_ids_disjoint": all(
            binding_ids[left].isdisjoint(binding_ids[right])
            for left, right in itertools.combinations(binding_ids, 2)
        ),
        "entity_pools_disjoint": all(
            set(NAME_POOLS[left]).isdisjoint(NAME_POOLS[right])
            for left, right in itertools.combinations(NAME_POOLS, 2)
        ),
        "relation_vocabularies_disjoint": all(
            set(RELATIONS[left]).isdisjoint(RELATIONS[right])
            for left, right in itertools.combinations(RELATIONS, 2)
        ),
        "cuda_used": False,
        "model_loaded": False,
        "sealed_read": False,
    }
    static_path = OUT_DIR / "phase518_static_audit.json"
    static_path.write_text(
        json.dumps(static, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(contract_path)
    print(static_path)


if __name__ == "__main__":
    main()
