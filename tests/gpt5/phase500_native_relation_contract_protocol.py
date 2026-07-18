#!/usr/bin/env python3
"""Freeze Phase500 staged native relation behavior contracts.

The full denominator is generated before model execution, but downstream
scripts may read a split only after the preceding gate authorizes it. Paired
worlds keep claims, rules, fact counts, and fact token multisets fixed.
"""

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
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase500_native_relation_contract_protocol"

FUNCTION_CLASSES = (
    "direct_symmetric",
    "direct_directed",
    "single_step_rule",
    "transitive_closure",
    "nontransitive_exclusion",
)
POLARITIES = ("positive", "explicit_negative", "reverse_query")
VOCAB_SYSTEMS = ("natural_names", "structured_ids", "historical_ids")
OBSERVERS = ("true_false", "mapped_ab", "mapped_01")
SURFACES = ("identity", "native_plain_candidate")
POSITION_ROLES = (
    "target_evidence_end",
    "distractor_evidence_end",
    "claim_entity_end",
    "claim_relation_end",
    "claim_end",
    "prompt_end",
)

SPLITS = {
    "function_polarity_calibration": {"index": 0, "pairs": 24, "vocabs": ("natural_names",), "observers": ("true_false",), "sealed": False},
    "vocab_observer_calibration": {"index": 1, "pairs": 24, "vocabs": VOCAB_SYSTEMS, "observers": OBSERVERS, "sealed": False},
    "independent_confirmation": {"index": 2, "pairs": 48, "vocabs": VOCAB_SYSTEMS, "observers": OBSERVERS, "sealed": False},
    "open_conditional_physical": {"index": 3, "pairs": 24, "vocabs": VOCAB_SYSTEMS, "observers": OBSERVERS, "sealed": False},
    "sealed_native_contract": {"index": 4, "pairs": 24, "vocabs": VOCAB_SYSTEMS, "observers": OBSERVERS, "sealed": True},
}

NATURAL_NAME_POOLS = {
    0: ("Alex", "Blake", "Casey", "Dana", "Erin", "Finn", "Gray", "Harper", "Jamie", "Kelly", "Morgan", "Riley"),
    1: ("Alice", "Brian", "Chloe", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack", "Laura", "Noah"),
    2: ("Aaron", "Bella", "Clara", "Dylan", "Eva", "Felix", "Gina", "Hugo", "Ivy", "Jason", "Lena", "Owen"),
    3: ("Amelia", "Ben", "Cora", "Daniel", "Ella", "Fred", "Hannah", "Ian", "Julia", "Kevin", "Maya", "Nolan"),
    4: ("Ava", "Caleb", "Daisy", "Ethan", "Faith", "George", "Hazel", "Isaac", "Jade", "Liam", "Mia", "Oscar"),
}


def stable_hash(*parts: object, n: int = 20) -> str:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


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


def token_counter(texts: list[str]) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z0-9_]+", " ".join(texts).lower()))


def entity(split_index: int, function_index: int, pair_index: int, slot: int, vocab: str) -> str:
    if vocab == "natural_names":
        pool = NATURAL_NAME_POOLS[split_index]
        return pool[(pair_index * 5 + function_index * 3 + slot) % len(pool)]
    if vocab == "structured_ids":
        return f"Entity-{split_index}{function_index}-{pair_index:03d}-{slot}"
    if vocab == "historical_ids":
        return f"u{split_index}{function_index}{pair_index:03d}{slot}"
    raise ValueError(vocab)


def attribute(split_index: int, function_index: int, pair_index: int, slot: int, vocab: str, kind: str) -> str:
    if vocab == "natural_names":
        natural = {
            "group": ("amber group", "cobalt group"),
            "marker": ("round marker", "square marker"),
            "container": ("north box", "south box", "east box", "west box"),
        }[kind]
        return natural[slot % len(natural)]
    if vocab == "structured_ids":
        return f"{kind.title()}-{split_index}{function_index}-{pair_index:03d}-{slot}"
    return f"{kind[0]}{split_index}{function_index}{pair_index:03d}{slot}"


def build_worlds(split_index: int, function_class: str, pair_index: int, vocab: str) -> dict[str, Any]:
    fi = FUNCTION_CLASSES.index(function_class)
    e = [entity(split_index, fi, pair_index, slot, vocab) for slot in range(8)]
    if len(set(e[:5])) < 5:
        raise RuntimeError("Entity collision inside a world")
    if function_class == "direct_symmetric":
        rules = ["Pairing is mutual.", "Only explicitly listed pairs are paired."]
        supporting = [f"{e[0]} is paired with {e[1]}.", f"{e[2]} is paired with {e[3]}."]
        opposing = [f"{e[0]} is paired with {e[2]}.", f"{e[1]} is paired with {e[3]}."]
        claim = f"{e[0]} is paired with {e[1]}."
        relation = "paired with"
        claim_entity = e[0]
    elif function_class == "direct_directed":
        rules = ["Mentoring is directed and not automatically reciprocal.", "Only explicitly listed mentoring links hold."]
        supporting = [f"{e[0]} mentors {e[1]}.", f"{e[2]} mentors {e[3]}."]
        opposing = [f"{e[0]} mentors {e[2]}.", f"{e[1]} mentors {e[3]}."]
        claim = f"{e[1]} is mentored by {e[0]}."
        relation = "mentored by"
        claim_entity = e[1]
    elif function_class == "single_step_rule":
        g0 = attribute(split_index, fi, pair_index, 0, vocab, "group")
        g1 = attribute(split_index, fi, pair_index, 1, vocab, "group")
        m0 = attribute(split_index, fi, pair_index, 0, vocab, "marker")
        m1 = attribute(split_index, fi, pair_index, 1, vocab, "marker")
        rules = [
            f"Every member of {g0} carries {m0}.",
            f"Every member of {g1} carries {m1}.",
            "Each listed person belongs to exactly one listed group.",
        ]
        supporting = [f"{e[0]} belongs to {g0}.", f"{e[1]} belongs to {g1}."]
        opposing = [f"{e[0]} belongs to {g1}.", f"{e[1]} belongs to {g0}."]
        claim = f"{e[0]} carries {m0}."
        relation = "carries"
        claim_entity = e[0]
    elif function_class == "transitive_closure":
        rules = [
            "Precedence is transitive: if one item precedes a second and the second precedes a third, the first precedes the third.",
            "Precedence is one-way and irreflexive.",
        ]
        supporting = [f"{e[0]} precedes {e[1]}.", f"{e[1]} precedes {e[2]}.", f"{e[3]} precedes {e[4]}."]
        opposing = [f"{e[2]} precedes {e[1]}.", f"{e[1]} precedes {e[0]}.", f"{e[4]} precedes {e[3]}."]
        claim = f"{e[0]} precedes {e[2]}."
        relation = "precedes"
        claim_entity = e[0]
    elif function_class == "nontransitive_exclusion":
        rules = ["Points-directly-to is directed and non-transitive.", "Only explicitly listed direct links hold."]
        supporting = [
            f"{e[0]} points directly to {e[1]}.",
            f"{e[1]} points directly to {e[2]}.",
            f"{e[0]} points directly to {e[2]}.",
            f"{e[3]} points directly to {e[4]}.",
        ]
        opposing = [
            f"{e[0]} points directly to {e[1]}.",
            f"{e[1]} points directly to {e[2]}.",
            f"{e[2]} points directly to {e[0]}.",
            f"{e[4]} points directly to {e[3]}.",
        ]
        claim = f"{e[0]} points directly to {e[2]}."
        relation = "points directly to"
        claim_entity = e[0]
    else:
        raise ValueError(function_class)
    return {
        "rules": rules,
        "supporting": supporting,
        "opposing": opposing,
        "positive_claim": claim,
        "claim_entity": claim_entity,
        "claim_relation": relation,
    }


def distractors(split_index: int, function_class: str, pair_index: int, vocab: str, length: str) -> list[str]:
    count = 2 if length == "compact" else 6
    fi = FUNCTION_CLASSES.index(function_class)
    rows = []
    for index in range(count):
        item = entity(split_index, fi, pair_index, 5 + (index % 3), vocab)
        box = attribute(split_index, fi, pair_index, index, vocab, "container")
        rows.append(f"{item} is stored in {box}.")
    return rows


def order_facts(target: list[str], extra: list[str], order: str) -> list[str]:
    if order == "target_first":
        return target + extra
    if order == "distractor_first":
        return extra + target
    if order != "interleaved":
        raise ValueError(order)
    rows = []
    for left, right in itertools.zip_longest(extra, target):
        if left is not None:
            rows.append(left)
        if right is not None:
            rows.append(right)
    return rows


def observer_contract(observer: str, flip: bool) -> dict[str, str]:
    if observer == "true_false":
        return {"mapping_line": "", "true_candidate": " true", "false_candidate": " false", "answer_marker": "Answer:"}
    if observer == "mapped_ab":
        if flip:
            line = "Output code: A means false and B means true."
            true_candidate, false_candidate = " B", " A"
        else:
            line = "Output code: A means true and B means false."
            true_candidate, false_candidate = " A", " B"
        return {"mapping_line": line, "true_candidate": true_candidate, "false_candidate": false_candidate, "answer_marker": "Code:"}
    if observer == "mapped_01":
        if flip:
            line = "Output code: 0 means false and 1 means true."
            true_candidate, false_candidate = "1", "0"
        else:
            line = "Output code: 0 means true and 1 means false."
            true_candidate, false_candidate = "0", "1"
        # The prompt owns the boundary space. This keeps 0/1 as one token in
        # all three frozen tokenizers instead of scoring a shared space token.
        return {"mapping_line": line, "true_candidate": true_candidate, "false_candidate": false_candidate, "answer_marker": "Code: "}
    raise ValueError(observer)


def render(
    surface: str,
    observer: str,
    flip: bool,
    polarity: str,
    rules: list[str],
    facts: list[str],
    target_facts: list[str],
    extra: list[str],
    positive_claim: str,
    claim_entity: str,
    claim_relation: str,
) -> dict[str, Any]:
    if polarity == "explicit_negative":
        displayed_claim = f"It is not the case that {positive_claim[:-1]}."
        question_identity = "Is the displayed claim true or false?"
        question_plain = "Does the proposition hold in this world?"
    elif polarity == "reverse_query":
        displayed_claim = positive_claim
        question_identity = "Is the displayed claim false?"
        question_plain = "Does the proposition fail to hold in this world?"
    elif polarity == "positive":
        displayed_claim = positive_claim
        question_identity = "Is the displayed claim true or false?"
        question_plain = "Does the proposition hold in this world?"
    else:
        raise ValueError(polarity)
    contract = observer_contract(observer, flip)
    rules_block = "\n".join(f"- {rule}" for rule in rules)
    facts_block = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(facts))
    mapping = f"\n{contract['mapping_line']}" if contract["mapping_line"] else ""
    if surface == "identity":
        prompt = (
            "Use only this closed synthetic world.\n"
            f"Rules:\n{rules_block}\nFacts:\n{facts_block}\n"
            f"Claim:\n{displayed_claim}{mapping}\nQuestion: {question_identity}\n{contract['answer_marker']}"
        )
    elif surface == "native_plain_candidate":
        prompt = (
            "Judge only from the miniature world below.\n"
            f"World rules:\n{rules_block}\nWorld evidence:\n{facts_block}\n"
            f"Proposition:\n{displayed_claim}{mapping}\nTask: {question_plain}\n{contract['answer_marker']}"
        )
    else:
        raise ValueError(surface)
    claim_start = prompt.rfind(displayed_claim)
    entity_offset = displayed_claim.find(claim_entity)
    relation_offset = displayed_claim.find(claim_relation)
    if min(claim_start, entity_offset, relation_offset) < 0:
        raise RuntimeError("Claim anchor missing")
    last_target = max(target_facts, key=lambda fact: prompt.rfind(fact))
    last_extra = max(extra, key=lambda fact: prompt.rfind(fact))
    roles = {
        "target_evidence_end": prompt.rfind(last_target) + len(last_target),
        "distractor_evidence_end": prompt.rfind(last_extra) + len(last_extra),
        "claim_entity_end": claim_start + entity_offset + len(claim_entity),
        "claim_relation_end": claim_start + relation_offset + len(claim_relation),
        "claim_end": claim_start + len(displayed_claim),
        "prompt_end": len(prompt),
    }
    if tuple(roles) != POSITION_ROLES:
        raise RuntimeError("Position role order drift")
    return {
        "surface": surface,
        "observer": observer,
        "mapping_flip": flip,
        "prompt": prompt,
        "true_candidate": contract["true_candidate"],
        "false_candidate": contract["false_candidate"],
        "role_char_ends": roles,
    }


def controls(pair_count: int) -> list[tuple[str, str, int]]:
    repeats = pair_count // 6
    rows = list(itertools.product(("compact", "extended"), ("target_first", "distractor_first", "interleaved"), range(repeats)))
    if len(rows) != pair_count:
        raise RuntimeError("Pair count must be divisible by six")
    return rows


def build_split(split: str) -> list[dict[str, Any]]:
    spec = SPLITS[split]
    rows = []
    for function_class in FUNCTION_CLASSES:
        for polarity in POLARITIES:
            for vocab in spec["vocabs"]:
                for pair_index, (length, fact_order, replicate) in enumerate(controls(spec["pairs"])):
                    world = build_worlds(spec["index"], function_class, pair_index, vocab)
                    extra = distractors(spec["index"], function_class, pair_index, vocab, length)
                    pair_id = f"p500-{split}-{function_class}-{polarity}-{vocab}-{pair_index:03d}-{stable_hash(split, function_class, polarity, vocab, pair_index)}"
                    flip = bool((pair_index + FUNCTION_CLASSES.index(function_class) + POLARITIES.index(polarity)) % 2)
                    for world_role, positive_truth in (("supporting_world", True), ("opposing_world", False)):
                        target = world["supporting"] if positive_truth else world["opposing"]
                        facts = order_facts(target, extra, fact_order)
                        truth = positive_truth if polarity == "positive" else not positive_truth
                        sample_id = f"p500-{stable_hash(pair_id, world_role)}"
                        variants = [
                            render(
                                surface,
                                observer,
                                flip,
                                polarity,
                                world["rules"],
                                facts,
                                target,
                                extra,
                                world["positive_claim"],
                                world["claim_entity"],
                                world["claim_relation"],
                            )
                            for observer in spec["observers"]
                            for surface in SURFACES
                        ]
                        exact_match = world["positive_claim"] in facts
                        local_prediction = exact_match if polarity == "positive" else not exact_match
                        rows.append({
                            "schema_version": "phase500_native_contract_sample.v1",
                            "sample_id": sample_id,
                            "source_pair_id": pair_id,
                            "split": split,
                            "sealed": bool(spec["sealed"]),
                            "function_class": function_class,
                            "polarity": polarity,
                            "vocab_system": vocab,
                            "pair_index": pair_index,
                            "replicate": replicate,
                            "length_control": length,
                            "fact_order": fact_order,
                            "world_role": world_role,
                            "positive_world_truth": positive_truth,
                            "truth_value": truth,
                            "rules": world["rules"],
                            "target_facts": target,
                            "distractor_facts": extra,
                            "facts": facts,
                            "positive_claim": world["positive_claim"],
                            "claim_entity": world["claim_entity"],
                            "claim_relation": world["claim_relation"],
                            "deterministic_baselines": {
                                "bag_of_tokens_can_distinguish_pair": False,
                                "positive_claim_exact_fact": exact_match,
                                "exact_match_with_polarity_prediction": local_prediction,
                            },
                            "variants": variants,
                        })
    return rows


def audit_split(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = SPLITS[split]
    expected = len(FUNCTION_CLASSES) * len(POLARITIES) * len(spec["vocabs"]) * spec["pairs"] * 2
    if len(rows) != expected:
        raise RuntimeError(f"{split}: expected {expected}, got {len(rows)}")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["source_pair_id"]].append(row)
        expected_variants = len(spec["observers"]) * len(SURFACES)
        if len(row["variants"]) != expected_variants:
            raise RuntimeError("Variant count drift")
    for pair_id, pair in by_pair.items():
        if len(pair) != 2:
            raise RuntimeError(f"Incomplete pair {pair_id}")
        left, right = pair
        if left["positive_claim"] != right["positive_claim"] or left["rules"] != right["rules"]:
            raise RuntimeError(f"Claim/rule drift {pair_id}")
        if len(left["facts"]) != len(right["facts"]):
            raise RuntimeError(f"Fact-count drift {pair_id}")
        if token_counter(left["facts"]) != token_counter(right["facts"]):
            raise RuntimeError(f"Fact-token drift {pair_id}")
        if {left["truth_value"], right["truth_value"]} != {False, True}:
            raise RuntimeError(f"Truth imbalance {pair_id}")
    baseline = {}
    for function_class in FUNCTION_CLASSES:
        selected = [row for row in rows if row["function_class"] == function_class]
        count = sum(row["deterministic_baselines"]["exact_match_with_polarity_prediction"] == row["truth_value"] for row in selected)
        baseline[function_class] = {"n": len(selected), "count": count, "rate": count / len(selected)}
    return {
        "row_count": len(rows),
        "variant_count": sum(len(row["variants"]) for row in rows),
        "pair_count": len(by_pair),
        "function_counts": dict(sorted(Counter(row["function_class"] for row in rows).items())),
        "polarity_counts": dict(sorted(Counter(row["polarity"] for row in rows).items())),
        "vocab_counts": dict(sorted(Counter(row["vocab_system"] for row in rows).items())),
        "fixed_claim_pair_pass": True,
        "fixed_fact_token_multiset_pair_pass": True,
        "deterministic_exact_match_baseline": baseline,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_rows = {split: build_split(split) for split in SPLITS}
    audits = {split: audit_split(split, rows) for split, rows in split_rows.items()}
    files = {}
    for split, rows in split_rows.items():
        path = OUT_DIR / f"phase500_{split}.jsonl"
        write_jsonl(path, rows)
        files[split] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "variants": sum(len(row["variants"]) for row in rows),
            "sealed": SPLITS[split]["sealed"],
        }
    contract = {
        "schema_version": "phase500_native_relation_contract.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_run",
        "function_classes": list(FUNCTION_CLASSES),
        "polarities": list(POLARITIES),
        "vocab_systems": list(VOCAB_SYSTEMS),
        "observers": list(OBSERVERS),
        "surfaces": list(SURFACES),
        "position_roles": list(POSITION_ROLES),
        "stage_order": ["function_polarity_calibration", "vocab_observer_calibration", "independent_confirmation", "open_conditional_physical"],
        "behavior_gate": {
            "identity_lcb95_min": 0.85,
            "native_plain_lcb95_min": 0.85,
            "surface_intersection_lcb95_min": 0.85,
            "paired_world_lcb95_min": 0.80,
            "observer_consistency_lcb95_min": 0.85,
        },
        "selection_rules": {
            "stage_a_to_b": "model-specific function_class x polarity cell must pass the frozen true_false natural_names gate",
            "stage_b_to_c": "all three observers and their semantic consistency must pass for a model x function x polarity x vocab cell",
            "shared_physical": "at least two models independently confirm the identical function x polarity x vocab contract",
            "no_failed_cell_pooling": True,
        },
        "projection": {"dimension": 64, "seeds": {"qwen3": 500031, "glm4": 500037, "deepseek7b": 500041}},
        "evidence_boundaries": {
            "sealed_read": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "role_sequence_is_compute_transport": False,
        },
        "split_files": files,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
    }
    contract_path = OUT_DIR / "phase500_frozen_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "phase500_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run",
        "model_loaded": False,
        "cuda_used": False,
        "splits": audits,
        "contract_sha256": sha256_file(contract_path),
        "authorization": {"function_polarity_calibration": True},
    }
    audit_path = OUT_DIR / "phase500_static_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(contract_path)
    print(audit_path)


if __name__ == "__main__":
    main()
