#!/usr/bin/env python3
"""Freeze Phase509 relation, label-binding, and joint contracts.

The protocol separates relation evaluation from arbitrary-label compilation.
All splits are generated before model execution. Downstream runners read only
the split authorized by the preceding frozen gate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
OUT_DIR = ROOT / "tests/gpt5/result/phase509_dual_contract_protocol"

SURFACES = ("identity", "native_plain_candidate")
LABEL_SYSTEMS = ("mapped_ab", "mapped_01")
MAPPING_ORDERS = ("state_first", "mapping_first")
LENGTHS = ("compact", "extended")
FACT_ORDERS = ("target_first", "distractor_first", "interleaved")

SPLITS = {
    "calibration": {"index": 0, "relation_pairs": 48, "binding_repeats": 2, "sealed": False},
    "confirmation": {"index": 1, "relation_pairs": 96, "binding_repeats": 4, "sealed": False},
    "joint_confirmation": {"index": 2, "relation_pairs": 96, "binding_repeats": 0, "sealed": False},
    "physical_fit": {"index": 3, "relation_pairs": 48, "binding_repeats": 1, "sealed": False},
    "physical_prediction": {"index": 4, "relation_pairs": 96, "binding_repeats": 2, "sealed": False},
    "sealed": {"index": 5, "relation_pairs": 96, "binding_repeats": 2, "sealed": True},
}

NAME_POOLS = {
    0: ("Arlo", "Briar", "Celia", "Damon", "Elise", "Flint", "Greta", "Harlan", "Indra", "Jonas", "Kira", "Lucan"),
    1: ("Anya", "Basil", "Clio", "Derek", "Esme", "Fabian", "Gemma", "Heath", "Ines", "Jasper", "Lola", "Micah"),
    2: ("Alden", "Bianca", "Cyrus", "Delia", "Emil", "Freya", "Gavin", "Helena", "Ivor", "Juno", "Leah", "Milo"),
    3: ("Alma", "Bennett", "Cleo", "Dario", "Eden", "Faye", "Gideon", "Holly", "Idris", "Josie", "Kellan", "Mara"),
    4: ("Abel", "Bea", "Cedric", "Della", "Eli", "Flora", "Grant", "Hope", "Imani", "Joel", "Keira", "Miles"),
    5: ("Ari", "Belle", "Calvin", "Dora", "Ewan", "Fern", "Glen", "Hera", "Ivan", "Jill", "Kian", "Mona"),
}

RELATIONS = {
    0: (("mentors", "is mentored by"), ("supervises", "is supervised by")),
    1: (("guides", "is guided by"), ("coaches", "is coached by")),
    2: (("advises", "is advised by"), ("tutors", "is tutored by")),
    3: (("trains", "is trained by"), ("directs", "is directed by")),
    4: (("instructs", "is instructed by"), ("sponsors", "is sponsored by")),
    5: (("briefs", "is briefed by"), ("assists", "is assisted by")),
}

STATE_CUES = {
    0: {
        True: ("The supplied record supports the proposition.", "The verdict accepts the proposition.", "The check confirms that the proposition holds.", "The record validates the proposition."),
        False: ("The supplied record rejects the proposition.", "The verdict rejects the proposition.", "The check denies that the proposition holds.", "The record invalidates the proposition."),
    },
    1: {
        True: ("The evidence endorses the proposition.", "The assessment accepts the stated proposition.", "The review finds the proposition to hold.", "The decision validates the stated proposition."),
        False: ("The evidence contradicts the proposition.", "The assessment refuses the stated proposition.", "The review finds that the proposition does not hold.", "The decision invalidates the stated proposition."),
    },
    2: {
        True: ("The proposition receives an affirmative verdict.", "The record agrees with the proposition.", "The evaluation confirms the proposition.", "The proposition survives the check."),
        False: ("The proposition receives a negative verdict.", "The record disagrees with the proposition.", "The evaluation rejects the proposition.", "The proposition fails the check."),
    },
    3: {
        True: ("The proposition is accepted by the evaluator.", "The available record favors the proposition.", "The adjudication upholds the proposition.", "The conclusion endorses the proposition."),
        False: ("The proposition is declined by the evaluator.", "The available record opposes the proposition.", "The adjudication overturns the proposition.", "The conclusion rejects the proposition."),
    },
    4: {
        True: ("The proposition is upheld after review.", "The final assessment favors the proposition.", "The evidence is compatible with the proposition.", "The proposition passes evaluation."),
        False: ("The proposition is overturned after review.", "The final assessment opposes the proposition.", "The evidence is incompatible with the proposition.", "The proposition fails evaluation."),
    },
    5: {
        True: ("The proposition is approved by the record.", "The ruling supports the proposition.", "The finding agrees with the proposition.", "The proposition is retained."),
        False: ("The proposition is disapproved by the record.", "The ruling opposes the proposition.", "The finding conflicts with the proposition.", "The proposition is discarded."),
    },
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


def entity(split_index: int, pair_index: int, slot: int) -> str:
    pool = NAME_POOLS[split_index]
    return pool[(pair_index * 5 + slot) % len(pool)]


def controls(pair_count: int) -> list[tuple[str, str, int]]:
    repeats = pair_count // 6
    values = list(itertools.product(LENGTHS, FACT_ORDERS, range(repeats)))
    if len(values) != pair_count:
        raise RuntimeError("relation pair count must be divisible by six")
    return values


def order_facts(target: list[str], distractors: list[str], fact_order: str) -> list[str]:
    if fact_order == "target_first":
        return target + distractors
    if fact_order == "distractor_first":
        return distractors + target
    if fact_order != "interleaved":
        raise ValueError(fact_order)
    rows = []
    for left, right in itertools.zip_longest(distractors, target):
        if left is not None:
            rows.append(left)
        if right is not None:
            rows.append(right)
    return rows


def relation_world(split_index: int, pair_index: int) -> dict[str, Any]:
    names = [entity(split_index, pair_index, slot) for slot in range(8)]
    if len(set(names[:5])) < 5:
        raise RuntimeError("entity collision")
    active, passive = RELATIONS[split_index][pair_index % len(RELATIONS[split_index])]
    support = [f"{names[0]} {active} {names[1]}.", f"{names[2]} {active} {names[3]}."]
    oppose = [f"{names[0]} {active} {names[2]}.", f"{names[1]} {active} {names[3]}."]
    claim = f"{names[1]} {passive} {names[0]}."
    rules = [
        f"The relation expressed by '{active}' is directed and is not automatically reciprocal.",
        f"Only explicitly listed '{active}' links hold in this world.",
    ]
    return {
        "active_relation": active,
        "passive_relation": passive,
        "support": support,
        "oppose": oppose,
        "claim": claim,
        "claim_entity": names[1],
        "names": names,
        "rules": rules,
    }


def distractors(world: dict[str, Any], length: str) -> list[str]:
    count = 2 if length == "compact" else 6
    places = ("north shelf", "south shelf", "east shelf", "west shelf")
    return [
        f"{world['names'][4 + index % 4]} is listed beside the {places[index % len(places)]}."
        for index in range(count)
    ]


def natural_observer() -> dict[str, str]:
    return {
        "mapping_line": "",
        "answer_marker": "Answer:",
        "true_candidate": " true",
        "false_candidate": " false",
    }


def mapped_observer(label_system: str, flip: bool, template_index: int = 0) -> dict[str, str]:
    if label_system == "mapped_ab":
        if flip:
            line = (
                "Use B when the proposition holds and A when it does not hold."
                if template_index == 0
                else "The code for a holding proposition is B; the code for a non-holding proposition is A."
            )
            true_candidate, false_candidate = " B", " A"
        else:
            line = (
                "Use A when the proposition holds and B when it does not hold."
                if template_index == 0
                else "The code for a holding proposition is A; the code for a non-holding proposition is B."
            )
            true_candidate, false_candidate = " A", " B"
        return {
            "mapping_line": line,
            "answer_marker": "Code:",
            "true_candidate": true_candidate,
            "false_candidate": false_candidate,
        }
    if label_system == "mapped_01":
        if flip:
            line = (
                "Use 1 when the proposition holds and 0 when it does not hold."
                if template_index == 0
                else "The code for a holding proposition is 1; the code for a non-holding proposition is 0."
            )
            true_candidate, false_candidate = "1", "0"
        else:
            line = (
                "Use 0 when the proposition holds and 1 when it does not hold."
                if template_index == 0
                else "The code for a holding proposition is 0; the code for a non-holding proposition is 1."
            )
            true_candidate, false_candidate = "0", "1"
        return {
            "mapping_line": line,
            "answer_marker": "Code: ",
            "true_candidate": true_candidate,
            "false_candidate": false_candidate,
        }
    raise ValueError(label_system)


def relation_prompt(
    surface: str,
    rules: list[str],
    facts: list[str],
    target: list[str],
    extra: list[str],
    claim: str,
    claim_entity: str,
    passive_relation: str,
    observer: dict[str, str],
) -> dict[str, Any]:
    rules_block = "\n".join(f"- {item}" for item in rules)
    facts_block = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(facts))
    mapping = f"\n{observer['mapping_line']}" if observer["mapping_line"] else ""
    if surface == "identity":
        prompt = (
            "Use only the following closed world.\n"
            f"Rules:\n{rules_block}\nFacts:\n{facts_block}\n"
            f"Claim:\n{claim}{mapping}\nQuestion: Is the claim true or false?\n{observer['answer_marker']}"
        )
    elif surface == "native_plain_candidate":
        prompt = (
            "Judge the proposition only from this miniature world.\n"
            f"World rules:\n{rules_block}\nWorld evidence:\n{facts_block}\n"
            f"Proposition:\n{claim}{mapping}\nTask: Decide whether the proposition holds.\n{observer['answer_marker']}"
        )
    else:
        raise ValueError(surface)
    claim_start = prompt.rfind(claim)
    entity_offset = claim.find(claim_entity)
    relation_offset = claim.find(passive_relation)
    if min(claim_start, entity_offset, relation_offset) < 0:
        raise RuntimeError("relation prompt anchor missing")
    target_end = max(prompt.rfind(item) + len(item) for item in target)
    extra_end = max(prompt.rfind(item) + len(item) for item in extra)
    mapping_end = (
        prompt.rfind(observer["mapping_line"]) + len(observer["mapping_line"])
        if observer["mapping_line"]
        else claim_start + len(claim)
    )
    return {
        "surface": surface,
        "prompt": prompt,
        "true_candidate": observer["true_candidate"],
        "false_candidate": observer["false_candidate"],
        "role_char_ends": {
            "target_evidence_end": target_end,
            "distractor_evidence_end": extra_end,
            "claim_entity_end": claim_start + entity_offset + len(claim_entity),
            "claim_relation_end": claim_start + relation_offset + len(passive_relation),
            "claim_end": claim_start + len(claim),
            "mapping_instruction_end": mapping_end,
            "prompt_end": len(prompt),
        },
    }


def build_relation_split(split: str) -> list[dict[str, Any]]:
    spec = SPLITS[split]
    rows = []
    for pair_index, (length, fact_order, replicate) in enumerate(controls(spec["relation_pairs"])):
        world = relation_world(spec["index"], pair_index)
        extra = distractors(world, length)
        pair_id = f"p509-r-{split}-{pair_index:03d}-{stable_hash(split, 'relation', pair_index)}"
        for world_role, truth in (("supporting_world", True), ("opposing_world", False)):
            target = world["support"] if truth else world["oppose"]
            facts = order_facts(target, extra, fact_order)
            variants = [
                relation_prompt(
                    surface,
                    world["rules"],
                    facts,
                    target,
                    extra,
                    world["claim"],
                    world["claim_entity"],
                    world["passive_relation"],
                    natural_observer(),
                )
                for surface in SURFACES
            ]
            rows.append({
                "schema_version": "phase509_relation_sample.v1",
                "contract_type": "relation_evaluation",
                "sample_id": f"p509-r-{stable_hash(pair_id, world_role)}",
                "source_pair_id": pair_id,
                "split": split,
                "sealed": bool(spec["sealed"]),
                "truth_value": truth,
                "world_role": world_role,
                "pair_index": pair_index,
                "replicate": replicate,
                "length_control": length,
                "fact_order": fact_order,
                "relation_verb": world["active_relation"],
                "claim": world["claim"],
                "rules": world["rules"],
                "facts": facts,
                "target_facts": target,
                "distractors": extra,
                "variants": variants,
            })
    return rows


def binding_prompt(
    surface: str,
    state_cue: str,
    observer: dict[str, str],
    mapping_order: str,
) -> dict[str, Any]:
    mapping = observer["mapping_line"]
    final_instruction = "Return exactly one requested code symbol."
    if surface == "identity":
        state_block = f"Evaluated state:\n{state_cue}"
        mapping_block = f"Code rule:\n{mapping}"
    elif surface == "native_plain_candidate":
        state_block = f"The evaluation report says: {state_cue}"
        mapping_block = f"Apply this output convention: {mapping}"
    else:
        raise ValueError(surface)
    blocks = [state_block, mapping_block] if mapping_order == "state_first" else [mapping_block, state_block]
    prompt = "\n".join([*blocks, final_instruction, observer["answer_marker"]])
    state_end = prompt.rfind(state_cue) + len(state_cue)
    mapping_end = prompt.rfind(mapping) + len(mapping)
    instruction_end = prompt.rfind(final_instruction) + len(final_instruction)
    if min(state_end, mapping_end, instruction_end) < 0:
        raise RuntimeError("binding prompt anchor missing")
    return {
        "surface": surface,
        "prompt": prompt,
        "true_candidate": observer["true_candidate"],
        "false_candidate": observer["false_candidate"],
        "role_char_ends": {
            "semantic_state_end": state_end,
            "mapping_instruction_end": mapping_end,
            "final_instruction_end": instruction_end,
            "prompt_end": len(prompt),
        },
    }


def build_binding_split(split: str) -> list[dict[str, Any]]:
    spec = SPLITS[split]
    if spec["binding_repeats"] == 0:
        return []
    rows = []
    cues = STATE_CUES[spec["index"]]
    for label_system, cue_index, mapping_template, mapping_order, truth, replicate, flip in itertools.product(
        LABEL_SYSTEMS,
        range(4),
        range(2),
        MAPPING_ORDERS,
        (False, True),
        range(spec["binding_repeats"]),
        (False, True),
    ):
        state_cue = cues[truth][cue_index]
        observer = mapped_observer(label_system, flip, mapping_template)
        reversal_id = (
            f"p509-b-rev-{split}-{label_system}-{cue_index}-{mapping_template}-{mapping_order}-{int(truth)}-{replicate}-"
            f"{stable_hash(split, label_system, cue_index, mapping_template, mapping_order, truth, replicate)}"
        )
        rows.append({
            "schema_version": "phase509_binding_sample.v1",
            "contract_type": "label_binding",
            "sample_id": f"p509-b-{stable_hash(reversal_id, flip)}",
            "mapping_reversal_id": reversal_id,
            "split": split,
            "sealed": bool(spec["sealed"]),
            "truth_value": truth,
            "label_system": label_system,
            "mapping_flip": flip,
            "mapping_order": mapping_order,
            "mapping_template": mapping_template,
            "cue_index": cue_index,
            "replicate": replicate,
            "state_cue": state_cue,
            "mapping_line": observer["mapping_line"],
            "variants": [
                binding_prompt(surface, state_cue, observer, mapping_order)
                for surface in SURFACES
            ],
        })
    return rows


def build_joint_split(split: str) -> list[dict[str, Any]]:
    spec = SPLITS[split]
    relation_rows = build_relation_split(split)
    rows = []
    for relation in relation_rows:
        for label_system, flip in itertools.product(LABEL_SYSTEMS, (False, True)):
            observer = mapped_observer(label_system, flip, relation["pair_index"] % 2)
            variants = []
            for surface in SURFACES:
                base = relation["variants"][SURFACES.index(surface)]
                facts = relation["facts"]
                target = relation["target_facts"]
                extra = relation["distractors"]
                world = relation_world(spec["index"], relation["pair_index"])
                variants.append(
                    relation_prompt(
                        surface,
                        relation["rules"],
                        facts,
                        target,
                        extra,
                        relation["claim"],
                        world["claim_entity"],
                        world["passive_relation"],
                        observer,
                    )
                )
                if not base["prompt"]:
                    raise RuntimeError("empty base relation prompt")
            rows.append({
                "schema_version": "phase509_joint_sample.v1",
                "contract_type": "relation_label_joint",
                "sample_id": f"p509-j-{stable_hash(relation['sample_id'], label_system, flip)}",
                "source_pair_id": relation["source_pair_id"],
                "mapping_reversal_id": f"p509-j-rev-{relation['sample_id']}-{label_system}",
                "relation_sample_id": relation["sample_id"],
                "split": split,
                "sealed": bool(spec["sealed"]),
                "truth_value": relation["truth_value"],
                "world_role": relation["world_role"],
                "pair_index": relation["pair_index"],
                "length_control": relation["length_control"],
                "fact_order": relation["fact_order"],
                "relation_verb": relation["relation_verb"],
                "label_system": label_system,
                "mapping_flip": flip,
                "variants": variants,
            })
    return rows


def audit_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_groups.setdefault(row["source_pair_id"], []).append(row)
    fixed_claim = True
    fixed_tokens = True
    for pair in pair_groups.values():
        if len(pair) != 2 or {item["truth_value"] for item in pair} != {False, True}:
            raise RuntimeError("invalid relation pair")
        fixed_claim &= len({item["claim"] for item in pair}) == 1
        fixed_tokens &= token_counter(pair[0]["facts"]) == token_counter(pair[1]["facts"])
    variant_count = sum(len(row["variants"]) for row in rows)
    fixed_true = sum(row["truth_value"] for row in rows)
    return {
        "row_count": len(rows),
        "variant_count": variant_count,
        "pair_count": len(pair_groups),
        "truth_counts": {"true": fixed_true, "false": len(rows) - fixed_true},
        "fixed_claim_pair_pass": fixed_claim,
        "fixed_fact_token_multiset_pair_pass": fixed_tokens,
        "fixed_output_baseline_rate": 0.5,
    }


def audit_binding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "variant_count": 0, "reversal_group_count": 0}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["mapping_reversal_id"], []).append(row)
    reversal_complete = all(
        len(items) == 2 and {item["mapping_flip"] for item in items} == {False, True}
        for items in groups.values()
    )
    label_counts = Counter(row["label_system"] for row in rows)
    truth_counts = Counter(str(row["truth_value"]).lower() for row in rows)
    flip_counts = Counter(str(row["mapping_flip"]).lower() for row in rows)
    candidates_absent_from_state = all(
        not re.search(r"(?:^|\s)(?:A|B|0|1)(?:\s|$)", row["state_cue"])
        for row in rows
    )
    return {
        "row_count": len(rows),
        "variant_count": sum(len(row["variants"]) for row in rows),
        "reversal_group_count": len(groups),
        "reversal_pairs_complete": reversal_complete,
        "label_system_counts": dict(sorted(label_counts.items())),
        "truth_counts": dict(sorted(truth_counts.items())),
        "mapping_flip_counts": dict(sorted(flip_counts.items())),
        "candidate_symbols_absent_from_state_cue": candidates_absent_from_state,
        "fixed_output_baseline_rate": 0.5,
        "mapping_parser_oracle_rate": 1.0,
    }


def audit_joint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "variant_count": 0, "paired_condition_count": 0}
    paired: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
    reversal: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["source_pair_id"], row["label_system"], row["mapping_flip"])
        paired.setdefault(key, []).append(row)
        reversal.setdefault(row["mapping_reversal_id"], []).append(row)
    paired_complete = all(
        len(items) == 2 and {item["truth_value"] for item in items} == {False, True}
        for items in paired.values()
    )
    reversal_complete = all(
        len(items) == 2 and {item["mapping_flip"] for item in items} == {False, True}
        for items in reversal.values()
    )
    return {
        "row_count": len(rows),
        "variant_count": sum(len(row["variants"]) for row in rows),
        "paired_condition_count": len(paired),
        "mapping_reversal_group_count": len(reversal),
        "paired_worlds_complete": paired_complete,
        "mapping_reversals_complete": reversal_complete,
        "label_system_counts": dict(sorted(Counter(row["label_system"] for row in rows).items())),
        "mapping_flip_counts": dict(sorted(Counter(str(row["mapping_flip"]).lower() for row in rows).items())),
        "fixed_output_baseline_rate": 0.5,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, Any] = {}
    static_splits: dict[str, Any] = {}
    for split in SPLITS:
        relation_rows = build_relation_split(split)
        binding_rows = build_binding_split(split)
        joint_rows = build_joint_split(split) if split in {"joint_confirmation", "physical_fit", "physical_prediction", "sealed"} else []
        for contract, rows in (
            ("relation", relation_rows),
            ("binding", binding_rows),
            ("joint", joint_rows),
        ):
            path = OUT_DIR / f"phase509_{split}_{contract}.jsonl"
            write_jsonl(path, rows)
            split_files[f"{split}_{contract}"] = {
                "path": str(path.relative_to(ROOT)),
                "rows": len(rows),
                "variants": sum(len(row.get("variants", [])) for row in rows),
                "sha256": sha256_file(path),
                "sealed": bool(SPLITS[split]["sealed"]),
            }
        static_splits[split] = {
            "relation": audit_relation(relation_rows),
            "binding": audit_binding(binding_rows),
            "joint": audit_joint(joint_rows),
        }

    contract = {
        "schema_version": "phase509_dual_contract_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_run",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "contracts": {
            "R": "natural true/false direct-directed relation evaluation",
            "B": "explicit semantic-state to arbitrary-label compilation",
            "J": "relation evaluation composed with arbitrary-label compilation",
        },
        "gates": {
            "R": {
                "identity_lcb95_min": 0.90,
                "native_plain_lcb95_min": 0.90,
                "surface_intersection_lcb95_min": 0.90,
                "paired_world_lcb95_min": 0.85,
            },
            "B": {
                "label_system_lcb95_min": 0.95,
                "mapping_flip_lcb95_min": 0.95,
                "surface_intersection_lcb95_min": 0.95,
                "mapping_reversal_lcb95_min": 0.90,
                "free_event_is_separate_gate": True,
            },
            "S": {
                "free_event_lcb95_min": 0.90,
                "mean_non_candidate_mass_max": 0.05,
            },
            "J": {
                "label_system_lcb95_min": 0.85,
                "surface_intersection_lcb95_min": 0.85,
                "paired_world_lcb95_min": 0.80,
                "mapping_reversal_lcb95_min": 0.80,
            },
        },
        "selection_rules": {
            "R_confirmation": "model must pass R calibration",
            "B_confirmation": "model must pass B calibration",
            "J_confirmation": "model must independently confirm both R and B",
            "model_specific_physical": "model may enter only the physical subgraph whose behavior contract it confirmed",
            "shared_physical": "at least two models must confirm the identical sub-contract",
            "no_failed_pooling": True,
            "product_accuracy_is_only_descriptive_null": True,
        },
        "evidence_boundaries": {
            "behavioral_dissociation_is_internal_module_proof": False,
            "physical_observation_is_causal": False,
            "head_channel_neuron_scan": False,
            "sealed_read": False,
        },
        "split_files": split_files,
    }
    contract_path = OUT_DIR / "phase509_frozen_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    static_audit = {
        "schema_version": "phase509_static_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_pass_no_model_run",
        "contract_sha256": sha256_file(contract_path),
        "cuda_used": False,
        "model_loaded": False,
        "splits": static_splits,
    }
    audit_path = OUT_DIR / "phase509_static_audit.json"
    audit_path.write_text(
        json.dumps(static_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(contract_path)
    print(audit_path)


if __name__ == "__main__":
    main()
