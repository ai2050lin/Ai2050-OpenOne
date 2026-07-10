#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase317"
SCHEMA_VERSION = "4.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def aliases(value: str) -> list[str]:
    values = [value, value.lower(), value.capitalize()]
    if value.lower() == "yes":
        values.extend(["true", "correct"])
    elif value.lower() == "no":
        values.extend(["false", "incorrect"])
    return list(dict.fromkeys(values))


def split_for(replica: int) -> str:
    return ["discovery", "calibration", "heldout"][replica]


def template_for(replica: int) -> str:
    return ["template_a", "template_b", "template_c_open"][replica]


def case_row(
    family: str,
    mechanism: str,
    target_group: str,
    replica: int,
    prompt: str,
    source_surface: str,
    query_surface: str,
    target: str,
    distractors: list[str],
    domain: str,
    item_key: str,
) -> dict[str, Any]:
    split = split_for(replica)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": f"phase317:{family}:{mechanism}:{target_group}:{replica}",
        "family_id": family,
        "mechanism_id": mechanism,
        "target_group": target_group,
        "replica": replica,
        "split": split,
        "template_id": template_for(replica),
        "domain_id": domain,
        "item_key": item_key,
        "prompt": prompt,
        "source_surface": source_surface,
        "query_surface": query_surface,
        "target": target,
        "target_aliases": aliases(target),
        "distractor_aliases": list(dict.fromkeys(distractors)),
        "independent_case": True,
        "open_template_heldout": split == "heldout",
    }


KNOWLEDGE: dict[str, list[tuple[str, str, list[str]]]] = {
    "category_binding": [
        ("robin", "animal", ["tool", "plant", "material"]),
        ("salmon", "animal", ["tool", "plant", "material"]),
        ("otter", "animal", ["tool", "plant", "material"]),
        ("hammer", "tool", ["animal", "plant", "material"]),
        ("wrench", "tool", ["animal", "plant", "material"]),
        ("chisel", "tool", ["animal", "plant", "material"]),
        ("oak", "plant", ["animal", "tool", "material"]),
        ("fern", "plant", ["animal", "tool", "material"]),
        ("cactus", "plant", ["animal", "tool", "material"]),
        ("copper", "material", ["animal", "tool", "plant"]),
        ("granite", "material", ["animal", "tool", "plant"]),
        ("rubber", "material", ["animal", "tool", "plant"]),
    ],
    "color_binding": [
        ("lemon", "yellow", ["black", "white", "green"]),
        ("sunflower", "yellow", ["black", "white", "green"]),
        ("banana", "yellow", ["black", "white", "green"]),
        ("coal", "black", ["yellow", "white", "green"]),
        ("raven", "black", ["yellow", "white", "green"]),
        ("ink", "black", ["yellow", "white", "green"]),
        ("snow", "white", ["yellow", "black", "green"]),
        ("chalk", "white", ["yellow", "black", "green"]),
        ("salt", "white", ["yellow", "black", "green"]),
        ("grass", "green", ["yellow", "black", "white"]),
        ("moss", "green", ["yellow", "black", "white"]),
        ("spinach", "green", ["yellow", "black", "white"]),
    ],
    "function_binding": [
        ("knife", "cutting", ["writing", "sitting", "opening"]),
        ("scissors", "cutting", ["writing", "sitting", "opening"]),
        ("saw", "cutting", ["writing", "sitting", "opening"]),
        ("pen", "writing", ["cutting", "sitting", "opening"]),
        ("pencil", "writing", ["cutting", "sitting", "opening"]),
        ("marker", "writing", ["cutting", "sitting", "opening"]),
        ("chair", "sitting", ["cutting", "writing", "opening"]),
        ("stool", "sitting", ["cutting", "writing", "opening"]),
        ("bench", "sitting", ["cutting", "writing", "opening"]),
        ("key", "opening", ["cutting", "writing", "sitting"]),
        ("opener", "opening", ["cutting", "writing", "sitting"]),
        ("handle", "opening", ["cutting", "writing", "sitting"]),
    ],
    "material_binding": [
        ("window", "glass", ["wood", "paper", "rubber"]),
        ("bottle", "glass", ["wood", "paper", "rubber"]),
        ("mirror", "glass", ["wood", "paper", "rubber"]),
        ("table", "wood", ["glass", "paper", "rubber"]),
        ("shelf", "wood", ["glass", "paper", "rubber"]),
        ("cabinet", "wood", ["glass", "paper", "rubber"]),
        ("notebook", "paper", ["glass", "wood", "rubber"]),
        ("newspaper", "paper", ["glass", "wood", "rubber"]),
        ("envelope", "paper", ["glass", "wood", "rubber"]),
        ("tire", "rubber", ["glass", "wood", "paper"]),
        ("eraser", "rubber", ["glass", "wood", "paper"]),
        ("gasket", "rubber", ["glass", "wood", "paper"]),
    ],
}


def knowledge_prompt(mechanism: str, obj: str, target: str, replica: int) -> tuple[str, str]:
    relation = {
        "category_binding": ("category", "kind"),
        "color_binding": ("color", "color"),
        "function_binding": ("purpose", "used for"),
        "material_binding": ("material", "made from"),
    }[mechanism]
    label, query = relation
    if replica == 0:
        prompt = f"Record: {obj} has {label} {target}. Query: The {label} of {obj} is ___. Answer:"
    elif replica == 1:
        prompt = f"Stored fact - {obj}: {label} = {target}. Fill the missing {label} for {obj}. Response:"
    else:
        prompt = f"Use this one record only. For {obj}, the recorded {label} is {target}. What {label} was recorded for {obj}? Reply:"
    return prompt, query


def knowledge_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, items in KNOWLEDGE.items():
        groups: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        for obj, target, distractors in items:
            groups[target].append((obj, distractors))
        for target, target_items in groups.items():
            for replica, (obj, distractors) in enumerate(target_items):
                prompt, query = knowledge_prompt(mechanism, obj, target, replica)
                rows.append(case_row("content_knowledge", mechanism, target, replica, prompt, target, query, target, distractors, "independent_multi_domain", obj))
    return rows


SYNTAX_GROUPS: dict[str, list[tuple[str, str, str, list[str]]]] = {
    "role_binding": [
        ("The pilot guided the sailor", "subject", "pilot", ["sailor", "guided"]),
        ("The nurse greeted the artist", "subject", "nurse", ["artist", "greeted"]),
        ("The farmer thanked the baker", "subject", "farmer", ["baker", "thanked"]),
        ("The pilot guided the sailor", "object", "sailor", ["pilot", "guided"]),
        ("The nurse greeted the artist", "object", "artist", ["nurse", "greeted"]),
        ("The farmer thanked the baker", "object", "baker", ["farmer", "thanked"]),
        ("The singer followed the dancer", "subject", "singer", ["dancer", "followed"]),
        ("The driver called the mechanic", "subject", "driver", ["mechanic", "called"]),
        ("The teacher praised the student", "subject", "teacher", ["student", "praised"]),
        ("The singer followed the dancer", "object", "dancer", ["singer", "followed"]),
        ("The driver called the mechanic", "object", "mechanic", ["driver", "called"]),
        ("The teacher praised the student", "object", "student", ["teacher", "praised"]),
    ],
    "number_agreement": [
        ("The dog near the trees", "singular", "runs", ["run", "running"]),
        ("The chef with the bowls", "singular", "cooks", ["cook", "cooking"]),
        ("The bird beside the windows", "singular", "sings", ["sing", "singing"]),
        ("The dogs near the tree", "plural", "run", ["runs", "running"]),
        ("The chefs with the bowl", "plural", "cook", ["cooks", "cooking"]),
        ("The birds beside the window", "plural", "sing", ["sings", "singing"]),
        ("The child among the adults", "singular", "writes", ["write", "writing"]),
        ("The robot near the boxes", "singular", "moves", ["move", "moving"]),
        ("The horse behind the carts", "singular", "walks", ["walk", "walking"]),
        ("The children among the adult", "plural", "write", ["writes", "writing"]),
        ("The robots near the box", "plural", "move", ["moves", "moving"]),
        ("The horses behind the cart", "plural", "walk", ["walks", "walking"]),
    ],
    "pronoun_number": [
        ("The lantern was bright", "singular", "it", ["they", "them"]),
        ("The engine was loud", "singular", "it", ["they", "them"]),
        ("The window was open", "singular", "it", ["they", "them"]),
        ("The books were heavy", "plural", "they", ["it", "its"]),
        ("The workers were ready", "plural", "they", ["it", "its"]),
        ("The keys were missing", "plural", "they", ["it", "its"]),
        ("The camera was new", "singular", "it", ["they", "them"]),
        ("The bridge was narrow", "singular", "it", ["they", "them"]),
        ("The package was sealed", "singular", "it", ["they", "them"]),
        ("The lamps were dim", "plural", "they", ["it", "its"]),
        ("The players were tired", "plural", "they", ["it", "its"]),
        ("The doors were locked", "plural", "they", ["it", "its"]),
    ],
    "tense_selection": [
        ("Today the dog", "present", "walks", ["walked", "walk"]),
        ("Today the chef", "present", "cooks", ["cooked", "cook"]),
        ("Today the robot", "present", "moves", ["moved", "move"]),
        ("Yesterday the dog", "past", "walked", ["walks", "walk"]),
        ("Yesterday the chef", "past", "cooked", ["cooks", "cook"]),
        ("Yesterday the robot", "past", "moved", ["moves", "move"]),
        ("Today the child", "present", "writes", ["wrote", "write"]),
        ("Today the bird", "present", "sings", ["sang", "sing"]),
        ("Today the driver", "present", "calls", ["called", "call"]),
        ("Yesterday the child", "past", "wrote", ["writes", "write"]),
        ("Yesterday the bird", "past", "sang", ["sings", "sing"]),
        ("Yesterday the driver", "past", "called", ["calls", "call"]),
    ],
}


def syntax_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, items in SYNTAX_GROUPS.items():
        by_group: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
        for text, group, target, distractors in items:
            by_group[group].append((text, target, distractors))
        for group, group_items in by_group.items():
            if len(group_items) != 6:
                raise AssertionError(f"{mechanism}:{group} expected six cases")
            for local_index, (text, target, distractors) in enumerate(group_items):
                replica = local_index // 2
                group_track = f"{group}_{local_index % 2}"
                if mechanism == "role_binding":
                    if replica == 0:
                        prompt = f"Sentence: {text}. Identify its grammatical {group}. Answer:"
                    elif replica == 1:
                        prompt = f"Read '{text}'. Which word fills the {group} role? Response:"
                    else:
                        prompt = f"Consider the clause '{text}'. Name the noun acting as the {group}. Reply:"
                    source = target
                    query = group
                elif mechanism == "number_agreement":
                    prompts = [
                        f"Complete with the agreeing verb: {text} ___. Answer:",
                        f"Choose the verb form that matches the subject: {text} ___. Response:",
                        f"Finish this clause using subject-verb agreement: {text} ___. Reply:",
                    ]
                    prompt, source, query = prompts[replica], text.split()[1], "agree"
                elif mechanism == "pronoun_number":
                    prompts = [
                        f"Complete with a matching pronoun: {text}, so ___. Answer:",
                        f"Select a pronoun with the same number: {text}; ___. Response:",
                        f"Continue the sentence with a number-agreeing pronoun: {text}. ___. Reply:",
                    ]
                    prompt, source, query = prompts[replica], text.split()[1], "pronoun"
                else:
                    prompts = [
                        f"Complete with the correct tense: {text} ___. Answer:",
                        f"Use the time cue to choose a verb form: {text} ___. Response:",
                        f"Finish the clause in the tense required by the first word: {text} ___. Reply:",
                    ]
                    prompt, source, query = prompts[replica], text.split()[0], "tense"
                rows.append(case_row("syntax_structure", mechanism, group_track, replica, prompt, source, query, target, distractors, "independent_syntax", text))
    return rows


REASONING_SETS: dict[str, list[tuple[str, str, str, str]]] = {
    "direct_validity": [
        ("Every red object is warm. The box is red.", "Is the box warm?", "yes", "warm"),
        ("Every metal object conducts. The rod is metal.", "Does the rod conduct?", "yes", "conducts"),
        ("Every bird has wings. The robin is a bird.", "Does the robin have wings?", "yes", "wings"),
        ("No red object is cold. The box is red.", "Is the box cold?", "no", "cold"),
        ("No wooden object conducts. The rod is wooden.", "Does the rod conduct?", "no", "conducts"),
        ("No fish has feathers. The salmon is a fish.", "Does the salmon have feathers?", "no", "feathers"),
        ("Every square has corners. The tile is square.", "Does the tile have corners?", "yes", "corners"),
        ("Every key opens a lock. This item is a key.", "Can this item open a lock?", "yes", "opens"),
        ("Every sealed parcel is protected. This parcel is sealed.", "Is this parcel protected?", "yes", "protected"),
        ("No circle has corners. The tile is circular.", "Does the tile have corners?", "no", "corners"),
        ("No sealed door is open. This door is sealed.", "Is this door open?", "no", "open"),
        ("No frozen liquid is warm. This liquid is frozen.", "Is this liquid warm?", "no", "warm"),
    ],
    "two_hop_validity": [
        ("All robins are birds. All birds have wings. Ria is a robin.", "Does Ria have wings?", "yes", "wings"),
        ("All copper items are metal. All metal items conduct. The rod is copper.", "Does the rod conduct?", "yes", "conduct"),
        ("All roses are flowers. All flowers are plants. Mira has a rose.", "Is Mira's rose a plant?", "yes", "plants"),
        ("All robins are birds. All mammals are warm. Ria is a robin.", "Must Ria be warm?", "no", "warm"),
        ("All copper items are metal. All plants grow. The rod is copper.", "Must the rod grow?", "no", "grow"),
        ("All roses are flowers. All tools are useful. Mira has a rose.", "Must Mira's rose be useful?", "no", "useful"),
        ("All squares are polygons. All polygons have edges. The tile is square.", "Does the tile have edges?", "yes", "edges"),
        ("All keys are tools. All tools are objects. This item is a key.", "Is this item an object?", "yes", "objects"),
        ("All sparrows are birds. All birds breathe. Pip is a sparrow.", "Does Pip breathe?", "yes", "breathe"),
        ("All squares are polygons. All animals breathe. The tile is square.", "Must the tile breathe?", "no", "breathe"),
        ("All keys are tools. All birds fly. This item is a key.", "Must this item fly?", "no", "fly"),
        ("All cups are containers. All engines rotate. This item is a cup.", "Must this item rotate?", "no", "rotate"),
    ],
    "transitive_validity": [
        ("A is taller than B. B is taller than C.", "Is A taller than C?", "yes", "taller"),
        ("D is older than E. E is older than F.", "Is D older than F?", "yes", "older"),
        ("G is heavier than H. H is heavier than I.", "Is G heavier than I?", "yes", "heavier"),
        ("A is taller than B. B is taller than C.", "Is C taller than A?", "no", "taller"),
        ("D is older than E. E is older than F.", "Is F older than D?", "no", "older"),
        ("G is heavier than H. H is heavier than I.", "Is I heavier than G?", "no", "heavier"),
        ("J is faster than K. K is faster than L.", "Is J faster than L?", "yes", "faster"),
        ("M is brighter than N. N is brighter than O.", "Is M brighter than O?", "yes", "brighter"),
        ("P is wider than Q. Q is wider than R.", "Is P wider than R?", "yes", "wider"),
        ("J is faster than K. K is faster than L.", "Is L faster than J?", "no", "faster"),
        ("M is brighter than N. N is brighter than O.", "Is O brighter than M?", "no", "brighter"),
        ("P is wider than Q. Q is wider than R.", "Is R wider than P?", "no", "wider"),
    ],
    "conjunction_validity": [
        ("If an item is red and round, it is marked. The token is red and round.", "Is the token marked?", "yes", "round"),
        ("If a person is trained and ready, they can enter. Ana is trained and ready.", "Can Ana enter?", "yes", "ready"),
        ("If a device is charged and connected, it works. The phone is charged and connected.", "Does the phone work?", "yes", "connected"),
        ("If an item is red and round, it is marked. The token is red but not round.", "Must the token be marked?", "no", "round"),
        ("If a person is trained and ready, they can enter. Ana is trained but not ready.", "Can Ana enter by this rule?", "no", "ready"),
        ("If a device is charged and connected, it works. The phone is charged but disconnected.", "Must the phone work?", "no", "connected"),
        ("If a door is closed and locked, it is secure. The door is closed and locked.", "Is the door secure?", "yes", "locked"),
        ("If a signal is strong and stable, it is accepted. The signal is strong and stable.", "Is the signal accepted?", "yes", "stable"),
        ("If a card is signed and stamped, it is valid. The card is signed and stamped.", "Is the card valid?", "yes", "stamped"),
        ("If a door is closed and locked, it is secure. The door is closed but unlocked.", "Must the door be secure?", "no", "locked"),
        ("If a signal is strong and stable, it is accepted. The signal is strong but unstable.", "Must the signal be accepted?", "no", "stable"),
        ("If a card is signed and stamped, it is valid. The card is signed but unstamped.", "Must the card be valid?", "no", "stamped"),
    ],
}


def reasoning_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, items in REASONING_SETS.items():
        grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for facts, question, target, source in items:
            grouped[target].append((facts, question, source))
        for target, target_items in grouped.items():
            if len(target_items) != 6:
                raise AssertionError(f"{mechanism}:{target} expected six cases")
            for local_index, (facts, question, source) in enumerate(target_items):
                replica = local_index // 2
                variant = local_index % 2
                if replica == 0:
                    prompt = f"Facts and rule: {facts} Question: {question} Answer yes or no. Answer:"
                elif replica == 1:
                    prompt = f"Given only these statements: {facts} Decide the query '{question}' with yes or no. Response:"
                else:
                    prompt = f"Treat the following as a temporary world. {facts} Based solely on it, {question} Reply yes or no:"
                rows.append(case_row("reasoning_constraint", mechanism, f"{target}_{variant}", replica, prompt, source, question.split()[0], target, ["no" if target == "yes" else "yes"], "independent_rules", f"{target}:{variant}"))
    return rows


def build_case_bank() -> list[dict[str, Any]]:
    rows = knowledge_cases() + syntax_cases() + reasoning_cases()
    expected = 3 * 4 * 12
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} cases, got {len(rows)}")
    return rows


def target_label(row: dict[str, Any]) -> str:
    group = str(row["target_group"])
    prefix, separator, suffix = group.rpartition("_")
    return prefix if separator and suffix.isdigit() else group


def build_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_key[(str(row["family_id"]), str(row["mechanism_id"]), str(row["split"]))].append(row)
    pairs: list[dict[str, Any]] = []
    for (family, mechanism, split), rows in sorted(by_key.items()):
        ordered = sorted(rows, key=lambda r: (target_label(r), str(r["case_id"])))
        for recipient in ordered:
            different = [r for r in ordered if target_label(r) != target_label(recipient)]
            if not different:
                raise AssertionError(f"no counterfactual donor for {recipient['case_id']}")
            donor = different[int(digest(recipient["case_id"]), 16) % len(different)]
            same_target_pool = [r for r in cases if r["family_id"] == family and r["mechanism_id"] == mechanism and target_label(r) == target_label(recipient) and r["case_id"] != recipient["case_id"]]
            same_target = sorted(same_target_pool, key=lambda r: str(r["case_id"]))[0]
            unrelated_pool = [r for r in cases if r["family_id"] == family and r["mechanism_id"] != mechanism and r["split"] == split]
            unrelated = sorted(unrelated_pool, key=lambda r: str(r["case_id"]))[int(digest([recipient["case_id"], "unrelated"]), 16) % len(unrelated_pool)]
            pairs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "pair_id": f"phase317:pair:{family}:{mechanism}:{split}:{len(pairs):03d}",
                    "family_id": family,
                    "mechanism_id": mechanism,
                    "split": split,
                    "recipient_case_id": recipient["case_id"],
                    "donor_case_id": donor["case_id"],
                    "same_target_control_case_id": same_target["case_id"],
                    "unrelated_control_case_id": unrelated["case_id"],
                    "recipient_target": recipient["target"],
                    "donor_target": donor["target"],
                    "targets_differ": recipient["target"] != donor["target"],
                    "template_id": recipient["template_id"],
                    "independent_pair": True,
                }
            )
    expected = 3 * 4 * 12
    if len(pairs) != expected:
        raise AssertionError(f"expected {expected} pairs, got {len(pairs)}")
    return pairs


def prepare() -> dict[str, Any]:
    cases = build_case_bank()
    pairs = build_pairs(cases)
    model_plan = [{**pair, "model": model, "model_pair_id": f"{pair['pair_id']}:{model}"} for model in MODELS for pair in pairs]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "prepared",
        "scientific_denominator": {
            "base_independent_cases": len(cases),
            "base_independent_pairs": len(pairs),
            "planned_model_pairs": len(model_plan),
            "family_count": len({r["family_id"] for r in cases}),
            "mechanisms_per_family": 4,
            "cases_per_mechanism": 12,
            "templates_per_mechanism": 3,
        },
        "family_counts": dict(Counter(str(r["family_id"]) for r in cases)),
        "split_counts": dict(Counter(str(r["split"]) for r in cases)),
        "pair_split_counts": dict(Counter(str(r["split"]) for r in pairs)),
        "models": MODELS,
        "case_bank_hash": digest(cases),
        "pair_bank_hash": digest(pairs),
        "controls": ["same_target_natural_state", "unrelated_mechanism_natural_state", "wrong_position", "feature_permutation"],
        "heldout_policy": "template_c_open and heldout objects/rules are not used for layer or component selection",
    }
    for base in [V2, LEGACY_V2]:
        write_jsonl(base / "phase317_natural_source_case_bank.jsonl", cases)
        write_jsonl(base / "phase317_natural_source_pair_bank.jsonl", pairs)
        write_jsonl(base / "phase317_natural_source_model_plan_rows.jsonl", model_plan)
        write_json(base / "phase317_natural_source_case_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    prepare()
