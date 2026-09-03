#!/usr/bin/env python3
"""Freeze a broad natural-continuation and active full-coordinate response campaign."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2314 = RESULT / "phase2314_c4961_c5040_multistep_atlas_cleanup"
OUT = RESULT / "phase2315_c5041_c5100_active_response_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/natural_active_response_bilingual.jsonl"
CONFIG = OUT / "config/frozen_campaign.json"
sys.path.insert(0, str(TESTS))

import model_utils  # noqa: E402


PHASE = 2315
CAMPAIGN = "C5041-C5100"
FAMILIES = (
    "agent_patient",
    "attitude_event",
    "comparison_order",
    "location_binding",
    "possession_query",
    "relative_binding",
    "temporal_order",
    "taxonomy_chain",
)
LANGUAGES = ("en", "zh")
SURFACES = ("narrative", "reported")
PARTITIONS = ("discovery", "confirmation", "fresh_confirmation", "fresh_lockbox")
UNITS = 32
PARTITION_BY_UNIT = {
    **{value: "discovery" for value in range(0, 8)},
    **{value: "confirmation" for value in range(8, 16)},
    **{value: "fresh_confirmation" for value in range(16, 24)},
    **{value: "fresh_lockbox" for value in range(24, 32)},
}
QPOINTS_4B = tuple(range(38))
SOURCE_QPOINTS_4B = (10, 20, 30)
TARGET_OFFSETS = (1, 4, "final_norm")
BASE_PROBES = 8
PAIR_PROBES = ((0, 1), (2, 3), (4, 5), (6, 7))
PERTURBATION_DOSE = 0.01
SEQUENCE_GATE = 0.75
FREE_IDENTITY_GATE = 0.50
ACTIVE_ROWS_PER_PARTITION_FAMILY = 4
MODEL_ORDER = ("qwen3", "qwen3_14b", "glm4", "deepseek7b")


EN_FIRST = (
    "Mara", "Elias", "Nora", "Jonas", "Iris", "Caleb", "Talia", "Damon",
    "Selene", "Owen", "Priya", "Felix", "Ada", "Milo", "Rhea", "Simon",
    "Lena", "Victor", "Mina", "Julian", "Clara", "Theo", "Anya", "Roman",
    "Elena", "Hugo", "Sofia", "Arun", "Celia", "Noel", "Daria", "Isaac",
)
EN_LAST = (
    "Vale", "Rowan", "Meyer", "Chen", "Silva", "Hart", "Klein", "Patel",
    "Stone", "Reed", "Park", "Cole", "Arden", "Bell", "Dunn", "Frost",
    "Gray", "Hayes", "Iqbal", "James", "King", "Lane", "Moore", "Nash",
    "Ortiz", "Price", "Quinn", "Ross", "Shaw", "Tran", "Usher", "Wells",
)
EN_OBJECTS = (
    "atlas", "notebook", "camera", "lantern", "violin", "compass", "vase", "ledger",
    "telescope", "satchel", "tablet", "hammer", "scarf", "goblet", "key", "parcel",
    "sketchbook", "drum", "microscope", "jacket", "ring", "basket", "map", "clock",
    "book", "helmet", "pencil", "radio", "bottle", "ticket", "ribbon", "folder",
)
EN_LOCATIONS = (
    "archive", "junction", "gallery", "workshop", "library", "station", "courtyard", "studio",
    "cellar", "balcony", "terminal", "museum", "hallway", "market", "garden", "laboratory",
    "office", "theater", "harbor", "depot", "kitchen", "attic", "lobby", "warehouse",
    "chapel", "observatory", "garage", "classroom", "clinic", "bakery", "pavilion", "dock",
)
EN_ATTRIBUTES = (
    "calm", "careful", "patient", "agile", "precise", "curious", "steady", "quiet",
    "alert", "generous", "formal", "cheerful", "focused", "punctual", "gentle", "brisk",
)
EN_PARENT = (
    "reference item", "display item", "measuring item", "travel item", "music item", "writing item",
    "household item", "record item", "optical item", "carrying item", "signal item", "craft item",
    "clothing item", "ceremonial item", "access item", "shipping item",
)
EN_TOP = (
    "archive material", "exhibit material", "technical material", "field equipment",
    "performance equipment", "office material", "domestic material", "administrative material",
    "scientific equipment", "personal equipment", "communication equipment", "workshop equipment",
    "wearable material", "collection material", "security equipment", "delivery material",
)

ZH_SURNAMES = tuple("\u8d75\u94b1\u5b59\u674e\u5468\u5434\u90d1\u738b\u51af\u9648\u891a\u536b\u848b\u6c88\u97e9\u6768")
ZH_GIVEN = (
    "\u5b89\u5b81", "\u5b5f\u7136", "\u767d\u5ddd", "\u6b27\u9633", "\u9648\u66e6", "\u5b50\u8c26", "\u96e8\u6850", "\u6d69\u5b87",
    "\u82e5\u5b81", "\u6587\u8f69", "\u601d\u8fdc", "\u6e05\u548c", "\u660e\u73e0", "\u666f\u884c", "\u8212\u7136", "\u51cc\u4e91",
    "\u661f\u6cb3", "\u5b81\u590f", "\u4e66\u8a00", "\u5b89\u6b4c", "\u4ea6\u8fb0", "\u5b50\u58a8", "\u65b0\u6708", "\u9752\u5c9a",
    "\u9e64\u8f69", "\u60a0\u7136", "\u4f73\u97f3", "\u9510\u6cfd", "\u5e73\u5b89", "\u77e5\u8fdc", "\u4e91\u8212", "\u542f\u660e",
)
ZH_OBJECTS = (
    "\u5730\u56fe\u518c", "\u7b14\u8bb0\u672c", "\u76f8\u673a", "\u63d0\u706f", "\u5c0f\u63d0\u7434", "\u7f57\u76d8", "\u82b1\u74f6", "\u8d26\u672c",
    "\u671b\u8fdc\u955c", "\u80cc\u5305", "\u5e73\u677f", "\u94c1\u9524", "\u56f4\u5dfe", "\u9ad8\u811a\u676f", "\u94a5\u5319", "\u5305\u88f9",
    "\u901f\u5199\u672c", "\u624b\u9f13", "\u663e\u5fae\u955c", "\u5939\u514b", "\u6212\u6307", "\u7bee\u5b50", "\u5730\u56fe", "\u65f6\u949f",
    "\u4e66\u7c4d", "\u5934\u76d4", "\u94c5\u7b14", "\u6536\u97f3\u673a", "\u6c34\u74f6", "\u7968\u636e", "\u4e1d\u5e26", "\u6587\u4ef6\u5939",
)
ZH_LOCATIONS = (
    "\u6863\u6848\u5ba4", "\u4ea4\u6c47\u5904", "\u753b\u5eca", "\u5de5\u574a", "\u56fe\u4e66\u9986", "\u8f66\u7ad9", "\u5ead\u9662", "\u5de5\u4f5c\u5ba4",
    "\u5730\u7a96", "\u9633\u53f0", "\u7ec8\u7aef\u5ba4", "\u535a\u7269\u9986", "\u8d70\u5eca", "\u5e02\u573a", "\u82b1\u56ed", "\u5b9e\u9a8c\u5ba4",
    "\u529e\u516c\u5ba4", "\u5267\u9662", "\u6e2f\u53e3", "\u8d27\u573a", "\u53a8\u623f", "\u9601\u697c", "\u5927\u5385", "\u4ed3\u5e93",
    "\u793c\u5802", "\u5929\u6587\u53f0", "\u8f66\u5e93", "\u6559\u5ba4", "\u8bca\u6240", "\u9762\u5305\u623f", "\u4ead\u5b50", "\u7801\u5934",
)
ZH_ATTRIBUTES = (
    "\u6c89\u7740", "\u4ed4\u7ec6", "\u8010\u5fc3", "\u654f\u6377", "\u7cbe\u51c6", "\u597d\u5947", "\u7a33\u91cd", "\u5b89\u9759",
    "\u8b66\u89c9", "\u6177\u6168", "\u6b63\u5f0f", "\u5f00\u6717", "\u4e13\u6ce8", "\u5b88\u65f6", "\u6e29\u548c", "\u5229\u843d",
)
ZH_PARENT = (
    "\u53c2\u8003\u7269\u54c1", "\u5c55\u793a\u7269\u54c1", "\u6d4b\u91cf\u7269\u54c1", "\u65c5\u884c\u7269\u54c1", "\u97f3\u4e50\u7269\u54c1", "\u4e66\u5199\u7269\u54c1",
    "\u5bb6\u5c45\u7269\u54c1", "\u8bb0\u5f55\u7269\u54c1", "\u5149\u5b66\u7269\u54c1", "\u643a\u5e26\u7269\u54c1", "\u4fe1\u53f7\u7269\u54c1", "\u5de5\u827a\u7269\u54c1",
    "\u670d\u9970\u7269\u54c1", "\u793c\u4eea\u7269\u54c1", "\u901a\u884c\u7269\u54c1", "\u8fd0\u8f93\u7269\u54c1",
)
ZH_TOP = (
    "\u6863\u6848\u6750\u6599", "\u5c55\u9648\u6750\u6599", "\u6280\u672f\u6750\u6599", "\u91ce\u5916\u88c5\u5907", "\u6f14\u51fa\u88c5\u5907", "\u529e\u516c\u6750\u6599",
    "\u751f\u6d3b\u6750\u6599", "\u7ba1\u7406\u6750\u6599", "\u79d1\u7814\u88c5\u5907", "\u4e2a\u4eba\u88c5\u5907", "\u901a\u4fe1\u88c5\u5907", "\u5de5\u574a\u88c5\u5907",
    "\u7a7f\u6234\u6750\u6599", "\u6536\u85cf\u6750\u6599", "\u5b89\u5168\u88c5\u5907", "\u914d\u9001\u6750\u6599",
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def answer_ids(tokenizer, text: str, language: str) -> list[int]:
    values = tokenizer.encode((" " + text) if language == "en" else text, add_special_tokens=False)
    if not values:
        raise RuntimeError(("empty_answer", text, language))
    return [int(value) for value in values]


def all_subsequence_positions(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not subsequence:
        return []
    return [index for index in range(len(sequence) - len(subsequence) + 1)
            if sequence[index:index + len(subsequence)] == subsequence]


def role_token_positions(tokenizer, prompt_ids: list[int], text: str, language: str) -> list[int]:
    variants = [text]
    if language == "en":
        variants.extend((" " + text, '"' + text, ' "' + text))
    positions: set[int] = set()
    for variant in variants:
        token_ids = [int(value) for value in tokenizer.encode(variant, add_special_tokens=False)]
        for start in all_subsequence_positions(prompt_ids, token_ids):
            positions.update(range(start, start + len(token_ids)))
    return sorted(positions)


def person(language: str, index: int) -> str:
    if language == "en":
        return f"{EN_FIRST[index % 32]} {EN_LAST[(index * 7 + index // 4) % 32]}"
    return ZH_SURNAMES[index % len(ZH_SURNAMES)] + ZH_GIVEN[(index * 5 + index // 3) % 32]


def lexicon(language: str, unit: int) -> dict[str, str]:
    if language == "en":
        return {
            "a": person(language, unit), "b": person(language, unit + 41),
            "holder": person(language, unit + 73),
            "object_a": EN_OBJECTS[unit], "object_b": EN_OBJECTS[(unit + 13) % 32],
            "location_a": EN_LOCATIONS[unit], "location_b": EN_LOCATIONS[(unit + 17) % 32],
            "attribute": EN_ATTRIBUTES[unit % 16],
            "parent_a": EN_PARENT[unit % 16], "parent_b": EN_PARENT[(unit + 9) % 16],
            "top_a": EN_TOP[unit % 16], "top_b": EN_TOP[(unit + 9) % 16],
        }
    return {
        "a": person(language, unit), "b": person(language, unit + 41),
        "holder": person(language, unit + 73),
        "object_a": ZH_OBJECTS[unit], "object_b": ZH_OBJECTS[(unit + 13) % 32],
        "location_a": ZH_LOCATIONS[unit], "location_b": ZH_LOCATIONS[(unit + 17) % 32],
        "attribute": ZH_ATTRIBUTES[unit % 16],
        "parent_a": ZH_PARENT[unit % 16], "parent_b": ZH_PARENT[(unit + 9) % 16],
        "top_a": ZH_TOP[unit % 16], "top_b": ZH_TOP[(unit + 9) % 16],
    }


def cue(language: str, unit: int) -> tuple[str, str]:
    if language == "en":
        values = (
            " A later line records that",
            " The same facts are continued by the sentence",
            " The record then states that",
            " A final factual line says that",
        )
    else:
        values = (
            "\u540e\u7eed\u8bb0\u5f55\u5199\u9053\uff1a",
            "\u540c\u4e00\u4e8b\u5b9e\u5728\u4e0b\u4e00\u53e5\u4e2d\u8868\u8ff0\u4e3a\uff1a",
            "\u8bb0\u5f55\u968f\u540e\u8bf4\u660e\uff1a",
            "\u6700\u540e\u4e00\u884c\u4e8b\u5b9e\u5199\u9053\uff1a",
        )
    return values[unit % 4], f"cue_{unit % 4}"


def reported(language: str, surface: str, facts: str) -> str:
    if surface == "narrative":
        return facts
    if language == "en":
        return f'A witness reported, "{facts}"'
    return f"\u4e00\u4f4d\u8bb0\u5f55\u5458\u8bf4\uff1a\u201c{facts}\u201d"


def compile_semantics(family: str, language: str, unit: int, state: int, surface: str) -> dict:
    v = lexicon(language, unit)
    a, b, obj, obj2 = v["a"], v["b"], v["object_a"], v["object_b"]
    target_first = (unit % 2 == 0)
    if family in ("agent_patient", "attitude_event", "comparison_order", "relative_binding",
                  "temporal_order"):
        target, wrong = ((a, b) if state == 0 else (b, a))
    elif family == "location_binding":
        target, wrong = ((v["location_a"], v["location_b"]) if state == 0
                         else (v["location_b"], v["location_a"]))
    elif family == "possession_query":
        target, wrong = ((obj, obj2) if state == 0 else (obj2, obj))
    elif family == "taxonomy_chain":
        target, wrong = ((v["top_a"], v["top_b"]) if state == 0 else (v["top_b"], v["top_a"]))
    else:
        raise KeyError(family)

    if language == "en":
        if family == "agent_patient":
            facts = f"{target} handed the {obj} to {wrong}."
            future = f"{target} was the person who handed over the {obj}."
            false = f"{wrong} was the person who handed over the {obj}."
            roles = {"primary": target, "secondary": wrong, "context": obj, "relation": "handed"}
            graph = [[target, "giver", obj], [obj, "recipient", wrong]]
        elif family == "attitude_event":
            clauses = [f"{target} ate the {obj}", f"{wrong} ate the {obj2}"]
            if not target_first:
                clauses.reverse()
            facts = f"{v['holder']} likes the fact that {clauses[0]}; {clauses[1]}."
            future = f"{target} was the person who ate the {obj}."
            false = f"{wrong} was the person who ate the {obj}."
            roles = {"primary": target, "secondary": wrong, "context": obj, "relation": "likes",
                     "holder": v["holder"]}
            graph = [[v["holder"], "attitude", "like"], [target, "ate", obj], [wrong, "ate", obj2]]
        elif family == "comparison_order":
            facts = f"{target} was more {v['attribute']} than {wrong}."
            future = f"{target} was the more {v['attribute']} person."
            false = f"{wrong} was the more {v['attribute']} person."
            roles = {"primary": target, "secondary": wrong, "context": v["attribute"], "relation": "more"}
            graph = [[target, "more", v["attribute"]], [wrong, "less", v["attribute"]]]
        elif family == "location_binding":
            facts = f"The {obj} was in the {target} and not in the {wrong}."
            future = f"The {obj} was located in the {target}."
            false = f"The {obj} was located in the {wrong}."
            roles = {"primary": obj, "secondary": target, "context": wrong, "relation": "located in"}
            graph = [[obj, "located_in", target], [obj, "not_in", wrong]]
        elif family == "possession_query":
            clauses = [f"{a} owned the {target}", f"{b} owned the {wrong}"]
            if not target_first:
                clauses.reverse()
            facts = f"{clauses[0]}; {clauses[1]}."
            future = f"{a} owned the {target}."
            false = f"{a} owned the {wrong}."
            roles = {"primary": a, "secondary": b, "context": target, "distractor": wrong, "relation": "owned"}
            graph = [[a, "owns", target], [b, "owns", wrong]]
        elif family == "relative_binding":
            facts = f"The editor who thanked {target} later interviewed {wrong}."
            future = f"The editor thanked {target}."
            false = f"The editor thanked {wrong}."
            roles = {"primary": target, "secondary": wrong, "context": "editor", "relation": "thanked"}
            graph = [["editor", "thanked", target], ["editor", "interviewed", wrong]]
        elif family == "temporal_order":
            facts = f"{target} arrived before {wrong}."
            future = f"{target} arrived before {wrong}."
            false = f"{wrong} arrived before {target}."
            roles = {"primary": target, "secondary": wrong, "context": "arrival", "relation": "before"}
            graph = [[target, "before", wrong]]
        else:
            parent = v["parent_a"] if state == 0 else v["parent_b"]
            clauses = [f"every {obj} is a {parent}", f"every {parent} is {target}",
                       f"no {parent} is {wrong}"]
            if not target_first:
                clauses = [clauses[2], clauses[0], clauses[1]]
            facts = "Within this closed catalog, " + "; ".join(clauses) + "."
            future = f"The {obj} belongs to the {target} category."
            false = f"The {obj} belongs to the {wrong} category."
            roles = {"primary": obj, "secondary": parent, "context": target, "distractor": wrong,
                     "relation": "is a"}
            graph = [[obj, "is_a", parent], [parent, "is_a", target], [parent, "not_is_a", wrong]]
    else:
        if family == "agent_patient":
            facts = f"{target}\u628a{obj}\u4ea4\u7ed9\u4e86{wrong}\u3002"
            future = f"{target}\u662f\u4ea4\u51fa{obj}\u7684\u4eba\u3002"
            false = f"{wrong}\u662f\u4ea4\u51fa{obj}\u7684\u4eba\u3002"
            roles = {"primary": target, "secondary": wrong, "context": obj, "relation": "\u4ea4\u7ed9"}
            graph = [[target, "giver", obj], [obj, "recipient", wrong]]
        elif family == "attitude_event":
            clauses = [f"{target}\u5403\u4e86{obj}", f"{wrong}\u5403\u4e86{obj2}"]
            if not target_first:
                clauses.reverse()
            facts = f"{v['holder']}\u559c\u6b22\u201c{clauses[0]}\u201d\u8fd9\u4ef6\u4e8b\uff1b{clauses[1]}\u3002"
            future = f"{target}\u662f\u5403\u4e0b{obj}\u7684\u4eba\u3002"
            false = f"{wrong}\u662f\u5403\u4e0b{obj}\u7684\u4eba\u3002"
            roles = {"primary": target, "secondary": wrong, "context": obj, "relation": "\u559c\u6b22",
                     "holder": v["holder"]}
            graph = [[v["holder"], "attitude", "like"], [target, "ate", obj], [wrong, "ate", obj2]]
        elif family == "comparison_order":
            facts = f"{target}\u6bd4{wrong}\u66f4{v['attribute']}\u3002"
            future = f"{target}\u662f\u66f4{v['attribute']}\u7684\u4eba\u3002"
            false = f"{wrong}\u662f\u66f4{v['attribute']}\u7684\u4eba\u3002"
            roles = {"primary": target, "secondary": wrong, "context": v["attribute"], "relation": "\u66f4"}
            graph = [[target, "more", v["attribute"]], [wrong, "less", v["attribute"]]]
        elif family == "location_binding":
            facts = f"{obj}\u5728{target}\uff0c\u4e0d\u5728{wrong}\u3002"
            future = f"{obj}\u4f4d\u4e8e{target}\u3002"
            false = f"{obj}\u4f4d\u4e8e{wrong}\u3002"
            roles = {"primary": obj, "secondary": target, "context": wrong, "relation": "\u4f4d\u4e8e"}
            graph = [[obj, "located_in", target], [obj, "not_in", wrong]]
        elif family == "possession_query":
            clauses = [f"{a}\u62e5\u6709{target}", f"{b}\u62e5\u6709{wrong}"]
            if not target_first:
                clauses.reverse()
            facts = f"{clauses[0]}\uff1b{clauses[1]}\u3002"
            future = f"{a}\u62e5\u6709{target}\u3002"
            false = f"{a}\u62e5\u6709{wrong}\u3002"
            roles = {"primary": a, "secondary": b, "context": target, "distractor": wrong, "relation": "\u62e5\u6709"}
            graph = [[a, "owns", target], [b, "owns", wrong]]
        elif family == "relative_binding":
            facts = f"\u90a3\u4f4d\u611f\u8c22\u8fc7{target}\u7684\u7f16\u8f91\uff0c\u540e\u6765\u91c7\u8bbf\u4e86{wrong}\u3002"
            future = f"\u7f16\u8f91\u611f\u8c22\u8fc7{target}\u3002"
            false = f"\u7f16\u8f91\u611f\u8c22\u8fc7{wrong}\u3002"
            roles = {"primary": target, "secondary": wrong, "context": "\u7f16\u8f91", "relation": "\u611f\u8c22"}
            graph = [["editor", "thanked", target], ["editor", "interviewed", wrong]]
        elif family == "temporal_order":
            facts = f"{target}\u6bd4{wrong}\u5148\u5230\u3002"
            future = f"{target}\u5728{wrong}\u4e4b\u524d\u5230\u8fbe\u3002"
            false = f"{wrong}\u5728{target}\u4e4b\u524d\u5230\u8fbe\u3002"
            roles = {"primary": target, "secondary": wrong, "context": "\u5230\u8fbe", "relation": "\u4e4b\u524d"}
            graph = [[target, "before", wrong]]
        else:
            parent = v["parent_a"] if state == 0 else v["parent_b"]
            clauses = [f"\u6bcf\u4e2a{obj}\u90fd\u662f{parent}", f"\u6bcf\u4e2a{parent}\u90fd\u5c5e\u4e8e{target}",
                       f"\u6ca1\u6709{parent}\u5c5e\u4e8e{wrong}"]
            if not target_first:
                clauses = [clauses[2], clauses[0], clauses[1]]
            facts = "\u5728\u8fd9\u4efd\u5c01\u95ed\u76ee\u5f55\u4e2d\uff0c" + "\uff1b".join(clauses) + "\u3002"
            future = f"{obj}\u5c5e\u4e8e{target}\u7c7b\u522b\u3002"
            false = f"{obj}\u5c5e\u4e8e{wrong}\u7c7b\u522b\u3002"
            roles = {"primary": obj, "secondary": parent, "context": target, "distractor": wrong,
                     "relation": "\u5c5e\u4e8e"}
            graph = [[obj, "is_a", parent], [parent, "is_a", target], [parent, "not_is_a", wrong]]

    prefix_cue, cue_id = cue(language, unit)
    prompt = reported(language, surface, facts) + prefix_cue
    return {
        "future_prompt": prompt,
        "future_target_text": future,
        "future_wrong_text": false,
        "identity_target": target,
        "identity_wrong": wrong,
        "role_values": roles,
        "semantic_graph": graph,
        "target_mention_order": "first" if target_first else "last",
        "cue_id": cue_id,
    }


def compile_rows() -> tuple[list[dict], dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for language in LANGUAGES:
            for surface in SURFACES:
                for unit in range(UNITS):
                    for state in (0, 1):
                        semantics = compile_semantics(family, language, unit, state, surface)
                        prompt_ids = [int(value) for value in tokenizer.encode(
                            semantics["future_prompt"], add_special_tokens=False)]
                        role_positions = {}
                        for key, text in semantics["role_values"].items():
                            role_positions[key] = role_token_positions(
                                tokenizer, prompt_ids, text, language
                            )
                        rows.append({
                            "case_id": f"c5041-{family}-{language}-{surface}-u{unit:02d}-s{state}",
                            "design_index": len(rows), "family": family,
                            "family_index": family_index, "language": language, "surface": surface,
                            "unit": unit, "state": state, "partition": PARTITION_BY_UNIT[unit],
                            **semantics,
                            "future_prompt_ids": prompt_ids,
                            "future_target_ids": answer_ids(tokenizer, semantics["future_target_text"], language),
                            "future_wrong_ids": answer_ids(tokenizer, semantics["future_wrong_text"], language),
                            "identity_target_ids": answer_ids(tokenizer, semantics["identity_target"], language),
                            "identity_wrong_ids": answer_ids(tokenizer, semantics["identity_wrong"], language),
                            "role_positions": role_positions,
                            "boundary_position": len(prompt_ids) - 1,
                            "interface": "raw_declarative_natural_sentence_continuation",
                        })
    rows.sort(key=lambda row: row["design_index"])

    pair_groups: dict[tuple, set[int]] = defaultdict(set)
    cell_counts = Counter()
    shorter_correct = []
    first_correct = []
    collisions = []
    missing_roles = []
    for row in rows:
        pair_groups[(row["family"], row["language"], row["surface"], row["unit"])].add(row["state"])
        cell_counts[(row["family"], row["language"], row["surface"], row["partition"], row["state"])] += 1
        target_length = len(row["future_target_ids"])
        wrong_length = len(row["future_wrong_ids"])
        shorter_correct.append(1.0 if target_length < wrong_length else
                               0.0 if target_length > wrong_length else 0.5)
        first_correct.append(row["target_mention_order"] == "first")
        if row["identity_target"] == row["identity_wrong"]:
            collisions.append(row["case_id"])
        if not row["role_positions"].get("primary"):
            missing_roles.append(row["case_id"])
    state_pairs_complete = all(value == {0, 1} for value in pair_groups.values())
    exactly_balanced_cells = len(set(cell_counts.values())) == 1
    audit = {
        "rows": len(rows), "families": list(FAMILIES),
        "language_counts": dict(Counter(row["language"] for row in rows)),
        "surface_counts": dict(Counter(row["surface"] for row in rows)),
        "partition_counts": dict(Counter(row["partition"] for row in rows)),
        "future_forms": dict(Counter(row["cue_id"] for row in rows)),
        "state_pairs_complete": state_pairs_complete,
        "exactly_balanced_cells": exactly_balanced_cells,
        "identity_collisions": len(collisions), "missing_primary_role_positions": len(missing_roles),
        "zero_model_shorter_candidate_accuracy": sum(shorter_correct) / len(shorter_correct),
        "zero_model_first_mention_accuracy": sum(first_correct) / len(first_correct),
        "zero_model_state_only_accuracy": 0.5,
        "prompt_token_min_max": [min(len(row["future_prompt_ids"]) for row in rows),
                                  max(len(row["future_prompt_ids"]) for row in rows)],
        "future_token_min_max": [min(len(row["future_target_ids"]) for row in rows),
                                  max(len(row["future_target_ids"]) for row in rows)],
        "semantic_uniqueness": (
            "Each state writes one explicit positive relation and an explicit competing fact or negation; "
            "state pairs swap the target while preserving the family, lexicon unit, language, and surface."
        ),
        "machine_naturality": (
            "Eight ordinary declarative constructions use four continuation cues and two paired surfaces; "
            "taxonomy is explicitly marked as a controlled closed catalog."
        ),
        "independent_human_blind_review": "NA_not_run",
    }
    return rows, audit


def frozen_config() -> dict:
    value = {
        "phase": PHASE, "campaign": CAMPAIGN, "frozen_before_model_load": True,
        "source_evidence": [2309, 2310, 2311, 2312, 2313, 2314],
        "research_object": (
            "model-local full-coordinate HiddenState responses to frozen signed structured perturbations, "
            "kept separate from natural continuation behavior and local output gradients"
        ),
        "families": list(FAMILIES), "languages": list(LANGUAGES),
        "surfaces": list(SURFACES), "partitions": list(PARTITIONS), "units": UNITS,
        "sequence_gate": SEQUENCE_GATE, "free_identity_gate": FREE_IDENTITY_GATE,
        "qpoints_qwen4b": list(QPOINTS_4B), "source_qpoints_qwen4b": list(SOURCE_QPOINTS_4B),
        "target_offsets": list(TARGET_OFFSETS), "base_rademacher_probes": BASE_PROBES,
        "pair_probes": [list(value) for value in PAIR_PROBES],
        "perturbation_dose_relative_hidden_l2": PERTURBATION_DOSE,
        "active_rows_per_partition_family": ACTIVE_ROWS_PER_PARTITION_FAMILY,
        "models_sequential": list(MODEL_ORDER),
        "null_models": ["zero_response", "global_probe_response", "family_probe_response",
                        "family_state_probe_response", "family_language_probe_response",
                        "family_surface_probe_response"],
        "observation_policy": "all behavior rows observed; route failures eliminate only mechanism claims",
        "fresh_policy": "directions, source layers, signs, candidate models and thresholds freeze before fresh readout",
        "coordinate_policy": "all original coordinates retained; no top-k, PCA, cosine selection or coordinate reordering",
        "identifiability_boundary": (
            "the intervention identifies directional responses only on a frozen 8-direction span; "
            "it does not identify an arbitrary d_by_d Jacobian"
        ),
        "claim_ladder": ["behavior", "observational_field", "directional_prediction",
                         "frozen_structured_control", "crossmodel_functional_topology"],
        "failure_policy": "route_level_elimination; execute every preregistered independent branch",
        "stop_condition": "all frozen branches attempted and audited; no post-unblinding threshold edits",
    }
    if CONFIG.exists():
        previous = json.loads(CONFIG.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError(("frozen_config_changed", previous, value))
    else:
        save(CONFIG, value)
    return value


def append_memo(result: dict) -> None:
    marker = f"## Phase {PHASE}:"
    existing = MEMO.read_text(encoding="utf-8")
    correction_marker = "### Phase 2315 预模型审计修正"
    if marker in existing and not result.get("all_checks_passed"):
        return
    if marker in existing:
        if correction_marker in existing:
            return
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        correction = f"""

### Phase 2315 预模型审计修正 [{stamp}]

首次编译在模型加载前被审计阻断：英文角色跨度没有覆盖词首空格 tokenizer 变体，导致 `576` 行 primary 角色未定位；候选长度零模型把等长情况记为 `0` 而不是随机决策的 `0.5`，得到伪偏差 `0.169921875`。未加载任何模型、未读取行为或 HiddenState。修复后按原冻结对象重新编译，角色位置改为实际 token 覆盖集合，等长按半分计；最终审计 `{json.dumps(result['audit'], ensure_ascii=False)}`，检查 `{json.dumps(result['checks'], ensure_ascii=False)}`，材料 SHA256 `{result['hashes']['material']}`。全部预模型检查通过后，Phase2316 才获得 Qwen3-4B 运行授权。
"""
        with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(correction)
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 八构式自然续写与全坐标主动响应大合同（{CAMPAIGN}） [{stamp}]

**证据审查与过度结论修正。** Phase2309--2314 真正建立了完整多 token 候选计分、自由生成、全检查点 HiddenState、固定输出方向和局部梯度的分账；基础同坐标传动在 fresh 上的相对 MSE 约为 `0.90--1.05`，没有形成有用的同坐标齿轮。Phase2312 的 `20/20` confirmation 与 `20/20` fresh 只验证当前样本、当前输出边界的一阶 Taylor 控制，不证明共享语义方向、流形、全息编码或“精密微积分齿轮”。teacher forcing 候选计分也不等于自由未来规划。本期保留附件中“非对角响应必须靠新干预识别”“物理坐标不能跨模型对齐”“先观察后因果”的正确方向，拒绝在数据之前预设 SVD 模态、Grassmann 流形、Koopman 算子或完整 Jacobian。

**测试原理、材料与用例。** 模型加载前冻结八个构式族、英中两语、叙述/转述两表面、32 个词汇单元、两种事实状态和四个隔离分区，共 `{result['audit']['rows']}` 行。统一 `rather than` 尾句被删除，改用四种自然续写接口。例如施事--受事材料以 `Mara Vale handed the atlas to ... A later line records that` 为前缀，正确未来是完整句 `Mara Vale was the person who handed over the atlas.`；态度--事件、比较、位置、领属、关系从句、时间顺序和两跳封闭分类各自使用类型正确的完整句。taxonomy 明确限定为人工封闭目录，不冒充开放世界知识。独立人类盲评仍为 `NA`，因此自然度只能称机器审计通过。

$$
S(y_{{1:K}}\mid x)=\sum_{{k=1}}^K\log p(y_k\mid x,y_{{<k}}),\qquad
\mathcal H_i=\bigl(H_{{i,q,p,j}}\bigr)_{{q,p,j}}.
$$

主动系统识别只在冻结的八维 Rademacher 扰动张成空间中估计方向响应：

$$
D_{{i,q\to t}}(r_k)=\frac{{H_t(h_q+\epsilon\lVert h_q\rVert r_k)-H_t(h_q-\epsilon\lVert h_q\rVert r_k)}}{{2\epsilon\lVert h_q\rVert}},
$$

并用成对扰动检查最基础的局部叠加误差：

$$
E_{{ab}}=\frac{{\lVert D(r_a+r_b)-D(r_a)-D(r_b)\rVert_2^2}}{{\lVert D(r_a+r_b)\rVert_2^2+\varepsilon}}.
$$

这只能识别冻结方向上的模型局部响应，不能恢复任意 `2560 x 2560` Jacobian。

**冻结门、零模型与停止条件。** 完整候选未来在语言、表面、分区和提及顺序切片的总分与长度均值准确率须达到 `{SEQUENCE_GATE}`；自由生成身份命中率 `{FREE_IDENTITY_GATE}` 只决定结构控制资格，不阻断全场观察。零模型包括零响应、全族平均、族条件平均以及状态/语言/表面条件平均。fresh 揭盲前冻结扰动方向、正负剂量、源层、目标层、模型竞争和门槛；失败只淘汰对应路线，全部预注册分支执行后结束。跨模型只比较相对深度、族区分增益和叠加误差，不比较坐标编号。

**审计结果与相关文件。** 材料审计 `{json.dumps(result['audit'], ensure_ascii=False)}`；冻结配置 `{json.dumps(result['config'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`；哈希 `{json.dumps(result['hashes'], ensure_ascii=False)}`。脚本 `tests/glm5/phase2315_c5041_c5100_active_response_contract.py`；结果 `tests/glm5/result/phase2315_c5041_c5100_active_response_contract`。

**理论进展、硬伤与结论。** 理论主体继续保持“条件化输出场闭合理论”，本 Phase 没有新增机制阳性，只把研究对象从不可识别的完整非对角矩阵收紧为可执行的冻结方向响应图。硬伤包括研究者编写模板、无独立人类盲评、封闭 taxonomy、构式仍受控，以及八方向只覆盖极小的输入子空间。当前线性代数、逐坐标代数与局部差分足够表达合同；新数学必须等待多个构式和 fresh 词汇中出现稳定、不能由这些基础量解释的残差。下一步依次运行 Qwen3-4B 行为与全场、主动方向图、锁箱控制、顺序跨模型功能拓扑、可视化发布与原场清理。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("all_checks_passed"):
            append_memo(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
    parent = json.loads((P2314 / "analysis/final.json").read_text(encoding="utf-8"))
    if not parent["all_checks_passed"]:
        raise RuntimeError("Phase2314 is not authorized")
    config = frozen_config()
    rows, audit = compile_rows()
    write_rows(MATERIAL, rows)
    checks = {
        "parent_authorized": True,
        "config_frozen_before_model": config["frozen_before_model_load"],
        "row_count": len(rows) == len(FAMILIES) * len(LANGUAGES) * len(SURFACES) * UNITS * 2,
        "state_pairs_complete": audit["state_pairs_complete"],
        "balanced_cells": audit["exactly_balanced_cells"],
        "no_identity_collision": audit["identity_collisions"] == 0,
        "all_primary_roles_compiled": audit["missing_primary_role_positions"] == 0,
        "shorter_candidate_zero_model_at_chance": audit["zero_model_shorter_candidate_accuracy"] == 0.5,
        "mention_order_zero_model_at_chance": audit["zero_model_first_mention_accuracy"] == 0.5,
        "human_review_honestly_na": audit["independent_human_blind_review"] == "NA_not_run",
        "no_topk_pca_or_cosine_selection": True,
        "full_jacobian_not_claimed": True,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "timestamp": datetime.now().astimezone().isoformat(), "status": "closed",
        "audit": audit, "config": config,
        "hashes": {"material": file_hash(MATERIAL), "config": file_hash(CONFIG)},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "strict_conclusion": (
            "A balanced eight-construction natural-continuation and frozen directional-response campaign is now "
            "compiled. No model or mechanism result is produced in this phase."
        ),
        "next_authorization": "Run Qwen3-4B behavior, boundary full field, and representative all-token field.",
    }
    save(final_path, result)
    append_memo(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
