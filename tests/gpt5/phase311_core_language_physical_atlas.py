#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.stdout.reconfigure(encoding="utf-8")

import phase268_attention_mlp_continuation_path_attribution as p268  # noqa: E402
import phase305_internal_semantic_physical_path_probe as p305  # noqa: E402
import phase307_three_position_semantic_path_trace as p307  # noqa: E402
import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402


PHASE = "Phase311"
SCHEMA_VERSION = "3.0.0"
BRANCH_ID = "gpt5_pattern_family_atlas"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ROUND_DEFAULT = "core_language_physical_atlas"
RESULT_ROOT = ROOT / "tests/gpt5/result"
OUT = RESULT_ROOT / "phase311_core_language_physical_atlas"
V2 = RESULT_ROOT / "pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def digest(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def aliases(value: str) -> list[str]:
    vals = [value, value.lower(), value.capitalize()]
    if value.lower() == "yes":
        vals += ["true", "correct"]
    if value.lower() == "no":
        vals += ["false", "incorrect"]
    return list(dict.fromkeys(x for x in vals if x))


def split_for(index: int) -> str:
    if index < 3:
        return "discovery"
    if index == 3:
        return "calibration"
    return "heldout"


def case_row(
    family: str,
    mechanism: str,
    index: int,
    prompt: str,
    source_surface: str,
    query_surface: str,
    target: str,
    distractors: list[str],
    domain: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "branch_id": BRANCH_ID,
        "created_at": now(),
        "case_id_base": f"phase311:{family}:{mechanism}:{index:02d}",
        "family_id": family,
        "mechanism_id": mechanism,
        "domain_id": domain,
        "item_index": index,
        "split": split_for(index),
        "prompt": prompt,
        "source_surface": source_surface,
        "query_surface": query_surface,
        "target": target,
        "target_aliases": aliases(target),
        "distractor_aliases": list(dict.fromkeys(distractors)),
        "independent_case": True,
        "measurement_status": "planned",
    }


KNOWLEDGE = {
    "category_binding": [
        ("robin", "animal", ["tool", "plant", "material"]),
        ("hammer", "tool", ["animal", "plant", "material"]),
        ("oak", "plant", ["animal", "tool", "material"]),
        ("copper", "material", ["animal", "tool", "plant"]),
        ("violin", "instrument", ["animal", "tool", "plant"]),
    ],
    "color_binding": [
        ("lemon", "yellow", ["red", "blue", "black"]),
        ("coal", "black", ["yellow", "white", "green"]),
        ("snow", "white", ["black", "red", "blue"]),
        ("grass", "green", ["white", "yellow", "black"]),
        ("sky", "blue", ["green", "black", "red"]),
    ],
    "function_binding": [
        ("knife", "cutting", ["writing", "sitting", "calling"]),
        ("pen", "writing", ["cutting", "opening", "sitting"]),
        ("chair", "sitting", ["writing", "calling", "cutting"]),
        ("key", "opening", ["sitting", "writing", "calling"]),
        ("phone", "calling", ["opening", "cutting", "sitting"]),
    ],
    "part_binding": [
        ("car", "wheel", ["wing", "root", "page"]),
        ("bird", "wing", ["wheel", "roof", "page"]),
        ("tree", "root", ["wing", "page", "wheel"]),
        ("book", "page", ["root", "roof", "wheel"]),
        ("house", "roof", ["page", "wing", "root"]),
    ],
    "habitat_binding": [
        ("fish", "water", ["desert", "field", "ice"]),
        ("camel", "desert", ["water", "ice", "forest"]),
        ("penguin", "ice", ["desert", "field", "forest"]),
        ("farmer", "field", ["water", "ice", "desert"]),
        ("monkey", "forest", ["field", "ice", "water"]),
    ],
    "material_binding": [
        ("window", "glass", ["wood", "wool", "paper"]),
        ("table", "wood", ["glass", "wool", "rubber"]),
        ("sweater", "wool", ["paper", "glass", "rubber"]),
        ("notebook", "paper", ["wood", "glass", "wool"]),
        ("tire", "rubber", ["paper", "wood", "glass"]),
    ],
    "comparison_binding": [
        ("elephant|mouse", "elephant", ["mouse", "equal"]),
        ("tower|shed", "tower", ["shed", "equal"]),
        ("river|stream", "river", ["stream", "equal"]),
        ("train|bicycle", "train", ["bicycle", "equal"]),
        ("mountain|hill", "mountain", ["hill", "equal"]),
    ],
    "negated_attribute": [
        ("block|brown|blue", "brown", ["blue", "red", "green"]),
        ("flag|red|green", "red", ["green", "blue", "white"]),
        ("cup|white|black", "white", ["black", "yellow", "red"]),
        ("door|green|yellow", "green", ["yellow", "black", "blue"]),
        ("stone|black|white", "black", ["white", "green", "yellow"]),
    ],
}


def knowledge_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, items in KNOWLEDGE.items():
        for index, (key, target, distractors) in enumerate(items):
            if mechanism == "category_binding":
                prompt = f"Fact: A {key} belongs to the category {target}. Complete: A {key} is a kind of ___. Answer:"
                source, query = key, "kind of"
            elif mechanism == "color_binding":
                prompt = f"Fact: The usual color of {key} is {target}. Complete: The color of {key} is ___. Answer:"
                source, query = key, "color"
            elif mechanism == "function_binding":
                prompt = f"Fact: A {key} is commonly used for {target}. Complete: A {key} is used for ___. Answer:"
                source, query = key, "used for"
            elif mechanism == "part_binding":
                prompt = f"Fact: A {key} has a {target}. Complete: A part of a {key} is a ___. Answer:"
                source, query = key, "part"
            elif mechanism == "habitat_binding":
                prompt = f"Fact: A {key} is commonly associated with {target}. Complete: The habitat of the {key} is ___. Answer:"
                source, query = key, "habitat"
            elif mechanism == "material_binding":
                prompt = f"Fact: The {key} is made of {target}. Complete: Its material is ___. Answer:"
                source, query = key, "material"
            elif mechanism == "comparison_binding":
                left, right = key.split("|")
                prompt = f"Fact: The {left} is larger than the {right}. Complete: The larger object is the ___. Answer:"
                source, query = left, "larger object"
            else:
                obj, positive, negative = key.split("|")
                prompt = f"Fact: The {obj} is {positive}, not {negative}. Complete: The {obj} is ___. Answer:"
                source, query = obj, "not"
            rows.append(case_row("content_knowledge", mechanism, index, prompt, source, query, target, distractors, "multi_domain"))
    return rows


SYNTAX = {
    "subject_role": [
        ("The dogs chase cats", "dogs", ["cats", "chase"]),
        ("Mira opens doors", "Mira", ["doors", "opens"]),
        ("Birds build nests", "Birds", ["nests", "build"]),
        ("The child reads books", "child", ["books", "reads"]),
        ("Robots move boxes", "Robots", ["boxes", "move"]),
    ],
    "object_role": [
        ("The dogs chase cats", "cats", ["dogs", "chase"]),
        ("Mira opens doors", "doors", ["Mira", "opens"]),
        ("Birds build nests", "nests", ["Birds", "build"]),
        ("The child reads books", "books", ["child", "reads"]),
        ("Robots move boxes", "boxes", ["Robots", "move"]),
    ],
    "singular_agreement": [
        ("The dog near the trees", "runs", ["run", "running"]),
        ("The chef with the bowls", "cooks", ["cook", "cooking"]),
        ("The bird beside the windows", "sings", ["sing", "singing"]),
        ("The child among the adults", "writes", ["write", "writing"]),
        ("The robot near the boxes", "moves", ["move", "moving"]),
    ],
    "plural_agreement": [
        ("The dogs near the tree", "run", ["runs", "running"]),
        ("The chefs with the bowl", "cook", ["cooks", "cooking"]),
        ("The birds beside the window", "sing", ["sings", "singing"]),
        ("The children among the adult", "write", ["writes", "writing"]),
        ("The robots near the box", "move", ["moves", "moving"]),
    ],
    "past_tense": [
        ("Yesterday the dog", "walked", ["walk", "walks"]),
        ("Yesterday the chef", "cooked", ["cook", "cooks"]),
        ("Yesterday the bird", "sang", ["sing", "sings"]),
        ("Yesterday the child", "wrote", ["write", "writes"]),
        ("Yesterday the robot", "moved", ["move", "moves"]),
    ],
    "pronoun_number": [
        ("The lantern was bright, so", "it", ["they", "them"]),
        ("The books were heavy, so", "they", ["it", "its"]),
        ("The engine was loud, so", "it", ["they", "them"]),
        ("The workers were ready, so", "they", ["it", "its"]),
        ("The window was open, so", "it", ["they", "them"]),
    ],
    "adjective_attachment": [
        ("small red ball|red", "ball", ["small", "red"]),
        ("old stone bridge|old", "bridge", ["stone", "old"]),
        ("bright blue lamp|blue", "lamp", ["bright", "blue"]),
        ("long wooden table|wooden", "table", ["long", "wooden"]),
        ("quiet green room|green", "room", ["quiet", "green"]),
    ],
    "relative_clause_role": [
        ("The dog that chased the cat barked|barked", "dog", ["cat", "chased"]),
        ("The chef who greeted the guest smiled|smiled", "chef", ["guest", "greeted"]),
        ("The bird that followed the plane landed|landed", "bird", ["plane", "followed"]),
        ("The child who carried the bag rested|rested", "child", ["bag", "carried"]),
        ("The robot that moved the box stopped|stopped", "robot", ["box", "moved"]),
    ],
}


def syntax_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, items in SYNTAX.items():
        for index, (key, target, distractors) in enumerate(items):
            if mechanism == "subject_role":
                prompt = f"In the sentence '{key}', identify the grammatical subject. Answer:"
                source, query = key.split()[0], "subject"
            elif mechanism == "object_role":
                prompt = f"In the sentence '{key}', identify the direct object. Answer:"
                source, query = key.split()[0], "direct object"
            elif mechanism in {"singular_agreement", "plural_agreement"}:
                prompt = f"Complete with the grammatically agreeing verb: {key} ___. Answer:"
                source, query = key.split()[1], "agreeing verb"
            elif mechanism == "past_tense":
                prompt = f"Complete with the correct past-tense verb: {key} ___. Answer:"
                source, query = key.split()[-1], "past-tense"
            elif mechanism == "pronoun_number":
                prompt = f"Complete with the pronoun that agrees in number: {key} ___. Answer:"
                source, query = key.split()[1], "agrees in number"
            elif mechanism == "adjective_attachment":
                sentence, adjective = key.split("|")
                prompt = f"In the phrase '{sentence}', the adjective '{adjective}' describes the ___. Answer:"
                source, query = adjective, "describes"
            else:
                sentence, action = key.split("|")
                prompt = f"In the sentence '{sentence}', what performed the action '{action}'? Answer:"
                source, query = sentence.split()[1], "performed"
            rows.append(case_row("syntax_structure", mechanism, index, prompt, source, query, target, distractors, "controlled_syntax"))
    return rows


REASONING = {
    "direct_entailment": [
        ("Every red object is warm. The box is red.", "Is the box warm?", "yes"),
        ("Every metal object conducts. The rod is metal.", "Does the rod conduct?", "yes"),
        ("Every bird has wings. The robin is a bird.", "Does the robin have wings?", "yes"),
        ("Every square has corners. The tile is square.", "Does the tile have corners?", "yes"),
        ("Every key opens a lock. This item is a key.", "Can this item open a lock?", "yes"),
    ],
    "direct_contradiction": [
        ("No red object is cold. The box is red.", "Is the box cold?", "no"),
        ("No wooden object conducts. The rod is wooden.", "Does the rod conduct?", "no"),
        ("No fish has feathers. The salmon is a fish.", "Does the salmon have feathers?", "no"),
        ("No circle has corners. The tile is circular.", "Does the tile have corners?", "no"),
        ("No sealed door is open. This door is sealed.", "Is this door open?", "no"),
    ],
    "two_hop_entailment": [
        ("All robins are birds. All birds have wings. Ria is a robin.", "Does Ria have wings?", "yes"),
        ("All copper items are metal. All metal items conduct. The rod is copper.", "Does the rod conduct?", "yes"),
        ("All roses are flowers. All flowers are plants. Mira has a rose.", "Is Mira's rose a plant?", "yes"),
        ("All squares are polygons. All polygons have edges. The tile is square.", "Does the tile have edges?", "yes"),
        ("All keys are tools. All tools are objects. This item is a key.", "Is this item an object?", "yes"),
    ],
    "two_hop_blocked": [
        ("All robins are birds. All mammals are warm. Ria is a robin.", "Must Ria be warm?", "no"),
        ("All copper items are metal. All plants grow. The rod is copper.", "Must the rod grow?", "no"),
        ("All roses are flowers. All tools are useful. Mira has a rose.", "Must Mira's rose be useful?", "no"),
        ("All squares are polygons. All animals breathe. The tile is square.", "Must the tile breathe?", "no"),
        ("All keys are tools. All birds fly. This item is a key.", "Must this item fly?", "no"),
    ],
    "transitive_order": [
        ("A is taller than B. B is taller than C.", "Is A taller than C?", "yes"),
        ("D is older than E. E is older than F.", "Is D older than F?", "yes"),
        ("G is heavier than H. H is heavier than I.", "Is G heavier than I?", "yes"),
        ("J is faster than K. K is faster than L.", "Is J faster than L?", "yes"),
        ("M is brighter than N. N is brighter than O.", "Is M brighter than O?", "yes"),
    ],
    "reversed_order_control": [
        ("A is taller than B. B is taller than C.", "Is C taller than A?", "no"),
        ("D is older than E. E is older than F.", "Is F older than D?", "no"),
        ("G is heavier than H. H is heavier than I.", "Is I heavier than G?", "no"),
        ("J is faster than K. K is faster than L.", "Is L faster than J?", "no"),
        ("M is brighter than N. N is brighter than O.", "Is O brighter than M?", "no"),
    ],
    "conjunction_rule": [
        ("If an item is red and round, it is marked. The token is red and round.", "Is the token marked?", "yes"),
        ("If an item is blue and square, it is stored. The block is blue and square.", "Is the block stored?", "yes"),
        ("If a person is trained and ready, they can enter. Ana is trained and ready.", "Can Ana enter?", "yes"),
        ("If a device is charged and connected, it works. The phone is charged and connected.", "Does the phone work?", "yes"),
        ("If a door is closed and locked, it is secure. The door is closed and locked.", "Is the door secure?", "yes"),
    ],
    "missing_conjunct_control": [
        ("If an item is red and round, it is marked. The token is red but not round.", "Must the token be marked?", "no"),
        ("If an item is blue and square, it is stored. The block is blue but not square.", "Must the block be stored?", "no"),
        ("If a person is trained and ready, they can enter. Ana is trained but not ready.", "Can Ana enter by this rule?", "no"),
        ("If a device is charged and connected, it works. The phone is charged but disconnected.", "Must the phone work?", "no"),
        ("If a door is closed and locked, it is secure. The door is closed but unlocked.", "Must the door be secure?", "no"),
    ],
}


def reasoning_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism, items in REASONING.items():
        for index, (facts, question, target) in enumerate(items):
            prompt = f"Facts and rule: {facts} Question: {question} Answer yes or no. Answer:"
            source = facts.split()[0].strip(".,")
            rows.append(case_row("reasoning_constraint", mechanism, index, prompt, source, "Question", target, ["no" if target == "yes" else "yes"], "controlled_rules"))
    return rows


def build_case_bank() -> list[dict[str, Any]]:
    rows = knowledge_cases() + syntax_cases() + reasoning_cases()
    expected = 3 * 8 * 5
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} independent base cases, got {len(rows)}")
    return rows


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(str(text), add_special_tokens=False)]


def find_span(haystack: list[int], needle: list[int], prefer_last: bool = False) -> tuple[int | None, int | None]:
    if not needle or len(needle) > len(haystack):
        return None, None
    matches = [i for i in range(len(haystack) - len(needle) + 1) if haystack[i : i + len(needle)] == needle]
    if not matches:
        return None, None
    start = matches[-1] if prefer_last else matches[0]
    return start, start + len(needle) - 1


def locate_surface(tokenizer: Any, ids: list[int], surface: str, fallback: int, prefer_last: bool = False) -> dict[str, Any]:
    for candidate in [surface, " " + surface]:
        start, end = find_span(ids, encode_ids(tokenizer, candidate), prefer_last=prefer_last)
        if start is not None and end is not None:
            return {
                "token_position": end,
                "token_start": start,
                "token_end": end,
                "token_match_confidence": 1.0,
                "match_surface": surface,
                "multi_token_pooling_method": "last_token_of_span",
            }
    return {
        "token_position": fallback,
        "token_start": None,
        "token_end": None,
        "token_match_confidence": 0.25,
        "match_surface": surface,
        "multi_token_pooling_method": "fallback_single_position",
    }


def locate_positions(tokenizer: Any, case: dict[str, Any], prompt: str, last_pos: int) -> dict[str, dict[str, Any]]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    ids = [int(x) for x in encoded["input_ids"][0].tolist()]
    source = locate_surface(tokenizer, ids, str(case["source_surface"]), max(0, min(last_pos, 1)))
    query = locate_surface(tokenizer, ids, str(case["query_surface"]), max(0, last_pos - 2), prefer_last=True)
    return {
        "source": source,
        "query": query,
        "last": {
            "token_position": last_pos,
            "token_start": last_pos,
            "token_end": last_pos,
            "token_match_confidence": 1.0,
            "match_surface": "<last>",
            "multi_token_pooling_method": "last_context_token",
        },
    }


def semantic_groups(tokenizer: Any, case: dict[str, Any]) -> tuple[list[int], list[int], list[str], list[str]]:
    targets = [str(x) for x in case["target_aliases"]]
    distractors = [str(x) for x in case["distractor_aliases"] if str(x) not in targets]
    return p305.token_ids(tokenizer, targets), p305.token_ids(tokenizer, distractors), targets, distractors


def provenance(model_obj: Any, tokenizer: Any, model_name: str, attn_impl: str) -> dict[str, Any]:
    config = model_obj.config.to_dict() if hasattr(model_obj, "config") else {"class": type(model_obj).__name__}
    tokenizer_data = {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_ids": getattr(tokenizer, "all_special_ids", None),
    }
    return {
        "branch_id": BRANCH_ID,
        "run_id": f"phase311:{model_name}:{digest([now(), model_name])}",
        "git_commit": git_commit(),
        "model_name": model_name,
        "model_name_or_path": str(getattr(model_obj.config, "_name_or_path", model_name)),
        "model_hash": digest(config),
        "tokenizer_hash": digest(tokenizer_data),
        "attention_implementation": attn_impl,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(next(model_obj.parameters()).device),
    }


def trace_case(
    model_obj: Any,
    tokenizer: Any,
    device: torch.device,
    case: dict[str, Any],
    prov: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt = str(case["prompt"])
    captured, final_logits, last_pos = p268.capture_components(model_obj, tokenizer, device, prompt)
    positions = locate_positions(tokenizer, case, prompt, last_pos)
    target_ids, distractor_ids, target_aliases, distractor_aliases = semantic_groups(tokenizer, case)
    final_norm = p268.get_final_norm(model_obj)
    final_readout = p305.semantic_readout(final_logits, target_ids, distractor_ids)
    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for role, meta in positions.items():
        rows, summary = p307.decompose_position(
            model_obj,
            final_norm,
            captured,
            int(meta["token_position"]),
            case,
            role,
            target_ids,
            distractor_ids,
            target_aliases,
            distractor_aliases,
        )
        shared = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "branch_id": BRANCH_ID,
            "run_id": prov["run_id"],
            "git_commit": prov["git_commit"],
            "model_hash": prov["model_hash"],
            "tokenizer_hash": prov["tokenizer_hash"],
            "family_id": case["family_id"],
            "mechanism_id": case["mechanism_id"],
            "domain_id": case["domain_id"],
            "item_index": case["item_index"],
            "split": case["split"],
            "independent_case": True,
            "source_surface": case["source_surface"],
            "query_surface": case["query_surface"],
            "token_start": meta["token_start"],
            "token_end": meta["token_end"],
            "token_match_confidence": meta["token_match_confidence"],
            "match_surface": meta["match_surface"],
            "multi_token_pooling_method": meta["multi_token_pooling_method"],
        }
        for row in rows:
            row.update(shared)
            row["atlas_record_type"] = "core_language_component_event"
        summary.update(shared)
        summary.update(
            {
                "atlas_record_type": "core_language_position_summary",
                "actual_final_target_logit": round(safe_float(final_readout["target_semantic_logit"]), 6),
                "actual_final_distractor_logit": round(safe_float(final_readout["distractor_semantic_logit"]), 6),
                "actual_final_semantic_margin": round(safe_float(final_readout["semantic_margin"]), 6),
                "actual_final_semantic_winner": final_readout["semantic_winner"],
                "actual_target_token_id": final_readout["target_token_id"],
                "actual_distractor_token_id": final_readout["distractor_token_id"],
            }
        )
        component_rows.extend(rows)
        summary_rows.append(summary)
    case_result = {
        **{k: case[k] for k in case},
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "model": case["model"],
        "case_id": case["case_id"],
        "run_id": prov["run_id"],
        "measurement_status": "measured",
        "actual_final_semantic_margin": round(safe_float(final_readout["semantic_margin"]), 6),
        "actual_final_semantic_winner": final_readout["semantic_winner"],
        "position_match_confidence_mean": mean_safe([safe_float(v["token_match_confidence"]) for v in positions.values()]),
    }
    return component_rows, summary_rows, case_result


def prepare() -> dict[str, Any]:
    cases = build_case_bank()
    model_plan = []
    for model in MODELS:
        for case in cases:
            model_plan.append({**case, "model": model, "case_id": f"{case['case_id_base']}:{model}"})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "prepared",
        "base_independent_cases": len(cases),
        "planned_model_cases": len(model_plan),
        "family_counts": dict(Counter(str(r["family_id"]) for r in cases)),
        "mechanism_counts": dict(Counter(f"{r['family_id']}:{r['mechanism_id']}" for r in cases)),
        "split_counts": dict(Counter(str(r["split"]) for r in cases)),
        "models": MODELS,
        "git_commit": git_commit(),
    }
    write_jsonl(V2 / "phase311_core_language_case_bank.jsonl", cases)
    write_jsonl(V2 / "phase311_core_language_model_plan_rows.jsonl", model_plan)
    write_json(V2 / "phase311_case_bank_summary.json", payload)
    write_jsonl(LEGACY_V2 / "phase311_core_language_case_bank.jsonl", cases)
    write_jsonl(LEGACY_V2 / "phase311_core_language_model_plan_rows.jsonl", model_plan)
    write_json(LEGACY_V2 / "phase311_case_bank_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def summarize_model(model: str, planned: list[dict[str, Any]], measured: list[dict[str, Any]], components: list[dict[str, Any]], summaries: list[dict[str, Any]], missing: list[dict[str, Any]], prov: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(measured) + len(missing) == len(planned) else "partial",
        "model": model,
        "provenance": prov,
        "planned_independent_cases": len(planned),
        "valid_independent_cases": len(measured),
        "missing_independent_cases": len(missing),
        "layer_component_rows": len(components),
        "position_summary_rows": len(summaries),
        "family_counts": dict(Counter(str(r["family_id"]) for r in measured)),
        "mechanism_counts": dict(Counter(str(r["mechanism_id"]) for r in measured)),
        "split_counts": dict(Counter(str(r["split"]) for r in measured)),
        "target_winner_counts": dict(Counter(str(r["actual_final_semantic_winner"]) for r in measured)),
        "target_winner_rate": mean_safe([1.0 if r["actual_final_semantic_winner"] == "target" else 0.0 for r in measured]),
        "token_match_confidence_mean": mean_safe([safe_float(r["position_match_confidence_mean"]) for r in measured]),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT / args.round_name
    base_cases = build_case_bank()
    planned = [{**row, "model": args.model, "case_id": f"{row['case_id_base']}:{args.model}"} for row in base_cases]
    if args.limit:
        planned = planned[: int(args.limit)]
    model_obj = tokenizer = None
    components: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    prov: dict[str, Any] = {}
    try:
        model_obj, tokenizer, device, attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prov = provenance(model_obj, tokenizer, args.model, attn_impl)
        for index, case in enumerate(planned, 1):
            try:
                rows, position_rows, result = trace_case(model_obj, tokenizer, device, case, prov)
                components.extend(rows)
                summaries.extend(position_rows)
                measured.append(result)
            except Exception as exc:  # noqa: BLE001
                missing.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": args.model,
                        "case_id": case["case_id"],
                        "family_id": case["family_id"],
                        "mechanism_id": case["mechanism_id"],
                        "reason": repr(exc),
                    }
                )
            print(f"{args.model}: core physical path {index}/{len(planned)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    payload = summarize_model(args.model, planned, measured, components, summaries, missing, prov)
    write_json(out_dir / f"phase311_{args.model}_summary.json", payload)
    write_jsonl(out_dir / f"phase311_{args.model}_case_result_rows.jsonl", measured)
    write_jsonl(out_dir / f"phase311_{args.model}_component_rows.jsonl", components)
    write_jsonl(out_dir / f"phase311_{args.model}_position_summary_rows.jsonl", summaries)
    write_jsonl(out_dir / f"phase311_{args.model}_missing_rows.jsonl", missing)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def collect(round_name: str) -> dict[str, Any]:
    out_dir = OUT / round_name
    model_summaries = []
    cases: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        path = out_dir / f"phase311_{model}_summary.json"
        if path.exists():
            model_summaries.append(json.loads(path.read_text(encoding="utf-8")))
        cases.extend(read_jsonl(out_dir / f"phase311_{model}_case_result_rows.jsonl"))
        components.extend(read_jsonl(out_dir / f"phase311_{model}_component_rows.jsonl"))
        summaries.extend(read_jsonl(out_dir / f"phase311_{model}_position_summary_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase311_{model}_missing_rows.jsonl"))
    planned = len(build_case_bank()) * len(MODELS)
    by_family = Counter(str(r["family_id"]) for r in cases)
    family_winner = {
        family: mean_safe([1.0 if r["actual_final_semantic_winner"] == "target" else 0.0 for r in cases if r["family_id"] == family])
        for family in sorted(by_family)
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete" if len(cases) + len(missing) == planned and len(model_summaries) == len(MODELS) else "partial",
        "round_name": round_name,
        "planned_independent_model_cases": planned,
        "valid_independent_model_cases": len(cases),
        "missing_independent_model_cases": len(missing),
        "layer_component_rows": len(components),
        "position_summary_rows": len(summaries),
        "model_summaries": model_summaries,
        "family_counts": dict(by_family),
        "family_target_winner_rate": family_winner,
        "overall_target_winner_rate": mean_safe([1.0 if r["actual_final_semantic_winner"] == "target" else 0.0 for r in cases]),
        "token_match_confidence_mean": mean_safe([safe_float(r["token_match_confidence"]) for r in summaries]),
        "git_commit": git_commit(),
    }
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase311_core_language_physical_atlas_summary.json", payload)
        write_jsonl(base / "phase311_core_language_case_result_rows.jsonl", cases)
        write_jsonl(base / "phase311_core_language_component_rows.jsonl", components)
        write_jsonl(base / "phase311_core_language_position_summary_rows.jsonl", summaries)
        write_jsonl(base / "phase311_core_language_missing_rows.jsonl", missing)
    write_json(out_dir / "phase311_core_language_physical_atlas_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.summarize:
        collect(args.round_name)
    elif args.model:
        run_model(args)
    else:
        prepare()
        for model in MODELS:
            args.model = model
            run_model(args)
        collect(args.round_name)


if __name__ == "__main__":
    main()
