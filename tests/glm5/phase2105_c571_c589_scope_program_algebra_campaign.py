#!/usr/bin/env python3
"""C571-C589 scope-factorized language-program response campaign.

The campaign observes token embeddings and every post-block HiddenState
coordinate.  It never reads attention or MLP internals and never uses PCA,
Top-K selection, or magnitude truncation as a scientific selector.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c589_scope_program_algebra_atlas.json"
REGISTRY = ROOT / "ai2050_research_os/registry/field_datasets.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import model_utils
import phase1797_c263_c272_state_operator_common as compiler
import phase2076_c542_c559_typed_operation_response_passport_campaign as parent


PHASES = {
    f"C{campaign}": (2105 + campaign - 571, slug)
    for campaign, slug in (
        (571, "evidence_audit_and_scope_program_master_contract"),
        (572, "language_program_ontology_and_large_material_freeze"),
        (573, "compiler_semantic_balance_naturalness_and_qwen_behavior"),
        (574, "qwen_qualified_all_token_all_coordinate_capture"),
        (575, "full_field_observation_and_coordinate_response_atlas"),
        (576, "fixed_query_atomic_response_forward_prediction"),
        (577, "complete_voice_scope_factorial_decomposition"),
        (578, "translation_language_layout_factorial_decomposition"),
        (579, "behavior_qualified_path_depth_response"),
        (580, "discourse_voice_and_path_paraphrase_composition"),
        (581, "conditional_full_coordinate_system_identification"),
        (582, "bidirectional_response_equivalence_graph"),
        (583, "future_response_signature_and_predictive_state_quotient"),
        (584, "causal_eligibility_without_route_wide_stop"),
        (585, "qualified_local_state_guidance_or_registered_na"),
        (586, "sequential_cross_model_functional_topology"),
        (587, "nested_attitude_event_flagship"),
        (588, "recursive_knowledge_graph_flagship"),
        (589, "parameter_visualization_cleanup_and_campaign_synthesis"),
    )
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
QPOINTS = (0, 8, 16, 24, 32, 37)
SURFACES = ("record", "dialogue")
UNITS = 18
CONTROL_MARGIN = 0.02
BEHAVIOR_SLICE_GATE = 0.75
PREDICTION_GATE = 0.75
ORDER_BITS = (0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0)

ATOMIC_SPECS = {
    "discourse_permutation": ("temporal", "causal", "conjunction"),
    "fact_voice_fixed_query": ("inspect", "praise", "carry"),
    "evidence_paraphrase": ("purchase", "repair", "select"),
    "double_negation_surface": ("open", "visit", "mark"),
    "clause_packaging": ("inspect", "praise", "carry"),
    "path_depth": ("temporal", "sequence", "taxonomy"),
    "translation_language": ("inspect", "temporal", "taxonomy"),
    "relation_lexicalization": ("purchase", "repair", "select"),
}

NAMES_A = parent.NAMES_A
NAMES_B = parent.NAMES_B
OBJECTS_A = parent.OBJECTS_A
OBJECTS_B = parent.OBJECTS_B
MIDDLES = parent.MIDDLES
TARGETS = parent.TARGETS
NOISES = parent.NOISES


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def producer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    out.mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": producer_hash(), **protocol,
    })
    save(out / "audit/internal_checks.json", checks)
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError((name, checks))
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    save(out / "analysis/summary.json", headline)
    save(out / "audit/internal_checks_post.json", checks)
    value = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "all_checks_passed": all(bool(item) for item in checks.values()),
        "headline": headline, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", value)
    if not value["all_checks_passed"]:
        raise RuntimeError((name, checks))
    return value


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    if unit < 10:
        return "discovery"
    if unit < 14:
        return "confirmation"
    return "lockbox"


def options(truth: bool, order: int) -> tuple[str, int]:
    if order == 0:
        return "(A) Yes (B) No", 0 if truth else 1
    return "(A) No (B) Yes", 1 if truth else 0


def option_order(unit: int, *indices: int) -> int:
    return ORDER_BITS[unit] ^ (sum(indices) % 2)


def wrap(surface: str, facts: list[str], question: str) -> str:
    body = " ".join(facts)
    if surface == "record":
        return f"A verified record states: {body} Based only on this record, {question}"
    return f"Analyst: {body} Reviewer: Using only those statements, {question}"


def values(unit: int) -> dict:
    return {
        "a": NAMES_A[unit], "b": NAMES_B[unit], "x": OBJECTS_A[unit],
        "y": OBJECTS_B[unit], "mid": MIDDLES[unit], "target": TARGETS[unit],
        "noise": NOISES[unit],
    }


def relation_words(domain: str) -> tuple[str, str]:
    return {
        "inspect": ("inspected", "examined"),
        "praise": ("praised", "commended"),
        "carry": ("carried", "transported"),
        "purchase": ("bought", "purchased"),
        "repair": ("fixed", "repaired"),
        "select": ("chose", "selected"),
        "open": ("opened", "opened"),
        "visit": ("visited", "visited"),
        "mark": ("marked", "marked"),
        "temporal": ("preceded", "came before"),
        "sequence": ("comes before", "precedes"),
        "taxonomy": ("is a member of", "belongs to"),
        "causal": ("enabled", "made possible"),
        "conjunction": ("was paired with", "accompanied"),
    }[domain]


def make_row(
    *, case_id: str, panel: str, family: str, domain: str, surface: str,
    unit: int, cell: str, facts: list[str], question: str, truth: bool,
    roles: dict[str, str], semantic_graph: dict, order_offset: int = 0,
    variant: int | None = None, factors: dict | None = None,
) -> dict:
    order = option_order(unit, order_offset, SURFACES.index(surface) if surface in SURFACES else 0)
    choices, gold = options(truth, order)
    core = wrap(surface if surface in SURFACES else "record", facts, question)
    return {
        "case_id": case_id, "panel": panel, "family": family,
        "operation_type": family, "operation_domain": domain,
        "domain_id": f"{panel}:{family}:{domain}", "surface": surface,
        "construction": f"{panel}:{surface}", "unit": unit, "cell": cell,
        "variant": variant, "factors": factors or {}, "partition": partition(unit),
        "truth": truth, "correct_answer": "Yes" if truth else "No",
        "wrong_answer": "No" if truth else "Yes", "option_order": order,
        "gold_position": gold, "facts": facts, "question": question,
        "prompt_core": core, "prompt": f"{core} {choices}. Reply with only A or B.",
        "free_prompt": f"{core} Answer only Yes or No.",
        "role_values": roles, "semantic_graph": semantic_graph,
    }


def atomic_case(family: str, domain: str, surface: str, unit: int, variant: int) -> dict:
    u = values(unit)
    a, b, x, y, mid, target, noise = (u[k] for k in ("a", "b", "x", "y", "mid", "target", "noise"))
    truth = unit % 2 == 0
    query_object = x if truth else y
    r0, r1 = relation_words(domain)
    relation = r0

    if family == "discourse_permutation":
        facts = [f"{a} {r0} {x}.", f"{b} catalogued the {noise} separately."]
        if variant:
            facts.reverse()
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["evidence_order"]
    elif family == "fact_voice_fixed_query":
        facts = [f"{a} {r0} {x}." if not variant else f"{x} was {r0} by {a}.", f"{b} noted the {noise} separately."]
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["fact_voice"]
    elif family == "evidence_paraphrase":
        facts = [f"{a} {r0} {x}." if not variant else f"The record says that {x} was {r1} by {a}.", f"{b} logged the {noise} separately."]
        question = f"Is it true that {a} {r0} {query_object}?"
        relation = r0 if not variant else r1
        changed = ["evidence_paraphrase", "lexical_realization"]
    elif family == "double_negation_surface":
        facts = [f"{a} {r0} {x}." if not variant else f"It is not false that {a} {r0} {x}.", f"{b} noted the {noise} separately."]
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["double_negation_surface"]
    elif family == "clause_packaging":
        facts = [f"{a} {r0} {x}.", f"{b} catalogued the {noise}."] if not variant else [f"{a} {r0} {x} while {b} catalogued the {noise}."]
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["clause_packaging"]
    elif family == "path_depth":
        query_object = target if truth else y
        if not variant:
            facts = [f"{a} {r0} {target}.", f"{b} catalogued the {noise} separately."]
        else:
            facts = [f"{a} {r0} {mid}.", f"{mid} {r0} {target}.", f"{b} catalogued the {noise} separately."]
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["evidence_path_depth"]
        x = target
    elif family == "translation_language":
        zh = {"inspect": "检查了", "temporal": "早于", "taxonomy": "属于"}[domain]
        if not variant:
            facts = [f"{a} {r0} {x}.", f"{b} noted the {noise} separately."]
            relation = r0
        else:
            facts = [f"{a}{zh}{x}。", f"{b}另行记录了{noise}。"]
            relation = zh
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["evidence_language"]
    elif family == "relation_lexicalization":
        relation = r0 if not variant else r1
        facts = [f"{a} {relation} {x}.", f"{b} logged the {noise} separately."]
        question = f"Is it true that {a} {r0} {query_object}?"
        changed = ["relation_lexicalization"]
    else:
        raise KeyError(family)

    return make_row(
        case_id=f"c572-atomic-{family}-{domain}-{surface}-u{unit:02d}-v{variant}",
        panel="atomic", family=family, domain=domain, surface=surface, unit=unit,
        cell=f"v{variant}", variant=variant, facts=facts, question=question, truth=truth,
        roles={"primary": a, "secondary": b, "relation": relation, "context": x, "query": a},
        semantic_graph={
            "input_type": "evidence_query_program", "output_type": "binary_truth",
            "scope": "evidence_only", "invariants": ["query", "truth", "output_protocol", "entity_roles"],
            "changed": changed, "family": family, "domain": domain,
        }, order_offset=list(ATOMIC_SPECS).index(family) + list(ATOMIC_SPECS[family]).index(domain),
    )


def voice_scope_case(domain: str, surface: str, unit: int, fact_voice: int, query_voice: int) -> dict:
    u = values(unit); a, b, x, y, noise = u["a"], u["b"], u["x"], u["y"], u["noise"]
    relation = relation_words(domain)[0]; truth = unit % 2 == 0; target = x if truth else y
    fact = f"{a} {relation} {x}." if not fact_voice else f"{x} was {relation} by {a}."
    question = f"Is it true that {a} {relation} {target}?" if not query_voice else f"Is it true that {target} was {relation} by {a}?"
    return make_row(
        case_id=f"c572-voice-{domain}-{surface}-u{unit:02d}-f{fact_voice}q{query_voice}",
        panel="voice_scope_factorial", family="voice_scope", domain=domain, surface=surface, unit=unit,
        cell=f"f{fact_voice}q{query_voice}", factors={"fact_voice": fact_voice, "query_voice": query_voice},
        facts=[fact, f"{b} noted the {noise} separately."], question=question, truth=truth,
        roles={"primary": a, "secondary": b, "relation": relation, "context": x, "query": a},
        semantic_graph={"input_type": "voice_factorial", "scope": ["fact", "query"],
                        "invariants": ["truth", "output_protocol", "thematic_roles"],
                        "changed": [name for name, bit in (("fact_voice", fact_voice), ("query_voice", query_voice)) if bit]},
        order_offset=20 + list(("inspect", "praise", "carry")).index(domain),
    )


def translation_layout_case(domain: str, unit: int, language: int, layout: int) -> dict:
    u = values(unit); a, b, x, y, noise = u["a"], u["b"], u["x"], u["y"], u["noise"]
    relation = relation_words(domain)[0]; truth = unit % 2 == 0; target = x if truth else y
    zh = {"inspect": "检查了", "temporal": "早于", "taxonomy": "属于"}[domain]
    facts = [f"{a} {relation} {x}.", f"{b} noted the {noise} separately."] if not language else [f"{a}{zh}{x}。", f"{b}另行记录了{noise}。"]
    if layout:
        facts = ["Evidence item one: " + facts[0], "Evidence item two: " + facts[1]]
    return make_row(
        case_id=f"c572-translation-{domain}-u{unit:02d}-l{language}p{layout}",
        panel="translation_layout_factorial", family="translation_layout", domain=domain,
        surface="record", unit=unit, cell=f"l{language}p{layout}",
        factors={"language": language, "layout": layout}, facts=facts,
        question=f"Is it true that {a} {relation} {target}?", truth=truth,
        roles={"primary": a, "secondary": b, "relation": zh if language else relation, "context": x, "query": a},
        semantic_graph={"input_type": "translation_layout_factorial", "scope": "evidence_only",
                        "invariants": ["query", "truth", "output_protocol", "entity_roles"],
                        "changed": [name for name, bit in (("language", language), ("layout", layout)) if bit]},
        order_offset=30 + list(("inspect", "temporal", "taxonomy")).index(domain),
    )


def composition_case(panel: str, domain: str, unit: int, a_bit: int, b_bit: int) -> dict:
    u = values(unit); a, b, x, y, mid, target, noise = (u[k] for k in ("a", "b", "x", "y", "mid", "target", "noise"))
    truth = unit % 2 == 0
    if panel == "discourse_voice_composition":
        relation = relation_words(domain)[0]; query_target = x if truth else y
        target_fact = f"{a} {relation} {x}." if not a_bit else f"{x} was {relation} by {a}."
        facts = [target_fact, f"{b} catalogued the {noise} separately."]
        if b_bit: facts.reverse()
        question = f"Is it true that {a} {relation} {query_target}?"
        factor_names = ("fact_voice", "discourse_order")
    else:
        relation = relation_words(domain)[0]; query_target = target if truth else y
        if not a_bit:
            core = f"{a} {relation} {target}."
        else:
            core = f"{a} {relation} {mid}. {mid} {relation} {target}."
        if b_bit:
            core = "The relevant evidence says that " + core
        facts = [core, f"{b} catalogued the {noise} separately."]
        question = f"Is it true that {a} {relation} {query_target}?"
        factor_names = ("path_depth", "surface_paraphrase")
        x = target
    return make_row(
        case_id=f"c572-{panel}-{domain}-u{unit:02d}-{a_bit}{b_bit}",
        panel=panel, family=panel, domain=domain, surface="record", unit=unit,
        cell=f"a{a_bit}b{b_bit}", factors={factor_names[0]: a_bit, factor_names[1]: b_bit},
        facts=facts, question=question, truth=truth,
        roles={"primary": a, "secondary": b, "relation": relation, "context": x, "query": a},
        semantic_graph={"input_type": "two_factor_composition", "scope": "evidence_only",
                        "invariants": ["query", "truth", "output_protocol"],
                        "changed": [name for name, bit in zip(factor_names, (a_bit, b_bit)) if bit]},
        order_offset=40 + (0 if panel == "discourse_voice_composition" else 4),
    )


def nested_attitude_case(domain: str, surface: str, unit: int, outer_neg: int, inner_neg: int) -> dict:
    u = values(unit); a, b, x, y, noise = u["a"], u["b"], u["x"], u["y"], u["noise"]
    verb = {"like": "likes", "regret": "regrets", "remember": "remembers"}[domain]
    obj = x; inner = f"{b} {'not ' if inner_neg else ''}eating {obj}"
    verb_surface = verb[:-1] if outer_neg else verb
    statement = f"{a} {'does not ' + verb_surface if outer_neg else verb_surface} {inner}."
    query_obj = x if unit % 2 == 0 else y; truth = unit % 2 == 0
    query_inner = f"{b} {'not ' if inner_neg else ''}eating {query_obj}"
    query = f"Is it true that {a} {'does not ' + verb_surface if outer_neg else verb_surface} {query_inner}?"
    return make_row(
        case_id=f"c572-nested-{domain}-{surface}-u{unit:02d}-o{outer_neg}i{inner_neg}",
        panel="nested_attitude_flagship", family="nested_attitude", domain=domain, surface=surface,
        unit=unit, cell=f"o{outer_neg}i{inner_neg}", factors={"outer_negation": outer_neg, "inner_negation": inner_neg},
        facts=[statement, f"{b} catalogued the {noise} separately."], question=query, truth=truth,
        roles={"primary": a, "secondary": b, "relation": verb_surface, "context": x, "query": a},
        semantic_graph={"input_type": "nested_attitude_event", "scope": ["attitude", "embedded_event"],
                        "invariants": ["role_binding", "output_protocol"],
                        "changed": [name for name, bit in (("outer_negation", outer_neg), ("inner_negation", inner_neg)) if bit]},
        order_offset=50 + list(("like", "regret", "remember")).index(domain),
    )


def graph_case(domain: str, unit: int, source_kind: int, depth: int, shortcut: int) -> dict:
    u = values(unit); a, b, x, y, mid, target, noise = (u[k] for k in ("a", "b", "x", "y", "mid", "target", "noise"))
    if source_kind == 0:
        source = {"taxonomy": "apple", "part_whole": "wheel", "temporal": "dawn"}[domain]
        middle1 = {"taxonomy": "fruit", "part_whole": "axle", "temporal": "morning"}[domain]
        middle2 = {"taxonomy": "food", "part_whole": "vehicle", "temporal": "noon"}[domain]
        final_target = {"taxonomy": "object", "part_whole": "machine", "temporal": "evening"}[domain]
    else:
        source, middle1, middle2, final_target = a.lower(), mid, x, target
    relation = {"taxonomy": "is a kind of", "part_whole": "is part of", "temporal": "precedes"}[domain]
    chain = [(source, middle1), (middle1, middle2), (middle2, final_target)]
    facts = [f"{left} {relation} {right}." for left, right in chain[:depth]]
    endpoint = chain[depth - 1][1]
    if shortcut and depth > 1:
        facts.append(f"{source} {relation} {endpoint}.")
    facts.append(f"{b} catalogued the {noise} separately.")
    truth = unit % 2 == 0; queried = endpoint if truth else y
    return make_row(
        case_id=f"c572-graph-{domain}-u{unit:02d}-s{source_kind}d{depth}k{shortcut}",
        panel="recursive_knowledge_flagship", family="recursive_knowledge", domain=domain,
        surface="record", unit=unit, cell=f"s{source_kind}d{depth}k{shortcut}",
        factors={"source_kind": source_kind, "depth": depth, "shortcut": shortcut},
        facts=facts, question=f"Is it true that {source} {relation} {queried}?", truth=truth,
        roles={"primary": source, "secondary": b, "relation": relation, "context": endpoint, "query": source},
        semantic_graph={"input_type": "recursive_relation_graph", "scope": "evidence_path",
                        "invariants": ["query_direction", "output_protocol"],
                        "changed": ["source_kind", "depth", "shortcut"]},
        order_offset=60 + list(("taxonomy", "part_whole", "temporal")).index(domain),
    )


def material() -> list[dict]:
    rows: list[dict] = []
    for family, domains in ATOMIC_SPECS.items():
        for domain, surface, unit, variant in itertools.product(domains, SURFACES, range(UNITS), (0, 1)):
            rows.append(atomic_case(family, domain, surface, unit, variant))
    for domain, surface, unit, f, q in itertools.product(("inspect", "praise", "carry"), SURFACES, range(UNITS), (0, 1), (0, 1)):
        rows.append(voice_scope_case(domain, surface, unit, f, q))
    for domain, unit, language, layout in itertools.product(("inspect", "temporal", "taxonomy"), range(UNITS), (0, 1), (0, 1)):
        rows.append(translation_layout_case(domain, unit, language, layout))
    for panel, domains in (("discourse_voice_composition", ("inspect", "praise", "carry")), ("path_paraphrase_composition", ("temporal", "sequence", "taxonomy"))):
        for domain, unit, a, b in itertools.product(domains, range(UNITS), (0, 1), (0, 1)):
            rows.append(composition_case(panel, domain, unit, a, b))
    for domain, surface, unit, outer, inner in itertools.product(("like", "regret", "remember"), SURFACES, range(UNITS), (0, 1), (0, 1)):
        rows.append(nested_attitude_case(domain, surface, unit, outer, inner))
    for domain, unit, source_kind, depth, shortcut in itertools.product(("taxonomy", "part_whole", "temporal"), range(UNITS), (0, 1), (1, 2, 3), (0, 1)):
        if depth == 1 and shortcut == 1:
            continue
        rows.append(graph_case(domain, unit, source_kind, depth, shortcut))
    return rows


def material_path() -> Path:
    return OUTS["C572"] / "material/scope_program_cases.jsonl"


def compiled_path() -> Path:
    return OUTS["C573"] / "compiled/qwen3_scope_program_cases.jsonl"


def behavior_path() -> Path:
    return OUTS["C573"] / "behavior/qwen3_behavior.jsonl"


def capture_path() -> Path:
    return OUTS["C574"] / "raw/qwen3_role_mean_states.float16.npy"


def capture_last_path() -> Path:
    return OUTS["C574"] / "raw/qwen3_role_last_states.float16.npy"


def full_shard_dir() -> Path:
    return OUTS["C574"] / "raw/qwen3_full_token_shards"


def capture_index_path() -> Path:
    return OUTS["C574"] / "raw/hidden_index.jsonl"


def qualified_path() -> Path:
    return OUTS["C573"] / "behavior/qualified_slices.json"


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    p = np.asarray(prediction, np.float64).reshape(-1)
    y = np.asarray(truth, np.float64).reshape(-1)
    error = p - y
    denom = math.sqrt(float(np.mean(y * y))) + 1e-12
    pnorm = float(np.linalg.norm(p)); ynorm = float(np.linalg.norm(y))
    return {
        "nrmse": math.sqrt(float(np.mean(error * error))) / denom,
        "coordinate_balanced_mae": float(np.mean(np.abs(error) / (np.abs(y) + np.median(np.abs(y)) + 1e-6))),
        "cosine": float(np.dot(p, y) / (pnorm * ynorm + 1e-12)),
        "sign_agreement": float(np.mean(np.sign(p) == np.sign(y))),
        "prediction_rms": math.sqrt(float(np.mean(p * p))),
        "target_rms": math.sqrt(float(np.mean(y * y))),
    }


def scaled_like(control: np.ndarray, reference: np.ndarray) -> np.ndarray:
    c = np.asarray(control, np.float32); r = np.asarray(reference, np.float32)
    return c * (float(np.linalg.norm(r)) / (float(np.linalg.norm(c)) + 1e-12))


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def role_state(states: np.ndarray, row: dict, q: int, role: str, view: str = "mean") -> np.ndarray:
    if states.ndim == 4:
        return np.asarray(states[int(row["hidden_index"]), q, ROLES.index(role)], np.float32)
    points = [int(value) for value in row["role_positions"][role]]
    values_ = np.asarray(states[int(row["hidden_index"]), q, points], np.float32)
    return values_.mean(axis=0) if view == "mean" else values_[-1]


def role_bundle(states: np.ndarray, row: dict, q: int, view: str = "mean") -> np.ndarray:
    return np.stack([role_state(states, row, q, role, view) for role in ROLES])


def index_rows() -> list[dict]:
    return read_rows(capture_index_path())


def pair_rows(index: list[dict], panel: str, family: str, part: str | None = None) -> list[tuple[dict, dict]]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in index:
        if row["panel"] != panel or row["family"] != family or row.get("variant") is None:
            continue
        if part is not None and row["partition"] != part:
            continue
        key = (row["operation_domain"], row["surface"], row["unit"])
        groups[key][int(row["variant"])] = row
    return [(values_[0], values_[1]) for values_ in groups.values() if set(values_) == {0, 1}]


def cell_rows(index: list[dict], panel: str, part: str | None = None) -> list[dict[tuple, dict[str, dict]]]:
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in index:
        if row["panel"] != panel or (part is not None and row["partition"] != part):
            continue
        key = (row["operation_domain"], row["surface"], row["unit"])
        groups[key][row["cell"]] = row
    return [{key: value} for key, value in groups.items()]


def pair_delta(states: np.ndarray, pair: tuple[dict, dict], q: int, view: str = "mean") -> np.ndarray:
    return role_bundle(states, pair[1], q, view) - role_bundle(states, pair[0], q, view)


def factorial_effects(states: np.ndarray, cells: dict[str, dict], q: int, labels: tuple[str, str] = ("a", "b")) -> dict[str, np.ndarray]:
    keys = {
        "00": next((key for key in cells if key.endswith("0") and key[-2] == "0"), None),
        "01": next((key for key in cells if key.endswith("1") and key[-2] == "0"), None),
        "10": next((key for key in cells if key.endswith("0") and key[-2] == "1"), None),
        "11": next((key for key in cells if key.endswith("1") and key[-2] == "1"), None),
    }
    if any(value is None for value in keys.values()):
        raise RuntimeError((cells.keys(), keys))
    h00, h01, h10, h11 = (role_bundle(states, cells[keys[name]], q) for name in ("00", "01", "10", "11"))
    return {
        labels[0]: 0.5 * ((h10 - h00) + (h11 - h01)),
        labels[1]: 0.5 * ((h01 - h00) + (h11 - h10)),
        "interaction": h11 - h10 - h01 + h00,
        "additive_prediction": h10 + h01 - h00,
        "joint_target": h11,
    }


def c571() -> None:
    out = begin("C571", {
        "status": "scope_program_master_contract_frozen",
        "parents": ["C560 artifact audit 157/157", "C569 artifact audit 290/290", "C570 route adjudication"],
        "strict_evidence_audit": {
            "artifact_audits": "internal reproducibility and ledger consistency only; not external replication",
            "old_voice": "joint fact+query rewrite passport, not a pure voice operator",
            "glm": "within-model topology only; no coordinate identity with Qwen",
            "new_math": "middle-level object hypothesis only; foundational upgrade unauthorized",
        },
        "campaign_object": "scope-factorized language programs and their behavior-qualified full-coordinate response fields",
        "routes": [
            "fixed-query atomic operations", "2x2 voice scope", "2x2 translation-layout",
            "path depth", "two composition panels", "conditional system identification",
            "response equivalence", "future response signatures", "qualified local guidance",
            "cross-model topology", "nested attitude", "recursive knowledge graph",
        ],
        "failure_policy": "route-level NA or rejection; one failed family never stops other registered routes",
        "observation_policy": "all tokens and all physical coordinates; no attention/MLP, PCA, Top-K, or magnitude truncation",
        "human_naturalness": "NA_not_run; machine lint cannot replace independent blind human review",
    }, {
        "parent_c560": load(RESULT / "phase2094_c560_typed_operation_response_passport_campaign_independent_audit/analysis/final.json")["all_checks_passed"],
        "parent_c570": load(RESULT / "phase2104_c570_next_exact_object_route_adjudication/analysis/final.json")["all_checks_passed"],
        "phase_continuity": PHASES["C571"][0] == 2105,
    })
    save(out / "analysis/evidence_adjudication.json", {
        "retain": [
            "some typed full-coordinate responses predict held-out units",
            "response transfer is strongly scope and query-contract dependent",
            "GLM4 has a within-model voice-response topology",
            "full-state, multi-token conditional objects remain better candidates than fixed coordinates",
        ],
        "tighten": [
            "157/157 and 290/290 are artifact audits, not scientific replications",
            "C555 is local downstream state guidance, not necessity or unique causation",
            "C564 refutes a pure scope-free voice passport under the registered fresh contract",
            "no empirical composition algebra or cross-model functional isomorphism has yet closed",
        ],
    })
    close("C571", {
        "status": "evidence_audit_and_contract_closed",
        "overclaims_removed": 4, "independent_routes": 12,
        "strict_conclusion": "The next object is a typed, scope-indexed response family. New mathematics remains a hypothesis, not a result.",
    }, {"audit_written": (out / "analysis/evidence_adjudication.json").exists()}, "C572_material_freeze")


def c572() -> None:
    out = begin("C572", {
        "status": "language_program_ontology_and_material_frozen",
        "units": UNITS, "partitions": {"discovery": "0-9", "confirmation": "10-13", "lockbox": "14-17"},
        "panels": ["atomic", "voice_scope_factorial", "translation_layout_factorial",
                   "discourse_voice_composition", "path_paraphrase_composition",
                   "nested_attitude_flagship", "recursive_knowledge_flagship"],
        "frozen_axes": ["object", "material", "partition", "model", "nulls", "gates", "failure branches"],
    }, {"parent": final("C571")["all_checks_passed"]})
    rows = material()
    write_rows(material_path(), rows)
    ontology = {
        "schema": "ai2050.scope_program_ontology.v1",
        "language_program": "L=(V,E,tau,rho,sigma,kappa,Q,Omega)",
        "operation": "o_eta:L->L' with explicit scope, invariants, query policy and output policy",
        "atomic_specs": {key: list(value) for key, value in ATOMIC_SPECS.items()},
        "panels": sorted({row["panel"] for row in rows}),
        "registered_nulls": ["zero", "wrong-family equal-norm", "wrong-scope", "wrong-role", "coordinate-shift"],
        "gates": {
            "behavior_slice": BEHAVIOR_SLICE_GATE,
            "prediction_pass_rate": PREDICTION_GATE,
            "prediction_margin": CONTROL_MARGIN,
            "causal": "only families passing behavior and forward-prediction eligibility",
        },
    }
    save(out / "material/language_program_ontology.json", ontology)
    counts = {panel: sum(row["panel"] == panel for row in rows) for panel in ontology["panels"]}
    parts = {part: sum(row["partition"] == part for row in rows) for part in ("discovery", "confirmation", "lockbox")}
    unique = len({row["case_id"] for row in rows})
    close("C572", {
        "status": "ontology_and_material_closed", "rows": len(rows), "unique_cases": unique,
        "panel_counts": counts, "partition_counts": parts, "families": len({row["family"] for row in rows}),
        "domains": len({(row["family"], row["operation_domain"]) for row in rows}),
        "examples": {panel: next(row["prompt"] for row in rows if row["panel"] == panel) for panel in ontology["panels"]},
    }, {"nonempty": bool(rows), "unique": unique == len(rows), "all_roles": all(set(row["role_values"]) == set(ROLES[:-1]) for row in rows)}, "C573_compile_behavior")


def c573() -> None:
    out = begin("C573", {
        "status": "compiler_semantic_balance_naturalness_and_behavior_frozen",
        "model": "Qwen3-4B BF16 CUDA", "behavior_before_hiddenstate": True,
        "semantic_uniqueness": "registered program graph and exact changed/invariant ledgers",
        "naturalness": "machine syntax lint plus human blind review explicitly NA",
        "behavior_gate": f"each panel-family-domain slice accuracy >= {BEHAVIOR_SLICE_GATE}",
    }, {"parent": final("C572")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = read_rows(material_path())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    compiled = compiler.compile_qwen(tokenizer, rows)
    write_rows(compiled_path(), compiled)
    prompts = [row["prompt"] for row in rows]
    widths = [len(row["prompt_ids"]) for row in compiled]
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows: duplicate_groups[row["prompt"]].append(row)
    shared = [items for items in duplicate_groups.values() if len(items) > 1]
    cross_partition = [items for items in shared if len({row["partition"] for row in items}) > 1]
    inconsistent = [items for items in shared if len({(row["truth"], row["gold_position"]) for row in items}) > 1]
    malformed = [row["case_id"] for row in rows if "  " in row["prompt"] or not row["question"].endswith("?")]
    semantic_bad = [row["case_id"] for row in rows if not row["semantic_graph"].get("invariants") or "output_protocol" not in row["semantic_graph"]["invariants"]]
    balance = {}
    for key, items in itertools.groupby(sorted(rows, key=lambda r:(r["panel"],r["family"],r["operation_domain"],r["surface"])), key=lambda r:(r["panel"],r["family"],r["operation_domain"],r["surface"])):
        vals = list(items); balance["|".join(key)] = {"rows": len(vals), "truth_rate": float(np.mean([r["truth"] for r in vals])), "a_first_rate": float(np.mean([r["option_order"] == 0 for r in vals]))}

    model = None; behavior = []
    try:
        model, tokenizer, device, placement = parent.previous.model_base().load_bf16("qwen3")
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(compiled), 12):
            batch = compiled[start:start+12]; width = max(len(row["prompt_ids"]) for row in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]; ids[i,:len(seq)] = torch.tensor(seq,device=device); mask[i,:len(seq)] = 1
            pos = mask.long().cumsum(-1)-1; pos.masked_fill_(mask == 0, 0)
            with torch.inference_mode(): logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            for i, row in enumerate(batch):
                length=len(row["prompt_ids"]); scores=[float(logits[i,length-1,c[0]]) for c in row["candidate_ids"]]; pred=int(scores[1]>scores[0])
                behavior.append({"case_id":row["case_id"],"panel":row["panel"],"family":row["family"],"operation_domain":row["operation_domain"],"surface":row["surface"],"unit":row["unit"],"partition":row["partition"],"cell":row["cell"],"variant":row.get("variant"),"gold_position":row["gold_position"],"prediction":pred,"correct":pred==row["gold_position"],"candidate_scores":scores})
            if start % 360 == 0 or start + len(batch) == len(compiled): print(f"[C573 behavior] {start+len(batch)}/{len(compiled)}",flush=True)
    finally:
        parent.previous.model_base().release_bf16(model); gc.collect()
    write_rows(behavior_path(), behavior)
    slices: dict[str, dict] = {}
    for key, group in itertools.groupby(sorted(behavior,key=lambda r:(r["panel"],r["family"],r["operation_domain"])), key=lambda r:(r["panel"],r["family"],r["operation_domain"])):
        vals=list(group); acc=float(np.mean([r["correct"] for r in vals])); slices["|".join(key)]={"rows":len(vals),"accuracy":acc,"qualified":acc>=BEHAVIOR_SLICE_GATE}
    qualified=sorted(key for key,value in slices.items() if value["qualified"]); save(qualified_path(),{"gate":BEHAVIOR_SLICE_GATE,"qualified":qualified,"slices":slices})
    headline={"status":"compile_audit_behavior_closed","rows":len(rows),"compiled_rows":len(compiled),"unique_prompts":len(set(prompts)),"shared_prompt_groups":len(shared),"cross_partition_shared_groups":len(cross_partition),"inconsistent_shared_groups":len(inconsistent),"max_width":max(widths),"malformed":len(malformed),"semantic_bad":len(semantic_bad),"human_naturalness":"NA_not_run","balance":balance,"behavior_accuracy":float(np.mean([r["correct"] for r in behavior])),"behavior_slices":slices,"qualified_slices":len(qualified),"total_slices":len(slices),"placement":placement}
    close("C573",headline,{"rows":len(compiled)==len(rows),"width":max(widths)<=180,"roles":all(set(row["role_positions"])==set(ROLES) for row in compiled),"no_cross_partition_duplicates":not cross_partition,"consistent_duplicates":not inconsistent,"syntax":not malformed,"semantics":not semantic_bad,"behavior_complete":len(behavior)==len(rows),"some_qualified":bool(qualified)},"C574_qualified_capture")


def c574() -> None:
    out = begin("C574", {
        "status": "qwen_qualified_full_field_capture_frozen", "model": "Qwen3-4B BF16 CUDA",
        "selection": "only predeclared panel-family-domain slices passing C573 behavior gate",
        "tensor": "qualified_sample x 38 checkpoints x all prompt tokens x 2560 coordinates",
        "coordinate_policy": "complete physical field; no PCA, Top-K, thresholding, or role-only storage",
    }, {"parent": final("C573")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    legacy_invalid = OUTS["C574"] / "raw/qwen3_full_token_states.float16.npy"
    if legacy_invalid.exists() and not capture_index_path().exists():
        # A previous run crashed before the first index commit while zero-filling
        # this monolithic file.  It contains no attributable sample and is not a
        # scientific artifact; remove it before creating bounded shards.
        legacy_invalid.unlink()
    rows = read_rows(material_path()); compiled_all = read_rows(compiled_path()); behavior = {r["case_id"]:r for r in read_rows(behavior_path())}; qualified=set(load(qualified_path())["qualified"])
    selected=[]
    for row, comp in zip(rows, compiled_all):
        key=f"{row['panel']}|{row['family']}|{row['operation_domain']}"
        if key in qualified: selected.append((row,comp))
    selected.sort(key=lambda item: len(item[1]["prompt_ids"]))
    width=max(len(comp["prompt_ids"]) for _,comp in selected); n=len(selected)
    capture_path().parent.mkdir(parents=True,exist_ok=True);full_shard_dir().mkdir(parents=True,exist_ok=True)
    mean_states=np.lib.format.open_memmap(capture_path(),mode="w+",dtype=np.float16,shape=(n,CHECKPOINTS,len(ROLES),DIM))
    last_states=np.lib.format.open_memmap(capture_last_path(),mode="w+",dtype=np.float16,shape=(n,CHECKPOINTS,len(ROLES),DIM))
    model=None; hooks=[]; captured=[]; index=[]; headline={};shard_ledger=[]
    try:
        model,tokenizer,device,placement=parent.previous.model_base().load_bf16("qwen3"); quant=parent.previous.model_base().quantization_audit(model); base=model.model
        def hook(_module,_args,output): captured.append(output[0] if isinstance(output,tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook)); hooks.extend(layer.register_forward_hook(hook) for layer in base.layers); hooks.append(base.norm.register_forward_hook(hook))
        pad=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        shard_size=32
        for shard_start in range(0,n,shard_size):
            shard_items=selected[shard_start:shard_start+shard_size];shard_width=max(len(comp["prompt_ids"]) for _,comp in shard_items);shard_id=shard_start//shard_size;shard_path=full_shard_dir()/f"shard_{shard_id:04d}.float16.npy"
            shard=np.lib.format.open_memmap(shard_path,mode="w+",dtype=np.float16,shape=(len(shard_items),CHECKPOINTS,shard_width,DIM))
            for local_start in range(0,len(shard_items),4):
                batch=shard_items[local_start:local_start+4];ids=torch.full((len(batch),shard_width),pad,dtype=torch.long,device=device);mask=torch.zeros_like(ids);lengths=[];weights=torch.zeros((len(batch),len(ROLES),shard_width),dtype=torch.float32,device=device);last_pos=torch.zeros((len(batch),len(ROLES)),dtype=torch.long,device=device)
                for i,(_row,comp) in enumerate(batch):
                    seq=comp["prompt_ids"];lengths.append(len(seq));ids[i,:len(seq)]=torch.tensor(seq,device=device);mask[i,:len(seq)]=1
                    for role_i,role in enumerate(ROLES):
                        points=[int(v) for v in comp["role_positions"][role]];weights[i,role_i,points]=1.0/len(points);last_pos[i,role_i]=points[-1]
                pos=mask.long().cumsum(-1)-1;pos.masked_fill_(mask==0,0);captured.clear()
                with torch.inference_mode():model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
                if len(captured)!=CHECKPOINTS:raise RuntimeError(("checkpoints",len(captured)))
                for q,state in enumerate(captured):
                    state32=state.float();mean_states[shard_start+local_start:shard_start+local_start+len(batch),q]=torch.einsum("brt,btd->brd",weights,state32).cpu().numpy().astype(np.float16);gather=last_pos[:,:,None].expand(-1,-1,DIM);last_states[shard_start+local_start:shard_start+local_start+len(batch),q]=torch.gather(state32,1,gather).cpu().numpy().astype(np.float16)
                    for i,length in enumerate(lengths):shard[local_start+i,q,:length]=state[i,:length].float().cpu().numpy().astype(np.float16)
                for i,(row,comp) in enumerate(batch):
                    global_i=shard_start+local_start+i;index.append({"hidden_index":global_i,"shard":shard_path.name,"shard_index":local_start+i,"case_id":row["case_id"],"panel":row["panel"],"family":row["family"],"operation_domain":row["operation_domain"],"surface":row["surface"],"unit":row["unit"],"partition":row["partition"],"cell":row["cell"],"variant":row.get("variant"),"factors":row.get("factors",{}),"truth":row["truth"],"length":lengths[i],"role_positions":comp["role_positions"],"behavior_correct":behavior[row["case_id"]]["correct"]})
            shard.flush();del shard;mean_states.flush();last_states.flush();shard_ledger.append({"shard":shard_path.name,"rows":len(shard_items),"width":shard_width,"bytes":shard_path.stat().st_size})
            print(f"[C574 capture] {min(shard_start+len(shard_items),n)}/{n} shard={shard_id:04d} width={shard_width}",flush=True)
        write_rows(capture_index_path(),index)
        save(out/"raw/shard_ledger.json",shard_ledger);raw_bytes=sum(item["bytes"] for item in shard_ledger)+capture_path().stat().st_size+capture_last_path().stat().st_size
        headline={"status":"qwen_qualified_full_field_closed","rows":n,"qualified_slices":len(qualified),"role_mean_shape":list(mean_states.shape),"role_last_shape":list(last_states.shape),"full_token_shards":len(shard_ledger),"max_width":width,"raw_bytes":raw_bytes,"role_mean_sha256":sha(capture_path()),"role_last_sha256":sha(capture_last_path()),"placement":placement,"quantization":quant,"behavior_correct_rate_in_capture":float(np.mean([r["behavior_correct"] for r in index]))}
    finally:
        for h in hooks: h.remove()
        mean_states.flush();last_states.flush();del mean_states,last_states;parent.previous.model_base().release_bf16(model);gc.collect()
    close("C574",headline,{"rows":headline["rows"]==n and n>0,"role_shapes":headline["role_mean_shape"]==[n,38,6,2560] and headline["role_last_shape"]==[n,38,6,2560],"shards":headline["full_token_shards"]>0,"bf16":headline["quantization"]["has_bf16_parameters"] and not headline["quantization"]["has_quantized_modules"],"index":len(read_rows(capture_index_path()))==n},"C575_observation")


def atomic_prototypes(states: np.ndarray, index: list[dict], part: str = "discovery") -> dict[str, np.ndarray]:
    book: dict[str, np.ndarray] = {}
    for family in ATOMIC_SPECS:
        for domain in ATOMIC_SPECS[family]:
            for surface in SURFACES:
                pairs = [pair for pair in pair_rows(index, "atomic", family, part) if pair[0]["operation_domain"] == domain and pair[0]["surface"] == surface]
                if not pairs:
                    continue
                for q in QPOINTS:
                    response = np.stack([pair_delta(states, pair, q) for pair in pairs])
                    book[f"{family}|{domain}|{surface}|q{q}"] = response.mean(axis=0).astype(np.float32)
    return book


def c575() -> None:
    out = begin("C575", {
        "status": "full_field_observation_frozen", "mode": "observation before mechanism naming",
        "objects": ["signed response", "role-checkpoint-coordinate amplitude", "coordinate quantiles", "emergence profile"],
        "forbidden": ["PCA", "Top-K scientific selection", "magnitude truncation", "attention", "MLP"],
    }, {"parent": final("C574")["all_checks_passed"]})
    index=index_rows(); states=np.load(capture_path(),mmap_mode="r"); book={}; summaries={}
    try:
        book=atomic_prototypes(states,index)
        for key,proto in book.items():
            absolute=np.abs(proto)
            summaries[key]={
                "shape":list(proto.shape), "rms_by_role":[float(np.sqrt(np.mean(row*row))) for row in proto],
                "positive_rate_by_role":[float(np.mean(row>0)) for row in proto],
                "absolute_quantiles_by_role":[[float(value) for value in np.quantile(row,(0,.25,.5,.75,.9,.99,1))] for row in absolute],
                "nonzero_coordinates":int(np.count_nonzero(proto)),
            }
        save_npz(out/"analysis/discovery_atomic_prototypes.npz",book)
    finally:
        close_mmap(states); del states
    family_profiles={}
    for family in ATOMIC_SPECS:
        keys=[key for key in summaries if key.startswith(family+"|")]
        by_q={}
        for q in QPOINTS:
            selected=[summaries[key] for key in keys if key.endswith(f"|q{q}")]
            if selected: by_q[str(q)]=float(np.mean([np.mean(item["rms_by_role"]) for item in selected]))
        family_profiles[family]=by_q
    headline={"status":"full_field_observation_closed","prototype_fields":len(book),"summaries":summaries,"family_checkpoint_rms":family_profiles,"strict_interpretation":"These are registered paired response summaries, not latent variables stored by the model."}
    close("C575",headline,{"prototypes":len(book)>0,"all_full":all(value.shape==(len(ROLES),DIM) for value in book.values()),"finite":finite(summaries)},"C576_forward_prediction")


def load_atomic_book() -> dict[str,np.ndarray]:
    with np.load(OUTS["C575"]/"analysis/discovery_atomic_prototypes.npz",allow_pickle=False) as z:
        return {key:np.asarray(z[key],np.float32) for key in z.files}


def wrong_atomic(book: dict[str,np.ndarray], family: str, domain: str, surface: str, q: int, reference: np.ndarray) -> tuple[str,np.ndarray]:
    candidates=[(key,value) for key,value in book.items() if key.endswith(f"|{surface}|q{q}") and not key.startswith(family+"|")]
    if not candidates: raise RuntimeError((family,domain,surface,q))
    key,value=sorted(candidates,key=lambda item:item[0])[0]
    return key,scaled_like(value,reference)


def c576() -> None:
    out=begin("C576",{
        "status":"fixed_query_atomic_prediction_frozen","train":"discovery units 0-9",
        "tests":["confirmation units 10-13","lockbox units 14-17"],"checkpoints":[24,37],
        "controls":["zero","equal-norm wrong family"],"gate":"correct NRMSE beats both controls by >=0.02",
    },{"parent":final("C575")["all_checks_passed"]})
    index=index_rows(); states=np.load(capture_path(),mmap_mode="r"); book=load_atomic_book(); metrics={}; gates={}
    try:
        for family,domains in ATOMIC_SPECS.items():
            for domain,surface,part,q in itertools.product(domains,SURFACES,("confirmation","lockbox"),(24,37)):
                pairs=[pair for pair in pair_rows(index,"atomic",family,part) if pair[0]["operation_domain"]==domain and pair[0]["surface"]==surface]
                proto_key=f"{family}|{domain}|{surface}|q{q}"
                if not pairs or proto_key not in book: continue
                truth=np.stack([pair_delta(states,pair,q) for pair in pairs]); proto=book[proto_key]
                wrong_key,wrong=wrong_atomic(book,family,domain,surface,q,proto)
                values={"pairs":len(pairs),"correct":metric(np.broadcast_to(proto,truth.shape),truth),"zero":metric(np.zeros_like(truth),truth),"wrong":metric(np.broadcast_to(wrong,truth.shape),truth),"wrong_key":wrong_key}
                key=f"{family}|{domain}|{surface}|{part}|q{q}"; metrics[key]=values
                gates[key]=values["correct"]["nrmse"]<=values["zero"]["nrmse"]-CONTROL_MARGIN and values["correct"]["nrmse"]<=values["wrong"]["nrmse"]-CONTROL_MARGIN
    finally:
        close_mmap(states); del states
    families={}
    for family in ATOMIC_SPECS:
        vals=[value for key,value in gates.items() if key.startswith(family+"|")]
        families[family]={"passed":int(sum(vals)),"total":len(vals),"pass_rate":float(np.mean(vals)) if vals else 0.0,"candidate":bool(vals) and float(np.mean(vals))>=PREDICTION_GATE}
    headline={"status":"fixed_query_atomic_prediction_closed","metrics":metrics,"gates":gates,"family_summary":families,"candidate_families":[key for key,value in families.items() if value["candidate"]],"strict_interpretation":"A passed family is a scope-specific response passport candidate, not a context-free semantic operator."}
    close("C576",headline,{"tests":bool(metrics),"finite":finite(metrics),"complete":len(gates)==len(metrics)},"C577_voice_factorial")


def factorial_groups(index: list[dict], panel: str, part: str) -> list[tuple[tuple,dict[str,dict]]]:
    groups:dict[tuple,dict[str,dict]]=defaultdict(dict)
    for row in index:
        if row["panel"]==panel and row["partition"]==part:
            groups[(row["operation_domain"],row["surface"],row["unit"])][row["cell"]]=row
    return sorted(groups.items())


def voice_effect(states:np.ndarray,cells:dict[str,dict],q:int)->dict[str,np.ndarray]:
    h00=role_bundle(states,cells["f0q0"],q); h01=role_bundle(states,cells["f0q1"],q); h10=role_bundle(states,cells["f1q0"],q); h11=role_bundle(states,cells["f1q1"],q)
    return {"fact":.5*((h10-h00)+(h11-h01)),"query":.5*((h01-h00)+(h11-h10)),"interaction":h11-h10-h01+h00}


def run_factorial_prediction(panel:str,effect_fn,labels:tuple[str,...],cell_count:int)->tuple[dict,dict,dict]:
    index=index_rows(); states=np.load(capture_path(),mmap_mode="r"); metrics={}; gates={}; prototypes={}
    try:
        for domain in sorted({row["operation_domain"] for row in index if row["panel"]==panel}):
            for surface in sorted({row["surface"] for row in index if row["panel"]==panel and row["operation_domain"]==domain}):
                discovery=[cells for (key,cells) in factorial_groups(index,panel,"discovery") if key[0]==domain and key[1]==surface and len(cells)==cell_count]
                for q in (16,24,37):
                    if not discovery: continue
                    responses={label:np.stack([effect_fn(states,cells,q)[label] for cells in discovery]) for label in labels}
                    for label in labels: prototypes[f"{domain}|{surface}|{label}|q{q}"]=responses[label].mean(axis=0)
                for part in ("confirmation","lockbox"):
                    tests=[cells for (key,cells) in factorial_groups(index,panel,part) if key[0]==domain and key[1]==surface and len(cells)==cell_count]
                    for q in (16,24,37):
                        for label in labels:
                            proto=prototypes.get(f"{domain}|{surface}|{label}|q{q}")
                            if proto is None or not tests: continue
                            target=np.stack([effect_fn(states,cells,q)[label] for cells in tests]); other=next(item for item in labels if item!=label); wrong=scaled_like(prototypes[f"{domain}|{surface}|{other}|q{q}"],proto)
                            value={"samples":len(tests),"correct":metric(np.broadcast_to(proto,target.shape),target),"zero":metric(np.zeros_like(target),target),"wrong_effect":metric(np.broadcast_to(wrong,target.shape),target)}
                            key=f"{domain}|{surface}|{label}|{part}|q{q}";metrics[key]=value;gates[key]=value["correct"]["nrmse"]<=value["zero"]["nrmse"]-CONTROL_MARGIN and value["correct"]["nrmse"]<=value["wrong_effect"]["nrmse"]-CONTROL_MARGIN
    finally:
        close_mmap(states);del states
    return metrics,gates,prototypes


def c577() -> None:
    out=begin("C577",{
        "status":"voice_scope_2x2_frozen","cells":["AA","AP","PA","PP"],
        "formulae":{"fact":"0.5*((PA-AA)+(PP-AP))","query":"0.5*((AP-AA)+(PP-PA))","interaction":"PP-PA-AP+AA"},
        "gate":"held-out effect prediction beats zero and equal-norm other effect by >=0.02",
    },{"parent":final("C576")["all_checks_passed"]})
    metrics,gates,protos=run_factorial_prediction("voice_scope_factorial",voice_effect,("fact","query","interaction"),4)
    save_npz(out/"analysis/voice_scope_effect_prototypes.npz",protos)
    by_effect={label:{"passed":int(sum(value for key,value in gates.items() if f"|{label}|" in key)),"total":sum(f"|{label}|" in key for key in gates)} for label in ("fact","query","interaction")}
    for value in by_effect.values(): value["pass_rate"]=value["passed"]/max(value["total"],1)
    headline={"status":"voice_scope_factorial_closed","metrics":metrics,"gates":gates,"effect_summary":by_effect,"pure_fact_voice_candidate":by_effect["fact"]["pass_rate"]>=PREDICTION_GATE,"strict_interpretation":"The 2x2 design separates fact voice, query voice and their interaction; none is named a neural operator without held-out and causal qualification."}
    close("C577",headline,{"tests":bool(metrics),"finite":finite(metrics),"prototypes":bool(protos)},"C578_translation_factorial")


def translation_effect(states:np.ndarray,cells:dict[str,dict],q:int)->dict[str,np.ndarray]:
    h00=role_bundle(states,cells["l0p0"],q);h01=role_bundle(states,cells["l0p1"],q);h10=role_bundle(states,cells["l1p0"],q);h11=role_bundle(states,cells["l1p1"],q)
    return {"language":.5*((h10-h00)+(h11-h01)),"layout":.5*((h01-h00)+(h11-h10)),"interaction":h11-h10-h01+h00}


def c578() -> None:
    out=begin("C578",{
        "status":"translation_language_layout_2x2_frozen","cells":["English/plain","English/list","Chinese/plain","Chinese/list"],
        "query_policy":"English query fixed in all cells","output_policy":"truth and candidate order fixed within unit",
        "gate":"held-out effect prediction beats zero and equal-norm other effect by >=0.02",
    },{"parent":final("C577")["all_checks_passed"]})
    metrics,gates,protos=run_factorial_prediction("translation_layout_factorial",translation_effect,("language","layout","interaction"),4)
    save_npz(out/"analysis/translation_layout_effect_prototypes.npz",protos)
    summary={label:{"passed":int(sum(value for key,value in gates.items() if f"|{label}|" in key)),"total":sum(f"|{label}|" in key for key in gates)} for label in ("language","layout","interaction")}
    for value in summary.values():value["pass_rate"]=value["passed"]/max(value["total"],1)
    headline={"status":"translation_layout_factorial_closed","metrics":metrics,"gates":gates,"effect_summary":summary,"language_specific_candidate":summary["language"]["pass_rate"]>=PREDICTION_GATE,"strict_interpretation":"The language main effect remains a bilingual evidence-interface response, not a universal translation operator."}
    close("C578",headline,{"tests":bool(metrics),"finite":finite(metrics),"prototypes":bool(protos)},"C579_path_depth")


def c579() -> None:
    out=begin("C579",{
        "status":"path_depth_behavior_qualified_response_frozen","object":"direct evidence versus valid two-step evidence with fixed query/truth/output",
        "branches":["behavior-qualified internal route","behavior-unqualified descriptive stop for that slice only"],
    },{"parent":final("C578")["all_checks_passed"]})
    behavior=final("C573")["headline"]["behavior_slices"]; prediction=final("C576")["headline"]["family_summary"]["path_depth"]
    path_slices={key:value for key,value in behavior.items() if "|path_depth|" in key}
    headline={"status":"path_depth_route_closed","behavior_slices":path_slices,"all_behavior_qualified":all(value["qualified"] for value in path_slices.values()),"prediction":prediction,"path_response_candidate":prediction["candidate"],"strict_interpretation":"A path-depth response compares different evidence programs. It is not yet recursive graph composition or transitive reasoning machinery."}
    close("C579",headline,{"three_domains":len(path_slices)==3,"prediction_present":prediction["total"]>0},"C580_composition")


def composition_effect(states:np.ndarray,cells:dict[str,dict],q:int)->dict[str,np.ndarray]:
    h00=role_bundle(states,cells["a0b0"],q);h01=role_bundle(states,cells["a0b1"],q);h10=role_bundle(states,cells["a1b0"],q);h11=role_bundle(states,cells["a1b1"],q)
    return {"factor_a":.5*((h10-h00)+(h11-h01)),"factor_b":.5*((h01-h00)+(h11-h10)),"interaction":h11-h10-h01+h00,"additive_error":h11-(h10+h01-h00)}


def c580() -> None:
    out=begin("C580",{
        "status":"two_composition_panels_frozen","panels":["discourse x fact voice","path depth x evidence paraphrase"],
        "object":"second-order full-coordinate interaction, never cosine-only composition",
        "gate":"discovery interaction predicts confirmation/lockbox and beats zero plus equal-norm wrong panel by >=0.02",
    },{"parent":final("C579")["all_checks_passed"]})
    index=index_rows();states=np.load(capture_path(),mmap_mode="r");metrics={};gates={};protos={}
    panels=("discourse_voice_composition","path_paraphrase_composition")
    try:
        for panel in panels:
            domains=sorted({row["operation_domain"] for row in index if row["panel"]==panel})
            for domain in domains:
                discovery=[cells for (key,cells) in factorial_groups(index,panel,"discovery") if key[0]==domain and len(cells)==4]
                for q in (16,24,37):
                    if discovery:protos[f"{panel}|{domain}|q{q}"]=np.stack([composition_effect(states,cells,q)["interaction"] for cells in discovery]).mean(axis=0)
        for panel in panels:
            wrong_panel=panels[1-panels.index(panel)]
            for domain in sorted({row["operation_domain"] for row in index if row["panel"]==panel}):
                for part,q in itertools.product(("confirmation","lockbox"),(16,24,37)):
                    tests=[cells for (key,cells) in factorial_groups(index,panel,part) if key[0]==domain and len(cells)==4];pk=f"{panel}|{domain}|q{q}"
                    if not tests or pk not in protos:continue
                    target=np.stack([composition_effect(states,cells,q)["interaction"] for cells in tests]);proto=protos[pk]
                    wrong_candidates=[value for key,value in protos.items() if key.startswith(wrong_panel+"|") and key.endswith(f"|q{q}")];wrong=scaled_like(np.mean(wrong_candidates,axis=0),proto)
                    value={"samples":len(tests),"correct":metric(np.broadcast_to(proto,target.shape),target),"zero":metric(np.zeros_like(target),target),"wrong_panel":metric(np.broadcast_to(wrong,target.shape),target)}
                    key=f"{panel}|{domain}|{part}|q{q}";metrics[key]=value;gates[key]=value["correct"]["nrmse"]<=value["zero"]["nrmse"]-CONTROL_MARGIN and value["correct"]["nrmse"]<=value["wrong_panel"]["nrmse"]-CONTROL_MARGIN
    finally:
        close_mmap(states);del states
    save_npz(out/"analysis/composition_interaction_prototypes.npz",protos)
    summary={panel:{"passed":int(sum(value for key,value in gates.items() if key.startswith(panel+"|"))),"total":sum(key.startswith(panel+"|") for key in gates)} for panel in panels}
    for value in summary.values():value["pass_rate"]=value["passed"]/max(value["total"],1);value["candidate"]=value["pass_rate"]>=PREDICTION_GATE
    headline={"status":"composition_panels_closed","metrics":metrics,"gates":gates,"panel_summary":summary,"any_composition_candidate":any(value["candidate"] for value in summary.values()),"strict_interpretation":"A predictable interaction residual is an empirical second-order response law, not yet an associative algebra or reasoning closure."}
    close("C580",headline,{"tests":bool(metrics),"finite":finite(metrics),"both_panels":len(summary)==2},"C581_system_identification")


def fit_coordinate_model(x: np.ndarray, y: np.ndarray, base: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray]:
    # x/y/base: sample x role x coordinate.  Fit each physical coordinate independently.
    features=[x]
    if base is not None: features.append(base*x)
    features.append(np.ones_like(x))
    design=np.stack(features,axis=-1).astype(np.float64)
    target=y.astype(np.float64)
    xtx=np.einsum("nrdk,nrdj->rdkj",design,design)
    xty=np.einsum("nrdk,nrd->rdk",design,target)
    eye=np.eye(design.shape[-1],dtype=np.float64)[None,None]*1e-4
    beta=np.linalg.solve(xtx+eye,xty[...,None])[...,0].astype(np.float32)
    return beta, np.asarray(features[-1][0],np.float32)


def predict_coordinate_model(x:np.ndarray,beta:np.ndarray,base:np.ndarray|None=None)->np.ndarray:
    features=[x]
    if base is not None:features.append(base*x)
    features.append(np.ones_like(x))
    design=np.stack(features,axis=-1)
    return np.einsum("nrdk,rdk->nrd",design,beta).astype(np.float32)


def c581() -> None:
    out=begin("C581",{
        "status":"conditional_system_identification_tournament_frozen",
        "object":"predict later paired response from earlier paired response and optional base HiddenState",
        "models":["identity","discovery target mean","coordinate affine","state-guarded coordinate bilinear"],
        "transitions":["q0->q8","q8->q16","q16->q24","q24->q32","q32->q37"],
        "selection":"no coordinate selection; one explicit coefficient tuple per role-coordinate",
    },{"parent":final("C580")["all_checks_passed"]})
    index=index_rows();states=np.load(capture_path(),mmap_mode="r");metrics={};winners={}
    transitions=((0,8),(8,16),(16,24),(24,32),(32,37))
    try:
        for family in ATOMIC_SPECS:
            train=pair_rows(index,"atomic",family,"discovery");test=pair_rows(index,"atomic",family,"lockbox")
            if not train or not test:continue
            for q0,q1 in transitions:
                xtr=np.stack([pair_delta(states,p,q0) for p in train]);ytr=np.stack([pair_delta(states,p,q1) for p in train]);btr=np.stack([role_bundle(states,p[0],q0) for p in train])
                xte=np.stack([pair_delta(states,p,q0) for p in test]);yte=np.stack([pair_delta(states,p,q1) for p in test]);bte=np.stack([role_bundle(states,p[0],q0) for p in test])
                beta_aff,_=fit_coordinate_model(xtr,ytr);beta_guard,_=fit_coordinate_model(xtr,ytr,btr)
                preds={"identity":xte,"target_mean":np.broadcast_to(ytr.mean(axis=0),yte.shape),"coordinate_affine":predict_coordinate_model(xte,beta_aff),"state_guarded_bilinear":predict_coordinate_model(xte,beta_guard,bte)}
                key=f"{family}|q{q0}->q{q1}";metrics[key]={name:metric(pred,yte) for name,pred in preds.items()};winners[key]=min(preds,key=lambda name:metrics[key][name]["nrmse"])
    finally:
        close_mmap(states);del states
    counts={name:sum(value==name for value in winners.values()) for name in ("identity","target_mean","coordinate_affine","state_guarded_bilinear")}
    headline={"status":"conditional_system_identification_closed","metrics":metrics,"winners":winners,"winner_counts":counts,"simple_dynamic_candidate":counts["coordinate_affine"]+counts["state_guarded_bilinear"]>counts["identity"],"strict_interpretation":"Winning predicts registered response trajectories. It does not identify the unique physical transition law."}
    close("C581",headline,{"tests":bool(metrics),"finite":finite(metrics),"winners":len(winners)==len(metrics)},"C582_equivalence")


def c582() -> None:
    out=begin("C582",{
        "status":"bidirectional_response_equivalence_frozen","checkpoint":37,
        "definition":"two families are candidate-equivalent only if each discovery passport predicts the other's held-out response nearly as well as the target family's own passport and beats zero",
        "tolerance":"cross NRMSE <= own NRMSE + 0.05 in both directions",
    },{"parent":final("C581")["all_checks_passed"]})
    index=index_rows();states=np.load(capture_path(),mmap_mode="r");book=load_atomic_book();matrix={};edges=[]
    try:
        targets={}
        for family in ATOMIC_SPECS:
            pairs=pair_rows(index,"atomic",family,"lockbox")
            if pairs:targets[family]=np.stack([pair_delta(states,p,37) for p in pairs])
        prototypes={family:np.mean([value for key,value in book.items() if key.startswith(family+"|") and key.endswith("|q37")],axis=0) for family in targets}
        for source,source_proto in prototypes.items():
            matrix[source]={}
            for target,target_values in targets.items():
                pred=scaled_like(source_proto,prototypes[target]);matrix[source][target]=metric(np.broadcast_to(pred,target_values.shape),target_values)
        for a,b in itertools.combinations(sorted(targets),2):
            own_a=matrix[a][a]["nrmse"];own_b=matrix[b][b]["nrmse"]
            equivalent=matrix[a][b]["nrmse"]<=own_b+0.05 and matrix[b][a]["nrmse"]<=own_a+0.05 and matrix[a][b]["nrmse"]<1.0 and matrix[b][a]["nrmse"]<1.0
            if equivalent:edges.append({"a":a,"b":b,"a_to_b":matrix[a][b]["nrmse"],"b_to_a":matrix[b][a]["nrmse"]})
    finally:
        close_mmap(states);del states
    headline={"status":"response_equivalence_graph_closed","families":sorted(matrix),"matrix":matrix,"candidate_equivalence_edges":edges,"edge_count":len(edges),"strict_interpretation":"This is a predictive response-equivalence graph under a finite test registry, not proof that the underlying physical states are identical."}
    close("C582",headline,{"matrix":bool(matrix),"finite":finite(matrix),"symmetric_tests":all(set(row)==set(matrix) for row in matrix.values())},"C583_future_signatures")


def c583() -> None:
    out=begin("C583",{
        "status":"future_response_signature_frozen",
        "signature":"concatenate signed role-coordinate responses at q16, q24 and q37",
        "quotient":"finite registered response quotient only; no claim over all possible interventions/readouts",
        "gate":"discovery family signature predicts lockbox signature better than zero and equal-norm wrong family by >=0.02",
    },{"parent":final("C582")["all_checks_passed"]})
    index=index_rows();states=np.load(capture_path(),mmap_mode="r");metrics={};gates={};signatures={}
    try:
        for family in ATOMIC_SPECS:
            train=pair_rows(index,"atomic",family,"discovery");test=pair_rows(index,"atomic",family,"lockbox")
            if not train or not test:continue
            train_sig=np.stack([np.concatenate([pair_delta(states,p,q).reshape(-1) for q in (16,24,37)]) for p in train]);test_sig=np.stack([np.concatenate([pair_delta(states,p,q).reshape(-1) for q in (16,24,37)]) for p in test]);proto=train_sig.mean(axis=0);signatures[family]=proto.astype(np.float32)
            wrong_family=next(name for name in ATOMIC_SPECS if name!=family and pair_rows(index,"atomic",name,"discovery"));wrong_train=pair_rows(index,"atomic",wrong_family,"discovery");wrong_proto=np.stack([np.concatenate([pair_delta(states,p,q).reshape(-1) for q in (16,24,37)]) for p in wrong_train]).mean(axis=0);wrong=scaled_like(wrong_proto,proto)
            value={"samples":len(test),"correct":metric(np.broadcast_to(proto,test_sig.shape),test_sig),"zero":metric(np.zeros_like(test_sig),test_sig),"wrong":metric(np.broadcast_to(wrong,test_sig.shape),test_sig),"wrong_family":wrong_family};metrics[family]=value;gates[family]=value["correct"]["nrmse"]<=value["zero"]["nrmse"]-CONTROL_MARGIN and value["correct"]["nrmse"]<=value["wrong"]["nrmse"]-CONTROL_MARGIN
    finally:
        close_mmap(states);del states
    save_npz(out/"analysis/future_response_signatures.npz",signatures)
    headline={"status":"future_response_signature_closed","metrics":metrics,"gates":gates,"qualified_signatures":[key for key,value in gates.items() if value],"finite_registry_quotient_candidate":sum(gates.values())>=2,"strict_interpretation":"Xi is estimated only over the registered operation/readout set; it cannot establish universal functional equivalence."}
    close("C583",headline,{"tests":bool(metrics),"finite":finite(metrics),"signatures":len(signatures)==len(metrics)},"C584_eligibility")


def c584() -> None:
    out=begin("C584",{
        "status":"causal_eligibility_frozen","policy":"family-level authorization; rejection never stops unrelated routes",
        "requirements":["behavior-qualified slices exist","atomic forward-prediction candidate","future-signature gate","q24 and q37 registered prototypes"],
    },{"parent":final("C583")["all_checks_passed"]})
    behavior=final("C573")["headline"]["behavior_slices"];prediction=final("C576")["headline"]["family_summary"];signatures=final("C583")["headline"]["gates"]
    requirements={};authorized=[]
    for family in ATOMIC_SPECS:
        slices=[value for key,value in behavior.items() if f"|{family}|" in key]
        req={"behavior":bool(slices) and all(value["qualified"] for value in slices),"prediction":prediction.get(family,{}).get("candidate",False),"future_signature":signatures.get(family,False),"prototypes":prediction.get(family,{}).get("total",0)>0}
        requirements[family]=req
        if all(req.values()):authorized.append(family)
    close("C584",{"status":"causal_eligibility_closed","requirements":requirements,"authorized_families":authorized,"authorized_count":len(authorized),"rejected_families":[key for key in ATOMIC_SPECS if key not in authorized]}, {"complete":len(requirements)==len(ATOMIC_SPECS)},"C585_causal_or_na")


def c585() -> None:
    authorized=final("C584")["headline"]["authorized_families"]
    out=begin("C585",{
        "status":"qualified_local_state_guidance_frozen","authorized_families":authorized,
        "intervention":"add discovery q24 role-mean response at registered role-last tokens of one lockbox base per domain",
        "controls":["natural base","equal-norm wrong-family patch","natural transformed target"],
        "readouts":["q37 role-last state distance","A/B output stability"],
        "gate":"correct q37 NRMSE improves over base and wrong by >=0.02; semantic-equivalent output must remain correct",
    },{"parent":final("C584")["all_checks_passed"]})
    if not authorized:
        close("C585",{"status":"causal_registered_na","ran":False,"metrics":{},"causal_families":[],"reason":"no family met all preregistered eligibility conditions"},{"na":True},"C586_cross_model")
        return
    rows=read_rows(material_path());compiled_by_id={row["case_id"]:row for row in read_rows(compiled_path())};index=index_rows();states=np.load(capture_last_path(),mmap_mode="r");model=None;metrics={};family_gates=defaultdict(list)
    try:
        model,tokenizer,device,_=parent.previous.model_base().load_bf16("qwen3")
        last_book={}
        for family in ATOMIC_SPECS:
            for domain in ATOMIC_SPECS[family]:
                pairs=[p for p in pair_rows(index,"atomic",family,"discovery") if p[0]["operation_domain"]==domain and p[0]["surface"]=="record"]
                if pairs:last_book[f"{family}|{domain}"]=np.stack([pair_delta(states,p,24) for p in pairs]).mean(axis=0)
        for family in authorized:
            for domain in ATOMIC_SPECS[family]:
                pairs=[p for p in pair_rows(index,"atomic",family,"lockbox") if p[0]["operation_domain"]==domain and p[0]["surface"]=="record"]
                if not pairs:continue
                left,right=pairs[0];comp=compiled_by_id[left["case_id"]];ids=torch.tensor([comp["prompt_ids"]],dtype=torch.long,device=device);mask=torch.ones_like(ids);pos=torch.arange(ids.shape[1],device=device)[None]
                proto=last_book[f"{family}|{domain}"]
                wrong_family=next(name for name in authorized if name!=family) if len(authorized)>1 else next(name for name in ATOMIC_SPECS if name!=family and any(key.startswith(name+"|") for key in last_book))
                wrong_values=[value for key,value in last_book.items() if key.startswith(wrong_family+"|")];wrong=scaled_like(np.mean(wrong_values,axis=0),proto)
                correct_final,correct_logits=parent.patched_forward(model,ids,mask,pos,comp["role_positions"],proto,24);wrong_final,wrong_logits=parent.patched_forward(model,ids,mask,pos,comp["role_positions"],wrong,24)
                gather=lambda state:np.stack([state[int(comp["role_positions"][role][-1])] for role in ROLES])
                target=role_bundle(states,right,37,"last");base=role_bundle(states,left,37,"last");correct_state=gather(correct_final);wrong_state=gather(wrong_final)
                candidate_ids=comp["candidate_ids"];correct_pred=int(correct_logits[candidate_ids[1][0]]>correct_logits[candidate_ids[0][0]]);wrong_pred=int(wrong_logits[candidate_ids[1][0]]>wrong_logits[candidate_ids[0][0]])
                gold=next(row["gold_position"] for row in rows if row["case_id"]==left["case_id"])
                value={"base":metric(base,target),"correct_patch":metric(correct_state,target),"wrong_patch":metric(wrong_state,target),"correct_output_stable":correct_pred==gold,"wrong_output_stable":wrong_pred==gold,"wrong_family":wrong_family};key=f"{family}|{domain}";metrics[key]=value
                family_gates[family].append(value["correct_patch"]["nrmse"]<=value["base"]["nrmse"]-CONTROL_MARGIN and value["correct_patch"]["nrmse"]<=value["wrong_patch"]["nrmse"]-CONTROL_MARGIN and value["correct_output_stable"])
    finally:
        close_mmap(states);del states;parent.previous.model_base().release_bf16(model);gc.collect()
    causal=[family for family,vals in family_gates.items() if vals and all(vals)]
    headline={"status":"qualified_local_state_guidance_closed","ran":True,"metrics":metrics,"family_gates":dict(family_gates),"causal_families":causal,"strict_interpretation":"Passing is local state sufficiency for this patch compiler on semantically equivalent prompts; it is not necessity, uniqueness, or an output-changing semantic intervention."}
    close("C585",headline,{"tests":bool(metrics),"finite":finite(metrics)},"C586_cross_model")


def c586() -> None:
    out=begin("C586",{
        "status":"sequential_cross_model_topology_frozen","models":["GLM4-9B","DeepSeek-R1-Distill-Qwen-7B"],
        "sequence":"one isolated subprocess at a time; release GPU before next model",
        "subset":"six families, record surface, balanced discovery and lockbox units",
        "comparison":"within-model family confusion, relative checkpoint profile and role topology; never coordinate-number identity",
        "behavior_policy":"capture HiddenState only when model-specific subset behavior >=0.75",
    },{"parent":final("C585")["all_checks_passed"]})
    worker=TESTS/"phase2120_c586_cross_model_scope_program_worker.py";results={}
    for model_name in ("glm4","deepseek7b"):
        result_path=out/f"analysis/{model_name}_worker_result.json"
        completed=subprocess.run([sys.executable,str(worker),"--model",model_name,"--output",str(result_path)],cwd=str(ROOT),capture_output=True,text=True,check=False)
        (out/f"audit/{model_name}_stdout.txt").parent.mkdir(parents=True,exist_ok=True);(out/f"audit/{model_name}_stdout.txt").write_text(completed.stdout,encoding="utf-8");(out/f"audit/{model_name}_stderr.txt").write_text(completed.stderr,encoding="utf-8")
        value=load(result_path) if result_path.exists() else {"status":"worker_failed_without_result"};value["returncode"]=completed.returncode;results[model_name]=value
        if torch.cuda.is_available():torch.cuda.empty_cache()
    candidates=[name for name,value in results.items() if value.get("status")=="closed" and value.get("functional_candidate",False)]
    headline={"status":"sequential_cross_model_branch_closed","models":results,"within_model_candidates":candidates,"cross_model_functional_isomorphism":False,"strict_interpretation":"Model-internal topology can replicate without any shared physical coordinate basis. A cross-model isomorphism requires agreement of registered confusion, role and relative-depth structures."}
    close("C586",headline,{"workers_returned":all(value.get("returncode") in (0,2) for value in results.values()),"sequential_results":len(results)==2,"finite":finite(results)},"C587_nested_flagship")


def nested_effect(states:np.ndarray,cells:dict[str,dict],q:int)->dict[str,np.ndarray]:
    h00=role_bundle(states,cells["o0i0"],q);h01=role_bundle(states,cells["o0i1"],q);h10=role_bundle(states,cells["o1i0"],q);h11=role_bundle(states,cells["o1i1"],q)
    return {"outer":.5*((h10-h00)+(h11-h01)),"inner":.5*((h01-h00)+(h11-h10)),"interaction":h11-h10-h01+h00}


def c587() -> None:
    out=begin("C587",{
        "status":"nested_attitude_event_flagship_frozen","program":"ATTITUDE(experiencer, EVENT(agent, action, patient))",
        "factors":["outer attitude negation","inner event negation"],"domains":["like","regret","remember"],
        "roles":["experiencer","embedded agent","attitude predicate","patient","query","boundary"],
        "gate":"held-out full-coordinate outer/inner/interaction response prediction against zero and wrong effect",
    },{"parent":final("C586")["all_checks_passed"]})
    metrics,gates,protos=run_factorial_prediction("nested_attitude_flagship",nested_effect,("outer","inner","interaction"),4)
    save_npz(out/"analysis/nested_attitude_effect_prototypes.npz",protos)
    behavior={key:value for key,value in final("C573")["headline"]["behavior_slices"].items() if key.startswith("nested_attitude_flagship|")}
    summary={label:{"passed":int(sum(value for key,value in gates.items() if f"|{label}|" in key)),"total":sum(f"|{label}|" in key for key in gates)} for label in ("outer","inner","interaction")}
    for value in summary.values():value["pass_rate"]=value["passed"]/max(value["total"],1)
    headline={"status":"nested_attitude_flagship_closed","behavior_slices":behavior,"metrics":metrics,"gates":gates,"effect_summary":summary,"nested_composition_candidate":summary["interaction"]["pass_rate"]>=PREDICTION_GATE,"strict_interpretation":"The registered English templates test scope-indexed response interactions, not natural-language attitude understanding in general."}
    close("C587",headline,{"behavior":bool(behavior),"finite":finite(metrics),"tests_or_na":bool(metrics) or not all(v["qualified"] for v in behavior.values())},"C588_graph_flagship")


def c588() -> None:
    out=begin("C588",{
        "status":"recursive_knowledge_graph_flagship_frozen","domains":["taxonomy","part_whole","temporal"],
        "axes":["natural versus pseudoword source","depth 1/2/3","direct shortcut absent/present","truth-balanced fixed query"],
        "tests":["depth increment response","shortcut response","natural/pseudoword transfer"],
        "boundary":"transitivity is licensed only where the external relation contract defines it",
    },{"parent":final("C587")["all_checks_passed"]})
    index=index_rows();states=np.load(capture_path(),mmap_mode="r");behavior={key:value for key,value in final("C573")["headline"]["behavior_slices"].items() if key.startswith("recursive_knowledge_flagship|")};metrics={};gates={};protos={}
    try:
        graph=[row for row in index if row["panel"]=="recursive_knowledge_flagship"]
        lookup={(row["operation_domain"],row["unit"],row["factors"]["source_kind"],row["factors"]["depth"],row["factors"]["shortcut"]):row for row in graph}
        for domain,source_kind,effect,q in itertools.product(("taxonomy","part_whole","temporal"),(0,1),("depth12","depth23","shortcut2","shortcut3"),(16,24,37)):
            def response(unit:int)->np.ndarray|None:
                if effect=="depth12":left=lookup.get((domain,unit,source_kind,1,0));right=lookup.get((domain,unit,source_kind,2,0))
                elif effect=="depth23":left=lookup.get((domain,unit,source_kind,2,0));right=lookup.get((domain,unit,source_kind,3,0))
                elif effect=="shortcut2":left=lookup.get((domain,unit,source_kind,2,0));right=lookup.get((domain,unit,source_kind,2,1))
                else:left=lookup.get((domain,unit,source_kind,3,0));right=lookup.get((domain,unit,source_kind,3,1))
                return None if left is None or right is None else role_bundle(states,right,q)-role_bundle(states,left,q)
            train=[response(unit) for unit in range(10)];train=[v for v in train if v is not None]
            if not train:continue
            proto=np.mean(train,axis=0);pk=f"{domain}|s{source_kind}|{effect}|q{q}";protos[pk]=proto
            for part,units in (("confirmation",range(10,14)),("lockbox",range(14,18))):
                test=[response(unit) for unit in units];test=[v for v in test if v is not None]
                if not test:continue
                target=np.stack(test);wrong_source=1-source_kind;wrong_key=f"{domain}|s{wrong_source}|{effect}|q{q}";wrong=scaled_like(protos.get(wrong_key,proto[::-1].copy()),proto)
                value={"samples":len(test),"correct":metric(np.broadcast_to(proto,target.shape),target),"zero":metric(np.zeros_like(target),target),"wrong_source":metric(np.broadcast_to(wrong,target.shape),target)};key=f"{pk}|{part}";metrics[key]=value;gates[key]=value["correct"]["nrmse"]<=value["zero"]["nrmse"]-CONTROL_MARGIN and value["correct"]["nrmse"]<=value["wrong_source"]["nrmse"]-CONTROL_MARGIN
    finally:
        close_mmap(states);del states
    save_npz(out/"analysis/recursive_graph_effect_prototypes.npz",protos)
    summary={effect:{"passed":int(sum(value for key,value in gates.items() if f"|{effect}|" in key)),"total":sum(f"|{effect}|" in key for key in gates)} for effect in ("depth12","depth23","shortcut2","shortcut3")}
    for value in summary.values():value["pass_rate"]=value["passed"]/max(value["total"],1)
    headline={"status":"recursive_knowledge_flagship_closed","behavior_slices":behavior,"metrics":metrics,"gates":gates,"effect_summary":summary,"recursive_response_candidate":summary["depth12"]["pass_rate"]>=PREDICTION_GATE and summary["depth23"]["pass_rate"]>=PREDICTION_GATE,"strict_interpretation":"Depth-response regularity is not proof that pretrained knowledge and in-context graph reasoning share one operator."}
    close("C588",headline,{"behavior":bool(behavior),"finite":finite(metrics),"tests_or_na":bool(metrics) or not all(v["qualified"] for v in behavior.values())},"C589_visual_synthesis")


def register_visual() -> None:
    entry={"id":"c589_scope_program_algebra_atlas","title":"C589 Scope-Program Algebra Full-Coordinate Atlas","phase":2123,"campaign":"C571-C589","path":"vis_data/research_kernel/c589_scope_program_algebra_atlas.json","schema":"ai2050.scope_program_algebra_atlas.v1","description":"Fixed-query, scope-factorial, composition and flagship full-coordinate response fields."}
    if REGISTRY.exists():
        data=load(REGISTRY);container=data.setdefault("datasets",[]) if isinstance(data,dict) else data
        if not any(item.get("id")==entry["id"] for item in container):container.append(entry);save(REGISTRY,data)
    if CATALOG.exists():
        data=load(CATALOG);container=data.setdefault("datasets",[]) if isinstance(data,dict) else data
        if not any(item.get("id")==entry["id"] for item in container):container.append(entry);save(CATALOG,data)


def npz_vectors(path:Path,limit_prefixes:tuple[str,...]|None=None)->dict[str,list]:
    if not path.exists():return {}
    with np.load(path,allow_pickle=False) as z:
        keys=sorted(z.files)
        if limit_prefixes:keys=[key for key in keys if key.startswith(limit_prefixes)]
        return {key:np.asarray(z[key],np.float32).tolist() for key in keys}


def c589() -> None:
    out=begin("C589",{
        "status":"visualization_cleanup_synthesis_frozen","visual":"full 2560-coordinate signed vectors for registered representative roles/checkpoints",
        "retention":"retain the Qwen raw field because it is displayed and needed for unanticipated low-amplitude analyses; remove only temporary or failed-worker partials",
        "theory_name":"Conditional Output Field Closure Theory","organizing_principle":"Reuse-Difference-Conditioning (RDC)",
    },{"parent":final("C588")["all_checks_passed"]})
    atomic=npz_vectors(OUTS["C575"]/"analysis/discovery_atomic_prototypes.npz")
    voice=npz_vectors(OUTS["C577"]/"analysis/voice_scope_effect_prototypes.npz")
    translation=npz_vectors(OUTS["C578"]/"analysis/translation_layout_effect_prototypes.npz")
    composition=npz_vectors(OUTS["C580"]/"analysis/composition_interaction_prototypes.npz")
    nested=npz_vectors(OUTS["C587"]/"analysis/nested_attitude_effect_prototypes.npz")
    graph=npz_vectors(OUTS["C588"]/"analysis/recursive_graph_effect_prototypes.npz")
    cross=final("C586")["headline"]["models"]
    atlas={"schema":"ai2050.scope_program_algebra_atlas.v1","phase":2123,"campaign":"C571-C589","model":"Qwen3-4B plus model-relative GLM4/DeepSeek panels","coordinates":DIM,"roles":list(ROLES),"checkpoints":list(QPOINTS),"coordinate_policy":"all physical coordinates retained; arrays are role x 2560 and never Top-K","panels":{"atomic":atomic,"voice_scope":voice,"translation_layout":translation,"composition":composition,"nested_attitude":nested,"recursive_graph":graph},"cross_model":cross,"behavior":final("C573")["headline"]["behavior_slices"],"prediction":final("C576")["headline"]["family_summary"],"causal":final("C585")["headline"],"warnings":["Response fields are paired experimental constructions, not stored model variables.","Cross-model coordinates are not aligned.","Human blind naturalness was not run."]}
    save(VISUAL,atlas);register_visual()
    removed=[]
    for path in RESULT.glob("phase21*_c5*/**/*.tmp"):
        try:path.unlink();removed.append(str(path.relative_to(ROOT)))
        except OSError:pass
    qwen_bytes=final("C574")["headline"]["raw_bytes"] if capture_path().exists() else 0
    prediction=final("C576")["headline"];composition_result=final("C580")["headline"];system_id=final("C581")["headline"];equivalence=final("C582")["headline"];future=final("C583")["headline"];causal=final("C585")["headline"];cross_result=final("C586")["headline"];nested_result=final("C587")["headline"];graph_result=final("C588")["headline"]
    empirical_gates={
        "stable_object":len(prediction["candidate_families"])>=2,
        "heldout_prediction":bool(prediction["candidate_families"]),
        "dynamic_law":system_id["simple_dynamic_candidate"],
        "composition":composition_result["any_composition_candidate"],
        "causal":bool(causal.get("causal_families",[])),
        "cross_model":cross_result["cross_model_functional_isomorphism"],
    }
    headline={"status":"visualization_cleanup_and_synthesis_closed","visual":str(VISUAL.relative_to(ROOT)),"visual_bytes":VISUAL.stat().st_size,"raw_field_retained":True,"raw_field_bytes":qwen_bytes,"temporary_files_removed":removed,"atomic_candidates":prediction["candidate_families"],"voice_scope":final("C577")["headline"]["effect_summary"],"translation_layout":final("C578")["headline"]["effect_summary"],"composition":composition_result["panel_summary"],"system_identification":system_id["winner_counts"],"response_equivalence_edges":equivalence["candidate_equivalence_edges"],"future_signatures":future["qualified_signatures"],"causal_families":causal.get("causal_families",[]),"cross_model_candidates":cross_result["within_model_candidates"],"nested_candidate":nested_result["nested_composition_candidate"],"recursive_graph_candidate":graph_result["recursive_response_candidate"],"empirical_theory_gates":empirical_gates,"new_foundational_mathematics_authorized":False,"strict_conclusion":"The campaign can establish scope-indexed predictive response laws and local guidance where gates pass. It cannot by itself establish a unique circuit, a universal language algebra, or the need for new foundational mathematics."}
    close("C589",headline,{"visual":VISUAL.exists() and VISUAL.stat().st_size>0,"full_coordinates":DIM==2560,"raw_retained":capture_path().exists(),"finite":finite(headline)},"C590_independent_audit")


FUNCTIONS={name:globals()[name.lower()] for name in PHASES}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--start",choices=list(PHASES),default="C571");parser.add_argument("--stop",choices=list(PHASES),default="C589");args=parser.parse_args()
    names=list(PHASES);start=names.index(args.start);stop=names.index(args.stop)
    if stop<start:raise SystemExit("--stop precedes --start")
    for name in names[start:stop+1]:
        print(f"\n=== {name} / Phase {PHASES[name][0]} ===",flush=True);FUNCTIONS[name]()


if __name__=="__main__":
    main()
