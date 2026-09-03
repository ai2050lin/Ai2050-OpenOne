#!/usr/bin/env python3
"""C600-C605 language-program transport and output-boundary campaign.

The scientific interface is token embeddings plus HiddenState checkpoints.
Attention, MLP internals, PCA, Top-K selection, and magnitude truncation are
not used.  Every retained response keeps every signed physical coordinate.
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c605_language_transport_output_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
sys.path.insert(0, str(TESTS))

import model_utils
import phase1331_relational_measurement_core as text_core
import phase1797_c263_c272_state_operator_common as compiler
import phase2076_c542_c559_typed_operation_response_passport_campaign as passport
import phase2125_c591_c598_fresh_scope_lockbox_campaign as previous


PHASES = {
    "C600": (2134, "evidence_adjudication_language_program_and_output_contract"),
    "C601": (2135, "qwen_behavior_and_all_token_all_coordinate_observation"),
    "C602": (2136, "full_coordinate_state_guard_system_identification"),
    "C603": (2137, "sequential_transport_and_flagship_composition"),
    "C604": (2138, "bidirectional_output_deletion_and_rescue"),
    "C605": (2139, "cross_model_functional_topology_visualization_and_theory"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower()}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

DIM = 2560
CHECKPOINTS = 38
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
QPOINTS = (0, 8, 16, 24, 32, 37)
SURFACES = ("ledger", "briefing")
UNITS = 16
CONTROL_MARGIN = 0.02
BEHAVIOR_GATE = 0.75
OPEN_GATE = 0.50
PREDICTION_GATE = 0.75
SYSTEM = "Use only the supplied statements. Give exactly the requested answer and no explanation."
COLORS = ("red", "blue", "green", "black")
MULTI_COLORS = ("dark red", "light blue", "bright green", "deep black")

NAMES_A = (
    "Arlen", "Beatrix", "Corwin", "Delia", "Evander", "Freya", "Gareth", "Helena",
    "Isolde", "Jasper", "Kendra", "Leander", "Marina", "Neville", "Oriana", "Quentin",
)
NAMES_B = (
    "Rowena", "Silas", "Tabitha", "Ulric", "Verena", "Wesley", "Xenia", "Yvette",
    "Zelda", "Bram", "Clara", "Derek", "Estelle", "Felix", "Greer", "Hector",
)
OBJECTS = (
    "compass", "lantern", "vase", "notebook", "satchel", "medallion", "goblet", "banner",
    "parcel", "tablet", "flask", "helmet", "ribbon", "mirror", "casket", "scroll",
)
NOISES = (
    "harbor", "meadow", "tower", "gallery", "orchard", "library", "station", "courtyard",
    "workshop", "observatory", "market", "archive", "garden", "museum", "theater", "laboratory",
)

ATOMIC_FAMILIES = (
    "evidence_order",
    "fact_voice",
    "evidence_paraphrase",
    "double_negation",
    "clause_packaging",
    "path_depth",
    "translation_surface",
    "relation_lexicalization",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def begin(name: str, protocol: dict, checks: dict) -> Path:
    out = OUTS[name]
    out.mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name, "timestamp_utc": now(),
        "protocol": protocol, "parent_checks": checks,
    })
    return out


def close(name: str, headline: dict, checks: dict, authorization: str) -> dict:
    out = OUTS[name]
    all_passed = bool(checks) and all(bool(v) for v in checks.values())
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": now(), "all_checks_passed": all_passed,
        "headline": headline, "checks": checks, "next_authorization": authorization,
    }
    save(out / "analysis/final.json", result)
    save(out / "audit/checks.json", {"checks": checks, "all_checks_passed": all_passed})
    return result


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def material_path() -> Path:
    return OUTS["C600"] / "material/language_transport_output_material.jsonl"


def compiled_path() -> Path:
    return OUTS["C600"] / "compiled/qwen3_language_transport_output.jsonl"


def behavior_path() -> Path:
    return OUTS["C601"] / "behavior/qwen3_behavior.jsonl"


def qualified_path() -> Path:
    return OUTS["C601"] / "behavior/qualified_slices.json"


def capture_index_path() -> Path:
    return OUTS["C601"] / "raw/hidden_index.jsonl"


def mean_path() -> Path:
    return OUTS["C601"] / "raw/role_mean.float16.npy"


def last_path() -> Path:
    return OUTS["C601"] / "raw/role_last.float16.npy"


def shard_dir() -> Path:
    return OUTS["C601"] / "raw/full_token_shards"


def partition(unit: int) -> str:
    if unit < 8:
        return "discovery"
    if unit < 12:
        return "confirmation"
    return "lockbox"


def wrap(surface: str, facts: list[str], question: str, answer_instruction: str) -> str:
    if surface == "ledger":
        body = " ".join(facts)
        return f"Archive ledger: {body} Query: {question} {answer_instruction}"
    if surface == "briefing":
        body = " ".join(f"Statement {i + 1}: {fact}" for i, fact in enumerate(facts))
        return f"A coordinator gives this briefing. {body} Question: {question} {answer_instruction}"
    raise KeyError(surface)


def row(
    *, case_id: str, panel: str, family: str, domain: str, surface: str, unit: int,
    cell: str, facts: list[str], question: str, answer: str, candidates: list[str],
    role_values: dict[str, str], factors: dict[str, Any], operation: dict[str, Any],
) -> dict:
    instruction = "Answer with exactly one word:" if " " not in answer else "Answer with exactly two words:"
    prompt = wrap(surface, facts, question, instruction)
    return {
        "case_id": case_id, "panel": panel, "family": family, "operation_type": family,
        "operation_domain": domain, "domain_id": f"{panel}:{family}:{domain}",
        "surface": surface, "unit": unit, "cell": cell, "partition": partition(unit),
        "facts": facts, "question": question, "prompt": prompt, "answer": answer,
        "answer_candidates": candidates, "role_values": role_values, "factors": factors,
        "language_program": operation, "human_naturalness": "NA_not_run",
    }


def unit_values(unit: int) -> tuple[str, str, str, str, str, str]:
    a, b, obj = NAMES_A[unit], NAMES_B[unit], OBJECTS[unit]
    c0, c1 = COLORS[unit % 4], COLORS[(unit + 1) % 4]
    c2 = COLORS[(unit + 2) % 4]
    return a, b, obj, c0, c1, c2


def atomic_case(family: str, surface: str, unit: int, variant: int) -> dict:
    a, b, obj, c0, c1, c2 = unit_values(unit)
    noise = NOISES[unit]
    relation = "marked"
    if family == "evidence_order":
        facts = [f"{a}'s {obj} is marked {c0}.", f"{b} visited the {noise}."]
        if variant:
            facts.reverse()
        question, answer = f"What color marks {a}'s {obj}?", c0
        relation = "marked"
    elif family == "fact_voice":
        facts = ([f"{a} assigned the {c0} seal to {b}."] if not variant
                 else [f"The {c0} seal was assigned to {b} by {a}."])
        question, answer, relation = "What color was the assigned seal?", c0, "assigned"
    elif family == "evidence_paraphrase":
        facts = ([f"The label on {a}'s {obj} is {c0}."] if not variant
                 else [f"{a}'s {obj} bears a {c0} label."])
        question, answer, relation = f"What is the label color on {a}'s {obj}?", c0, "label"
    elif family == "double_negation":
        facts = ([f"{a}'s {obj} is {c0}."] if not variant
                 else [f"It is not the case that {a}'s {obj} is not {c0}."])
        question, answer, relation = f"What color is {a}'s {obj}?", c0, "color"
    elif family == "clause_packaging":
        facts = ([f"{a}'s {obj} is {c0}; meanwhile {b}'s {obj} is {c1}."] if not variant
                 else [f"{a}'s {obj}: {c0}.", f"{b}'s {obj}: {c1}."])
        question, answer, relation = f"What color belongs to {a}'s {obj}?", c0, "belongs"
    elif family == "path_depth":
        facts = ([f"The terminal code for {a} is {c2}."] if not variant
                 else [f"{a} maps to {c0}.", f"{c0} maps to {c2}."])
        question = f"What terminal code is reached from {a}?"
        answer, relation = c2, ("terminal code" if not variant else "maps")
    elif family == "translation_surface":
        facts = ([f"The seal for {a} is {c0}."] if not variant
                 else [f"El sello de {a} es {c0}."])
        question = (f"What color is {a}'s seal?" if not variant else f"¿De qué color es el sello de {a}?")
        answer, relation = c0, ("seal" if not variant else "sello")
    elif family == "relation_lexicalization":
        facts = ([f"{a} has a {c0} marker."] if not variant
                 else [f"A {c0} marker belongs to {a}."])
        question, answer, relation = f"What marker color is linked to {a}?", c0, "marker"
    else:
        raise KeyError(family)
    return row(
        case_id=f"c600-atomic-{family}-{surface}-u{unit:02d}-v{variant}", panel="atomic",
        family=family, domain="mixed", surface=surface, unit=unit, cell=f"v{variant}",
        facts=facts, question=question, answer=answer, candidates=list(COLORS),
        role_values={"primary": a, "secondary": b if b in " ".join(facts + [question]) else a,
                     "relation": relation, "context": answer, "query": a},
        factors={"variant": variant},
        operation={"input_type": "record", "output_type": "open_color", "scope": family,
                   "invariants": ["referent", "answer", "output_protocol"],
                   "changed": [family], "operation": family},
    )


def factorial_case(surface: str, domain: str, unit: int, query_switch: int, value_swap: int) -> dict:
    a, b, obj, c0, c1, _ = unit_values(unit)
    left, right = (c1, c0) if value_swap else (c0, c1)
    query_name = b if query_switch else a
    answer = right if query_switch else left
    noun = {"badge": "badge", "locker": "locker tag", "route": "route marker"}[domain]
    facts = [f"{a}'s {noun} is {left}.", f"{b}'s {noun} is {right}."]
    question = f"What color is {query_name}'s {noun}?"
    return row(
        case_id=f"c600-factorial-{domain}-{surface}-u{unit:02d}-q{query_switch}v{value_swap}",
        panel="readout_factorial", family="query_value_factorial", domain=domain,
        surface=surface, unit=unit, cell=f"q{query_switch}v{value_swap}", facts=facts,
        question=question, answer=answer, candidates=list(COLORS),
        role_values={"primary": a, "secondary": b, "relation": noun, "context": answer, "query": query_name},
        factors={"query_switch": query_switch, "value_swap": value_swap},
        operation={"input_type": "two_record_map", "output_type": "open_color", "scope": "fact_or_query",
                   "invariants": ["entities", "candidate_vocabulary"],
                   "changed": ["query_entity" if query_switch else "none", "value_assignment" if value_swap else "none"]},
    )


def sequence_case(surface: str, domain: str, unit: int, cell: str) -> dict:
    a, b, obj, c0, c1, c2 = unit_values(unit)
    noun = "status light" if domain == "status" else "access flag"
    mapping = {a: c0, b: c1}
    query_name = a
    if cell == "A":
        query_name = b
    elif cell == "B":
        mapping[a] = c2
    elif cell == "AB":
        query_name = b
        mapping[b] = c2
    elif cell == "BA":
        mapping[a] = c2
        query_name = b
    elif cell != "S0":
        raise KeyError(cell)
    facts = [f"{a}'s {noun} is {mapping[a]}.", f"{b}'s {noun} is {mapping[b]}."]
    question = f"What color is {query_name}'s {noun}?"
    return row(
        case_id=f"c600-sequence-{domain}-{surface}-u{unit:02d}-{cell}", panel="sequence_program",
        family="query_then_contextual_overwrite", domain=domain, surface=surface, unit=unit,
        cell=cell, facts=facts, question=question, answer=mapping[query_name], candidates=list(COLORS),
        role_values={"primary": a, "secondary": b, "relation": noun,
                     "context": mapping[query_name], "query": query_name},
        factors={"stage": cell},
        operation={"input_type": "stateful_map", "output_type": "open_color", "scope": "current_query",
                   "program": {"S0": [], "A": ["query_switch"], "B": ["overwrite_current"],
                               "AB": ["query_switch", "overwrite_current"],
                               "BA": ["overwrite_current", "query_switch"]}[cell]},
    )


def multiword_case(surface: str, unit: int, query_switch: int) -> dict:
    a, b, obj, _, _, _ = unit_values(unit)
    left, right = MULTI_COLORS[unit % 4], MULTI_COLORS[(unit + 1) % 4]
    query_name = b if query_switch else a
    answer = right if query_switch else left
    facts = [f"{a}'s signal is {left}.", f"{b}'s signal is {right}."]
    return row(
        case_id=f"c600-multiword-{surface}-u{unit:02d}-q{query_switch}", panel="multiword_readout",
        family="multiword_query_switch", domain="signal", surface=surface, unit=unit,
        cell=f"q{query_switch}", facts=facts, question=f"What is {query_name}'s signal color?",
        answer=answer, candidates=list(MULTI_COLORS),
        role_values={"primary": a, "secondary": b, "relation": "signal", "context": answer, "query": query_name},
        factors={"query_switch": query_switch},
        operation={"input_type": "two_record_map", "output_type": "open_two_token_color",
                   "scope": "query", "changed": ["query_entity"] if query_switch else []},
    )


def attitude_case(surface: str, unit: int, outer_neg: int, inner_neg: int, query_scope: int) -> dict:
    a, b, obj, c0, _, _ = unit_values(unit)
    inner = f"{b} carries the {c0} {obj}"
    if inner_neg:
        inner = f"{b} does not carry the {c0} {obj}"
    attitude = f"{a} likes the report that {inner}."
    if outer_neg:
        attitude = f"{a} does not like the report that {inner}."
    facts = [attitude]
    if query_scope:
        question = f"According to the report, does {b} carry the {c0} {obj}?"
        answer = "No" if inner_neg else "Yes"
        query = b
    else:
        question = f"Does {a} like the report?"
        answer = "No" if outer_neg else "Yes"
        query = a
    return row(
        case_id=f"c600-attitude-{surface}-u{unit:02d}-o{outer_neg}i{inner_neg}q{query_scope}",
        panel="attitude_flagship", family="attitude_event_scope", domain="like_carry",
        surface=surface, unit=unit, cell=f"o{outer_neg}i{inner_neg}q{query_scope}", facts=facts,
        question=question, answer=answer, candidates=["Yes", "No"],
        role_values={"primary": a, "secondary": b, "relation": "report", "context": obj, "query": query},
        factors={"outer_neg": outer_neg, "inner_neg": inner_neg, "query_scope": query_scope},
        operation={"input_type": "nested_attitude_event", "output_type": "binary_word",
                   "scope": "inner" if query_scope else "outer",
                   "changed": ["outer_negation" if outer_neg else "none", "inner_negation" if inner_neg else "none"]},
    )


def graph_case(surface: str, unit: int, graph_type: str, cell: str) -> dict:
    a = NAMES_A[unit]
    if graph_type == "natural":
        start, n1, n2, n3 = "apple", "fruit", "food", "object"
        candidates = ["fruit", "food", "object", "plant"]
    else:
        start, n1, n2, n3 = a, "red", "blue", "green"
        candidates = list(COLORS)
    if cell == "direct":
        facts, answer, hops = [f"{start} is a {n1}."], n1, 1
    elif cell == "two":
        facts, answer, hops = [f"{start} is a {n1}.", f"{n1} is a {n2}."], n2, 2
    elif cell == "three":
        facts, answer, hops = [f"{start} is a {n1}.", f"{n1} is a {n2}.", f"{n2} is an {n3}."], n3, 3
    elif cell == "shortcut":
        facts, answer, hops = [f"{start} is a {n1}.", f"{n1} is a {n2}.", f"{start} is a {n2}."], n2, 2
    elif cell == "wrong_middle":
        facts, answer, hops = [f"{start} is a {n1}.", f"stone is a {n2}."], n1, 1
    elif cell == "reverse":
        facts, answer, hops = [f"{n1} includes {start}."], n1, 1
    else:
        raise KeyError(cell)
    if graph_type == "natural":
        facts.append(f"Separately, {a} catalogued a {OBJECTS[unit]} near the {NOISES[unit]}.")
    question = f"What category is reached from {start} after {hops} valid link{'s' if hops != 1 else ''}?"
    return row(
        case_id=f"c600-graph-{graph_type}-{surface}-u{unit:02d}-{cell}", panel="knowledge_graph_flagship",
        family="typed_path_program", domain=graph_type, surface=surface, unit=unit, cell=cell,
        facts=facts, question=question, answer=answer, candidates=candidates,
        role_values={"primary": start, "secondary": n1, "relation": ("includes" if cell == "reverse" else "is"),
                     "context": answer, "query": start},
        factors={"path_cell": cell, "hops": hops, "graph_type": graph_type},
        operation={"input_type": "typed_graph", "output_type": "open_category", "scope": "path",
                   "path_cell": cell, "hops": hops, "changed": ["path_structure"]},
    )


def make_material() -> list[dict]:
    rows: list[dict] = []
    for family, surface, unit, variant in itertools.product(ATOMIC_FAMILIES, SURFACES, range(UNITS), (0, 1)):
        rows.append(atomic_case(family, surface, unit, variant))
    for domain, surface, unit, q, v in itertools.product(("badge", "locker", "route"), SURFACES, range(UNITS), (0, 1), (0, 1)):
        rows.append(factorial_case(surface, domain, unit, q, v))
    for domain, surface, unit, cell in itertools.product(("status", "access"), SURFACES, range(UNITS), ("S0", "A", "B", "AB", "BA")):
        rows.append(sequence_case(surface, domain, unit, cell))
    for surface, unit, query_switch in itertools.product(SURFACES, range(UNITS), (0, 1)):
        rows.append(multiword_case(surface, unit, query_switch))
    for surface, unit, outer, inner, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1), (0, 1), (0, 1)):
        rows.append(attitude_case(surface, unit, outer, inner, query_scope))
    for graph_type, surface, unit, cell in itertools.product(("natural", "pseudo"), SURFACES, range(UNITS), ("direct", "two", "three", "shortcut", "wrong_middle", "reverse")):
        rows.append(graph_case(surface, unit, graph_type, cell))
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    compiled = []
    for source in rows:
        ids = text_core.chat_ids(tokenizer, SYSTEM, source["prompt"])
        candidate_ids = [tokenizer.encode(" " + answer, add_special_tokens=False) for answer in source["answer_candidates"]]
        if not all(candidate_ids):
            raise RuntimeError((source["case_id"], candidate_ids))
        positions = {}
        for role, value in source["role_values"].items():
            spans = compiler.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((source["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        gold = source["answer_candidates"].index(source["answer"])
        compiled.append({**source, "prompt_ids": ids, "candidate_ids": candidate_ids,
                         "gold_position": gold, "role_positions": positions})
    return compiled


def candidate_sequence_scores(model, device, prompt_ids: list[int], candidates: list[list[int]]) -> list[float]:
    rows = []
    for candidate in candidates:
        seq = prompt_ids + candidate
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        pos = torch.arange(len(seq), device=device)[None]
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
        score = 0.0
        for offset, token in enumerate(candidate):
            at = len(prompt_ids) - 1 + offset
            score += float(torch.log_softmax(logits[0, at].float(), dim=-1)[token])
        rows.append(score / len(candidate))
    return rows


def c600() -> None:
    out = begin("C600", {
        "status": "language_program_output_master_contract_frozen",
        "evidence_adjudication": [
            "C571-C599 support typed scope-conditioned full-coordinate response transfer",
            "state guidance is not output control",
            "ordinary interaction residuals are not curvature",
            "no foundational mathematics is authorized",
        ],
        "panels": ["eight atomic families", "open readout 2x2", "true sequential programs",
                   "multi-token output", "nested attitude", "natural and pseudo knowledge paths"],
        "partitions": "unit-disjoint discovery 0-7, confirmation 8-11, lockbox 12-15",
        "behavior_before_hiddenstate": True,
        "human_naturalness": "NA_not_run; machine semantic ledger only",
        "coordinate_policy": "all signed coordinates; no PCA, Top-K, or magnitude threshold",
        "route_policy": "failure closes one branch only",
    }, {"parent_c599": previous.load(previous.OUTS["C598"] / "analysis/final.json")["all_checks_passed"]})
    rows = make_material()
    write_rows(material_path(), rows)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    compiled = compile_rows(tokenizer, rows)
    write_rows(compiled_path(), compiled)
    old_prompts = set()
    if previous.material_path().exists():
        old_prompts = {r["prompt"] for r in previous.read_rows(previous.material_path())}
    prompt_groups = defaultdict(list)
    for item in rows:
        prompt_groups[item["prompt"]].append(item)
    cross_partition_duplicates = [key for key, values in prompt_groups.items() if len({v["partition"] for v in values}) > 1]
    answer_counts = defaultdict(int)
    for item in rows:
        answer_counts[item["answer"]] += 1
    token_lengths = defaultdict(set)
    for item in compiled:
        for answer, ids in zip(item["answer_candidates"], item["candidate_ids"]):
            token_lengths[answer].add(len(ids))
    headline = {
        "status": "language_program_output_material_closed", "rows": len(rows),
        "unique_case_ids": len({r["case_id"] for r in rows}), "unique_prompts": len(prompt_groups),
        "panel_counts": {p: sum(r["panel"] == p for r in rows) for p in sorted({r["panel"] for r in rows})},
        "family_counts": {p: sum(r["family"] == p for r in rows) for p in sorted({r["family"] for r in rows})},
        "partition_counts": {p: sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")},
        "answer_counts": dict(answer_counts), "candidate_token_lengths": {k: sorted(v) for k, v in token_lengths.items()},
        "old_prompt_overlap": sum(r["prompt"] in old_prompts for r in rows),
        "cross_partition_duplicate_prompts": len(cross_partition_duplicates),
        "max_prompt_tokens": max(len(r["prompt_ids"]) for r in compiled),
        "examples": {p: next(r["prompt"] for r in rows if r["panel"] == p) for p in sorted({r["panel"] for r in rows})},
        "human_naturalness": "NA_not_run",
    }
    close("C600", headline, {
        "rows": len(rows) >= 1500, "case_unique": len({r["case_id"] for r in rows}) == len(rows),
        "compiled": len(compiled) == len(rows), "partition_isolation": not cross_partition_duplicates,
        "fresh_prompts": not any(r["prompt"] in old_prompts for r in rows),
        "multi_token_present": any(len(ids) > 1 for r in compiled for ids in r["candidate_ids"]),
        "program_breadth": len({r["family"] for r in rows}) >= 13,
    }, "C601_qwen_behavior_and_observation")


def c601() -> None:
    out = begin("C601", {
        "status": "qwen_behavior_and_full_field_frozen", "model": "Qwen3-4B BF16 CUDA",
        "behavior": "candidate sequence likelihood plus unrestricted next-token exact match",
        "qualification": "family-domain candidate accuracy >=0.75; unrestricted output separately reported",
        "capture": "qualified slices only; embedding, every post-block checkpoint, final norm, every token and coordinate",
        "storage": "16-row bounded float16 shards plus all-coordinate role mean and last tensors",
    }, {"parent": final("C600")["all_checks_passed"], "cuda": torch.cuda.is_available()})
    rows = read_rows(material_path())
    compiled = read_rows(compiled_path())
    model = None
    behavior: list[dict] = []
    try:
        model, tokenizer, device, placement = passport.previous.model_base().load_bf16("qwen3")
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        single = [item for item in compiled if all(len(ids) == 1 for ids in item["candidate_ids"])]
        for start in range(0, len(single), 12):
            batch = single[start:start + 12]
            width = max(len(r["prompt_ids"]) for r in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, item in enumerate(batch):
                seq = item["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, device=device)
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            for i, item in enumerate(batch):
                at = len(item["prompt_ids"]) - 1
                scores = [float(logits[i, at, candidate[0]]) for candidate in item["candidate_ids"]]
                pred = int(np.argmax(scores))
                open_token = int(torch.argmax(logits[i, at]).item())
                gold_token = int(item["candidate_ids"][item["gold_position"]][0])
                behavior.append({"case_id": item["case_id"], "panel": item["panel"], "family": item["family"],
                                 "operation_domain": item["operation_domain"], "unit": item["unit"],
                                 "partition": item["partition"], "candidate_prediction": pred,
                                 "candidate_correct": pred == item["gold_position"], "open_first_token": open_token,
                                 "open_correct": open_token == gold_token, "candidate_scores": scores})
            if start % 480 == 0 or start + len(batch) == len(single):
                print(f"[C601 behavior one-token] {min(start + len(batch), len(single))}/{len(single)}", flush=True)
        multi = [item for item in compiled if any(len(ids) > 1 for ids in item["candidate_ids"])]
        for i, item in enumerate(multi):
            scores = candidate_sequence_scores(model, device, item["prompt_ids"], item["candidate_ids"])
            pred = int(np.argmax(scores))
            behavior.append({"case_id": item["case_id"], "panel": item["panel"], "family": item["family"],
                             "operation_domain": item["operation_domain"], "unit": item["unit"],
                             "partition": item["partition"], "candidate_prediction": pred,
                             "candidate_correct": pred == item["gold_position"], "open_first_token": None,
                             "open_correct": False, "candidate_scores": scores, "multi_token_scored": True})
            if i % 16 == 0 or i + 1 == len(multi):
                print(f"[C601 behavior multi-token] {i + 1}/{len(multi)}", flush=True)
    finally:
        passport.previous.model_base().release_bf16(model)
        gc.collect()
    behavior.sort(key=lambda item: item["case_id"])
    write_rows(behavior_path(), behavior)
    grouped = defaultdict(list)
    for item in behavior:
        grouped[(item["panel"], item["family"], item["operation_domain"])].append(item)
    slices = {}
    for key, values in sorted(grouped.items()):
        candidate_acc = float(np.mean([v["candidate_correct"] for v in values]))
        one_token = [v for v in values if v["open_first_token"] is not None]
        open_acc = float(np.mean([v["open_correct"] for v in one_token])) if one_token else None
        slices["|".join(key)] = {"rows": len(values), "candidate_accuracy": candidate_acc,
                                  "open_first_token_accuracy": open_acc,
                                  "qualified": candidate_acc >= BEHAVIOR_GATE}
    qualified = sorted(key for key, value in slices.items() if value["qualified"])
    save(qualified_path(), {"gate": BEHAVIOR_GATE, "open_gate": OPEN_GATE, "qualified": qualified, "slices": slices})

    by_case = {r["case_id"]: r for r in behavior}
    selected = [(r, c) for r, c in zip(rows, compiled)
                if f"{r['panel']}|{r['family']}|{r['operation_domain']}" in qualified]
    selected.sort(key=lambda item: len(item[1]["prompt_ids"]))
    n = len(selected)
    estimated = sum(CHECKPOINTS * len(c["prompt_ids"]) * DIM * 2 for _, c in selected)
    estimated += 2 * n * CHECKPOINTS * len(ROLES) * DIM * 2
    free_before = shutil.disk_usage(RESULT).free
    if free_before < estimated + (10 << 30):
        raise RuntimeError({"free": free_before, "estimated": estimated, "headroom": 10 << 30})
    shard_dir().mkdir(parents=True, exist_ok=True)
    mean_states = np.lib.format.open_memmap(mean_path(), mode="w+", dtype=np.float16,
                                            shape=(n, CHECKPOINTS, len(ROLES), DIM))
    last_states = np.lib.format.open_memmap(last_path(), mode="w+", dtype=np.float16,
                                            shape=(n, CHECKPOINTS, len(ROLES), DIM))
    model = None
    hooks = []
    captured: list[torch.Tensor] = []
    index: list[dict] = []
    ledger: list[dict] = []
    capture_headline = {}
    try:
        model, tokenizer, device, placement_capture = passport.previous.model_base().load_bf16("qwen3")
        quant = passport.previous.model_base().quantization_audit(model)
        base = model.model
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for shard_start in range(0, n, 16):
            items = selected[shard_start:shard_start + 16]
            width = max(len(c["prompt_ids"]) for _, c in items)
            shard_path = shard_dir() / f"shard_{shard_start // 16:04d}.float16.npy"
            shard = np.lib.format.open_memmap(shard_path, mode="w+", dtype=np.float16,
                                              shape=(len(items), CHECKPOINTS, width, DIM))
            for local_start in range(0, len(items), 4):
                batch = items[local_start:local_start + 4]
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                weights = torch.zeros((len(batch), len(ROLES), width), dtype=torch.float32, device=device)
                last_pos = torch.zeros((len(batch), len(ROLES)), dtype=torch.long, device=device)
                lengths = []
                for i, (_source, item) in enumerate(batch):
                    seq = item["prompt_ids"]
                    lengths.append(len(seq))
                    ids[i, :len(seq)] = torch.tensor(seq, device=device)
                    mask[i, :len(seq)] = 1
                    for role_i, role in enumerate(ROLES):
                        points = [int(v) for v in item["role_positions"][role]]
                        weights[i, role_i, points] = 1.0 / len(points)
                        last_pos[i, role_i] = points[-1]
                pos = mask.long().cumsum(-1) - 1
                pos.masked_fill_(mask == 0, 0)
                captured.clear()
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                if len(captured) != CHECKPOINTS:
                    raise RuntimeError((len(captured), CHECKPOINTS))
                target_slice = slice(shard_start + local_start, shard_start + local_start + len(batch))
                for q, state in enumerate(captured):
                    state32 = state.float()
                    mean_states[target_slice, q] = torch.einsum("brt,btd->brd", weights, state32).cpu().numpy().astype(np.float16)
                    gather = last_pos[:, :, None].expand(-1, -1, DIM)
                    last_states[target_slice, q] = torch.gather(state32, 1, gather).cpu().numpy().astype(np.float16)
                    for i, length in enumerate(lengths):
                        shard[local_start + i, q, :length] = state[i, :length].float().cpu().numpy().astype(np.float16)
                for i, (source, item) in enumerate(batch):
                    index.append({"hidden_index": shard_start + local_start + i,
                                  "shard": shard_path.name, "shard_index": local_start + i,
                                  "case_id": source["case_id"], "panel": source["panel"],
                                  "family": source["family"], "operation_domain": source["operation_domain"],
                                  "surface": source["surface"], "unit": source["unit"],
                                  "partition": source["partition"], "cell": source["cell"],
                                  "factors": source["factors"], "answer": source["answer"],
                                  "length": lengths[i], "role_positions": item["role_positions"],
                                  "behavior_correct": by_case[source["case_id"]]["candidate_correct"],
                                  "open_correct": by_case[source["case_id"]]["open_correct"]})
            shard.flush(); del shard
            mean_states.flush(); last_states.flush()
            ledger.append({"shard": shard_path.name, "rows": len(items), "width": width, "bytes": shard_path.stat().st_size})
            print(f"[C601 capture] {min(shard_start + len(items), n)}/{n}", flush=True)
        write_rows(capture_index_path(), index)
        save(out / "raw/shard_ledger.json", ledger)
        capture_headline = {"rows": n, "role_mean_shape": list(mean_states.shape), "role_last_shape": list(last_states.shape),
                            "full_token_shards": len(ledger), "raw_bytes": sum(v["bytes"] for v in ledger) + mean_path().stat().st_size + last_path().stat().st_size,
                            "estimated_bytes": estimated, "free_disk_before": free_before,
                            "placement": placement_capture, "quantization": quant}
    finally:
        for handle in hooks:
            handle.remove()
        mean_states.flush(); last_states.flush(); del mean_states, last_states
        passport.previous.model_base().release_bf16(model)
        gc.collect()
    headline = {
        "status": "qwen_behavior_and_full_field_closed", "material_rows": len(rows),
        "behavior_rows": len(behavior), "candidate_accuracy": float(np.mean([r["candidate_correct"] for r in behavior])),
        "open_one_token_accuracy": float(np.mean([r["open_correct"] for r in behavior if r["open_first_token"] is not None])),
        "qualified_slices": len(qualified), "total_slices": len(slices), "slices": slices,
        "capture": capture_headline, "human_naturalness": "NA_not_run",
    }
    close("C601", headline, {"behavior_complete": len(behavior) == len(rows), "qualified_nonempty": bool(qualified),
                             "capture_rows": capture_headline["rows"] == n and n > 0,
                             "shape": capture_headline["role_last_shape"] == [n, 38, 6, 2560],
                             "all_coordinates": capture_headline["quantization"]["has_bf16_parameters"],
                             "finite": finite(headline)}, "C602_state_guard_system_identification")


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    p = np.asarray(prediction, np.float64)
    y = np.asarray(truth, np.float64)
    e = p - y
    den = math.sqrt(float(np.mean(y * y))) + 1e-12
    flat_p, flat_y = p.reshape(-1), y.reshape(-1)
    coordinate_rms = np.sqrt(np.mean(y.reshape(y.shape[0], -1) ** 2, axis=0))
    coordinate_error = np.sqrt(np.mean(e.reshape(e.shape[0], -1) ** 2, axis=0))
    valid = coordinate_rms > 1e-5
    balanced = float(np.mean(np.minimum(coordinate_error[valid] / coordinate_rms[valid], 10.0))) if valid.any() else 0.0
    return {"nrmse": math.sqrt(float(np.mean(e * e))) / den,
            "balanced_nrmse": balanced,
            "cosine": float(np.dot(flat_p, flat_y) / (np.linalg.norm(flat_p) * np.linalg.norm(flat_y) + 1e-12)),
            "sign_agreement": float(np.mean(np.sign(flat_p) == np.sign(flat_y))),
            "truth_rms": den - 1e-12, "error_rms": math.sqrt(float(np.mean(e * e)))}


def role_state(states: np.ndarray, item: dict, q: int) -> np.ndarray:
    return np.asarray(states[int(item["hidden_index"]), q], np.float32)


def transition_pairs(index: list[dict]) -> list[dict]:
    lookup = {(r["panel"], r["family"], r["operation_domain"], r["surface"], r["unit"], r["cell"]): r for r in index if r["behavior_correct"]}
    pairs: list[dict] = []
    def add(panel: str, family: str, domain: str, surface: str, unit: int, left: str, right: str, operation: str, context: str = "base"):
        lk = (panel, family, domain, surface, unit, left)
        rk = (panel, family, domain, surface, unit, right)
        if lk in lookup and rk in lookup:
            pairs.append({"panel": panel, "family": family, "domain": domain, "surface": surface,
                          "unit": unit, "partition": partition(unit), "left_cell": left, "right_cell": right,
                          "operation": operation, "context": context, "left": lookup[lk], "right": lookup[rk]})
    for family, surface, unit in itertools.product(ATOMIC_FAMILIES, SURFACES, range(UNITS)):
        add("atomic", family, "mixed", surface, unit, "v0", "v1", f"atomic:{family}")
    for domain, surface, unit in itertools.product(("badge", "locker", "route"), SURFACES, range(UNITS)):
        add("readout_factorial", "query_value_factorial", domain, surface, unit, "q0v0", "q1v0", "factor:query", "value0")
        add("readout_factorial", "query_value_factorial", domain, surface, unit, "q0v1", "q1v1", "factor:query", "value1")
        add("readout_factorial", "query_value_factorial", domain, surface, unit, "q0v0", "q0v1", "factor:value", "query0")
        add("readout_factorial", "query_value_factorial", domain, surface, unit, "q1v0", "q1v1", "factor:value", "query1")
    for domain, surface, unit in itertools.product(("status", "access"), SURFACES, range(UNITS)):
        add("sequence_program", "query_then_contextual_overwrite", domain, surface, unit, "S0", "A", "sequence:query_switch", "base")
        add("sequence_program", "query_then_contextual_overwrite", domain, surface, unit, "B", "BA", "sequence:query_switch", "after_overwrite")
        add("sequence_program", "query_then_contextual_overwrite", domain, surface, unit, "S0", "B", "sequence:overwrite_current", "base")
        add("sequence_program", "query_then_contextual_overwrite", domain, surface, unit, "A", "AB", "sequence:overwrite_current", "after_query")
    for surface, unit in itertools.product(SURFACES, range(UNITS)):
        add("multiword_readout", "multiword_query_switch", "signal", surface, unit, "q0", "q1", "multiword:query_switch")
    for surface, unit, inner, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1), (0, 1)):
        add("attitude_flagship", "attitude_event_scope", "like_carry", surface, unit,
            f"o0i{inner}q{query_scope}", f"o1i{inner}q{query_scope}", "attitude:outer_negation", f"inner{inner}:q{query_scope}")
    for surface, unit, outer, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1), (0, 1)):
        add("attitude_flagship", "attitude_event_scope", "like_carry", surface, unit,
            f"o{outer}i0q{query_scope}", f"o{outer}i1q{query_scope}", "attitude:inner_negation", f"outer{outer}:q{query_scope}")
    for graph_type, surface, unit in itertools.product(("natural", "pseudo"), SURFACES, range(UNITS)):
        add("knowledge_graph_flagship", "typed_path_program", graph_type, surface, unit, "direct", "two", "graph:one_to_two")
        add("knowledge_graph_flagship", "typed_path_program", graph_type, surface, unit, "two", "three", "graph:two_to_three")
    return pairs


def fit_coordinate_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    centered = x - xm
    gain = np.sum(centered * (y - ym), axis=0) / (np.sum(centered * centered, axis=0) + 1e-4)
    bias = ym - gain * xm
    return gain, bias


def fit_guarded(x: np.ndarray, h: np.ndarray, y: np.ndarray) -> np.ndarray:
    xf, hf, yf = x.reshape(x.shape[0], -1), h.reshape(h.shape[0], -1), y.reshape(y.shape[0], -1)
    ones = np.ones_like(xf)
    features = np.stack((xf, hf, ones), axis=-1)
    gram = np.einsum("ndk,ndl->dkl", features, features)
    gram += np.eye(3, dtype=np.float64)[None] * 1e-3
    rhs = np.einsum("ndk,nd->dk", features, yf)
    beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
    return beta.astype(np.float32)


def predict_guarded(x: np.ndarray, h: np.ndarray, beta: np.ndarray) -> np.ndarray:
    shape = x.shape
    xf, hf = x.reshape(x.shape[0], -1), h.reshape(h.shape[0], -1)
    pred = xf * beta[:, 0] + hf * beta[:, 1] + beta[:, 2]
    return pred.reshape(shape)


def c602() -> None:
    out = begin("C602", {
        "status": "full_coordinate_state_guard_tournament_frozen",
        "object": "predict next-checkpoint operation response from current response and unmodified base state",
        "models": ["identity", "target mean", "coordinate affine", "state-guarded coordinate model", "nearest-state upper bound"],
        "controls": ["wrong operation", "shuffle pairing"],
        "transitions": ["q8->q16", "q16->q24", "q24->q32", "q32->q37"],
        "gate": "guarded NRMSE beats identity, mean and affine by >=0.02 on unit lockbox",
        "coordinate_policy": "all 6x2560 signed role coordinates",
    }, {"parent": final("C601")["all_checks_passed"]})
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    index = read_rows(capture_index_path())
    states = np.load(last_path(), mmap_mode="r")
    pairs = transition_pairs(index)
    metrics: dict[str, dict] = {}
    gates: dict[str, bool] = {}
    prototypes: dict[str, np.ndarray] = {}
    try:
        grouped = defaultdict(list)
        for pair in pairs:
            grouped[(pair["operation"], pair["context"])].append(pair)
        operations = sorted({key[0] for key in grouped})
        for operation in operations:
            operation_pairs = [p for p in pairs if p["operation"] == operation]
            train = [p for p in operation_pairs if p["partition"] == "discovery"]
            test = [p for p in operation_pairs if p["partition"] == "lockbox"]
            if len(train) < 4 or len(test) < 2:
                continue
            for q in QPOINTS:
                responses = np.stack([role_state(states, p["right"], q) - role_state(states, p["left"], q) for p in train])
                prototypes[f"{operation}|q{q}"] = responses.mean(axis=0).astype(np.float32)
            for q0, q1 in ((8, 16), (16, 24), (24, 32), (32, 37)):
                xtr = np.stack([role_state(states, p["right"], q0) - role_state(states, p["left"], q0) for p in train])
                ytr = np.stack([role_state(states, p["right"], q1) - role_state(states, p["left"], q1) for p in train])
                htr = np.stack([role_state(states, p["left"], q0) for p in train])
                xte = np.stack([role_state(states, p["right"], q0) - role_state(states, p["left"], q0) for p in test])
                yte = np.stack([role_state(states, p["right"], q1) - role_state(states, p["left"], q1) for p in test])
                hte = np.stack([role_state(states, p["left"], q0) for p in test])
                gain, bias = fit_coordinate_affine(xtr, ytr)
                affine = xte * gain + bias
                beta = fit_guarded(xtr, htr, ytr)
                guarded = predict_guarded(xte, hte, beta)
                mean = np.broadcast_to(ytr.mean(axis=0), yte.shape)
                distances = np.sum((hte.reshape(len(hte), -1)[:, None] - htr.reshape(len(htr), -1)[None]) ** 2, axis=-1)
                nearest = ytr[np.argmin(distances, axis=1)]
                wrong_operation = next((o for o in operations if o != operation and f"{o}|q{q1}" in prototypes), None)
                wrong = (np.broadcast_to(prototypes[f"{wrong_operation}|q{q1}"], yte.shape)
                         if wrong_operation else np.zeros_like(yte))
                value = {"samples_train": len(train), "samples_test": len(test),
                         "identity": metric(xte, yte), "target_mean": metric(mean, yte),
                         "coordinate_affine": metric(affine, yte), "state_guarded": metric(guarded, yte),
                         "nearest_state_upper": metric(nearest, yte), "wrong_operation": metric(wrong, yte),
                         "wrong_operation_name": wrong_operation}
                candidates = {k: v["nrmse"] for k, v in value.items() if isinstance(v, dict) and "nrmse" in v}
                value["winner"] = min(candidates, key=candidates.get)
                value["guarded_compression_gap"] = value["state_guarded"]["nrmse"] - value["nearest_state_upper"]["nrmse"]
                key = f"{operation}|q{q0}->q{q1}"
                metrics[key] = value
                gates[key] = all(value["state_guarded"]["nrmse"] <= value[name]["nrmse"] - CONTROL_MARGIN
                                 for name in ("identity", "target_mean", "coordinate_affine"))
        np.savez_compressed(out / "analysis/discovery_response_prototypes.npz", **prototypes)
    finally:
        previous.previous.close_mmap(states)
        del states
    by_operation = {}
    for operation in sorted({key.split("|q")[0] for key in metrics}):
        values = [v for k, v in gates.items() if k.startswith(operation + "|q")]
        winners = [metrics[k]["winner"] for k in metrics if k.startswith(operation + "|q")]
        by_operation[operation] = {"passed": int(sum(values)), "total": len(values),
                                   "pass_rate": float(np.mean(values)) if values else 0.0,
                                   "winners": {name: winners.count(name) for name in sorted(set(winners))},
                                   "state_guard_candidate": bool(values) and float(np.mean(values)) >= PREDICTION_GATE}
    headline = {"status": "state_guard_system_identification_closed", "pair_count": len(pairs),
                "operation_count": len(by_operation), "metrics": metrics, "gates": gates,
                "by_operation": by_operation,
                "state_guard_candidates": [k for k, v in by_operation.items() if v["state_guard_candidate"]],
                "strict_interpretation": "A win is predictive dependence on the observed base state, not a unique causal guard or a new mathematical structure."}
    close("C602", headline, {"pairs": len(pairs) >= 300, "operations": len(by_operation) >= 10,
                             "metrics": bool(metrics), "prototypes": bool(prototypes), "finite": finite(headline)},
          "C603_sequential_transport_and_flagships")


def effect(states: np.ndarray, cells: dict[str, dict], q: int, keys: tuple[str, str, str, str]) -> np.ndarray:
    h00, h10, h01, h11 = (role_state(states, cells[k], q) for k in keys)
    return h11 - h10 - h01 + h00


def c603() -> None:
    out = begin("C603", {
        "status": "sequential_transport_and_flagship_composition_frozen",
        "tests": ["2x2 query-value interaction", "A then B versus B then A", "unseen sequential composition",
                  "nested attitude outer-inner-query interaction", "natural and pseudo 1/2/3-hop graph"],
        "transport_rule": "edge prototypes fit on discovery units; lockbox units used once",
        "holonomy_rule": "registered NA unless independent fitted loop edges qualify",
        "gate": "correct sequential prediction beats additive/mean/wrong-order by >=0.02",
    }, {"parent": final("C602")["all_checks_passed"]})
    index = read_rows(capture_index_path())
    states = np.load(last_path(), mmap_mode="r")
    lookup = {(r["panel"], r["family"], r["operation_domain"], r["surface"], r["unit"], r["cell"]): r for r in index if r["behavior_correct"]}
    results: dict[str, Any] = {"factorial": {}, "sequence": {}, "attitude": {}, "graph": {}}
    try:
        for q in (16, 24, 32, 37):
            train_j, test_j = [], []
            for domain, surface, unit in itertools.product(("badge", "locker", "route"), SURFACES, range(UNITS)):
                keys = [("readout_factorial", "query_value_factorial", domain, surface, unit, f"q{a}v{b}") for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
                if all(k in lookup for k in keys):
                    j = effect(states, {f"q{a}v{b}": lookup[("readout_factorial", "query_value_factorial", domain, surface, unit, f"q{a}v{b}")] for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))}, q, ("q0v0", "q1v0", "q0v1", "q1v1"))
                    (train_j if partition(unit) == "discovery" else test_j if partition(unit) == "lockbox" else []).append(j)
            if train_j and test_j:
                tr, te = np.stack(train_j), np.stack(test_j)
                proto = tr.mean(axis=0)
                results["factorial"][f"q{q}"] = {"samples": len(te), "prototype": metric(np.broadcast_to(proto, te.shape), te),
                                                      "zero": metric(np.zeros_like(te), te),
                                                      "gate": metric(np.broadcast_to(proto, te.shape), te)["nrmse"] <= metric(np.zeros_like(te), te)["nrmse"] - CONTROL_MARGIN}

        for q in (16, 24, 32, 37):
            train = defaultdict(list)
            tests = []
            for domain, surface, unit in itertools.product(("status", "access"), SURFACES, range(UNITS)):
                cells = {cell: lookup.get(("sequence_program", "query_then_contextual_overwrite", domain, surface, unit, cell)) for cell in ("S0", "A", "B", "AB", "BA")}
                if not all(cells.values()):
                    continue
                h = {cell: role_state(states, item, q) for cell, item in cells.items()}
                item = {"h": h, "domain": domain, "surface": surface, "unit": unit}
                if partition(unit) == "discovery":
                    train["A0"].append(h["A"] - h["S0"])
                    train["B0"].append(h["B"] - h["S0"])
                    train["B_after_A"].append(h["AB"] - h["A"])
                    train["A_after_B"].append(h["BA"] - h["B"])
                    train["AB"].append(h["AB"] - h["S0"])
                    train["BA"].append(h["BA"] - h["S0"])
                elif partition(unit) == "lockbox":
                    tests.append(item)
            if train and tests:
                p = {k: np.stack(v).mean(axis=0) for k, v in train.items()}
                actual_ab = np.stack([t["h"]["AB"] for t in tests])
                actual_ba = np.stack([t["h"]["BA"] for t in tests])
                seq_ab = np.stack([t["h"]["A"] + p["B_after_A"] for t in tests])
                seq_ba = np.stack([t["h"]["B"] + p["A_after_B"] for t in tests])
                add_ab = np.stack([t["h"]["S0"] + p["A0"] + p["B0"] for t in tests])
                mean_ab = np.stack([t["h"]["S0"] + p["AB"] for t in tests])
                wrong_ab = seq_ba
                value = {"samples": len(tests), "AB_sequential": metric(seq_ab, actual_ab),
                         "BA_sequential": metric(seq_ba, actual_ba), "AB_additive": metric(add_ab, actual_ab),
                         "AB_target_mean": metric(mean_ab, actual_ab), "AB_wrong_order": metric(wrong_ab, actual_ab)}
                value["AB_gate"] = all(value["AB_sequential"]["nrmse"] <= value[name]["nrmse"] - CONTROL_MARGIN
                                       for name in ("AB_additive", "AB_target_mean", "AB_wrong_order"))
                results["sequence"][f"q{q}"] = value

        for q in (16, 24, 32, 37):
            train, test = [], []
            for surface, unit, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1)):
                cells = {f"o{o}i{i}q{query_scope}": lookup.get(("attitude_flagship", "attitude_event_scope", "like_carry", surface, unit, f"o{o}i{i}q{query_scope}")) for o, i in itertools.product((0, 1), repeat=2)}
                if not all(cells.values()):
                    continue
                j = effect(states, cells, q, (f"o0i0q{query_scope}", f"o1i0q{query_scope}", f"o0i1q{query_scope}", f"o1i1q{query_scope}"))
                (train if partition(unit) == "discovery" else test if partition(unit) == "lockbox" else []).append(j)
            if train and test:
                tr, te = np.stack(train), np.stack(test); proto = tr.mean(axis=0)
                results["attitude"][f"q{q}"] = {"samples": len(te), "interaction": metric(np.broadcast_to(proto, te.shape), te),
                                                     "zero": metric(np.zeros_like(te), te),
                                                     "gate": metric(np.broadcast_to(proto, te.shape), te)["nrmse"] <= metric(np.zeros_like(te), te)["nrmse"] - CONTROL_MARGIN}

        for q in (16, 24, 32, 37):
            for graph_type in ("natural", "pseudo"):
                train12, train23, tests = [], [], []
                for surface, unit in itertools.product(SURFACES, range(UNITS)):
                    cells = {cell: lookup.get(("knowledge_graph_flagship", "typed_path_program", graph_type, surface, unit, cell)) for cell in ("direct", "two", "three")}
                    if not all(cells.values()):
                        continue
                    h = {cell: role_state(states, item, q) for cell, item in cells.items()}
                    if partition(unit) == "discovery":
                        train12.append(h["two"] - h["direct"]); train23.append(h["three"] - h["two"])
                    elif partition(unit) == "lockbox":
                        tests.append(h)
                if train12 and tests:
                    p12, p23 = np.stack(train12).mean(axis=0), np.stack(train23).mean(axis=0)
                    actual2 = np.stack([t["two"] for t in tests]); actual3 = np.stack([t["three"] for t in tests])
                    pred2 = np.stack([t["direct"] + p12 for t in tests]); pred3 = np.stack([t["two"] + p23 for t in tests])
                    wrong3 = np.stack([t["two"] + p12 for t in tests])
                    value = {"samples": len(tests), "one_to_two": metric(pred2, actual2),
                             "two_to_three": metric(pred3, actual3), "wrong_edge": metric(wrong3, actual3)}
                    value["gate"] = value["two_to_three"]["nrmse"] <= value["wrong_edge"]["nrmse"] - CONTROL_MARGIN
                    results["graph"][f"{graph_type}|q{q}"] = value
    finally:
        previous.previous.close_mmap(states); del states
    summary = {
        "factorial": {"passed": sum(v["gate"] for v in results["factorial"].values()), "total": len(results["factorial"])},
        "sequence": {"passed": sum(v["AB_gate"] for v in results["sequence"].values()), "total": len(results["sequence"])},
        "attitude": {"passed": sum(v["gate"] for v in results["attitude"].values()), "total": len(results["attitude"])},
        "graph": {"passed": sum(v["gate"] for v in results["graph"].values()), "total": len(results["graph"])},
    }
    headline = {"status": "sequential_transport_and_flagships_closed", "results": results,
                "summary": summary, "cocycle_candidate": summary["sequence"]["passed"] >= 3,
                "holonomy": "NA_no_independent_closed_loop_edges_qualified",
                "strict_interpretation": "A sequential pass is a learned conditional transport candidate. It is not curvature or holonomy."}
    close("C603", headline, {"factorial": bool(results["factorial"]), "sequence": bool(results["sequence"]),
                             "attitude": bool(results["attitude"]), "graph": bool(results["graph"]),
                             "finite": finite(headline)}, "C604_bidirectional_output_causal")


def scaled_like(control: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(control) * (float(np.linalg.norm(reference)) / (float(np.linalg.norm(control)) + 1e-12))


def patched_forward_multi(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                          role_positions: dict, patches: list[tuple[int, np.ndarray, tuple[int, ...]]]) -> tuple[np.ndarray, np.ndarray]:
    base = model.model
    handles = []
    final_state: list[torch.Tensor] = []
    by_q = defaultdict(list)
    for q, response, role_order in patches:
        by_q[int(q)].append((np.asarray(response, np.float32), role_order))
    for q, values in by_q.items():
        if not (1 <= q <= len(base.layers)):
            raise ValueError(q)
        def make_hook(items):
            def patch_hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                for response, role_order in items:
                    for target_i, role in enumerate(ROLES):
                        source_i = role_order[target_i]
                        pos = int(role_positions[role][-1])
                        changed[0, pos] = changed[0, pos] + torch.tensor(response[source_i], dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return patch_hook
        handles.append(base.layers[q - 1].register_forward_hook(make_hook(values)))
    handles.append(base.norm.register_forward_hook(lambda _m, _a, output: final_state.append(output.detach())))
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                           use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    at = int(attention_mask.sum()) - 1
    return final_state[-1][0].float().cpu().numpy(), output.logits[0, at].float().cpu().numpy()


def patched_greedy_text(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        role_positions: dict, patches: list[tuple], max_new_tokens: int = 6) -> str:
    base = model.model
    handles = []
    by_q = defaultdict(list)
    for q, response, role_order in patches:
        by_q[int(q)].append((np.asarray(response, np.float32), role_order))
    for q, values in by_q.items():
        def make_hook(items):
            def patch_hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                changed = tensor.clone()
                for response, role_order in items:
                    for target_i, role in enumerate(ROLES):
                        source_i = role_order[target_i]
                        pos = int(role_positions[role][-1])
                        if pos < changed.shape[1]:
                            changed[0, pos] = changed[0, pos] + torch.tensor(
                                response[source_i], dtype=changed.dtype, device=changed.device)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            return patch_hook
        handles.append(base.layers[q - 1].register_forward_hook(make_hook(values)))
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def answer_matches(text: str, answer: str) -> bool:
    clean_text = " ".join(text.strip().lower().split()).strip(".,;:!?\"'")
    clean_answer = " ".join(answer.strip().lower().split()).strip(".,;:!?\"'")
    return clean_text == clean_answer or clean_text.startswith(clean_answer + " ")


def c604() -> None:
    out = begin("C604", {
        "status": "bidirectional_output_deletion_rescue_frozen",
        "eligible": "output-changing operations with discovery response prototypes and behavior-correct lockbox pairs",
        "layers": [16, 24, 32], "doses": [0.5, 1.0, 1.5],
        "controls": ["zero", "wrong operation", "wrong role", "wrong checkpoint", "opposite sign"],
        "necessity": "subtract discovery q24 response from natural target",
        "rescue": "q24 deletion followed by q32 correct or wrong downstream response",
        "output": "registered candidate logits, raw next-token identity, and six-token greedy decoded text; multi-token causal output remains NA",
        "gate": "state guidance and target output reported separately in both directions",
    }, {"parent": final("C603")["all_checks_passed"]})
    rows = {r["case_id"]: r for r in read_rows(material_path())}
    compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    index = read_rows(capture_index_path())
    states = np.load(last_path(), mmap_mode="r")
    pairs = transition_pairs(index)
    proto_file = np.load(OUTS["C602"] / "analysis/discovery_response_prototypes.npz")
    proto = {k: np.asarray(proto_file[k], np.float32) for k in proto_file.files}
    proto_file.close()
    eligible_ops = ("factor:query", "factor:value", "sequence:query_switch", "sequence:overwrite_current",
                    "attitude:outer_negation", "attitude:inner_negation")
    eligible_pairs = [p for p in pairs if p["operation"] in eligible_ops and p["partition"] == "lockbox"]
    selected_pairs = []
    for operation in eligible_ops:
        selected_pairs.extend([p for p in eligible_pairs if p["operation"] == operation][:4])
    model = None
    records = []
    try:
        model, tokenizer, device, placement = passport.previous.model_base().load_bf16("qwen3")
        wrong_name = next(name for name in eligible_ops if name != selected_pairs[0]["operation"] and f"{name}|q24" in proto) if selected_pairs else None
        for pair_i, pair in enumerate(selected_pairs):
            operation = pair["operation"]
            if f"{operation}|q24" not in proto or f"{operation}|q32" not in proto:
                continue
            p24, p32 = proto[f"{operation}|q24"], proto[f"{operation}|q32"]
            wrong_op = next((name for name in eligible_ops if name != operation and f"{name}|q24" in proto), wrong_name)
            wrong24 = scaled_like(proto[f"{wrong_op}|q24"], p24) if wrong_op else p24[::-1]
            wrong32 = scaled_like(proto[f"{wrong_op}|q32"], p32) if wrong_op and f"{wrong_op}|q32" in proto else p32[::-1]
            for source, target, sign, direction in ((pair["left"], pair["right"], 1.0, "forward"),
                                                     (pair["right"], pair["left"], -1.0, "reverse")):
                source_comp, target_comp = compiled[source["case_id"]], compiled[target["case_id"]]
                if any(len(ids) != 1 for ids in source_comp["candidate_ids"]):
                    continue
                ids = torch.tensor([source_comp["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
                target_gold = int(target_comp["gold_position"])
                candidates = source_comp["candidate_ids"]
                target_token = int(target_comp["candidate_ids"][target_gold][0])
                identity_order = tuple(range(len(ROLES)))
                swapped_order = (1, 0, 3, 2, 4, 5)
                interventions = {
                    "zero": [],
                    "dose_0.5": [(24, sign * 0.5 * p24, identity_order)],
                    "correct": [(24, sign * p24, identity_order)],
                    "dose_1.5": [(24, sign * 1.5 * p24, identity_order)],
                    "wrong_operation": [(24, sign * wrong24, identity_order)],
                    "wrong_role": [(24, sign * p24, swapped_order)],
                    "wrong_checkpoint": [(16, sign * p24, identity_order)],
                    "opposite_sign": [(24, -sign * p24, identity_order)],
                }
                target_state = role_state(states, target, 37)
                source_values = {}
                for name, patches in interventions.items():
                    final_state, logits = patched_forward_multi(model, ids, mask, pos, source_comp["role_positions"], patches)
                    gathered = np.stack([final_state[int(source_comp["role_positions"][role][-1])] for role in ROLES])
                    scores = [float(logits[candidate[0]]) for candidate in candidates]
                    source_values[name] = {"state_to_target": metric(gathered, target_state),
                                           "candidate_prediction": int(np.argmax(scores)),
                                           "open_token": int(np.argmax(logits)), "target_token": target_token,
                                           "open_target": int(np.argmax(logits)) == target_token,
                                           "target_margin": scores[target_gold] - max(v for i, v in enumerate(scores) if i != target_gold),
                                           "candidate_scores": scores}
                generated = patched_greedy_text(model, tokenizer, ids, mask, source_comp["role_positions"],
                                                interventions["correct"])
                source_values["correct"]["generated_text"] = generated
                source_values["correct"]["generated_target"] = answer_matches(generated, target_comp["answer"])
                target_ids = torch.tensor([target_comp["prompt_ids"]], dtype=torch.long, device=device)
                target_mask = torch.ones_like(target_ids); target_pos = torch.arange(target_ids.shape[1], device=device)[None]
                target_interventions = {
                    "target_natural": [],
                    "deletion": [(24, -sign * p24, identity_order)],
                    "rescue": [(24, -sign * p24, identity_order), (32, sign * p32, identity_order)],
                    "wrong_rescue": [(24, -sign * p24, identity_order), (32, sign * wrong32, identity_order)],
                }
                target_values = {}
                for name, patches in target_interventions.items():
                    final_state, logits = patched_forward_multi(model, target_ids, target_mask, target_pos, target_comp["role_positions"], patches)
                    scores = [float(logits[candidate[0]]) for candidate in target_comp["candidate_ids"]]
                    target_values[name] = {"candidate_prediction": int(np.argmax(scores)),
                                           "open_token": int(np.argmax(logits)), "target_token": target_token,
                                           "open_target": int(np.argmax(logits)) == target_token,
                                           "target_margin": scores[target_gold] - max(v for i, v in enumerate(scores) if i != target_gold),
                                           "candidate_scores": scores}
                    generated = patched_greedy_text(model, tokenizer, target_ids, target_mask,
                                                    target_comp["role_positions"], patches)
                    target_values[name]["generated_text"] = generated
                    target_values[name]["generated_target"] = answer_matches(generated, target_comp["answer"])
                records.append({"operation": operation, "domain": pair["domain"], "surface": pair["surface"],
                                "unit": pair["unit"], "direction": direction, "source": source["case_id"],
                                "target": target["case_id"], "source_interventions": source_values,
                                "target_interventions": target_values, "wrong_operation_name": wrong_op})
            if pair_i % 8 == 0 or pair_i + 1 == len(selected_pairs):
                print(f"[C604 causal] {pair_i + 1}/{len(selected_pairs)}", flush=True)
    finally:
        passport.previous.model_base().release_bf16(model)
        previous.previous.close_mmap(states); del states
        gc.collect()
    write_rows(out / "analysis/intervention_records.jsonl", records)
    summary = {}
    for operation in eligible_ops:
        op_rows = [r for r in records if r["operation"] == operation]
        directions = {}
        for direction in ("forward", "reverse"):
            values = [r for r in op_rows if r["direction"] == direction]
            directions[direction] = {
                "tests": len(values),
                "state_guidance": sum(r["source_interventions"]["correct"]["state_to_target"]["nrmse"] <= r["source_interventions"]["zero"]["state_to_target"]["nrmse"] - CONTROL_MARGIN and r["source_interventions"]["correct"]["state_to_target"]["nrmse"] <= r["source_interventions"]["wrong_operation"]["state_to_target"]["nrmse"] - CONTROL_MARGIN for r in values),
                "candidate_output": sum(r["source_interventions"]["correct"]["candidate_prediction"] == compiled[r["target"]]["gold_position"] for r in values),
                "open_output": sum(r["source_interventions"]["correct"]["open_target"] for r in values),
                "generated_output": sum(r["source_interventions"]["correct"]["generated_target"] for r in values),
                "necessity": sum(r["target_interventions"]["deletion"]["target_margin"] < r["target_interventions"]["target_natural"]["target_margin"] - 0.1 for r in values),
                "rescue": sum(r["target_interventions"]["rescue"]["target_margin"] > r["target_interventions"]["deletion"]["target_margin"] + 0.1 and r["target_interventions"]["rescue"]["target_margin"] > r["target_interventions"]["wrong_rescue"]["target_margin"] + 0.1 for r in values),
                "generated_natural": sum(r["target_interventions"]["target_natural"]["generated_target"] for r in values),
                "generated_after_deletion": sum(r["target_interventions"]["deletion"]["generated_target"] for r in values),
                "generated_after_rescue": sum(r["target_interventions"]["rescue"]["generated_target"] for r in values),
            }
        summary[operation] = directions
    headline = {"status": "bidirectional_output_deletion_rescue_closed", "records": len(records),
                "summary": summary, "multi_token_causal": "NA_no_behavior_qualified_output_changing_multi-token response entered patching",
                "strict_interpretation": "Output, state guidance, necessity and rescue are separate gates. A directional pass does not imply a general semantic compiler."}
    close("C604", headline, {"records": bool(records), "directions": all(any(v[d]["tests"] > 0 for d in ("forward", "reverse")) for v in summary.values()),
                             "controls": all("wrong_operation" in r["source_interventions"] and "wrong_rescue" in r["target_interventions"] for r in records),
                             "finite": finite(headline)}, "C605_cross_model_visual_theory")


def register_visual() -> None:
    entry = {"id": "c605_language_transport_output_atlas", "title": "C605 Language Transport and Output Boundary Atlas",
             "phase": 2139, "campaign": "C600-C605",
             "path": "vis_data/research_kernel/c605_language_transport_output_atlas.json",
             "schema": "ai2050.language_transport_output_atlas.v1",
             "description": "All-coordinate Qwen field, state-guard tournament, sequential composition, bidirectional output interventions and model-relative topology."}
    catalog = load(CATALOG) if CATALOG.exists() else {"artifacts": []}
    artifacts = catalog.setdefault("artifacts", [])
    artifacts[:] = [item for item in artifacts if item.get("id") != entry["id"]]
    artifacts.append(entry)
    save(CATALOG, catalog)


def c605() -> None:
    out = begin("C605", {
        "status": "cross_model_visual_theory_frozen",
        "models": ["Qwen3-4B completed", "Qwen3-14B FP16 offload", "GLM4 BF16/registered loader", "DeepSeek7B behavior first"],
        "cross_model": "model-relative layer depth, role topology, operation confusion and output direction only",
        "visual": "one exact all-token Qwen field plus all signed operation response coordinates and model-relative representatives",
        "cleanup": "remove only bulk full-token shards after exact displayed representative is materialized; retain all role tensors and metrics",
        "theory_name": "Conditional Output Field Closure Theory",
        "new_math_gate": "closed unless stable object, unseen composition, bidirectional causal output, necessity, cross-model function and compression all pass",
    }, {"parent": final("C604")["all_checks_passed"]})
    worker = TESTS / "phase2139_c605_cross_model_output_worker.py"
    q14_worker = TESTS / "phase2139_c605_qwen14_output_worker.py"
    worker_results = {}
    for model_name in ("glm4", "deepseek7b"):
        target = out / f"analysis/{model_name}_worker.json"
        completed = subprocess.run([sys.executable, str(worker), "--model", model_name,
                                    "--material", str(material_path()), "--output", str(target)],
                                   cwd=str(ROOT), capture_output=True, text=True, check=False)
        (out / f"audit/{model_name}_stdout.txt").parent.mkdir(parents=True, exist_ok=True)
        (out / f"audit/{model_name}_stdout.txt").write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8")
        if target.exists():
            worker_results[model_name] = load(target)
        else:
            worker_results[model_name] = {"status": "worker_missing", "returncode": completed.returncode,
                                          "hiddenstate_ran": False, "functional_candidate": False}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    q14_target = out / "analysis/qwen14_worker.json"
    completed = subprocess.run([sys.executable, str(q14_worker), "--material", str(material_path()),
                                "--output", str(q14_target)], cwd=str(ROOT), capture_output=True, text=True, check=False)
    (out / "audit/qwen14_stdout.txt").write_text(completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8")
    worker_results["qwen3_14b"] = load(q14_target) if q14_target.exists() else {
        "status": "worker_missing", "returncode": completed.returncode, "hiddenstate_ran": False,
        "functional_candidate": False}

    compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    index = read_rows(capture_index_path())
    representative = next(r for r in index if r["partition"] == "lockbox" and r["behavior_correct"])
    raw_path = shard_dir() / representative["shard"]
    raw = np.load(raw_path, mmap_mode="r")
    exact = np.asarray(raw[int(representative["shard_index"]), :, :int(representative["length"])], np.float32)
    prompt_ids = compiled[representative["case_id"]]["prompt_ids"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    response_file = np.load(OUTS["C602"] / "analysis/discovery_response_prototypes.npz")
    response_fields = {key: np.asarray(response_file[key], np.float32).tolist() for key in response_file.files if key.endswith("|q24") or key.endswith("|q37")}
    response_file.close()
    atlas = {
        "schema": "ai2050.language_transport_output_atlas.v1", "phase": 2139, "campaign": "C600-C605",
        "coordinate_policy": "all signed physical coordinates; no PCA, Top-K, magnitude threshold, attention or MLP internals",
        "language_program_ledger": final("C600")["headline"],
        "qwen3_4b": {"coordinates": DIM, "checkpoints": ["embedding"] + [f"block_{i:02d}_post" for i in range(36)] + ["final_norm"],
                       "representative": {"case_id": representative["case_id"], "token_ids": prompt_ids,
                                          "tokens": tokens, "shape": list(exact.shape), "states": exact.tolist()},
                       "response_fields": response_fields},
        "state_guard": final("C602")["headline"]["by_operation"],
        "composition": final("C603")["headline"]["summary"],
        "causal": final("C604")["headline"]["summary"],
        "cross_model": worker_results,
        "warnings": ["Activation coordinates are not model weights or named semantic neurons.",
                     "Human blind naturalness was not run.", "Physical coordinate numbers are never aligned across models.",
                     "Interaction and loop errors are not called curvature."],
    }
    save(VISUAL, atlas)
    del raw, exact
    cleaned = []
    if shard_dir().exists():
        resolved = shard_dir().resolve()
        if not str(resolved).lower().startswith(str(ROOT.resolve()).lower()):
            raise RuntimeError(resolved)
        size = sum(path.stat().st_size for path in shard_dir().rglob("*") if path.is_file())
        shutil.rmtree(shard_dir())
        cleaned.append({"path": str(shard_dir().relative_to(ROOT)), "bytes": size,
                        "reason": "one exact field is in C605 atlas; all-case role-coordinate tensors and derived analyses remain"})
    for model_name, value in worker_results.items():
        raw_value = value.get("raw_path")
        if not raw_value:
            continue
        candidate = ROOT / raw_value
        if candidate.exists():
            resolved = candidate.resolve()
            if not str(resolved).lower().startswith(str(ROOT.resolve()).lower()):
                raise RuntimeError(resolved)
            size = candidate.stat().st_size
            candidate.unlink()
            cleaned.append({"path": raw_value, "bytes": size, "reason": "model representative and all-coordinate topology are in C605 atlas"})
    register_visual()
    behavior = final("C601")["headline"]
    system_id = final("C602")["headline"]
    composition = final("C603")["headline"]
    causal = final("C604")["headline"]
    empirical = {
        "qwen_behavior": behavior["candidate_accuracy"] >= BEHAVIOR_GATE,
        "open_output": behavior["open_one_token_accuracy"] >= OPEN_GATE,
        "state_guard": bool(system_id["state_guard_candidates"]),
        "unseen_sequential": bool(composition["cocycle_candidate"]),
        "bidirectional_output": any(all(v[d]["candidate_output"] >= max(1, int(.75 * v[d]["tests"])) for d in ("forward", "reverse")) for v in causal["summary"].values()),
        "necessity_and_rescue": any(all(v[d]["necessity"] > 0 and v[d]["rescue"] > 0 for d in ("forward", "reverse")) for v in causal["summary"].values()),
        "cross_model_functional": sum(bool(v.get("functional_candidate")) for v in worker_results.values()) >= 2,
        "human_naturalness": False,
        "new_math": False,
    }
    headline = {"status": "cross_model_visual_theory_closed", "workers": worker_results,
                "visual": str(VISUAL.relative_to(ROOT)), "visual_bytes": VISUAL.stat().st_size,
                "cleaned": cleaned, "cleaned_bytes": sum(v["bytes"] for v in cleaned),
                "retained_role_mean": str(mean_path().relative_to(ROOT)), "retained_role_last": str(last_path().relative_to(ROOT)),
                "empirical_gates": empirical,
                "theory": {"name": "Conditional Output Field Closure Theory", "principle": "Reuse-Difference-Conditioning",
                           "object": "typed scope- and state-conditioned full-coordinate response transport",
                           "foundational_math_authorized": False},
                "strict_interpretation": "The campaign compares conditional response laws and output boundaries. It does not establish a unique circuit, a fiber bundle, curvature, or foundational new mathematics."}
    close("C605", headline, {"workers": len(worker_results) == 3, "visual": VISUAL.exists() and VISUAL.stat().st_size > 1_000_000,
                             "catalog": CATALOG.exists(), "retained_roles": mean_path().exists() and last_path().exists(),
                             "raw_cleaned": not shard_dir().exists(), "finite": finite(headline)},
          "C606_independent_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(PHASES) + ("all",), default="all")
    args = parser.parse_args()
    stages = tuple(PHASES) if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"=== {stage} phase={PHASES[stage][0]} ===", flush=True)
        globals()[stage.lower()]()
        print(json.dumps(final(stage), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
