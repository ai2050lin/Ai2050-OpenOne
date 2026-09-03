#!/usr/bin/env python3
"""C607-C611 natural open-generation and conditional transport campaign.

The scientific interface is embedding, HiddenState checkpoints and output
logits/text. Attention, MLP internals, gradients, PCA, Top-K and magnitude
truncation are not used. Retained fields keep every signed coordinate.
"""
from __future__ import annotations

import argparse
import csv
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
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
sys.path.insert(0, str(TESTS))

import model_utils
import phase1331_relational_measurement_core as text_core
import phase1797_c263_c272_state_operator_common as compiler
import phase2076_c542_c559_typed_operation_response_passport_campaign as passport
import phase2134_c600_c605_language_transport_campaign as previous


PHASES = {
    "C607": (2141, "natural_open_generation_master_contract_and_behavior"),
    "C608": (2142, "all_coordinate_conditional_operator_tomography"),
    "C609": (2143, "language_program_algebra_and_unseen_composition"),
    "C610": (2144, "generation_level_bidirectional_deletion_rescue"),
    "C611": (2145, "cross_model_functional_topology_visualization_theory"),
}
OUTS = {name: RESULT / f"phase{phase}_{name.lower()}_{slug}" for name, (phase, slug) in PHASES.items()}
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c611_natural_output_compiler_atlas.json"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"

SYSTEM = "Use only the supplied information. Reply with only the requested short answer and no explanation."
SURFACES = ("open", "options")
UNITS = 8
ROLES = previous.ROLES
QPOINTS = previous.QPOINTS
DIM = 2560
CHECKPOINTS = 38
BEHAVIOR_GATE = 0.75
CONTROL_MARGIN = 0.02

NAMES_A = ("Mabel", "Cedric", "Iris", "Jonas", "Lydia", "Orson", "Petra", "Rufus")
NAMES_B = ("Nolan", "Daphne", "Felix", "Greta", "Hugo", "Karin", "Lucan", "Mira")
NAMES_C = ("Tobias", "Una", "Victor", "Willa", "Xavier", "Yara", "Zane", "Bianca")
NAMES_D = ("Caleb", "Dora", "Elias", "Flora", "Gavin", "Hazel", "Ivan", "Julia")
COLORS = ("deep violet", "pale amber", "soft crimson", "bright teal",
          "muted indigo", "warm ochre", "cool magenta", "vivid cyan")
ACTIONS = ("baking bread", "planting herbs", "painting murals", "repairing clocks",
           "weaving baskets", "mapping caves", "polishing lenses", "carving stamps")
OBJECTS = ("cedar tower", "silver post", "granite arch", "copper mast",
           "willow gate", "bronze pillar", "marble beacon", "iron frame")
EVENTS = ("morning briefing", "evening inspection", "harbor opening", "archive closing",
          "garden survey", "museum audit", "station handover", "library review")
CAUSES = ("cracked valve", "loose cable", "blocked vent", "empty tank",
          "bent latch", "worn gasket", "jammed relay", "split hose")
EFFECTS = ("warning alarm", "power outage", "pressure drop", "engine stall",
           "door failure", "coolant leak", "signal loss", "pump shutdown")
PARTS = ("front wheel", "rear panel", "upper hinge", "side handle",
         "lower bracket", "inner spring", "outer casing", "center axle")
ROUTES = ("north route", "south route", "east route", "west route",
          "river route", "forest route", "coastal route", "valley route")
SPANISH = ("verde oscuro", "azul claro", "rojo suave", "violeta intenso",
           "indigo tenue", "ocre calido", "magenta frio", "cian vivo")
NUMBERS = ("twelve", "seven", "nine", "four")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def begin(name: str, protocol: dict, dependencies: dict) -> Path:
    out = OUTS[name]
    (out / "protocol").mkdir(parents=True, exist_ok=True)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    save(out / "protocol/preregistration.json", {
        "phase": PHASES[name][0], "campaign": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "dependencies": dependencies,
        "claim_boundary": "execution completion is not an empirical mechanism pass",
    })
    print(f"=== {name} phase={PHASES[name][0]} ===", flush=True)
    return out


def close(name: str, headline: dict, checks: dict, next_authorization: str) -> dict:
    result = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": bool(checks) and all(checks.values()),
        "headline": headline, "checks": checks, "next_authorization": next_authorization,
    }
    save(OUTS[name] / "analysis/final.json", result)
    print(json.dumps({"phase": result["phase"], "campaign": name,
                      "all_checks_passed": result["all_checks_passed"],
                      "headline_summary": summarize_headline(headline), "checks": checks},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def summarize_headline(value: dict) -> dict:
    keep = ("status", "rows", "candidate_accuracy", "generated_accuracy", "qualified_slices",
            "capture_rows", "pair_count", "operation_count", "operator_candidates", "records",
            "summary", "workers", "visual", "visual_bytes", "cleaned_bytes", "empirical_gates")
    return {k: value[k] for k in keep if k in value and k != "workers"} | ({
        "workers": {k: {x: v.get(x) for x in ("status", "rows", "candidate_accuracy", "generated_accuracy",
                                               "hiddenstate_ran", "checkpoints", "coordinates", "functional_candidate")}
                    for k, v in value.get("workers", {}).items()}
    } if "workers" in value else {})


def final(name: str) -> dict:
    return load(OUTS[name] / "analysis/final.json")


def partition(unit: int) -> str:
    if unit < 4:
        return "discovery"
    if unit < 6:
        return "confirmation"
    return "lockbox"


def rotate(values: tuple[str, ...], shift: int) -> list[str]:
    n = len(values)
    return [values[(i + shift) % n] for i in range(n)]


def render(surface: str, facts: str, question: str, candidates: list[str]) -> str:
    if surface == "open":
        return f"Field note: {facts} {question} Reply with only the answer phrase."
    choices = "; ".join(candidates)
    return f"A coordinator recorded: {facts} {question} Choices in arbitrary order: {choices}. Return only the answer phrase."


def base_row(panel: str, family: str, domain: str, surface: str, unit: int, cell: str,
             facts: str, question: str, answer: str, candidates: list[str], roles: dict,
             factors: dict) -> dict:
    ordered = rotate(tuple(candidates), (unit + len(cell) + (1 if surface == "options" else 0)) % len(candidates))
    prompt = render(surface, facts, question, ordered)
    case_id = f"c607|{panel}|{family}|{domain}|{surface}|u{unit:02d}|{cell}"
    return {
        "case_id": case_id, "panel": panel, "family": family, "operation_domain": domain,
        "surface": surface, "unit": unit, "partition": partition(unit), "cell": cell,
        "prompt": prompt, "answer": answer, "answer_candidates": ordered,
        "role_values": roles, "factors": factors,
    }


def atomic_case(family: str, surface: str, unit: int, cell: int) -> dict:
    a, b, c, d = NAMES_A[unit], NAMES_B[unit], NAMES_C[unit], NAMES_D[unit]
    i0, i1 = unit % 8, (unit + 1) % 8
    if family == "attribute_binding":
        facts = f"{a}'s travel case is {COLORS[i0]}, while {b}'s travel case is {COLORS[i1]}."
        target = a if cell == 0 else b
        answer = COLORS[i0] if cell == 0 else COLORS[i1]
        question = f"What color is {target}'s travel case?"
        candidates = list(COLORS)
        roles = {"primary": a, "secondary": b, "relation": "travel case", "context": COLORS[i0], "query": target}
    elif family == "attitude_event":
        facts = f"{a} enjoys {ACTIONS[i0]}, whereas {b} enjoys {ACTIONS[i1]}."
        target = a if cell == 0 else b
        answer = ACTIONS[i0] if cell == 0 else ACTIONS[i1]
        question = f"Which activity does {target} enjoy?"
        candidates = list(ACTIONS)
        roles = {"primary": a, "secondary": b, "relation": "enjoys", "context": ACTIONS[i0], "query": target}
    elif family == "comparison":
        facts = f"The {OBJECTS[i0]} is taller than the {OBJECTS[i1]}."
        answer = OBJECTS[i0] if cell == 0 else OBJECTS[i1]
        question = "Which object is taller?" if cell == 0 else "Which object is shorter?"
        candidates = list(OBJECTS)
        roles = {"primary": OBJECTS[i0], "secondary": OBJECTS[i1], "relation": "taller than", "context": OBJECTS[i0], "query": "taller" if cell == 0 else "shorter"}
    elif family == "temporal_order":
        facts = f"The {EVENTS[i0]} happened before the {EVENTS[i1]}."
        answer = EVENTS[i0] if cell == 0 else EVENTS[i1]
        question = "Which event happened earlier?" if cell == 0 else "Which event happened later?"
        candidates = list(EVENTS)
        roles = {"primary": EVENTS[i0], "secondary": EVENTS[i1], "relation": "before", "context": EVENTS[i0], "query": "earlier" if cell == 0 else "later"}
    elif family == "causal_binding":
        facts = f"A {CAUSES[i0]} caused the {EFFECTS[i0]}, while a {CAUSES[i1]} caused the {EFFECTS[i1]}."
        effect = EFFECTS[i0] if cell == 0 else EFFECTS[i1]
        answer = CAUSES[i0] if cell == 0 else CAUSES[i1]
        question = f"What caused the {effect}?"
        candidates = list(CAUSES)
        roles = {"primary": EFFECTS[i0], "secondary": EFFECTS[i1], "relation": "caused", "context": CAUSES[i0], "query": effect}
    elif family == "coreference":
        facts = f"After {a} handed a map to {b}, the former signed the receipt and the latter filed it."
        answer = a if cell == 0 else b
        question = "Who does 'the former' refer to?" if cell == 0 else "Who does 'the latter' refer to?"
        candidates = [a, b, c, d]
        roles = {"primary": a, "secondary": b, "relation": "handed", "context": "receipt", "query": "former" if cell == 0 else "latter"}
    elif family == "part_whole":
        whole0, whole1 = f"{a}'s rover", f"{b}'s drone"
        facts = f"{whole0} includes a {PARTS[i0]}, while {whole1} includes a {PARTS[i1]}."
        target = whole0 if cell == 0 else whole1
        answer = PARTS[i0] if cell == 0 else PARTS[i1]
        question = f"Which component belongs to {target}?"
        candidates = list(PARTS)
        roles = {"primary": whole0, "secondary": whole1, "relation": "includes", "context": PARTS[i0], "query": target}
    elif family == "contrast_focus":
        facts = f"The first bulletin recommends the {ROUTES[i0]}, but the second recommends the {ROUTES[i1]}."
        answer = ROUTES[i0] if cell == 0 else ROUTES[i1]
        question = "Which route does the first bulletin recommend?" if cell == 0 else "Which route does the second bulletin recommend?"
        candidates = list(ROUTES)
        roles = {"primary": "first bulletin", "secondary": "second", "relation": "recommends", "context": ROUTES[i0], "query": "first" if cell == 0 else "second"}
    elif family == "translation":
        facts = f"In this glossary, '{SPANISH[i0]}' means {COLORS[i0]}, and '{SPANISH[i1]}' means {COLORS[i1]}."
        source = SPANISH[i0] if cell == 0 else SPANISH[i1]
        answer = COLORS[i0] if cell == 0 else COLORS[i1]
        question = f"What does '{source}' mean?"
        candidates = list(COLORS)
        roles = {"primary": SPANISH[i0], "secondary": SPANISH[i1], "relation": "means", "context": COLORS[i0], "query": source}
    elif family == "spatial_relation":
        facts = f"The {OBJECTS[i0]} stands to the left of the {OBJECTS[i1]}."
        answer = OBJECTS[i0] if cell == 0 else OBJECTS[i1]
        question = "Which object is on the left?" if cell == 0 else "Which object is on the right?"
        candidates = list(OBJECTS)
        roles = {"primary": OBJECTS[i0], "secondary": OBJECTS[i1], "relation": "left of", "context": OBJECTS[i0], "query": "left" if cell == 0 else "right"}
    elif family == "ownership":
        object0, object1 = f"{COLORS[i0]} lantern", f"{COLORS[i1]} compass"
        facts = f"{a} owns the {object0}, while {b} owns the {object1}."
        target = object0 if cell == 0 else object1
        answer = a if cell == 0 else b
        question = f"Who owns the {target}?"
        candidates = [a, b, c, d]
        roles = {"primary": a, "secondary": b, "relation": "owns", "context": object0, "query": target}
    elif family == "quantity":
        amount0, amount1 = (("twelve", "seven") if unit % 2 == 0 else ("nine", "four"))
        facts = f"{a} stored {amount0} coins, while {b} stored {amount1} coins. The first amount is larger."
        answer = a if cell == 0 else b
        question = "Who stored more coins?" if cell == 0 else "Who stored fewer coins?"
        candidates = [a, b, c, d]
        roles = {"primary": a, "secondary": b, "relation": "stored", "context": amount0, "query": "more" if cell == 0 else "fewer"}
    else:
        raise ValueError(family)
    return base_row("atomic", family, "natural", surface, unit, str(cell), facts, question,
                    answer, candidates, roles, {"cell": cell})


ATOMIC_FAMILIES = (
    "attribute_binding", "attitude_event", "comparison", "temporal_order",
    "causal_binding", "coreference", "part_whole", "contrast_focus",
    "translation", "spatial_relation", "ownership", "quantity",
)


def scope_case(surface: str, unit: int, outer: int, inner: int, query_scope: str) -> dict:
    a, b = NAMES_A[unit], NAMES_B[unit]
    gate = OBJECTS[unit % len(OBJECTS)]
    outer_text = "does not affirm" if outer else "affirms"
    inner_text = "did not open" if inner else "opened"
    facts = f"Regarding the report, {a} {outer_text} that {b} {inner_text} the {gate}."
    if query_scope == "attitude":
        question = f"Is {a}'s affirmation positive?"
        answer = "No" if outer else "Yes"
        query = "affirmation"
    else:
        question = f"Does the embedded clause state that {b} opened the {gate}?"
        answer = "No" if inner else "Yes"
        query = "embedded clause"
    return base_row("scope_factorial", "attitude_scope", query_scope, surface, unit,
                    f"o{outer}i{inner}q{query_scope}", facts, question, answer,
                    ["Yes", "No", "Unknown", "Unclear"],
                    {"primary": a, "secondary": b, "relation": "report", "context": gate, "query": query},
                    {"outer": outer, "inner": inner, "query_scope": query_scope})


NATURAL_CHAINS = (
    ("wren", "bird", "animal", "living thing", "machine"),
    ("salmon", "fish", "animal", "living thing", "mineral"),
    ("orchid", "flower", "plant", "living thing", "vehicle"),
    ("oak", "tree", "plant", "living thing", "instrument"),
    ("violin", "instrument", "artifact", "physical object", "animal"),
    ("hammer", "tool", "artifact", "physical object", "plant"),
    ("sapphire", "gem", "mineral", "physical object", "animal"),
    ("canoe", "boat", "vehicle", "physical object", "flower"),
)


def graph_chain(unit: int, graph_type: str) -> tuple[str, str, str, str, str]:
    if graph_type == "natural":
        return NATURAL_CHAINS[unit]
    return tuple(f"{stem}{unit}" for stem in ("daxor", "mepin", "tavon", "zulic", "qorin"))


def graph_case(surface: str, unit: int, graph_type: str, cell: str) -> dict:
    n0, n1, n2, n3, wrong = graph_chain(unit, graph_type)
    edges = [f"a {n0} is a {n1}"]
    if cell in ("depth2", "depth3", "shortcut", "wrong_middle", "reverse"):
        edges.append(f"a {n1} is a {n2}")
    if cell in ("depth3", "shortcut", "wrong_middle", "reverse"):
        edges.append(f"a {n2} is a {n3}")
    if cell == "shortcut":
        edges.append(f"the {n0} is also explicitly listed as a {n3}")
    if cell == "wrong_middle":
        edges.append(f"the {n0} is displayed beside a {wrong}, but 'beside' is not a type link")
    facts = "; ".join(edges) + "."
    if cell == "depth1":
        answer, question = n1, f"Starting from {n0}, which category is reached after one type link?"
    elif cell == "depth2":
        answer, question = n2, f"Starting from {n0}, which category is reached after two type links?"
    elif cell in ("depth3", "shortcut", "wrong_middle"):
        answer, question = n3, f"Starting from {n0}, which category is reached after three valid type links?"
    else:
        answer, question = n3, f"Which category immediately contains {n2}?"
    return base_row("graph_program", "typed_graph_path", graph_type, surface, unit, cell,
                    facts, question, answer, [n1, n2, n3, wrong],
                    {"primary": n0, "secondary": n1, "relation": "is a", "context": n1, "query": n0 if cell != "reverse" else n2},
                    {"graph_type": graph_type, "cell": cell})


def sequence_case(surface: str, unit: int, cell: str) -> dict:
    a, b = NAMES_A[unit], NAMES_B[unit]
    c0, c1, c2 = COLORS[unit % len(COLORS)], COLORS[(unit + 1) % len(COLORS)], COLORS[(unit + 2) % len(COLORS)]
    if cell == "S0":
        facts = f"{a}'s status light is {c0}, and {b}'s status light is {c1}."
        question, answer = f"What is {a}'s current status color?", c0
    elif cell == "A":
        facts = f"{a}'s status light is {c0}, and {b}'s status light is {c1}."
        question, answer = f"What is {b}'s current status color?", c1
    elif cell == "B":
        facts = f"{a}'s status light was {c0}, then an update changed it to {c2}; {b}'s remains {c1}."
        question, answer = f"What is {a}'s current status color?", c2
    elif cell == "AB":
        facts = f"Focus shifts to {b}; an update changes {b}'s status light from {c1} to {c2}, while {a}'s remains {c0}."
        question, answer = f"What is {b}'s current status color?", c2
    else:
        facts = f"An update changes {a}'s status light from {c0} to {c2}; the final query then shifts to {b}, whose light remains {c1}."
        question, answer = f"What is {b}'s current status color?", c1
    return base_row("sequence_program", "query_context_order", "status", surface, unit, cell,
                    facts, question, answer, list(COLORS),
                    {"primary": a, "secondary": b, "relation": "status light", "context": c0, "query": a if cell in ("S0", "B") else b},
                    {"cell": cell})


def make_material() -> list[dict]:
    rows = []
    for family, surface, unit, cell in itertools.product(ATOMIC_FAMILIES, SURFACES, range(UNITS), (0, 1)):
        rows.append(atomic_case(family, surface, unit, cell))
    for surface, unit, outer, inner, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1), (0, 1), ("attitude", "event")):
        rows.append(scope_case(surface, unit, outer, inner, query_scope))
    for graph_type, surface, unit, cell in itertools.product(("natural", "pseudo"), SURFACES, range(UNITS),
                                                              ("depth1", "depth2", "depth3", "shortcut", "wrong_middle", "reverse")):
        rows.append(graph_case(surface, unit, graph_type, cell))
    for surface, unit, cell in itertools.product(SURFACES, range(UNITS), ("S0", "A", "B", "AB", "BA")):
        rows.append(sequence_case(surface, unit, cell))
    return rows


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    compiled = []
    for row in rows:
        ids = text_core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        candidate_ids = [tokenizer.encode(" " + answer, add_special_tokens=False) for answer in row["answer_candidates"]]
        if not all(candidate_ids):
            raise RuntimeError((row["case_id"], candidate_ids))
        positions = {}
        for role, value in row["role_values"].items():
            spans = compiler.graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids,
                         "gold_position": row["answer_candidates"].index(row["answer"]),
                         "role_positions": positions})
    return compiled


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split()).strip(".,;:!?\"'`()[]")


def generated_prediction(text: str, candidates: list[str]) -> int:
    value = normalize(text)
    matches = []
    for i, candidate in enumerate(candidates):
        target = normalize(candidate)
        if value == target or value.startswith(target + " ") or value.startswith(target + "."):
            matches.append(i)
    return matches[0] if len(matches) == 1 else -1


def batch_candidate_scores(model, device, compiled: list[dict], batch_size: int = 16) -> list[list[float]]:
    expanded = []
    for row_i, item in enumerate(compiled):
        for candidate_i, candidate in enumerate(item["candidate_ids"]):
            expanded.append((row_i, candidate_i, item["prompt_ids"], candidate))
    expanded.sort(key=lambda v: len(v[2]) + len(v[3]))
    scores = [[0.0] * len(item["candidate_ids"]) for item in compiled]
    pad = int(model.config.pad_token_id if model.config.pad_token_id is not None else 0)
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start:start + batch_size]
        width = max(len(prompt) + len(candidate) for _, _, prompt, candidate in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, (_, _, prompt, candidate) in enumerate(batch):
            seq = prompt + candidate
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, :len(seq)] = 1
        pos = mask.long().cumsum(-1) - 1
        pos.masked_fill_(mask == 0, 0)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                           use_cache=False, return_dict=True).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for i, (row_i, candidate_i, prompt, candidate) in enumerate(batch):
            value = 0.0
            for offset, token in enumerate(candidate):
                value += float(logp[i, len(prompt) - 1 + offset, token])
            scores[row_i][candidate_i] = value / len(candidate)
        if start % 512 == 0 or start + len(batch) == len(expanded):
            print(f"[candidate score] {min(start + len(batch), len(expanded))}/{len(expanded)}", flush=True)
    return scores


def greedy_text(model, tokenizer, device, prompt_ids: list[int], max_new_tokens: int = 10) -> str:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                   max_new_tokens=max_new_tokens, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    return tokenizer.decode(generated[0, len(prompt_ids):], skip_special_tokens=True).strip()


def material_path() -> Path:
    return OUTS["C607"] / "material/natural_open_material.jsonl"


def compiled_path() -> Path:
    return OUTS["C607"] / "material/qwen_compiled.jsonl"


def behavior_path() -> Path:
    return OUTS["C607"] / "behavior/qwen_behavior.jsonl"


def qualified_path() -> Path:
    return OUTS["C607"] / "behavior/qualified_slices.json"


def c607() -> None:
    out = begin("C607", {
        "object": "natural and controlled open-generation behavior across broad language programs",
        "families": list(ATOMIC_FAMILIES) + ["attitude_scope", "typed_graph_path", "query_context_order"],
        "controls": ["answer identity", "visible candidate order", "single/multi token", "surface", "unit partition", "scope", "path depth"],
        "behavior": ["full candidate sequence likelihood", "10-token greedy generation", "semantic exact candidate match"],
        "qualification": "candidate and generated accuracy each >=0.75 per panel-family-domain slice",
        "human_blind": "required for external validity but unavailable; template frozen and status pending",
    }, {"C606": load(RESULT / "phase2140_c606_language_transport_independent_audit/analysis/final.json")["all_checks_passed"]})
    rows = make_material()
    ids = [r["case_id"] for r in rows]
    prompts = [r["prompt"] for r in rows]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise RuntimeError("non-unique material")
    model = None
    behavior = []
    try:
        model, tokenizer, device, placement = passport.previous.model_base().load_bf16("qwen3")
        compiled = compile_rows(tokenizer, rows)
        write_rows(compiled_path(), compiled)
        candidate_scores = batch_candidate_scores(model, device, compiled)
        for i, (item, scores) in enumerate(zip(compiled, candidate_scores)):
            text = greedy_text(model, tokenizer, device, item["prompt_ids"])
            gen_pred = generated_prediction(text, item["answer_candidates"])
            behavior.append({
                "case_id": item["case_id"], "panel": item["panel"], "family": item["family"],
                "operation_domain": item["operation_domain"], "surface": item["surface"],
                "unit": item["unit"], "partition": item["partition"],
                "candidate_prediction": int(np.argmax(scores)),
                "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                "candidate_scores": scores, "generated_text": text,
                "generated_prediction": gen_pred, "generated_correct": gen_pred == item["gold_position"],
            })
            if i % 32 == 0 or i + 1 == len(compiled):
                print(f"[C607 generation] {i + 1}/{len(compiled)}", flush=True)
    finally:
        passport.previous.model_base().release_bf16(model)
        gc.collect()
    write_rows(material_path(), rows)
    write_rows(behavior_path(), behavior)
    grouped = defaultdict(list)
    for row in behavior:
        grouped[(row["panel"], row["family"], row["operation_domain"])].append(row)
    slices = {}
    for key, values in sorted(grouped.items()):
        ca = float(np.mean([v["candidate_correct"] for v in values]))
        ga = float(np.mean([v["generated_correct"] for v in values]))
        slices["|".join(key)] = {"rows": len(values), "candidate_accuracy": ca,
                                  "generated_accuracy": ga, "qualified": ca >= BEHAVIOR_GATE and ga >= BEHAVIOR_GATE}
    qualified = sorted(k for k, v in slices.items() if v["qualified"])
    save(qualified_path(), {"gate": BEHAVIOR_GATE, "qualified": qualified, "slices": slices})
    review = out / "external/human_blind_review_template.csv"
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "prompt", "semantic_unique_0_1", "natural_1_5", "reviewer_id", "notes"])
        for row in rows[::max(1, len(rows) // 64)][:64]:
            writer.writerow([row["case_id"], row["prompt"], "", "", "", ""])
    family_counts = {k: sum(r["family"] == k for r in rows) for k in sorted({r["family"] for r in rows})}
    headline = {
        "status": "natural_open_behavior_closed", "rows": len(rows), "unique_prompts": len(set(prompts)),
        "candidate_accuracy": float(np.mean([r["candidate_correct"] for r in behavior])),
        "generated_accuracy": float(np.mean([r["generated_correct"] for r in behavior])),
        "qualified_slices": len(qualified), "total_slices": len(slices), "slices": slices,
        "family_counts": family_counts,
        "partition_counts": {p: sum(r["partition"] == p for r in rows) for p in ("discovery", "confirmation", "lockbox")},
        "human_blind_naturalness": "NA_pending_external_review",
        "review_template": str(review.relative_to(ROOT)),
        "strict_interpretation": "Machine behavior qualification is not human naturalness or proof of the intended algorithm.",
    }
    close("C607", headline, {"rows": len(rows) >= 700, "unique": len(rows) == len(set(ids)) == len(set(prompts)),
                              "behavior_complete": len(behavior) == len(rows), "qualified_nonempty": bool(qualified),
                              "review_not_fabricated": headline["human_blind_naturalness"].startswith("NA"),
                              "finite": finite(headline)}, "C608_all_coordinate_tomography")


def capture_index_path() -> Path:
    return OUTS["C608"] / "raw/hidden_index.jsonl"


def last_path() -> Path:
    return OUTS["C608"] / "raw/role_last.float16.npy"


def mean_path() -> Path:
    return OUTS["C608"] / "raw/role_mean.float16.npy"


def shard_dir() -> Path:
    return OUTS["C608"] / "raw/full_token_shards"


def transition_pairs(index: list[dict]) -> list[dict]:
    by_id = {r["case_id"]: r for r in index}
    pairs = []
    def add(left_id: str, right_id: str, operation: str):
        if left_id in by_id and right_id in by_id:
            left, right = by_id[left_id], by_id[right_id]
            if left["generated_correct"] and right["generated_correct"]:
                pairs.append({"left": left, "right": right, "operation": operation,
                              "partition": left["partition"], "surface": left["surface"],
                              "unit": left["unit"], "domain": left["operation_domain"]})
    for family, surface, unit in itertools.product(ATOMIC_FAMILIES, SURFACES, range(UNITS)):
        add(f"c607|atomic|{family}|natural|{surface}|u{unit:02d}|0",
            f"c607|atomic|{family}|natural|{surface}|u{unit:02d}|1", f"atomic:{family}")
    for surface, unit, inner, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1), ("attitude", "event")):
        add(f"c607|scope_factorial|attitude_scope|{query_scope}|{surface}|u{unit:02d}|o0i{inner}q{query_scope}",
            f"c607|scope_factorial|attitude_scope|{query_scope}|{surface}|u{unit:02d}|o1i{inner}q{query_scope}",
            f"scope:outer:{query_scope}")
    for surface, unit, outer, query_scope in itertools.product(SURFACES, range(UNITS), (0, 1), ("attitude", "event")):
        add(f"c607|scope_factorial|attitude_scope|{query_scope}|{surface}|u{unit:02d}|o{outer}i0q{query_scope}",
            f"c607|scope_factorial|attitude_scope|{query_scope}|{surface}|u{unit:02d}|o{outer}i1q{query_scope}",
            f"scope:inner:{query_scope}")
    for graph_type, surface, unit in itertools.product(("natural", "pseudo"), SURFACES, range(UNITS)):
        prefix = f"c607|graph_program|typed_graph_path|{graph_type}|{surface}|u{unit:02d}|"
        add(prefix + "depth1", prefix + "depth2", f"graph:{graph_type}:one_to_two")
        add(prefix + "depth2", prefix + "depth3", f"graph:{graph_type}:two_to_three")
    for surface, unit in itertools.product(SURFACES, range(UNITS)):
        prefix = f"c607|sequence_program|query_context_order|status|{surface}|u{unit:02d}|"
        add(prefix + "S0", prefix + "A", "sequence:query_switch")
        add(prefix + "S0", prefix + "B", "sequence:overwrite")
    return pairs


def role_state(states: np.ndarray, row: dict, q: int) -> np.ndarray:
    return np.asarray(states[int(row["hidden_index"]), int(q)], dtype=np.float32)


def fit_operator_metrics(states: np.ndarray, pairs: list[dict]) -> tuple[dict, dict, dict[str, np.ndarray]]:
    metrics, by_operation, prototypes = {}, {}, {}
    operations = sorted({p["operation"] for p in pairs})
    for operation in operations:
        train = [p for p in pairs if p["operation"] == operation and p["partition"] == "discovery"]
        test = [p for p in pairs if p["operation"] == operation and p["partition"] == "lockbox"]
        if len(train) < 4 or len(test) < 2:
            continue
        for q in QPOINTS:
            prototypes[f"{operation}|q{q}"] = np.mean(
                [role_state(states, p["right"], q) - role_state(states, p["left"], q) for p in train], axis=0).astype(np.float32)
    for operation in operations:
        op_values = []
        train = [p for p in pairs if p["operation"] == operation and p["partition"] == "discovery"]
        test = [p for p in pairs if p["operation"] == operation and p["partition"] == "lockbox"]
        if len(train) < 4 or len(test) < 2:
            continue
        for q0, q1 in ((8, 16), (16, 24), (24, 32), (32, 37)):
            xtr = np.stack([role_state(states, p["right"], q0) - role_state(states, p["left"], q0) for p in train])
            ytr = np.stack([role_state(states, p["right"], q1) - role_state(states, p["left"], q1) for p in train])
            htr = np.stack([role_state(states, p["left"], q0) for p in train])
            xte = np.stack([role_state(states, p["right"], q0) - role_state(states, p["left"], q0) for p in test])
            yte = np.stack([role_state(states, p["right"], q1) - role_state(states, p["left"], q1) for p in test])
            hte = np.stack([role_state(states, p["left"], q0) for p in test])
            gain, bias = previous.fit_coordinate_affine(xtr, ytr)
            beta = previous.fit_guarded(xtr, htr, ytr)
            distance = np.sum((hte.reshape(len(hte), -1)[:, None] - htr.reshape(len(htr), -1)[None]) ** 2, axis=-1)
            nearest = ytr[np.argmin(distance, axis=1)]
            wrong_op = next((o for o in operations if o != operation and f"{o}|q{q1}" in prototypes), None)
            wrong = np.broadcast_to(prototypes[f"{wrong_op}|q{q1}"], yte.shape) if wrong_op else np.zeros_like(yte)
            values = {
                "identity": previous.metric(xte, yte),
                "mean": previous.metric(np.broadcast_to(ytr.mean(axis=0), yte.shape), yte),
                "affine": previous.metric(xte * gain + bias, yte),
                "guarded": previous.metric(previous.predict_guarded(xte, hte, beta), yte),
                "nearest_state": previous.metric(nearest, yte),
                "wrong_operation": previous.metric(wrong, yte),
                "samples_train": len(train), "samples_test": len(test), "wrong_operation_name": wrong_op,
            }
            candidates = {k: v["nrmse"] for k, v in values.items() if isinstance(v, dict)}
            values["winner"] = min(candidates, key=candidates.get)
            values["gate"] = values["nearest_state"]["nrmse"] <= min(values[n]["nrmse"] for n in ("identity", "mean", "affine", "wrong_operation")) - CONTROL_MARGIN
            key = f"{operation}|q{q0}->q{q1}"
            metrics[key] = values
            op_values.append(values["gate"])
        by_operation[operation] = {"passed": sum(op_values), "total": len(op_values),
                                   "pass_rate": float(np.mean(op_values)) if op_values else 0.0,
                                   "conditional_candidate": len(op_values) == 4 and sum(op_values) >= 3}
    return metrics, by_operation, prototypes


def patched_capture(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor,
                    role_positions: dict, patch_q: int, response: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    base = model.model
    captured = {}
    handles = []
    def patch_hook(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        changed = tensor.clone()
        for role_i, role in enumerate(ROLES):
            at = int(role_positions[role][-1])
            changed[0, at] = changed[0, at] + torch.tensor(response[role_i], dtype=changed.dtype, device=changed.device)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handles.append(base.layers[patch_q - 1].register_forward_hook(patch_hook))
    for q in (24, 32):
        def make_capture(q_value):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[q_value] = tensor.detach()
            return hook
        handles.append(base.layers[q - 1].register_forward_hook(make_capture(q)))
    handles.append(base.norm.register_forward_hook(lambda _m, _a, output: captured.__setitem__(37, output.detach())))
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                           use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    roles = {}
    for q, tensor in captured.items():
        roles[q] = np.stack([tensor[0, int(role_positions[role][-1])].float().cpu().numpy() for role in ROLES])
    return roles, output.logits[0, -1].float().cpu().numpy()


def c608() -> None:
    out = begin("C608", {
        "object": "all-coordinate state-conditioned operator tomography",
        "capture": "qualified slices; all tokens, embedding, every block, final norm, all 2560 coordinates",
        "models": ["identity", "mean", "coordinate affine", "state guarded", "nearest base state", "wrong operation"],
        "gate": "nearest-state NRMSE beats identity, mean, affine and wrong operation by >=0.02 on unit lockbox in >=3/4 transitions",
        "perturbation": "q24 positive/negative doses plus wrong-operation control; all downstream role coordinates retained",
    }, {"C607": final("C607")["all_checks_passed"]})
    rows = read_rows(material_path())
    compiled = read_rows(compiled_path())
    behavior = {r["case_id"]: r for r in read_rows(behavior_path())}
    qualified = set(load(qualified_path())["qualified"])
    selected = [(r, c) for r, c in zip(rows, compiled)
                if f"{r['panel']}|{r['family']}|{r['operation_domain']}" in qualified]
    selected.sort(key=lambda v: len(v[1]["prompt_ids"]))
    n = len(selected)
    estimated = sum(CHECKPOINTS * len(c["prompt_ids"]) * DIM * 2 for _, c in selected) + 2 * n * CHECKPOINTS * len(ROLES) * DIM * 2
    free_before = shutil.disk_usage(RESULT).free
    if free_before < estimated + (8 << 30):
        raise RuntimeError({"free": free_before, "estimated": estimated})
    shard_dir().mkdir(parents=True, exist_ok=True)
    mean_states = np.lib.format.open_memmap(mean_path(), mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    last_states = np.lib.format.open_memmap(last_path(), mode="w+", dtype=np.float16, shape=(n, CHECKPOINTS, len(ROLES), DIM))
    index, ledger, model, hooks, captured = [], [], None, [], []
    try:
        model, tokenizer, device, placement = passport.previous.model_base().load_bf16("qwen3")
        base = model.model
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(hook))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for shard_start in range(0, n, 12):
            items = selected[shard_start:shard_start + 12]
            width = max(len(c["prompt_ids"]) for _, c in items)
            shard_path = shard_dir() / f"shard_{shard_start // 12:04d}.float16.npy"
            shard = np.lib.format.open_memmap(shard_path, mode="w+", dtype=np.float16,
                                              shape=(len(items), CHECKPOINTS, width, DIM))
            for local in range(0, len(items), 4):
                batch = items[local:local + 4]
                ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                weights = torch.zeros((len(batch), len(ROLES), width), dtype=torch.float32, device=device)
                lasts = torch.zeros((len(batch), len(ROLES)), dtype=torch.long, device=device)
                lengths = []
                for i, (_r, item) in enumerate(batch):
                    seq = item["prompt_ids"]; lengths.append(len(seq))
                    ids[i, :len(seq)] = torch.tensor(seq, device=device); mask[i, :len(seq)] = 1
                    for role_i, role in enumerate(ROLES):
                        points = [int(v) for v in item["role_positions"][role]]
                        weights[i, role_i, points] = 1.0 / len(points); lasts[i, role_i] = points[-1]
                pos = mask.long().cumsum(-1) - 1; pos.masked_fill_(mask == 0, 0)
                captured.clear()
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                if len(captured) != CHECKPOINTS:
                    raise RuntimeError(len(captured))
                target_slice = slice(shard_start + local, shard_start + local + len(batch))
                for q, state in enumerate(captured):
                    state32 = state.float()
                    mean_states[target_slice, q] = torch.einsum("brt,btd->brd", weights, state32).cpu().numpy().astype(np.float16)
                    gather = lasts[:, :, None].expand(-1, -1, DIM)
                    last_states[target_slice, q] = torch.gather(state32, 1, gather).cpu().numpy().astype(np.float16)
                    for i, length in enumerate(lengths):
                        shard[local + i, q, :length] = state[i, :length].float().cpu().numpy().astype(np.float16)
                for i, (source, item) in enumerate(batch):
                    b = behavior[source["case_id"]]
                    index.append({"hidden_index": shard_start + local + i, "shard": shard_path.name,
                                  "shard_index": local + i, "length": lengths[i], "case_id": source["case_id"],
                                  "panel": source["panel"], "family": source["family"],
                                  "operation_domain": source["operation_domain"], "surface": source["surface"],
                                  "unit": source["unit"], "partition": source["partition"], "cell": source["cell"],
                                  "factors": source["factors"], "answer": source["answer"],
                                  "role_positions": item["role_positions"], "candidate_correct": b["candidate_correct"],
                                  "generated_correct": b["generated_correct"]})
            shard.flush(); del shard
            mean_states.flush(); last_states.flush()
            ledger.append({"shard": shard_path.name, "rows": len(items), "width": width, "bytes": shard_path.stat().st_size})
            print(f"[C608 capture] {min(shard_start + len(items), n)}/{n}", flush=True)
        write_rows(capture_index_path(), index); save(out / "raw/shard_ledger.json", ledger)
    finally:
        for handle in hooks:
            handle.remove()
        hooks.clear()
        captured.clear()
        passport.previous.model_base().release_bf16(model)
        model = None
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mean_states.flush(); last_states.flush(); del mean_states, last_states
        gc.collect()

    states = np.load(last_path(), mmap_mode="r")
    pairs = transition_pairs(index)
    metrics, by_operation, prototypes = fit_operator_metrics(states, pairs)
    np.savez_compressed(out / "analysis/operator_prototypes.npz", **prototypes)
    operator_candidates = [k for k, v in by_operation.items() if v["conditional_candidate"]]
    compiled_by_id = {r["case_id"]: r for r in compiled}
    tomography_records, tomography_arrays = [], []
    model = None
    try:
        model, tokenizer, device, placement2 = passport.previous.model_base().load_bf16("qwen3")
        for operation in operator_candidates:
            train = [p for p in pairs if p["operation"] == operation and p["partition"] == "discovery"]
            tests = [p for p in pairs if p["operation"] == operation and p["partition"] == "lockbox"][:2]
            wrong_op = next((o for o in operator_candidates if o != operation and f"{o}|q24" in prototypes), None)
            for pair in tests:
                source = pair["left"]
                h = role_state(states, source, 24)
                distances = [np.sum((h - role_state(states, p["left"], 24)) ** 2) for p in train]
                donor = train[int(np.argmin(distances))]
                response = role_state(states, donor["right"], 24) - role_state(states, donor["left"], 24)
                wrong = previous.scaled_like(prototypes[f"{wrong_op}|q24"], response) if wrong_op else response[::-1]
                item = compiled_by_id[source["case_id"]]
                ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
                for label, vector in (("dose_-1", -response), ("dose_-0.5", -.5 * response),
                                      ("dose_0.5", .5 * response), ("dose_1", response), ("wrong_operation", wrong)):
                    role_fields, logits = patched_capture(model, ids, mask, pos, item["role_positions"], 24, vector)
                    arr = np.stack([role_fields[q] for q in (24, 32, 37)]).astype(np.float16)
                    array_i = len(tomography_arrays); tomography_arrays.append(arr)
                    scores = [float(np.mean(logits[candidate])) for candidate in item["candidate_ids"]]
                    tomography_records.append({"operation": operation, "case_id": source["case_id"],
                                                "label": label, "array_index": array_i,
                                                "candidate_prediction": int(np.argmax(scores)), "scores": scores,
                                                "field_rms": float(np.sqrt(np.mean(np.asarray(arr, np.float32) ** 2)))})
    finally:
        passport.previous.model_base().release_bf16(model)
        model = None
        passport.close_mmap(states); del states; gc.collect()
    tomography = np.stack(tomography_arrays) if tomography_arrays else np.empty((0, 3, 6, DIM), np.float16)
    np.save(out / "raw/tomography_role_fields.float16.npy", tomography)
    write_rows(out / "analysis/tomography_records.jsonl", tomography_records)
    headline = {
        "status": "all_coordinate_conditional_tomography_closed", "capture_rows": n,
        "capture_shape": [n, CHECKPOINTS, len(ROLES), DIM], "full_token_shards": len(ledger),
        "full_token_bytes": sum(v["bytes"] for v in ledger), "pair_count": len(pairs),
        "operation_count": len(by_operation), "metrics": metrics, "by_operation": by_operation,
        "operator_candidates": operator_candidates, "tomography_records": len(tomography_records),
        "tomography_shape": list(tomography.shape),
        "strict_interpretation": "Nearest-state wins identify conditional predictive dependence, not a unique guard, Jacobian or causal circuit.",
    }
    close("C608", headline, {"capture": n > 0, "shape": headline["capture_shape"][1:] == [38, 6, 2560],
                              "pairs": len(pairs) >= 200, "metrics": bool(metrics),
                              "tomography": len(tomography_records) > 0 if operator_candidates else True,
                              "finite": finite(headline)}, "C609_language_program_algebra")


def interaction_metric(states: np.ndarray, index: list[dict], q: int) -> dict:
    by_key = {(r["surface"], r["unit"], r["operation_domain"], r["cell"]): r
              for r in index if r["panel"] == "scope_factorial" and r["generated_correct"]}
    train, test = [], []
    for surface, unit, scope in itertools.product(SURFACES, range(UNITS), ("attitude", "event")):
        cells = [by_key.get((surface, unit, scope, f"o{o}i{i}q{scope}")) for o, i in ((0, 0), (1, 0), (0, 1), (1, 1))]
        if all(cells):
            h00, h10, h01, h11 = [role_state(states, row, q) for row in cells]
            value = h11 - h10 - h01 + h00
            (train if partition(unit) == "discovery" else test if partition(unit) == "lockbox" else []).append(value)
    if not train or not test:
        return {"samples": len(test), "gate": False}
    truth = np.stack(test); proto = np.mean(train, axis=0)
    return {"samples": len(test), "prototype": previous.metric(np.broadcast_to(proto, truth.shape), truth),
            "zero": previous.metric(np.zeros_like(truth), truth),
            "gate": previous.metric(np.broadcast_to(proto, truth.shape), truth)["nrmse"] <= 1.0 - CONTROL_MARGIN}


def c609() -> None:
    out = begin("C609", {
        "object": "scope interaction, graph depth and true ordered-update algebra",
        "tests": ["second-order scope interaction", "depth1->2->3 composition", "AB versus BA order residual"],
        "controls": ["zero", "mean additive", "wrong middle state", "wrong order"],
        "claim_boundary": "no curvature, holonomy, group or category name without independent laws",
    }, {"C608": final("C608")["all_checks_passed"]})
    index = read_rows(capture_index_path()); states = np.load(last_path(), mmap_mode="r")
    results = {"scope": {}, "graph": {}, "sequence": {}}
    try:
        for q in (16, 24, 32, 37):
            results["scope"][f"q{q}"] = interaction_metric(states, index, q)
        by_id = {r["case_id"]: r for r in index}
        for graph_type in ("natural", "pseudo"):
            for q in (16, 24, 32, 37):
                train12, train23, train_h2, tests = [], [], [], []
                for surface, unit in itertools.product(SURFACES, range(UNITS)):
                    prefix = f"c607|graph_program|typed_graph_path|{graph_type}|{surface}|u{unit:02d}|"
                    r1, r2, r3 = (by_id.get(prefix + c) for c in ("depth1", "depth2", "depth3"))
                    if not all((r1, r2, r3)) or not all(r["generated_correct"] for r in (r1, r2, r3)):
                        continue
                    v12 = role_state(states, r2, q) - role_state(states, r1, q)
                    v23 = role_state(states, r3, q) - role_state(states, r2, q)
                    if partition(unit) == "discovery":
                        train12.append(v12); train23.append(v23); train_h2.append(role_state(states, r2, q))
                    elif partition(unit) == "lockbox":
                        tests.append((role_state(states, r1, q), role_state(states, r2, q), role_state(states, r3, q), v12))
                if train12 and tests:
                    tr12, tr23, trh2 = np.stack(train12), np.stack(train23), np.stack(train_h2)
                    truth = np.stack([h3 - h1 for h1, _h2, h3, _v12 in tests])
                    additive = np.broadcast_to(tr12.mean(0) + tr23.mean(0), truth.shape)
                    sequential, wrong_middle = [], []
                    for h1, h2, _h3, v12 in tests:
                        dist = np.sum((trh2.reshape(len(trh2), -1) - h2.reshape(1, -1)) ** 2, axis=1)
                        sequential.append(v12 + tr23[int(np.argmin(dist))])
                        wrong_middle.append(v12 + tr23[int(np.argmax(dist))])
                    seq = np.stack(sequential); wrong = np.stack(wrong_middle)
                    sm, am, wm = previous.metric(seq, truth), previous.metric(additive, truth), previous.metric(wrong, truth)
                    results["graph"][f"{graph_type}|q{q}"] = {"samples": len(tests), "sequential": sm,
                                                                     "additive": am, "wrong_middle": wm,
                                                                     "gate": sm["nrmse"] <= min(am["nrmse"], wm["nrmse"]) - CONTROL_MARGIN}
        for q in (16, 24, 32, 37):
            train, test = [], []
            for surface, unit in itertools.product(SURFACES, range(UNITS)):
                prefix = f"c607|sequence_program|query_context_order|status|{surface}|u{unit:02d}|"
                ab, ba = by_id.get(prefix + "AB"), by_id.get(prefix + "BA")
                if not ab or not ba or not (ab["generated_correct"] and ba["generated_correct"]):
                    continue
                residual = role_state(states, ab, q) - role_state(states, ba, q)
                (train if partition(unit) == "discovery" else test if partition(unit) == "lockbox" else []).append(residual)
            if train and test:
                truth = np.stack(test); proto = np.mean(train, axis=0)
                pm = previous.metric(np.broadcast_to(proto, truth.shape), truth); zm = previous.metric(np.zeros_like(truth), truth)
                results["sequence"][f"q{q}"] = {"samples": len(test), "order_residual": pm, "zero": zm,
                                                    "gate": pm["nrmse"] <= zm["nrmse"] - CONTROL_MARGIN}
    finally:
        passport.close_mmap(states); del states
    summary = {}
    for name, ledger in results.items():
        eligible = [v for v in ledger.values() if v.get("samples", 0) > 0]
        summary[name] = {"passed": sum(v.get("gate", False) for v in eligible),
                         "total": len(eligible), "registered_missing": len(ledger) - len(eligible)}
    headline = {"status": "language_program_algebra_closed", "results": results, "summary": summary,
                "associativity": "NA_no_independent_three_operation_parenthesizations",
                "inverse": "NA_pair_reversal_is_algebraically_trivial_and_not_counted",
                "holonomy": "NA_no_independent_closed_loop",
                "strict_interpretation": "A passed ledger is a finite predictive composition candidate, not a global algebra."}
    close("C609", headline, {"scope": bool(results["scope"]), "graph": bool(results["graph"]),
                              "sequence": bool(results["sequence"]), "finite": finite(headline)},
          "C610_generation_causal")


def candidate_scores_from_logits(logits: np.ndarray, candidate_ids: list[list[int]]) -> list[float]:
    return [float(np.mean(logits[candidate])) for candidate in candidate_ids]


def nearest_response(states: np.ndarray, train: list[dict], source: dict, q: int) -> tuple[np.ndarray, dict]:
    h = role_state(states, source, q)
    distances = [np.sum((h - role_state(states, pair["left"], q)) ** 2) for pair in train]
    donor = train[int(np.argmin(distances))]
    return role_state(states, donor["right"], q) - role_state(states, donor["left"], q), donor


def role_mask(response: np.ndarray, keep: list[int]) -> np.ndarray:
    value = np.zeros_like(response)
    value[keep] = response[keep]
    return value


def c610() -> None:
    out = begin("C610", {
        "object": "sample-conditioned bidirectional output, generation-level deletion/rescue and role cuts",
        "eligible": "C608 conditional candidates with generated-correct discovery and lockbox pairs",
        "source_tests": ["q16", "q24", "q32", "q16+q24", "wrong operation", "wrong role", "wrong sign"],
        "target_tests": ["natural", "q24 deletion", "adaptive rescue only after generated failure", "wrong rescue", "single-role cuts"],
        "side_effect": "same patch on unrelated generated-correct prompt",
    }, {"C609": final("C609")["all_checks_passed"]})
    index = read_rows(capture_index_path()); states = np.load(last_path(), mmap_mode="r")
    pairs = transition_pairs(index)
    operators = final("C608")["headline"]["operator_candidates"]
    compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    prototypes_file = np.load(OUTS["C608"] / "analysis/operator_prototypes.npz")
    prototypes = {k: np.asarray(prototypes_file[k], np.float32) for k in prototypes_file.files}; prototypes_file.close()
    prototype_operations = sorted({key.rsplit("|q", 1)[0] for key in prototypes})
    records, model = [], None
    try:
        model, tokenizer, device, placement = passport.previous.model_base().load_bf16("qwen3")
        for operation in operators:
            train = [p for p in pairs if p["operation"] == operation and p["partition"] == "discovery"]
            tests = [p for p in pairs if p["operation"] == operation and p["partition"] == "lockbox"][:2]
            if not train:
                continue
            wrong_op = next((o for o in prototype_operations if o != operation and f"{o}|q24" in prototypes), None)
            for pair in tests:
                for source, target, sign, direction in ((pair["left"], pair["right"], 1.0, "forward"),
                                                         (pair["right"], pair["left"], -1.0, "reverse")):
                    source_item, target_item = compiled[source["case_id"]], compiled[target["case_id"]]
                    response24, donor24 = nearest_response(states, train, source if sign > 0 else target, 24)
                    response16, _ = nearest_response(states, train, source if sign > 0 else target, 16)
                    response32, _ = nearest_response(states, train, source if sign > 0 else target, 32)
                    response24 *= sign; response16 *= sign; response32 *= sign
                    wrong = previous.scaled_like(prototypes[f"{wrong_op}|q24"], response24) if wrong_op else response24[::-1]
                    ids = torch.tensor([source_item["prompt_ids"]], dtype=torch.long, device=device)
                    mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
                    target_answer = target_item["answer"]
                    target_pos_in_source = source_item["answer_candidates"].index(target_answer)
                    identity = tuple(range(len(ROLES))); swapped = (1, 0, 3, 2, 4, 5)
                    interventions = {
                        "zero": [], "q16": [(16, response16, identity)], "q24": [(24, response24, identity)],
                        "q32": [(32, response32, identity)],
                        "q16_q24": [(16, .5 * response16, identity), (24, .5 * response24, identity)],
                        "wrong_operation": [(24, wrong, identity)], "wrong_role": [(24, response24, swapped)],
                        "wrong_sign": [(24, -response24, identity)],
                    }
                    source_values = {}
                    target_state = role_state(states, target, 37)
                    for name, patches in interventions.items():
                        final_state, logits = previous.patched_forward_multi(model, ids, mask, pos, source_item["role_positions"], patches)
                        gathered = np.stack([final_state[int(source_item["role_positions"][role][-1])] for role in ROLES])
                        scores = candidate_scores_from_logits(logits, source_item["candidate_ids"])
                        source_values[name] = {"state_to_target": previous.metric(gathered, target_state),
                                               "candidate_prediction": int(np.argmax(scores)),
                                               "target_margin": scores[target_pos_in_source] - max(v for i, v in enumerate(scores) if i != target_pos_in_source)}
                    generated = previous.patched_greedy_text(model, tokenizer, ids, mask, source_item["role_positions"], interventions["q24"], max_new_tokens=10)
                    source_values["q24"]["generated_text"] = generated
                    source_values["q24"]["generated_target"] = generated_prediction(generated, source_item["answer_candidates"]) == target_pos_in_source

                    target_ids = torch.tensor([target_item["prompt_ids"]], dtype=torch.long, device=device)
                    target_mask = torch.ones_like(target_ids); target_pos = torch.arange(target_ids.shape[1], device=device)[None]
                    natural = previous.patched_greedy_text(model, tokenizer, target_ids, target_mask, target_item["role_positions"], [], max_new_tokens=10)
                    deletion_patch = [(24, -response24, identity)]
                    deletion = previous.patched_greedy_text(model, tokenizer, target_ids, target_mask, target_item["role_positions"], deletion_patch, max_new_tokens=10)
                    natural_ok = generated_prediction(natural, target_item["answer_candidates"]) == target_item["gold_position"]
                    deletion_ok = generated_prediction(deletion, target_item["answer_candidates"]) == target_item["gold_position"]
                    target_values = {"natural_text": natural, "natural_ok": natural_ok,
                                     "deletion_text": deletion, "deletion_ok": deletion_ok,
                                     "rescue_eligible": natural_ok and not deletion_ok}
                    if target_values["rescue_eligible"]:
                        rescue_patches = deletion_patch + [(32, response32, identity)]
                        wrong_rescue_patches = deletion_patch + [(32, previous.scaled_like(prototypes.get(f"{wrong_op}|q32", wrong), response32), identity)]
                        rescue = previous.patched_greedy_text(model, tokenizer, target_ids, target_mask, target_item["role_positions"], rescue_patches, max_new_tokens=10)
                        wrong_rescue = previous.patched_greedy_text(model, tokenizer, target_ids, target_mask, target_item["role_positions"], wrong_rescue_patches, max_new_tokens=10)
                        target_values.update({"rescue_text": rescue,
                                              "rescue_ok": generated_prediction(rescue, target_item["answer_candidates"]) == target_item["gold_position"],
                                              "wrong_rescue_text": wrong_rescue,
                                              "wrong_rescue_ok": generated_prediction(wrong_rescue, target_item["answer_candidates"]) == target_item["gold_position"]})
                    cuts = {}
                    for role_i, role in enumerate(ROLES):
                        _, logits = previous.patched_forward_multi(model, target_ids, target_mask, target_pos,
                                                                  target_item["role_positions"], [(24, -role_mask(response24, [role_i]), identity)])
                        scores = candidate_scores_from_logits(logits, target_item["candidate_ids"])
                        cuts[role] = {"gold_margin": scores[target_item["gold_position"]] - max(v for i, v in enumerate(scores) if i != target_item["gold_position"])}
                    unrelated = next(r for r in index if r["partition"] == "lockbox" and r["family"] != source["family"] and r["generated_correct"])
                    unrelated_item = compiled[unrelated["case_id"]]
                    uid = torch.tensor([unrelated_item["prompt_ids"]], dtype=torch.long, device=device)
                    umask = torch.ones_like(uid); upos = torch.arange(uid.shape[1], device=device)[None]
                    _, ulogits = previous.patched_forward_multi(model, uid, umask, upos, unrelated_item["role_positions"], [(24, response24, identity)])
                    uscores = candidate_scores_from_logits(ulogits, unrelated_item["candidate_ids"])
                    side_ok = int(np.argmax(uscores)) == unrelated_item["gold_position"]
                    records.append({"operation": operation, "direction": direction, "source": source["case_id"],
                                    "target": target["case_id"], "donor": donor24["left"]["case_id"],
                                    "source_values": source_values, "target_values": target_values,
                                    "role_cuts": cuts, "side_effect_control_ok": side_ok,
                                    "wrong_operation_name": wrong_op})
                    print(f"[C610] {len(records)}", flush=True)
    finally:
        passport.previous.model_base().release_bf16(model)
        model = None
        passport.close_mmap(states); del states; gc.collect()
    write_rows(out / "analysis/causal_records.jsonl", records)
    summary = {}
    for operation in sorted({r["operation"] for r in records}):
        op_rows = [r for r in records if r["operation"] == operation]
        summary[operation] = {}
        for direction in ("forward", "reverse"):
            values = [r for r in op_rows if r["direction"] == direction]
            eligible = [r for r in values if r["target_values"]["rescue_eligible"]]
            summary[operation][direction] = {
                "tests": len(values),
                "state_guidance": sum(r["source_values"]["q24"]["state_to_target"]["nrmse"] <= r["source_values"]["zero"]["state_to_target"]["nrmse"] - CONTROL_MARGIN and r["source_values"]["q24"]["state_to_target"]["nrmse"] <= r["source_values"]["wrong_operation"]["state_to_target"]["nrmse"] - CONTROL_MARGIN for r in values),
                "generated_target": sum(r["source_values"]["q24"]["generated_target"] for r in values),
                "deletion_broke_generation": sum(r["target_values"]["rescue_eligible"] for r in values),
                "rescue_eligible": len(eligible),
                "specific_rescue": sum(r["target_values"].get("rescue_ok", False) and not r["target_values"].get("wrong_rescue_ok", False) for r in eligible),
                "side_effect_control_ok": sum(r["side_effect_control_ok"] for r in values),
            }
    totals = {k: sum(d[k] for op in summary.values() for d in op.values()) for k in
              ("tests", "state_guidance", "generated_target", "deletion_broke_generation", "rescue_eligible", "specific_rescue", "side_effect_control_ok")}
    bidirectional = [op for op, dirs in summary.items() if all(dirs[d]["tests"] > 0 and dirs[d]["generated_target"] / dirs[d]["tests"] >= .75 for d in ("forward", "reverse"))]
    headline = {"status": "generation_level_causal_closed", "records": len(records), "summary": summary,
                "totals": totals, "bidirectional_generation_candidates": bidirectional,
                "strict_interpretation": "Rescue is counted only when deletion first broke a naturally correct generation; role cuts are coarse role-level, not minimal coordinate circuits."}
    close("C610", headline, {"records": bool(records) if operators else True,
                              "controls": all("wrong_operation" in r["source_values"] and "role_cuts" in r for r in records),
                              "adaptive_rescue": all(not r["target_values"].get("rescue_ok", False) or r["target_values"]["rescue_eligible"] for r in records),
                              "finite": finite(headline)}, "C611_cross_model_visual_theory")


def run_worker(command: list[str]) -> dict:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    print(result.stdout[-2000:], flush=True)
    if result.returncode:
        print(result.stderr[-4000:], flush=True)
    return {"returncode": result.returncode, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-4000:]}


def register_visual() -> None:
    entry = {"id": "c611_natural_output_compiler_atlas", "title": "C611 Natural Output Compiler Atlas",
             "phase": 2145, "campaign": "C607-C611",
             "path": "vis_data/research_kernel/c611_natural_output_compiler_atlas.json",
             "schema": "ai2050.natural_output_compiler_atlas.v1",
             "description": "Exact token-by-checkpoint all-coordinate field, conditional operation passports, program algebra, generation causal ledger and model-relative topology."}
    catalog = load(CATALOG) if CATALOG.exists() else {"artifacts": []}
    artifacts = catalog.setdefault("artifacts", [])
    artifacts[:] = [item for item in artifacts if item.get("id") != entry["id"]]
    artifacts.append(entry); save(CATALOG, catalog)


def c611() -> None:
    out = begin("C611", {
        "object": "cross-model natural-output qualification and model-relative response topology",
        "models": ["GLM4", "DeepSeek-7B", "Qwen3-14B"],
        "qualification": "candidate and greedy generated accuracy each >=0.75 before HiddenState",
        "comparison": ["relative layer depth", "role RMS topology", "operation confusion", "generation direction"],
        "forbidden": ["same coordinate id", "PCA", "Top-K", "magnitude truncation"],
        "visual": "one exact Qwen full-token field plus every operation prototype coordinate and model-relative representatives",
    }, {"C610": final("C610")["all_checks_passed"]})
    worker = TESTS / "phase2145_c611_natural_cross_model_worker.py"
    q14_worker = TESTS / "phase2145_c611_natural_qwen14_worker.py"
    outputs = {}
    supervisor = {}
    for model_name in ("glm4", "deepseek7b"):
        path = out / f"analysis/{model_name}_worker.json"
        existing = load(path) if path.exists() else None
        if existing and existing.get("status") in ("closed", "behavior_unqualified"):
            supervisor[model_name] = {"returncode": 0, "resumed_from_frozen_worker": True}
        else:
            supervisor[model_name] = run_worker([str(ROOT / ".venv/Scripts/python.exe"), str(worker), "--model", model_name,
                                                 "--material", str(material_path()), "--output", str(path)])
        if not path.exists():
            save(path, {"status": "supervisor_error", "model": model_name, "hiddenstate_ran": False,
                        "functional_candidate": False, **supervisor[model_name]})
        outputs[model_name] = load(path)
    q14_path = out / "analysis/qwen14_worker.json"
    existing = load(q14_path) if q14_path.exists() else None
    if existing and existing.get("status") in ("closed", "behavior_unqualified"):
        supervisor["qwen3_14b"] = {"returncode": 0, "resumed_from_frozen_worker": True}
    else:
        supervisor["qwen3_14b"] = run_worker([str(ROOT / ".venv/Scripts/python.exe"), str(q14_worker),
                                               "--material", str(material_path()), "--output", str(q14_path)])
    if not q14_path.exists():
        save(q14_path, {"status": "supervisor_error", "model": "Qwen3-14B", "hiddenstate_ran": False,
                        "functional_candidate": False, **supervisor["qwen3_14b"]})
    outputs["qwen3_14b"] = load(q14_path)
    print("[C611] frozen workers loaded", flush=True)

    index = read_rows(capture_index_path()); compiled = {r["case_id"]: r for r in read_rows(compiled_path())}
    representative = next(r for r in index if r["partition"] == "lockbox" and r["generated_correct"])
    print(f"[C611] representative {representative['case_id']}", flush=True)
    shard = np.load(shard_dir() / representative["shard"], mmap_mode="r")
    exact = np.array(shard[representative["shard_index"], :, :representative["length"]],
                     dtype=np.float16, copy=True)
    print(f"[C611] exact field {list(exact.shape)}", flush=True)
    prompt_ids = compiled[representative["case_id"]]["prompt_ids"]
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/hf/qwen3-4b"), local_files_only=True,
                                               trust_remote_code=True, use_fast=False)
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    del tokenizer
    print("[C611] tokenizer released", flush=True)
    mmap = getattr(shard, "_mmap", None)
    if mmap is not None: mmap.close()
    del shard
    proto_file = np.load(OUTS["C608"] / "analysis/operator_prototypes.npz")
    prototypes = {k: np.asarray(proto_file[k], np.float16).tolist() for k in proto_file.files}; proto_file.close()
    print(f"[C611] prototypes {len(prototypes)}", flush=True)
    tomography = np.load(OUTS["C608"] / "raw/tomography_role_fields.float16.npy", mmap_mode="r")
    tomo_records = read_rows(OUTS["C608"] / "analysis/tomography_records.jsonl")
    tomo_display = []
    for record in tomo_records[:min(12, len(tomo_records))]:
        tomo_display.append({**record, "role_fields": np.asarray(tomography[record["array_index"]], np.float16).tolist()})
    mmap = getattr(tomography, "_mmap", None)
    if mmap is not None: mmap.close()
    del tomography
    print(f"[C611] tomography {len(tomo_display)}", flush=True)
    worker_atlas = {}
    worker_summary = {}
    for name, value in outputs.items():
        worker_atlas[name] = value.get("representative")
        worker_summary[name] = {k: value.get(k) for k in ("status", "rows", "candidate_accuracy", "generated_accuracy",
                                                          "hiddenstate_ran", "checkpoints", "coordinates", "functional_candidate", "metric_passes", "metric_total")}
    atlas = {
        "schema": "ai2050.natural_output_compiler_atlas.v1", "phase": 2145,
        "coordinate_policy": "all signed physical coordinates; no PCA, Top-K, magnitude threshold, attention or MLP internals",
        "qwen3_4b": {"coordinates": 2560, "checkpoints": 38,
                      "representative": {"case_id": representative["case_id"], "token_ids": prompt_ids,
                                         "tokens": tokens, "shape": list(exact.shape), "field": exact.tolist()},
                      "operation_prototypes": prototypes, "tomography": tomo_display},
        "behavior": final("C607")["headline"],
        "operators": {"by_operation": final("C608")["headline"]["by_operation"],
                      "operator_candidates": final("C608")["headline"]["operator_candidates"]},
        "algebra": final("C609")["headline"]["summary"],
        "causal": {"totals": final("C610")["headline"]["totals"],
                   "bidirectional_generation_candidates": final("C610")["headline"]["bidirectional_generation_candidates"]},
        "cross_model": worker_atlas,
    }
    print("[C611] atlas assembled", flush=True)
    save(VISUAL, atlas); register_visual()
    print(f"[C611] visual saved bytes={VISUAL.stat().st_size}", flush=True)
    cleaned = []
    for path, reason in ((shard_dir(), "exact displayed field and all role tensors retained"),
                         (OUTS["C608"] / "raw/tomography_role_fields.float16.npy", "registered tomography subset and summaries retained")):
        if path.exists():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
            cleaned.append({"path": str(path.relative_to(ROOT)), "bytes": size, "reason": reason})
    for name in ("glm4", "deepseek7b", "qwen3_14b"):
        raw = out / f"raw/{name}/role_last.float16.npy"
        if raw.exists():
            size = raw.stat().st_size; raw.unlink()
            cleaned.append({"path": str(raw.relative_to(ROOT)), "bytes": size, "reason": "model-relative representative and metrics retained in atlas"})
    empirical = {
        "qwen_natural_open": final("C607")["headline"]["generated_accuracy"] >= BEHAVIOR_GATE,
        "conditional_operator": bool(final("C608")["headline"]["operator_candidates"]),
        "unseen_composition": any(v["passed"] > 0 for v in final("C609")["headline"]["summary"].values()),
        "bidirectional_generation": bool(final("C610")["headline"]["bidirectional_generation_candidates"]),
        "generation_necessity": final("C610")["headline"]["totals"]["deletion_broke_generation"] > 0,
        "specific_rescue": final("C610")["headline"]["totals"]["specific_rescue"] > 0,
        "cross_model_functional": sum(bool(v.get("functional_candidate")) for v in outputs.values()) >= 2,
        "human_naturalness": False,
    }
    empirical["new_math"] = all(empirical.values())
    print("[C611] cleanup and empirical ledger complete", flush=True)
    headline = {
        "status": "cross_model_visual_theory_closed", "workers": worker_summary, "worker_supervisor": supervisor,
        "visual": str(VISUAL.relative_to(ROOT)), "visual_bytes": VISUAL.stat().st_size,
        "cleaned": cleaned, "cleaned_bytes": sum(v["bytes"] for v in cleaned),
        "retained_role_mean": str(mean_path().relative_to(ROOT)), "retained_role_last": str(last_path().relative_to(ROOT)),
        "empirical_gates": empirical,
        "theory": {"name": "Conditional Output Field Closure Theory", "principle": "Reuse-Difference-Conditioning",
                   "object": "typed state-conditioned all-coordinate transport with a separate generation boundary",
                   "foundational_math_authorized": empirical["new_math"]},
        "strict_interpretation": "Model-relative signatures are not physical-coordinate conjugacy. Pending human review keeps natural external validity open.",
    }
    close("C611", headline, {"workers": len(outputs) == 3 and all(v.get("status") in ("closed", "behavior_unqualified") for v in outputs.values()),
                              "visual": VISUAL.exists(), "catalog": CATALOG.exists(),
                              "retained_roles": mean_path().exists() and last_path().exists(),
                              "bulk_cleaned": not shard_dir().exists(), "finite": finite(headline)},
          "C612_independent_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*PHASES.keys(), "all"), default="all")
    args = parser.parse_args()
    stages = list(PHASES) if args.stage == "all" else [args.stage]
    for stage in stages:
        globals()[stage.lower()]()


if __name__ == "__main__":
    main()
