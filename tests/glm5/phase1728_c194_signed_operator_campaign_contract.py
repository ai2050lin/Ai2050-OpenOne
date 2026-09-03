#!/usr/bin/env python3
"""C194: freeze the signed role-checkpoint operator campaign and broad language panels."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1728_c194_signed_operator_campaign_contract"
C193 = RESULT / "phase1727_c193_program_scaffold_failure_decomposition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base

PHASE, CAMPAIGN = 1728, "C194"
DIM, WIDTH = 2560, 256
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
ANCHOR_UNITS = (1, 2, 5, 6)


NATURAL_PROGRAMS = {
    "attitude_event": [
        ("Mira", "Nolan", "carve", "statue", "Tessa", "mend", "sail"),
        ("Ravi", "Elena", "paint", "mural", "Damon", "polish", "vase"),
        ("Sofia", "Caleb", "bake", "tart", "Iris", "fold", "map"),
        ("Jonah", "Priya", "repair", "lantern", "Owen", "weave", "basket"),
        ("Leah", "Marco", "plant", "cedar", "Nina", "tune", "violin"),
        ("Hugo", "Amara", "design", "bridge", "Felix", "pack", "parcel"),
        ("Zara", "Theo", "translate", "letter", "Uma", "clean", "window"),
        ("Evan", "Lina", "inspect", "engine", "Galen", "draw", "portrait"),
    ],
    "agent_patient": [
        ("Mira", "Nolan", "lifted", "crate", "Tessa", "opened", "gate"),
        ("Ravi", "Elena", "repaired", "clock", "Damon", "washed", "cup"),
        ("Sofia", "Caleb", "carried", "lantern", "Iris", "folded", "flag"),
        ("Jonah", "Priya", "measured", "beam", "Owen", "painted", "fence"),
        ("Leah", "Marco", "trimmed", "hedge", "Nina", "tuned", "piano"),
        ("Hugo", "Amara", "mapped", "island", "Felix", "packed", "case"),
        ("Zara", "Theo", "sealed", "envelope", "Uma", "cleaned", "mirror"),
        ("Evan", "Lina", "tested", "sensor", "Galen", "drew", "diagram"),
    ],
    "possession": [
        ("Mira", "Nolan", "owns", "amber key", "Tessa", "keeps", "silver bell"),
        ("Ravi", "Elena", "owns", "red journal", "Damon", "keeps", "blue scarf"),
        ("Sofia", "Caleb", "owns", "brass compass", "Iris", "keeps", "linen bag"),
        ("Jonah", "Priya", "owns", "glass token", "Owen", "keeps", "paper crown"),
        ("Leah", "Marco", "owns", "cedar box", "Nina", "keeps", "ivory comb"),
        ("Hugo", "Amara", "owns", "copper lamp", "Felix", "keeps", "wool cape"),
        ("Zara", "Theo", "owns", "green notebook", "Uma", "keeps", "black ribbon"),
        ("Evan", "Lina", "owns", "steel gauge", "Galen", "keeps", "clay cup"),
    ],
    "location": [
        ("Mira", "Nolan", "rests inside", "lantern", "cabinet", "sail", "locker"),
        ("Ravi", "Elena", "rests inside", "journal", "drawer", "scarf", "trunk"),
        ("Sofia", "Caleb", "rests inside", "compass", "case", "bag", "closet"),
        ("Jonah", "Priya", "rests inside", "token", "vault", "crown", "chest"),
        ("Leah", "Marco", "rests inside", "box", "attic", "comb", "basket"),
        ("Hugo", "Amara", "rests inside", "lamp", "studio", "cape", "wardrobe"),
        ("Zara", "Theo", "rests inside", "notebook", "satchel", "ribbon", "pouch"),
        ("Evan", "Lina", "rests inside", "gauge", "workshop", "cup", "pantry"),
    ],
    "comparison": [
        ("Mira", "Nolan", "arrived before", "Tessa", "Damon", "arrived after", "Elena"),
        ("Ravi", "Elena", "finished before", "Damon", "Sofia", "finished after", "Caleb"),
        ("Sofia", "Caleb", "spoke before", "Iris", "Jonah", "spoke after", "Priya"),
        ("Jonah", "Priya", "departed before", "Owen", "Leah", "departed after", "Marco"),
        ("Leah", "Marco", "registered before", "Nina", "Hugo", "registered after", "Amara"),
        ("Hugo", "Amara", "reported before", "Felix", "Zara", "reported after", "Theo"),
        ("Zara", "Theo", "returned before", "Uma", "Evan", "returned after", "Lina"),
        ("Evan", "Lina", "responded before", "Galen", "Mira", "responded after", "Nolan"),
    ],
    "contrast": [
        ("Mira", "Nolan", "praised", "red vase", "Tessa", "blue bowl", "but"),
        ("Ravi", "Elena", "selected", "oak chair", "Damon", "pine desk", "but"),
        ("Sofia", "Caleb", "visited", "north garden", "Iris", "south tower", "but"),
        ("Jonah", "Priya", "packed", "green coat", "Owen", "white hat", "but"),
        ("Leah", "Marco", "sketched", "stone arch", "Nina", "glass dome", "but"),
        ("Hugo", "Amara", "tested", "large motor", "Felix", "small pump", "but"),
        ("Zara", "Theo", "translated", "old letter", "Uma", "new memo", "but"),
        ("Evan", "Lina", "inspected", "front axle", "Galen", "rear wheel", "but"),
    ],
    "negation": [
        ("Mira", "Nolan", "did not carry", "lantern", "Tessa", "carried", "lantern"),
        ("Ravi", "Elena", "did not open", "journal", "Damon", "opened", "journal"),
        ("Sofia", "Caleb", "did not move", "compass", "Iris", "moved", "compass"),
        ("Jonah", "Priya", "did not polish", "token", "Owen", "polished", "token"),
        ("Leah", "Marco", "did not pack", "box", "Nina", "packed", "box"),
        ("Hugo", "Amara", "did not light", "lamp", "Felix", "lit", "lamp"),
        ("Zara", "Theo", "did not copy", "notebook", "Uma", "copied", "notebook"),
        ("Evan", "Lina", "did not test", "gauge", "Galen", "tested", "gauge"),
    ],
    "causation": [
        ("Mira", "storm", "caused", "outage", "heat", "expansion", "report"),
        ("Ravi", "virus", "caused", "fever", "rain", "puddle", "report"),
        ("Sofia", "surge", "caused", "shutdown", "wind", "motion", "report"),
        ("Jonah", "drought", "caused", "famine", "light", "shadow", "report"),
        ("Leah", "toxin", "caused", "illness", "exercise", "strength", "report"),
        ("Hugo", "collision", "caused", "fracture", "music", "joy", "report"),
        ("Zara", "stress", "caused", "error", "water", "rust", "report"),
        ("Evan", "spark", "caused", "fire", "cold", "frost", "report"),
    ],
    "translation": [
        ("Mira", "luma", "means", "apple", "naro", "river", "glossary"),
        ("Ravi", "savi", "means", "bridge", "keto", "forest", "glossary"),
        ("Sofia", "mepo", "means", "lantern", "tavi", "garden", "glossary"),
        ("Jonah", "riku", "means", "compass", "bena", "window", "glossary"),
        ("Leah", "faro", "means", "violin", "doma", "island", "glossary"),
        ("Hugo", "peli", "means", "engine", "zuno", "letter", "glossary"),
        ("Zara", "wemi", "means", "notebook", "gako", "tower", "glossary"),
        ("Evan", "cora", "means", "sensor", "javi", "basket", "glossary"),
    ],
    "type_chain": [
        ("Mira", "sparrow", "is a kind of", "bird", "animal", "mallet", "tool"),
        ("Ravi", "salmon", "is a kind of", "fish", "animal", "kettle", "appliance"),
        ("Sofia", "cactus", "is a kind of", "plant", "organism", "trumpet", "instrument"),
        ("Jonah", "bicycle", "is a kind of", "vehicle", "machine", "fork", "utensil"),
        ("Leah", "ruby", "is a kind of", "mineral", "material", "falcon", "bird"),
        ("Hugo", "soprano", "is a kind of", "singer", "artist", "cedar", "tree"),
        ("Zara", "tablet", "is a kind of", "device", "machine", "rose", "plant"),
        ("Evan", "cello", "is a kind of", "instrument", "artifact", "otter", "animal"),
    ],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partition(unit: int) -> str:
    return "discovery" if unit < 3 else ("confirmation" if unit < 6 else "fresh")


def ordered(correct: str, wrong: str, order: int):
    return ((f"(A) {correct} (B) {wrong}", 0, [correct, wrong]) if order == 1 else (f"(A) {wrong} (B) {correct}", 1, [wrong, correct]))


def natural_case(program: str, unit: int, values, surface: int, order: int) -> dict:
    outer, a, relation, obj, b, distractor, extra = values
    composition = program in {"attitude_event", "contrast", "negation", "type_chain"}
    if program == "attitude_event":
        correct, wrong, query = a, b, obj
        if surface == 0:
            statement = f"{outer} enjoys watching {a} {relation} the {obj}, while {b} will {distractor} the {extra}."
            question = f"Who will {relation} the {obj}?"
        else:
            statement = f"While {b} will {distractor} the {extra}, {outer} likes seeing {a} {relation} the {obj}."
            question = f"The person who will {relation} the {obj} is which one?"
        primary, secondary, context = outer, a, obj
    elif program == "agent_patient":
        correct, wrong, query = a, b, obj
        statement = (f"{a} {relation} the {obj}, and {b} {distractor} the {extra}." if surface == 0 else f"The {extra} was {distractor} by {b}; the {obj} was {relation} by {a}.")
        question = f"Who handled the {obj}?"
        primary, secondary, context = a, obj, b
    elif program == "possession":
        correct, wrong, query = a, b, obj
        statement = (f"{a} {relation} the {obj}, while {b} {distractor} the {extra}." if surface == 0 else f"The {obj} is owned by {a}; the {extra} is kept by {b}.")
        question = f"Who owns the {obj}?"
        primary, secondary, context = obj, a, b
        if surface == 1:
            relation = "owned by"
    elif program == "location":
        correct, wrong, query = obj, extra, a
        statement = (f"The {relation} {obj} the {a}; the {distractor} {extra} the {b}." if surface == 0 else f"Inside the {obj} is the {a}, whereas inside the {extra} is the {b}.")
        question = f"Where is the {a}?"
        primary, secondary, context = a, obj, b
        if surface == 1:
            relation = "Inside"
    elif program == "comparison":
        correct, wrong, query = a, obj, a
        statement = (f"{a} {relation} {obj}; {b} {distractor} {extra}." if surface == 0 else f"Before {obj}, {a} completed the event; after {extra}, {b} completed the other event.")
        question = f"Who was earlier, {a} or {obj}?"
        primary, secondary, context = a, obj, b
        if surface == 1:
            relation = "Before"
    elif program == "contrast":
        correct, wrong, query = b, a, distractor
        statement = (f"{a} {relation} the {obj}, {extra} {b} {relation} the {distractor}." if surface == 0 else f"Although {a} {relation} the {obj}, {b} {relation} the {distractor}.")
        question = f"Who {relation} the {distractor}?"
        primary, secondary, context = a, b, distractor
        relation = extra if surface == 0 else "Although"
    elif program == "negation":
        correct, wrong, query = b, a, obj
        statement = (f"{a} {relation} the {obj}; {b} {distractor} the {extra}." if surface == 0 else f"It was {b}, not {a}, who {distractor} the {extra}.")
        question = f"Who actually handled the {obj}?"
        primary, secondary, context = a, b, obj
        if surface == 1:
            relation = "not"
    elif program == "causation":
        correct, wrong, query = a, b, obj
        statement = (f"The {a} {relation} the {obj}; the {b} {relation} the {distractor}." if surface == 0 else f"The {obj} resulted from the {a}, whereas the {distractor} resulted from the {b}.")
        question = f"What caused the {obj}?"
        primary, secondary, context = a, obj, b
        if surface == 1:
            relation = "resulted from"
    elif program == "translation":
        correct, wrong, query = obj, distractor, a
        statement = (f"In the {extra}, '{a}' {relation} {obj}, while '{b}' {relation} {distractor}." if surface == 0 else f"The {extra} translates '{a}' as {obj} and '{b}' as {distractor}.")
        question = f"What does '{a}' mean?"
        primary, secondary, context = a, obj, b
        if surface == 1:
            relation = "translates"
    elif program == "type_chain":
        correct, wrong, query = obj, extra, a
        statement = (f"A {a} {relation} a {obj}; a {obj} {relation} an {b}. A {distractor} {relation} a {extra}." if surface == 0 else f"The hierarchy places {a} under {obj} and {obj} under {b}; {distractor} is under {extra}.")
        question = f"Which direct category contains the {a}?"
        primary, secondary, context = a, obj, b
        if surface == 1:
            relation = "under"
    else:
        raise ValueError(program)
    options, gold, candidates = ordered(correct, wrong, order)
    prompt = f"Read the statement. {statement} {question} {options}. Reply with only A or B."
    return {
        "case_id": "", "program": program, "unit": unit, "partition": partition(unit), "surface": surface,
        "order": order, "gold_position": gold, "answer_candidates": candidates, "correct_answer": correct,
        "wrong_answer": wrong, "prompt": prompt, "composition": composition,
        "role_values": {"primary": primary, "secondary": secondary, "relation": relation, "context": context, "query": query},
    }


def natural_material() -> list[dict]:
    rows = []
    for program, units in NATURAL_PROGRAMS.items():
        for unit, values in enumerate(units):
            for surface, order in itertools.product((0, 1), (1, -1)):
                row = natural_case(program, unit, values, surface, order)
                row["case_id"] = f"c194-natural-{len(rows):04d}"
                rows.append(row)
    return rows


def compile_qwen(tokenizer, rows: list[dict]) -> list[dict]:
    candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
    if any(len(x) != 1 for x in candidates):
        raise RuntimeError(candidates)
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, "Answer the question from the supplied statement. Reply exactly A or B.", row["prompt"])
        positions = {}
        for role, value in row["role_values"].items():
            spans = graph_base.name_spans(tokenizer, ids, value)
            if not spans:
                raise RuntimeError((row["case_id"], role, value, row["prompt"]))
            positions[role] = spans[-1] if role == "query" else spans[0]
        positions["boundary"] = [len(ids) - 1]
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidates, "role_positions": positions})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C193 / "audit/independent_final_audit.json")
    natural = natural_material()
    compiled = compile_qwen(graph_base.tokenizer(), natural)
    checks = {
        "authorization": parent["all_checks_passed"] and "C194_change_measurement_object" in parent["authorization"],
        "programs": len(NATURAL_PROGRAMS) == 10,
        "natural_cases": len(natural) == 320,
        "partitions": {p: sum(r["partition"] == p for r in natural) for p in ("discovery", "confirmation", "fresh")} == {"discovery": 120, "confirmation": 120, "fresh": 80},
        "candidate_balance": float(np.mean([r["gold_position"] == 0 for r in natural])) == 0.5,
        "surface_balance": float(np.mean([r["surface"] == 1 for r in natural])) == 0.5,
        "semantic_unique": all(r["correct_answer"] != r["wrong_answer"] for r in natural),
        "roles": all(set(r["role_positions"]) == set(ROLES) for r in compiled),
        "width": max(len(r["prompt_ids"]) for r in compiled) < WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/natural_cases.jsonl", natural)
    core.write_rows(OUT / "compiled/qwen3_natural.jsonl", compiled)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(),
        "status": "signed_operator_campaign_frozen",
        "evidence_audit": {
            "retained": [
                "C172-C193 establish a typed local-response instrument and a stable coarse role/checkpoint scaffold candidate",
                "C192 strict cross-program/unit/phrase family retrieval failed",
                "C193 shows strong relation-phrase conditioning and only exploratory residual organization",
            ],
            "corrected_overclaims": [
                "nearest-neighbor neighborhoods are not mathematical equivalence classes because symmetry and transitivity were not established",
                "relation phrase is nested in family and was not randomized independently, so C193 does not causally locate the failure in wording alone",
                "coarse local propagation can reflect generic residual dynamics or task scaffolding; it is not yet a language-specific universal backbone",
                "C172-C193 did not test the natural sentence 'I like eating apples' or prove a new mathematical theory",
            ],
        },
        "object": "signed role x checkpoint x source-coordinate x target-coordinate response trajectory",
        "qwen_graph_panel": "reuse all 112 behavior-correct C192 anchors; perturb q23 relation-role frozen 64 coordinates; observe q24 and q25 six-role full fields",
        "natural_panel": "10 programs x 8 lexical units x 2 surfaces x 2 candidate orders",
        "natural_programs": list(NATURAL_PROGRAMS),
        "natural_anchor_units": list(ANCHOR_UNITS),
        "partitions": {"discovery": [0, 1, 2], "confirmation": [3, 4, 5], "fresh": [6, 7]},
        "behavior_gates": {"global_min": 0.80, "program_partition_min": 0.65},
        "observation_policy": "behavior failure registers execution-interface missingness but does not stop descriptive observation or other routes",
        "system_identification": {"source_coordinates": 64, "target_coordinates": 2560, "doses": [0.25, 0.5, 1.0], "orthogonal_patterns": 16},
        "model_tournament": ["identity", "global_gain", "role_gain", "coordinate_gain", "role_coordinate_gain", "program_role_coordinate_gain", "family_role_coordinate_gain"],
        "natural_composition_holdout": ["attitude_event", "contrast", "negation", "type_chain"],
        "causal_controls": ["delete_discovery_field", "matched_restore", "wrong_family_restore", "wrong_role_restore", "wrong_checkpoint_restore"],
        "cross_model": "model-specific behavior interfaces, common-correct samples, relative checkpoint and role topology only; never same coordinate ids",
        "theory_upgrade_gate": ["cross vocabulary", "cross paraphrase", "cross program", "full signed trajectory prediction", "unseen composition prediction", "selective deletion and rescue", "two-model functional topology"],
        "forbidden": ["attention", "MLP", "weights", "PCA", "static target-energy threshold tuning", "post-reveal contract changes", "claiming a new theory from descriptive fit"],
        "claim_boundary": "Qwen3 controlled and natural micro-programs plus typed cross-model topology; no complete language mechanism or new mathematics is presupposed",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C195_signed_q23_q24_q25_trajectory_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "programs": list(NATURAL_PROGRAMS), "max_width": max(len(r["prompt_ids"]) for r in compiled)}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
        "material": len(core.rows(OUT / "material/natural_cases.jsonl")) == 320,
        "compiled": len(core.rows(OUT / "compiled/qwen3_natural.jsonl")) == 320,
    }
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": protocol["evidence_audit"], "next_authorization": protocol["authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "close")); args = parser.parse_args()
    contract() if args.command == "contract" else close()


if __name__ == "__main__":
    main()
