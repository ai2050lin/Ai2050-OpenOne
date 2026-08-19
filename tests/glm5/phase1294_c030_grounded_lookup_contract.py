#!/usr/bin/env python3
"""Phase 1294: freeze the corrected C030 grounded lookup contract.

C029 remains immutable and closed.  This new campaign reuses only the selected
functional object and pure generator scaffolding; it uses disjoint lexical
material and a stronger grammar-aware audit before any model weight load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1292_c029_object_attribute_convergence_contract as scaffold  # noqa: E402


PHASE = 1294
CAMPAIGN = "C030"
SCRIPT = Path(__file__).resolve()
AUDITOR = TEST_ROOT / "phase1294_c030_grounded_lookup_contract_audit.py"
SCAFFOLD = TEST_ROOT / "phase1292_c029_object_attribute_convergence_contract.py"
ERRATUM = TEST_ROOT / "result/phase1293_c029_naturalness_erratum/analysis/final.json"
C029_MACHINE = TEST_ROOT / "result/phase1292_c029_object_attribute_convergence_contract/audit/tokenizer_semantic_program_audit.json"
OUT = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_grounded_lookup_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_grammar_type_review.json"
MACHINE_AUDIT = OUT / "audit/tokenizer_semantic_program_audit.json"
INDEPENDENT_AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"

VALUE_BANKS = {
    "discovery": {
        "color": ("teal", "maroon", "beige"),
        "material": ("bamboo", "aluminum", "wool"),
        "location": ("gallery", "garage", "studio"),
        "size": ("petite", "balanced", "enormous"),
        "shape": ("diamond", "crescent", "elliptical"),
        "status": ("verified", "flagged", "scheduled"),
    },
    "confirmation": {
        "color": ("cyan", "magenta", "olive"),
        "material": ("canvas", "brick", "silicone"),
        "location": ("terminal", "archive", "balcony"),
        "size": ("slender", "ordinary", "bulky"),
        "shape": ("pentagonal", "tubular", "wedge"),
        "status": ("cleared", "blocked", "reviewed"),
    },
    "holdout": {
        "color": ("turquoise", "indigo", "peach"),
        "material": ("cardboard", "resin", "sandstone"),
        "location": ("pavilion", "tunnel", "mezzanine"),
        "size": ("microscopic", "midrange", "colossal"),
        "shape": ("octagonal", "arched", "faceted"),
        "status": ("confirmed", "withdrawn", "deferred"),
    },
}

EXTRA_NAMES = (
    "Alicia", "Alison", "Allison", "Audrey", "Blake", "Bradley", "Bryan", "Candace",
    "Caroline", "Carrie", "Casey", "Cecilia", "Cody", "Colin", "Connor", "Curtis",
    "Derek", "Dominic", "Elaine", "Erica", "Erin", "Felix", "Frances", "Francis",
    "Gordon", "Grant", "Gregory", "Hailey", "Harry", "Hazel", "Heidi", "Holly",
    "Hugh", "Jacqueline", "Jade", "Jamie", "Jared", "Jenna", "Jesse", "Jill",
    "Joel", "Johnny", "Joshua", "Julian", "Julie", "Kara", "Karl", "Katherine",
    "Katie", "Kayla", "Kelsey", "Kristen", "Kristin", "Leah", "Leo", "Logan",
    "Lorraine", "Madeline", "Marcus", "Marilyn", "Mason", "Maurice", "Melanie", "Melody",
    "Mia", "Miranda", "Monica", "Morgan", "Natalie", "Neil", "Nina", "Norman",
    "Oscar", "Paige", "Paula", "Perry", "Ralph", "Randy", "Rebecca", "Renee",
    "Robin", "Ruby", "Samantha", "Sara", "Seth", "Sheila", "Spencer", "Stella",
    "Tammy", "Tara", "Taylor", "Todd", "Tracy", "Trevor", "Valerie", "Vanessa",
    "Veronica", "Victor", "Virginia", "Warren", "Whitney", "Zachary",
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    tmp.replace(path)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def corrected_record(entity: str, fields: dict[str, str], surface: str) -> str:
    if surface == "catalog_prose":
        color_article = article(fields["color"])
        shape_article = article(fields["shape"])
        return (
            f"The sample named {entity} has {color_article} {fields['color']} color, is made of {fields['material']}, "
            f"is stored in the {fields['location']} area, is {fields['size']} in size, "
            f"has {shape_article} {fields['shape']} shape, and is marked {fields['status']}."
        )
    return (
        f"{entity} - color: {fields['color']}; material: {fields['material']}; "
        f"storage area: {fields['location']}; size: {fields['size']}; "
        f"shape: {fields['shape']}; status: {fields['status']}."
    )


def corrected_query(attribute_name: str, value: str, surface: str) -> str:
    if surface == "inventory_ledger":
        label = "storage area" if attribute_name == "location" else scaffold.ATTRIBUTE_LEXEME[attribute_name]
        return f"Which listed sample has {label}: {value}?"
    if attribute_name == "color":
        return f"According to the catalog, which sample has {article(value)} {value} color?"
    if attribute_name == "material":
        return f"According to the catalog, which sample is made of {value}?"
    if attribute_name == "location":
        return f"According to the catalog, which sample is stored in the {value} area?"
    if attribute_name == "size":
        return f"According to the catalog, which sample is {value} in size?"
    if attribute_name == "shape":
        return f"According to the catalog, which sample has {article(value)} {value} shape?"
    return f"According to the catalog, which sample is marked {value}?"


def choose_disjoint_names(tokenizer: Any) -> tuple[str, ...]:
    prior = set(load(C029_MACHINE)["token_audit"]["selected_names"])
    pool = tuple(dict.fromkeys((*scaffold.NAME_POOL, *EXTRA_NAMES)))
    eligible = [
        name for name in pool
        if name not in prior and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    needed = len(scaffold.PARTITIONS) * scaffold.PROFILES_PER_PARTITION * 3
    if len(eligible) < needed:
        raise RuntimeError(f"only {len(eligible)} new one-token names; need {needed}")
    return tuple(eligible[:needed])


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_names = set(load(C029_MACHINE)["token_audit"]["selected_names"])
    names = choose_disjoint_names(tokenizer)
    original = {
        "VALUE_BANKS": scaffold.VALUE_BANKS,
        "NAME_POOL": scaffold.NAME_POOL,
        "record_clause": scaffold.record_clause,
        "query_clause": scaffold.query_clause,
    }
    try:
        scaffold.VALUE_BANKS = VALUE_BANKS
        scaffold.NAME_POOL = names
        scaffold.record_clause = corrected_record
        scaffold.query_clause = corrected_query
        rows, token_audit = scaffold.build_cases(tokenizer)
    finally:
        scaffold.VALUE_BANKS = original["VALUE_BANKS"]
        scaffold.NAME_POOL = original["NAME_POOL"]
        scaffold.record_clause = original["record_clause"]
        scaffold.query_clause = original["query_clause"]

    for row in rows:
        row["schema_version"] = "phase1294.c030.case.v1"
        row["case_id"] = "c030-" + digest({
            "group": row["group_id"], "state": row["binding_state"], "prompt": row["candidate_prompt"]
        })[:20]
    token_audit.update({
        "selected_names": list(names),
        "prior_name_overlap": sorted(prior_names & set(names)),
        "new_name_pool_eligible_count": sum(
            name not in prior_names and len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
            for name in tuple(dict.fromkeys((*scaffold.NAME_POOL, *EXTRA_NAMES)))
        ),
    })
    return rows, token_audit


def grammar_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    prototypes: dict[str, dict[str, str]] = {}
    for row in rows:
        prototypes.setdefault(f"{row['surface']}|{row['attribute']}", {
            "candidate_prompt": row["candidate_prompt"],
            "generation_prompt": row["generation_prompt"],
        })
        text = row["candidate_prompt"]
        if "  " in text or text.count("?") != 1 or not text.endswith("Answer:"):
            issues.append({"case_id": row["case_id"], "kind": "surface_form"})
        for match in re.finditer(r"\b(a|an) ([A-Za-z-]+) (color|shape)\b", text):
            if match.group(1) != article(match.group(2)):
                issues.append({"case_id": row["case_id"], "kind": "article_mismatch"})
        if row["surface"] == "catalog_prose" and "is stored in the " in text and " area" not in text:
            issues.append({"case_id": row["case_id"], "kind": "location_collocation"})
        if any(fragment in text.lower() for fragment in (
            "does not apply the", "unassigned alternative the", "stored in the rooftop,", "a azure", "a emerald"
        )):
            issues.append({"case_id": row["case_id"], "kind": "known_c029_defect"})
    return {
        "reviewed_before_any_c030_weight_load": True,
        "reviewer_type": "researcher prototype review plus article/collocation-aware deterministic full replay",
        "independent_human_panel": False,
        "type_signature": "(WorldState, Attribute, Value) -> Entity",
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
        "issues": issues,
        "all_checks_passed": not issues,
        "limitation": "Controlled English passed explicit grammar/type checks; no independent human naturalness panel was available.",
    }


def protocol(rows: list[dict[str, Any]], token_audit: dict[str, Any], program: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "experiment_id": "EXP-C030-WP00-001",
        "schema_version": "phase1294.c030.preregistration.v1",
        "upstream": {
            "c029_erratum_sha256": sha(ERRATUM),
            "c029_authorization": load(ERRATUM)["authorization"],
            "scaffold_sha256": sha(SCAFFOLD),
            "retained_object_only": "object_attribute_inverse_lookup",
            "no_c029_model_output_existed": True,
        },
        "construct": {
            "world_state": "finite explicit map Entity x Attribute -> Value",
            "query": "unique inverse lookup (Attribute, Value) -> Entity",
            "type_signature": "(WorldState, Attribute, Value) -> Entity",
            "operation_requested_from_model": False,
            "gold_source": "explicit mapping, independently recomputed",
        },
        "material": {
            "partitions": list(scaffold.PARTITIONS),
            "profiles_per_partition": scaffold.PROFILES_PER_PARTITION,
            "attributes": list(scaffold.ATTRIBUTES),
            "panels": list(scaffold.PANELS),
            "surfaces": list(scaffold.SURFACES),
            "candidate_orders": list(scaffold.CANDIDATE_ORDERS),
            "binding_states": list(scaffold.BINDING_STATES),
            "case_count": len(rows),
            "independent_profile_count": len(scaffold.PARTITIONS) * scaffold.PROFILES_PER_PARTITION,
            "typed_query_count": len(scaffold.PARTITIONS) * scaffold.PROFILES_PER_PARTITION * len(scaffold.ATTRIBUTES),
            "candidate_sequences": len(rows) * 3,
            "generation_cases": 2 * scaffold.PROFILES_PER_PARTITION * len(scaffold.ATTRIBUTES) * len(scaffold.PANELS) * len(scaffold.SURFACES) * 2,
            "c029_entity_overlap": token_audit["prior_name_overlap"],
            "c029_value_overlap": [],
            "material_sha256": sha(MATERIAL),
            "naturalness_sha256": sha(NATURALNESS),
        },
        "model": {
            "behavior": ["qwen3-4b-fp16-cuda-no-quantization"],
            "other_models_authorized": False,
            "formal_behavior_runs": 1,
            "system_prompt": scaffold.SYSTEM_PROMPT,
            "native_chat_template": True,
            "enable_thinking": False,
        },
        "tokenizer_audit": token_audit,
        "zero_models": program,
        "grammar_type_review": {
            "passed": review["all_checks_passed"],
            "issue_count": len(review["issues"]),
            "independent_human_panel": False,
            "limitation": review["limitation"],
        },
        "thresholds": scaffold.THRESHOLDS,
        "behavior_gate": "all frozen candidate, invariance, generation, finite, and shortcut ledgers pass",
        "hidden_if_behavior_passes": {
            "phase": 1296,
            "object": "same C030 lookup; multi-event residual future-response path",
            "events": ["queried record entity", "queried record value", "query value", "answer boundary"],
            "selection": "earliest adjacent discovery residual-depth band passing active-over-controls transfer",
            "confirmation": "frozen event and depth on confirmation and holdout",
        },
        "failure_and_stop_branches": {
            "phase1294_audit_pass": "authorize_phase1295_qwen3_behavior_only",
            "any_phase1295_ledger_fails": "close_c030_without_hidden",
            "all_phase1295_ledgers_pass": "authorize_phase1296_multievent_response",
            "phase1296_fails": "close_c030_without_path_claim",
            "phase1296_passes": "authorize_phase1297_path_cut_and_independent_rescue",
            "phase1297_fails": "close_c030_with_bounded_response_path_only",
            "phase1297_passes": "complete_c030_qwen_closure; cross-model requires a new contract",
        },
        "freeze_rules": [
            "No C030 model weight may load before this contract and independent audit pass.",
            "No object, material, split, model, zero model, threshold, parser, or stop branch may change after creation.",
            "Candidate behavior and list-free generation must both pass before hidden state measurement.",
            "After unblinding only the preregistered branch may run; every failure closes C030.",
            "No prompt repair, threshold relaxation, surface deletion, seed rerun, or other-model vote is allowed.",
        ],
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR), "scaffold": sha(SCAFFOLD)},
        "model_weights_loaded": False,
    }
    return {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_digest": digest(timeless)}


def build(force: bool) -> None:
    if load(ERRATUM)["authorization"] != "close_c029_before_behavior":
        raise RuntimeError("C029 erratum is not closed")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True)
    rows, token_audit = build_material(tokenizer)
    write_jsonl(MATERIAL, rows)
    review = grammar_review(rows)
    save(NATURALNESS, review)
    program = scaffold.program_audit(rows)
    prior_values = {
        value for partition in scaffold.VALUE_BANKS.values() for values in partition.values() for value in values
    }
    current_values = {
        value for partition in VALUE_BANKS.values() for values in partition.values() for value in values
    }
    machine = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "token_audit": token_audit,
        "program_audit": program,
        "c029_value_overlap": sorted(prior_values & current_values),
        "grammar_type_review_passed": review["all_checks_passed"],
        "all_machine_checks_passed": (
            review["all_checks_passed"]
            and not token_audit["prior_name_overlap"]
            and not (prior_values & current_values)
            and token_audit["all_candidates_single_token"]
            and program["shortcut_ceiling"] <= scaffold.THRESHOLDS["shortcut_program_accuracy_max"]
            and program["active_same_bag_different_gold_pairs"] == program["active_pair_count"]
        ),
    }
    save(MACHINE_AUDIT, machine)
    save(ENVIRONMENT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "model_weights_loaded": False,
        "tokenizer_only": True,
    })
    frozen = protocol(rows, token_audit, program, review)
    save(PROTOCOL, frozen)
    print(canonical({
        "phase": PHASE, "campaign": CAMPAIGN, "cases": len(rows),
        "grammar_issues": len(review["issues"]), "shortcut_ceiling": program["shortcut_ceiling"],
        "protocol_digest": frozen["protocol_digest"],
    }))


def finalize() -> None:
    frozen = load(PROTOCOL)
    audit = load(INDEPENDENT_AUDIT)
    if not audit["all_checks_passed"]:
        raise RuntimeError("independent C030 audit failed")
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "verdict": "corrected_grounded_lookup_contract_frozen_and_independently_audited",
        "protocol_digest": frozen["protocol_digest"],
        "material_sha256": frozen["material"]["material_sha256"],
        "model_weights_loaded": False,
        "audit_passed": True,
        "authorization": "phase1295_qwen3_behavior_only",
    }
    save(FINAL, final)
    print(canonical(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "finalize"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.force) if args.command == "build" else finalize()


if __name__ == "__main__":
    main()
