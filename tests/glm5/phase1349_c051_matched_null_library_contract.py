#!/usr/bin/env python3
"""Phase1349: freeze the C051 behavior-qualified matched-null library."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from model_utils import MODEL_CONFIGS

PHASE, CAMPAIGN = 1349, "C051"
OUT = TESTS / "result/phase1349_c051_matched_null_library_contract"
PARENT = TESTS / "result/phase1348_c050_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("qualification", "validation", "confirmation", "holdout")
LABELS = ("amber", "cobalt", "ivory", "scarlet")
SYSTEM = "Answer the stated yes-or-no question. Output only yes or no."
ITEMS = {
    "qualification": {
        "amber": ("lantern", "meadow", "orbit", "kettle"),
        "cobalt": ("ribbon", "harbor", "comet", "basket"),
        "ivory": ("canyon", "typewriter", "pebble", "window"),
        "scarlet": ("forest", "compass", "pillow", "tunnel"),
    },
    "validation": {
        "amber": ("anchor", "gazebo", "chimney", "marble"),
        "cobalt": ("helmet", "lagoon", "parcel", "hourglass"),
        "ivory": ("glacier", "hammock", "acorn", "bridge"),
        "scarlet": ("castle", "feather", "magnet", "pocket"),
    },
    "confirmation": {
        "amber": ("apron", "valley", "rocket", "teacup"),
        "cobalt": ("barrel", "garden", "planet", "whistle"),
        "ivory": ("cabin", "thermos", "oyster", "spiral"),
        "scarlet": ("desert", "fountain", "mirror", "windmill"),
    },
    "holdout": {
        "amber": ("attic", "clover", "engine", "tablet"),
        "cobalt": ("balcony", "island", "ladder", "wallet"),
        "ivory": ("bakery", "jungle", "needle", "tower"),
        "scarlet": ("cellar", "kernel", "notebook", "village"),
    },
}

ROLE_TEMPLATES = {
    "record": (
        'Record card:\nItem: "{record_item}"\nCategory: "{record_label}"\n'
        'Side note: "{note_item}" and "{note_label}".\n'
        'Is the Item exactly "{query_item}" and the Category exactly "{query_label}"?'
    ),
    "prose": (
        'A catalog records "{record_item}" as the Item and "{record_label}" as the Category. '
        'A side note also mentions "{note_item}" and "{note_label}". '
        'Is the recorded Item exactly "{query_item}" and the recorded Category exactly "{query_label}"?'
    ),
}
STATUS_TEMPLATES = {
    "record": (
        'Record card:\nItem: "{query_item}"\nCategory: "{query_label}"\nStatus: {status}.\n'
        'Does the Status say approved?'
    ),
    "prose": (
        'A catalog entry lists Item "{query_item}", Category "{query_label}", and marks its Status as {status}. '
        'Is the stated Status approved?'
    ),
}


def prior_words() -> set[str]:
    found: set[str] = set()
    for path in (TESTS / "result").glob("phase13*/material/frozen_concept_graph.json"):
        try:
            found.update(str(row["word"]) for row in core.load(path).get("concepts", []))
        except Exception:
            pass
    return found


def concepts() -> list[dict]:
    return [
        {
            "word": word,
            "partition": partition,
            "assigned_label": label,
            "semantic_role": "literal record-field value; no real-world membership claim is intended",
        }
        for partition in PARTITIONS
        for label in LABELS
        for word in ITEMS[partition][label]
    ]


def add_role_quartet(rows, partition, label_a, label_b, index, offset, surface, template):
    item_a = ITEMS[partition][label_a][index]
    item_b = ITEMS[partition][label_b][(index + offset) % 4]
    key = f"role_bound_lexical:{partition}:{label_a}__{label_b}:{index}:o{offset}:{surface}"
    cells = (
        ("aa", item_a, label_a, item_b, label_b, item_a, label_a, True, "both_match"),
        ("ab", item_a, label_a, item_b, label_b, item_a, label_b, False, "category_mismatch"),
        ("ba", item_a, label_a, item_b, label_b, item_b, label_a, False, "item_mismatch"),
        ("bb", item_b, label_b, item_a, label_a, item_b, label_b, True, "both_match"),
    )
    for cell, rec_i, rec_l, note_i, note_l, query_i, query_l, truth, mismatch in cells:
        prompt = template.format(
            record_item=rec_i, record_label=rec_l, note_item=note_i, note_label=note_l,
            query_item=query_i, query_label=query_l,
        )
        rows.append({
            "case_id": f"c051-b-{len(rows):05d}", "panel": "role_bound_lexical",
            "partition": partition, "label_pair": f"{label_a}__{label_b}", "pair_index": index,
            "pair_offset": offset, "surface": surface, "quartet_key": key, "cell": cell,
            "mismatch_type": mismatch, "query_item": query_i, "query_label": query_l,
            "record_item": rec_i, "record_label": rec_l, "note_item": note_i, "note_label": note_l,
            "truth": truth, "prompt": prompt + " Output only yes or no.",
            "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
        })


def add_status_quartet(rows, partition, label_a, label_b, index, offset, surface, template):
    item_a = ITEMS[partition][label_a][index]
    item_b = ITEMS[partition][label_b][(index + offset) % 4]
    key = f"explicit_status:{partition}:{label_a}__{label_b}:{index}:o{offset}:{surface}"
    cells = (
        ("aa", item_a, label_a, True), ("ab", item_a, label_b, False),
        ("ba", item_b, label_a, False), ("bb", item_b, label_b, True),
    )
    for cell, query_i, query_l, truth in cells:
        status = "approved" if truth else "rejected"
        prompt = template.format(query_item=query_i, query_label=query_l, status=status)
        rows.append({
            "case_id": f"c051-b-{len(rows):05d}", "panel": "explicit_status",
            "partition": partition, "label_pair": f"{label_a}__{label_b}", "pair_index": index,
            "pair_offset": offset, "surface": surface, "quartet_key": key, "cell": cell,
            "mismatch_type": status, "query_item": query_i, "query_label": query_l,
            "record_item": query_i, "record_label": query_l, "note_item": None, "note_label": None,
            "truth": truth, "status": status, "prompt": prompt + " Output only yes or no.",
            "candidates": ["yes", "no"], "gold_position": 0 if truth else 1,
        })


def cases() -> list[dict]:
    rows: list[dict] = []
    for partition in PARTITIONS:
        for label_a, label_b in combinations(LABELS, 2):
            for index in range(4):
                for offset in (0, 1):
                    for surface, template in ROLE_TEMPLATES.items():
                        add_role_quartet(rows, partition, label_a, label_b, index, offset, surface, template)
                    for surface, template in STATUS_TEMPLATES.items():
                        add_status_quartet(rows, partition, label_a, label_b, index, offset, surface, template)
    return rows


def tokenizer(model_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def compile_for(model_name, rows):
    tok = tokenizer(model_name)
    return [
        {
            "case_id": row["case_id"],
            "prompt_ids": core.chat_ids(tok, SYSTEM, row["prompt"]),
            "candidate_ids": [[int(x) for x in tok.encode(c, add_special_tokens=False)] for c in ("yes", "no")],
            "boundary_position": len(core.chat_ids(tok, SYSTEM, row["prompt"])) - 1,
        }
        for row in rows
    ]


def main():
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c050_behavior" or not parent_audit.get("all_checks_passed"):
        raise RuntimeError("C050 must be formally closed")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1349 already exists")

    concept_rows, case_rows = concepts(), cases()
    groups = defaultdict(list)
    for row in case_rows:
        groups[row["quartet_key"]].append(row)
    selected = {row["word"] for row in concept_rows}
    panel_counts = Counter(row["panel"] for row in case_rows)
    truth_cells = Counter((row["panel"], row["partition"], row["surface"], row["truth"]) for row in case_rows)
    role_rows = [row for row in case_rows if row["panel"] == "role_bound_lexical"]
    zero_models = {
        "always_yes": sum(row["truth"] for row in case_rows) / len(case_rows),
        "always_no": sum(not row["truth"] for row in case_rows) / len(case_rows),
        "role_presence_only": sum(row["truth"] for row in role_rows) / len(role_rows),
        "role_item_only": sum((row["record_item"] == row["query_item"]) == row["truth"] for row in role_rows) / len(role_rows),
        "role_category_only": sum((row["record_label"] == row["query_label"]) == row["truth"] for row in role_rows) / len(role_rows),
        "status_keyword_positive_sentinel": 1.0,
    }
    checks = {
        "parent_closed": True,
        "fresh_items": not (selected & prior_words()),
        "concept_count": len(concept_rows) == 64 and len(selected) == 64,
        "case_count": len(case_rows) == 3072,
        "panel_counts": panel_counts == {"role_bound_lexical": 1536, "explicit_status": 1536},
        "quartets": len(groups) == 768 and all(len(v) == 4 for v in groups.values()),
        "balanced_every_cell": all(value == 96 for value in truth_cells.values()) and len(truth_cells) == 32,
        "role_strings_all_present": all(
            row["query_item"] in row["prompt"] and row["query_label"] in row["prompt"]
            and row["note_item"] in row["prompt"] and row["note_label"] in row["prompt"]
            for row in role_rows
        ),
        "role_mismatch_balance": Counter(row["mismatch_type"] for row in role_rows)
            == {"both_match": 768, "category_mismatch": 384, "item_mismatch": 384},
        "constant_zero_models": zero_models["always_yes"] == 0.5 and zero_models["always_no"] == 0.5,
        "presence_zero_model": zero_models["role_presence_only"] == 0.5,
        "single_role_zero_models": zero_models["role_item_only"] == 0.75
            and zero_models["role_category_only"] == 0.75,
        "semantic_uniqueness": all(
            "exactly" in row["prompt"] if row["panel"] == "role_bound_lexical" else "Status" in row["prompt"]
            for row in case_rows
        ),
        "controlled_naturalness": all(len(row["prompt"].split()) <= 55 for row in case_rows),
    }
    compiled = {model: compile_for(model, case_rows) for model in MODELS}
    checks["tokenizer_compilation"] = all(
        len(rows) == len(case_rows) and all(row["candidate_ids"][0] and row["candidate_ids"][1] for row in rows)
        for rows in compiled.values()
    )

    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "counts": {"concepts": len(concept_rows), "cases": len(case_rows), "quartets": len(groups),
                   "panels": dict(panel_counts)},
        "zero_models": zero_models,
        "human_review": {"independent_blind_review": False,
                         "scope": "machine and investigator audit of controlled English only"},
    }
    core.save(OUT / "audit/pre_model_material_zero_audit.json", preaudit)
    if not preaudit["all_checks_passed"]:
        raise RuntimeError(json.dumps(preaudit, indent=2))

    core.save(OUT / "material/frozen_concept_graph.json", {"schema": "c051.null-library.v1", "concepts": concept_rows})
    core.write_rows(OUT / "material/frozen_cases.jsonl", case_rows)
    for model, rows in compiled.items():
        core.write_rows(OUT / f"compiled/{model}_cases.jsonl", rows)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "research_object": "behavior-qualified reusable semantic null interfaces; no hidden-state claim",
        "models": list(MODELS), "required_common_models": ["qwen3", "glm4"],
        "panels": ["role_bound_lexical", "explicit_status"], "partitions": list(PARTITIONS),
        "behavior_gate": {
            "finite": True, "executor_rank_agreement_min": 1.0, "executor_max_abs_diff_max": 1e-6,
            "panel_overall_accuracy_min": 0.98, "panel_partition_accuracy_min": 0.97,
            "panel_surface_accuracy_min": 0.97, "panel_truth_accuracy_min": 0.97,
            "role_mismatch_accuracy_min": 0.95, "quartet_all_correct_fraction_min": 0.90,
            "required_common_models": ["qwen3", "glm4"], "deepseek_is_supplementary": True,
        },
        "zero_models": zero_models,
        "semantic_contract": {
            "role_bound_lexical": "both query strings occur, but truth requires exact Item and Category role binding",
            "explicit_status": "truth depends only on the explicit approved/rejected Status field",
            "constant_strategy_must_fail": True, "yes_no_balanced_within_each_panel_partition_surface": True,
        },
        "branching": {
            "qwen_or_glm_fail": "close C051 without hidden states and without C052",
            "qwen_and_glm_pass": "authorize a separately frozen C052 formation-clock contract on new material",
            "deepseek_fail": "record model-specific boundary; does not veto Qwen-GLM common library",
        },
        "stop_rule": "After first model reveal, do not change items, labels, templates, panels, partitions, models, gates, or branches.",
        "hidden_state_boundary": "No hidden state, attention, MLP, parameter, probe, or causal experiment is authorized in C051.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1350_c051_null_behavior"
    core.save(OUT / "protocol/preregistration.json", protocol)
    hashes = {
        "material": core.sha(OUT / "material/frozen_cases.jsonl"),
        "concepts": core.sha(OUT / "material/frozen_concept_graph.json"),
        **{model: core.sha(OUT / f"compiled/{model}_cases.jsonl") for model in MODELS},
    }
    core.save(OUT / "protocol/file_hashes.json", hashes)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
