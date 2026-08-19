#!/usr/bin/env python3
"""Independent zero-model audit for Phase 1297 C031."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))
from model_utils import MODEL_CONFIGS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = TEST_ROOT / "result/phase1297_c031_event_interval_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_event_interval_cases.jsonl"
NATURALNESS = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
AUDIT = OUT / "audit/independent_final_audit.json"
MAIN = TEST_ROOT / "phase1297_c031_event_interval_contract.py"
SCRIPT = Path(__file__).resolve()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows() -> list[dict[str, Any]]:
    with MATERIAL.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    protocol, material = load(PROTOCOL), rows()
    review, machine = load(NATURALNESS), load(MACHINE)
    checks: list[dict[str, Any]] = []
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "phase_campaign", (protocol["phase"], protocol["campaign"]) == (1297, "C031"), [protocol["phase"], protocol["campaign"]])
    add(checks, "material_hash", protocol["material"]["material_sha256"] == sha(MATERIAL), sha(MATERIAL))
    add(checks, "naturalness_hash", protocol["material"]["naturalness_sha256"] == sha(NATURALNESS), sha(NATURALNESS))
    add(checks, "case_count", len(material) == 6912, len(material))
    add(checks, "case_ids_unique", len({row["case_id"] for row in material}) == len(material), len(material))
    dims = Counter((row["partition"], row["panel"], row["surface"], row["binding_state"], row["candidate_order"]) for row in material)
    add(checks, "factorial_cells_balanced", len(dims) == 3 * 4 * 2 * 2 * 3 and len(set(dims.values())) == 1, {"cells": len(dims), "counts": sorted(set(dims.values()))})
    add(checks, "attributes_complete", set(row["attribute"] for row in material) == {"color", "material", "location", "size", "shape", "status"}, sorted({row["attribute"] for row in material}))
    semantic = all(sum(fields[row["attribute"]] == row["target_value"] for fields in row["assignments"].values()) == 1 for row in material)
    gold = all([entity for entity, fields in row["assignments"].items() if fields[row["attribute"]] == row["target_value"]] == [row["gold_candidate"]] for row in material)
    add(checks, "semantic_unique", semantic, semantic)
    add(checks, "gold_recomputes", gold, gold)
    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in material:
        paired[row["group_id"]].append(row)
    pair_ok = all(len(pair) == 2 and {row["binding_state"] for row in pair} == {0, 1} for pair in paired.values())
    add(checks, "pairs_complete", pair_ok and len(paired) == 3456, len(paired))
    active_ok = all(len(pair) == 2 and pair[0]["gold_candidate"] != pair[1]["gold_candidate"] for key, pair in paired.items() if "|active|" in key)
    null_ok = all(len(pair) == 2 and pair[0]["gold_candidate"] == pair[1]["gold_candidate"] for key, pair in paired.items() if "|matched_null|" in key)
    add(checks, "active_changes_gold", active_ok, active_ok)
    add(checks, "matched_null_preserves_gold", null_ok, null_ok)
    add(checks, "all_spans_typed", all(len(row["typed_spans"]["query"]) == 1 and len(row["typed_spans"]["answer_boundary"]) == 1 and len(row["typed_spans"]["records"]) == 3 for row in material), "all rows")
    text_ok = all(row["candidate_prompt"].endswith("Answer:") and row["candidate_prompt"].count("?") == 1 and "  " not in row["candidate_prompt"] for row in material)
    add(checks, "surface_form", text_ok, text_ok)
    article_ok = True
    for row in material:
        for article, word in re.findall(r"\b(a|an) ([A-Za-z-]+) (?:color|shape)\b", row["candidate_prompt"]):
            article_ok &= article == ("an" if word[0].lower() in "aeiou" else "a")
    add(checks, "articles", article_ok, article_ok)
    add(checks, "review_passes", review["all_checks_passed"] and review["semantic_uniqueness_recomputed_for_all_cases"], review)
    add(checks, "machine_passes", machine["all_machine_checks_passed"], machine)
    add(checks, "no_prior_overlap", not protocol["material"]["c029_c030_entity_overlap"] and not protocol["material"]["c029_c030_value_overlap"], protocol["material"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True)
    names = sorted({entity for row in material for entity in row["entities"]})
    add(checks, "names_single_token", len(names) == 72 and all(len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1 for name in names), len(names))
    add(checks, "candidate_suffix_single_token", all(len(tokenizer.encode(" " + candidate, add_special_tokens=False)) == 1 for row in material for candidate in row["candidates"]), "all candidates")
    shortcuts = {
        "candidate_first": sum(row["candidates"][0] == row["gold_candidate"] for row in material) / len(material),
        "record_first": sum(row["record_order"][0] == row["gold_candidate"] for row in material) / len(material),
        "entity_first": sum(row["entities"][0] == row["gold_candidate"] for row in material) / len(material),
    }
    add(checks, "shortcut_ceiling", max(shortcuts.values()) <= 0.70, shortcuts)
    branch = protocol["failure_and_stop_branches"]
    add(checks, "branch_chain_frozen", branch["phase1298_fail"] == "close_c031_as_numerically_unqualified" and branch["phase1300_fail"] == "close_c031_without_path_claim" and branch["phase1301_fail"] == "close_c031_with_bounded_descriptive_path_only", branch)
    add(checks, "event_registry_frozen", protocol["event_registry"]["primary_transfer_events"] == ["user_answer_cue_end", "assistant_answer_boundary"] and protocol["event_registry"]["query_clause_end_is_required_for_description_not_for_path_gate"], protocol["event_registry"])
    add(checks, "weights_not_loaded", protocol["model_weights_loaded"] is False, protocol["model_weights_loaded"])
    add(checks, "calibration_precedes_behavior_hidden", "phase1298" in branch["phase1297_audit_pass"] and "phase1299" in branch["phase1298_pass"] and "phase1300" in branch["phase1299_pass"], branch)
    passed = all(check["passed"] for check in checks)
    result = {
        "phase": 1297,
        "campaign": "C031",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(check["passed"] for check in checks),
        "total_count": len(checks),
        "all_checks_passed": passed,
        "authorization": "phase1298_numerical_calibration_only" if passed else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
