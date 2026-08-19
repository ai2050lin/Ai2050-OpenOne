#!/usr/bin/env python3
"""Independent audit for Phase1320 C037 contract and typed event compiler."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1320_c037_event_isomorphism_boundary_contract"
P = OUT / "protocol/preregistration.json"
SOURCE = OUT / "material/frozen_isomorphic_lookup_cases.jsonl"
PAIRS = OUT / "material/frozen_isomorphic_lookup_pairs.jsonl"
NATURAL = OUT / "material/pre_model_semantic_naturalness_review.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
CAL = OUT / "analysis/known_truth_isomorphism_boundary_calibration.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
MAIN = T / "phase1320_c037_event_isomorphism_boundary_contract.py"
SELF = Path(__file__).resolve()


def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def rows(path: Path) -> list[dict[str, Any]]: return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()
def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    protocol, source, pairs, machine, natural, cal, final = load(P), rows(SOURCE), rows(PAIRS), load(MACHINE), load(NATURAL), load(CAL), load(FINAL)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source: grouped.setdefault(row["group_id"], []).append(row)
    changes = {panel: [] for panel in ("active", "matched_null", "record_reorder", "self_repeat")}
    for pair in pairs: changes[pair["panel"]].append(pair["states"][0]["gold_value"] != pair["states"][1]["gold_value"])
    checks = {
        "source_hashes": protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SELF)},
        "parent_terminal": load(T / "result/phase1319_c036_embedding_full_state_field/analysis/final.json").get("authorization") == "close_c036_at_descriptive_field_boundary"
            and load(T / "result/phase1319_c036_embedding_full_state_field/audit/independent_final_audit.json").get("all_checks_passed"),
        "dependency_hashes": protocol["dependencies"] == {
            "c036_final": sha(T / "result/phase1319_c036_embedding_full_state_field/analysis/final.json"),
            "c036_audit": sha(T / "result/phase1319_c036_embedding_full_state_field/audit/independent_final_audit.json")},
        "material_hashes": protocol["material"]["source_sha256"] == sha(SOURCE) and protocol["material"]["pairs_sha256"] == sha(PAIRS)
            and protocol["material"]["naturalness_sha256"] == sha(NATURAL) and protocol["material"]["machine_sha256"] == sha(MACHINE),
        "counts": len(source) == 1152 and len(pairs) == 576 and len(grouped) == 576,
        "balance": Counter(row["partition"] for row in source) == Counter({"discovery": 384, "confirmation": 384, "holdout": 384}),
        "pair_states": all(len(v) == 2 and {x["binding_state"] for x in v} == {0, 1} for v in grouped.values()),
        "active_only_changes": all(changes["active"]) and all(not x for panel in changes if panel != "active" for x in changes[panel]),
        "semantic_unique": all(row["candidates"].count(row["gold_value"]) == 1 and row["assignments"][row["query_entity"]][row["attribute"]] == row["gold_value"] for row in source),
        "new_tokens": not machine["token_audit"]["prior_name_overlap"] and not machine["token_audit"]["prior_value_overlap"],
        "single_tokens": machine["token_audit"]["all_names_single_token"] and machine["token_audit"]["all_values_single_token"] and machine["token_audit"]["all_attributes_single_token"],
        "typed_phi": machine["token_audit"]["phi_bijection_within_pairs"] and machine["token_audit"]["slot_count"] == 10,
        "true_boundary": machine["token_audit"]["true_boundary_strictly_after_string_marker"] and machine["token_audit"]["boundary_definition"] == "last compiled chat-prefix token",
        "shortcuts": max(machine["shortcut_accuracy"].values()) <= 0.5,
        "machine_pass": machine["all_machine_checks_passed"],
        "naturalness": natural["all_checks_passed"] and natural["independent_human_panel"] is False,
        "known_truth": cal["all_gates_passed"] and all(v == 1.0 for v in cal["metrics"].values()),
        "calibration_scope": "not attention-derived" in cal["claim_boundary"],
        "no_attention_probe": "No attention/probe role discovery" in protocol["hard_stops"],
        "behavior_before_hidden": protocol["behavior"]["hidden_states_read"] is False,
        "fixed_failure_stop": protocol["field"]["failure_authorization"] == "close_c037_at_isomorphic_field_boundary" and any("first failed gate" in x for x in protocol["hard_stops"]),
        "final": final["authorization"] == "phase1321_qwen3_behavior_only" and final["all_gates_passed"],
        "no_model": protocol["model_weights_loaded"] is False,
    }
    passed = all(checks.values()); authorization = "run_phase1321_preregistration" if passed else "close_c037_before_model"
    save(AUDIT, {"phase": 1320, "campaign": "C037", "checks": checks, "passed": sum(checks.values()), "total": len(checks),
                 "all_checks_passed": passed, "authorization": authorization,
                 "claim_boundary": "Audits explicit material-truth phi and compiled boundary; does not infer latent semantic roles."})
    print(json.dumps({"passed": sum(checks.values()), "total": len(checks), "authorization": authorization}, ensure_ascii=False))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
