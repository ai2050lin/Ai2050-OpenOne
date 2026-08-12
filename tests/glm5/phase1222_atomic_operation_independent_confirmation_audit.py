#!/usr/bin/env python3
"""Independent audits for Phase 1222 material and revealed behavior."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1222_atomic_operation_independent_confirmation as p


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def world_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "world_id": row["world_id"],
        "split": row["split"],
        "world_index": row["world_index"],
        "local_index": row["local_index"],
        "track": row["track"],
        "objects": row["objects"],
        "relations": row["relations"],
        "values": row["values"],
        "assignments": row["base_assignments"],
        "links": row["links"],
    }


def vocabulary(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    value_set: set[str] = set()
    for row in rows:
        if "values" in row:
            value_set.update(
                value for values in row["values"].values() for value in values
            )
        else:
            value_set.update(
                value
                for fields in row["base_assignments"].values()
                for value in fields.values()
            )
    return {
        "objects": {value for row in rows for value in row["objects"]},
        "relations": {value for row in rows for value in row["relations"]},
        "values": value_set,
    }


def run_preaudit() -> dict[str, Any]:
    protocol = p.read_json(p.PROTOCOL_PATH)
    rows = p.read_jsonl(p.MATERIAL_PATH)
    manifest = p.read_jsonl(p.MANIFEST_PATH)
    token_audit = p.read_json(p.TOKEN_AUDIT_PATH)
    checks: list[dict[str, Any]] = []

    embedded = dict(protocol)
    claimed_protocol_digest = embedded.pop("protocol_digest", None)
    add_check(
        checks,
        "protocol_embedded_digest",
        claimed_protocol_digest == p.digest(embedded),
        claimed_protocol_digest,
    )
    add_check(
        checks,
        "frozen_source_hashes",
        protocol["source_hashes"]["main"] == p.file_sha256(p.SCRIPT)
        and protocol["source_hashes"]["audit"] == p.file_sha256(p.AUDIT_SCRIPT),
        protocol["source_hashes"],
    )
    upstream = p.read_json(p.UPSTREAM_FINAL)
    add_check(
        checks,
        "upstream_identity_and_stop",
        upstream.get("final_digest") == p.EXPECTED_UPSTREAM_FINAL_DIGEST
        and not upstream.get("authorized_next", {}).get("automatic_execution"),
        {"final_digest": upstream.get("final_digest"), "automatic": upstream.get("authorized_next", {}).get("automatic_execution")},
    )
    add_check(
        checks,
        "formal_counts",
        len(rows) == p.EXPECTED_ROWS and len(manifest) == p.EXPECTED_ROWS,
        {"rows": len(rows), "manifest": len(manifest), "expected": p.EXPECTED_ROWS},
    )
    add_check(
        checks,
        "formal_digests",
        protocol["material"]["material_digest"] == p.digest(rows)
        and protocol["material"]["manifest_digest"] == p.digest(manifest)
        and protocol["material"]["tokenizer_audit_digest"] == token_audit["tokenizer_audit_digest"],
        protocol["material"],
    )
    add_check(
        checks,
        "row_identifier_uniqueness",
        len({row["item_id"] for row in rows}) == len(rows)
        and len({row["row_digest"] for row in rows}) == len(rows)
        and all(row["row_digest"] == p.digest({k: v for k, v in row.items() if k != "row_digest"}) for row in rows),
        len({row["item_id"] for row in rows}),
    )
    add_check(
        checks,
        "world_count_and_split_balance",
        len({row["world_id"] for row in rows}) == p.EXPECTED_WORLDS
        and all(len({row["world_id"] for row in rows if row["split"] == split}) == p.WORLDS_PER_SPLIT for split in p.SPLITS),
        {split: len({row["world_id"] for row in rows if row["split"] == split}) for split in p.SPLITS},
    )
    cell_counts = Counter((row["split"], row["track"], row["operation"], row["panel"]) for row in rows)
    expected_cell = p.WORLDS_PER_SPLIT // len(p.TRACKS)
    add_check(
        checks,
        "complete_balanced_factorial",
        len(cell_counts) == len(p.SPLITS) * len(p.TRACKS) * len(p.OPERATIONS) * len(p.PANELS)
        and set(cell_counts.values()) == {expected_cell},
        {"cell_count": len(cell_counts), "counts": sorted(set(cell_counts.values()))},
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    add_check(
        checks,
        "four_panels_per_world_operation",
        len(groups) == p.EXPECTED_WORLDS * len(p.OPERATIONS)
        and all({row["panel"] for row in values} == set(p.PANELS) for values in groups.values()),
        len(groups),
    )
    add_check(
        checks,
        "operation_roles_frozen",
        all(
            row["operation_role"]
            == ("target" if row["operation"] in p.TARGET_OPERATIONS else "diagnostic")
            for row in rows
        ),
        {"targets": list(p.TARGET_OPERATIONS), "diagnostics": list(p.DIAGNOSTIC_OPERATIONS)},
    )

    replay_truth = True
    replay_prompts = True
    for row in rows:
        world = world_from_row(row)
        state = p.operation_state(world, row["operation"], row["panel"])
        replay_truth = replay_truth and all(
            row[key] == state[state_key]
            for key, state_key in (
                ("gold", "gold"),
                ("candidates", "candidates"),
                ("display_assignments", "display_assignments"),
                ("computed_assignments", "computed_assignments"),
                ("derivation", "derivation"),
                ("fingerprint_by_candidate", "fingerprint_by_candidate"),
                ("shorter_path_candidates", "shorter_path_candidates"),
            )
        )
        prompts = p.render_prompts(world, state, row["panel"], row["candidate_order"])
        replay_prompts = replay_prompts and (
            row["candidate_prompt"] == prompts["candidate"]
            and row["open_prompt"] == prompts["open"]
            and row["null_prompt"] == prompts["null"]
        )
    add_check(checks, "truth_derivation_replay", replay_truth, replay_truth)
    add_check(checks, "prompt_replay", replay_prompts, replay_prompts)

    binding_changes = True
    for values in groups.values():
        canonical = next(row for row in values if row["panel"] == "canonical")
        permuted = next(row for row in values if row["panel"] == "binding_permutation")
        binding_changes = binding_changes and canonical["gold"] != permuted["gold"]
        binding_changes = binding_changes and sorted(canonical["candidates"]) == sorted(permuted["candidates"])
        binding_changes = binding_changes and sorted(
            value for fields in canonical["display_assignments"].values() for value in fields.values()
        ) == sorted(value for fields in permuted["display_assignments"].values() for value in fields.values())
    add_check(checks, "same_lexicon_binding_permutation_changes_gold", binding_changes, binding_changes)
    add_check(
        checks,
        "shorter_paths_distinct_from_gold",
        all(row["gold"] not in row["shorter_path_candidates"] for row in rows),
        Counter(row["operation"] for row in rows if row["shorter_path_candidates"]),
    )
    add_check(
        checks,
        "candidate_orders_are_permutations",
        all(sorted(row["candidate_order"]) == sorted(row["candidates"]) for row in rows),
        True,
    )
    permutation_shapes = {
        tuple(row["candidates"].index(value) for value in row["candidate_order"]) for row in rows
    }
    add_check(
        checks,
        "all_24_candidate_permutations_used",
        len(permutation_shapes) == 24,
        len(permutation_shapes),
    )
    position_cells = {
        key: Counter(row["gold_position"] for row in rows if (row["split"], row["track"], row["operation"]) == key)
        for key in {(row["split"], row["track"], row["operation"]) for row in rows}
    }
    add_check(
        checks,
        "gold_position_near_balance",
        all(max(counts.values()) - min(counts.values()) <= 4 and set(counts) == {0, 1, 2, 3} for counts in position_cells.values()),
        {"max_cell_spread": max(max(v.values()) - min(v.values()) for v in position_cells.values())},
    )

    manifest_by_id = {row["item_id"]: row for row in manifest}
    row_manifest_alignment = len(manifest_by_id) == len(rows)
    token_lengths_valid = True
    manifest_digests_valid = True
    prompt_contracts_valid = True
    for row in rows:
        item = manifest_by_id.get(row["item_id"])
        if item is None:
            row_manifest_alignment = False
            continue
        row_manifest_alignment = row_manifest_alignment and item["row_digest"] == row["row_digest"]
        manifest_digests_valid = manifest_digests_valid and item["manifest_row_digest"] == p.digest(
            {key: value for key, value in item.items() if key != "manifest_row_digest"}
        )
        world_lengths = [len(item["world_candidate_token_ids"][value]) for value in row["candidate_order"]]
        null_lengths = [len(item["null_candidate_token_ids"][value]) for value in row["candidate_order"]]
        token_lengths_valid = token_lengths_valid and len(set(world_lengths)) == 1 and world_lengths == null_lengths
        prompt_contracts_valid = prompt_contracts_valid and (
            "CHOICES:" in row["candidate_prompt"]
            and "CHOICES:" not in row["open_prompt"]
            and "NO CURRENT-WORLD RECORDS ARE SUPPLIED" in row["null_prompt"]
        )
    add_check(checks, "row_manifest_alignment", row_manifest_alignment, row_manifest_alignment)
    add_check(checks, "manifest_row_digests", manifest_digests_valid, manifest_digests_valid)
    add_check(checks, "equal_candidate_lengths_world_and_null", token_lengths_valid, token_audit["equal_candidate_token_length_row_rate"])
    add_check(checks, "option_open_null_prompt_contracts", prompt_contracts_valid, prompt_contracts_valid)

    split_vocab: dict[str, dict[str, set[str]]] = {}
    for split in p.SPLITS:
        split_vocab[split] = vocabulary([row for row in rows if row["split"] == split])
    split_disjoint = True
    for index, left in enumerate(p.SPLITS):
        for right in p.SPLITS[index + 1 :]:
            for kind in ("objects", "values"):
                split_disjoint = split_disjoint and not (split_vocab[left][kind] & split_vocab[right][kind])
    add_check(checks, "split_partitioned_objects_and_values", split_disjoint, split_disjoint)

    previous_rows = p.read_jsonl(p1221_material_path())
    previous_vocab = vocabulary(previous_rows)
    current_vocab = vocabulary(rows)
    overlap = {
        key: sorted(previous_vocab[key] & current_vocab[key])
        for key in ("objects", "relations", "values")
    }
    add_check(
        checks,
        "phase1221_exact_vocabulary_disjoint",
        all(not values for values in overlap.values()),
        overlap,
    )
    add_check(
        checks,
        "protocol_has_no_family_gate",
        protocol["atomic_gate"]["family_or_global_conjunction"] is False
        and protocol["atomic_gate"]["failure_closes_only_exact_operation_track"] is True,
        protocol["atomic_gate"],
    )
    add_check(
        checks,
        "behavior_only_instrument_contract",
        all(protocol["interface"][key] is False for key in ("hidden_states", "attentions", "hooks", "interventions")),
        protocol["claim_boundary"],
    )
    add_check(
        checks,
        "tokenizer_audit_embedded_digest",
        token_audit["tokenizer_audit_digest"]
        == p.digest({key: value for key, value in token_audit.items() if key != "tokenizer_audit_digest"}),
        token_audit["tokenizer_audit_digest"],
    )

    report: dict[str, Any] = {
        "phase": p.PHASE,
        "audit_stage": "preaudit",
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "all_checks_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    report["audit_digest"] = p.digest(report)
    p.write_json(p.PREAUDIT_PATH, report)
    return report


def p1221_material_path() -> Path:
    return (
        p.TEST_ROOT
        / "result/phase1221_typed_operation_behavior_and_error_fingerprints/material/typed_worlds.jsonl"
    )


def recompute_behavior_row(
    material: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    sum_map = {key: value["sum_log_probability"] for key, value in row["candidate_scores"].items()}
    mean_map = {key: value["mean_log_probability"] for key, value in row["candidate_scores"].items()}
    null_map = {key: value["sum_log_probability"] for key, value in row["null_candidate_scores"].items()}
    context_map = {key: sum_map[key] - null_map[key] for key in sum_map}
    sum_set = p.argmax_set(sum_map)
    mean_set = p.argmax_set(mean_map)
    null_set = p.argmax_set(null_map)
    context_set = p.argmax_set(context_map)
    sum_prediction = sum_set[0] if len(sum_set) == 1 else None
    context_prediction = context_set[0] if len(context_set) == 1 else None
    null_prediction = null_set[0] if len(null_set) == 1 else None
    ordered = sorted(sum_map.values(), reverse=True)
    return {
        "context_map": context_map,
        "sum_set": sum_set,
        "mean_set": mean_set,
        "null_set": null_set,
        "context_set": context_set,
        "sum_prediction": sum_prediction,
        "context_prediction": context_prediction,
        "null_prediction": null_prediction,
        "candidate_correct": sum_prediction == material["gold"],
        "context_correct": context_prediction == material["gold"],
        "null_gold_prediction": null_prediction == material["gold"],
        "sum_margin": ordered[0] - ordered[1],
        "fingerprint": material["fingerprint_by_candidate"].get(sum_prediction, "tie_or_unregistered"),
    }


def run_result_audit() -> dict[str, Any]:
    protocol = p.read_json(p.PROTOCOL_PATH)
    material = p.read_jsonl(p.MATERIAL_PATH)
    raw = p.read_jsonl(p.RAW_PATH)
    summary = p.read_json(p.RUN_SUMMARY_PATH)
    final = p.read_json(p.FINAL_PATH)
    checks: list[dict[str, Any]] = []
    preaudit = p.read_json(p.PREAUDIT_PATH)
    add_check(checks, "preaudit_passed", preaudit.get("all_checks_passed") is True, preaudit.get("audit_digest"))
    add_check(
        checks,
        "revealed_counts",
        len(raw) == len(material) == p.EXPECTED_ROWS and summary["case_count"] == p.EXPECTED_ROWS,
        {"raw": len(raw), "material": len(material), "summary": summary["case_count"]},
    )
    add_check(
        checks,
        "raw_and_summary_digests",
        summary["raw_digest"] == p.digest(raw)
        and summary["summary_digest"] == p.digest({key: value for key, value in summary.items() if key != "summary_digest"}),
        summary["summary_digest"],
    )
    material_by_id = {row["item_id"]: row for row in material}
    alignment = len({row["item_id"] for row in raw}) == len(raw)
    derived = True
    finite = True
    row_digests = True
    generation_parse = True
    for row in raw:
        source = material_by_id.get(row["item_id"])
        if source is None:
            alignment = False
            continue
        alignment = alignment and row["row_digest"] == source["row_digest"]
        replay = recompute_behavior_row(source, row)
        derived = derived and (
            row["context_adjusted_scores"] == replay["context_map"]
            and row["sum_argmax_set"] == replay["sum_set"]
            and row["mean_argmax_set"] == replay["mean_set"]
            and row["null_argmax_set"] == replay["null_set"]
            and row["context_argmax_set"] == replay["context_set"]
            and row["sum_prediction"] == replay["sum_prediction"]
            and row["context_prediction"] == replay["context_prediction"]
            and row["null_prediction"] == replay["null_prediction"]
            and row["candidate_correct"] == replay["candidate_correct"]
            and row["context_correct"] == replay["context_correct"]
            and row["null_gold_prediction"] == replay["null_gold_prediction"]
            and abs(row["sum_margin"] - replay["sum_margin"]) <= 1e-12
            and row["error_fingerprint"] == replay["fingerprint"]
            and row["sum_mean_argmax_set_agreement"] == (replay["sum_set"] == replay["mean_set"])
        )
        finite = finite and row["all_candidate_scores_finite"] and row["all_null_scores_finite"]
        option_prediction, option_exact = p1220_parse(row["option_generated_text"], source["candidate_order"])
        open_prediction, open_exact = p1220_parse(row["open_generated_text"], source["candidate_order"])
        generation_parse = generation_parse and (
            option_prediction == row["option_generation_prediction"]
            and option_exact == row["option_generation_exact"]
            and open_prediction == row["open_generation_prediction"]
            and open_exact == row["open_generation_exact"]
            and row["option_generation_correct"] == (option_prediction == source["gold"])
            and row["open_generation_correct"] == (open_prediction == source["gold"])
        )
        row_digests = row_digests and row["behavior_row_digest"] == p.digest(
            {key: value for key, value in row.items() if key != "behavior_row_digest"}
        )
    add_check(checks, "material_behavior_alignment", alignment, alignment)
    add_check(checks, "argmax_set_and_context_recomputation", derived, derived)
    add_check(checks, "generation_parser_recomputation", generation_parse, generation_parse)
    add_check(checks, "behavior_row_digests", row_digests, row_digests)
    add_check(checks, "all_scored_logits_finite", finite, finite)

    recomputed = p.summarize_behavior(raw)
    add_check(
        checks,
        "aggregate_and_gate_recomputation",
        recomputed == final["behavior"],
        {"authorized": recomputed["authorized_target_operation_tracks"]},
    )
    diagnostic_keys = [f"{operation}|{track}" for operation in p.DIAGNOSTIC_OPERATIONS for track in p.TRACKS]
    add_check(
        checks,
        "diagnostics_never_authorized",
        all(not final["behavior"]["operation_track_authorization"][key] for key in diagnostic_keys),
        diagnostic_keys,
    )
    target_keys = {f"{operation}|{track}" for operation in p.TARGET_OPERATIONS for track in p.TRACKS}
    authorized = set(final["behavior"]["authorized_target_operation_tracks"])
    add_check(
        checks,
        "authorization_scope_is_atomic_target_only",
        authorized <= target_keys
        and final["authorized_next"]["authorized_operation_tracks"] == sorted(authorized)
        and final["authorized_next"]["automatic_execution"] == bool(authorized),
        sorted(authorized),
    )
    add_check(
        checks,
        "final_digest_and_protocol_binding",
        final["final_digest"] == p.digest({key: value for key, value in final.items() if key != "final_digest"})
        and final["protocol_digest"] == protocol["protocol_digest"]
        and final["run_summary_digest"] == summary["summary_digest"],
        final["final_digest"],
    )
    precision = summary["precision_audit"]
    add_check(
        checks,
        "fp16_nonquantized_execution",
        precision.get("all_parameters_fp16") is True
        and precision.get("quantized_parameter_count", 0) == 0,
        precision,
    )
    add_check(
        checks,
        "claim_boundary_preserved",
        final["claim_boundary"] == protocol["claim_boundary"]
        and final["claim_boundary"]["hidden_state"] is False
        and final["claim_boundary"]["causal"] is False,
        final["claim_boundary"],
    )

    report: dict[str, Any] = {
        "phase": p.PHASE,
        "audit_stage": "result",
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "all_checks_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    report["audit_digest"] = p.digest(report)
    p.write_json(p.RESULT_AUDIT_PATH, report)
    return report


def p1220_parse(text: str, candidates: list[str]) -> tuple[str | None, bool]:
    return p.p1220.parse_generated(text, candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "result"))
    args = parser.parse_args()
    report = run_preaudit() if args.stage == "pre" else run_result_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
