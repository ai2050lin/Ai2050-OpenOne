#!/usr/bin/env python3
"""Independent pre- and post-execution audit for Phase 1221."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS


PHASE = 1221
MAIN_SCRIPT = TEST_ROOT / "phase1221_typed_operation_behavior_and_error_fingerprints.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT_ROOT = TEST_ROOT / "result/phase1221_typed_operation_behavior_and_error_fingerprints"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/typed_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
UPSTREAM_MATERIAL = TEST_ROOT / "result/phase1220_object_relation_value_master_task/material/master_task.jsonl"

SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
TRACKS = ("natural", "symbolic")
PANELS = ("canonical", "record_order", "paraphrase", "matched_carrier")
FAMILIES: dict[str, tuple[str, ...]] = {
    "core": ("direct", "query_object", "query_relation", "binding_swap", "inverse_lookup"),
    "link": ("link_marker_control", "link_object", "link_then_value", "reverse_link_object", "reverse_link_then_value"),
    "compose": ("object_relation_compose", "binding_query_compose", "double_link_relation", "inverse_link_compose", "link_binding_compose"),
}
EXPECTED_WORLDS = 768
EXPECTED_ROWS = 15360
EXPECTED_ROWS_PER_SPLIT = 3840
EXPECTED_WORLDS_PER_FAMILY_SPLIT = 64
EXPECTED_ROWS_PER_OPERATION_TRACK_SPLIT = 128
FINITE_MIN = 1.0
CANDIDATE_MIN = 0.90
GENERATION_MIN = 0.85
WORST_PANEL_MIN = 0.80
SURFACE_GROUP_MIN = 0.75
SUM_MEAN_AGREEMENT_MIN = 1.0
TIE_TOLERANCE = 1e-7
BINDING_OPERATIONS = {"binding_swap", "binding_query_compose", "link_binding_compose"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add(checks: dict[str, bool], name: str, value: Any) -> None:
    checks[name] = bool(value)


def validate_embedded(value: dict[str, Any], key: str) -> bool:
    expected = value.get(key)
    candidate = {name: item for name, item in value.items() if name != key}
    return isinstance(expected, str) and digest(candidate) == expected


def predecessor(links: dict[str, str], target: str) -> str:
    values = [source for source, linked in links.items() if linked == target]
    if len(values) != 1:
        raise RuntimeError("non-unique predecessor")
    return values[0]


def recompute_gold(row: dict[str, Any]) -> str:
    objects = row["objects"]
    relations = row["relations"]
    assignments = row["rendered_assignments"]
    links = row["links"]
    operation = row["operation"]
    source, second, third = objects[:3]
    relation, other_relation = relations[:2]

    if operation in {"direct", "query_relation", "binding_swap", "link_marker_control"}:
        return assignments[source][row["target_relation"]]
    if operation == "query_object":
        return assignments[second][relation]
    if operation == "inverse_lookup":
        target_value = assignments[third][relation]
        matches = [obj for obj in objects if assignments[obj][relation] == target_value]
        if len(matches) != 1:
            raise RuntimeError("non-unique inverse lookup")
        return matches[0]
    if operation == "link_object":
        return links[source]
    if operation == "link_then_value":
        return assignments[links[source]][relation]
    if operation == "reverse_link_object":
        return predecessor(links, source)
    if operation == "reverse_link_then_value":
        return assignments[predecessor(links, source)][relation]
    if operation in {"object_relation_compose", "binding_query_compose"}:
        return assignments[second][other_relation]
    if operation == "double_link_relation":
        return assignments[links[links[source]]][relation]
    if operation == "inverse_link_compose":
        target_value = assignments[third][other_relation]
        found = [obj for obj in objects if assignments[obj][other_relation] == target_value]
        if len(found) != 1:
            raise RuntimeError("non-unique composed inverse lookup")
        return assignments[links[found[0]]][relation]
    if operation == "link_binding_compose":
        return assignments[links[source]][relation]
    raise KeyError(operation)


def render_native(tokenizer: Any, system_prompt: str, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def world_signature(row: dict[str, Any]) -> str:
    return digest([row["objects"], row["relations"], row["base_assignments"], row["links"]])


def preaudit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    checks: dict[str, bool] = {}

    add(checks, "protocol_embedded_digest", validate_embedded(protocol, "protocol_digest"))
    add(checks, "main_hash_frozen", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT))
    add(checks, "audit_hash_frozen", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT))
    add(checks, "phase", protocol.get("phase") == PHASE)
    add(checks, "row_count", len(rows) == EXPECTED_ROWS)
    add(checks, "manifest_count", len(manifest) == EXPECTED_ROWS)
    add(checks, "world_count", len({row["world_id"] for row in rows}) == EXPECTED_WORLDS)
    add(checks, "unique_item_ids", len({row["item_id"] for row in rows}) == EXPECTED_ROWS)
    add(checks, "material_digest", protocol["material"]["material_digest"] == digest(rows))
    add(checks, "manifest_digest", protocol["material"]["manifest_digest"] == digest(manifest))
    add(checks, "token_audit_embedded", validate_embedded(token_audit, "tokenizer_audit_digest"))
    add(checks, "token_audit_match", protocol["material"]["tokenizer_audit_digest"] == token_audit["tokenizer_audit_digest"])
    add(checks, "candidate_lengths_equal", token_audit["equal_candidate_token_length_row_rate"] == 1.0)
    add(checks, "all_rows_generated", all(row["generation_required"] for row in rows))

    material_by_id = {row["item_id"]: row for row in rows}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    add(checks, "manifest_items_exact", set(material_by_id) == set(manifest_by_id))

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        add(checks, f"{split}_row_count", len(split_rows) == EXPECTED_ROWS_PER_SPLIT)
        add(checks, f"{split}_world_count", len({row["world_id"] for row in split_rows}) == EXPECTED_WORLDS // len(SPLITS))
        for family, operations in FAMILIES.items():
            family_rows = [row for row in split_rows if row["family"] == family]
            add(checks, f"{split}_{family}_worlds", len({row["world_id"] for row in family_rows}) == EXPECTED_WORLDS_PER_FAMILY_SPLIT)
            for track in TRACKS:
                for operation in operations:
                    cell = [row for row in family_rows if row["track"] == track and row["operation"] == operation]
                    add(checks, f"cell_{split}_{family}_{track}_{operation}", len(cell) == EXPECTED_ROWS_PER_OPERATION_TRACK_SPLIT)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    world_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    permutation_cells: dict[tuple[str, str, str, str], set[tuple[int, ...]]] = defaultdict(set)
    truth_ok = True
    row_digest_ok = True
    binding_semantics_ok = True
    carrier_ok = True
    candidate_set_ok = True
    prompt_binding_ok = True
    for row in rows:
        groups[row["group_id"]].append(row)
        world_members[row["world_id"]].append(row)
        row_digest_ok = row_digest_ok and validate_embedded(row, "row_digest")
        truth_ok = truth_ok and recompute_gold(row) == row["gold"]
        candidate_set_ok = candidate_set_ok and len(set(row["candidates"])) == 4 and row["gold"] in row["candidates"]
        permutation = tuple(row["candidates"].index(candidate) for candidate in row["candidate_order"])
        permutation_cells[(row["split"], row["family"], row["track"], row["operation"])].add(permutation)
        is_binding = row["operation"] in BINDING_OPERATIONS
        binding_semantics_ok = binding_semantics_ok and (
            (is_binding and row["display_assignments"] != row["rendered_assignments"] and row["transformation_instruction"])
            or (not is_binding and row["display_assignments"] == row["rendered_assignments"] and row["transformation_instruction"] is None)
        )
        prompt_binding_ok = prompt_binding_ok and (
            not is_binding or row["transformation_instruction"] in row["prompt"]
        )
        if row["panel"] == "matched_carrier":
            carrier_ok = carrier_ok and "catalog seal" in row["prompt"].lower()
        else:
            carrier_ok = carrier_ok and "catalog seal" not in row["prompt"].lower()
    add(checks, "row_embedded_digests", row_digest_ok)
    add(checks, "truth_independently_recomputed", truth_ok)
    add(checks, "candidate_sets_valid", candidate_set_ok)
    add(checks, "binding_is_executed_not_prerendered", binding_semantics_ok and prompt_binding_ok)
    add(checks, "matched_carrier_is_panel_specific", carrier_ok)
    add(
        checks,
        "surface_groups_complete",
        len(groups) == EXPECTED_WORLDS * 5
        and all(
            len(values) == len(PANELS)
            and {row["panel"] for row in values} == set(PANELS)
            and len({row["gold"] for row in values}) == 1
            for values in groups.values()
        ),
    )
    add(checks, "world_operation_completeness", all(len(values) == 20 for values in world_members.values()))
    add(checks, "all_24_permutations_each_cell", all(len(values) == math.factorial(4) for values in permutation_cells.values()))

    signatures_by_split: dict[str, set[str]] = defaultdict(set)
    world_signatures = set()
    for world_rows in world_members.values():
        signature = world_signature(world_rows[0])
        world_signatures.add(signature)
        signatures_by_split[world_rows[0]["split"]].add(signature)
    add(checks, "all_world_contents_unique", len(world_signatures) == EXPECTED_WORLDS)
    add(
        checks,
        "split_world_contents_disjoint",
        all(not (signatures_by_split[left] & signatures_by_split[right]) for left, right in itertools.combinations(SPLITS, 2)),
    )

    upstream_rows = read_jsonl(UPSTREAM_MATERIAL)
    old_objects = {value for row in upstream_rows for value in row.get("entities", [])}
    old_candidates = {value for row in upstream_rows for value in row.get("candidates", [])}
    new_objects = {value for row in rows for value in row["objects"]}
    new_candidates = {value for row in rows for value in row["candidates"]}
    add(checks, "new_objects_disjoint_from_phase1220", not (old_objects & new_objects))
    add(checks, "new_candidates_disjoint_from_phase1220", not (old_candidates & new_candidates))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    tokenizer_ok = True
    equal_lengths_ok = True
    manifest_digest_ok = True
    for index, manifest_row in enumerate(manifest):
        material = material_by_id[manifest_row["item_id"]]
        rendered = render_native(tokenizer, protocol["interface"]["system_prompt"], material["prompt"])
        base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        tokenizer_ok = tokenizer_ok and base == manifest_row["input_ids"]
        lengths = set()
        for candidate in material["candidate_order"]:
            extended = [int(value) for value in tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)]
            suffix = extended[len(base) :]
            tokenizer_ok = tokenizer_ok and extended[: len(base)] == base
            tokenizer_ok = tokenizer_ok and suffix == manifest_row["candidate_token_ids"][candidate]
            lengths.add(len(suffix))
        equal_lengths_ok = equal_lengths_ok and len(lengths) == 1
        manifest_digest_ok = manifest_digest_ok and validate_embedded(manifest_row, "manifest_row_digest")
        if (index + 1) % 3840 == 0:
            print(f"[phase1221/preaudit-tokenizer] {index + 1}/{len(manifest)}", flush=True)
    add(checks, "tokenizer_exact_replay", tokenizer_ok)
    add(checks, "equal_lengths_exact_replay", equal_lengths_ok)
    add(checks, "manifest_row_digests", manifest_digest_ok)

    sentinel = dict(rows[0])
    sentinel["gold"] = next(value for value in sentinel["candidates"] if value != sentinel["gold"])
    add(checks, "truth_positive_sentinel", recompute_gold(sentinel) != sentinel["gold"])
    add(checks, "behavior_absent_before_preaudit", not RAW_PATH.exists() and not FINAL_PATH.exists())
    add(
        checks,
        "behavior_only_contract",
        not protocol["interface"]["hidden_states"]
        and not protocol["interface"]["attentions"]
        and not protocol["interface"]["hooks"]
        and not protocol["interface"]["interventions"],
    )

    result = {
        "phase": PHASE,
        "mode": "preaudit",
        "created_at": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "protocol_digest": protocol["protocol_digest"],
    }
    result["audit_digest"] = digest(result)
    return result


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def operation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_panel = {panel: rate([row for row in rows if row["panel"] == panel], "candidate_correct") for panel in PANELS}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    surface_success = [len(values) == len(PANELS) and all(row["candidate_correct"] for row in values) for values in groups.values()]
    errors = Counter(row["error_fingerprint"] for row in rows if not row["candidate_correct"])
    prediction_positions = Counter(row["prediction_position"] for row in rows if row["prediction_position"] is not None)
    correct_positions = Counter(row["gold_position"] for row in rows if row["candidate_correct"])
    return {
        "case_count": len(rows),
        "world_count": len({row["world_id"] for row in rows}),
        "finite_rate": rate(rows, "all_candidate_scores_finite"),
        "candidate_accuracy": rate(rows, "candidate_correct"),
        "generation_accuracy": rate(rows, "generation_correct"),
        "generation_exact_option_rate": rate(rows, "generation_normalized_exact"),
        "sum_mean_winner_agreement": rate(rows, "sum_mean_winner_agreement"),
        "candidate_by_panel": by_panel,
        "worst_panel_candidate": min(by_panel.values()),
        "surface_group_rate": sum(surface_success) / len(surface_success),
        "direct_prefill_fraction": sum(row["scoring_path"] == "direct_prefill_low_margin" for row in rows) / len(rows),
        "mean_sum_margin": sum(row["sum_margin"] for row in rows) / len(rows),
        "error_fingerprints": dict(sorted(errors.items())),
        "prediction_position_counts": {str(key): value for key, value in sorted(prediction_positions.items())},
        "correct_gold_position_counts": {str(key): value for key, value in sorted(correct_positions.items())},
    }


def metric_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "finite": metrics["finite_rate"] >= FINITE_MIN,
        "candidate": metrics["candidate_accuracy"] >= CANDIDATE_MIN,
        "generation": metrics["generation_accuracy"] >= GENERATION_MIN,
        "worst_panel": metrics["worst_panel_candidate"] >= WORST_PANEL_MIN,
        "surface_group": metrics["surface_group_rate"] >= SURFACE_GROUP_MIN,
        "sum_mean_agreement": metrics["sum_mean_winner_agreement"] >= SUM_MEAN_AGREEMENT_MIN,
    }


def independent_summary(raw: list[dict[str, Any]]) -> dict[str, Any]:
    operation_results: dict[str, Any] = {}
    operation_authorization: dict[str, bool] = {}
    family_authorization: dict[str, bool] = {}
    for family, operations in FAMILIES.items():
        for track in TRACKS:
            family_passes = []
            for operation in operations:
                split_results = {}
                split_passes = []
                for split in SPLITS:
                    selected = [
                        row for row in raw
                        if row["family"] == family and row["track"] == track
                        and row["operation"] == operation and row["split"] == split
                    ]
                    metrics = operation_metrics(selected)
                    gates = metric_gate(metrics)
                    passed = all(gates.values())
                    split_results[split] = {"metrics": metrics, "gates": gates, "passed": passed}
                    split_passes.append(passed)
                key = f"{family}|{track}|{operation}"
                authorized = all(split_passes)
                operation_results[key] = {"splits": split_results, "authorized": authorized}
                operation_authorization[key] = authorized
                family_passes.append(authorized)
            family_authorization[f"{family}|{track}"] = all(family_passes)
    authorized_scopes = sorted(key for key, value in family_authorization.items() if value)
    return {
        "operation_results": operation_results,
        "operation_authorization": operation_authorization,
        "family_authorization": family_authorization,
        "authorized_family_tracks": authorized_scopes,
        "any_family_track_authorized": bool(authorized_scopes),
        "unified_authorized": all(family_authorization.values()),
    }


def result_audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    raw = read_jsonl(RAW_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)
    final = read_json(FINAL_PATH)
    checks: dict[str, bool] = {}
    add(checks, "protocol_still_valid", validate_embedded(protocol, "protocol_digest"))
    add(checks, "main_hash_unchanged", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT))
    add(checks, "audit_hash_unchanged", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT))
    add(checks, "raw_count", len(raw) == EXPECTED_ROWS)
    add(checks, "unique_raw_items", len({row["item_id"] for row in raw}) == EXPECTED_ROWS)
    add(checks, "run_summary_embedded", validate_embedded(run_summary, "summary_digest"))
    add(checks, "raw_digest", run_summary["raw_digest"] == digest(raw))
    add(checks, "final_embedded", validate_embedded(final, "final_digest"))
    add(checks, "no_quantization", not run_summary["precision_audit"]["has_quantized_modules"])
    add(checks, "fp16_parameters", run_summary["precision_audit"]["has_fp16_parameters"])
    add(checks, "no_pending", not any(OUT_ROOT.rglob("*.pending")))

    material_by_id = {row["item_id"]: row for row in rows}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    digest_ok = True
    binding_ok = True
    score_ok = True
    generation_ok = True
    for row in raw:
        material = material_by_id[row["item_id"]]
        manifest_row = manifest_by_id[row["item_id"]]
        digest_ok = digest_ok and validate_embedded(row, "behavior_row_digest")
        binding_ok = binding_ok and row["row_digest"] == material["row_digest"]
        values = row["candidate_scores"]
        sum_order = sorted(values, key=lambda candidate: values[candidate]["sum_log_probability"], reverse=True)
        mean_order = sorted(values, key=lambda candidate: values[candidate]["mean_log_probability"], reverse=True)
        margin = values[sum_order[0]]["sum_log_probability"] - values[sum_order[1]]["sum_log_probability"]
        sum_prediction = None if abs(margin) <= TIE_TOLERANCE else sum_order[0]
        mean_prediction = mean_order[0]
        prediction_position = material["candidate_order"].index(sum_prediction) if sum_prediction in material["candidate_order"] else None
        score_ok = score_ok and set(values) == set(manifest_row["candidates"])
        score_ok = score_ok and sum_prediction == row["sum_prediction"]
        score_ok = score_ok and mean_prediction == row["mean_prediction"]
        score_ok = score_ok and abs(margin - row["sum_margin"]) < 1e-9
        score_ok = score_ok and bool(sum_prediction == material["gold"]) == row["candidate_correct"]
        score_ok = score_ok and (sum_prediction == mean_prediction) == row["sum_mean_winner_agreement"]
        score_ok = score_ok and prediction_position == row["prediction_position"]
        score_ok = score_ok and row["scoring_path"] in {"shared_prefix_cache", "direct_prefill_low_margin"}
        score_ok = score_ok and row["error_fingerprint"] == (
            material["fingerprint_by_candidate"].get(sum_prediction, "unregistered_candidate") if sum_prediction else "tie"
        )
        generation_ok = generation_ok and row["generation_correct"] == (row["generation_prediction"] == material["gold"])
    add(checks, "raw_embedded_digests", digest_ok)
    add(checks, "raw_material_binding", binding_ok)
    add(checks, "candidate_scores_recomputed", score_ok)
    add(checks, "generation_recomputed", generation_ok)

    independent = independent_summary(raw)
    authorized = independent["authorized_family_tracks"]
    add(checks, "summary_exact_recompute", independent == final["behavior"])
    add(checks, "status_recompute", final["status"] == ("typed_behavior_authorized" if authorized else "typed_behavior_no_family_authorized"))
    add(checks, "authorization_recompute", final["authorized_next"]["automatic_execution"] == bool(authorized))
    add(checks, "authorized_scopes_recompute", final["authorized_next"]["authorized_family_tracks"] == authorized)
    add(checks, "k198_scope", final["k_item"]["identifier"] == "K198" and final["k_item"]["scope"] == "Qwen3 FP16; generated worlds; behavior only")
    add(checks, "no_hidden_claim", not final["claim_boundary"]["hidden_state"] and not final["claim_boundary"]["causal"])

    result = {
        "phase": PHASE,
        "mode": "result",
        "created_at": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "recomputed_behavior_digest": digest(independent),
        "final_digest": final["final_digest"],
    }
    result["audit_digest"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preaudit", "result"))
    args = parser.parse_args()
    result = preaudit() if args.mode == "preaudit" else result_audit()
    path = PREAUDIT_PATH if args.mode == "preaudit" else RESULT_AUDIT_PATH
    if path.exists():
        raise RuntimeError(f"audit output already exists: {path}")
    write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
