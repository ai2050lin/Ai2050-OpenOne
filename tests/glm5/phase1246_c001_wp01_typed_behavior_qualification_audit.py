#!/usr/bin/env python3
"""Independent auditor for Phase1246 C001-WP01."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
OS_ROOT = ROOT / "research/ai2050_research_os"
PHASE = 1246
CONTRACT_ID = "EXP-C001-WP01-001"
RUN_ID = "RUN-EXP-C001-WP01-001-001"
MAIN = TEST_ROOT / "phase1246_c001_wp01_typed_behavior_qualification.py"
SELF = Path(__file__).resolve()
OUT_ROOT = TEST_ROOT / "result/phase1246_c001_wp01_typed_behavior_qualification"
LOCAL_CONTRACT_PATH = OUT_ROOT / "protocol/scientific_contract_snapshot.json"
MATERIAL_PATH = OUT_ROOT / "material/frozen_typed_worlds.jsonl"
TOKEN_MANIFEST_PATH = OUT_ROOT / "material/qwen3_token_manifest.jsonl"
FIXTURE_PATH = OUT_ROOT / "material/evaluator_fixtures.jsonl"
CAMERA_PATH = OUT_ROOT / "calibration/known_truth_response_camera.json"
PROGRAM_PATH = OUT_ROOT / "calibration/alternative_program_audit.json"
PLAN_PATH = OUT_ROOT / "protocol/frozen_execution_plan.json"
ENVIRONMENT_PATH = OUT_ROOT / "protocol/environment_snapshot.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
ADJUDICATION_PATH = OUT_ROOT / "analysis/typed_adjudication.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"
CONTRACT_PATH = OS_ROOT / f"contracts/{CONTRACT_ID}.json"
CONTRACT_INDEX_PATH = OS_ROOT / "registry/contracts.json"
RUNS_PATH = OS_ROOT / "registry/runs.json"

PARTITIONS = ("calibration", "discovery", "selection", "confirmation")
REQUIRED = ("discovery", "selection", "confirmation")
SEALED = {"selection", "confirmation"}
PROTOCOLS = ("candidate", "bare_short", "prompted_short", "fixed_sentence", "natural_sentence")
GENERATION_PROTOCOLS = PROTOCOLS[1:]
FORMAT_PROTOCOLS = ("bare_short", "prompted_short", "fixed_sentence")
COLLISIONS = (
    "target_change",
    "nontarget_noop",
    "query_switch",
    "same_bag_binding_swap",
    "order_invariance",
    "template_invariance",
)
THRESHOLDS = {
    "content_partition_accuracy": 0.95,
    "content_worst_template_name_accuracy": 0.90,
    "collision_complete_rate": 0.90,
    "alternative_program_advantage": 0.15,
    "format_partition_accuracy": 0.95,
    "format_worst_template_accuracy": 0.95,
    "natural_partition_accuracy": 0.95,
    "natural_worst_template_accuracy": 0.90,
    "cache_top1_agreement": 1.0,
    "correct_stop_rate": 0.95,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scientific_contract(contract: dict[str, Any]) -> dict[str, Any]:
    value = dict(contract)
    value.pop("status", None)
    value.pop("phase", None)
    value.pop("frozen_artifacts", None)
    return value


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def parse_independent(text: str, candidates: list[str], protocol: str, expected: str) -> dict[str, Any]:
    value = normalize(text)
    mentions = sorted(
        candidate for candidate in candidates
        if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", value)
    )
    prediction = mentions[0] if len(mentions) == 1 else None
    rejected = None
    if prediction is None:
        rejected = "none_or_multiple_candidates"
    else:
        match = re.search(rf"(?<!\w){re.escape(prediction)}(?!\w)", value)
        start, end = match.span() if match else (0, 0)
        before = value[max(0, start - 36):start]
        after = value[end:end + 36]
        if re.search(r"\b(?:not|wrong|incorrect|isn't|isnt)\b|rather than", before):
            prediction, rejected = None, "negated_candidate"
        elif re.search(r"\b(?:sorry|correction|instead)\b", after):
            prediction, rejected = None, "self_correction"
    escaped = "|".join(re.escape(candidate) for candidate in candidates)
    if protocol == "bare_short":
        format_valid = re.fullmatch(rf"(?:{escaped})", text.strip()) is not None
    elif protocol == "prompted_short":
        format_valid = re.fullmatch(rf"Marker name: (?:{escaped})\.", text.strip()) is not None
    elif protocol == "fixed_sentence":
        format_valid = re.fullmatch(rf"The canonical marker name is (?:{escaped})\.", text.strip()) is not None
    elif protocol == "natural_sentence":
        format_valid = prediction is not None and len(text.strip().split()) <= 20 and text.strip().endswith((".", "!", "?"))
    else:
        raise ValueError(protocol)
    gold = normalize(expected) if protocol == "bare_short" else normalize(expected).split()[-1].strip(string.punctuation)
    return {
        "prediction": prediction,
        "rejected_reason": rejected,
        "content_correct": prediction == gold,
        "format_valid": bool(format_valid),
        "exact": text.strip() == expected,
    }


def mean_bool(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else float("nan")


def check(name: str, condition: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "details": details}


def lexical_multiset(prompt: str) -> list[str]:
    return sorted(re.findall(r"[A-Za-z]+|\d+|[^\w\s]", prompt.lower()))


def pairwise_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = {
        "query_index": lambda row: row["query_index"],
        "gold_slot": lambda row: row["candidate_codes"].index(row["gold_code"]),
        "template": lambda row: row["template_index"],
        "name_world": lambda row: row["name_world"],
        "record_first": lambda row: row["record_order"][0],
    }
    results: dict[str, Any] = {}
    for partition in PARTITIONS:
        selected = [row for row in rows if row["partition"] == partition]
        for left_index, left in enumerate(features):
            for right in list(features)[left_index + 1:]:
                left_values = {features[left](row) for row in selected}
                right_values = {features[right](row) for row in selected}
                observed = {(features[left](row), features[right](row)) for row in selected}
                expected = {(a, b) for a in left_values for b in right_values}
                results[f"{partition}|{left}|{right}"] = {
                    "observed": len(observed),
                    "expected": len(expected),
                    "complete": observed == expected,
                }
    return results


def preaudit() -> None:
    required_paths = [
        MAIN, SELF, LOCAL_CONTRACT_PATH, MATERIAL_PATH, TOKEN_MANIFEST_PATH, FIXTURE_PATH,
        CAMERA_PATH, PROGRAM_PATH, PLAN_PATH, ENVIRONMENT_PATH, CONTRACT_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing preaudit inputs: {missing}")
    local = read_json(LOCAL_CONTRACT_PATH)
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    tokens = read_jsonl(TOKEN_MANIFEST_PATH)
    fixtures = read_jsonl(FIXTURE_PATH)
    camera = read_json(CAMERA_PATH)
    program = read_json(PROGRAM_PATH)
    plan = read_json(PLAN_PATH)
    environment = read_json(ENVIRONMENT_PATH)
    row_map = {row["row_id"]: row for row in rows}
    token_map = {row["row_id"]: row for row in tokens}
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        worlds[row["world_id"]].append(row)
    partition_worlds = {
        partition: {row["world_id"] for row in rows if row["partition"] == partition}
        for partition in PARTITIONS
    }
    partition_names = {
        partition: {name for row in rows if row["partition"] == partition for name in row["objects"]}
        for partition in PARTITIONS
    }
    answer_string_absent = all(
        not re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", row["prompts"][protocol].lower())
        for row in rows for protocol in PROTOCOLS for candidate in row["candidate_order"]
    )
    answer_ids_absent = all(
        not (set(token_map[row["row_id"]]["candidate_token_ids"].values()) &
             set(token_map[row["row_id"]]["input_ids"][protocol]))
        for row in rows for protocol in PROTOCOLS
    )
    row_digests = all(
        row["row_digest"] == digest({key: value for key, value in row.items() if key != "row_digest"}) for row in rows
    )
    token_digests = all(
        row["token_row_digest"] == digest({key: value for key, value in row.items() if key != "token_row_digest"})
        for row in tokens
    )
    expected_change = {"target_change", "query_switch", "same_bag_binding_swap"}
    collision_semantics = all(
        len(pair) == 2 and ((pair[0]["gold"] != pair[1]["gold"]) == (pair[0]["collision_group"] in expected_change))
        for pair in worlds.values()
    )
    same_bag = all(
        lexical_multiset(pair[0]["prompts"]["candidate"]) == lexical_multiset(pair[1]["prompts"]["candidate"])
        for pair in worlds.values() if pair[0]["collision_group"] == "same_bag_binding_swap"
    )
    pairwise = pairwise_coverage(rows)
    fixture_results = []
    for fixture in fixtures:
        parsed = parse_independent(fixture["text"], fixture["candidates"], fixture["protocol"], fixture["expected"])
        fixture_results.append(
            parsed["prediction"] == fixture["expected_prediction"]
            and parsed["content_correct"] == fixture["expected_content_correct"]
            and parsed["format_valid"] == fixture["expected_format_valid"]
        )
    checks = [
        check("scientific_contract_digest", digest(scientific_contract(contract)) == local["scientific_contract_digest"]),
        check("local_snapshot_digest", local["snapshot_digest"] == digest({k: v for k, v in local.items() if k != "snapshot_digest"})),
        check(
            "phase_contract_pre_model",
            contract["phase"] in (None, PHASE) and contract["status"] in {"preregistered", "ready"},
            {"phase": contract["phase"], "status": contract["status"]},
        ),
        check("row_count", len(rows) == 1024, len(rows)),
        check("token_row_count", len(tokens) == len(rows), len(tokens)),
        check("world_count", len(worlds) == 512, len(worlds)),
        check("two_states_per_world", all(len(pair) == 2 and {row["state"] for row in pair} == {0, 1} for pair in worlds.values())),
        check("partition_world_counts", all(len(partition_worlds[p]) == 128 for p in PARTITIONS), {p: len(v) for p, v in partition_worlds.items()}),
        check("partition_world_disjointness", all(not (partition_worlds[a] & partition_worlds[b]) for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:])),
        check("partition_name_disjointness", all(not (partition_names[a] & partition_names[b]) for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:])),
        check("row_digests", row_digests),
        check("token_row_digests", token_digests),
        check("material_digest", digest(rows) == local["material_digest"]),
        check("token_manifest_digest", digest(tokens) == local["token_manifest_digest"]),
        check("execution_order", plan["execution_order"] == [row["row_id"] for row in rows]),
        check("answer_strings_absent", answer_string_absent),
        check("answer_token_ids_absent", answer_ids_absent),
        check("native_single_candidate_tokens", all(len(set(row["candidate_token_ids"].values())) == 5 for row in tokens)),
        check("max_input_length", max(length for row in tokens for length in row["input_lengths"].values()) <= 256),
        check("collision_semantics", collision_semantics),
        check("same_bag_binding_swap", same_bag),
        check("all_collisions_per_partition", all({row["collision_group"] for row in rows if row["partition"] == p} == set(COLLISIONS) for p in PARTITIONS)),
        check("nonbijective_assignments", all(len(set(row["assignments"].values())) < 4 for row in rows)),
        check("unused_value", all(set(row["candidate_codes"]) - set(row["assignments"].values()) for row in rows)),
        check("pairwise_factor_coverage", all(cell["complete"] for cell in pairwise.values()), pairwise),
        check("alternative_program_gate", program["program_gate"] is True),
        check("known_truth_camera_gate", camera["camera_gate"] is True),
        check("known_truth_open_unknown", camera["checks"]["open_discovery_channel"] is True),
        check("known_truth_abstention", camera["checks"]["nonidentifiable_abstention"] is True),
        check("known_truth_basis_expansion", camera["checks"]["basis_expansion_recovers_twins"] is True),
        check("evaluator_fixtures", all(fixture_results), {"passed": sum(fixture_results), "total": len(fixture_results)}),
        check("cuda_environment", environment["cuda_available"] is True and environment["precision"] == "float16"),
        check("no_model_loaded_in_prepare", environment["model_weights_loaded"] is False),
        check("single_run_budget", plan["model_runs"] == 1 and plan["adaptive_rounds"] == 0),
        check("no_existing_raw_output", not RAW_PATH.exists() and not RUN_SUMMARY_PATH.exists()),
    ]
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1246.independent_preaudit.v1",
        "created_at_utc": utc_now(),
        "auditor": str(SELF),
        "auditor_sha256": file_sha256(SELF),
        "main_sha256": file_sha256(MAIN),
        "check_count": len(checks),
        "checks": checks,
        "pairwise_coverage": pairwise,
        "all_checks_passed": all(row["passed"] for row in checks),
        "authorization": "run_ready_candidate" if all(row["passed"] for row in checks) else "deny_model_run",
        "claim_boundary": "Preaudit validates construction, tokenizer and calibrated adjudication only; it contains no model result.",
    }
    value["preaudit_digest"] = digest(value)
    write_json(PREAUDIT_PATH, value)
    print(canonical_json({"status": "phase1246_preaudit", "passed": value["all_checks_passed"], "checks": len(checks), "digest": value["preaudit_digest"]}))
    if not value["all_checks_passed"]:
        raise SystemExit(2)


def content_correct(row: dict[str, Any], material: dict[str, Any], protocol: str) -> bool:
    if protocol == "candidate":
        return row["candidate"]["prediction"] == material["gold"]
    parsed = parse_independent(
        row["generations"][protocol]["text"], material["candidate_order"], protocol, material["expected_outputs"][protocol]
    )
    return bool(parsed["content_correct"])


def recompute_gates(raw: list[dict[str, Any]], material_rows: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, Any]]:
    material = {row["row_id"]: row for row in material_rows}
    program = read_json(PROGRAM_PATH)
    partition_accuracy = {}
    worst_template_name = {}
    strongest_alt = {}
    for partition in REQUIRED:
        rows = [row for row in raw if row["partition"] == partition]
        partition_accuracy[partition] = mean_bool(
            content_correct(row, material[row["row_id"]], protocol) for row in rows for protocol in PROTOCOLS
        )
        cells = []
        for template in range(4):
            for name_world in range(4):
                cell = [row for row in rows if row["template_index"] == template and row["name_world"] == name_world]
                if cell:
                    cells.append(mean_bool(
                        content_correct(row, material[row["row_id"]], protocol) for row in cell for protocol in PROTOCOLS
                    ))
        worst_template_name[partition] = min(cells)
        strongest_alt[partition] = program["partitions"][partition]["strongest_registered_ceiling"]
    collision_complete = {}
    for group in COLLISIONS:
        world_ids = sorted({row["world_id"] for row in raw if row["partition"] in REQUIRED and row["collision_group"] == group})
        collision_complete[group] = mean_bool(
            all(
                content_correct(row, material[row["row_id"]], protocol)
                for row in raw if row["world_id"] == world_id for protocol in PROTOCOLS
            )
            for world_id in world_ids
        )
    content_checks = {
        "partition_accuracy": min(partition_accuracy.values()) >= THRESHOLDS["content_partition_accuracy"],
        "worst_template_name": min(worst_template_name.values()) >= THRESHOLDS["content_worst_template_name_accuracy"],
        "collision_complete": min(collision_complete.values()) >= THRESHOLDS["collision_complete_rate"],
        "alternative_advantage": min(partition_accuracy[p] - strongest_alt[p] for p in REQUIRED) >= THRESHOLDS["alternative_program_advantage"],
        "finite_and_unique_candidate": all(row["candidate"]["all_vocab_logits_finite"] and not row["candidate"]["tie"] for row in raw),
    }
    format_partition = {}
    format_worst = {}
    for partition in REQUIRED:
        rows = [row for row in raw if row["partition"] == partition]
        format_partition[partition] = mean_bool(
            parse_independent(row["generations"][protocol]["text"], material[row["row_id"]]["candidate_order"], protocol, material[row["row_id"]]["expected_outputs"][protocol])["exact"]
            for row in rows for protocol in FORMAT_PROTOCOLS
        )
        format_worst[partition] = min(
            mean_bool(
                parse_independent(row["generations"][protocol]["text"], material[row["row_id"]]["candidate_order"], protocol, material[row["row_id"]]["expected_outputs"][protocol])["exact"]
                for row in rows if row["template_index"] == template for protocol in FORMAT_PROTOCOLS
            )
            for template in range(4)
        )
    fixture_results = []
    for fixture in read_jsonl(FIXTURE_PATH):
        parsed = parse_independent(fixture["text"], fixture["candidates"], fixture["protocol"], fixture["expected"])
        fixture_results.append(parsed["prediction"] == fixture["expected_prediction"] and parsed["content_correct"] == fixture["expected_content_correct"] and parsed["format_valid"] == fixture["expected_format_valid"])
    format_checks = {
        "partition_accuracy": min(format_partition.values()) >= THRESHOLDS["format_partition_accuracy"],
        "worst_template": min(format_worst.values()) >= THRESHOLDS["format_worst_template_accuracy"],
        "four_cell_and_adversarial_evaluator": all(fixture_results),
    }
    natural_partition = {}
    natural_worst = {}
    for partition in ("selection", "confirmation"):
        rows = [row for row in raw if row["partition"] == partition]
        natural_partition[partition] = mean_bool(
            content_correct(row, material[row["row_id"]], "natural_sentence") for row in rows
        )
        natural_worst[partition] = min(
            mean_bool(
                content_correct(row, material[row["row_id"]], "natural_sentence")
                for row in rows if row["template_index"] == template
            )
            for template in range(4)
        )
    natural_checks = {
        "partition_accuracy": min(natural_partition.values()) >= THRESHOLDS["natural_partition_accuracy"],
        "worst_template": min(natural_worst.values()) >= THRESHOLDS["natural_worst_template_accuracy"],
        "multi_reference_evaluator": all(fixture_results),
    }
    cache_rows = [row for row in raw if row["partition"] in SEALED and row["state"] == 0]
    cache_agreement = sum(row["cache_full_recompute"]["match_count"] for row in cache_rows) / sum(row["cache_full_recompute"]["step_count"] for row in cache_rows)
    stop_rows = [row for row in raw if row["partition"] in SEALED]
    stop_rate = mean_bool(row["generations"]["natural_sentence"]["model_stopped"] for row in stop_rows)
    cache_checks = {
        "cache_top1_agreement": cache_agreement == 1.0,
        "correct_stop_rate": stop_rate >= THRESHOLDS["correct_stop_rate"],
        "no_external_truncation_counted": all(row["generations"]["natural_sentence"]["stop_source"] == "model_eos" for row in stop_rows if row["generations"]["natural_sentence"]["model_stopped"]),
        "generation_logits_finite": all(row["generations"][p]["score_logits_finite"] for row in raw for p in GENERATION_PROTOCOLS),
    }
    gates = {
        "G-CONTENT": all(content_checks.values()),
        "G-FORMAT": all(format_checks.values()),
        "G-NATURAL": all(natural_checks.values()),
        "G-STOP-CACHE": all(cache_checks.values()),
    }
    details = {
        "content": {"partition_accuracy": partition_accuracy, "worst_template_name": worst_template_name, "collision_complete": collision_complete, "strongest_alt": strongest_alt, "checks": content_checks},
        "format": {"partition_accuracy": format_partition, "worst_template": format_worst, "checks": format_checks},
        "natural": {"partition_accuracy": natural_partition, "worst_template": natural_worst, "checks": natural_checks},
        "stop_cache": {"cache_agreement": cache_agreement, "stop_rate": stop_rate, "checks": cache_checks},
    }
    return gates, details


def final_audit() -> None:
    required_paths = [RAW_PATH, RUN_SUMMARY_PATH, ADJUDICATION_PATH, FINAL_PATH, PREAUDIT_PATH]
    if any(not path.is_file() for path in required_paths):
        raise RuntimeError("final audit inputs missing")
    local = read_json(LOCAL_CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    adjudication = read_json(ADJUDICATION_PATH)
    final = read_json(FINAL_PATH)
    gates, details = recompute_gates(raw, material)
    row_digests = all(row["behavior_row_digest"] == digest({k: v for k, v in row.items() if k != "behavior_row_digest"}) for row in raw)
    expected_verdict = "typed_behavior_qualified" if all(gates.values()) else ("partial_typed_qualification" if any(gates.values()) else "bounded_rejected")
    checks = [
        check("preaudit_passed", read_json(PREAUDIT_PATH)["all_checks_passed"] is True),
        check("raw_row_count", len(raw) == 1024, len(raw)),
        check("unique_raw_rows", len({row["row_id"] for row in raw}) == len(raw)),
        check("raw_row_digests", row_digests),
        check("raw_digest", summary["raw_digest"] == digest(raw)),
        check("summary_digest", summary["summary_digest"] == digest({k: v for k, v in summary.items() if k != "summary_digest"})),
        check("scientific_contract_digest", summary["scientific_contract_digest"] == local["scientific_contract_digest"]),
        check("fp16_cuda", summary["precision_audit"]["has_fp16_parameters"] and not summary["precision_audit"]["has_quantized_modules"]),
        check("gpu_budget", summary["gpu_budget_respected"] is True and summary["gpu_hours"] <= 2.0),
        check("no_hidden_state", summary["hidden_states_saved"] is False),
        check("no_attention", summary["attentions_saved"] is False),
        check("no_intervention", summary["interventions_performed"] is False),
        check("typed_gates_recomputed", adjudication["typed_gates"] == gates, {"expected": gates, "observed": adjudication["typed_gates"]}),
        check("verdict_recomputed", adjudication["verdict"] == expected_verdict),
        check("adjudication_digest", adjudication["adjudication_digest"] == digest({k: v for k, v in adjudication.items() if k != "adjudication_digest"})),
        check("final_gates", final["typed_gates"] == gates),
        check("final_verdict", final["verdict"] == expected_verdict),
        check("final_digest", final["final_digest"] == digest({k: v for k, v in final.items() if k != "final_digest"})),
        check("known_truth_not_promoted", final["known_truth_camera_gate"] is True and any("instrument calibration" in item for item in final["non_claims"])),
        check("auto_continue_false", final["auto_continue"] is False),
        check("contract_run_count", len([row for row in read_json(RUNS_PATH) if row["contract_id"] == CONTRACT_ID]) == 1),
    ]
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1246.independent_final_audit.v1",
        "created_at_utc": utc_now(),
        "auditor_sha256": file_sha256(SELF),
        "main_sha256": file_sha256(MAIN),
        "recomputed_gates": gates,
        "recomputed_details": details,
        "recomputed_verdict": expected_verdict,
        "check_count": len(checks),
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
        "claim_boundary": "Audit confirms typed behavior arithmetic and execution integrity only; no mechanism evidence exists in this phase.",
    }
    value["audit_digest"] = digest(value)
    write_json(FINAL_AUDIT_PATH, value)
    print(canonical_json({"status": "phase1246_final_audit", "passed": value["all_checks_passed"], "checks": len(checks), "verdict": expected_verdict, "digest": value["audit_digest"]}))
    if not value["all_checks_passed"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preaudit", "final"), required=True)
    args = parser.parse_args()
    if args.mode == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
