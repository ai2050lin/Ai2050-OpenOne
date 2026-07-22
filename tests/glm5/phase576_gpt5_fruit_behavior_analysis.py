#!/usr/bin/env python3
"""CPU-only independent audit of one Phase576 behavior stage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests/glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase576_gpt5_fruit_protocol as protocol  # noqa: E402
import phase576_gpt5_fruit_behavior as behavior_execution  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite/stomp Phase576 analysis: {path}")
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def stage_dir(stage: str) -> Path:
    return protocol.OUT_DIR / "open_behavior" / stage


def rows_path(model: str, stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_rows.jsonl.gz"


def summary_path(model: str, stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_summary.json"


def contract_path(model: str, stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_contract.json"


def status_path(model: str, stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{model}_{stage}_behavior_status.json"


def receipt_path(stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{stage}_behavior_execution_receipt.json"


def stage_started_path(stage: str) -> Path:
    return stage_dir(stage) / f"phase576_{stage}_behavior_stage_started.json"


def normalized_short_answer(text: str) -> str:
    compact = " ".join(text.strip().split()).casefold()
    return compact.strip(" .,!?:;\"'`()[]{}")


def independent_classification(row: dict[str, Any], text: str) -> dict[str, Any]:
    normalized = normalized_short_answer(text)
    exact_owners = {
        canonical
        for canonical, aliases in row["candidate_groups"].items()
        if normalized in {alias.casefold() for alias in aliases}
    }
    selected = next(iter(exact_owners)) if len(exact_owners) == 1 else None
    mentions = sorted({
        canonical
        for canonical, aliases in row["candidate_groups"].items()
        if any(re.search(
            rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE
        ) for alias in aliases)
    })
    if len(exact_owners) > 1 or len(mentions) > 1:
        event = "ambiguous_multiple_candidates"
    elif selected == row["target"]:
        event = "target_exact_short_answer"
    elif selected is not None:
        event = "registered_other_exact_short_answer"
    elif mentions:
        event = "candidate_mentioned_but_not_short_answer"
    else:
        event = "unrecoverable"
    correct = selected == row["target"] and len(exact_owners) == 1
    return {
        "normalized_generated": normalized,
        "selected_candidate": selected,
        "mentioned_candidates": mentions,
        "semantic_correct": correct,
        "strict_sequence_correct": correct,
        "semantic_event": event,
    }


def artifact_paths(model: str, stage: str) -> tuple[Path, Path, Path, Path]:
    return (
        rows_path(model, stage), summary_path(model, stage),
        contract_path(model, stage), status_path(model, stage),
    )


def blocked_report(model: str, stage: str, status: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "stage": stage,
        "behavior_gate_pass": False,
        "single_model_trace_authorized": False,
        "internal_trace_authorized": False,
        "blocked_reason": "behavior_execution_failed",
        "execution_status": status,
        "trace_case_ids": [],
        "internal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "sealed_model_access": False,
    }


def validate_status(
    model: str,
    stage: str,
    status: dict[str, Any],
    attempt: dict[str, Any],
    engineering: dict[str, Any],
) -> str:
    state = status.get("status")
    behavior_contract_file = contract_path(model, stage)
    expected_contract_hash = (
        sha256_file(behavior_contract_file)
        if behavior_contract_file.is_file() else None
    )
    base_valid = all((
        status.get("schema_version") == "phase576_behavior_model_status.v1",
        status.get("phase_id") == protocol.PHASE,
        status.get("model") == model,
        status.get("stage") == stage,
        status.get("model_order_index") == protocol.MODELS.index(model),
        isinstance(status.get("attempt_id"), str),
        bool(status.get("attempt_id")),
        attempt.get("attempt_id") == status.get("attempt_id"),
        status.get("sealed_model_access") is False,
        attempt.get("model") == model,
        attempt.get("status") == state,
        attempt.get("terminal_status_sha256") == sha256_file(
            status_path(model, stage)
        ),
        status.get("protocol_sha256") == sha256_file(protocol.PROTOCOL_PATH),
        status.get("stage_cases_sha256") == sha256_file(
            protocol.OPEN_SPLIT_CASE_PATHS[stage]
        ),
        status.get("behavior_source_sha256") == sha256_file(
            ROOT / "tests/glm5/phase576_gpt5_fruit_behavior.py"
        ),
        status.get("behavior_contract_sha256") == expected_contract_hash,
        status.get("engineering_qualification_sha256")
        == engineering["qualification_sha256"],
        status.get("engineering_execution_receipt_sha256")
        == engineering["execution_receipt_sha256"],
        status.get("runtime_identity") == engineering["runtime_identity"],
        status.get("cleanup_completed") is True,
        status.get("pytorch_cuda_allocated_after_release") == 0,
        isinstance(status.get("pytorch_cuda_reserved_after_release"), int),
        not isinstance(status.get("pytorch_cuda_reserved_after_release"), bool),
        status.get("pytorch_cuda_reserved_after_release", -1) >= 0,
    ))
    if not base_valid or state not in {"complete", "failed"}:
        raise RuntimeError(f"invalid behavior status identity for {stage}/{model}")
    if state == "complete":
        allocated = status.get("pytorch_cuda_allocated_after_release")
        reserved = status.get("pytorch_cuda_reserved_after_release")
        if not all((
            isinstance(status.get("completed_at_utc"), str),
            bool(status.get("completed_at_utc")),
            isinstance(allocated, int),
            not isinstance(allocated, bool),
            allocated == 0,
            isinstance(reserved, int),
            not isinstance(reserved, bool),
            reserved >= 0,
            behavior_contract_file.is_file(),
            status.get("behavior_rows_sha256") == sha256_file(
                rows_path(model, stage)
            ),
            status.get("behavior_summary_sha256") == sha256_file(
                summary_path(model, stage)
            ),
        )):
            raise RuntimeError(f"invalid complete status for {stage}/{model}")
    else:
        failure_stage = status.get("failure_stage")
        if not all((
            isinstance(status.get("failed_at_utc"), str),
            bool(status.get("failed_at_utc")),
            isinstance(status.get("error_type"), str),
            bool(status.get("error_type")),
            isinstance(status.get("error"), str),
            attempt.get("error_type") == status.get("error_type"),
             failure_stage in {
                 "precontract", "preexecution_status_publish",
                 "model_execution_or_cleanup", "missing_summary",
                 "terminal_status_publish",
             },
            (failure_stage == "precontract")
            is (not behavior_contract_file.is_file()),
            (failure_stage == "preexecution_status_publish")
            is behavior_contract_file.is_file()
            if failure_stage in {"precontract", "preexecution_status_publish"}
            else True,
        )):
            raise RuntimeError(f"invalid failed status for {stage}/{model}")
    return state


def validate_rows(
    model: str,
    stage: str,
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    tokenizer: Any,
    eos_ids: set[int],
) -> tuple[dict[str, bool], dict[str, Any], dict[str, int]]:
    case_by_id = {row["case_id"]: row for row in cases}
    expected_keys = {
        (case_id, repeat)
        for case_id in case_by_id
        for repeat in protocol.BEHAVIOR_REPEATS
    }
    actual_keys = [(row.get("case_id"), row.get("execution_repeat")) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise RuntimeError(f"{stage}/{model}: case x repeat registry is not exact")

    tokenizer_size = len(tokenizer)
    raw_special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    if not raw_special_ids or not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < tokenizer_size
        for value in raw_special_ids
    ):
        raise RuntimeError(f"{stage}/{model}: tokenizer special-token registry invalid")
    special_ids = set(int(value) for value in raw_special_ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool):
        raise RuntimeError(f"{stage}/{model}: tokenizer pad token is invalid")
    if not eos_ids or not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < tokenizer_size
        for value in eos_ids
    ):
        raise RuntimeError(f"{stage}/{model}: effective EOS registry invalid")

    stable_parts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    event_counts: dict[str, int] = defaultdict(int)
    capsule_counts: dict[str, int] = defaultdict(int)
    for observed in rows:
        case = case_by_id[observed["case_id"]]
        immutable = {
            "schema_version": "phase576_open_behavior_row.v2",
            "phase_id": protocol.PHASE,
            "model": model,
            "stage": stage,
            "split": stage,
            "relation": case["relation"],
            "interface": case["interface"],
            "surface_id": case["surface_id"],
            "order": case["order"],
            "focus_object_id": case["focus_object_id"],
            "focus_is_fruit": case["focus_is_fruit"],
            "contrast_group_id": case["contrast_group_id"],
            "contrast_label": case["contrast_label"],
            "independent_unit_id": case["independent_unit_id"],
            "target": case["target"],
            "observer_only": True,
            "activation_collected": False,
            "causal": False,
            "sealed_model_access": False,
        }
        for key, expected in immutable.items():
            if observed.get(key) != expected:
                raise RuntimeError(f"{stage}/{model}/{observed['case_id']}: drift in {key}")
        token_ids = observed.get("generated_token_ids_before_eos")
        if not isinstance(token_ids, list) or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < tokenizer_size
            for value in token_ids
        ):
            raise RuntimeError(f"{stage}/{model}: invalid generated token registry")
        if any(value in eos_ids for value in token_ids):
            raise RuntimeError(f"{stage}/{model}: pre-EOS content contains an EOS token")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        if not isinstance(observed.get("generated_text"), str) or decoded != observed[
            "generated_text"
        ]:
            raise RuntimeError(f"{stage}/{model}: generated text/token mismatch")
        token_count = observed.get("generated_token_count_before_eos")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count != len(token_ids)
        ):
            raise RuntimeError(f"{stage}/{model}: generated token count mismatch")
        if not isinstance(observed.get("eos_seen"), bool) or not isinstance(
            observed.get("budget_truncated"), bool
        ):
            raise RuntimeError(f"{stage}/{model}: non-boolean termination registry")
        eos_seen = observed["eos_seen"]
        first_eos = observed.get("first_eos_token_id")
        first_eos_index = observed.get("first_eos_index")
        full_suffix = observed.get("full_generated_suffix_token_ids")
        post_eos = observed.get("post_eos_token_ids")
        suffix_width = observed.get("generation_suffix_width")
        budget = observed["budget_truncated"]
        termination = observed.get("termination_event")
        if (
            not isinstance(full_suffix, list)
            or not full_suffix
            or len(full_suffix) > protocol.MAX_NEW_TOKENS
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < tokenizer_size
                for value in full_suffix
            )
            or suffix_width != len(full_suffix)
            or not isinstance(post_eos, list)
            or observed.get("post_eos_tokens_all_pad")
            is not all(value == pad_token_id for value in post_eos)
        ):
            raise RuntimeError(f"{stage}/{model}: full suffix registry invalid")
        if eos_seen:
            if (
                not isinstance(first_eos, int)
                or isinstance(first_eos, bool)
                or first_eos not in eos_ids
                or len(token_ids) + 1 > protocol.MAX_NEW_TOKENS
                or budget
                or termination != "eos"
                or first_eos_index != len(token_ids)
                or full_suffix[:first_eos_index] != token_ids
                or full_suffix[first_eos_index] != first_eos
                or full_suffix[first_eos_index + 1:] != post_eos
                or any(value != pad_token_id for value in post_eos)
            ):
                raise RuntimeError(f"{stage}/{model}: invalid EOS termination registry")
        else:
            if (
                first_eos is not None
                or len(token_ids) != protocol.MAX_NEW_TOKENS
                or not budget
                or termination != "budget"
                or first_eos_index is not None
                or full_suffix != token_ids
                or post_eos != []
            ):
                raise RuntimeError(f"{stage}/{model}: invalid non-EOS termination registry")
        content_special_ids = sorted(set(token_ids) & special_ids)
        capsule_counts["row_count"] += 1
        capsule_counts["eos_terminated"] += int(eos_seen)
        capsule_counts["budget_terminated"] += int(budget)
        capsule_counts["rows_with_special_token_before_eos"] += int(
            bool(content_special_ids)
        )
        rebuilt = independent_classification(case, observed["generated_text"])
        for key, expected in rebuilt.items():
            if observed.get(key) != expected:
                raise RuntimeError(
                    f"{stage}/{model}/{observed['case_id']}: independent classify mismatch {key}"
                )
        valid = bool(
            rebuilt["strict_sequence_correct"]
            and eos_seen
            and not budget
            and not content_special_ids
        )
        stable_parts[observed["case_id"]][observed["execution_repeat"]] = {
            "valid": valid,
            "normalized": rebuilt["normalized_generated"],
            "token_ids": token_ids,
            "first_eos_token_id": first_eos,
            "full_suffix": full_suffix,
        }
        event_counts[rebuilt["semantic_event"]] += 1

    stable = {}
    for case_id, repeats in stable_parts.items():
        first = repeats[protocol.BEHAVIOR_REPEATS[0]]
        second = repeats[protocol.BEHAVIOR_REPEATS[1]]
        stable[case_id] = bool(
            first["valid"] and second["valid"]
            and first["normalized"] == second["normalized"]
            and first["token_ids"] == second["token_ids"]
            and first["first_eos_token_id"] == second["first_eos_token_id"]
            and first["full_suffix"] == second["full_suffix"]
        )
    return (
        stable,
        dict(sorted(event_counts.items())),
        dict(sorted(capsule_counts.items())),
    )


def recompute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["relation"], row["interface"])].append(row)
    strata: dict[str, Any] = {}
    for key, bank in sorted(grouped.items()):
        strata["|".join(key)] = {
            "n": len(bank),
            "strict_correct": sum(item["strict_sequence_correct"] for item in bank),
            "strict_rate": sum(item["strict_sequence_correct"] for item in bank)
            / len(bank),
            "budget_truncated": sum(item["budget_truncated"] for item in bank),
        }
    return {
        "row_count": len(rows),
        "unique_case_count": len({row["case_id"] for row in rows}),
        "strict_correct": sum(row["strict_sequence_correct"] for row in rows),
        "strict_rate": sum(row["strict_sequence_correct"] for row in rows)
        / len(rows),
        "event_counts": dict(sorted(Counter(
            row["semantic_event"] for row in rows
        ).items())),
        "termination_counts": dict(sorted(Counter(
            row["termination_event"] for row in rows
        ).items())),
        "strata": strata,
    }


def unit_reports(
    cases: list[dict[str, Any]],
    stable: dict[str, bool],
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_unit[case["independent_unit_id"]].append(case)
    reports: dict[str, Any] = {}
    counts = {
        "category_direct": 0,
        "color_direct": 0,
        "category_selection": 0,
        "color_selection": 0,
    }
    for unit_id, bank in sorted(by_unit.items()):
        relation = bank[0]["relation"]
        interface = bank[0]["interface"]
        if relation not in protocol.RELATIONS or interface not in protocol.INTERFACES:
            raise RuntimeError(f"invalid unit family: {unit_id}")
        if any(
            row["relation"] != relation or row["interface"] != interface
            for row in bank
        ):
            raise RuntimeError(f"mixed relation/interface in unit: {unit_id}")
        stable_count = sum(stable[row["case_id"]] for row in bank)
        if interface == "direct":
            if (
                len(bank) != len(protocol.DIRECT_SURFACES)
                or {row["surface_id"] for row in bank}
                != set(protocol.DIRECT_SURFACES)
                or any(row["order"] is not None for row in bank)
                or len({row["focus_object_id"] for row in bank}) != 1
                or len({row["target"] for row in bank}) != 1
            ):
                raise RuntimeError(f"direct unit is not six surfaces: {unit_id}")
            passed = stable_count >= gate[
                "direct_unit_minimum_stable_surfaces_of_6"
            ]
            side_counts = None
        else:
            expected_grid = {
                (surface, order)
                for surface in protocol.SELECTION_SURFACES
                for order in protocol.SELECTION_ORDERS
            }
            if len(bank) != 2 * len(expected_grid):
                raise RuntimeError(f"selection unit is not 16 paired cases: {unit_id}")
            sides: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in bank:
                sides[row["contrast_label"]].append(row)
            if (
                len(sides) != 2
                or any(len(rows) != len(expected_grid) for rows in sides.values())
                or any(
                    {(row["surface_id"], row["order"]) for row in rows}
                    != expected_grid
                    for rows in sides.values()
                )
                or any(
                    len({row["focus_object_id"] for row in rows}) != 1
                    for rows in sides.values()
                )
                or len({row["focus_object_id"] for row in bank}) != 2
            ):
                raise RuntimeError(f"selection query-side registry invalid: {unit_id}")
            contrast_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in bank:
                contrast_groups[row["contrast_group_id"]].append(row)
            side_labels = set(sides)
            if (
                len(contrast_groups) != len(expected_grid)
                or any(len(rows) != 2 for rows in contrast_groups.values())
                or any(
                    {row["contrast_label"] for row in rows} != side_labels
                    or len({(row["surface_id"], row["order"]) for row in rows}) != 1
                    for rows in contrast_groups.values()
                )
            ):
                raise RuntimeError(f"selection contrast pairing invalid: {unit_id}")
            side_counts = {
                label: sum(stable[row["case_id"]] for row in rows)
                for label, rows in sorted(sides.items())
            }
            passed = (
                stable_count
                >= gate["selection_unit_minimum_stable_cases_of_16"]
                and all(
                    value
                    >= gate[
                        "selection_unit_minimum_stable_cases_each_query_side_of_8"
                    ]
                    for value in side_counts.values()
                )
            )
        key = f"{relation}_{interface}"
        counts[key] += int(passed)
        reports[unit_id] = {
            "relation": relation,
            "interface": interface,
            "case_count": len(bank),
            "stable_case_count": stable_count,
            "query_side_stable_counts": side_counts,
            "unit_pass": passed,
        }
    expected_families = {
        "category_direct": 12,
        "color_direct": 12,
        "category_selection": 6,
        "color_selection": 6,
    }
    observed_families = Counter(
        f"{report['relation']}_{report['interface']}"
        for report in reports.values()
    )
    if len(reports) != 36 or dict(observed_families) != expected_families:
        raise RuntimeError(
            "independent unit family denominator drift: "
            f"total={len(reports)}, families={dict(observed_families)}"
        )
    return reports, counts


def validate_gate_contract(gate: dict[str, Any]) -> None:
    integer_limits = {
        "minimum_stable_category_direct_units_of_12": 12,
        "minimum_stable_color_direct_units_of_12": 12,
        "minimum_stable_category_selection_units_of_6": 6,
        "minimum_stable_color_selection_units_of_6": 6,
        "direct_unit_minimum_stable_surfaces_of_6": 6,
        "selection_unit_minimum_stable_cases_of_16": 16,
        "selection_unit_minimum_stable_cases_each_query_side_of_8": 8,
    }
    checks: dict[str, bool] = {
        name: isinstance(gate.get(name), int)
        and not isinstance(gate.get(name), bool)
        and 0 <= gate[name] <= maximum
        for name, maximum in integer_limits.items()
    }
    rate = gate.get("minimum_case_level_stable_rate_diagnostic")
    checks["minimum_case_level_stable_rate_diagnostic"] = (
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and 0 <= rate <= 1
    )
    for name in (
        "surface_and_order_are_paired_repeats_not_independent_samples",
        "deterministic_generation_required",
        "valid_case_requires_exact_short_answer",
        "valid_case_requires_eos_before_budget",
        "model_failure_is_behavior_blocked_not_mechanism_absence",
    ):
        checks[name] = gate.get(name) is True
    if not all(checks.values()):
        raise RuntimeError(f"frozen behavior gate contract invalid: {checks}")


def analyze_model(
    model: str,
    stage: str,
    cases: list[dict[str, Any]],
    frozen: dict[str, Any],
    attempt: dict[str, Any],
    engineering: dict[str, Any],
) -> dict[str, Any]:
    status_file = status_path(model, stage)
    if not status_file.exists():
        raise RuntimeError(f"missing behavior status for {stage}/{model}")
    status = read_json(status_file)
    state = validate_status(model, stage, status, attempt, engineering)
    if state == "failed":
        return blocked_report(model, stage, status)
    rows_file, summary_file, contract_file, _ = artifact_paths(model, stage)
    for path in (rows_file, summary_file, contract_file):
        if not path.exists():
            raise RuntimeError(f"missing behavior artifact: {path}")
    summary = read_json(summary_file)
    contract = read_json(contract_file)
    rows = read_jsonl_gz(rows_file)
    source_key = "tests/glm5/phase576_gpt5_fruit_behavior.py"
    loaded = summary.get("loaded_model_identity", {})
    quant = loaded.get("loaded_quantization", {})
    stage_hash = sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    protocol_hash = sha256_file(protocol.PROTOCOL_PATH)
    expected_render_policy = {
        "qwen3_enable_thinking": False,
        "deepseek_empty_think_prefill_closed": True,
        "classification": (
            "exact registered short answer after terminal punctuation trim"
        ),
    }
    artifact_checks = {
        "summary_schema": summary.get("schema_version")
        == "phase576_open_behavior_summary.v2",
        "summary_identity": all((
            summary.get("phase_id") == protocol.PHASE,
            summary.get("model") == model,
            summary.get("stage") == stage,
            summary.get("attempt_id") == status.get("attempt_id"),
            isinstance(summary.get("created_at_utc"), str),
            bool(summary.get("created_at_utc")),
            isinstance(summary.get("elapsed_seconds"), (int, float)),
            not isinstance(summary.get("elapsed_seconds"), bool),
            summary.get("elapsed_seconds", -1) >= 0,
        )),
        "contract_schema": contract.get("schema_version")
        == "phase576_open_behavior_contract.v2",
        "contract_identity": all((
            contract.get("phase_id") == protocol.PHASE,
            contract.get("model") == model,
            contract.get("stage") == stage,
            contract.get("attempt_id") == status.get("attempt_id"),
            contract.get("model_order_index") == protocol.MODELS.index(model),
            isinstance(contract.get("created_at_utc"), str),
            bool(contract.get("created_at_utc")),
        )),
        "rows_hash": summary.get("rows_sha256") == sha256_file(rows_file),
        "stage_hash": summary.get("stage_cases_sha256")
        == stage_hash,
        "protocol_hash": summary.get("protocol_sha256")
        == protocol_hash,
        "contract_stage_hash": contract.get("stage_cases_sha256") == stage_hash,
        "contract_protocol_hash": contract.get("protocol_sha256") == protocol_hash,
        "summary_contract_hash": summary.get("behavior_contract_sha256")
        == sha256_file(contract_file),
        "engineering_chain": all((
            summary.get("engineering_qualification_sha256")
            == engineering["qualification_sha256"],
            summary.get("engineering_execution_receipt_sha256")
            == engineering["execution_receipt_sha256"],
            summary.get("runtime_identity") == engineering["runtime_identity"],
            contract.get("engineering_qualification_sha256")
            == engineering["qualification_sha256"],
            contract.get("engineering_execution_receipt_sha256")
            == engineering["execution_receipt_sha256"],
            contract.get("runtime_identity") == engineering["runtime_identity"],
        )),
        "frozen_model_identity": summary.get("frozen_model_artifact_identity")
        == frozen["model_artifact_identities"][model],
        "contract_model_identity": contract.get("model_artifact_identity")
        == frozen["model_artifact_identities"][model],
        "contract_source_registry": contract.get("frozen_stage_source_seals")
        == frozen["stage_source_seals"],
        "behavior_source": contract.get("behavior_source_sha256")
        == frozen["stage_source_seals"][source_key]["sha256"],
        "cuda_only": loaded.get("cuda_only_no_cpu_or_disk_offload") is True,
        "int8": quant.get("load_in_8bit") is True,
        "bf16_nonquantized": quant.get("floating_parameter_dtypes") == ["torch.bfloat16"],
        "sdpa": loaded.get("loaded_attn_implementation") == "sdpa",
        "loaded_model_key": loaded.get("model_key") == model,
        "loaded_on_cuda": isinstance(loaded.get("input_device"), str)
        and loaded["input_device"].startswith("cuda"),
        "contract_generation": all((
            contract.get("batch_size") == protocol.BEHAVIOR_BATCH_SIZE,
            contract.get("max_new_tokens") == protocol.MAX_NEW_TOKENS,
            contract.get("repeats") == list(protocol.BEHAVIOR_REPEATS),
            contract.get("do_sample") is False,
            contract.get("render_policy") == expected_render_policy,
        )),
        "contract_observer_only": all((
            contract.get("sealed_model_access") is False,
            contract.get("activation_collection") is False,
            contract.get("causal_intervention") is False,
        )),
        "summary_no_sealed": summary.get("sealed_model_access") is False,
    }
    if not all(artifact_checks.values()):
        raise RuntimeError(f"{stage}/{model}: behavior artifact check failed {artifact_checks}")
    raw_eos_ids = loaded.get("eos_identity", {}).get("effective_eos_token_ids")
    if (
        not isinstance(raw_eos_ids, list)
        or not raw_eos_ids
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_eos_ids
        )
        or raw_eos_ids != sorted(set(raw_eos_ids))
    ):
        raise RuntimeError(f"{stage}/{model}: missing effective EOS registry")
    eos_ids = set(raw_eos_ids)
    tokenizer = protocol.tokenizer_for(model)
    stable, event_counts, capsule_counts = validate_rows(
        model, stage, cases, rows, tokenizer, eos_ids
    )
    gate = frozen["behavior_gate"]
    validate_gate_contract(gate)
    units, counts = unit_reports(cases, stable, gate)
    expected_summary = recompute_summary(rows)
    summary_checks = {
        key: summary.get(key) == expected
        for key, expected in expected_summary.items()
    }
    if not all(summary_checks.values()):
        raise RuntimeError(
            f"{stage}/{model}: behavior summary mismatch {summary_checks}"
        )
    case_rate = sum(stable.values()) / len(stable)
    gate_checks = {
        "category_direct_units": counts["category_direct"]
        >= gate["minimum_stable_category_direct_units_of_12"],
        "color_direct_units": counts["color_direct"]
        >= gate["minimum_stable_color_direct_units_of_12"],
        "category_selection_units": counts["category_selection"]
        >= gate["minimum_stable_category_selection_units_of_6"],
        "color_selection_units": counts["color_selection"]
        >= gate["minimum_stable_color_selection_units_of_6"],
        "case_rate_diagnostic": case_rate
        >= gate["minimum_case_level_stable_rate_diagnostic"],
    }
    passed = all(gate_checks.values())
    return {
        "model": model,
        "stage": stage,
        "behavior_gate_pass": passed,
        "single_model_trace_authorized": passed,
        "internal_trace_authorized": passed,
        "blocked_reason": None if passed else "frozen_behavior_gate_failed",
        "stable_case_count": sum(stable.values()),
        "case_count": len(stable),
        "stable_case_rate_diagnostic": case_rate,
        "independent_unit_count": len(units),
        "stable_independent_unit_counts": counts,
        "independent_unit_reports": units,
        "gate_checks": gate_checks,
        "event_counts": event_counts,
        "generation_capsule_counts": capsule_counts,
        "trace_case_ids": sorted(stable) if passed else [],
        "trace_includes_incorrect_and_unrecoverable_negative_controls": passed,
        "behavior_rows_sha256": sha256_file(rows_file),
        "behavior_summary_sha256": sha256_file(summary_file),
        "behavior_contract_sha256": sha256_file(contract_file),
        "internal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "sealed_model_access": False,
    }


def validate_receipt(
    stage: str,
    receipt: dict[str, Any],
    frozen: dict[str, Any],
    engineering: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    stage_hash = sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    protocol_hash = sha256_file(protocol.PROTOCOL_PATH)
    attempts = receipt.get("attempts")
    failed = receipt.get("failed_models")
    completed = receipt.get("completed_models")
    started_file = stage_started_path(stage)
    started = read_json(started_file) if started_file.is_file() else {}
    receipt_checks = {
        "schema": receipt.get("schema_version")
        == "phase576_behavior_execution_receipt.v1",
        "identity": all((
            receipt.get("phase_id") == protocol.PHASE,
            receipt.get("stage") == stage,
            isinstance(receipt.get("created_at_utc"), str),
            bool(receipt.get("created_at_utc")),
        )),
        "order": receipt.get("models_attempted_in_order")
        == list(protocol.MODELS),
        "stage_hash": receipt.get("stage_cases_sha256") == stage_hash,
        "protocol_hash": receipt.get("protocol_sha256") == protocol_hash,
        "source_hash": receipt.get("behavior_source_sha256")
        == frozen["stage_source_seals"][
            "tests/glm5/phase576_gpt5_fruit_behavior.py"
        ]["sha256"],
        "engineering_chain": all((
            receipt.get("engineering_qualification_sha256")
            == engineering["qualification_sha256"],
            receipt.get("engineering_execution_receipt_sha256")
            == engineering["execution_receipt_sha256"],
            receipt.get("runtime_identity") == engineering["runtime_identity"],
        )),
        "stage_started": (
            started.get("schema_version") == "phase576_behavior_stage_started.v1"
            and started.get("phase_id") == protocol.PHASE
            and started.get("stage") == stage
            and started.get("models_planned_in_required_order")
            == list(protocol.MODELS)
            and started.get("stage_cases_sha256") == stage_hash
            and started.get("protocol_sha256") == protocol_hash
            and started.get("behavior_source_sha256")
            == frozen["stage_source_seals"][
                "tests/glm5/phase576_gpt5_fruit_behavior.py"
            ]["sha256"]
            and started.get("engineering_qualification_sha256")
            == engineering["qualification_sha256"]
            and started.get("engineering_execution_receipt_sha256")
            == engineering["execution_receipt_sha256"]
            and started.get("runtime_identity") == engineering["runtime_identity"]
            and started.get("sealed_model_access") is False
            and receipt.get("stage_started_sha256") == sha256_file(started_file)
        ) if started_file.is_file() else False,
        "sealed_model_access": receipt.get("sealed_model_access") is False,
        "attempt_registry": isinstance(attempts, list)
        and len(attempts) == len(protocol.MODELS),
        "completed_registry": isinstance(completed, list),
        "failed_registry": isinstance(failed, list),
        "terminal_complete": receipt.get("terminal_status") == "complete",
        "no_fatal_or_unattempted": receipt.get("fatal_error") is None
        and receipt.get("not_attempted_models") == [],
        "final_cuda_clean": receipt.get("final_pytorch_cuda_allocated") == 0
        and isinstance(receipt.get("final_pytorch_cuda_reserved"), int)
        and receipt.get("final_pytorch_cuda_reserved", -1) >= 0,
    }
    if not all(receipt_checks.values()):
        raise RuntimeError(f"invalid Phase576 behavior receipt: {receipt_checks}")
    if [item.get("model") for item in attempts] != list(protocol.MODELS):
        raise RuntimeError("behavior receipt attempt order drift")
    attempt_by_model: dict[str, dict[str, Any]] = {}
    for item in attempts:
        model = item.get("model")
        state = item.get("status")
        if state not in {"complete", "failed"} or model in attempt_by_model:
            raise RuntimeError("behavior receipt attempt registry invalid")
        if state == "failed" and not (
            isinstance(item.get("error_type"), str) and item["error_type"]
        ):
            raise RuntimeError("behavior receipt failed attempt lacks error type")
        attempt_by_model[model] = item
    expected_completed = [
        model for model in protocol.MODELS
        if attempt_by_model[model]["status"] == "complete"
    ]
    expected_failed = [
        model for model in protocol.MODELS
        if attempt_by_model[model]["status"] == "failed"
    ]
    if completed != expected_completed or [item.get("model") for item in failed] != expected_failed:
        raise RuntimeError("behavior receipt completed/failed model registry drift")
    for item in failed:
        model = item.get("model")
        if not all((
            isinstance(item.get("error_type"), str),
            bool(item.get("error_type")),
            item.get("error_type") == attempt_by_model[model].get("error_type"),
            isinstance(item.get("error"), str),
        )):
            raise RuntimeError(f"invalid failure receipt for {model}")
    return attempt_by_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=protocol.OPEN_SPLITS)
    args = parser.parse_args()
    stage = args.stage
    output_path = protocol.BEHAVIOR_DECISION_PATHS[stage]
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite Phase576 {stage} behavior decision")
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.STATIC_AUDIT_PATH)
    commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    protocol.verify_frozen_source_seals(frozen)
    protocol.verify_frozen_model_artifacts(frozen)
    qualification = behavior_execution.verify_engineering_qualification(frozen)
    engineering = {
        "qualification_sha256": sha256_file(
            protocol.ENGINEERING_QUALIFICATION_PATH
        ),
        "execution_receipt_sha256": qualification[
            "execution_receipt_sha256"
        ],
        "runtime_identity": qualification["runtime_identity"],
    }
    freeze_commit = protocol._verify_freeze_commit()
    protocol_hash = sha256_file(protocol.PROTOCOL_PATH)
    commitment_hash = sha256_file(protocol.SEALED_COMMITMENT_PATH)
    stage_hash = sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    seal_checks = {
        "protocol_phase": frozen.get("phase_id") == protocol.PHASE,
        "protocol_source_hash": frozen.get("source_script_sha256")
        == sha256_file(Path(protocol.__file__).resolve()),
        "audit_schema": audit.get("schema_version") == "phase576_static_audit.v2",
        "audit_phase": audit.get("phase_id") == protocol.PHASE,
        "audit_valid": audit.get("valid") is True and not audit.get("failures"),
        "audit_cpu_no_models": audit.get("model_weights_loaded") is False
        and audit.get("cuda_used") is False,
        "audit_grid_valid": audit.get("case_grid_audit", {}).get("valid") is True,
        "audit_cross_model_comparison_rule": isinstance(
            audit.get("cross_model_observational_comparison_rule_audit"), dict
        ) and bool(audit["cross_model_observational_comparison_rule_audit"])
        and all(
            audit["cross_model_observational_comparison_rule_audit"].values()
        ),
        "audit_protocol_hash": audit.get("protocol_sha256") == protocol_hash,
        "audit_commitment_hash": audit.get("sealed_commitment_sha256")
        == commitment_hash,
        "frozen_commitment_hash": frozen.get("sealed_commitment_sha256")
        == commitment_hash,
        "frozen_stage_hash": frozen.get("open_case_sha256_by_split", {}).get(stage)
        == stage_hash,
        "audit_stage_hash": audit.get("open_case_sha256_by_split", {}).get(stage)
        == stage_hash,
        "audit_no_prior_sealed_read": audit.get("prior_sealed_files_read") == [],
        "audit_no_sealed_analysis": audit.get(
            "sealed_model_or_result_read_for_analysis"
        ) is False,
        "audit_prior_open_identity": audit.get("prior_open_file_identities")
        == frozen.get("prior_open_file_identities"),
        "commitment_schema": commitment.get("schema_version")
        == "phase576_sealed_commitment.v2",
        "commitment_phase": commitment.get("phase_id") == protocol.PHASE,
        "holdout_not_blind": commitment.get("holdout_is_blind") is False,
        "sealed_definition_public": commitment.get(
            "sealed_definition_is_public_in_source"
        ) is True,
        "sealed_model_unopened": commitment.get("sealed_model_opened") is False,
        "sealed_model_access_zero": commitment.get("sealed_model_access_count") == 0,
        "sealed_analysis_access_zero": commitment.get(
            "sealed_result_analysis_access_count"
        ) == 0,
        "prior_sealed_unread": commitment.get("prior_sealed_files_read") is False,
        "freeze_commit_complete": freeze_commit.get("complete") is True,
        "freeze_lock_absent": not protocol.FREEZE_LOCK_PATH.exists(),
    }
    if not all(seal_checks.values()):
        raise RuntimeError(
            f"Phase576 seal invalid before behavior analysis: {seal_checks}"
        )
    if stage == "discovery":
        if any(path.exists() for path in (
            protocol.DISCOVERY_REGISTRY_PATH,
            protocol.CONFIRMATION_DECISION_PATH,
            protocol.HELDOUT_DECISION_PATH,
            protocol.BEHAVIOR_DECISION_PATHS["confirmation"],
            protocol.BEHAVIOR_DECISION_PATHS["heldout_recombination"],
        )):
            raise RuntimeError("future-stage artifact exists during discovery analysis")
    elif stage == "confirmation":
        discovery_registry = protocol.verify_discovery_registry(frozen)
        if discovery_registry.get("discovery_candidate_pass") is not True:
            raise RuntimeError(
                "confirmation analysis requires passed discovery evidence"
            )
        if (
            protocol.CONFIRMATION_DECISION_PATH.exists()
            or protocol.HELDOUT_DECISION_PATH.exists()
            or protocol.BEHAVIOR_DECISION_PATHS["heldout_recombination"].exists()
        ):
            raise RuntimeError("future-stage artifact exists during confirmation analysis")
    else:
        confirmation = protocol.verify_confirmation_decision(frozen)
        if confirmation.get("structure_confirmation_pass") is not True:
            raise RuntimeError("heldout analysis requires passed confirmation decision")
        if protocol.HELDOUT_DECISION_PATH.exists():
            raise RuntimeError("heldout structure decision exists before behavior analysis")
    validate_gate_contract(frozen.get("behavior_gate", {}))
    if not receipt_path(stage).exists():
        raise RuntimeError(f"missing Phase576 {stage} execution receipt")
    receipt = read_json(receipt_path(stage))
    attempt_by_model = validate_receipt(stage, receipt, frozen, engineering)
    cases = read_jsonl(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    if (
        len(cases) != 336
        or len({row.get("case_id") for row in cases}) != len(cases)
        or any(
            row.get("phase_id") != protocol.PHASE
            or row.get("split") != stage
            or row.get("sealed") is not False
            for row in cases
        )
    ):
        raise RuntimeError(f"Phase576 {stage} denominator drift")
    reports = [
        analyze_model(
            model, stage, cases, frozen, attempt_by_model[model], engineering
        )
        for model in protocol.MODELS
    ]
    qualified_models = [
        report["model"] for report in reports if report["behavior_gate_pass"]
    ]
    cross_model_comparison_authorized = (
        protocol.cross_model_observational_comparison_authorized(
            stage,
            {report["model"]: report["behavior_gate_pass"] for report in reports},
        )
    )
    payload = {
        "schema_version": "phase576_behavior_decision.v2",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "stage": stage,
        "models_in_required_execution_order": list(protocol.MODELS),
        "reports": reports,
        "qualified_models": qualified_models,
        "single_model_trace_authorized_models": qualified_models,
        "cross_model_observational_comparison_authorized": (
            cross_model_comparison_authorized
        ),
        "cross_model_observational_comparison_scope": (
            "this stage's behavior qualification and observational trace only; "
            "not an internal mechanism or causal claim"
        ),
        "blocked_models": [r["model"] for r in reports if not r["behavior_gate_pass"]],
        "stage_cases_sha256": stage_hash,
        "behavior_execution_receipt_sha256": sha256_file(receipt_path(stage)),
        "engineering_qualification_sha256": engineering[
            "qualification_sha256"
        ],
        "engineering_execution_receipt_sha256": engineering[
            "execution_receipt_sha256"
        ],
        "runtime_identity": engineering["runtime_identity"],
        "protocol_sha256": protocol_hash,
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "analysis_source_seal": frozen["stage_source_seals"][
            "tests/glm5/phase576_gpt5_fruit_behavior_analysis.py"
        ],
        "analysis_unit_definition": (
            "direct: object x relation; selection: semantic object pair; "
            "surface/order are paired repeats; statistical independence is not claimed"
        ),
        "trace_selection_rule": "all 336 preregistered cases for each qualified model",
        "cross_model_observational_comparison_rule": (
            "all qwen3, glm4, and deepseek7b models must pass this stage; "
            "a single qualified model authorizes only its own trace"
        ),
        "sealed_model_access": False,
        "internal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
    }
    protocol.verify_frozen_model_artifacts(frozen)
    if (
        sha256_file(protocol.ENGINEERING_QUALIFICATION_PATH)
        != engineering["qualification_sha256"]
        or behavior_execution.verify_engineering_qualification(frozen)
        != qualification
    ):
        raise RuntimeError("engineering qualification drift during behavior analysis")
    write_json(output_path, payload)
    print(json.dumps({
        "stage": stage,
        "qualified_models": payload["qualified_models"],
        "blocked_models": payload["blocked_models"],
        "cross_model_observational_comparison_authorized": payload[
            "cross_model_observational_comparison_authorized"
        ],
        "stable_independent_units": {
            report["model"]: report.get("stable_independent_unit_counts", {})
            for report in reports
        },
        "sealed_model_access": False,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
