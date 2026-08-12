#!/usr/bin/env python3
"""Audit Phase1005 protocol, ranking, gates, controls, and artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PHASE = 1005
ROOT = (
    Path(__file__).resolve().parent
    / "result"
    / "phase1005_blind_layerwise_source_compression"
)
OUT_ROOT = ROOT / "audit"
RUNS = (
    ("8bit", "qwen3"),
    ("8bit", "glm4"),
    ("8bit", "deepseek7b"),
    ("bf16", "qwen3"),
)
FORMAL_ROOTS = (
    ROOT / "8bit",
    ROOT / "bf16",
    ROOT / "analysis",
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: {error}"
                ) from error
    return rows


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def check(
    errors: list[dict[str, str]],
    condition: bool,
    code: str,
    detail: str,
) -> None:
    if not condition:
        errors.append({"code": code, "detail": detail})


def audit_donor(
    audit: dict[str, Any],
    context: str,
    errors: list[dict[str, str]],
) -> None:
    check(
        errors,
        audit["candidate_pool_source"]
        == "complete_frozen_protocol_model_domain_split_template",
        "donor_pool_source",
        context,
    )
    check(
        errors,
        int(audit["candidate_pool_count"]) == 512,
        "donor_pool_count",
        f"{context}: {audit['candidate_pool_count']}",
    )
    expected_recipients = int(audit["recipient_count"])
    check(
        errors,
        expected_recipients in {16, 64},
        "donor_recipient_count",
        f"{context}: {expected_recipients}",
    )
    check(
        errors,
        int(audit["unique_donor_count"]) == expected_recipients,
        "donor_unique_count",
        context,
    )
    check(
        errors,
        float(audit["unique_donor_fraction"]) == 1.0,
        "donor_unique_fraction",
        context,
    )
    check(
        errors,
        int(audit["maximum_donor_reuse"]) == 1,
        "donor_reuse",
        context,
    )
    check(
        errors,
        bool(audit["all_cross_world"]),
        "donor_cross_world",
        context,
    )
    check(
        errors,
        bool(audit["all_answer_contracts_hold"]),
        "donor_answer_contract",
        context,
    )
    if not audit["same_answer_control"]:
        check(
            errors,
            bool(audit["different_answer_value_sets_disjoint"]),
            "donor_value_disjoint",
            context,
        )


def discovery_gate(metrics: dict[str, Any]) -> bool:
    return all(
        float(item["donor_rate"]) >= 0.80
        and float(item["median_normalized_transfer"]) >= 0.50
        for item in metrics["template_metrics"].values()
    )


def confirmation_gate(event: dict[str, Any]) -> bool:
    different = event["confirmation_different_answer"]
    same = event["confirmation_same_answer"]
    noop = event["confirmation_target_noop"]
    return all(
        float(different["template_metrics"][template]["donor_rate"])
        >= 0.80
        and float(
            different["template_metrics"][template][
                "median_normalized_transfer"
            ]
        )
        >= 0.50
        and float(same["template_metrics"][template]["target_rate"])
        >= 0.95
        and float(noop["template_metrics"][template]["target_rate"])
        >= 0.99
        for template in sorted(different["template_metrics"])
    )


def ranking_key(item: tuple[str, dict[str, Any]]):
    event_id, summary = item
    templates = list(summary["template_metrics"].values())
    return (
        -min(float(value["donor_rate"]) for value in templates),
        -min(
            float(value["median_normalized_transfer"])
            for value in templates
        ),
        -float(summary["donor_rate"]),
        -float(summary["median_normalized_transfer"]),
        event_id,
    )


def audit_domain(
    root: Path,
    value: dict[str, Any],
    context: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    if value["status"] != "complete":
        check(
            errors,
            value["status"]
            == "upstream_behavior_parent_gate_failed",
            "unexpected_domain_skip",
            context,
        )
        check(
            errors,
            not value["compressed_single_position_found"]
            and int(value["compressed_event_pass_count"]) == 0,
            "skipped_domain_positive",
            context,
        )
        return {
            "context": context,
            "status": value["status"],
            "event_universe_count": 0,
            "discovery_rows": 0,
            "confirmation_rows": 0,
            "compressed_event_pass_count": 0,
        }

    check(
        errors,
        not value["selection_uses_semantic_labels"],
        "semantic_selection_leak",
        context,
    )
    check(
        errors,
        not value["selection_uses_confirmation"],
        "confirmation_selection_leak",
        context,
    )
    check(
        errors,
        int(value["discovery_n"]) == 16,
        "discovery_n",
        context,
    )
    check(
        errors,
        int(value["confirmation_n"]) == 64,
        "confirmation_n",
        context,
    )
    check(
        errors,
        int(value["frozen_event_count"]) == 12,
        "frozen_event_count",
        context,
    )
    for key in (
        "discovery_donor_audit",
        "discovery_same_answer_donor_audit",
        "confirmation_donor_audit",
        "confirmation_same_answer_donor_audit",
    ):
        audit_donor(value[key], f"{context}/{key}", errors)

    discovery_path = root / "discovery_rows.jsonl"
    confirmation_path = root / "confirmation_rows.jsonl"
    discovery_rows = count_jsonl(discovery_path)
    confirmation_rows = count_jsonl(confirmation_path)
    check(
        errors,
        discovery_rows
        == int(value["event_universe_count"]) * 16,
        "discovery_row_count",
        (
            f"{context}: {discovery_rows} vs "
            f"{value['event_universe_count'] * 16}"
        ),
    )
    check(
        errors,
        confirmation_rows == 12 * 64 * 3,
        "confirmation_row_count",
        f"{context}: {confirmation_rows}",
    )

    summaries = read_json(
        root / "discovery_event_summaries.json"
    )
    check(
        errors,
        len(summaries) == int(value["event_universe_count"]),
        "discovery_summary_count",
        context,
    )
    expected_ids = [
        event_id
        for event_id, _ in sorted(
            summaries.items(), key=ranking_key
        )[:12]
    ]
    stored_ids = [
        event["event_id"] for event in value["frozen_events"]
    ]
    check(
        errors,
        stored_ids == expected_ids,
        "frozen_ranking_drift",
        (
            f"{context}: expected={expected_ids} "
            f"stored={stored_ids}"
        ),
    )

    pass_count = 0
    for rank, event in enumerate(value["frozen_events"], 1):
        event_context = f"{context}/{event['event_id']}"
        check(
            errors,
            int(event["discovery_rank"]) == rank,
            "discovery_rank",
            event_context,
        )
        check(
            errors,
            not event["selection_uses_semantic_labels"]
            and not event["selection_uses_confirmation"],
            "event_selection_leak",
            event_context,
        )
        semantic = event["semantic_reconstruction_audit"]
        check(
            errors,
            semantic["revealed_after_selection"]
            and not semantic["selection_uses_this_audit"],
            "semantic_reveal_order",
            event_context,
        )
        gate = discovery_gate(
            event["discovery_metrics"]
        ) and confirmation_gate(event)
        check(
            errors,
            bool(event["compressed_event_gate_pass"]) == gate,
            "compressed_event_gate",
            event_context,
        )
        pass_count += gate
    check(
        errors,
        pass_count == int(value["compressed_event_pass_count"]),
        "compressed_event_pass_count",
        context,
    )
    check(
        errors,
        bool(value["compressed_single_position_found"])
        == bool(pass_count),
        "compressed_domain_flag",
        context,
    )
    return {
        "context": context,
        "status": value["status"],
        "event_universe_count": int(value["event_universe_count"]),
        "discovery_rows": discovery_rows,
        "confirmation_rows": confirmation_rows,
        "compressed_event_pass_count": pass_count,
    }


def audit_formal_files(
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    json_files = [ROOT / "preregistered_protocol.json"]
    jsonl_files = []
    for root in FORMAL_ROOTS:
        json_files.extend(root.rglob("*.json"))
        jsonl_files.extend(root.rglob("*.jsonl"))
    json_files = sorted(set(json_files))
    jsonl_files = sorted(set(jsonl_files))
    jsonl_rows = 0
    finite_float_count = 0
    phase_field_count = 0
    for path in json_files:
        values = [read_json(path)]
        for value in values:
            for item in walk(value):
                if isinstance(item, float):
                    finite_float_count += 1
                    check(
                        errors,
                        math.isfinite(item),
                        "non_finite",
                        str(path),
                    )
                if isinstance(item, dict) and "phase" in item:
                    phase_field_count += 1
                    check(
                        errors,
                        item["phase"] == PHASE,
                        "phase_mismatch",
                        f"{path}: {item['phase']}",
                    )
    for path in jsonl_files:
        for value in read_jsonl(path):
            jsonl_rows += 1
            for item in walk(value):
                if isinstance(item, float):
                    finite_float_count += 1
                    check(
                        errors,
                        math.isfinite(item),
                        "non_finite",
                        str(path),
                    )
                if isinstance(item, dict) and "phase" in item:
                    phase_field_count += 1
                    check(
                        errors,
                        item["phase"] == PHASE,
                        "phase_mismatch",
                        f"{path}: {item['phase']}",
                    )
    return {
        "json_file_count": len(json_files),
        "jsonl_file_count": len(jsonl_files),
        "jsonl_row_count": jsonl_rows,
        "finite_float_count": finite_float_count,
        "phase_field_count": phase_field_count,
    }


def main() -> None:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    protocol = read_json(ROOT / "preregistered_protocol.json")
    computed_digest = digest({
        key: value
        for key, value in protocol.items()
        if key != "preregistration_digest"
    })
    check(
        errors,
        int(protocol["protocol_revision"]) == 2,
        "protocol_revision",
        str(protocol["protocol_revision"]),
    )
    check(
        errors,
        protocol["preregistration_digest"] == computed_digest,
        "protocol_digest",
        (
            f"stored={protocol['preregistration_digest']} "
            f"computed={computed_digest}"
        ),
    )
    check(
        errors,
        not protocol["revision_audit"][
            "revision_1_result_used"
        ],
        "invalid_revision_used",
        "revision_1_result_used",
    )

    run_rows = []
    domain_rows = []
    for precision, model in RUNS:
        model_root = ROOT / precision / model
        summary = read_json(model_root / "summary.json")
        check(
            errors,
            summary["precision"] == precision,
            "precision_mismatch",
            f"{precision}/{model}: {summary['precision']}",
        )
        qualified = 0
        compressed_domains = 0
        compressed_events = 0
        for domain, value in summary["domains"].items():
            result = audit_domain(
                model_root / domain,
                value,
                f"{precision}/{model}/{domain}",
                errors,
            )
            domain_rows.append(result)
            qualified += value["status"] == "complete"
            compressed_domains += bool(
                value["compressed_single_position_found"]
            )
            compressed_events += int(
                value["compressed_event_pass_count"]
            )
        check(
            errors,
            qualified
            == int(summary["parent_qualified_domain_count"]),
            "qualified_domain_count",
            f"{precision}/{model}: {qualified}",
        )
        check(
            errors,
            compressed_domains
            == int(summary["compressed_domain_count"]),
            "compressed_domain_count",
            f"{precision}/{model}: {compressed_domains}",
        )
        check(
            errors,
            compressed_events
            == int(summary["compressed_event_pass_count"]),
            "compressed_model_event_count",
            f"{precision}/{model}: {compressed_events}",
        )
        run_rows.append({
            "precision": precision,
            "model": model,
            "qualified_domain_count": qualified,
            "compressed_domain_count": compressed_domains,
            "compressed_event_pass_count": compressed_events,
        })

    invalid_root = ROOT / "invalid_pre_semantic_boundary_fix"
    if invalid_root.exists():
        warnings.append({
            "code": "invalid_revision_artifacts_excluded",
            "detail": str(invalid_root),
        })
    formal_files = audit_formal_files(errors)
    result = {
        "schema_version": "phase1005_result_audit.v1",
        "phase": PHASE,
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "protocol_revision": int(protocol["protocol_revision"]),
        "stored_protocol_digest": protocol[
            "preregistration_digest"
        ],
        "computed_protocol_digest": computed_digest,
        "protocol_digest_match": protocol[
            "preregistration_digest"
        ]
        == computed_digest,
        "runs": run_rows,
        "domains": domain_rows,
        "formal_files": formal_files,
        "claim_boundary": (
            "The audit verifies the frozen ranking, controls, "
            "row counts, and operational gates. It does not prove "
            "that a NO-GO extends beyond the tested positions, "
            "tasks, models, or intervention interface."
        ),
    }
    write_json(OUT_ROOT / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
