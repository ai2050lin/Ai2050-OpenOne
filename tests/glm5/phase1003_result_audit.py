#!/usr/bin/env python3
"""Mechanical completeness and leakage audit for all Phase1003 artifacts."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1003_crossparadigm_protocol import (
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    write_json,
)
from phase1003_structural_stress_protocol import STRESS_ROOT
from phase1003_rollout_surface_protocol import (
    ROLLOUT_ROOT,
    SURFACES,
)


AUDIT_ROOT = OUT_ROOT / "audit"


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks.append({
        "check_id": check_id,
        "pass": bool(passed),
        "observed": observed,
        "expected": expected,
    })


def nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    result = []
    if isinstance(value, float) and not math.isfinite(value):
        result.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            result.extend(
                nonfinite_paths(item, f"{prefix}.{key}")
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(
                nonfinite_paths(item, f"{prefix}[{index}]")
            )
    return result


def parse_artifacts() -> dict[str, Any]:
    parse_failures = []
    nonfinite = []
    missing_schema = []
    phase_drift = []
    json_count = 0
    jsonl_count = 0
    jsonl_rows = 0
    for path in sorted(OUT_ROOT.rglob("*.json")):
        json_count += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_failures.append({
                "path": str(path.relative_to(ROOT)),
                "error": repr(exc),
            })
            continue
        found = nonfinite_paths(payload)
        if found:
            nonfinite.append({
                "path": str(path.relative_to(ROOT)),
                "locations": found[:20],
            })
        if isinstance(payload, dict):
            if "schema_version" not in payload:
                missing_schema.append(str(path.relative_to(ROOT)))
            if "phase" in payload and payload["phase"] != PHASE:
                phase_drift.append({
                    "path": str(path.relative_to(ROOT)),
                    "phase": payload["phase"],
                })
    for path in sorted(OUT_ROOT.rglob("*.jsonl")):
        jsonl_count += 1
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            jsonl_rows += 1
            try:
                payload = json.loads(line)
            except Exception as exc:
                parse_failures.append({
                    "path": str(path.relative_to(ROOT)),
                    "line": line_number,
                    "error": repr(exc),
                })
                continue
            found = nonfinite_paths(payload)
            if found:
                nonfinite.append({
                    "path": str(path.relative_to(ROOT)),
                    "line": line_number,
                    "locations": found[:20],
                })
            if isinstance(payload, dict):
                if "schema_version" not in payload:
                    missing_schema.append(
                        f"{path.relative_to(ROOT)}:{line_number}"
                    )
                if (
                    "phase" in payload
                    and payload["phase"] != PHASE
                ):
                    phase_drift.append({
                        "path": (
                            f"{path.relative_to(ROOT)}:"
                            f"{line_number}"
                        ),
                        "phase": payload["phase"],
                    })
    return {
        "json_file_count": json_count,
        "jsonl_file_count": jsonl_count,
        "jsonl_row_count": jsonl_rows,
        "parse_failures": parse_failures,
        "nonfinite_values": nonfinite,
        "missing_schema": missing_schema,
        "phase_drift": phase_drift,
    }


def line_count(path: Path) -> int:
    return sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def main_protocol_checks(
    checks: list[dict[str, Any]],
) -> None:
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    add_check(
        checks,
        "main_protocol_revision",
        prereg["protocol_revision"] == 4,
        prereg["protocol_revision"],
        4,
    )
    add_check(
        checks,
        "main_answer_surface_frozen_before_formal_internal_tests",
        (
            not prereg["revision_audit"][
                "formal_data_used_in_calibration"
            ]
            and not prereg["revision_audit"][
                "internal_results_observed_before_revision"
            ]
        ),
        {
            "formal_data_used_in_calibration": prereg[
                "revision_audit"
            ]["formal_data_used_in_calibration"],
            "internal_results_observed_before_revision": prereg[
                "revision_audit"
            ]["internal_results_observed_before_revision"],
        },
        {
            "formal_data_used_in_calibration": False,
            "internal_results_observed_before_revision": False,
        },
    )
    for model_name in MODELS:
        audit = read_json(
            OUT_ROOT
            / "protocol"
            / model_name
            / "protocol_audit.json"
        )
        add_check(
            checks,
            f"main_protocol_count:{model_name}",
            (
                audit["case_count"] == 8192
                and audit["pair_count"] == 4096
                and audit["selected_direction_count_per_domain_split"]
                == 64
            ),
            {
                "case_count": audit["case_count"],
                "pair_count": audit["pair_count"],
                "selected_direction_count_per_domain_split": audit[
                    "selected_direction_count_per_domain_split"
                ],
            },
            {
                "case_count": 8192,
                "pair_count": 4096,
                "selected_direction_count_per_domain_split": 64,
            },
        )
        add_check(
            checks,
            f"main_vocab_split:{model_name}",
            audit["discovery_confirmation_name_overlap"] == [],
            audit["discovery_confirmation_name_overlap"],
            [],
        )
    behavior = read_json(OUT_ROOT / "behavior" / "summary.json")
    add_check(
        checks,
        "main_behavior_all_models_complete",
        behavior["all_models_complete"],
        behavior["all_models_complete"],
        True,
    )


def main_causal_checks(
    checks: list[dict[str, Any]],
) -> None:
    anchor = read_json(OUT_ROOT / "anchor_subsets" / "summary.json")
    natural = read_json(OUT_ROOT / "anchor_natural" / "summary.json")
    cache = read_json(OUT_ROOT / "kv_replication" / "summary.json")
    layers = read_json(
        OUT_ROOT / "value_layer_relocalization" / "summary.json"
    )
    heads = read_json(
        OUT_ROOT / "value_head_localization" / "summary.json"
    )
    channels = read_json(
        OUT_ROOT
        / "value_channel_block_localization"
        / "summary.json"
    )
    for name, payload in (
        ("anchor", anchor),
        ("natural", natural),
        ("cache", cache),
        ("layers", layers),
        ("heads", heads),
        ("channels", channels),
    ):
        add_check(
            checks,
            f"main_complete:{name}",
            payload["all_models_complete"],
            payload["all_models_complete"],
            True,
        )
    for model_name, model_summary in anchor["models"].items():
        for domain, summary in model_summary["domains"].items():
            add_check(
                checks,
                f"anchor_no_confirmation_selection:"
                f"{model_name}:{domain}",
                not summary["discovery_selection"][
                    "selection_uses_confirmation"
                ],
                summary["discovery_selection"][
                    "selection_uses_confirmation"
                ],
                False,
            )
            add_check(
                checks,
                f"anchor_noop:{model_name}:{domain}",
                all(
                    summary["controls"][split][
                        "empty_noop_prediction_agreement"
                    ] == 1.0
                    for split in ("discovery", "confirmation")
                ),
                {
                    split: summary["controls"][split][
                        "empty_noop_prediction_agreement"
                    ]
                    for split in ("discovery", "confirmation")
                },
                {"discovery": 1.0, "confirmation": 1.0},
            )
    for model_name, model_summary in natural["models"].items():
        for domain, summary in model_summary["domains"].items():
            noop = summary["condition_summary"]["target_noop"][
                "noop_sequence_agreement"
            ]
            add_check(
                checks,
                f"natural_noop:{model_name}:{domain}",
                noop == 1.0,
                noop,
                1.0,
            )
    for aggregate_name, aggregate in (
        ("layers", layers),
        ("heads", heads),
        ("channels", channels),
    ):
        for model_name, model_summary in aggregate["models"].items():
            leak_values = []

            def visit(value: Any) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "selection_uses_confirmation":
                            leak_values.append(item)
                        visit(item)
                elif isinstance(value, list):
                    for item in value:
                        visit(item)

            visit(model_summary)
            add_check(
                checks,
                f"no_confirmation_selection:{aggregate_name}:"
                f"{model_name}",
                all(value is False for value in leak_values),
                leak_values,
                "all false",
            )


def stress_checks(checks: list[dict[str, Any]]) -> None:
    prereg = read_json(
        STRESS_ROOT / "preregistered_protocol.json"
    )
    add_check(
        checks,
        "stress_internal_results_not_used_for_prompts",
        not prereg[
            "internal_results_used_to_define_stress_prompts"
        ],
        prereg[
            "internal_results_used_to_define_stress_prompts"
        ],
        False,
    )
    for model_name in MODELS:
        audit = read_json(
            STRESS_ROOT
            / "protocol"
            / model_name
            / "protocol_audit.json"
        )
        add_check(
            checks,
            f"stress_protocol_count:{model_name}",
            audit["case_count"] == 832,
            audit["case_count"],
            832,
        )
        add_check(
            checks,
            f"stress_vocab_split:{model_name}",
            audit["discovery_confirmation_name_overlap"] == [],
            audit["discovery_confirmation_name_overlap"],
            [],
        )
        behavior = read_json(
            STRESS_ROOT
            / "behavior"
            / model_name
            / "summary.json"
        )
        add_check(
            checks,
            f"stress_behavior_count:{model_name}",
            behavior["case_count"] == 832,
            behavior["case_count"],
            832,
        )
        add_check(
            checks,
            f"stress_behavior_teacher_rows:{model_name}",
            line_count(
                STRESS_ROOT
                / "behavior"
                / model_name
                / "teacher_rows.jsonl"
            ) == 832,
            line_count(
                STRESS_ROOT
                / "behavior"
                / model_name
                / "teacher_rows.jsonl"
            ),
            832,
        )
        causal = read_json(
            STRESS_ROOT
            / "causal"
            / model_name
            / "summary.json"
        )
        for task, summary in causal["tasks"].items():
            roles = summary["roles"]
            split_n = summary["teacher"]["discovery"][
                "conditions"
            ]["target_noop"]["n"]
            expected_teacher = 2 * split_n * (2 + len(roles))
            expected_natural = split_n * 2
            expected_cache = 2 * split_n * 4
            task_root = (
                STRESS_ROOT / "causal" / model_name / task
            )
            actual = {
                "teacher": line_count(
                    task_root / "teacher_rows.jsonl"
                ),
                "natural": line_count(
                    task_root / "natural_rows.jsonl"
                ),
                "cache": line_count(
                    task_root / "cache_rows.jsonl"
                ),
            }
            expected = {
                "teacher": expected_teacher,
                "natural": expected_natural,
                "cache": expected_cache,
            }
            add_check(
                checks,
                f"stress_row_counts:{model_name}:{task}",
                actual == expected,
                actual,
                expected,
            )
            noops = {
                "teacher_discovery": summary["teacher"][
                    "discovery"
                ]["conditions"]["target_noop"][
                    "prediction_agreement"
                ],
                "teacher_confirmation": summary["teacher"][
                    "confirmation"
                ]["conditions"]["target_noop"][
                    "prediction_agreement"
                ],
                "natural": summary["natural_confirmation"][
                    "conditions"
                ]["target_noop"]["noop_sequence_agreement"],
            }
            add_check(
                checks,
                f"stress_noops:{model_name}:{task}",
                all(value == 1.0 for value in noops.values()),
                noops,
                "all 1.0",
            )
            donors = summary["donor_audits"]
            add_check(
                checks,
                f"stress_donors:{model_name}:{task}",
                all(
                    item["all_cross_world"]
                    and item[
                        "all_donor_answers_differ_from_target"
                    ]
                    for item in donors.values()
                ),
                donors,
                "all cross-world and answer-changing",
            )
    aggregate = read_json(
        STRESS_ROOT / "causal" / "summary.json"
    )
    add_check(
        checks,
        "stress_causal_all_models_complete",
        aggregate["all_models_complete"],
        aggregate["all_models_complete"],
        True,
    )


def precision_checks(checks: list[dict[str, Any]]) -> None:
    audit = read_json(
        OUT_ROOT
        / "precision_audit"
        / "qwen3_bf16"
        / "summary.json"
    )
    add_check(
        checks,
        "bf16_audit_complete",
        audit["status"] == "complete",
        audit["status"],
        "complete",
    )
    add_check(
        checks,
        "bf16_noop_instruments",
        audit["all_noop_instruments_pass"],
        audit["all_noop_instruments_pass"],
        True,
    )
    add_check(
        checks,
        "bf16_cache_instruments",
        audit["all_cache_instruments_pass"],
        audit["all_cache_instruments_pass"],
        True,
    )
    add_check(
        checks,
        "bf16_value_exceeds_key",
        audit["value_exceeds_key_in_all_tasks"],
        audit["value_exceeds_key_in_all_tasks"],
        True,
    )
    add_check(
        checks,
        "bf16_pronoun_quantization_flip_removed",
        audit["tasks"]["pronoun"][
            "instrument_disagreement_removed_in_bf16"
        ],
        audit["tasks"]["pronoun"][
            "instrument_disagreement_removed_in_bf16"
        ],
        True,
    )


def rollout_checks(checks: list[dict[str, Any]]) -> None:
    prereg = read_json(
        ROLLOUT_ROOT / "preregistered_protocol.json"
    )
    add_check(
        checks,
        "rollout_surfaces_frozen_without_internal_results",
        (
            not prereg[
                "internal_results_used_to_select_surfaces"
            ]
            and not prereg[
                "surface_selection_uses_behavior_results"
            ]
        ),
        {
            "internal_results_used_to_select_surfaces": prereg[
                "internal_results_used_to_select_surfaces"
            ],
            "surface_selection_uses_behavior_results": prereg[
                "surface_selection_uses_behavior_results"
            ],
        },
        {
            "internal_results_used_to_select_surfaces": False,
            "surface_selection_uses_behavior_results": False,
        },
    )
    behavior_aggregate = read_json(
        ROLLOUT_ROOT / "behavior" / "summary.json"
    )
    causal_aggregate = read_json(
        ROLLOUT_ROOT / "causal" / "summary.json"
    )
    add_check(
        checks,
        "rollout_behavior_all_models_complete",
        behavior_aggregate["all_models_complete"],
        behavior_aggregate["all_models_complete"],
        True,
    )
    add_check(
        checks,
        "rollout_causal_all_models_complete",
        causal_aggregate["all_models_complete"],
        causal_aggregate["all_models_complete"],
        True,
    )
    for model_name in MODELS:
        protocol = read_json(
            ROLLOUT_ROOT
            / "protocol"
            / model_name
            / "protocol_audit.json"
        )
        add_check(
            checks,
            f"rollout_protocol:{model_name}",
            (
                protocol["case_count"] == 640
                and protocol[
                    "all_anchor_positions_in_unchanged_prompt_prefix"
                ]
            ),
            {
                "case_count": protocol["case_count"],
                "anchor_prefix": protocol[
                    "all_anchor_positions_in_unchanged_prompt_prefix"
                ],
            },
            {"case_count": 640, "anchor_prefix": True},
        )
        behavior = read_json(
            ROLLOUT_ROOT
            / "behavior"
            / model_name
            / "summary.json"
        )
        behavior_rows = line_count(
            ROLLOUT_ROOT
            / "behavior"
            / model_name
            / "rows.jsonl"
        )
        add_check(
            checks,
            f"rollout_behavior_rows:{model_name}",
            (
                behavior["case_count"] == 640
                and behavior_rows == 640
            ),
            {
                "summary": behavior["case_count"],
                "rows": behavior_rows,
            },
            {"summary": 640, "rows": 640},
        )
        causal = read_json(
            ROLLOUT_ROOT
            / "causal"
            / model_name
            / "summary.json"
        )
        for surface, summary in causal["surfaces"].items():
            rows = line_count(
                ROLLOUT_ROOT
                / "causal"
                / model_name
                / surface
                / "rows.jsonl"
            )
            add_check(
                checks,
                f"rollout_causal_rows:{model_name}:{surface}",
                rows == 384,
                rows,
                384,
            )
            noops = {
                split: summary["splits"][split]["conditions"][
                    "target_noop"
                ]["noop_sequence_agreement"]
                for split in ("discovery", "confirmation")
            }
            add_check(
                checks,
                f"rollout_noops:{model_name}:{surface}",
                all(value == 1.0 for value in noops.values()),
                noops,
                {"discovery": 1.0, "confirmation": 1.0},
            )
            add_check(
                checks,
                f"rollout_donors:{model_name}:{surface}",
                all(
                    audit["all_cross_world"]
                    and audit[
                        "all_donor_answers_differ_from_target"
                    ]
                    for audit in summary["donor_audits"].values()
                ),
                summary["donor_audits"],
                "all cross-world and answer-changing",
            )
    add_check(
        checks,
        "rollout_cross_model_fixed_surfaces",
        all(
            causal_aggregate["cross_model_causal_gates"][surface]
            for surface in (
                "bare",
                "short_sentence",
                "two_sentence",
            )
        ),
        {
            surface: causal_aggregate[
                "cross_model_causal_gates"
            ][surface]
            for surface in SURFACES
        },
        {
            "bare": True,
            "short_sentence": True,
            "two_sentence": True,
        },
    )


def script_checks(checks: list[dict[str, Any]]) -> None:
    expected = sorted(
        (ROOT / "tests" / "glm5").glob("phase1003*.py")
    )
    import py_compile

    failures = []
    for path in expected:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            failures.append({
                "path": str(path.relative_to(ROOT)),
                "error": repr(exc),
            })
    add_check(
        checks,
        "all_phase1003_scripts_compile",
        not failures,
        failures,
        [],
    )


def main() -> None:
    checks: list[dict[str, Any]] = []
    parse = parse_artifacts()
    add_check(
        checks,
        "all_json_parses",
        not parse["parse_failures"],
        parse["parse_failures"],
        [],
    )
    add_check(
        checks,
        "all_numeric_values_finite",
        not parse["nonfinite_values"],
        parse["nonfinite_values"],
        [],
    )
    add_check(
        checks,
        "all_artifacts_have_schema",
        not parse["missing_schema"],
        parse["missing_schema"],
        [],
    )
    add_check(
        checks,
        "all_artifacts_phase_1003",
        not parse["phase_drift"],
        parse["phase_drift"],
        [],
    )
    main_protocol_checks(checks)
    main_causal_checks(checks)
    stress_checks(checks)
    precision_checks(checks)
    rollout_checks(checks)
    script_checks(checks)
    failed = [check for check in checks if not check["pass"]]
    payload = {
        "schema_version": "phase1003_result_audit.v1",
        "phase": PHASE,
        "status": "pass" if not failed else "fail",
        "artifact_inventory": parse,
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(failed),
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
    write_json(AUDIT_ROOT / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
