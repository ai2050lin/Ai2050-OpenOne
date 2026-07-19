#!/usr/bin/env python3
"""Read-only Phase 978 development failure postmortem.

This report authenticates the frozen development admission and reconstructs
simple endpoint/censoring counts from the already generated rows.  It never
imports the holdout dataset, never loads model weights, and cannot change the
pre-registered gate decision.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase978_dev_admission_audit as audit  # noqa: E402
import phase978_legal_core as legal  # noqa: E402


PHASE = 978
SCHEMA_VERSION = 1
CONDITIONS = tuple(legal.CONDITIONS)
OUT = ROOT / "tests" / "glm5" / "result" / "phase978_legal_budget_stabilization"
ADMISSION_PATH = OUT / "admission_development.json"
POSTMORTEM_PATH = OUT / "postmortem_development.json"
W_OUT = ROOT / "tests" / "glm5" / "result" / "phase978_wrong_answer_safety"
W_MANIFEST_PATH = W_OUT / "manifest.json"
W_ROWS_PATH = W_OUT / "rows.jsonl"
W_SUMMARY_PATH = W_OUT / "summary.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def without_fields(value: dict[str, Any], *excluded: str) -> dict[str, Any]:
    blocked = set(excluded)
    return {key: item for key, item in value.items() if key not in blocked}


def authenticate_admission() -> dict[str, Any]:
    admission = load_json(ADMISSION_PATH, "Phase978 development admission")
    core = without_fields(admission, "admission_sha256", "audited_at_utc")
    require(admission.get("admission_sha256") == legal.sha256_json(core),
            "development admission self-hash invalid")
    require(admission.get("phase") == PHASE, "wrong admission phase")
    require(admission.get("decision_gate", {}).get("passed") is False,
            "postmortem is only defined for the frozen development NO-GO")
    require(admission.get("holdout_authorized") is False,
            "unexpected holdout authorization")
    require(admission.get("holdout_loaded") is False,
            "admission reports holdout access")
    require(admission.get("mechanism_authorized") is False,
            "admission unexpectedly authorizes mechanisms")
    require(admission.get("auditor_sha256") == audit.sha256_file(Path(audit.__file__)),
            "admission auditor changed after audit")
    return admission


def endpoint_interval(step: int | None) -> str:
    if step is None:
        return ">1536_censored"
    if step <= 256:
        return "<=256"
    if step <= 512:
        return "257-512"
    if step <= 1024:
        return "513-1024"
    require(step <= 1536, f"endpoint exceeds final cap: {step}")
    return "1025-1536"


def fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def condition_postmortem(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 64, "condition denominator is not 64")
    intervals = Counter(endpoint_interval(row.get("first_eos_step")) for row in rows)
    completed = [row for row in rows if bool(row.get("has_eos"))]
    censored = [row for row in rows if not bool(row.get("has_eos"))]

    valid = sum(bool(row.get("valid_mode_eos")) for row in completed)
    mode_valid_semantic_mismatch = sum(
        bool(row.get("mode_valid")) and not bool(row.get("semantic_match"))
        for row in completed
    )
    mode_invalid = sum(not bool(row.get("mode_valid")) for row in completed)
    require(valid + mode_valid_semantic_mismatch + mode_invalid == len(completed),
            "completed endpoint taxonomy is not exhaustive")

    hit512 = intervals["513-1024"] + intervals["1025-1536"] + intervals[">1536_censored"]
    hit1024 = intervals["1025-1536"] + intervals[">1536_censored"]
    recovered_by1024 = intervals["513-1024"]
    recovered_by1536 = intervals["1025-1536"]

    censored_snapshots = [
        {
            "id": str(row["id"]),
            "task": str(row["task"]),
            "mode_valid_at_cap": bool(row.get("mode_valid")),
            "final_region_valid_at_cap": bool(row.get("final_region_valid")),
            "semantic_match_at_cap": bool(row.get("semantic_match")),
            "think_structure_status_at_cap": str(row.get("think_structure_status")),
        }
        for row in sorted(censored, key=lambda value: str(value["id"]))
    ]
    return {
        "n": len(rows),
        "endpoint_intervals": {
            key: intervals[key] for key in
            ("<=256", "257-512", "513-1024", "1025-1536", ">1536_censored")
        },
        "staged_recovery": {
            "selected_after_hit512_n": hit512,
            "actual_eos_513_1024_n": recovered_by1024,
            "recovery_among_hit512_rate": fraction(recovered_by1024, hit512),
            "selected_after_hit1024_n": hit1024,
            "actual_eos_1025_1536_n": recovered_by1536,
            "recovery_among_hit1024_rate": fraction(recovered_by1536, hit1024),
            "still_censored_at_1536_n": intervals[">1536_censored"],
        },
        "completed_actual_eos_taxonomy": {
            "n": len(completed),
            "valid_mode_eos_n": valid,
            "mode_valid_but_semantic_mismatch_n": mode_valid_semantic_mismatch,
            "mode_invalid_n": mode_invalid,
            "mutually_exclusive_and_exhaustive": True,
        },
        "censoring": {
            "n": len(censored),
            "by_task": dict(sorted(Counter(str(row["task"]) for row in censored).items())),
            "cap_snapshots": censored_snapshots,
            "interpretation_boundary": (
                "These rows have no sampled EOS by 1536. Mode/semantic fields describe only "
                "the truncated cap snapshot and are not endpoint failures. Right censoring "
                "applies to endpoint time only."
            ),
        },
    }


def paired_thinking_description(
    final_rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    hard_censored = {
        item_id for (item_id, condition), row in final_rows.items()
        if condition == "hard_thinking" and not row["has_eos"]
    }
    soft_censored = {
        item_id for (item_id, condition), row in final_rows.items()
        if condition == "soft_thinking" and not row["has_eos"]
    }
    both_completed = sorted(
        item_id for item_id in {key[0] for key in final_rows}
        if final_rows[(item_id, "hard_thinking")]["has_eos"]
        and final_rows[(item_id, "soft_thinking")]["has_eos"]
    )
    timing = Counter()
    for item_id in both_completed:
        hard_step = int(final_rows[(item_id, "hard_thinking")]["first_eos_step"])
        soft_step = int(final_rows[(item_id, "soft_thinking")]["first_eos_step"])
        if hard_step < soft_step:
            timing["hard_earlier"] += 1
        elif soft_step < hard_step:
            timing["soft_earlier"] += 1
        else:
            timing["equal"] += 1
    return {
        "both_censored_n": len(hard_censored & soft_censored),
        "hard_only_censored_n": len(hard_censored - soft_censored),
        "soft_only_censored_n": len(soft_censored - hard_censored),
        "neither_censored_n": 64 - len(hard_censored | soft_censored),
        "both_completed_actual_eos_n": len(both_completed),
        "endpoint_order_among_both_completed": {
            key: timing[key] for key in ("hard_earlier", "soft_earlier", "equal")
        },
        "interpretation_boundary": (
            "This is a paired descriptive count only. The official hard/soft conditions use "
            "different complete decoding configurations, so it does not isolate a soft-switch "
            "causal effect."
        ),
    }


def authenticate_wrong_answer_auxiliary() -> dict[str, Any]:
    manifest = load_json(W_MANIFEST_PATH, "W/WP manifest")
    manifest_core = without_fields(manifest, "manifest_sha256", "created_at_utc")
    require(manifest.get("manifest_sha256") == legal.sha256_json(manifest_core),
            "W/WP manifest self-hash invalid")
    summary = load_json(W_SUMMARY_PATH, "W/WP summary")
    require(summary.get("manifest_sha256") == manifest["manifest_sha256"],
            "W/WP summary manifest mismatch")
    require(summary.get("complete") is True and summary.get("completed_rows") == 256,
            "W/WP summary is incomplete")
    require(summary.get("holdout_loaded") is False,
            "W/WP summary reports holdout access")
    require(summary.get("decision_status") == "AUXILIARY_COMPLETE_NO_MAIN_GATE",
            "W/WP contract changed")

    payload = W_ROWS_PATH.read_bytes()
    require(payload.endswith(b"\n"), "W/WP rows lack a final newline")
    keys: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid W/WP row {line_number}") from exc
        require(isinstance(row, dict), f"W/WP row {line_number} is not an object")
        claimed = row.get("row_sha256")
        require(claimed == legal.sha256_json(without_fields(row, "row_sha256")),
                f"W/WP row self-hash mismatch at line {line_number}")
        require(row.get("manifest_sha256") == manifest["manifest_sha256"],
                f"W/WP row manifest mismatch at line {line_number}")
        key = (str(row.get("id")), str(row.get("state")))
        require(key not in keys, f"duplicate W/WP row key {key}")
        keys.add(key)
    require(len(keys) == 256, "W/WP row denominator is not 256")

    state_fields = (
        "n", "eos_top1_n", "gap_negative_n", "mean_gap",
        "mean_max_eos_probability", "mean_eos_rank",
    )
    by_state = {
        state: {key: summary["by_state"][state][key] for key in state_fields}
        for state in ("C", "P", "W", "WP")
    }
    dataset = manifest.get("dataset", {})
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": audit.sha256_file(W_MANIFEST_PATH),
        "rows_file_sha256": audit.sha256_file(W_ROWS_PATH),
        "summary_file_sha256": audit.sha256_file(W_SUMMARY_PATH),
        "authenticated_rows": len(keys),
        "by_state": by_state,
        "paired": summary["paired"],
        "canonical_answer_matcher_false_n": dataset.get(
            "canonical_answer_matcher_false_n"),
        "canonical_answer_matcher_false_ids": dataset.get(
            "canonical_answer_matcher_false_ids"),
        "contract": summary["contract"],
        "interpretation_boundary": summary["conclusion_boundary"],
    }


def install_or_validate(report: dict[str, Any]) -> None:
    if POSTMORTEM_PATH.exists():
        prior = load_json(POSTMORTEM_PATH, "existing Phase978 postmortem")
        prior_core = without_fields(prior, "postmortem_sha256", "generated_at_utc")
        require(prior.get("postmortem_sha256") == legal.sha256_json(prior_core),
                "existing postmortem self-hash invalid")
        require(prior.get("postmortem_sha256") == report["postmortem_sha256"],
                "existing postmortem differs; refusing to overwrite")
        return
    legal.atomic_write_json(POSTMORTEM_PATH, report)


def run(write: bool) -> dict[str, Any]:
    audit.assert_no_holdout_import()
    admission = authenticate_admission()
    protocol = audit.authenticate_protocol()
    tok = audit.load_tokenizer()
    try:
        source_manifest, source, selected, items, source_audit = audit.authenticate_source(tok)
        dev_manifest, extended, extension_audit = audit.authenticate_extensions(
            protocol, tok, source_manifest, source, selected, items)
        checkpoints, intervals = audit.build_checkpoints(source, extended, items)
    finally:
        del tok
        gc.collect()
    audit.assert_no_holdout_import()

    require(checkpoints == admission["checkpoints"],
            "reconstructed checkpoints differ from admission")
    require(intervals == admission["eos_time_intervals"],
            "reconstructed endpoint intervals differ from admission")
    require(dev_manifest["manifest_sha256"] == admission["development_manifest_sha256"],
            "development manifest mismatch")
    require(extension_audit["rows_sha256"] == admission["development_rows_sha256"],
            "development rows mismatch")

    final_rows: dict[tuple[str, str], dict[str, Any]] = {}
    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [audit.resolve_endpoint(source, extended, item["id"], condition, 1536)
                for item in items]
        for row in rows:
            final_rows[(str(row["id"]), condition)] = row
        by_condition[condition] = condition_postmortem(rows)

    matcher_misses = sorted(
        str(item["id"]) for item in items
        if not legal.semantic_match(item["alias_groups"], item["answer"], item["exact"])
    )
    gate = admission["decision_gate"]
    failed_conditions = [
        condition for condition, block in gate["condition_checks"].items()
        if not block["passed"]
    ]
    require(failed_conditions == ["hard_thinking"],
            "unexpected frozen gate failure set")

    wrong_answer = authenticate_wrong_answer_auxiliary()
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "development_budget_failure_postmortem",
        "split": "development_design_data_only",
        "role": "read_only_exploratory_postmortem_cannot_change_gate",
        "postmortem_script_sha256": audit.sha256_file(Path(__file__)),
        "protocol_sha256": protocol["protocol_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "development_manifest_sha256": dev_manifest["manifest_sha256"],
        "development_rows_sha256": admission["development_rows_sha256"],
        "source_rows_sha256": source_audit["source_file_sha256"][
            "rows_development.jsonl"],
        "input_authentication": {
            "all_source_rows_recomputed": source_audit[
                "all_408_source_rows_recomputed"],
            "all_source_256_replays_exact": source_audit["all_source_256_replays_exact"],
            "all_extension_rows_recomputed": extension_audit["all_extension_rows_recomputed"],
            "all_extension_replays_exact": extension_audit["all_extension_replays_exact"],
            "holdout_module_imported": False,
            "model_weights_loaded": False,
        },
        "frozen_decision": {
            "passed": False,
            "failed_conditions": failed_conditions,
            "holdout_authorized": False,
            "holdout_loaded": False,
            "mechanism_authorized": False,
            "decision_unchanged": True,
        },
        "by_condition": by_condition,
        "paired_thinking_descriptive": paired_thinking_description(final_rows),
        "primary_matcher_self_consistency": {
            "canonical_answer_matcher_false_n": len(matcher_misses),
            "canonical_answer_matcher_false_ids": matcher_misses,
            "interpretation": (
                "External frozen labels remain authoritative. These misses expose an evaluator "
                "hard limit and cannot be repaired post hoc for the Phase978 gate."
            ),
        },
        "wrong_answer_teacher_forced_auxiliary": wrong_answer,
        "conclusion_boundaries": [
            "Actual sampled EOS, not EOS top-1 or logit gap, defines the natural endpoint gate.",
            "Right censoring concerns endpoint time only; a truncated cap snapshot is not a semantic or mode endpoint failure.",
            "Budget is an external protocol and observation variable, not a located internal causal node.",
            "The official hard/soft configurations differ in decoding details; their contrast is not an isolated thinking-control intervention.",
            "Development data are now design data. No post-hoc higher cap may revise the frozen Phase978 decision.",
        ],
        "next_stage_constraint": (
            "A later phase must use a new preregistration and new non-holdout design data to "
            "study the >1536 tail and evaluator separation. The sealed 128-item holdout remains "
            "closed until a fresh independent confirmatory gate passes."
        ),
        "holdout_loaded": False,
        "model_weights_loaded": False,
    }
    report = {
        **core,
        "postmortem_sha256": legal.sha256_json(core),
        "generated_at_utc": legal.utc_now(),
    }
    if write:
        install_or_validate(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Atomically install the self-hashed postmortem artifact.")
    args = parser.parse_args()
    result = run(bool(args.write))
    print(json.dumps({
        "postmortem_sha256": result["postmortem_sha256"],
        "decision_passed": result["frozen_decision"]["passed"],
        "failed_conditions": result["frozen_decision"]["failed_conditions"],
        "holdout_loaded": result["holdout_loaded"],
        "model_weights_loaded": result["model_weights_loaded"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
