#!/usr/bin/env python3
"""Detailed raw-row morphology for the sealed Phase 983 NO-GO.

The report separates strict terminal validity from visible answer content and
records literal thinking markers.  These post-hoc observations cannot modify
the preregistered decision or establish a latent reasoning state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""

GLM5 = Path(__file__).resolve().parent
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))
import phase983_cross_model_core as core  # noqa: E402


OUTPUT_PATH = core.OUT / "failure_morphology_detail.json"
CHECKPOINT = "2048"
_STANDALONE_FINAL = re.compile(r"(?:\A|\n)FINAL: ([AB])")


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_line(line: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid {label}") from exc
    core.require(isinstance(value, dict), f"{label} must be an object")
    return value


def verify_audit(audit: dict[str, Any]) -> None:
    core.require(audit.get("phase") == core.PHASE, "audit phase changed")
    core.require(audit.get("experiment") == core.EXPERIMENT,
                 "audit experiment changed")
    core.require(audit.get("integrity_decision") == "GO",
                 "audit integrity is not GO")
    core.require(audit.get("scientific_decision") == "NO-GO",
                 "detail morphology requires a sealed NO-GO")
    expected = core.sha256_json(core.without_fields(
        audit, "audit_sha256", "created_at_utc"))
    core.require(audit.get("audit_sha256") == expected,
                 "audit self-hash invalid")


def load_rows(model: str, audit: dict[str, Any]) -> list[dict[str, Any]]:
    path = core.OUT / model / "rows.jsonl"
    source = audit["source_hashes"][model]
    core.require(core.sha256_file(path) == source["rows_file_sha256"],
                 f"{model} row file hash changed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(handle, start=1):
            core.require(raw.endswith("\n"),
                         f"{model} row {line_number} lacks terminal newline")
            row = strict_json_line(raw, f"{model} row {line_number}")
            core.require(row.get("row_sha256") == core.sha256_json(
                core.without_fields(row, "row_sha256")),
                f"{model} row {line_number} self-hash invalid")
            core.require(row.get("model_key") == model,
                         f"{model} row model mismatch")
            core.require(row.get("phase") == core.PHASE
                         and row.get("experiment") == core.EXPERIMENT,
                         f"{model} row lineage mismatch")
            stream = row.get("stream")
            arm = row.get("arm")
            item_id = row.get("id")
            core.require(stream in core.STREAMS and arm in core.ARMS
                         and isinstance(item_id, str),
                         f"{model} row key invalid")
            key = (item_id, stream, arm)
            core.require(key not in seen, f"{model} duplicate row key")
            seen.add(key)
            checkpoint = row.get("checkpoints", {}).get(CHECKPOINT)
            core.require(isinstance(checkpoint, dict),
                         f"{model} decision checkpoint missing")
            core.require(checkpoint.get("terminal_state")
                         == row.get("decision_terminal_state"),
                         f"{model} decision state mismatch")
            rows.append(row)
    core.require(len(rows) == source["row_count"]
                 == core.EXPECTED_ROWS_PER_MODEL,
                 f"{model} row count changed")
    return rows


def length_bin(n_tokens: int) -> str:
    core.require(isinstance(n_tokens, int) and not isinstance(n_tokens, bool)
                 and 0 < n_tokens <= core.DECISION_CHECKPOINT,
                 "invalid generated-token count")
    if n_tokens <= 256:
        return "0001-0256"
    if n_tokens <= 512:
        return "0257-0512"
    if n_tokens <= 1024:
        return "0513-1024"
    if n_tokens <= 1536:
        return "1025-1536"
    return "1537-2048"


def invalid_contract_shape(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("protocol_subtype") == "EOS_WITH_UNEXPECTED_SPECIAL_TOKEN":
        return "unexpected_non_EOS_special_token"
    marker_count = checkpoint.get("marker_like_count")
    exact = checkpoint.get("exact_terminal_marker")
    parsed = checkpoint.get("parsed_label")
    core.require(isinstance(marker_count, int) and marker_count >= 0,
                 "marker-like count invalid")
    if marker_count == 0:
        return "no_FINAL_like_marker"
    if exact is True and marker_count > 1:
        return "exact_terminal_FINAL_plus_additional_marker_like_text"
    if exact is False and parsed is None:
        return "FINAL_like_text_but_not_unique_exact_terminal_contract"
    return "other_invalid_FINAL_contract"


def last_final_label(text: str) -> str | None:
    matches = _STANDALONE_FINAL.findall(str(text).strip())
    return matches[-1] if matches else None


def summarize_arm(
    rows: list[dict[str, Any]], native_prefill: str,
) -> dict[str, Any]:
    core.require(len(rows) == core.ITEM_COUNT, "arm cell is not N=256")
    states: Counter[str] = Counter()
    censor_subtypes: Counter[str] = Counter()
    protocol_subtypes: Counter[str] = Counter()
    invalid_shapes: Counter[str] = Counter()
    marker_count_histogram: Counter[str] = Counter()
    token_bins: Counter[str] = Counter()
    generated_open = 0
    generated_close = 0
    effective_open = 0
    effective_closed = 0
    exact_terminal_marker = 0
    exact_terminal_correct = 0
    last_final_present = 0
    last_final_correct = 0
    native_opens_think = "<think>" in native_prefill

    for row in rows:
        checkpoint = row["checkpoints"][CHECKPOINT]
        state = checkpoint["terminal_state"]
        states[state] += 1
        if state == "C":
            censor_subtypes[str(checkpoint["censor_subtype"])] += 1
        if state == "I_protocol":
            protocol_subtypes[str(checkpoint["protocol_subtype"])] += 1
            invalid_shapes[invalid_contract_shape(checkpoint)] += 1
        text = str(checkpoint["plain_text"])
        has_open = "<think>" in text
        has_close = "</think>" in text
        generated_open += has_open
        generated_close += has_close
        effective_open += native_opens_think or has_open
        effective_closed += (native_opens_think or has_open) and has_close
        exact = checkpoint["exact_terminal_marker"] is True
        exact_terminal_marker += exact
        exact_terminal_correct += (
            exact and checkpoint["parsed_label"] == row["gold_label"])
        label = last_final_label(text)
        last_final_present += label is not None
        last_final_correct += label is not None and label == row["gold_label"]
        marker_count_histogram[str(checkpoint["marker_like_count"])] += 1
        token_bins[length_bin(checkpoint["n_tokens"])] += 1

    for state in core.TERMINAL_STATES:
        states.setdefault(state, 0)
    core.require(sum(states.values()) == core.ITEM_COUNT,
                 "terminal states do not sum to 256")
    return {
        "N": core.ITEM_COUNT,
        "terminal_states": {state: states[state]
                            for state in core.TERMINAL_STATES},
        "C_subtypes": dict(sorted(censor_subtypes.items())),
        "I_protocol_subtypes": dict(sorted(protocol_subtypes.items())),
        "invalid_FINAL_contract_shapes": dict(sorted(invalid_shapes.items())),
        "strict_V_count": states["V"],
        "exact_terminal_FINAL_present": exact_terminal_marker,
        "exact_terminal_FINAL_correct_ignoring_EOS_or_hidden_special":
            exact_terminal_correct,
        "last_standalone_FINAL_present_exploratory": last_final_present,
        "last_standalone_FINAL_correct_exploratory": last_final_correct,
        "strict_V_minus_exact_terminal_correct":
            states["V"] - exact_terminal_correct,
        "marker_like_count_histogram":
            dict(sorted(marker_count_histogram.items(), key=lambda item: int(item[0]))),
        "generated_literal_think_open_count": generated_open,
        "generated_literal_think_close_count": generated_close,
        "native_prefill_literal_think_open": native_opens_think,
        "effective_prefill_or_generated_think_open_count": effective_open,
        "effective_think_open_and_generated_close_count": effective_closed,
        "generated_token_count_bins": {
            key: token_bins.get(key, 0)
            for key in ("0001-0256", "0257-0512", "0513-1024",
                        "1025-1536", "1537-2048")
        },
    }


def option_swap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_semantic: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_semantic[row["semantic_id"]].append(row)
    core.require(len(by_semantic) == core.SEMANTIC_INSTANCE_COUNT,
                 "semantic twin count changed")
    counts: Counter[str] = Counter()
    for twins in by_semantic.values():
        core.require(len(twins) == 2, "semantic twin is not a pair")
        core.require({row["swap_variant"] for row in twins}
                     == set(core.SWAP_SIDES), "swap sides changed")
        core.require({row["gold_label"] for row in twins}
                     == set(core.LABELS), "twin gold labels changed")
        states = [row["decision_terminal_state"] for row in twins]
        valid = [state == "V" for state in states]
        counts["same_terminal_state"] += states[0] == states[1]
        counts["both_V"] += all(valid)
        counts["exactly_one_V"] += sum(valid) == 1
        counts["neither_V"] += not any(valid)
    return {
        "semantic_pair_count": core.SEMANTIC_INSTANCE_COUNT,
        "same_terminal_state": counts["same_terminal_state"],
        "both_V": counts["both_V"],
        "exactly_one_V": counts["exactly_one_V"],
        "neither_V": counts["neither_V"],
        "twins_are_not_independent_samples": True,
    }


def validate_against_audit(
    model: str, stream: int, arm: str, summary: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    expected = audit["gate_decision"]["model_results"][model][
        "stream_results"][f"stream_{stream}"]["overall_accounting"][
            f"external_bundle_{arm}_counts"]
    core.require(summary["terminal_states"] == expected,
                 f"{model}/stream{stream}/{arm} disagrees with audit")


def build_payload(audit: dict[str, Any]) -> dict[str, Any]:
    verify_audit(audit)
    model_reports: dict[str, Any] = {}
    for model in core.MODEL_ORDER:
        rows = load_rows(model, audit)
        manifest = core.load_json(core.OUT / model / "manifest.json",
                                  f"{model} manifest")
        prefill = manifest["loaded_model_identity"][
            "native_generation_prefill"]["assistant_prefill_text"]
        core.require(isinstance(prefill, str), f"{model} prefill invalid")
        stream_reports: dict[str, Any] = {}
        for stream in core.STREAMS:
            arm_reports: dict[str, Any] = {}
            for arm in core.ARMS:
                cell = [row for row in rows
                        if row["stream"] == stream and row["arm"] == arm]
                summary = summarize_arm(cell, prefill)
                validate_against_audit(model, stream, arm, summary, audit)
                arm_reports[arm] = {
                    **summary,
                    "option_swap_twin_summary": option_swap_summary(cell),
                }
            stream_reports[f"stream_{stream}"] = arm_reports
        model_reports[model] = {
            "native_assistant_prefill_text": prefill,
            "native_prefill_opens_literal_think": "<think>" in prefill,
            "streams": stream_reports,
        }

    return {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "analysis_name": "sealed_NO_GO_raw_row_failure_morphology_detail",
        "analysis_scope": "CPU-only post-hoc descriptive raw-row audit",
        "source_audit_sha256": audit["audit_sha256"],
        "source_audit_file_sha256": core.sha256_file(core.COMBINED_AUDIT_PATH),
        "source_scientific_decision": audit["scientific_decision"],
        "primary_decision_unchanged": True,
        "can_set_primary_decision": False,
        "can_authorize_holdout": False,
        "can_authorize_mechanism": False,
        "models_pooled": False,
        "models": model_reports,
        "measurement_warnings": [
            "A/B are assigned natural-language instructions, not verified latent modes.",
            "Literal <think> markers measure visible serialization, not cognition.",
            "GLM4 has no comparable literal think tag, so adherence is unscored.",
            "Exact/last FINAL content counts are exploratory and cannot replace strict V.",
            "The global case-insensitive FINAL-like count intentionally makes the preregistered parser stricter than a last-label parser.",
            "Option-swapped twins and three streams are paired robustness views, not independent samples.",
        ],
        "key_observation": (
            "Qwen3 generated a literal <think> opener in all A and B rows, while "
            "DS7B's native assistant prefill opened <think> for both arms.  The "
            "intervention therefore cannot be interpreted as verified no-think "
            "versus think behavior."
        ),
    }


def verify_existing(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    core.require(set(existing) == set(expected) | {
        "detail_sha256", "created_at_utc"}, "detail schema changed")
    core.require(existing.get("detail_sha256") == core.sha256_json(expected),
                 "detail self-hash invalid")
    core.require(core.without_fields(
        existing, "detail_sha256", "created_at_utc") == expected,
        "detail payload changed")
    core.require(isinstance(existing.get("created_at_utc"), str)
                 and existing["created_at_utc"], "detail timestamp missing")


def static_self_test() -> dict[str, Any]:
    tests = {
        "length_256": length_bin(256) == "0001-0256",
        "length_257": length_bin(257) == "0257-0512",
        "length_2048": length_bin(2048) == "1537-2048",
        "last_FINAL": last_final_label("x\nFINAL: A") == "A",
        "nonstandalone_ignored": last_final_label("x FINAL: B") is None,
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    core.require(all(tests.values()), "detail self-test failed")
    return {"tests": tests, "files_written": False}


def run(write: bool, verify: bool) -> dict[str, Any]:
    if not write and not verify:
        return static_self_test()
    audit = core.load_json(core.COMBINED_AUDIT_PATH, "Phase983 combined audit")
    expected = build_payload(audit)
    if OUTPUT_PATH.exists():
        existing = core.load_json(OUTPUT_PATH, "Phase983 morphology detail")
        verify_existing(existing, expected)
        return {
            "existing": True,
            "files_written": False,
            "detail_sha256": existing["detail_sha256"],
            "detail_file_sha256": core.sha256_file(OUTPUT_PATH),
        }
    core.require(write, "detail report absent; --verify cannot create it")
    document = {
        **expected,
        "detail_sha256": core.sha256_json(expected),
        "created_at_utc": core.utc_now(),
    }
    core.atomic_write_json(OUTPUT_PATH, document)
    installed = core.load_json(OUTPUT_PATH, "installed Phase983 morphology detail")
    verify_existing(installed, expected)
    core.require(installed == document, "installed detail serialization changed")
    return {
        "existing": False,
        "files_written": True,
        "detail_sha256": installed["detail_sha256"],
        "detail_file_sha256": core.sha256_file(OUTPUT_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.write, args.verify), ensure_ascii=False,
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
