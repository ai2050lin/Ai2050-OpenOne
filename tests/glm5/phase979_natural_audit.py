#!/usr/bin/env python3
"""Independent CPU audit for Phase 979 natural 4x2x2 trajectories."""
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

import phase979_boundary_core as core  # noqa: E402
import phase979_diagnostic_dataset as dataset  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402


OUT = ROOT / "tests" / "glm5" / "result" / "phase979_three_boundary_factorial"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
MANIFEST_PATH = OUT / "manifest_natural.json"
ROWS_PATH = OUT / "rows_natural.jsonl"
AUDIT_PATH = OUT / "audit_natural.json"
EXPECTED_ROWS = 128 * 4 * 2 * 2
TASKS = tuple(dataset.TASKS)
SCREEN_THRESHOLDS = {
    "denominator_per_cell_stream": 128,
    "valid_stop_net_improvement_min": 13,
    "censored_net_reduction_min": 13,
    "eos_invalid_increase_max": 6,
    "tasks_with_valid_stop_improvement_min": 6,
    "must_pass_both_frozen_streams": True,
    "interpretation": (
        "A passing contrast is only a candidate for a fresh independent "
        "confirmation; it cannot open the old holdout or authorize a mechanism scan."
    ),
}


def assert_no_holdout_import() -> None:
    forbidden = [name for name in sys.modules if "holdout" in name.lower()]
    core.require(not forbidden, f"holdout-like module imported: {forbidden}")


def authenticate_protocol() -> dict[str, Any]:
    protocol = core.load_json(PROTOCOL_PATH, "Phase979 protocol")
    payload = core.without_fields(protocol, "protocol_sha256", "created_at_utc")
    core.require(protocol.get("protocol_sha256") == core.sha256_json(payload),
                 "Phase979 protocol self-hash invalid")
    core.require(protocol.get("phase") == core.PHASE, "wrong protocol phase")
    core.require(protocol.get("checkpoints") == list(core.CHECKPOINTS),
                 "checkpoint contract changed")
    core.require(protocol.get("max_new_tokens") == core.MAX_NEW_TOKENS,
                 "maximum budget changed")
    core.require(protocol.get("expected_natural_rows") == EXPECTED_ROWS,
                 "natural row denominator changed")
    core.require(protocol.get("controls") == core.CONTROL_POLICIES,
                 "control factorial changed")
    core.require(protocol.get("decoding_policies") == core.DECODING_POLICIES,
                 "decoding factorial changed")
    core.require(protocol.get("replicates") == list(core.REPLICATES),
                 "replicate contract changed")
    natural_contract = protocol.get("natural_contract", {})
    core.require(
        natural_contract.get("new_diagnostic_cap_not_phase978_revision") is True
        and natural_contract.get("single_rollout_per_row") is True
        and natural_contract.get("checkpoint_snapshots_are_prefixes_not_reruns") is True
        and natural_contract.get(
            "two_streams_are_seed_dependence_screen_not_variance_estimate") is True
        and natural_contract.get(
            "cap_categories_are_right_censored_snapshots_not_terminal_failures") is True
        and natural_contract.get("screen_thresholds") == SCREEN_THRESHOLDS,
        "natural diagnostic/decision boundary changed",
    )
    scripts = protocol.get("phase979_script_hashes", {})
    core.require(isinstance(scripts, dict), "protocol lacks script seal")
    for label, entry in scripts.items():
        core.require(isinstance(entry, dict) and "path" in entry and "sha256" in entry,
                     f"invalid script seal entry {label}")
        path = ROOT / str(entry["path"])
        core.require(path.is_file() and core.sha256_file(path) == entry["sha256"],
                     f"sealed script changed: {label}")
    commitments = protocol.get("phase978_commitments", {})
    core.require(commitments.get("development_gate_passed") is False
                 and commitments.get("holdout_authorized") is False
                 and commitments.get("holdout_loaded") is False,
                 "Phase978 NO-GO boundary missing")
    core.require(protocol.get("holdout_loaded") is False
                 and protocol.get("mechanism_authorized") is False,
                 "protocol crosses forbidden boundary")
    return protocol


def authenticate_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = core.load_json(MANIFEST_PATH, "Phase979 natural manifest")
    payload = core.without_fields(manifest, "manifest_sha256", "created_at_utc")
    core.require(manifest.get("manifest_sha256") == core.sha256_json(payload),
                 "natural manifest self-hash invalid")
    core.require(manifest.get("phase") == core.PHASE, "wrong manifest phase")
    core.require(manifest.get("protocol_sha256") == protocol["protocol_sha256"],
                 "manifest protocol mismatch")
    core.require(manifest.get("expected_rows") == EXPECTED_ROWS,
                 "manifest denominator mismatch")
    expected_identity = dataset.audit_items()["identity"]
    core.require(manifest.get("dataset_identity") == expected_identity,
                 "manifest dataset identity mismatch")
    core.require(manifest.get("holdout_loaded") is False,
                 "manifest reports holdout access")
    return manifest


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def read_rows(manifest_sha256: str) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    core.require(ROWS_PATH.is_file(), f"missing natural rows: {ROWS_PATH}")
    payload = ROWS_PATH.read_bytes()
    core.require(payload.endswith(b"\n"), "natural JSONL lacks final newline")
    records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed natural row {line_number}") from exc
        core.require(isinstance(row, dict), f"natural row {line_number} not object")
        claimed = row.get("row_sha256")
        core.require(claimed == core.sha256_json(core.without_fields(row, "row_sha256")),
                     f"natural row self-hash mismatch {line_number}")
        core.require(row.get("manifest_sha256") == manifest_sha256,
                     f"natural row manifest mismatch {line_number}")
        key = core.natural_key(row)
        core.require(key not in records, f"duplicate natural key {key}")
        records[key] = row
    core.require(len(records) == EXPECTED_ROWS,
                 f"expected {EXPECTED_ROWS} natural rows, got {len(records)}")
    return records


def expected_keys(items: list[dict[str, Any]]) -> set[tuple[str, str, str, int]]:
    return {
        (str(item["id"]), control, decoding, replicate)
        for item in items
        for control in core.CONTROL_POLICIES
        for decoding in core.DECODING_POLICIES
        for replicate in core.REPLICATES
    }


def validate_rows(
    protocol: dict[str, Any], manifest: dict[str, Any],
    records: dict[tuple[str, str, str, int], dict[str, Any]],
    tok, items: list[dict[str, Any]],
) -> dict[str, Any]:
    item_by_id = {str(item["id"]): item for item in items}
    core.require(set(records) == expected_keys(items), "natural key set mismatch")
    eos_ids = [int(value) for value in manifest["eos_token_ids"]]
    think_open = core.single_token_id(tok, "<think>")
    think_close = core.single_token_id(tok, "</think>")
    core.require(manifest.get("special_token_ids") == {
        "think_open": think_open, "think_close": think_close,
    }, "natural special token IDs changed")
    label_ids = {
        label: list(tok(label, add_special_tokens=False,
                        return_attention_mask=False).input_ids)
        for label in ("A", "B")
    }
    core.require(label_ids == {"A": [32], "B": [33]},
                 f"primary answer tokenization changed: {label_ids}")

    for key, row in records.items():
        item = item_by_id[key[0]]
        control, decoding, replicate = key[1], key[2], key[3]
        user_prompt, rendered, input_ids = core.render_prefix(tok, item, control)
        generated = row.get("generated_ids")
        core.require(isinstance(generated, list) and generated
                     and all(isinstance(value, int) and not isinstance(value, bool)
                             for value in generated), f"invalid generated IDs {key}")
        trimmed = core.trim_at_first_eos([int(value) for value in generated], eos_ids)
        core.require(trimmed == generated, f"untrimmed tokens after EOS {key}")
        eos_positions = core.positions_of(generated, set(eos_ids))
        core.require(
            (len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
            or (not eos_positions and len(generated) == core.MAX_NEW_TOKENS),
            f"invalid natural termination {key}",
        )
        recomputed = core.analyze_checkpoints(
            tok, item, control, generated, eos_ids, think_open, think_close,
        )
        expected_sampling = core.DECODING_POLICIES[decoding]
        expected_seed = core.stable_seed(item["id"], control, decoding, replicate)
        core.require(
            row.get("schema_version") == core.SCHEMA_VERSION
            and row.get("phase") == core.PHASE
            and row.get("protocol_sha256") == protocol["protocol_sha256"]
            and row.get("id") == item["id"]
            and row.get("task") == item["task"]
            and row.get("prompt") == item["prompt"]
            and row.get("answer") == item["answer"]
            and row.get("control_policy") == control
            and row.get("decoding_policy") == decoding
            and row.get("sampling") == expected_sampling
            and row.get("replicate") == replicate
            and row.get("seed") == expected_seed
            and row.get("effective_user_prompt") == user_prompt
            and row.get("rendered_prefix_sha256") == core.sha256_json(rendered)
            and row.get("input_ids") == input_ids
            and row.get("generated_ids") == generated
            and row.get("checkpoints") == recomputed
            and row.get("official_cell") == ((control, decoding) in core.OFFICIAL_CELLS)
            and row.get("max_new_tokens") == core.MAX_NEW_TOKENS
            and row.get("holdout_loaded") is False
            and row.get("mechanism_authorized") is False,
            f"natural metadata/derived mismatch {key}",
        )
    return {
        "all_expected_keys_exact": True,
        "all_rows_self_hashed": True,
        "all_prefixes_recomputed": True,
        "all_seeds_recomputed": True,
        "all_terminations_valid": True,
        "all_checkpoint_states_recomputed": True,
        "rows_file_sha256": core.sha256_file(ROWS_PATH),
    }


def checkpoint_metrics(rows: list[dict[str, Any]], checkpoint: int) -> dict[str, Any]:
    values = [row["checkpoints"][str(checkpoint)] for row in rows]
    n = len(values)
    states = Counter(str(value["terminal_state"]) for value in values)
    return {
        "n": n,
        "eos_observed_n": sum(bool(value["has_eos"]) for value in values),
        "close_observed_n": sum(bool(value["close_observed"]) for value in values),
        "answer_observed_n": sum(bool(value["answer_observed"]) for value in values),
        "mode_valid_snapshot_n": sum(bool(value["mode_valid"]) for value in values),
        "semantic_match_snapshot_n": sum(
            bool(value["semantic_match_at_snapshot"]) for value in values),
        "valid_stop_n": states["VALID_STOP"],
        "censored_n": sum(not bool(value["has_eos"]) for value in values),
        "eos_invalid_n": states["EOS_INVALID_MODE"] + states["EOS_INVALID_SEMANTIC"],
        "terminal_states": {state: states[state] for state in core.TERMINAL_STATES},
    }


def build_summaries(records: dict[tuple[str, str, str, int], dict[str, Any]]) -> dict[str, Any]:
    rows = list(records.values())
    output: dict[str, Any] = {}
    for checkpoint in core.CHECKPOINTS:
        groups: dict[str, Any] = {}
        for control in core.CONTROL_POLICIES:
            groups[control] = {}
            for decoding in core.DECODING_POLICIES:
                groups[control][decoding] = {}
                for replicate in core.REPLICATES:
                    selected = [row for row in rows
                                if row["control_policy"] == control
                                and row["decoding_policy"] == decoding
                                and row["replicate"] == replicate]
                    core.require(len(selected) == 128, "group denominator is not 128")
                    by_task = {
                        task: checkpoint_metrics(
                            [row for row in selected if row["task"] == task], checkpoint)
                        for task in TASKS
                    }
                    groups[control][decoding][str(replicate)] = {
                        "overall": checkpoint_metrics(selected, checkpoint),
                        "by_task": by_task,
                    }
        output[str(checkpoint)] = groups
    return output


def effect_screen(
    label: str, summaries2048: dict[str, Any],
    option_a: tuple[str, str], option_b: tuple[str, str],
) -> dict[str, Any]:
    replicate_deltas: dict[str, Any] = {}
    valid_deltas: list[int] = []
    for replicate in core.REPLICATES:
        a = summaries2048[option_a[0]][option_a[1]][str(replicate)]
        b = summaries2048[option_b[0]][option_b[1]][str(replicate)]
        valid_delta = int(b["overall"]["valid_stop_n"] - a["overall"]["valid_stop_n"])
        valid_deltas.append(valid_delta)
        replicate_deltas[str(replicate)] = {
            "option_b_minus_a_valid_stop_n": valid_delta,
            "option_b_minus_a_censored_n": int(
                b["overall"]["censored_n"] - a["overall"]["censored_n"]),
            "option_b_minus_a_eos_invalid_n": int(
                b["overall"]["eos_invalid_n"] - a["overall"]["eos_invalid_n"]),
            "by_task_valid_stop_delta": {
                task: int(b["by_task"][task]["valid_stop_n"]
                          - a["by_task"][task]["valid_stop_n"])
                for task in TASKS
            },
        }

    if all(delta >= 13 for delta in valid_deltas):
        candidate, baseline, sign = option_b, option_a, 1
    elif all(delta <= -13 for delta in valid_deltas):
        candidate, baseline, sign = option_a, option_b, -1
    else:
        candidate = baseline = None
        sign = 0

    checks: dict[str, Any] = {}
    if candidate is not None and baseline is not None:
        for replicate in core.REPLICATES:
            c = summaries2048[candidate[0]][candidate[1]][str(replicate)]
            b = summaries2048[baseline[0]][baseline[1]][str(replicate)]
            task_improved = sum(
                c["by_task"][task]["valid_stop_n"]
                > b["by_task"][task]["valid_stop_n"] for task in TASKS
            )
            checks[str(replicate)] = {
                "valid_stop_net_improvement_n": int(
                    c["overall"]["valid_stop_n"] - b["overall"]["valid_stop_n"]),
                "censored_net_reduction_n": int(
                    b["overall"]["censored_n"] - c["overall"]["censored_n"]),
                "eos_invalid_increase_n": int(
                    c["overall"]["eos_invalid_n"] - b["overall"]["eos_invalid_n"]),
                "tasks_with_valid_stop_improvement_n": task_improved,
                "passed": (
                    c["overall"]["valid_stop_n"] - b["overall"]["valid_stop_n"] >= 13
                    and b["overall"]["censored_n"] - c["overall"]["censored_n"] >= 13
                    and c["overall"]["eos_invalid_n"] - b["overall"]["eos_invalid_n"] <= 6
                    and task_improved >= 6
                ),
            }
    passed = bool(checks) and all(value["passed"] for value in checks.values())
    return {
        "comparison": label,
        "option_a": {"control": option_a[0], "decoding": option_a[1]},
        "option_b": {"control": option_b[0], "decoding": option_b[1]},
        "replicate_deltas": replicate_deltas,
        "candidate_direction_sign_b_minus_a": sign,
        "candidate": (None if candidate is None else {
            "control": candidate[0], "decoding": candidate[1]}),
        "replicate_checks": checks,
        "passed": passed,
        "interpretation": (
            "A PASS is only a design-stage candidate for a new independent confirmation. "
            "It does not authorize the Phase977 holdout or any mechanism scan."
        ),
    }


def build_effect_screens(summaries2048: dict[str, Any]) -> dict[str, Any]:
    screens: dict[str, Any] = {}
    for control in core.CONTROL_POLICIES:
        label = f"sampling_within_{control}"
        screens[label] = effect_screen(
            label, summaries2048,
            (control, "no_think_sampling"),
            (control, "thinking_sampling"),
        )
    for decoding in core.DECODING_POLICIES:
        label = f"hard_vs_soft_thinking_at_{decoding}"
        screens[label] = effect_screen(
            label, summaries2048,
            ("hard_thinking", decoding), ("soft_thinking", decoding),
        )
        label = f"hard_vs_soft_no_think_at_{decoding}"
        screens[label] = effect_screen(
            label, summaries2048,
            ("hard_no_think", decoding), ("soft_no_think", decoding),
        )
        label = f"hard_thinking_vs_no_think_at_{decoding}"
        screens[label] = effect_screen(
            label, summaries2048,
            ("hard_no_think", decoding), ("hard_thinking", decoding),
        )
        label = f"soft_thinking_vs_no_think_at_{decoding}"
        screens[label] = effect_screen(
            label, summaries2048,
            ("soft_no_think", decoding), ("soft_thinking", decoding),
        )
    return screens


def install_or_validate(report: dict[str, Any]) -> None:
    if AUDIT_PATH.exists():
        prior = core.load_json(AUDIT_PATH, "existing natural audit")
        prior_core = core.without_fields(prior, "audit_sha256", "audited_at_utc")
        core.require(prior.get("audit_sha256") == core.sha256_json(prior_core),
                     "existing natural audit self-hash invalid")
        core.require(prior.get("audit_sha256") == report["audit_sha256"],
                     "existing natural audit differs")
        return
    core.atomic_write_json(AUDIT_PATH, report)


def audit(write: bool) -> dict[str, Any]:
    assert_no_holdout_import()
    protocol = authenticate_protocol()
    manifest = authenticate_manifest(protocol)
    records = read_rows(manifest["manifest_sha256"])
    items = dataset.build_items()
    data_audit = dataset.audit_items(items)
    core.require(data_audit["passed"] is True, "diagnostic dataset audit failed")
    tok = load_tokenizer()
    try:
        row_audit = validate_rows(protocol, manifest, records, tok, items)
    finally:
        del tok
        gc.collect()
    assert_no_holdout_import()
    summaries = build_summaries(records)
    screens = build_effect_screens(summaries["2048"])
    passed_screens = sorted(label for label, value in screens.items() if value["passed"])
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": "natural_three_boundary_full_factorial_audit",
        "role": "design_diagnostic_only",
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "rows_file_sha256": core.sha256_file(ROWS_PATH),
        "dataset_identity": data_audit["identity"],
        "row_audit": row_audit,
        "checkpoints": summaries,
        "candidate_effect_screens": screens,
        "passed_candidate_screens": passed_screens,
        "new_independent_confirmation_candidate_exists": bool(passed_screens),
        "phase977_holdout_authorized": False,
        "mechanism_authorized": False,
        "holdout_loaded": False,
        "model_weights_loaded": False,
        "two_replicate_boundary": (
            "Two frozen replicate streams are only a minimum seed-dependence screen; "
            "they do not estimate the sampling distribution or stable variance."
        ),
        "decision_boundary": (
            "No Phase979 diagnostic result can revise Phase978, open the old sealed "
            "holdout, or authorize layer/span/cross-time experiments."
        ),
    }
    report = {
        **payload,
        "audit_sha256": core.sha256_json(payload),
        "audited_at_utc": core.utc_now(),
    }
    if write:
        install_or_validate(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(bool(args.write))
    print(json.dumps({
        "audit_sha256": result["audit_sha256"],
        "passed_candidate_screens": result["passed_candidate_screens"],
        "holdout_loaded": result["holdout_loaded"],
        "mechanism_authorized": result["mechanism_authorized"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
