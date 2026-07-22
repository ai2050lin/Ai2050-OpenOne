#!/usr/bin/env python3
"""Collect stage-isolated, formula-free Phase576 generation residual panels.

This collector reruns the already-qualified deterministic short-answer interface
and requires every generated capsule to match behavior repeat1.  It records all
layers at every rendered-prompt token and at every generated token that was
actually fed back through the cached autoregressive computation.  Frozen
semantic-role spans are labels, not a sampling filter.  This is a complete
residual-stream record of executed token positions, but it is not a complete
component trace, a teacher-forced replay, an intervention, or a mechanism test.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import msvcrt
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
GPT5 = ROOT / "tests/gpt5"
GLM5 = ROOT / "tests/glm5"
for search_path in (GPT5, GLM5):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import phase576_gpt5_fruit_protocol as protocol  # noqa: E402
import phase576_gpt5_fruit_behavior as behavior_execution  # noqa: E402
from phase576_gpt5_fruit_engineering_qualification import (  # noqa: E402
    runtime_identity,
)
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase983_cross_model_engine import (  # noqa: E402
    load_model_adapter,
    release_model_adapter,
)


PROMPT_ROLES = (
    "focus_object_last_token",
    "comparison_object_last_token_when_present",
    "query_anchor_last_token",
    "answer_boundary",
)
MAX_FEEDBACK_STEPS = protocol.MAX_NEW_TOKENS - 1
FEEDBACK_ROLES = tuple(
    f"generated_feedback_token_{index:02d}" for index in range(MAX_FEEDBACK_STEPS)
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_exclusive(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite Phase576 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale Phase576 temporary artifact: {temporary}")
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def save_torch_exclusive(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite Phase576 trace shard: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale Phase576 trace temporary: {temporary}")
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def behavior_dir(stage: str) -> Path:
    return protocol.OUT_DIR / "open_behavior" / stage


def behavior_rows_path(model: str, stage: str) -> Path:
    return behavior_dir(stage) / f"phase576_{model}_{stage}_behavior_rows.jsonl.gz"


def trace_stage_dir(stage: str) -> Path:
    return protocol.OUT_DIR / "natural_trace" / stage


def trace_model_dir(model: str, stage: str) -> Path:
    return protocol.trace_model_dir(stage, model)


def trace_contract_path(model: str, stage: str) -> Path:
    return trace_model_dir(model, stage) / "phase576_generation_trace_contract.json"


def trace_started_path(model: str, stage: str) -> Path:
    return trace_model_dir(model, stage) / "phase576_generation_trace_started.json"


def trace_completed_path(model: str, stage: str) -> Path:
    return trace_model_dir(model, stage) / "phase576_generation_trace_completed.json"


def trace_failed_path(model: str, stage: str) -> Path:
    return trace_model_dir(model, stage) / "phase576_generation_trace_failed.json"


def trace_manifest_path(model: str, stage: str) -> Path:
    return protocol.trace_manifest_path(stage, model)


def trace_stage_receipt_path(stage: str) -> Path:
    return trace_stage_dir(stage) / f"phase576_{stage}_trace_execution_receipt.json"


def trace_model_receipt_path(model: str, stage: str) -> Path:
    return protocol.trace_receipt_path(stage, model)


def quarantine_incomplete_trace_stage(stage: str) -> dict[str, Any] | None:
    source = trace_stage_dir(stage)
    if not source.exists():
        return None
    if trace_stage_receipt_path(stage).is_file():
        raise RuntimeError(f"Phase576 {stage} trace stage is already terminal")
    parent = source.parent.resolve(strict=True)
    if source.is_symlink() or source.resolve(strict=True).parent != parent:
        raise RuntimeError("refusing to quarantine aliased trace stage")
    inventory = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("refusing to quarantine trace stage containing symlink")
        if path.is_file():
            inventory.append({
                "path": str(path.relative_to(source)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = parent / f".{source.name}.aborted-{stamp}-pid{os.getpid()}"
    if destination.exists():
        raise RuntimeError("trace quarantine destination already exists")
    source.rename(destination)
    record = {
        "reason": "nonterminal_prior_stage_atomically_quarantined",
        "path": str(destination.relative_to(protocol.OUT_DIR)).replace("\\", "/"),
        "file_inventory": inventory,
        "file_inventory_sha256": stable_hash(inventory),
    }
    write_json_exclusive(destination / "phase576_quarantine_receipt.json", record)
    return record


def acquire_trace_stage_lease(stage: str) -> Any:
    path = protocol.OUT_DIR / "natural_trace" / f".phase576_{stage}.lease"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    if path.stat().st_size == 0:
        handle.write(b"0")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Phase576 {stage} trace stage is actively leased") from exc
    return handle


def release_trace_stage_lease(handle: Any) -> None:
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def partial_shard_registry(model: str, stage: str) -> list[dict[str, Any]]:
    directory = trace_model_dir(model, stage)
    if not directory.is_dir():
        return []
    return [
        {
            "path": str(path.relative_to(protocol.OUT_DIR)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.glob("phase576_generation_trace_shard_*.pt"))
    ]


def _audit_and_commitment_checks(
    frozen: dict[str, Any], audit: dict[str, Any], commitment: dict[str, Any],
) -> dict[str, bool]:
    return {
        "audit_schema_phase": audit.get("schema_version")
        == "phase576_static_audit.v2" and audit.get("phase_id") == protocol.PHASE,
        "audit_valid": audit.get("valid") is True and audit.get("failures") == [],
        "audit_cpu_no_models": audit.get("model_weights_loaded") is False
        and audit.get("cuda_used") is False,
        "audit_no_sealed_analysis": audit.get(
            "sealed_model_or_result_read_for_analysis"
        ) is False,
        "audit_grid_valid": audit.get("case_grid_audit", {}).get("valid") is True,
        "audit_cross_model_comparison_rule": isinstance(
            audit.get("cross_model_observational_comparison_rule_audit"), dict
        ) and bool(audit["cross_model_observational_comparison_rule_audit"])
        and all(
            audit["cross_model_observational_comparison_rule_audit"].values()
        ),
        "audit_protocol_hash": audit.get("protocol_sha256")
        == sha256_file(protocol.PROTOCOL_PATH),
        "audit_commitment_hash": audit.get("sealed_commitment_sha256")
        == sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "frozen_commitment_hash": frozen.get("sealed_commitment_sha256")
        == sha256_file(protocol.SEALED_COMMITMENT_PATH),
        "sealed_model_unopened": commitment.get("sealed_model_opened") is False,
        "sealed_model_access_zero": commitment.get("sealed_model_access_count") == 0,
        "sealed_analysis_access_zero": commitment.get(
            "sealed_result_analysis_access_count"
        ) == 0,
        "prior_sealed_unread": commitment.get("prior_sealed_files_read") is False,
    }


def _verify_predecessor(stage: str) -> dict[str, Any]:
    if stage == "discovery":
        forbidden = (
            protocol.DISCOVERY_REGISTRY_PATH,
            protocol.CONFIRMATION_DECISION_PATH,
            protocol.HELDOUT_DECISION_PATH,
            protocol.BEHAVIOR_DECISION_PATHS["confirmation"],
            protocol.BEHAVIOR_DECISION_PATHS["heldout_recombination"],
        )
        if any(path.exists() for path in forbidden):
            raise RuntimeError("future-stage artifact exists before discovery trace")
        return {"discovery_registry": None, "confirmation_decision": None}
    if stage == "confirmation":
        verifier = getattr(protocol, "verify_discovery_registry", None)
        if verifier is None:
            raise RuntimeError("strict discovery-registry verifier is unavailable")
        registry = verifier()
        if registry.get("discovery_candidate_pass") is not True:
            raise RuntimeError(
                "confirmation trace requires passed discovery evidence"
            )
        if (
            protocol.CONFIRMATION_DECISION_PATH.exists()
            or protocol.HELDOUT_DECISION_PATH.exists()
            or protocol.BEHAVIOR_DECISION_PATHS["heldout_recombination"].exists()
        ):
            raise RuntimeError("future-stage artifact exists before confirmation trace")
        return {
            "discovery_registry": registry,
            "confirmation_decision": None,
        }
    verifier = getattr(protocol, "verify_confirmation_decision", None)
    if verifier is None:
        raise RuntimeError("strict confirmation-decision verifier is unavailable")
    confirmation = verifier()
    if protocol.HELDOUT_DECISION_PATH.exists():
        raise RuntimeError("heldout decision exists before heldout trace")
    return {
        "discovery_registry": read_json(protocol.DISCOVERY_REGISTRY_PATH),
        "confirmation_decision": confirmation,
    }


def verify_stage_admission(
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if stage not in protocol.OPEN_SPLITS:
        raise RuntimeError(f"unsupported Phase576 stage: {stage}")
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.STATIC_AUDIT_PATH)
    commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    decision_path = protocol.BEHAVIOR_DECISION_PATHS[stage]
    if not decision_path.exists():
        raise RuntimeError(f"missing Phase576 {stage} behavior decision")
    decision = read_json(decision_path)
    protocol.verify_frozen_source_seals(frozen)
    protocol.verify_frozen_model_artifacts(frozen)
    freeze_commit = protocol._verify_freeze_commit()
    qualification = behavior_execution.verify_engineering_qualification(frozen)
    qualification_sha256 = sha256_file(protocol.ENGINEERING_QUALIFICATION_PATH)
    checks = _audit_and_commitment_checks(frozen, audit, commitment)
    source_key = "tests/glm5/phase576_gpt5_fruit_behavior_analysis.py"
    reports = decision.get("reports", [])
    qualified_from_reports = [
        item.get("model") for item in reports if item.get("behavior_gate_pass") is True
    ]
    checks.update({
        "protocol_schema": frozen.get("schema_version") == protocol.SCHEMA_VERSION,
        "audit_schema": audit.get("schema_version") == "phase576_static_audit.v2",
        "commitment_schema": commitment.get("schema_version")
        == "phase576_sealed_commitment.v2",
        "freeze_commit_complete": freeze_commit.get("complete") is True,
        "freeze_lock_absent": not protocol.FREEZE_LOCK_PATH.exists(),
        "prior_open_identity_chain": audit.get("prior_open_file_identities")
        == frozen.get("prior_open_file_identities"),
        "decision_schema": decision.get("schema_version")
        == "phase576_behavior_decision.v2",
        "decision_phase": decision.get("phase_id") == protocol.PHASE,
        "decision_stage": decision.get("stage") == stage,
        "decision_model_order": decision.get("models_in_required_execution_order")
        == list(protocol.MODELS),
        "report_model_order": [item.get("model") for item in reports]
        == list(protocol.MODELS),
        "qualified_registry": decision.get("qualified_models")
        == qualified_from_reports,
        "cross_model_gate": decision.get(
            "cross_model_observational_comparison_authorized"
        )
        is (len(qualified_from_reports) == len(protocol.MODELS)),
        "stage_case_hash": decision.get("stage_cases_sha256")
        == sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "protocol_hash": decision.get("protocol_sha256")
        == sha256_file(protocol.PROTOCOL_PATH),
        "analysis_source_hash": decision.get("analysis_source_sha256")
        == frozen["stage_source_seals"][source_key]["sha256"],
        "analysis_source_seal": decision.get("analysis_source_seal")
        == frozen["stage_source_seals"][source_key],
        "engineering_qualification_hash": decision.get(
            "engineering_qualification_sha256"
        ) == qualification_sha256,
        "engineering_receipt_hash": decision.get(
            "engineering_execution_receipt_sha256"
        ) == qualification.get("execution_receipt_sha256"),
        "qualified_runtime_identity": (
            qualification.get("runtime_identity") == runtime_identity()
            and decision.get("runtime_identity") == qualification.get("runtime_identity")
        ),
        "no_sealed": decision.get("sealed_model_access") is False,
        "no_intervention": decision.get("internal_intervention_authorized") is False,
        "no_mechanism_claim": decision.get("mechanism_claim_authorized") is False,
        "trace_source_sealed": frozen["stage_source_seals"][
            "tests/glm5/phase576_gpt5_fruit_natural_trace.py"
        ]["sha256"] == sha256_file(Path(__file__).resolve()),
        "trace_policy": all((
            frozen.get("trace_policy", {}).get("full_executed_residual_trajectory")
            is True,
            frozen.get("trace_policy", {}).get("full_token_trajectory") is True,
            frozen.get("trace_policy", {}).get(
                "batch_absorbing_eos_and_pad_feedback_positions_included"
            ) is True,
            frozen.get("trace_policy", {}).get("record_all_model_layers") is True,
            frozen.get("trace_policy", {}).get("residual_storage_dtype")
            == "bfloat16",
            frozen.get("trace_policy", {}).get(
                "raw_role_spans_are_labels_not_sampling_filters"
            ) is True,
            frozen.get("trace_policy", {}).get("attention_component_states_recorded")
            is False,
            frozen.get("trace_policy", {}).get("mlp_component_states_recorded")
            is False,
            frozen.get("trace_policy", {}).get("role_labels")
            == list(PROMPT_ROLES),
        )),
    })
    for report, model in zip(reports, protocol.MODELS):
        passed = report.get("behavior_gate_pass") is True
        checks[f"{model}_report_scope"] = (
            report.get("stage") == stage
            and report.get("internal_trace_authorized") is passed
            and report.get("single_model_trace_authorized") is passed
            and report.get("sealed_model_access") is False
            and report.get("internal_intervention_authorized") is False
            and report.get("mechanism_claim_authorized") is False
        )
        trace_ids = report.get("trace_case_ids", [])
        checks[f"{model}_trace_denominator"] = (
            len(trace_ids) == 336 and len(set(trace_ids)) == 336
        ) if passed else trace_ids == []
    if not all(checks.values()):
        raise RuntimeError(f"Phase576 {stage} trace admission failed: {checks}")
    predecessor = _verify_predecessor(stage)
    return frozen, decision, qualification, predecessor


def _span_tuple(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, dict)
        or set(value) != {"start", "end", "text"}
        or not isinstance(value.get("text"), str)
        or not isinstance(value.get("start"), int)
        or isinstance(value.get("start"), bool)
        or not isinstance(value.get("end"), int)
        or isinstance(value.get("end"), bool)
    ):
        raise RuntimeError(f"invalid frozen raw span for {label}")
    start, end = value["start"], value["end"]
    if start < 0 or end <= start:
        raise RuntimeError(f"empty/out-of-range frozen raw span for {label}")
    return start, end


def _token_position_for_raw_span(
    offsets: list[tuple[int, int]], rendered_raw_start: int,
    raw_span: tuple[int, int], label: str,
) -> int:
    absolute_start = rendered_raw_start + raw_span[0]
    absolute_end = rendered_raw_start + raw_span[1]
    overlapping = [
        index for index, (start, end) in enumerate(offsets)
        if end > start and start < absolute_end and end > absolute_start
    ]
    if not overlapping:
        raise RuntimeError(f"frozen raw span maps to no rendered token: {label}")
    return overlapping[-1]


def prepare_prompt(tokenizer: Any, model: str, row: dict[str, Any]) -> dict[str, Any]:
    spans = row.get("raw_role_char_spans")
    if not isinstance(spans, dict):
        raise RuntimeError(f"{row['case_id']}: missing frozen raw role spans")
    raw_prompt = row["raw_prompt"]
    rendered = render_chat(tokenizer, model, raw_prompt)
    raw_start = rendered.find(raw_prompt)
    if raw_start < 0 or rendered.find(raw_prompt, raw_start + 1) >= 0:
        raise RuntimeError(f"{row['case_id']}: raw prompt is not unique in rendering")
    encoded = tokenizer(
        rendered, add_special_tokens=True, return_offsets_mapping=True,
        return_attention_mask=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    attention = [int(value) for value in encoded["attention_mask"]]
    if len(input_ids) != len(offsets) or attention != [1] * len(input_ids):
        raise RuntimeError(f"{row['case_id']}: tokenizer offset contract failed")

    expected_fragments = {
        "focus_object": row["focus_object_label"],
        "query_anchor": row["query_anchor_fragment"],
    }
    role_positions: list[int] = []
    role_mask = [True, row["comparison_object_label"] is not None, True, True]
    for role in ("focus", "comparison", "query_anchor"):
        value = spans.get(role)
        if role == "comparison" and value is None:
            role_positions.append(0)
            continue
        start, end = _span_tuple(value, role)
        if end > len(raw_prompt):
            raise RuntimeError(f"{row['case_id']}: {role} span exceeds prompt")
        expected = (
            row["comparison_object_label"] if role == "comparison"
            else expected_fragments["focus_object" if role == "focus" else role]
        )
        if (
            value["text"].casefold() != expected.casefold()
            or raw_prompt[start:end].casefold() != expected.casefold()
        ):
            raise RuntimeError(f"{row['case_id']}: {role} frozen span text drift")
        role_positions.append(_token_position_for_raw_span(
            offsets, raw_start, (start, end), role,
        ))
    role_positions.append(len(input_ids) - 1)
    if any(position < 0 or position >= len(input_ids) for position in role_positions):
        raise RuntimeError(f"{row['case_id']}: role token position out of bounds")
    return {
        **row,
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "input_ids": input_ids,
        "unpad_role_positions": role_positions,
        "prompt_role_mask": role_mask,
    }


def behavior_repeat1_by_case(
    model: str, stage: str, cases: list[dict[str, Any]], report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows_file = behavior_rows_path(model, stage)
    if not rows_file.exists() or report.get("behavior_rows_sha256") != sha256_file(rows_file):
        raise RuntimeError(f"{stage}/{model}: behavior rows hash unavailable/drifted")
    rows = read_jsonl_gz(rows_file)
    repeat1 = [row for row in rows if row.get("execution_repeat") == "repeat1"]
    expected_ids = {row["case_id"] for row in cases}
    actual_ids = [row.get("case_id") for row in repeat1]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise RuntimeError(f"{stage}/{model}: repeat1 capsule registry is not exact")
    output = {row["case_id"]: row for row in repeat1}
    for case_id, row in output.items():
        if (
            row.get("schema_version") != "phase576_open_behavior_row.v2"
            or row.get("phase_id") != protocol.PHASE
            or row.get("model") != model
            or row.get("stage") != stage
            or row.get("split") != stage
            or row.get("sealed_model_access") is not False
        ):
            raise RuntimeError(f"{stage}/{model}/{case_id}: behavior capsule scope drift")
        content = row.get("generated_token_ids_before_eos")
        full_suffix = row.get("full_generated_suffix_token_ids")
        if not isinstance(content, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in content
        ):
            raise RuntimeError(f"{stage}/{model}/{case_id}: invalid behavior token capsule")
        eos_seen = row.get("eos_seen") is True
        first_eos = row.get("first_eos_token_id")
        if eos_seen != isinstance(first_eos, int):
            raise RuntimeError(f"{stage}/{model}/{case_id}: EOS capsule mismatch")
        capsule = list(content) + ([int(first_eos)] if eos_seen else [])
        if not capsule or len(capsule) > protocol.MAX_NEW_TOKENS:
            raise RuntimeError(f"{stage}/{model}/{case_id}: capsule length invalid")
        if not eos_seen and len(capsule) != protocol.MAX_NEW_TOKENS:
            raise RuntimeError(f"{stage}/{model}/{case_id}: non-EOS capsule is not budget-complete")
        if (
            not isinstance(full_suffix, list)
            or not full_suffix
            or len(full_suffix) > protocol.MAX_NEW_TOKENS
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in full_suffix
            )
            or full_suffix[:len(capsule)] != capsule
        ):
            raise RuntimeError(f"{stage}/{model}/{case_id}: full behavior suffix invalid")
        row["_expected_generated_capsule"] = capsule
        row["_expected_full_generated_suffix"] = list(full_suffix)
    return output


def preflight_locators(
    stage: str, cases: list[dict[str, Any]], qualified_models: list[str],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for model in qualified_models:
        tokenizer = protocol.tokenizer_for(model)
        prepared = [prepare_prompt(tokenizer, model, row) for row in cases]
        reports[model] = {
            "case_count": len(prepared),
            "prompt_token_min": min(len(row["input_ids"]) for row in prepared),
            "prompt_token_max": max(len(row["input_ids"]) for row in prepared),
            "role_registry_sha256": stable_hash([
                {
                    "case_id": row["case_id"],
                    "positions": row["unpad_role_positions"],
                    "mask": row["prompt_role_mask"],
                    "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                }
                for row in prepared
            ]),
        }
        del tokenizer, prepared
        gc.collect()
    return {
        "stage": stage,
        "models": reports,
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def _actual_capsules(
    sequences: torch.Tensor, prompt_width: int, eos_ids: set[int],
) -> list[list[int]]:
    result: list[list[int]] = []
    for sequence in sequences:
        suffix = [int(value) for value in sequence[prompt_width:].tolist()]
        eos_at = next((i for i, value in enumerate(suffix) if value in eos_ids), None)
        result.append(suffix if eos_at is None else suffix[:eos_at + 1])
    return result


def collect_batch(
    adapter: Any, model: str, batch: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full(
        (len(batch), width), int(adapter.pad_token_id), dtype=torch.long,
        device=adapter.input_device,
    )
    attention_mask = torch.zeros(
        (len(batch), width), dtype=torch.long, device=adapter.input_device,
    )
    padded_roles: list[list[int]] = []
    for index, row in enumerate(batch):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=adapter.input_device)
        shift = width - len(row["input_ids"])
        input_ids[index, shift:] = values
        attention_mask[index, shift:] = 1
        padded_roles.append([shift + value for value in row["unpad_role_positions"]])

    eos_ids = {int(value) for value in adapter.eos_identity["effective_eos_token_ids"]}
    with torch.inference_mode():
        generated = adapter.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=protocol.MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=adapter.pad_token_id,
            eos_token_id=sorted(eos_ids),
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    actual_full_suffixes = [
        [int(value) for value in sequence[width:].tolist()]
        for sequence in generated.sequences
    ]
    actual_capsules = _actual_capsules(generated.sequences, width, eos_ids)
    expected_capsules = [row["_expected_generated_capsule"] for row in batch]
    expected_full_suffixes = [
        row["_expected_full_generated_suffix"] for row in batch
    ]
    if (
        actual_capsules != expected_capsules
        or actual_full_suffixes != expected_full_suffixes
    ):
        mismatches = [
            row["case_id"]
            for row, actual_capsule, expected_capsule, actual_full, expected_full
            in zip(
                batch, actual_capsules, expected_capsules,
                actual_full_suffixes, expected_full_suffixes,
            )
            if actual_capsule != expected_capsule or actual_full != expected_full
        ]
        raise RuntimeError(
            f"{model}: trace regeneration differs from behavior repeat1: {mismatches[:8]}"
        )
    hidden_steps = generated.hidden_states
    if not isinstance(hidden_steps, tuple) or not hidden_steps:
        raise RuntimeError(f"{model}: generate did not return hidden states")
    initial_layers = hidden_steps[0]
    if not isinstance(initial_layers, tuple) or not initial_layers:
        raise RuntimeError(f"{model}: initial generation step has no hidden states")
    layer_count = len(initial_layers)
    hidden_size = int(initial_layers[0].shape[-1])
    if any(len(step) != layer_count for step in hidden_steps):
        raise RuntimeError(f"{model}: hidden layer count changes across generation")
    if layer_count != int(adapter.config.num_hidden_layers) + 1:
        raise RuntimeError(f"{model}: hidden layer count differs from model config")
    if hidden_size != int(adapter.config.hidden_size):
        raise RuntimeError(f"{model}: hidden width differs from model config")
    generated_width = int(generated.sequences.shape[1]) - width
    if len(hidden_steps) != generated_width:
        raise RuntimeError(f"{model}: hidden step count differs from generated width")
    for step_index, layer_bank in enumerate(hidden_steps):
        expected_width = width if step_index == 0 else 1
        for layer_index, state in enumerate(layer_bank):
            if (
                not isinstance(state, torch.Tensor)
                or list(state.shape) != [len(batch), expected_width, hidden_size]
                or state.dtype != torch.bfloat16
                or state.device != adapter.input_device
            ):
                raise RuntimeError(
                    f"{model}: hidden tensor contract drift at step={step_index}, "
                    f"layer={layer_index}"
                )

    executed_prompt_mask = attention_mask.bool()
    prefill_tensor = torch.stack(list(initial_layers), dim=1)
    prefill_tensor = prefill_tensor.masked_fill(
        ~executed_prompt_mask[:, None, :, None], 0,
    )

    executed_feedback_count = len(hidden_steps) - 1
    if any(
        len(suffix) != len(hidden_steps) for suffix in actual_full_suffixes
    ):
        raise RuntimeError(f"{model}: full suffix width differs from hidden steps")
    feedback_tensors = []
    for feedback_index in range(MAX_FEEDBACK_STEPS):
        generation_step = feedback_index + 1
        if generation_step < len(hidden_steps):
            state = torch.stack(
                [layer[:, -1, :] for layer in hidden_steps[generation_step]], dim=1,
            )
        else:
            state = torch.zeros(
                (len(batch), layer_count, hidden_size),
                dtype=prefill_tensor.dtype, device=adapter.input_device,
            )
        valid = torch.full(
            (len(batch),), feedback_index < executed_feedback_count,
            dtype=torch.bool, device=adapter.input_device,
        )
        state = state.masked_fill(~valid[:, None, None], 0)
        feedback_tensors.append(state[:, :, None, :])
    feedback_tensor = torch.cat(feedback_tensors, dim=2)
    if not bool(torch.isfinite(prefill_tensor.float()).all().item()):
        raise RuntimeError(f"{model}: non-finite prefill residual before storage")
    if not bool(torch.isfinite(feedback_tensor.float()).all().item()):
        raise RuntimeError(f"{model}: non-finite feedback residual before storage")
    stored_prefill = prefill_tensor.to(dtype=torch.bfloat16).cpu().contiguous()
    stored_feedback = feedback_tensor.to(dtype=torch.bfloat16).cpu().contiguous()
    if not bool(torch.isfinite(stored_prefill.float()).all().item()):
        raise RuntimeError(f"{model}: non-finite prefill residual after BF16 conversion")
    if not bool(torch.isfinite(stored_feedback.float()).all().item()):
        raise RuntimeError(f"{model}: non-finite feedback residual after BF16 conversion")

    rows = []
    for index, row in enumerate(batch):
        rows.append({
            "case_id": row["case_id"],
            "independent_unit_id": row["independent_unit_id"],
            "relation": row["relation"],
            "interface": row["interface"],
            "surface_id": row["surface_id"],
            "order": row["order"],
            "rendered_prompt_sha256": row["rendered_prompt_sha256"],
            "rendered_prompt_token_ids": row["input_ids"],
            "unpad_prompt_role_positions": row["unpad_role_positions"],
            "padded_prompt_role_positions": padded_roles[index],
            "prompt_role_mask": row["prompt_role_mask"],
            "behavior_repeat": "repeat1",
            "generated_capsule_token_ids": actual_capsules[index],
            "full_generated_suffix_token_ids": actual_full_suffixes[index],
            "feedback_token_ids": actual_full_suffixes[index][:-1],
            "feedback_mask": [
                feedback_index < executed_feedback_count
                for feedback_index in range(MAX_FEEDBACK_STEPS)
            ],
        })
    identity = {
        "batch_size": len(batch),
        "prompt_padded_width": width,
        "generation_iteration_count": len(hidden_steps),
        "hidden_state_count": layer_count,
        "hidden_size": hidden_size,
        "runtime_dtype": str(prefill_tensor.dtype),
        "stored_dtype": str(stored_prefill.dtype),
        "prefill_position_count": width,
        "feedback_slot_count": len(FEEDBACK_ROLES),
        "executed_feedback_position_count": executed_feedback_count,
    }
    del (
        generated, hidden_steps, initial_layers, prefill_tensor, feedback_tensors,
        feedback_tensor, input_ids, attention_mask, executed_prompt_mask,
    )
    return {
        "prefill_residual": stored_prefill,
        "feedback_residual": stored_feedback,
    }, rows, identity


def run_model(
    model: str, stage: str, frozen: dict[str, Any], decision: dict[str, Any],
    cases: list[dict[str, Any]], report: dict[str, Any],
    candidate_specification_sha256: str | None,
    discovery_registry_sha256: str | None,
    confirmation_decision_sha256: str | None,
    qualification_sha256: str,
    engineering_receipt_sha256: str,
    qualified_runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    out_dir = trace_model_dir(model, stage)
    if out_dir.exists():
        raise RuntimeError(f"refusing to reuse Phase576 trace model directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    behavior_rows = behavior_repeat1_by_case(model, stage, cases, report)
    contract = {
        "schema_version": "phase576_generation_trace_contract.v2",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "model": model,
        "stage": stage,
        "model_order_index": protocol.MODELS.index(model),
        "case_count": len(cases),
        "case_ids_sha256": stable_hash([row["case_id"] for row in cases]),
        "stage_cases_sha256": sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "behavior_rows_sha256": sha256_file(behavior_rows_path(model, stage)),
        "behavior_decision_sha256": sha256_file(protocol.BEHAVIOR_DECISION_PATHS[stage]),
        "candidate_specification_sha256": candidate_specification_sha256,
        "discovery_registry_sha256": discovery_registry_sha256,
        "confirmation_decision_sha256": confirmation_decision_sha256,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": qualified_runtime_identity,
        "trace_source_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_stage_source_seals": frozen["stage_source_seals"],
        "model_artifact_identity": frozen["model_artifact_identities"][model],
        "controlled_generation_interface": True,
        "deterministic_generation_reexecution": True,
        "behavior_repeat1_capsule_identity_required": True,
        "teacher_forced_replay": False,
        "cached_autoregressive_generation": True,
        "all_layers": True,
        "prompt_role_labels": list(PROMPT_ROLES),
        "feedback_slots": list(FEEDBACK_ROLES),
        "all_rendered_prompt_token_positions": True,
        "all_actually_executed_feedback_token_positions": True,
        "batch_absorbing_eos_and_pad_feedback_positions_included": True,
        "full_vectors_at_every_executed_residual_position": True,
        "complete_component_trajectory": False,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "stored_dtype": "bfloat16",
        "finite_values_required": True,
        "internal_intervention": False,
        "causal": False,
        "sealed_model_access": False,
    }
    write_json_exclusive(trace_contract_path(model, stage), contract)
    write_json_exclusive(trace_started_path(model, stage), {
        "schema_version": "phase576_generation_trace_started.v1",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "model": model,
        "stage": stage,
        "status": "running",
        "contract_sha256": sha256_file(trace_contract_path(model, stage)),
        "sealed_model_access": False,
    })

    adapter = None
    manifest: dict[str, Any] | None = None
    failure: BaseException | None = None
    started = time.time()
    try:
        protocol.verify_frozen_model_artifacts(frozen, (model,))
        adapter = load_model_adapter(model)
        adapter.tokenizer.padding_side = "left"
        loaded_quantization = adapter.identity.get("loaded_quantization", {})
        if not all((
            adapter.identity.get("model_key") == model,
            adapter.identity.get("cuda_only_no_cpu_or_disk_offload") is True,
            adapter.identity.get("loaded_attn_implementation") == "sdpa",
            loaded_quantization.get("load_in_8bit") is True,
            loaded_quantization.get("floating_parameter_dtypes")
            == ["torch.bfloat16"],
            adapter.input_device.type == "cuda",
        )):
            raise RuntimeError(f"{model}: loaded trace model identity is invalid")
        prepared = []
        for row in cases:
            item = prepare_prompt(adapter.tokenizer, model, row)
            item["_expected_generated_capsule"] = behavior_rows[
                row["case_id"]
            ]["_expected_generated_capsule"]
            prepared.append(item)
        shards = []
        repeat_delta = None
        for shard_index, start in enumerate(range(0, len(prepared), protocol.TRACE_BATCH_SIZE)):
            batch = prepared[start:start + protocol.TRACE_BATCH_SIZE]
            residual, row_registry, identity = collect_batch(adapter, model, batch)
            if shard_index == 0:
                repeated, repeated_registry, repeated_identity = collect_batch(
                    adapter, model, batch,
                )
                if repeated_registry != row_registry or repeated_identity != identity:
                    raise RuntimeError(f"{model}: repeat trace metadata drift")
                deltas = {
                    name: float(
                        (residual[name].float() - repeated[name].float())
                        .abs().max().item()
                    )
                    for name in sorted(residual)
                }
                repeat_delta = max(deltas.values())
                if repeat_delta != 0.0 or any(
                    not torch.equal(residual[name], repeated[name])
                    for name in residual
                ):
                    raise RuntimeError(
                        f"{model}: repeated BF16 trace is not exact (delta={repeat_delta})"
                    )
                del repeated, repeated_registry
            shard_path = out_dir / f"phase576_generation_trace_shard_{shard_index:04d}.pt"
            save_torch_exclusive(shard_path, {
                "schema_version": "phase576_generation_residual_shard.v2",
                "phase_id": protocol.PHASE,
                "model": model,
                "stage": stage,
                "case_rows": row_registry,
                "prompt_role_labels": list(PROMPT_ROLES),
                "feedback_slots": list(FEEDBACK_ROLES),
                "prefill_residual": residual["prefill_residual"],
                "feedback_residual": residual["feedback_residual"],
                "prefill_attention_mask": [
                    [False] * (identity["prefill_position_count"] - len(row["rendered_prompt_token_ids"]))
                    + [True] * len(row["rendered_prompt_token_ids"])
                    for row in row_registry
                ],
                "tensor_identity": identity,
                "all_layers": True,
                "all_executed_residual_positions": True,
                "batch_absorbing_eos_and_pad_feedback_positions_included": True,
                "complete_component_trajectory": False,
                "teacher_forced_replay": False,
                "causal": False,
                "sealed_model_access": False,
            })
            shards.append({
                "path": str(shard_path.relative_to(protocol.OUT_DIR)).replace("\\", "/"),
                "case_ids": [row["case_id"] for row in row_registry],
                "size_bytes": shard_path.stat().st_size,
                "sha256": sha256_file(shard_path),
                **identity,
            })
            del residual, row_registry
            done = min(start + protocol.TRACE_BATCH_SIZE, len(prepared))
            if shard_index == 0 or done == len(prepared) or shard_index % 16 == 15:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {stage}/{model} trace {done}/{len(prepared)}",
                    flush=True,
                )
        manifest = {
            "schema_version": protocol.TRACE_MANIFEST_SCHEMA_VERSION,
            "phase_id": protocol.PHASE,
            "created_at_utc": now(),
            "model": model,
            "stage": stage,
            "case_count": len(prepared),
            "independent_unit_count": len({row["independent_unit_id"] for row in prepared}),
            "shard_count": len(shards),
            "shards": shards,
            "prompt_role_labels": list(PROMPT_ROLES),
            "feedback_slots": list(FEEDBACK_ROLES),
            "all_executed_residual_positions": True,
            "batch_absorbing_eos_and_pad_feedback_positions_included": True,
            "complete_component_trajectory": False,
            "repeat_first_batch_max_abs_delta_bf16": repeat_delta,
            "repeat_first_batch_exact_bf16": repeat_delta == 0.0,
            "all_generated_capsules_match_behavior_repeat1": True,
            "all_values_finite_before_and_after_bf16_conversion": True,
            "loaded_model_identity": adapter.identity,
            "frozen_model_artifact_identity": frozen["model_artifact_identities"][model],
            "contract_sha256": sha256_file(trace_contract_path(model, stage)),
            "behavior_decision_sha256": sha256_file(
                protocol.BEHAVIOR_DECISION_PATHS[stage]
            ),
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "trace_source_sha256": sha256_file(Path(__file__).resolve()),
            "candidate_specification_sha256": candidate_specification_sha256,
            "discovery_registry_sha256": discovery_registry_sha256,
            "confirmation_decision_sha256": confirmation_decision_sha256,
            "stage_cases_sha256": sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
            "engineering_qualification_sha256": qualification_sha256,
            "engineering_execution_receipt_sha256": engineering_receipt_sha256,
            "runtime_identity": qualified_runtime_identity,
            "elapsed_seconds_before_release": time.time() - started,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
            "trace_complete": True,
            "internal_intervention": False,
            "causal": False,
            "sealed_case_payload_parsed_for_analysis": False,
            "sealed_model_access": False,
            "prior_sealed_files_read": False,
        }
    except BaseException as exc:
        failure = exc
        exc.__traceback__ = None
    finally:
        try:
            release_model_adapter(adapter)
        except BaseException as release_exc:
            if failure is None:
                failure = release_exc
                release_exc.__traceback__ = None
        adapter = None
        gc.collect()
        allocated = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        reserved = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        if allocated != 0 and failure is None:
            failure = RuntimeError(
                f"{model}: nonzero PyTorch CUDA allocation after trace release: {allocated}"
            )
        if allocated == 0:
            try:
                protocol.verify_frozen_model_artifacts(frozen, (model,))
            except BaseException as identity_exc:
                if failure is None:
                    failure = identity_exc
                    identity_exc.__traceback__ = None

    if failure is not None:
        write_json_exclusive(trace_failed_path(model, stage), {
            "schema_version": "phase576_generation_trace_failed.v1",
            "phase_id": protocol.PHASE,
            "created_at_utc": now(),
            "model": model,
            "stage": stage,
            "status": "failed",
            "error_type": type(failure).__name__,
            "error": str(failure),
            "trace_contract_sha256": sha256_file(trace_contract_path(model, stage)),
            "started_status_sha256": sha256_file(trace_started_path(model, stage)),
            "partial_shards": partial_shard_registry(model, stage),
            "trace_source_sha256": sha256_file(Path(__file__).resolve()),
            "engineering_qualification_sha256": qualification_sha256,
            "engineering_execution_receipt_sha256": engineering_receipt_sha256,
            "runtime_identity": qualified_runtime_identity,
            "confirmation_decision_sha256": confirmation_decision_sha256,
            "pytorch_cuda_allocated_after_release": allocated,
            "pytorch_cuda_reserved_after_release": reserved,
            "sealed_model_access": False,
        })
        raise failure
    if manifest is None:
        raise RuntimeError(f"{model}: trace completed without a manifest")
    write_json_exclusive(trace_manifest_path(model, stage), manifest)
    write_json_exclusive(trace_completed_path(model, stage), {
        "schema_version": "phase576_generation_trace_completed.v1",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "model": model,
        "stage": stage,
        "status": "complete",
        "manifest_sha256": sha256_file(trace_manifest_path(model, stage)),
        "contract_sha256": sha256_file(trace_contract_path(model, stage)),
        "trace_source_sha256": sha256_file(Path(__file__).resolve()),
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": qualified_runtime_identity,
        "confirmation_decision_sha256": confirmation_decision_sha256,
        "pytorch_cuda_allocated_after_release": allocated,
        "pytorch_cuda_reserved_after_release": reserved,
        "sealed_model_access": False,
    })
    return manifest


def _stage_dependency_hashes(
    stage: str, predecessor: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    if stage == "discovery":
        return None, None, None
    registry = predecessor["discovery_registry"]
    if not isinstance(registry, dict):
        raise RuntimeError(f"{stage}: verified discovery predecessor is unavailable")
    return (
        registry["candidate_specification_sha256"],
        sha256_file(protocol.DISCOVERY_REGISTRY_PATH),
        sha256_file(protocol.CONFIRMATION_DECISION_PATH)
        if stage == "heldout_recombination" else None,
    )


def write_model_receipt(
    model: str,
    stage: str,
    stage_pass: bool,
    trace_attempt_status: str,
    candidate_specification_sha256: str | None,
    discovery_registry_sha256: str | None,
    confirmation_decision_sha256: str | None,
    qualification_sha256: str,
    engineering_receipt_sha256: str,
    qualified_runtime_identity: dict[str, Any],
) -> None:
    if trace_attempt_status not in {"complete", "behavior_blocked", "failed"}:
        raise RuntimeError(f"invalid trace attempt status: {trace_attempt_status}")
    if stage_pass is not (trace_attempt_status == "complete"):
        raise RuntimeError("trace pass/status mismatch")
    contract_file = trace_contract_path(model, stage)
    manifest_file = trace_manifest_path(model, stage)
    completed_file = trace_completed_path(model, stage)
    failed_file = trace_failed_path(model, stage)
    cleanup_status = (
        read_json(completed_file) if completed_file.is_file()
        else read_json(failed_file) if failed_file.is_file()
        else {"pytorch_cuda_allocated_after_release": 0}
    )
    write_json_exclusive(trace_model_receipt_path(model, stage), {
        "schema_version": protocol.TRACE_RECEIPT_SCHEMA_VERSION,
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "model": model,
        "stage": stage,
        "stage_pass": stage_pass,
        "trace_attempt_status": trace_attempt_status,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "stage_cases_sha256": sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "behavior_decision_sha256": sha256_file(
            protocol.BEHAVIOR_DECISION_PATHS[stage]
        ),
        "candidate_specification_sha256": candidate_specification_sha256,
        "discovery_registry_sha256": discovery_registry_sha256,
        "confirmation_decision_sha256": confirmation_decision_sha256,
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": qualified_runtime_identity,
        "trace_contract_sha256": (
            sha256_file(contract_file) if contract_file.is_file() else None
        ),
        "trace_manifest_sha256": (
            sha256_file(manifest_file) if manifest_file.is_file() else None
        ),
        "completed_status_sha256": (
            sha256_file(completed_file) if completed_file.is_file() else None
        ),
        "failed_status_sha256": (
            sha256_file(failed_file) if failed_file.is_file() else None
        ),
        "pytorch_cuda_allocated_after_release": cleanup_status.get(
            "pytorch_cuda_allocated_after_release"
        ),
        "sealed_case_payload_parsed_for_analysis": False,
        "sealed_model_access": False,
        "prior_sealed_files_read": False,
    })


def quarantine_publication_temp(path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if not temporary.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = temporary.with_name(
        temporary.name + f".publication-failed-{stamp}-pid{os.getpid()}"
    )
    if destination.exists():
        raise RuntimeError("publication-temp quarantine destination exists")
    temporary.rename(destination)


def recover_completed_model_publication(
    model: str,
    stage: str,
    frozen: dict[str, Any],
    candidate_specification_sha256: str | None,
    discovery_registry_sha256: str | None,
    confirmation_decision_sha256: str | None,
    qualification_sha256: str,
    engineering_receipt_sha256: str,
    qualified_runtime_identity: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_file = trace_manifest_path(model, stage)
    if not manifest_file.is_file():
        return None
    contract_file = trace_contract_path(model, stage)
    failed_file = trace_failed_path(model, stage)
    if not contract_file.is_file() or failed_file.exists():
        raise RuntimeError(f"{stage}/{model}: manifest has invalid terminal siblings")
    if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
        raise RuntimeError(f"{stage}/{model}: cannot recover publication with live CUDA")
    protocol.verify_frozen_model_artifacts(frozen, (model,))
    manifest = read_json(manifest_file)
    behavior_decision = read_json(protocol.BEHAVIOR_DECISION_PATHS[stage])
    qualification = read_json(protocol.ENGINEERING_QUALIFICATION_PATH)
    if not all((
        manifest.get("schema_version") == protocol.TRACE_MANIFEST_SCHEMA_VERSION,
        manifest.get("phase_id") == protocol.PHASE,
        manifest.get("model") == model,
        manifest.get("stage") == stage,
        manifest.get("trace_complete") is True,
        manifest.get("contract_sha256") == sha256_file(contract_file),
        manifest.get("behavior_decision_sha256")
        == sha256_file(protocol.BEHAVIOR_DECISION_PATHS[stage]),
        manifest.get("protocol_sha256") == sha256_file(protocol.PROTOCOL_PATH),
        manifest.get("stage_cases_sha256")
        == sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        manifest.get("candidate_specification_sha256")
        == candidate_specification_sha256,
        manifest.get("discovery_registry_sha256") == discovery_registry_sha256,
        manifest.get("confirmation_decision_sha256")
        == confirmation_decision_sha256,
        manifest.get("engineering_qualification_sha256") == qualification_sha256,
        manifest.get("engineering_execution_receipt_sha256")
        == engineering_receipt_sha256,
        manifest.get("runtime_identity") == qualified_runtime_identity,
        manifest.get("trace_source_sha256")
        == sha256_file(Path(__file__).resolve()),
        manifest.get("frozen_model_artifact_identity")
        == frozen["model_artifact_identities"][model],
        manifest.get("repeat_first_batch_exact_bf16") is True,
        manifest.get("all_generated_capsules_match_behavior_repeat1") is True,
        manifest.get("all_executed_residual_positions") is True,
        manifest.get("batch_absorbing_eos_and_pad_feedback_positions_included")
        is True,
        manifest.get("internal_intervention") is False,
        manifest.get("causal") is False,
        manifest.get("sealed_model_access") is False,
    )):
        raise RuntimeError(f"{stage}/{model}: cannot recover invalid manifest")
    protocol._verify_trace_shard_closure(
        stage, model, manifest, frozen, behavior_decision, qualification,
    )
    completed_file = trace_completed_path(model, stage)
    if not completed_file.is_file():
        quarantine_publication_temp(completed_file)
        write_json_exclusive(completed_file, {
            "schema_version": "phase576_generation_trace_completed.v1",
            "phase_id": protocol.PHASE,
            "created_at_utc": now(),
            "model": model,
            "stage": stage,
            "status": "complete",
            "manifest_sha256": sha256_file(manifest_file),
            "contract_sha256": sha256_file(contract_file),
            "trace_source_sha256": sha256_file(Path(__file__).resolve()),
            "engineering_qualification_sha256": qualification_sha256,
            "engineering_execution_receipt_sha256": engineering_receipt_sha256,
            "runtime_identity": qualified_runtime_identity,
            "confirmation_decision_sha256": confirmation_decision_sha256,
            "pytorch_cuda_allocated_after_release": 0,
            "pytorch_cuda_reserved_after_release": (
                int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
            ),
            "sealed_model_access": False,
        })
    receipt_file = trace_model_receipt_path(model, stage)
    if not receipt_file.is_file():
        quarantine_publication_temp(receipt_file)
        write_model_receipt(
            model, stage, True, "complete",
            candidate_specification_sha256, discovery_registry_sha256,
            confirmation_decision_sha256, qualification_sha256,
            engineering_receipt_sha256, qualified_runtime_identity,
        )
    else:
        receipt = read_json(receipt_file)
        if not (
            receipt.get("stage_pass") is True
            and receipt.get("trace_attempt_status") == "complete"
            and receipt.get("trace_manifest_sha256") == sha256_file(manifest_file)
            and receipt.get("completed_status_sha256") == sha256_file(completed_file)
        ):
            raise RuntimeError(f"{stage}/{model}: existing receipt blocks recovery")
    return manifest


def _run_stage_with_lease(stage: str) -> dict[str, Any]:
    frozen, decision, qualification, predecessor = verify_stage_admission(stage)
    (
        candidate_specification_sha256,
        discovery_registry_sha256,
        confirmation_decision_sha256,
    ) = (
        _stage_dependency_hashes(stage, predecessor)
    )
    qualification_sha256 = sha256_file(protocol.ENGINEERING_QUALIFICATION_PATH)
    engineering_receipt_sha256 = qualification["execution_receipt_sha256"]
    qualified_runtime_identity = qualification["runtime_identity"]
    cases = read_jsonl(protocol.OPEN_SPLIT_CASE_PATHS[stage])
    if (
        len(cases) != 336
        or len({row.get("case_id") for row in cases}) != 336
        or any(row.get("split") != stage or row.get("sealed") is not False for row in cases)
    ):
        raise RuntimeError(f"Phase576 {stage} case denominator invalid")
    reports = decision["reports"]
    qualified = decision["qualified_models"]
    case_ids = {row["case_id"] for row in cases}
    for report in reports:
        if report["model"] in qualified and set(report["trace_case_ids"]) != case_ids:
            raise RuntimeError(f"{stage}/{report['model']}: trace is not the full denominator")
    locator_report = preflight_locators(stage, cases, qualified)
    if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
        raise RuntimeError("nonzero PyTorch CUDA allocation at trace-stage baseline")
    quarantined_attempt = quarantine_incomplete_trace_stage(stage)
    trace_stage_dir(stage).mkdir(parents=True, exist_ok=False)
    write_json_exclusive(trace_stage_dir(stage) / "phase576_trace_stage_started.json", {
        "schema_version": "phase576_trace_stage_started.v1",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "stage": stage,
        "behavior_decision_sha256": sha256_file(protocol.BEHAVIOR_DECISION_PATHS[stage]),
        "qualified_models": qualified,
        "locator_preflight": locator_report,
        "candidate_specification_sha256": candidate_specification_sha256,
        "discovery_registry_sha256": discovery_registry_sha256,
        "confirmation_decision_sha256": confirmation_decision_sha256,
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": qualified_runtime_identity,
        "quarantined_incomplete_attempt": quarantined_attempt,
        "sealed_model_access": False,
    })

    manifests: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fatal_error: dict[str, Any] | None = None
    report_by_model = {report["model"]: report for report in reports}
    for model in protocol.MODELS:
        if model not in qualified:
            trace_model_dir(model, stage).mkdir(parents=True, exist_ok=False)
            write_model_receipt(
                model, stage, False, "behavior_blocked",
                candidate_specification_sha256, discovery_registry_sha256,
                confirmation_decision_sha256, qualification_sha256,
                engineering_receipt_sha256, qualified_runtime_identity,
            )
            attempts.append({"model": model, "status": "behavior_blocked"})
            print(f"{stage}/{model}: behavior-blocked; trace skipped", flush=True)
            continue
        if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
            fatal_error = {
                "stage": "before_next_model",
                "model": model,
                "error_type": "CudaReleaseError",
                "error": "PyTorch CUDA allocation remains before next trace model",
            }
            break
        try:
            manifest = run_model(
                model, stage, frozen, decision, cases, report_by_model[model],
                candidate_specification_sha256, discovery_registry_sha256,
                confirmation_decision_sha256, qualification_sha256,
                engineering_receipt_sha256, qualified_runtime_identity,
            )
            write_model_receipt(
                model, stage, True, "complete",
                candidate_specification_sha256, discovery_registry_sha256,
                confirmation_decision_sha256, qualification_sha256,
                engineering_receipt_sha256, qualified_runtime_identity,
            )
            manifests.append(manifest)
            attempts.append({"model": model, "status": "complete"})
        except BaseException as exc:
            recovered_manifest = recover_completed_model_publication(
                model,
                stage,
                frozen,
                candidate_specification_sha256,
                discovery_registry_sha256,
                confirmation_decision_sha256,
                qualification_sha256,
                engineering_receipt_sha256,
                qualified_runtime_identity,
            )
            if recovered_manifest is not None:
                manifests.append(recovered_manifest)
                attempts.append({"model": model, "status": "complete"})
                print(
                    f"{stage}/{model}: recovered complete trace publication",
                    flush=True,
                )
                continue
            failed_file = trace_failed_path(model, stage)
            if not failed_file.exists():
                model_dir = trace_model_dir(model, stage)
                model_dir.mkdir(parents=True, exist_ok=True)
                allocated = (
                    int(torch.cuda.memory_allocated())
                    if torch.cuda.is_available() else 0
                )
                reserved = (
                    int(torch.cuda.memory_reserved())
                    if torch.cuda.is_available() else 0
                )
                contract_file = trace_contract_path(model, stage)
                started_file = trace_started_path(model, stage)
                write_json_exclusive(failed_file, {
                    "schema_version": "phase576_generation_trace_failed.v1",
                    "phase_id": protocol.PHASE,
                    "created_at_utc": now(),
                    "model": model,
                    "stage": stage,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "trace_contract_sha256": (
                        sha256_file(contract_file) if contract_file.is_file() else None
                    ),
                    "started_status_sha256": (
                        sha256_file(started_file) if started_file.is_file() else None
                    ),
                    "partial_shards": partial_shard_registry(model, stage),
                    "trace_source_sha256": sha256_file(Path(__file__).resolve()),
                    "engineering_qualification_sha256": qualification_sha256,
                    "engineering_execution_receipt_sha256": engineering_receipt_sha256,
                    "runtime_identity": qualified_runtime_identity,
                    "confirmation_decision_sha256": confirmation_decision_sha256,
                    "pytorch_cuda_allocated_after_release": allocated,
                    "pytorch_cuda_reserved_after_release": reserved,
                    "sealed_model_access": False,
                })
            if not trace_model_receipt_path(model, stage).exists():
                write_model_receipt(
                    model, stage, False, "failed",
                    candidate_specification_sha256, discovery_registry_sha256,
                    confirmation_decision_sha256, qualification_sha256,
                    engineering_receipt_sha256, qualified_runtime_identity,
                )
            failures.append({
                "model": model,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            attempts.append({
                "model": model, "status": "failed", "error_type": type(exc).__name__,
            })
            if torch.cuda.is_available() and torch.cuda.memory_allocated() != 0:
                fatal_error = {
                    "stage": "failed_model_cleanup",
                    "model": model,
                    "error_type": "CudaReleaseError",
                    "error": f"{model}: trace failed with CUDA allocation retained",
                }
                break
    final_allocated = (
        int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    )
    final_reserved = (
        int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
    )
    if final_allocated != 0 and fatal_error is None:
        fatal_error = {
            "stage": "final_cuda_cleanup",
            "model": None,
            "error_type": "CudaReleaseError",
            "error": "PyTorch CUDA allocation remains after trace stage",
        }
    completed = [item["model"] for item in manifests]
    all_models_behavior_qualified = qualified == list(protocol.MODELS)
    all_models_trace_complete = completed == list(protocol.MODELS)
    receipt = {
        "schema_version": "phase576_trace_execution_receipt.v2",
        "phase_id": protocol.PHASE,
        "created_at_utc": now(),
        "stage": stage,
        "models_considered_in_required_order": [item["model"] for item in attempts],
        "attempts": attempts,
        "qualified_models": qualified,
        "completed_models": completed,
        "failed_models": failures,
        "not_attempted_models": [
            model for model in protocol.MODELS
            if model not in {item["model"] for item in attempts}
        ],
        "fatal_error": fatal_error,
        "terminal_status": "failed" if fatal_error is not None else "complete",
        "behavior_decision_sha256": sha256_file(protocol.BEHAVIOR_DECISION_PATHS[stage]),
        "stage_cases_sha256": sha256_file(protocol.OPEN_SPLIT_CASE_PATHS[stage]),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "trace_source_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_specification_sha256": candidate_specification_sha256,
        "discovery_registry_sha256": discovery_registry_sha256,
        "confirmation_decision_sha256": confirmation_decision_sha256,
        "engineering_qualification_sha256": qualification_sha256,
        "engineering_execution_receipt_sha256": engineering_receipt_sha256,
        "runtime_identity": qualified_runtime_identity,
        "trace_manifest_sha256_by_model": {
            item["model"]: sha256_file(trace_manifest_path(item["model"], stage))
            for item in manifests
        },
        "trace_receipt_sha256_by_model": {
            model: sha256_file(trace_model_receipt_path(model, stage))
            for model in protocol.MODELS
            if trace_model_receipt_path(model, stage).is_file()
        },
        "all_models_behavior_qualified": all_models_behavior_qualified,
        "all_models_trace_complete": all_models_trace_complete,
        "single_model_observation_allowed": True,
        "cross_model_observational_comparison_authorized": (
            all_models_behavior_qualified and all_models_trace_complete
        ),
        "cross_model_common_structure_claim_authorized": False,
        "internal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "final_pytorch_cuda_allocated": final_allocated,
        "final_pytorch_cuda_reserved": final_reserved,
        "sealed_model_access": False,
    }
    write_json_exclusive(trace_stage_receipt_path(stage), receipt)
    if fatal_error is not None:
        raise RuntimeError(f"Phase576 {stage} trace fatal: {fatal_error}")
    return receipt


def run_stage(stage: str) -> dict[str, Any]:
    lease = acquire_trace_stage_lease(stage)
    try:
        return _run_stage_with_lease(stage)
    finally:
        release_trace_stage_lease(lease)


def locator_self_test() -> dict[str, Any]:
    results = {}
    for stage in protocol.OPEN_SPLITS:
        cases = protocol.build_split(stage)
        results[stage] = preflight_locators(stage, cases, list(protocol.MODELS))
    return {
        "passed": True,
        "stages": results,
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", choices=protocol.OPEN_SPLITS)
    group.add_argument("--locator-self-test", action="store_true")
    args = parser.parse_args()
    result = locator_self_test() if args.locator_self_test else run_stage(args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
