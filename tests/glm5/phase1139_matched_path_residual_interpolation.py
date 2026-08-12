#!/usr/bin/env python3
"""Matched-path residual interpolation audit for temporal binding.

This phase does not reopen Phase1138. It uses the untouched confirmation
cohort to test whether the descriptive late residual transition has a smooth
or sharply nonlinear downstream response. Target states are taken live from
the exact candidate-scoring pass, so alpha=0 is an identity intervention.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import quantization_audit, release_fp16  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1138_temporal_residual_onset as phase1138  # noqa: E402


PHASE = 1139
MODELS = ("qwen3_4b", "qwen3_14b")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_FRACTION = 0.80
OUT_ROOT = ROOT / "tests/glm5/result/phase1139_matched_path_residual_interpolation"
SOURCE1138 = ROOT / "tests/glm5/result/phase1138_temporal_residual_onset"
SOURCE_ITEMS = source.SOURCE
EXPECTED_CURVES = 13 * 6
EXPECTED_RECORDS = EXPECTED_CURVES * len(ALPHAS)
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction": 0.99,
    "identity_max_abs_margin_drift": 0.005,
    "main_valid_fraction": 0.95,
    "endpoint_flip_fraction": 0.90,
    "monotonic_fraction": 0.90,
    "main_to_same_answer_span_ratio": 2.0,
    "smooth_max_linear_deviation": 0.08,
    "smooth_max_step": 0.35,
    "phase_min_linear_deviation": 0.10,
    "phase_min_max_step": 0.45,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def descriptive_pass(row: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    """Phase1138 single-depth gate with only the self-patch term omitted."""
    return bool(
        row["finite_fraction"] >= thresholds["finite_fraction"]
        and row["behavior_valid_fraction"] >= thresholds["behavior_valid_fraction"]
        and row["main_median_recovery"] >= thresholds["main_median_recovery"]
        and row["main_positive_fraction"] >= thresholds["main_positive_fraction"]
        and row["original_median_recovery"] >= thresholds["panel_median_recovery"]
        and row["swapped_median_recovery"] >= thresholds["panel_median_recovery"]
        and row["specificity_advantage"] >= thresholds["specificity_advantage"]
    )


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1139 protocol after model output exists")

    prereg1138 = read_json(SOURCE1138 / "protocol/preregistration.json")
    audit1138 = read_json(SOURCE1138 / "audit/independent_result_audit.json")
    selection1138 = read_json(SOURCE1138 / "analysis/discovery_selection.json")
    confirmation1138 = read_json(SOURCE1138 / "analysis/causal_confirmation.json")
    thresholds1138 = prereg1138["thresholds"]

    per_model_descriptive: dict[str, list[float]] = {}
    for model_name in MODELS:
        rows = selection1138["models"][model_name]["depth_metrics"]
        per_model_descriptive[model_name] = [
            float(row["requested_fraction"])
            for row in rows
            if float(row["requested_fraction"]) <= MAX_FRACTION
            and descriptive_pass(row, thresholds1138)
        ]
    shared = sorted(set(per_model_descriptive[MODELS[0]]) & set(per_model_descriptive[MODELS[1]]))
    selected_fraction = shared[0] if shared else None
    confirmation_ids = list(prereg1138["behavior_conditioning"]["cohorts"]["confirmation"])

    checks = {
        "phase1138_audit_passed": bool(audit1138["all_checks_passed"]),
        "phase1138_confirmation_was_not_run": confirmation1138["confirmation_run"] is False,
        "phase1138_auto_continue_was_false": confirmation1138["auto_continue"] is False,
        "phase1138_common_frozen_gate_was_empty": selection1138["shared_passing_requested_fractions"] == [],
        "descriptive_fraction_exists": selected_fraction is not None,
        "descriptive_fraction_is_0_7": selected_fraction == 0.7,
        "confirmation_cohort_has_13_items": len(confirmation_ids) == 13,
        "alpha_grid_is_frozen": ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0),
        "no_model_output_before_protocol": not (OUT_ROOT / "runs").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1139 protocol checks failed: {checks}")

    core = {
        "schema_version": "phase1139_matched_path_interpolation_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "On the untouched Phase1138 confirmation cohort, test whether the frozen late residual band has "
            "a smooth or sharply nonlinear downstream response under candidate-path-matched interpolation."
        ),
        "epistemic_scope": (
            "This is a response-curve adjudication, not a reopening of Phase1138, not a component scan, "
            "and not evidence for an attractor, phase transition, or semantic module by itself."
        ),
        "source": {
            "phase1138_protocol_digest": prereg1138["protocol_digest"],
            "phase1138_selection_digest": selection1138["selection_digest"],
            "phase1138_confirmation_digest": confirmation1138["confirmation_digest"],
            "phase1138_audit_digest": audit1138["audit_digest"],
            "source_items_sha256": sha256_file(SOURCE_ITEMS),
            "script_sha256": sha256_file(Path(__file__)),
            "labels": "external_machine_consensus_not_human_gold",
        },
        "models": prereg1138["models"],
        "cohort": {
            "split": "confirmation",
            "item_ids": confirmation_ids,
            "item_count": len(confirmation_ids),
            "untouched_by_phase1138_hidden_scan": True,
        },
        "selection": {
            "rule": (
                "earliest fraction no later than 0.80 where both Phase1138 discovery endpoints passed every "
                "single-depth descriptive term except the self-patch identity threshold"
            ),
            "per_model_descriptive_fractions": per_model_descriptive,
            "shared_descriptive_fractions": shared,
            "selected_requested_fraction": selected_fraction,
        },
        "intervention": {
            "position": "answer_boundary",
            "component": "whole residual stream after frozen layer",
            "target_state": "live state from the exact candidate-scoring forward pass",
            "source_state": "candidate-matched source state captured from the opposite or same-answer condition",
            "formula": "H(alpha)=(1-alpha)*H_target_live+alpha*H_source_candidate_matched",
            "alphas": list(ALPHAS),
            "curve_kinds": ["main", "same_answer_temporal_control"],
            "main_curves_per_item": 4,
            "control_curves_per_item": 2,
            "expected_curves_per_model": EXPECTED_CURVES,
            "expected_records_per_model": EXPECTED_RECORDS,
        },
        "metrics": {
            "identity_drift": "alpha=0 patched margin minus an unhooked margin from identical rows and batching",
            "normalized_curve": "u(alpha)=(m(alpha)-m(0))/(m(1)-m(0))",
            "linear_deviation": "median over internal alphas of abs(u(alpha)-alpha)",
            "max_step": "maximum adjacent increase in normalized recovery",
            "monotonic_fraction": "fraction of adjacent normalized increments at least -0.02",
            "control_span": "absolute endpoint margin change for same-answer temporal donors",
        },
        "thresholds": THRESHOLDS,
        "decision": {
            "instrument_failed": "either endpoint fails finite or identity-equivalence gates",
            "phase_like": (
                "both endpoints pass instrument and endpoint gates, median linear deviation >=0.10, "
                "median max step >=0.45, and dominant max-step interval agrees"
            ),
            "smooth": (
                "both endpoints pass instrument and endpoint gates, median linear deviation <=0.08, "
                "and median max step <=0.35"
            ),
            "mixed": "all other qualified outcomes",
        },
        "hard_stops": [
            "do not revise Phase1138 or K73",
            "do not interpret a nonlinear logit response as proof of an attractor",
            "do not run attention, MLP, head, neuron, SAE, TDA, or full-Jacobian searches in this phase",
            "do not alter the selected depth, alpha grid, cohort, thresholds, or curve classes after model output",
            "do not treat same-family agreement as cross-architecture conservation",
        ],
        "auto_continue_rule": (
            "Phase1139 always stops after adjudication because its confirmation cohort is consumed and any "
            "component or tangent-space test requires a separately frozen independent material axis."
        ),
    }
    prereg = dict(core)
    prereg["protocol_digest"] = digest(core)
    audit_core = {
        "schema_version": "phase1139_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "command": "protocol",
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "selected_requested_fraction": selected_fraction,
        "protocol_digest": prereg["protocol_digest"],
    }, ensure_ascii=False), flush=True)


class LiveResidualInterpolation:
    def __init__(
        self,
        layer,
        positions: torch.Tensor,
        sources: torch.Tensor,
        alphas: torch.Tensor,
    ):
        self.layer = layer
        self.positions = positions
        self.sources = sources
        self.alphas = alphas
        self.handle = None
        self.calls = 0

    def _hook(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("interpolation layer did not return a tensor")
        positions = self.positions.to(value.device)
        sources = self.sources.to(value.device, dtype=value.dtype)
        alphas = self.alphas.to(value.device, dtype=value.dtype).unsqueeze(-1)
        batch = torch.arange(value.shape[0], device=value.device)
        live = value[batch, positions, :]
        mixed = live + alphas * (sources - live)
        zero = alphas.squeeze(-1) == 0
        if bool(zero.any()):
            mixed[zero] = live[zero]
        patched = value.clone()
        patched[batch, positions, :] = mixed
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def candidate_vector_key(case_id: str, candidate_key: str) -> str:
    return f"{case_id}|candidate={candidate_key}"


def capture_candidate_vectors(
    model,
    layer,
    depth: int,
    cases: dict[str, dict[str, Any]],
    tokenizer,
    pad_id: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    rows = [
        source.tokenize_case(tokenizer, case, candidate_key)
        for case in cases.values()
        for candidate_key in ("old", "new")
    ]
    capture = source.ResidualCapture([layer], [1])
    capture.register()
    vectors: dict[str, torch.Tensor] = {}
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start : start + batch_size]
                input_ids, attention_mask = source.pad_sequences(batch_rows, pad_id, device)
                positions = torch.tensor(
                    [int(row["prompt_length"]) - 1 for row in batch_rows],
                    dtype=torch.long,
                    device=device,
                )
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                capture.validate()
                values = capture.values[1]
                for slot, row in enumerate(batch_rows):
                    vectors[candidate_vector_key(str(row["case_id"]), str(row["candidate_key"]))] = values[slot].clone()
                del output, input_ids, attention_mask, positions
    finally:
        capture.close()
    return vectors


def build_curves(item_ids: list[str]) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []

    def add(
        item_id: str,
        kind: str,
        target_state: str,
        source_state: str,
        base_key: str,
        desired_key: str,
        panel: str,
    ) -> None:
        curves.append({
            "curve_id": f"{item_id}|{kind}|{target_state}|{source_state}",
            "item_id": item_id,
            "curve_kind": kind,
            "panel": panel,
            "target_state": target_state,
            "target_case_id": source.state_id(item_id, target_state),
            "source_state": source_state,
            "source_case_id": source.state_id(item_id, source_state),
            "base_key": base_key,
            "desired_key": desired_key,
        })

    for item_id in sorted(item_ids):
        add(item_id, "main", "original_pre", "original_post", "old", "new", "original")
        add(item_id, "main", "original_post", "original_pre", "new", "old", "original")
        add(item_id, "main", "swapped_pre", "swapped_post", "new", "old", "swapped")
        add(item_id, "main", "swapped_post", "swapped_pre", "old", "new", "swapped")
        add(
            item_id,
            "same_answer_temporal_control",
            "original_pre",
            "original_pre_early",
            "old",
            "new",
            "original",
        )
        add(
            item_id,
            "same_answer_temporal_control",
            "original_post",
            "original_post_late",
            "new",
            "old",
            "original",
        )
    return curves


def score_interpolation_batch(
    model,
    layer,
    entries: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    vectors: dict[str, torch.Tensor],
    tokenizer,
    pad_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    owners: list[dict[str, Any]] = []
    source_vectors: list[torch.Tensor] = []
    alpha_rows: list[float] = []
    for entry in entries:
        case = cases[str(entry["target_case_id"])]
        for candidate_key in ("old", "new"):
            row = source.tokenize_case(tokenizer, case, candidate_key)
            expanded.append(row)
            owners.append(entry)
            source_vectors.append(
                vectors[candidate_vector_key(str(entry["source_case_id"]), candidate_key)]
            )
            alpha_rows.append(float(entry["alpha"]))

    input_ids, attention_mask = source.pad_sequences(expanded, pad_id, device)
    positions = torch.tensor(
        [int(row["prompt_length"]) - 1 for row in expanded],
        dtype=torch.long,
        device=device,
    )
    replacements = torch.stack(source_vectors, dim=0)
    alphas = torch.tensor(alpha_rows, dtype=torch.float32, device=device)
    unhooked_scores = None
    with torch.inference_mode():
        if all(math.isclose(float(entry["alpha"]), 0.0, abs_tol=1e-12) for entry in entries):
            clean_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            unhooked_scores = source.scores_from_logits(clean_output.logits, expanded)
            del clean_output
        with LiveResidualInterpolation(layer, positions, replacements, alphas) as patch:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if patch.calls != 1:
            raise RuntimeError(f"interpolation hook called {patch.calls} times")
        patched_scores = source.scores_from_logits(output.logits, expanded)

    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    clean_grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for index, (row, entry, score) in enumerate(zip(expanded, owners, patched_scores)):
        record_id = str(entry["record_id"])
        grouped[record_id][str(row["candidate_key"])] = (
            float(score["logp_mean"]) if score["finite"] else math.nan
        )
        if unhooked_scores is not None:
            clean_score = unhooked_scores[index]
            clean_grouped[record_id][str(row["candidate_key"])] = (
                float(clean_score["logp_mean"]) if clean_score["finite"] else math.nan
            )

    del output, input_ids, attention_mask, positions, replacements, alphas, patched_scores
    result = []
    for entry in entries:
        record_id = str(entry["record_id"])
        row = dict(entry)
        scores = grouped[record_id]
        base_key = str(entry["base_key"])
        desired_key = str(entry["desired_key"])
        values = [scores[base_key], scores[desired_key]]
        finite = all(math.isfinite(value) for value in values)
        margin = scores[desired_key] - scores[base_key] if finite else None
        identity_drift = None
        unhooked_margin = None
        if record_id in clean_grouped:
            clean = clean_grouped[record_id]
            clean_values = [clean[base_key], clean[desired_key]]
            if all(math.isfinite(value) for value in clean_values):
                unhooked_margin = clean[desired_key] - clean[base_key]
                identity_drift = margin - unhooked_margin if margin is not None else None
        row.update({
            "scores": scores if finite else None,
            "oriented_margin": margin,
            "unhooked_oriented_margin": unhooked_margin,
            "identity_margin_drift": identity_drift,
            "finite": finite,
        })
        result.append(row)
    return result


def run_command(model_name: str) -> None:
    if model_name not in MODELS:
        raise RuntimeError("invalid Phase1139 endpoint")
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    audit = read_json(OUT_ROOT / "protocol/audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1139 protocol audit failed")
    output_root = OUT_ROOT / "runs" / model_name
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing Phase1139 output for {model_name}")

    selected_ids = list(prereg["cohort"]["item_ids"])
    items = read_jsonl(SOURCE_ITEMS)
    selected_items, cases = source.causal_cases(items, "confirmation", selected_ids)
    curves = build_curves(selected_ids)
    if len(curves) != EXPECTED_CURVES:
        raise RuntimeError("curve count drift")

    model = None
    started = time.time()
    records: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, placement = phase1138.load_model(model_name, prereg)
        precision = quantization_audit(model)
        expected = prereg["models"][model_name]
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(expected["expected_parameter_count"]):
            raise RuntimeError(f"{model_name} parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError(f"{model_name} FP16/no-quantization gate failed")

        layers = get_layers(model)
        requested_fraction = float(prereg["selection"]["selected_requested_fraction"])
        depth_rows = [
            row
            for row in phase1138.depth_rows_for_model(len(layers))
            if math.isclose(float(row["requested_fraction"]), requested_fraction, abs_tol=1e-12)
        ]
        if len(depth_rows) != 1:
            raise RuntimeError("selected fraction did not map to exactly one depth")
        depth = int(depth_rows[0]["depth"])
        layer = layers[depth - 1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        batch_size = int(expected["batch_size"])
        vectors = capture_candidate_vectors(
            model,
            layer,
            depth,
            cases,
            tokenizer,
            int(pad_id),
            device,
            batch_size,
        )

        entries = []
        for alpha in ALPHAS:
            for curve in curves:
                entry = dict(curve)
                entry.update({
                    "schema_version": "phase1139_interpolation_record.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "split": "confirmation",
                    "depth": depth,
                    "relative_depth": depth / len(layers),
                    "requested_fraction": requested_fraction,
                    "alpha": alpha,
                    "record_id": f"{curve['curve_id']}|alpha={alpha:.2f}",
                })
                entries.append(entry)

        entries_per_batch = max(1, batch_size // 2)
        for alpha in ALPHAS:
            alpha_entries = [
                entry
                for entry in entries
                if math.isclose(float(entry["alpha"]), alpha, abs_tol=1e-12)
            ]
            for start in range(0, len(alpha_entries), entries_per_batch):
                batch_entries = alpha_entries[start : start + entries_per_batch]
                records.extend(score_interpolation_batch(
                    model,
                    layer,
                    batch_entries,
                    cases,
                    vectors,
                    tokenizer,
                    int(pad_id),
                    device,
                ))
            print(json.dumps({
                "phase": PHASE,
                "model": model_name,
                "alpha": alpha,
                "records": len(records),
            }), flush=True)

        if len(records) != EXPECTED_RECORDS:
            raise RuntimeError(f"record count drift: {len(records)}")
        core = {
            "schema_version": "phase1139_interpolation_run_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": placement,
            "layer_count": len(layers),
            "depth": depth,
            "relative_depth": depth / len(layers),
            "requested_fraction": requested_fraction,
            "item_count": len(selected_items),
            "curve_count": len(curves),
            "record_count": len(records),
            "finite_fraction": sum(bool(row["finite"]) for row in records) / len(records),
            "elapsed_seconds": time.time() - started,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "record_digest": digest(records),
            "evidence_scope": "same_family_confirmation_response_curve_only",
        }
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        write_jsonl(output_root / "records.jsonl", records)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": PHASE,
            "command": "run",
            "model": model_name,
            "records": len(records),
            "finite_fraction": summary["finite_fraction"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            if model_name == "qwen3_4b":
                release_fp16(model)
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def curve_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["curve_id"])].append(row)
    curves = []
    for curve_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        alpha_values = [float(row["alpha"]) for row in ordered]
        margins = [row["oriented_margin"] for row in ordered]
        finite = all(row["finite"] and value is not None for row, value in zip(ordered, margins))
        base = float(margins[0]) if finite else None
        endpoint = float(margins[-1]) if finite else None
        span = endpoint - base if finite else None
        valid_main = bool(
            finite
            and ordered[0]["curve_kind"] == "main"
            and base is not None
            and endpoint is not None
            and base < 0.0
            and endpoint > 0.0
            and span is not None
            and span > EPSILON
        )
        normalized = None
        linear_deviation = None
        max_step = None
        monotonic_fraction = None
        max_step_end_alpha = None
        if valid_main:
            normalized = [(float(value) - base) / span for value in margins]
            linear_deviation = statistics.median(
                abs(value - alpha)
                for value, alpha in zip(normalized[1:-1], alpha_values[1:-1])
            )
            steps = [normalized[index + 1] - normalized[index] for index in range(len(normalized) - 1)]
            max_index = max(range(len(steps)), key=lambda index: steps[index])
            max_step = steps[max_index]
            monotonic_fraction = sum(step >= -0.02 for step in steps) / len(steps)
            max_step_end_alpha = alpha_values[max_index + 1]
        identity = ordered[0]["identity_margin_drift"]
        curves.append({
            "curve_id": curve_id,
            "model": ordered[0]["model"],
            "item_id": ordered[0]["item_id"],
            "curve_kind": ordered[0]["curve_kind"],
            "panel": ordered[0]["panel"],
            "target_state": ordered[0]["target_state"],
            "source_state": ordered[0]["source_state"],
            "alphas": alpha_values,
            "margins": margins if finite else None,
            "finite": finite,
            "identity_margin_drift": identity,
            "endpoint_span": span,
            "valid_main": valid_main,
            "normalized_curve": normalized,
            "linear_deviation": linear_deviation,
            "max_step": max_step,
            "monotonic_fraction": monotonic_fraction,
            "max_step_end_alpha": max_step_end_alpha,
        })
    return curves


def summarize_model(model_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    curves = curve_metrics(records)
    main = [row for row in curves if row["curve_kind"] == "main"]
    valid_main = [row for row in main if row["valid_main"]]
    controls = [row for row in curves if row["curve_kind"] == "same_answer_temporal_control" and row["finite"]]
    identity = [
        abs(float(row["identity_margin_drift"]))
        for row in curves
        if row["identity_margin_drift"] is not None
    ]
    main_span = median(row["endpoint_span"] for row in valid_main)
    control_span = median(abs(float(row["endpoint_span"])) for row in controls if row["endpoint_span"] is not None)
    ratio = (
        main_span / max(control_span, EPSILON)
        if main_span is not None and control_span is not None
        else None
    )
    aggregate_curve = {
        str(alpha): median(
            row["normalized_curve"][index]
            for row in valid_main
            if row["normalized_curve"] is not None
        )
        for index, alpha in enumerate(ALPHAS)
    }
    intervals = Counter(row["max_step_end_alpha"] for row in valid_main)
    dominant_interval = None
    dominant_interval_fraction = None
    if intervals:
        dominant_interval, count = sorted(intervals.items(), key=lambda pair: (-pair[1], pair[0]))[0]
        dominant_interval_fraction = count / len(valid_main)

    metrics = {
        "model": model_name,
        "record_count": len(records),
        "curve_count": len(curves),
        "main_curve_count": len(main),
        "same_answer_control_curve_count": len(controls),
        "finite_fraction": sum(bool(row["finite"]) for row in records) / max(len(records), 1),
        "identity_max_abs_margin_drift": max(identity) if identity else None,
        "identity_median_abs_margin_drift": median(identity),
        "main_valid_fraction": len(valid_main) / max(len(main), 1),
        "endpoint_flip_fraction": len(valid_main) / max(len(main), 1),
        "main_endpoint_span_median": main_span,
        "same_answer_endpoint_abs_span_median": control_span,
        "main_to_same_answer_span_ratio": ratio,
        "median_linear_deviation": median(row["linear_deviation"] for row in valid_main),
        "median_max_step": median(row["max_step"] for row in valid_main),
        "median_monotonic_fraction": median(row["monotonic_fraction"] for row in valid_main),
        "dominant_max_step_end_alpha": dominant_interval,
        "dominant_max_step_interval_fraction": dominant_interval_fraction,
        "aggregate_normalized_curve": aggregate_curve,
    }
    qualified = bool(
        metrics["finite_fraction"] >= THRESHOLDS["finite_fraction"]
        and metrics["identity_max_abs_margin_drift"] is not None
        and metrics["identity_max_abs_margin_drift"] <= THRESHOLDS["identity_max_abs_margin_drift"]
        and metrics["main_valid_fraction"] >= THRESHOLDS["main_valid_fraction"]
        and metrics["endpoint_flip_fraction"] >= THRESHOLDS["endpoint_flip_fraction"]
        and metrics["median_monotonic_fraction"] is not None
        and metrics["median_monotonic_fraction"] >= THRESHOLDS["monotonic_fraction"]
        and metrics["main_to_same_answer_span_ratio"] is not None
        and metrics["main_to_same_answer_span_ratio"] >= THRESHOLDS["main_to_same_answer_span_ratio"]
    )
    metrics["qualified"] = qualified
    metrics["phase_like"] = bool(
        qualified
        and metrics["median_linear_deviation"] >= THRESHOLDS["phase_min_linear_deviation"]
        and metrics["median_max_step"] >= THRESHOLDS["phase_min_max_step"]
    )
    metrics["smooth"] = bool(
        qualified
        and metrics["median_linear_deviation"] <= THRESHOLDS["smooth_max_linear_deviation"]
        and metrics["median_max_step"] <= THRESHOLDS["smooth_max_step"]
    )
    metrics["curve_digest"] = digest(curves)
    return {"metrics": metrics, "curves": curves}


def finalize_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    model_results = {}
    all_curves = {}
    for model_name in MODELS:
        summary_path = OUT_ROOT / "runs" / model_name / "summary.json"
        records_path = OUT_ROOT / "runs" / model_name / "records.jsonl"
        if not summary_path.exists() or not records_path.exists():
            raise RuntimeError(f"missing Phase1139 output for {model_name}")
        summary = read_json(summary_path)
        records = read_jsonl(records_path)
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError("protocol digest drift")
        result = summarize_model(model_name, records)
        model_results[model_name] = result["metrics"]
        all_curves[model_name] = result["curves"]

    if not all(model_results[name]["qualified"] for name in MODELS):
        outcome = "instrument_failed_or_unqualified"
    elif all(model_results[name]["phase_like"] for name in MODELS):
        intervals = {model_results[name]["dominant_max_step_end_alpha"] for name in MODELS}
        outcome = "phase_like" if len(intervals) == 1 else "mixed"
    elif all(model_results[name]["smooth"] for name in MODELS):
        outcome = "smooth"
    else:
        outcome = "mixed"

    core = {
        "schema_version": "phase1139_matched_path_interpolation_final.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": model_results,
        "cross_endpoint": {
            "both_qualified": all(model_results[name]["qualified"] for name in MODELS),
            "dominant_intervals": {
                name: model_results[name]["dominant_max_step_end_alpha"] for name in MODELS
            },
            "outcome": outcome,
        },
        "phase_transition_confirmed": outcome == "phase_like",
        "smooth_interpolation_confirmed": outcome == "smooth",
        "attractor_claim_authorized": False,
        "component_scan_authorized": False,
        "phase1138_reopened": False,
        "auto_continue": False,
        "next_action": (
            "Use the response-curve result only to design a separately frozen independent-material tangent or "
            "component experiment; do not reuse this confirmation cohort."
        ),
        "claim_boundary": (
            "A phase-like outcome is only nonlinear downstream response along one counterfactual chord. "
            "It does not prove an attractor, manifold topology, semantic primitive, or local circuit."
        ),
    }
    final = dict(core)
    final["final_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/final.json", final)
    for model_name, curves in all_curves.items():
        write_jsonl(OUT_ROOT / "analysis" / f"curves.{model_name}.jsonl", curves)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize",
        "outcome": outcome,
        "phase_transition_confirmed": final["phase_transition_confirmed"],
        "smooth_interpolation_confirmed": final["smooth_interpolation_confirmed"],
        "auto_continue": final["auto_continue"],
        "final_digest": final["final_digest"],
    }, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("model", choices=MODELS)
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.model)
    elif args.command == "finalize":
        finalize_command()
    else:
        raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
