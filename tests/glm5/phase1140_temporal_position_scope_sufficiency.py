#!/usr/bin/env python3
"""Prospective position-scope sufficiency test for temporal binding.

The experiment uses behavior-qualified natural-use items that have never been
used for hidden-state intervention. At the frozen relative depth 0.7, it
compares two nested live-state interventions:

1. answer_boundary: patch only the state predicting the first candidate token;
2. candidate_prediction_span: patch every state used to score the candidate.

Both first-token and full-candidate margins are retained. This separates state
insufficiency from a mismatch between intervention position and readout span.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import quantization_audit, release_fp16  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1138_temporal_residual_onset as phase1138  # noqa: E402


PHASE = 1140
MODELS = ("qwen3_4b", "qwen3_14b")
SPLITS = ("discovery", "confirmation")
SCOPES = ("answer_boundary", "candidate_prediction_span")
ALPHAS = (0.0, 1.0)
REQUESTED_FRACTION = 0.7
COHORT_SIZE = 24
EXPECTED_CURVES = COHORT_SIZE * 6
EXPECTED_RECORDS = EXPECTED_CURVES * len(SCOPES) * len(ALPHAS)
MAIN_STATES = {"original_pre", "original_post", "swapped_pre", "swapped_post"}
PROPERTY_QUOTAS = {
    "P54": 8,
    "P286": 7,
    "P6": 4,
    "P488": 3,
    "P35": 1,
    "P169": 1,
}
OUT_ROOT = ROOT / "tests/glm5/result/phase1140_temporal_position_scope_sufficiency"
SOURCE1135 = ROOT / "tests/glm5/result/phase1135_temporal_binding_intervention"
SOURCE1137 = ROOT / "tests/glm5/result/phase1137_qwen14b_temporal_binding_endpoint"
SOURCE1138 = ROOT / "tests/glm5/result/phase1138_temporal_residual_onset"
SOURCE1139 = ROOT / "tests/glm5/result/phase1139_matched_path_residual_interpolation"
SOURCE_ITEMS = source.SOURCE
Q4_DECISIONS = SOURCE1135 / "analysis/behavior_decisions.qwen3.jsonl"
Q14_DECISIONS = SOURCE1137 / "analysis/behavior_decisions.qwen3_14b.jsonl"
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction": 0.99,
    "identity_max_abs_margin_drift": 0.005,
    "baseline_valid_fraction": 0.99,
    "main_endpoint_flip_fraction": 0.95,
    "panel_endpoint_flip_fraction": 0.95,
    "main_positive_change_fraction": 0.99,
    "main_to_same_answer_span_ratio": 2.0,
    "same_answer_control_flip_fraction": 0.10,
    "span_rescue_min_improvement": 0.15,
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


def stable_order(label: str, values: Iterable[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"phase1140|{label}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def behavior_all_four(path: Path) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        if row["split"] == "natural_use" and row["state"] in MAIN_STATES:
            grouped[str(row["item_id"])].append(row)
    return {
        item_id
        for item_id, rows in grouped.items()
        if len(rows) == 4
        and all(bool(row["finite"]) and row["correct"] is True for row in rows)
    }


def freeze_cohorts(
    items: list[dict[str, Any]],
    shared_ids: set[str],
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    item_index = {str(row["item_id"]): row for row in items}
    by_property: dict[str, list[str]] = defaultdict(list)
    for item_id in shared_ids:
        item = item_index[item_id]
        if item["split"] != "natural_use":
            raise RuntimeError("non-natural-use item entered the Phase1140 pool")
        by_property[str(item["property_id"])].append(item_id)

    discovery: list[str] = []
    confirmation: list[str] = []
    selection_rows = {}
    for property_id, quota in PROPERTY_QUOTAS.items():
        ordered = stable_order(property_id, by_property[property_id])
        if len(ordered) < quota * 2:
            raise RuntimeError(f"insufficient {property_id} items for frozen cohorts")
        discovery.extend(ordered[:quota])
        confirmation.extend(ordered[quota : quota * 2])
        selection_rows[property_id] = {
            "available": len(ordered),
            "per_cohort_quota": quota,
            "discovery": ordered[:quota],
            "confirmation": ordered[quota : quota * 2],
        }

    discovery = stable_order("discovery", discovery)
    confirmation = stable_order("confirmation", confirmation)
    used = set(discovery) | set(confirmation)
    reserve = stable_order("reserve", shared_ids - used)
    cohorts = {"discovery": discovery, "confirmation": confirmation}
    audit = {
        "selection_rows": selection_rows,
        "discovery_count": len(discovery),
        "confirmation_count": len(confirmation),
        "reserve_count": len(reserve),
        "disjoint": set(discovery).isdisjoint(confirmation),
        "all_from_shared_pool": used <= shared_ids,
    }
    return cohorts, reserve, audit


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1140 protocol after model output exists")

    items = read_jsonl(SOURCE_ITEMS)
    item_index = {str(row["item_id"]): row for row in items}
    q4_ids = behavior_all_four(Q4_DECISIONS)
    q14_ids = behavior_all_four(Q14_DECISIONS)
    shared_ids = q4_ids & q14_ids
    cohorts, reserve, cohort_audit = freeze_cohorts(items, shared_ids)

    prereg1138 = read_json(SOURCE1138 / "protocol/preregistration.json")
    cohort1138 = read_json(SOURCE1138 / "protocol/behavior_conditioned_cohorts.json")
    audit1139 = read_json(SOURCE1139 / "audit/independent_result_audit.json")
    final1139 = read_json(SOURCE1139 / "analysis/final.json")
    prior_hidden_ids = set(cohort1138["discovery"]["shared_item_ids"])
    prior_hidden_ids.update(cohort1138["confirmation"]["shared_item_ids"])
    selected_ids = set(cohorts["discovery"]) | set(cohorts["confirmation"])

    property_counts = {
        split: {
            property_id: sum(
                item_index[item_id]["property_id"] == property_id
                for item_id in item_ids
            )
            for property_id in PROPERTY_QUOTAS
        }
        for split, item_ids in cohorts.items()
    }
    checks = {
        "phase1139_audit_passed": bool(audit1139["all_checks_passed"]),
        "phase1139_auto_continue_was_false": final1139["auto_continue"] is False,
        "natural_use_shared_all_four_count_is_111": len(shared_ids) == 111,
        "cohort_sizes_are_24": all(len(ids) == COHORT_SIZE for ids in cohorts.values()),
        "cohorts_disjoint": cohort_audit["disjoint"],
        "cohorts_avoid_prior_hidden_items": selected_ids.isdisjoint(prior_hidden_ids),
        "reserve_has_63_items": len(reserve) == 63,
        "property_quotas_exact": all(
            property_counts[split] == PROPERTY_QUOTAS for split in SPLITS
        ),
        "frozen_depth_is_0_7": REQUESTED_FRACTION == 0.7,
        "nested_scopes_frozen": SCOPES == (
            "answer_boundary",
            "candidate_prediction_span",
        ),
        "alpha_endpoints_frozen": ALPHAS == (0.0, 1.0),
        "no_model_output_before_protocol": not (OUT_ROOT / "runs").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1140 protocol checks failed: {checks}")

    core = {
        "schema_version": "phase1140_temporal_position_scope_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "On a hidden-intervention-naive natural-use axis, determine whether Phase1139 endpoint "
            "insufficiency reflects an answer-boundary/readout-span mismatch by comparing nested "
            "answer-boundary and full candidate-prediction-span residual interventions."
        ),
        "epistemic_scope": (
            "Same-family FP16 causal sufficiency and measurement-alignment test only. The materials "
            "remain deterministic Wikidata templates with external-machine consensus labels."
        ),
        "source": {
            "items_sha256": sha256_file(SOURCE_ITEMS),
            "qwen4_decisions_sha256": sha256_file(Q4_DECISIONS),
            "qwen14_decisions_sha256": sha256_file(Q14_DECISIONS),
            "phase1138_protocol_digest": prereg1138["protocol_digest"],
            "phase1139_final_digest": final1139["final_digest"],
            "phase1139_audit_digest": audit1139["audit_digest"],
            "script_sha256": sha256_file(Path(__file__)),
            "labels": "external_machine_consensus_not_human_gold",
        },
        "models": prereg1138["models"],
        "material": {
            "split": "natural_use",
            "shared_behavior_all_four_pool_count": len(shared_ids),
            "shared_behavior_all_four_pool_digest": digest(sorted(shared_ids)),
            "cohorts": cohorts,
            "reserve_item_ids": reserve,
            "cohort_audit": cohort_audit,
            "property_quotas": PROPERTY_QUOTAS,
            "property_counts": property_counts,
            "hidden_intervention_naive": True,
        },
        "intervention": {
            "requested_fraction": REQUESTED_FRACTION,
            "position_scopes": list(SCOPES),
            "alphas": list(ALPHAS),
            "component": "whole residual stream after frozen layer",
            "target_state": "live state from the exact candidate-scoring forward pass",
            "source_state": "same-item, candidate-path-matched donor state",
            "answer_boundary": "patch only prompt_length-1",
            "candidate_prediction_span": (
                "patch prompt_length-1 through prompt_length+candidate_token_count-2"
            ),
            "first_token_and_full_candidate_ledgers": True,
            "curve_kinds": ["main", "same_answer_temporal_control"],
            "main_curves_per_item": 4,
            "control_curves_per_item": 2,
            "expected_curves_per_model_split": EXPECTED_CURVES,
            "expected_records_per_model_split": EXPECTED_RECORDS,
        },
        "thresholds": THRESHOLDS,
        "selection": {
            "minimal_nested_scope_rule": (
                "select answer_boundary if both discovery endpoints qualify; otherwise select "
                "candidate_prediction_span if both qualify; otherwise deny confirmation"
            ),
            "span_rescue_rule": (
                "selected candidate span plus at least 0.15 absolute endpoint-flip improvement "
                "over answer boundary in both endpoints"
            ),
        },
        "hard_stops": [
            "do not reuse Phase1138 or Phase1139 hidden cohorts",
            "do not change cohorts, depth, scopes, thresholds, or readout ledgers after output",
            "do not select a scope separately for each model",
            "do not drop multi-token candidates",
            "do not run attention, MLP, head, neuron, SAE, TDA, or Jacobian searches in Phase1140",
            "do not call first-token success full-candidate sufficiency",
            "do not call full-span success a semantic module",
            "a failed discovery denies confirmation",
        ],
        "auto_continue_rule": (
            "Only two-endpoint independent confirmation of one frozen minimal scope authorizes a "
            "separately frozen component-mediation phase on reserve items."
        ),
    }
    prereg = dict(core)
    prereg["protocol_digest"] = digest(core)
    audit_core = {
        "schema_version": "phase1140_protocol_audit.v1",
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
        "shared_pool": len(shared_ids),
        "cohorts": {key: len(value) for key, value in cohorts.items()},
        "reserve": len(reserve),
        "protocol_digest": prereg["protocol_digest"],
    }, ensure_ascii=False), flush=True)


class CandidatePathCapture:
    def __init__(self, layer):
        self.layer = layer
        self.positions: list[torch.Tensor] | None = None
        self.values: list[torch.Tensor] = []
        self.handle = None
        self.calls = 0

    def _hook(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or self.positions is None:
            raise RuntimeError("candidate path capture not initialized")
        self.values = [
            value[index, positions.to(value.device), :].detach().float().cpu()
            for index, positions in enumerate(self.positions)
        ]
        self.calls += 1
        return output

    def begin(self, positions: list[torch.Tensor]) -> None:
        self.positions = positions
        self.values = []
        self.calls = 0

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.positions = None
        self.values = []


class LivePathInterpolation:
    def __init__(
        self,
        layer,
        positions: list[torch.Tensor],
        sources: list[torch.Tensor],
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
            raise RuntimeError("path interpolation layer did not return a tensor")
        patched = value.clone()
        for index, (position_row, source_row) in enumerate(
            zip(self.positions, self.sources)
        ):
            positions = position_row.to(value.device)
            source_value = source_row.to(value.device, dtype=value.dtype)
            if source_value.shape[0] != positions.shape[0]:
                raise RuntimeError("source and target path lengths differ")
            live = value[index, positions, :]
            alpha = self.alphas[index].to(value.device, dtype=value.dtype)
            if float(alpha.item()) == 0.0:
                mixed = live
            else:
                mixed = live + alpha * (source_value - live)
            patched[index, positions, :] = mixed
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def prediction_positions(row: dict[str, Any], device: torch.device) -> torch.Tensor:
    start = int(row["prompt_length"]) - 1
    count = len(row["continuation_ids"])
    return torch.arange(start, start + count, dtype=torch.long, device=device)


def vector_key(case_id: str, candidate_key: str) -> str:
    return f"{case_id}|candidate={candidate_key}"


def capture_candidate_paths(
    model,
    layer,
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
    vectors: dict[str, torch.Tensor] = {}
    with CandidatePathCapture(layer) as capture:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                input_ids, attention_mask = source.pad_sequences(batch, pad_id, device)
                positions = [prediction_positions(row, device) for row in batch]
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                if capture.calls != 1 or len(capture.values) != len(batch):
                    raise RuntimeError("candidate path capture call drift")
                for index, row in enumerate(batch):
                    key = vector_key(str(row["case_id"]), str(row["candidate_key"]))
                    vectors[key] = capture.values[index].clone()
                del output, input_ids, attention_mask, positions
    return vectors


def score_token_paths(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_logits = []
    targets = []
    ownership = []
    for index, row in enumerate(rows):
        prompt_length = int(row["prompt_length"])
        for offset, token_id in enumerate(row["continuation_ids"]):
            selected_logits.append(logits[index, prompt_length - 1 + offset, :].float())
            targets.append(int(token_id))
            ownership.append(index)
    matrix = torch.stack(selected_logits)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=matrix.device)
    token_logp = -F.cross_entropy(matrix, target_tensor, reduction="none")
    grouped: list[list[float]] = [[] for _ in rows]
    for owner, value in zip(ownership, token_logp.detach().cpu().tolist()):
        grouped[owner].append(float(value))
    result = []
    for row, values in zip(rows, grouped):
        total = sum(values)
        mean = total / len(values)
        first = values[0]
        result.append({
            "token_count": len(values),
            "first_token_id": int(row["continuation_ids"][0]),
            "first_logp": first,
            "full_logp_mean": mean,
            "full_logp_sum": total,
            "finite": all(math.isfinite(value) for value in values),
        })
    del matrix, target_tensor, token_logp
    return result


def build_curves(item_ids: list[str]) -> list[dict[str, Any]]:
    curves = []

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


def score_intervention_batch(
    model,
    layer,
    entries: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    vectors: dict[str, torch.Tensor],
    tokenizer,
    pad_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    expanded = []
    owners = []
    source_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    alpha_rows = []
    for entry in entries:
        target_case = cases[str(entry["target_case_id"])]
        source_case = cases[str(entry["source_case_id"])]
        for candidate_key in ("old", "new"):
            target_row = source.tokenize_case(tokenizer, target_case, candidate_key)
            source_row = source.tokenize_case(tokenizer, source_case, candidate_key)
            if target_row["continuation_ids"] != source_row["continuation_ids"]:
                raise RuntimeError("same-item candidate tokenization changed across temporal states")
            all_positions = prediction_positions(target_row, device)
            donor = vectors[vector_key(str(entry["source_case_id"]), candidate_key)]
            if entry["scope"] == "answer_boundary":
                positions = all_positions[:1]
                donor = donor[:1]
            else:
                positions = all_positions
            expanded.append(target_row)
            owners.append(entry)
            source_rows.append(donor)
            position_rows.append(positions)
            alpha_rows.append(float(entry["alpha"]))

    input_ids, attention_mask = source.pad_sequences(expanded, pad_id, device)
    alphas = torch.tensor(alpha_rows, dtype=torch.float32, device=device)
    clean_scores = None
    with torch.inference_mode():
        if all(math.isclose(float(entry["alpha"]), 0.0, abs_tol=1e-12) for entry in entries):
            clean_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            clean_scores = score_token_paths(clean_output.logits, expanded)
            del clean_output
        with LivePathInterpolation(
            layer,
            position_rows,
            source_rows,
            alphas,
        ) as patch:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if patch.calls != 1:
            raise RuntimeError(f"path interpolation hook called {patch.calls} times")
        patched_scores = score_token_paths(output.logits, expanded)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    clean_grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, (row, entry, score_row) in enumerate(
        zip(expanded, owners, patched_scores)
    ):
        record_id = str(entry["record_id"])
        grouped[record_id][str(row["candidate_key"])] = score_row
        if clean_scores is not None:
            clean_grouped[record_id][str(row["candidate_key"])] = clean_scores[index]

    result = []
    for entry in entries:
        record_id = str(entry["record_id"])
        scores = grouped[record_id]
        base_key = str(entry["base_key"])
        desired_key = str(entry["desired_key"])
        finite = bool(scores[base_key]["finite"] and scores[desired_key]["finite"])
        first_informative = (
            scores["old"]["first_token_id"] != scores["new"]["first_token_id"]
        )
        full_margin = (
            scores[desired_key]["full_logp_mean"]
            - scores[base_key]["full_logp_mean"]
            if finite
            else None
        )
        first_margin = (
            scores[desired_key]["first_logp"] - scores[base_key]["first_logp"]
            if finite and first_informative
            else None
        )
        clean_full = None
        clean_first = None
        full_drift = None
        first_drift = None
        if record_id in clean_grouped:
            clean = clean_grouped[record_id]
            clean_finite = bool(clean[base_key]["finite"] and clean[desired_key]["finite"])
            if clean_finite:
                clean_full = (
                    clean[desired_key]["full_logp_mean"]
                    - clean[base_key]["full_logp_mean"]
                )
                full_drift = full_margin - clean_full if full_margin is not None else None
                if first_informative:
                    clean_first = (
                        clean[desired_key]["first_logp"]
                        - clean[base_key]["first_logp"]
                    )
                    first_drift = (
                        first_margin - clean_first if first_margin is not None else None
                    )
        row = dict(entry)
        row.update({
            "finite": finite,
            "first_token_informative": first_informative,
            "candidate_token_counts": {
                key: int(scores[key]["token_count"]) for key in ("old", "new")
            },
            "candidate_first_token_ids": {
                key: int(scores[key]["first_token_id"]) for key in ("old", "new")
            },
            "full_oriented_margin": full_margin,
            "first_oriented_margin": first_margin,
            "unhooked_full_oriented_margin": clean_full,
            "unhooked_first_oriented_margin": clean_first,
            "identity_full_margin_drift": full_drift,
            "identity_first_margin_drift": first_drift,
            "scores": scores if finite else None,
        })
        result.append(row)

    del output, input_ids, attention_mask, alphas, patched_scores
    return result


def curve_rows(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["scope"] == scope:
            grouped[str(row["curve_id"])].append(row)
    curves = []
    for curve_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != list(ALPHAS):
            raise RuntimeError("curve alpha grid drift")
        base, endpoint = ordered
        full_finite = bool(base["finite"] and endpoint["finite"])
        full_base = base["full_oriented_margin"]
        full_endpoint = endpoint["full_oriented_margin"]
        full_change = (
            float(full_endpoint) - float(full_base)
            if full_finite and full_base is not None and full_endpoint is not None
            else None
        )
        first_informative = bool(base["first_token_informative"])
        first_base = base["first_oriented_margin"]
        first_endpoint = endpoint["first_oriented_margin"]
        first_change = (
            float(first_endpoint) - float(first_base)
            if first_informative
            and first_base is not None
            and first_endpoint is not None
            else None
        )
        curves.append({
            "curve_id": curve_id,
            "model": base["model"],
            "split": base["split"],
            "scope": scope,
            "item_id": base["item_id"],
            "curve_kind": base["curve_kind"],
            "panel": base["panel"],
            "target_state": base["target_state"],
            "source_state": base["source_state"],
            "finite": full_finite,
            "identity_full_margin_drift": base["identity_full_margin_drift"],
            "identity_first_margin_drift": base["identity_first_margin_drift"],
            "first_token_informative": first_informative,
            "max_candidate_token_count": max(base["candidate_token_counts"].values()),
            "full_baseline_margin": full_base,
            "full_endpoint_margin": full_endpoint,
            "full_margin_change": full_change,
            "full_baseline_valid": bool(full_finite and full_base is not None and full_base < 0),
            "full_endpoint_flip": bool(full_finite and full_endpoint is not None and full_endpoint > 0),
            "full_positive_change": bool(full_change is not None and full_change > 0),
            "first_baseline_margin": first_base,
            "first_endpoint_margin": first_endpoint,
            "first_margin_change": first_change,
            "first_baseline_valid": bool(
                first_informative and first_base is not None and first_base < 0
            ),
            "first_endpoint_flip": bool(
                first_informative and first_endpoint is not None and first_endpoint > 0
            ),
        })
    return curves


def scope_metrics(
    model_name: str,
    split: str,
    records: list[dict[str, Any]],
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curves = curve_rows(records, scope)
    main = [row for row in curves if row["curve_kind"] == "main"]
    controls = [
        row for row in curves if row["curve_kind"] == "same_answer_temporal_control"
    ]
    identity_full = [
        abs(float(row["identity_full_margin_drift"]))
        for row in curves
        if row["identity_full_margin_drift"] is not None
    ]
    identity_first = [
        abs(float(row["identity_first_margin_drift"]))
        for row in curves
        if row["identity_first_margin_drift"] is not None
    ]
    main_span = median(row["full_margin_change"] for row in main)
    control_span = median(
        abs(float(row["full_margin_change"]))
        for row in controls
        if row["full_margin_change"] is not None
    )
    ratio = (
        main_span / max(control_span, EPSILON)
        if main_span is not None and control_span is not None
        else None
    )
    first_main = [row for row in main if row["first_token_informative"]]
    panel_flip = {
        panel: (
            sum(row["full_endpoint_flip"] for row in main if row["panel"] == panel)
            / max(sum(row["panel"] == panel for row in main), 1)
        )
        for panel in ("original", "swapped")
    }
    metrics = {
        "model": model_name,
        "split": split,
        "scope": scope,
        "record_count": sum(row["scope"] == scope for row in records),
        "curve_count": len(curves),
        "main_curve_count": len(main),
        "control_curve_count": len(controls),
        "finite_fraction": (
            sum(row["finite"] for row in records if row["scope"] == scope)
            / max(sum(row["scope"] == scope for row in records), 1)
        ),
        "identity_full_max_abs_margin_drift": max(identity_full) if identity_full else None,
        "identity_first_max_abs_margin_drift": max(identity_first) if identity_first else None,
        "full_baseline_valid_fraction": (
            sum(row["full_baseline_valid"] for row in main) / max(len(main), 1)
        ),
        "full_main_endpoint_flip_fraction": (
            sum(row["full_endpoint_flip"] for row in main) / max(len(main), 1)
        ),
        "full_original_endpoint_flip_fraction": panel_flip["original"],
        "full_swapped_endpoint_flip_fraction": panel_flip["swapped"],
        "full_main_positive_change_fraction": (
            sum(row["full_positive_change"] for row in main) / max(len(main), 1)
        ),
        "full_main_margin_change_median": main_span,
        "full_control_abs_margin_change_median": control_span,
        "full_main_to_control_span_ratio": ratio,
        "full_control_endpoint_flip_fraction": (
            sum(row["full_endpoint_flip"] for row in controls)
            / max(len(controls), 1)
        ),
        "first_informative_main_count": len(first_main),
        "first_baseline_valid_fraction": (
            sum(row["first_baseline_valid"] for row in first_main)
            / max(len(first_main), 1)
        ),
        "first_endpoint_flip_fraction": (
            sum(row["first_endpoint_flip"] for row in first_main)
            / max(len(first_main), 1)
        ),
        "first_positive_change_fraction": (
            sum(
                row["first_margin_change"] is not None
                and row["first_margin_change"] > 0
                for row in first_main
            )
            / max(len(first_main), 1)
        ),
    }
    metrics["qualified"] = bool(
        metrics["finite_fraction"] >= THRESHOLDS["finite_fraction"]
        and metrics["identity_full_max_abs_margin_drift"] is not None
        and metrics["identity_full_max_abs_margin_drift"]
        <= THRESHOLDS["identity_max_abs_margin_drift"]
        and (
            metrics["identity_first_max_abs_margin_drift"] is None
            or metrics["identity_first_max_abs_margin_drift"]
            <= THRESHOLDS["identity_max_abs_margin_drift"]
        )
        and metrics["full_baseline_valid_fraction"]
        >= THRESHOLDS["baseline_valid_fraction"]
        and metrics["full_main_endpoint_flip_fraction"]
        >= THRESHOLDS["main_endpoint_flip_fraction"]
        and metrics["full_original_endpoint_flip_fraction"]
        >= THRESHOLDS["panel_endpoint_flip_fraction"]
        and metrics["full_swapped_endpoint_flip_fraction"]
        >= THRESHOLDS["panel_endpoint_flip_fraction"]
        and metrics["full_main_positive_change_fraction"]
        >= THRESHOLDS["main_positive_change_fraction"]
        and metrics["full_main_to_control_span_ratio"] is not None
        and metrics["full_main_to_control_span_ratio"]
        >= THRESHOLDS["main_to_same_answer_span_ratio"]
        and metrics["full_control_endpoint_flip_fraction"]
        <= THRESHOLDS["same_answer_control_flip_fraction"]
    )
    metrics["curve_digest"] = digest(curves)
    return metrics, curves


def analyze_records(
    model_name: str,
    split: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metrics = {}
    curves = {}
    for scope in SCOPES:
        metrics[scope], curves[scope] = scope_metrics(
            model_name,
            split,
            records,
            scope,
        )
    metrics["span_minus_boundary_endpoint_flip"] = (
        metrics["candidate_prediction_span"]["full_main_endpoint_flip_fraction"]
        - metrics["answer_boundary"]["full_main_endpoint_flip_fraction"]
    )
    metrics["span_rescue"] = bool(
        metrics["candidate_prediction_span"]["qualified"]
        and not metrics["answer_boundary"]["qualified"]
        and metrics["span_minus_boundary_endpoint_flip"]
        >= THRESHOLDS["span_rescue_min_improvement"]
    )
    return metrics, curves


def run_command(model_name: str, split: str) -> None:
    if model_name not in MODELS or split not in SPLITS:
        raise RuntimeError("invalid Phase1140 endpoint")
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1140 protocol audit failed")
    if split == "confirmation":
        selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
        if not selection["confirmation_authorized"]:
            raise RuntimeError("Phase1140 discovery denied confirmation")
    output_root = OUT_ROOT / "runs" / split / model_name
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite {output_root}")

    selected_ids = list(prereg["material"]["cohorts"][split])
    items = read_jsonl(SOURCE_ITEMS)
    selected_items, cases = source.causal_cases(items, split, selected_ids)
    curves = build_curves(selected_ids)
    if len(curves) != EXPECTED_CURVES:
        raise RuntimeError("Phase1140 curve count drift")

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
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError(f"{model_name} FP16/no-quantization gate failed")

        layers = get_layers(model)
        depth_rows = [
            row
            for row in phase1138.depth_rows_for_model(len(layers))
            if math.isclose(
                float(row["requested_fraction"]),
                REQUESTED_FRACTION,
                abs_tol=1e-12,
            )
        ]
        if len(depth_rows) != 1:
            raise RuntimeError("frozen depth did not map to one layer")
        depth = int(depth_rows[0]["depth"])
        layer = layers[depth - 1]
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        batch_size = int(expected["batch_size"])
        vectors = capture_candidate_paths(
            model,
            layer,
            cases,
            tokenizer,
            int(pad_id),
            device,
            batch_size,
        )

        entries = []
        for scope in SCOPES:
            for alpha in ALPHAS:
                for curve in curves:
                    entry = dict(curve)
                    entry.update({
                        "schema_version": "phase1140_position_scope_record.v1",
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "depth": depth,
                        "relative_depth": depth / len(layers),
                        "requested_fraction": REQUESTED_FRACTION,
                        "scope": scope,
                        "alpha": alpha,
                        "record_id": (
                            f"{curve['curve_id']}|scope={scope}|alpha={alpha:.2f}"
                        ),
                    })
                    entries.append(entry)

        entries_per_batch = max(1, batch_size // 2)
        for scope in SCOPES:
            for alpha in ALPHAS:
                current = [
                    entry
                    for entry in entries
                    if entry["scope"] == scope
                    and math.isclose(float(entry["alpha"]), alpha, abs_tol=1e-12)
                ]
                for start in range(0, len(current), entries_per_batch):
                    records.extend(score_intervention_batch(
                        model,
                        layer,
                        current[start : start + entries_per_batch],
                        cases,
                        vectors,
                        tokenizer,
                        int(pad_id),
                        device,
                    ))
                print(json.dumps({
                    "phase": PHASE,
                    "model": model_name,
                    "split": split,
                    "scope": scope,
                    "alpha": alpha,
                    "records": len(records),
                }), flush=True)

        if len(records) != EXPECTED_RECORDS:
            raise RuntimeError(f"record count drift: {len(records)}")
        metrics, curves_by_scope = analyze_records(model_name, split, records)
        core = {
            "schema_version": "phase1140_position_scope_run_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": placement,
            "layer_count": len(layers),
            "depth": depth,
            "relative_depth": depth / len(layers),
            "requested_fraction": REQUESTED_FRACTION,
            "item_count": len(selected_items),
            "curve_count": len(curves),
            "record_count": len(records),
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "record_digest": digest(records),
            "evidence_scope": "same_family_hidden_intervention_naive_natural_use_axis",
        }
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        write_jsonl(output_root / "records.jsonl", records)
        write_json(output_root / "summary.json", summary)
        for scope, rows in curves_by_scope.items():
            write_jsonl(output_root / f"curves.{scope}.jsonl", rows)
        print(json.dumps({
            "phase": PHASE,
            "command": "run",
            "model": model_name,
            "split": split,
            "records": len(records),
            "boundary_flip": metrics["answer_boundary"]["full_main_endpoint_flip_fraction"],
            "span_flip": metrics["candidate_prediction_span"]["full_main_endpoint_flip_fraction"],
            "span_rescue": metrics["span_rescue"],
            "summary_digest": summary["summary_digest"],
        }), flush=True)
    finally:
        if model is not None:
            if model_name == "qwen3_4b":
                release_fp16(model)
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def select_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    summaries = {
        model_name: read_json(
            OUT_ROOT / "runs/discovery" / model_name / "summary.json"
        )
        for model_name in MODELS
    }
    for summary in summaries.values():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError("Phase1140 discovery protocol digest drift")
    boundary_all = all(
        summaries[model]["metrics"]["answer_boundary"]["qualified"]
        for model in MODELS
    )
    span_all = all(
        summaries[model]["metrics"]["candidate_prediction_span"]["qualified"]
        for model in MODELS
    )
    if boundary_all:
        selected_scope = "answer_boundary"
    elif span_all:
        selected_scope = "candidate_prediction_span"
    else:
        selected_scope = None
    span_rescue_all = all(
        summaries[model]["metrics"]["span_rescue"] for model in MODELS
    )
    core = {
        "schema_version": "phase1140_discovery_selection.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": {
            model: {
                "answer_boundary": summaries[model]["metrics"]["answer_boundary"],
                "candidate_prediction_span": summaries[model]["metrics"][
                    "candidate_prediction_span"
                ],
                "span_minus_boundary_endpoint_flip": summaries[model]["metrics"][
                    "span_minus_boundary_endpoint_flip"
                ],
                "span_rescue": summaries[model]["metrics"]["span_rescue"],
            }
            for model in MODELS
        },
        "both_boundary_qualified": boundary_all,
        "both_span_qualified": span_all,
        "span_rescue_both_endpoints": span_rescue_all,
        "selected_scope": selected_scope,
        "confirmation_authorized": selected_scope is not None,
        "selection_rule_followed": True,
    }
    result = dict(core)
    result["selection_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/discovery_selection.json", result)
    print(json.dumps({
        "phase": PHASE,
        "command": "select",
        "selected_scope": selected_scope,
        "confirmation_authorized": result["confirmation_authorized"],
        "span_rescue_both": span_rescue_all,
        "selection_digest": result["selection_digest"],
    }), flush=True)


def finalize_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    confirmation_summaries = {}
    if selection["confirmation_authorized"]:
        for model_name in MODELS:
            path = OUT_ROOT / "runs/confirmation" / model_name / "summary.json"
            if not path.exists():
                raise RuntimeError(f"missing confirmation output for {model_name}")
            confirmation_summaries[model_name] = read_json(path)
    selected_scope = selection["selected_scope"]
    confirmed = bool(
        selected_scope is not None
        and all(
            confirmation_summaries[model]["metrics"][selected_scope]["qualified"]
            for model in MODELS
        )
    )
    confirmation_span_rescue = bool(
        confirmed
        and selected_scope == "candidate_prediction_span"
        and all(
            confirmation_summaries[model]["metrics"]["span_rescue"]
            for model in MODELS
        )
    )
    if not selection["confirmation_authorized"]:
        outcome = "discovery_failed"
    elif confirmed and selected_scope == "answer_boundary":
        outcome = "answer_boundary_sufficiency_confirmed"
    elif confirmed and confirmation_span_rescue:
        outcome = "candidate_span_rescue_confirmed"
    elif confirmed:
        outcome = "candidate_span_sufficiency_confirmed_without_rescue"
    else:
        outcome = "confirmation_failed"
    component_authorized = confirmed
    core = {
        "schema_version": "phase1140_position_scope_final.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "selected_scope": selected_scope,
        "confirmation_run": bool(confirmation_summaries),
        "confirmation_models": {
            model: summary["metrics"]
            for model, summary in confirmation_summaries.items()
        },
        "minimal_scope_sufficiency_confirmed": confirmed,
        "candidate_span_rescue_confirmed": confirmation_span_rescue,
        "outcome": outcome,
        "component_mediation_authorized": component_authorized,
        "component_mediation_material": (
            "Phase1140 reserve items only" if component_authorized else None
        ),
        "cross_architecture_claim_authorized": False,
        "semantic_module_claim_authorized": False,
        "auto_continue": component_authorized,
        "claim_boundary": (
            "A pass identifies a same-family, readout-aligned sufficient whole-residual scope at "
            "one frozen depth. It is not necessity, component mediation, semantic identity, or "
            "cross-architecture conservation."
        ),
    }
    final = dict(core)
    final["final_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(json.dumps({
        "phase": PHASE,
        "command": "finalize",
        "outcome": outcome,
        "selected_scope": selected_scope,
        "confirmed": confirmed,
        "auto_continue": final["auto_continue"],
        "final_digest": final["final_digest"],
    }), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("model", choices=MODELS)
    run.add_argument("split", choices=SPLITS)
    sub.add_parser("select")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.model, args.split)
    elif args.command == "select":
        select_command()
    elif args.command == "finalize":
        finalize_command()
    else:
        raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
