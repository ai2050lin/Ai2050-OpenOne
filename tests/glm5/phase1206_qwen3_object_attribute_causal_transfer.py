#!/usr/bin/env python3
"""Qwen3-only causal transfer gate for the frozen Phase1205 residual event.

The target is fixed before this phase: residual depth 25 at the actual
generation boundary.  Bidirectional active-state differences are injected into
the opposite active state and compared with matched-panel, norm-matched random,
and zero controls.  Other roles/depths are descriptive localization controls.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
import phase1205_qwen3_object_attribute_vertical_closure as phase1205  # noqa: E402


PHASE = 1206
MODEL = "qwen3"
MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SOURCE1205 = ROOT / "tests/glm5/result/phase1205_qwen3_object_attribute_vertical_closure"
SOURCE1203 = ROOT / "tests/glm5/result/phase1203_object_attribute_behavior_protocol"
SOURCE_PAIR_MANIFEST = SOURCE1205 / "protocol/pair_manifest.jsonl"
SOURCE1205_FINAL = SOURCE1205 / "analysis/final.json"
SOURCE1205_AUDIT = SOURCE1205 / "audit/independent_result_audit.json"
SOURCE_MANIFEST = SOURCE1203 / "protocol/model_manifests/qwen3.jsonl"

OUT_ROOT = ROOT / "tests/glm5/result/phase1206_qwen3_object_attribute_causal_transfer"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/preexecution_audit.json"
VECTOR_PATH = OUT_ROOT / "runs/captured_vectors.npz"
RAW_PATH = OUT_ROOT / "runs/intervention_scores.jsonl.gz"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
VERDICT_PATH = OUT_ROOT / "analysis/causal_transfer_verdict.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

AUDIT_SCRIPT = TEST_ROOT / "phase1206_qwen3_object_attribute_causal_transfer_audit.py"
RUNNER_SCRIPT = TEST_ROOT / "phase1206_run_sequential.py"
EXPECTED_PHASE1205_FINAL_DIGEST = "c38805a360c1a2e627c599e37a8fdb3045a83b988279b950d448797514cc759b"
EXPECTED_PHASE1205_AUDIT_DIGEST = "b9cb907c851c2fee5ae97cd66896e77eceedde5e5b7d46f5f63e94ccf6a12e44"
TARGET_DEPTH = 25
PREBAND_DEPTH = 24
HIDDEN_SIZE = 2560
BATCH_PAIRS = 8
PATCH_BATCH_SIZE = 16
RANDOM_SEED = 12060031
TIE_TOLERANCE = 1e-6
EPSILON = 1e-8

CONDITIONS = (
    {"id": "active_target_full", "kind": "panel_delta", "panel": "active", "depth": 25, "role": "generation_boundary", "scale": 1.0, "evidence": "primary"},
    {"id": "active_target_half", "kind": "panel_delta", "panel": "active", "depth": 25, "role": "generation_boundary", "scale": 0.5, "evidence": "dose_descriptive"},
    {"id": "matched_null_target", "kind": "panel_delta", "panel": "matched_null", "depth": 25, "role": "generation_boundary", "scale": 1.0, "evidence": "primary_control"},
    {"id": "surface_only_target", "kind": "panel_delta", "panel": "surface_only", "depth": 25, "role": "generation_boundary", "scale": 1.0, "evidence": "primary_control"},
    {"id": "semantic_neighbor_target", "kind": "panel_delta", "panel": "semantic_neighbor", "depth": 25, "role": "generation_boundary", "scale": 1.0, "evidence": "primary_control"},
    {"id": "random_target_r0", "kind": "random", "random_index": 0, "depth": 25, "role": "generation_boundary", "evidence": "primary_control"},
    {"id": "random_target_r1", "kind": "random", "random_index": 1, "depth": 25, "role": "generation_boundary", "evidence": "primary_control"},
    {"id": "random_target_r2", "kind": "random", "random_index": 2, "depth": 25, "role": "generation_boundary", "evidence": "primary_control"},
    {"id": "random_target_r3", "kind": "random", "random_index": 3, "depth": 25, "role": "generation_boundary", "evidence": "primary_control"},
    {"id": "zero_target", "kind": "zero", "depth": 25, "role": "generation_boundary", "evidence": "identity_control"},
    {"id": "active_answer_prefix", "kind": "panel_delta", "panel": "active", "depth": 25, "role": "answer_prefix", "scale": 1.0, "evidence": "role_descriptive"},
    {"id": "active_query_value", "kind": "panel_delta", "panel": "active", "depth": 25, "role": "query_value", "scale": 1.0, "evidence": "role_descriptive"},
    {"id": "active_preband_generation", "kind": "panel_delta", "panel": "active", "depth": 24, "role": "generation_boundary", "scale": 1.0, "evidence": "depth_descriptive"},
)
PRIMARY_CONTROLS = (
    "matched_null_target",
    "surface_only_target",
    "semantic_neighbor_target",
    "random_target_r0",
    "random_target_r1",
    "random_target_r2",
    "random_target_r3",
    "zero_target",
)
SPLITS = phase1205.SPLITS
THRESHOLDS = {
    "finite_fraction": 1.0,
    "baseline_behavior_accuracy": 1.0,
    "full_donor_behavior_accuracy": 1.0,
    "positive_donor_margin_shift_fraction": 0.95,
    "donor_choice_fraction": 0.80,
    "minimum_median_transfer_fraction": 0.50,
    "active_beats_all_primary_controls_fraction": 0.75,
    "minimum_median_active_minus_max_control_shift": 0.10,
    "minimum_each_direction_donor_choice_fraction": 0.75,
    "zero_patch_max_abs_logit_drift": 1e-4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    os.replace(temporary, path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_embedded_digest(value: dict[str, Any], key: str) -> None:
    candidate = {name: item for name, item in value.items() if name != key}
    if digest(candidate) != value.get(key):
        raise RuntimeError(f"embedded digest mismatch: {key}")


def source_hashes() -> dict[str, str]:
    return {
        "main": sha256_file(Path(__file__).resolve()),
        "audit": sha256_file(AUDIT_SCRIPT),
        "runner": sha256_file(RUNNER_SCRIPT),
    }


def protocol_command() -> None:
    if PROTOCOL_PATH.exists() or (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1206 after protocol or model output exists")
    final1205 = read_json(SOURCE1205_FINAL)
    audit1205 = read_json(SOURCE1205_AUDIT)
    validate_embedded_digest(final1205, "final_digest")
    validate_embedded_digest(audit1205, "audit_digest")
    checks = {
        "phase1205_final_digest": final1205["final_digest"] == EXPECTED_PHASE1205_FINAL_DIGEST,
        "phase1205_audit_digest": audit1205["audit_digest"] == EXPECTED_PHASE1205_AUDIT_DIGEST,
        "phase1205_audit_pass": audit1205["gate_pass"] is True,
        "phase1205_hidden_gate": final1205["hidden_specificity_gate"] is True,
        "phase1205_selected_depth": final1205["selected_depth"] == TARGET_DEPTH,
        "phase1205_causal_prereg_authorized": final1205["authorized_next"]["phase1206_qwen3_causal_preregistration"] is True,
        "phase1205_cross_claim_denied": final1205["authorized_next"]["cross_model_claim"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1206 upstream checks failed: {checks}")
    pairs = read_jsonl(SOURCE_PAIR_MANIFEST)
    active_pairs = [row for row in pairs if row["panel"] == "active"]
    if len(active_pairs) * 4 != len(pairs):
        raise RuntimeError("Phase1206 source quartet drift")
    expected_records = len(active_pairs) * 2 * len(CONDITIONS)
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1206.qwen3_object_attribute_causal_transfer.v1",
        "created_at": utc_now(),
        "objective": (
            "Test whether the frozen Qwen3 residual depth-25 generation-boundary active-state difference "
            "causally transfers the counterfactual object-attribute answer beyond matched and random controls."
        ),
        "scope": {
            "model": MODEL,
            "model_specific_only": True,
            "causal_transfer_claim": True,
            "causal_necessity_claim": False,
            "natural_use_claim": False,
            "cross_model_claim": False,
            "mechanism_closure_claim": False,
        },
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1205_final_digest": final1205["final_digest"],
            "phase1205_audit_digest": audit1205["audit_digest"],
            "pair_manifest_file_sha256": sha256_file(SOURCE_PAIR_MANIFEST),
            "pair_manifest_digest": digest(pairs),
            "qwen_manifest_file_sha256": sha256_file(SOURCE_MANIFEST),
            "eligible_group_count": len(active_pairs),
        },
        "model": {
            "path": str(MODEL_PATH.resolve()),
            "precision": "FP16",
            "quantization": "none",
            "placement": "full_cuda",
            "hidden_size": HIDDEN_SIZE,
            "capture_batch_pairs": BATCH_PAIRS,
            "patch_batch_size": PATCH_BATCH_SIZE,
            "logits_to_keep": 1,
        },
        "target": {
            "source": "Phase1205 earliest depth of earliest preregistered qualifying discovery band",
            "depth": TARGET_DEPTH,
            "role": "generation_boundary",
            "component": "residual",
            "not_refit_in_phase1206": True,
        },
        "conditions": list(CONDITIONS),
        "primary_controls": list(PRIMARY_CONTROLS),
        "random_control": {
            "seed": RANDOM_SEED,
            "count": 4,
            "norm": "per-group active target delta norm",
            "antisymmetric_between_directions": True,
        },
        "estimands": {
            "donor_margin": "logit(donor_gold)-logit(recipient_gold)",
            "donor_margin_shift": "patched donor margin minus unhooked recipient donor margin",
            "transfer_fraction": "donor_margin_shift divided by the full unhooked donor-minus-recipient margin change",
            "counterfactual_choice": "patched argmax over the three frozen candidates equals donor_gold",
        },
        "primary_gate": {
            "condition": "active_target_full",
            "splits": list(SPLITS),
            "thresholds": THRESHOLDS,
            "rule": "all split gates and the zero-patch identity audit must pass without target or threshold refitting",
            "role_and_depth_controls_are_descriptive": True,
        },
        "counts": {
            "active_pairs": len(active_pairs),
            "directions_per_pair": 2,
            "conditions": len(CONDITIONS),
            "expected_intervention_records": expected_records,
        },
        "authorization": {
            "run_after_independent_zero_output_audit": True,
            "head_or_neuron_search": False,
            "automatic_phase1207_execution": False,
            "phase1207_necessity_rescue_prereg_if_pass": True,
        },
        "stop_rules": [
            "If preexecution audit fails, do not load Qwen3.",
            "If any split causal-transfer gate fails, deny necessity/rescue target selection.",
            "A positive result establishes Qwen3-specific controlled causal transfer only, not natural necessity or full closure.",
            "No result may weaken K184 or create a cross-model claim.",
        ],
        "upstream_checks": checks,
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "active_pairs": len(active_pairs),
        "conditions": len(CONDITIONS),
        "expected_records": expected_records,
    }, ensure_ascii=False, indent=2))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_embedded_digest(protocol, "protocol_digest")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("Phase1206 source hash drift")
    if sha256_file(SOURCE_PAIR_MANIFEST) != protocol["upstream"]["pair_manifest_file_sha256"]:
        raise RuntimeError("Phase1206 pair manifest drift")
    if digest(read_jsonl(SOURCE_PAIR_MANIFEST)) != protocol["upstream"]["pair_manifest_digest"]:
        raise RuntimeError("Phase1206 pair manifest semantic drift")
    return protocol


class CaptureSelectedResiduals:
    def __init__(self, layers: list[Any]):
        self.layers = layers
        self.positions: dict[str, torch.Tensor] = {}
        self.values: dict[str, torch.Tensor] = {}
        self.calls: dict[str, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _hook(self, depth: int):
        def hook(module: Any, args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("capture layer did not return tensor")
            batch = torch.arange(value.shape[0], device=value.device)
            roles = ("generation_boundary",) if depth == PREBAND_DEPTH else (
                "generation_boundary", "answer_prefix", "query_value"
            )
            for role in roles:
                positions = self.positions[role].to(value.device)
                self.values[f"d{depth}_{role}"] = value[batch, positions, :].detach()
            self.calls[f"d{depth}"] += 1
            return output
        return hook

    def register(self) -> None:
        self.handles.append(self.layers[PREBAND_DEPTH - 1].register_forward_hook(self._hook(PREBAND_DEPTH)))
        self.handles.append(self.layers[TARGET_DEPTH - 1].register_forward_hook(self._hook(TARGET_DEPTH)))

    def begin(self, positions: dict[str, torch.Tensor]) -> None:
        self.positions = positions
        self.values = {}
        self.calls = defaultdict(int)

    def validate(self) -> None:
        expected = {
            "d24_generation_boundary",
            "d25_generation_boundary",
            "d25_answer_prefix",
            "d25_query_value",
        }
        if set(self.values) != expected or self.calls != {"d24": 1, "d25": 1}:
            raise RuntimeError(f"capture drift values={set(self.values)} calls={dict(self.calls)}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class DeltaPatch:
    def __init__(self, layer: Any, positions: torch.Tensor, deltas: torch.Tensor):
        self.layer = layer
        self.positions = positions
        self.deltas = deltas
        self.handle = None
        self.calls = 0

    def _hook(self, module: Any, args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("patch layer did not return tensor")
        positions = self.positions.to(value.device)
        deltas = self.deltas.to(value.device, dtype=value.dtype)
        batch = torch.arange(value.shape[0], device=value.device)
        patched = value.clone()
        patched[batch, positions, :] = value[batch, positions, :] + deltas
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


def placement_audit(model: Any) -> dict[str, Any]:
    devices = {str(parameter.device) for parameter in model.parameters()}
    return {
        "placement": "full_cuda" if devices == {"cuda:0"} else "mixed",
        "devices": sorted(devices),
        "quantization": "none",
    }


def score_batch(logits: torch.Tensor, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    last = logits[:, -1, :].float()
    all_finite = torch.isfinite(last).all(dim=-1).detach().cpu().numpy().astype(np.bool_)
    scores = np.empty((len(rows), 3), dtype=np.float32)
    for index, row in enumerate(rows):
        labels = list(row["entities"])
        token_ids = [int(row["manifest"]["candidate_token_ids"][label][0]) for label in labels]
        scores[index] = last[index, token_ids].detach().cpu().numpy()
    return scores, all_finite


def prediction(labels: list[str], scores: np.ndarray, finite: bool) -> str:
    if not finite or not np.isfinite(scores).all():
        return "NONFINITE"
    order = np.argsort(-scores, kind="stable")
    if float(scores[order[0]] - scores[order[1]]) <= TIE_TOLERANCE:
        return "UNRESOLVED_TIE"
    return labels[int(order[0])]


def random_delta(group_id: str, random_index: int, norm: float, direction: int) -> np.ndarray:
    seed_text = f"{RANDOM_SEED}|{group_id}|{random_index}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(HIDDEN_SIZE).astype(np.float32)
    vector /= float(np.linalg.norm(vector)) + EPSILON
    return vector * float(norm) * float(direction)


def run_command() -> None:
    protocol = verify_protocol()
    if VECTOR_PATH.exists() or RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1206 model output already exists")
    preaudit = read_json(PREAUDIT_PATH)
    validate_embedded_digest(preaudit, "audit_digest")
    if not preaudit.get("gate_pass") or preaudit.get("protocol_digest") != protocol["protocol_digest"]:
        raise RuntimeError("Phase1206 preexecution audit did not authorize run")

    pairs = read_jsonl(SOURCE_PAIR_MANIFEST)
    manifest_rows = read_jsonl(SOURCE_MANIFEST)
    manifest = {str(row["item_id"]): row for row in manifest_rows}
    panel_by_group = {
        (str(row["group_id"]), str(row["panel"])): row for row in pairs
    }
    active_pairs = [row for row in pairs if row["panel"] == "active"]
    pair_count = len(pairs)
    vector_keys = (
        "d24_generation_boundary",
        "d25_generation_boundary",
        "d25_answer_prefix",
        "d25_query_value",
    )
    vectors = {
        key: np.empty((pair_count, 2, HIDDEN_SIZE), dtype=np.float16)
        for key in vector_keys
    }
    baseline_scores = np.empty((pair_count, 2, 3), dtype=np.float32)
    baseline_finite = np.zeros((pair_count, 2), dtype=np.bool_)
    started = time.time()
    model = None
    capture = None
    try:
        model, tokenizer, device, precision = load_fp16(MODEL, MODEL_PATH)
        precision = quantization_audit(model)
        placement = placement_audit(model)
        if not (
            precision["has_fp16_parameters"]
            and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"]
            and set(precision["parameter_dtypes"]) == {"float16"}
            and placement["placement"] == "full_cuda"
        ):
            raise RuntimeError(f"Phase1206 precision/placement failed: {precision} {placement}")
        layers = list(get_layers(model))
        if len(layers) != 36:
            raise RuntimeError(f"Phase1206 layer count drift: {len(layers)}")
        capture = CaptureSelectedResiduals(layers)
        capture.register()
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_length[int(pair["input_length"])].append(pair)
        completed = 0
        with torch.inference_mode():
            for length in sorted(by_length):
                members = sorted(by_length[length], key=lambda row: int(row["pair_index"]))
                for start in range(0, len(members), BATCH_PAIRS):
                    batch_pairs = members[start : start + BATCH_PAIRS]
                    rows: list[dict[str, Any]] = []
                    input_rows: list[list[int]] = []
                    positions = {role: [] for role in ("generation_boundary", "answer_prefix", "query_value")}
                    for pair in batch_pairs:
                        for state in (0, 1):
                            item = manifest[str(pair[f"state{state}_item_id"])]
                            rows.append({"pair": pair, "state": state, "manifest": item, "entities": list(pair["entities"])})
                            input_rows.append(list(item["input_ids"]))
                            for role in positions:
                                positions[role].append(int(pair[f"state{state}_positions"][role]))
                    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    capture.begin({
                        role: torch.tensor(values, dtype=torch.long, device=device)
                        for role, values in positions.items()
                    })
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                    capture.validate()
                    scores, finite = score_batch(output.logits, rows)
                    for slot, row in enumerate(rows):
                        pair_index = int(row["pair"]["pair_index"])
                        state = int(row["state"])
                        baseline_scores[pair_index, state] = scores[slot]
                        baseline_finite[pair_index, state] = finite[slot]
                        for key in vector_keys:
                            vectors[key][pair_index, state] = capture.values[key][slot].cpu().numpy()
                    completed += len(batch_pairs)
                    del output, input_ids, attention_mask
                print(json.dumps({"phase": PHASE, "capture_length": length, "completed_pairs": completed}, separators=(",", ":")), flush=True)
        capture.close()
        capture = None

        baseline_correct: list[bool] = []
        for pair in pairs:
            labels = list(pair["entities"])
            for state in (0, 1):
                index = int(pair["pair_index"])
                pred = prediction(labels, baseline_scores[index, state], bool(baseline_finite[index, state]))
                baseline_correct.append(pred == pair[f"state{state}_gold"])
        if not all(baseline_correct) or not bool(baseline_finite.all()):
            raise RuntimeError("Phase1206 baseline behavior drifted")

        records: list[dict[str, Any]] = []
        active_by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in active_pairs:
            active_by_length[int(pair["input_length"])].append(pair)
        for condition in CONDITIONS:
            depth = int(condition["depth"])
            role = str(condition["role"])
            vector_key = f"d{depth}_{role}"
            for length in sorted(active_by_length):
                entries: list[dict[str, Any]] = []
                for pair in sorted(active_by_length[length], key=lambda row: int(row["pair_index"])):
                    for recipient_state in (0, 1):
                        entries.append({"pair": pair, "recipient_state": recipient_state, "donor_state": 1 - recipient_state})
                for start in range(0, len(entries), PATCH_BATCH_SIZE):
                    batch_entries = entries[start : start + PATCH_BATCH_SIZE]
                    rows: list[dict[str, Any]] = []
                    input_rows: list[list[int]] = []
                    patch_positions: list[int] = []
                    deltas: list[np.ndarray] = []
                    for entry in batch_entries:
                        pair = entry["pair"]
                        recipient_state = int(entry["recipient_state"])
                        donor_state = int(entry["donor_state"])
                        item = manifest[str(pair[f"state{recipient_state}_item_id"])]
                        rows.append({"pair": pair, "state": recipient_state, "manifest": item, "entities": list(pair["entities"])})
                        input_rows.append(list(item["input_ids"]))
                        patch_positions.append(int(pair[f"state{recipient_state}_positions"][role]))
                        if condition["kind"] == "zero":
                            delta = np.zeros(HIDDEN_SIZE, dtype=np.float32)
                        elif condition["kind"] == "random":
                            active_index = int(pair["pair_index"])
                            active_delta = (
                                vectors["d25_generation_boundary"][active_index, 1].astype(np.float32)
                                - vectors["d25_generation_boundary"][active_index, 0].astype(np.float32)
                            )
                            sign = 1 if recipient_state == 0 else -1
                            delta = random_delta(
                                str(pair["group_id"]),
                                int(condition["random_index"]),
                                float(np.linalg.norm(active_delta)),
                                sign,
                            )
                        else:
                            donor_pair = panel_by_group[(str(pair["group_id"]), str(condition["panel"]))]
                            donor_index = int(donor_pair["pair_index"])
                            delta = (
                                vectors[vector_key][donor_index, donor_state].astype(np.float32)
                                - vectors[vector_key][donor_index, recipient_state].astype(np.float32)
                            ) * float(condition.get("scale", 1.0))
                        deltas.append(delta)
                    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    position_tensor = torch.tensor(patch_positions, dtype=torch.long, device=device)
                    delta_tensor = torch.from_numpy(np.stack(deltas)).to(device=device, dtype=torch.float16)
                    with DeltaPatch(layers[depth - 1], position_tensor, delta_tensor) as patch:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                            logits_to_keep=1,
                        )
                    if patch.calls != 1:
                        raise RuntimeError(f"Phase1206 patch call drift: {patch.calls}")
                    patched_scores, patched_finite = score_batch(output.logits, rows)
                    for slot, entry in enumerate(batch_entries):
                        pair = entry["pair"]
                        pair_index = int(pair["pair_index"])
                        recipient_state = int(entry["recipient_state"])
                        donor_state = int(entry["donor_state"])
                        labels = list(pair["entities"])
                        records.append({
                            "record_id": f"{pair['group_id']}|r{recipient_state}d{donor_state}|{condition['id']}",
                            "group_id": str(pair["group_id"]),
                            "split": str(pair["split"]),
                            "world": str(pair["world"]),
                            "attribute": str(pair["attribute"]),
                            "template": str(pair["template"]),
                            "candidate_order": int(pair["candidate_order"]),
                            "recipient_state": recipient_state,
                            "donor_state": donor_state,
                            "condition": str(condition["id"]),
                            "condition_evidence": str(condition["evidence"]),
                            "depth": depth,
                            "role": role,
                            "candidate_labels": labels,
                            "recipient_gold": str(pair[f"state{recipient_state}_gold"]),
                            "donor_gold": str(pair[f"state{donor_state}_gold"]),
                            "recipient_unhooked_scores": [float(value) for value in baseline_scores[pair_index, recipient_state]],
                            "donor_unhooked_scores": [float(value) for value in baseline_scores[pair_index, donor_state]],
                            "patched_scores": [float(value) for value in patched_scores[slot]],
                            "recipient_unhooked_finite": bool(baseline_finite[pair_index, recipient_state]),
                            "donor_unhooked_finite": bool(baseline_finite[pair_index, donor_state]),
                            "patched_finite": bool(patched_finite[slot]),
                            "recipient_unhooked_prediction": prediction(labels, baseline_scores[pair_index, recipient_state], bool(baseline_finite[pair_index, recipient_state])),
                            "donor_unhooked_prediction": prediction(labels, baseline_scores[pair_index, donor_state], bool(baseline_finite[pair_index, donor_state])),
                            "patched_prediction": prediction(labels, patched_scores[slot], bool(patched_finite[slot])),
                            "delta_l2": float(np.linalg.norm(deltas[slot])),
                        })
                    del output, input_ids, attention_mask, position_tensor, delta_tensor
            print(json.dumps({"phase": PHASE, "completed_condition": condition["id"], "records": len(records)}, separators=(",", ":")), flush=True)

        if len(records) != protocol["counts"]["expected_intervention_records"]:
            raise RuntimeError("Phase1206 record count drift")
        if not all(row["patched_finite"] for row in records):
            raise RuntimeError("Phase1206 nonfinite patched logits")
        write_npz_atomic(
            VECTOR_PATH,
            **vectors,
            baseline_scores=baseline_scores,
            baseline_finite=baseline_finite,
        )
        write_jsonl_gz(RAW_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1206.qwen3_causal_transfer_run.v1",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "preexecution_audit_digest": preaudit["audit_digest"],
            "pair_count": pair_count,
            "active_pair_count": len(active_pairs),
            "condition_count": len(CONDITIONS),
            "record_count": len(records),
            "vector_shapes": {key: list(value.shape) for key, value in vectors.items()},
            "baseline_scores_shape": list(baseline_scores.shape),
            "vector_file_sha256": sha256_file(VECTOR_PATH),
            "raw_file_sha256": sha256_file(RAW_PATH),
            "raw_digest": digest(records),
            "baseline_behavior_accuracy": sum(baseline_correct) / len(baseline_correct),
            "baseline_finite_fraction": float(baseline_finite.mean()),
            "patched_finite_fraction": sum(bool(row["patched_finite"]) for row in records) / len(records),
            "precision_audit": precision,
            "placement": placement,
            "runtime": {
                "elapsed_seconds": time.time() - started,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "claim_boundary": protocol["scope"],
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(json.dumps({
            "phase": PHASE,
            "records": len(records),
            "baseline_accuracy": summary["baseline_behavior_accuracy"],
            "elapsed_seconds": summary["runtime"]["elapsed_seconds"],
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False, indent=2))
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)
        gc.collect()


def median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(np.median(np.asarray(items, dtype=np.float64))) if items else 0.0


def enriched_record(row: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    base_index = labels.index(str(row["recipient_gold"]))
    donor_index = labels.index(str(row["donor_gold"]))
    recipient = np.asarray(row["recipient_unhooked_scores"], dtype=np.float64)
    donor = np.asarray(row["donor_unhooked_scores"], dtype=np.float64)
    patched = np.asarray(row["patched_scores"], dtype=np.float64)
    base_margin = float(recipient[donor_index] - recipient[base_index])
    donor_margin = float(donor[donor_index] - donor[base_index])
    patched_margin = float(patched[donor_index] - patched[base_index])
    shift = patched_margin - base_margin
    full_shift = donor_margin - base_margin
    return {
        **row,
        "recipient_donor_margin": base_margin,
        "full_donor_margin": donor_margin,
        "patched_donor_margin": patched_margin,
        "donor_margin_shift": shift,
        "full_unhooked_margin_shift": full_shift,
        "transfer_fraction": shift / (full_shift + EPSILON),
        "positive_shift": shift > 0,
        "donor_choice": row["patched_prediction"] == row["donor_gold"],
        "recipient_correct": row["recipient_unhooked_prediction"] == row["recipient_gold"],
        "donor_correct": row["donor_unhooked_prediction"] == row["donor_gold"],
    }


def compute_split_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split]
    by_key = {(row["group_id"], row["recipient_state"], row["condition"]): row for row in members}
    targets = [row for row in members if row["condition"] == "active_target_full"]
    advantages: list[float] = []
    target_shifts: list[float] = []
    transfer: list[float] = []
    each_direction: dict[str, float] = {}
    for row in targets:
        key_base = (row["group_id"], row["recipient_state"])
        control_shifts = [by_key[(*key_base, control)]["donor_margin_shift"] for control in PRIMARY_CONTROLS]
        advantages.append(float(row["donor_margin_shift"]) - max(float(value) for value in control_shifts))
        target_shifts.append(float(row["donor_margin_shift"]))
        transfer.append(float(row["transfer_fraction"]))
    for state in (0, 1):
        direction_rows = [row for row in targets if int(row["recipient_state"]) == state]
        each_direction[f"state{state}_to_state{1-state}"] = (
            sum(bool(row["donor_choice"]) for row in direction_rows) / max(len(direction_rows), 1)
        )
    finite = [
        bool(row["recipient_unhooked_finite"] and row["donor_unhooked_finite"] and row["patched_finite"])
        for row in targets
    ]
    result = {
        "split": split,
        "target_record_count": len(targets),
        "finite_fraction": sum(finite) / max(len(finite), 1),
        "baseline_behavior_accuracy": sum(bool(row["recipient_correct"]) for row in targets) / max(len(targets), 1),
        "full_donor_behavior_accuracy": sum(bool(row["donor_correct"]) for row in targets) / max(len(targets), 1),
        "positive_donor_margin_shift_fraction": sum(bool(row["positive_shift"]) for row in targets) / max(len(targets), 1),
        "donor_choice_fraction": sum(bool(row["donor_choice"]) for row in targets) / max(len(targets), 1),
        "median_donor_margin_shift": median(target_shifts),
        "median_transfer_fraction": median(transfer),
        "active_beats_all_primary_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_active_minus_max_control_shift": median(advantages),
        "direction_donor_choice_fraction": each_direction,
    }
    result["pass"] = bool(
        result["finite_fraction"] >= THRESHOLDS["finite_fraction"]
        and result["baseline_behavior_accuracy"] >= THRESHOLDS["baseline_behavior_accuracy"]
        and result["full_donor_behavior_accuracy"] >= THRESHOLDS["full_donor_behavior_accuracy"]
        and result["positive_donor_margin_shift_fraction"] >= THRESHOLDS["positive_donor_margin_shift_fraction"]
        and result["donor_choice_fraction"] >= THRESHOLDS["donor_choice_fraction"]
        and result["median_transfer_fraction"] >= THRESHOLDS["minimum_median_transfer_fraction"]
        and result["active_beats_all_primary_controls_fraction"] >= THRESHOLDS["active_beats_all_primary_controls_fraction"]
        and result["median_active_minus_max_control_shift"] >= THRESHOLDS["minimum_median_active_minus_max_control_shift"]
        and min(result["direction_donor_choice_fraction"].values()) >= THRESHOLDS["minimum_each_direction_donor_choice_fraction"]
    )
    return result


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in (row["id"] for row in CONDITIONS):
        members = [row for row in rows if row["condition"] == condition]
        output[str(condition)] = {
            "count": len(members),
            "median_donor_margin_shift": median(row["donor_margin_shift"] for row in members),
            "median_transfer_fraction": median(row["transfer_fraction"] for row in members),
            "positive_shift_fraction": sum(bool(row["positive_shift"]) for row in members) / max(len(members), 1),
            "donor_choice_fraction": sum(bool(row["donor_choice"]) for row in members) / max(len(members), 1),
            "median_delta_l2": median(row["delta_l2"] for row in members),
        }
    return output


def analyze_command() -> None:
    protocol = verify_protocol()
    if VERDICT_PATH.exists() or RESULT_AUDIT_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("Phase1206 analysis output already exists")
    summary = read_json(RUN_SUMMARY_PATH)
    validate_embedded_digest(summary, "summary_digest")
    if sha256_file(RAW_PATH) != summary["raw_file_sha256"]:
        raise RuntimeError("Phase1206 raw file drift")
    raw = read_jsonl_gz(RAW_PATH)
    if digest(raw) != summary["raw_digest"]:
        raise RuntimeError("Phase1206 raw semantic drift")
    rows = [enriched_record(row) for row in raw]
    split_metrics = {split: compute_split_metrics(rows, split) for split in SPLITS}
    zero_rows = [row for row in rows if row["condition"] == "zero_target"]
    zero_max_drift = max(
        abs(float(patched) - float(base))
        for row in zero_rows
        for patched, base in zip(row["patched_scores"], row["recipient_unhooked_scores"])
    )
    identity_pass = zero_max_drift <= THRESHOLDS["zero_patch_max_abs_logit_drift"]
    causal_gate = bool(identity_pass and all(metrics["pass"] for metrics in split_metrics.values()))
    status = "qwen3_controlled_causal_transfer_qualified" if causal_gate else "qwen3_controlled_causal_transfer_not_qualified"
    k_statement = (
        "At the Phase1205-frozen Qwen3 depth-25 generation-boundary residual event, the bidirectional active-state "
        "difference causally transfers the counterfactual candidate beyond matched-panel and norm-matched random controls "
        "across all three splits; natural necessity and full closure remain untested."
        if causal_gate else
        "The Phase1205-frozen Qwen3 residual event did not satisfy the preregistered bidirectional causal-transfer, "
        "matched-control, split-repetition, and identity gates; necessity/rescue localization is denied."
    )
    verdict: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1206.qwen3_causal_transfer_verdict.v1",
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_summary_digest": summary["summary_digest"],
        "split_metrics": split_metrics,
        "zero_patch_max_abs_logit_drift": zero_max_drift,
        "zero_patch_identity_pass": identity_pass,
        "condition_summary": condition_summary(rows),
        "causal_transfer_gate": causal_gate,
        "status": status,
        "proposed_k_item_pending_independent_audit": {
            "id": "K186",
            "scope": "Qwen3-specific controlled causal transfer",
            "statement": k_statement,
        },
        "claim_boundary": {
            "qwen3_model_specific": True,
            "controlled_causal_transfer": causal_gate,
            "causal_necessity": False,
            "natural_use": False,
            "cross_model": False,
            "mechanism_closure": False,
        },
    }
    verdict["verdict_digest"] = digest(verdict)
    write_json(VERDICT_PATH, verdict)
    print(json.dumps({
        "status": status,
        "split_pass": {split: metrics["pass"] for split, metrics in split_metrics.items()},
        "zero_identity_pass": identity_pass,
        "causal_transfer_gate": causal_gate,
        "verdict_digest": verdict["verdict_digest"],
    }, ensure_ascii=False, indent=2))


def finalize_command() -> None:
    protocol = verify_protocol()
    verdict = read_json(VERDICT_PATH)
    audit = read_json(RESULT_AUDIT_PATH)
    validate_embedded_digest(verdict, "verdict_digest")
    validate_embedded_digest(audit, "audit_digest")
    if not audit.get("gate_pass") or audit.get("verdict_digest") != verdict["verdict_digest"]:
        raise RuntimeError("Phase1206 result audit failed")
    causal_gate = bool(verdict["causal_transfer_gate"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "status": verdict["status"],
        "protocol_digest": protocol["protocol_digest"],
        "verdict_digest": verdict["verdict_digest"],
        "independent_result_audit_digest": audit["audit_digest"],
        "target": protocol["target"],
        "causal_transfer_gate": causal_gate,
        "new_k_item": verdict["proposed_k_item_pending_independent_audit"],
        "evidence_scope": verdict["claim_boundary"],
        "authorized_next": {
            "phase1207_qwen3_necessity_rescue_preregistration": causal_gate,
            "automatic_phase1207_execution": False,
            "head_or_neuron_search": False,
            "cross_model_claim": False,
            "natural_use_claim": False,
            "mechanism_closure_claim": False,
        },
        "stop_rule": (
            "Controlled transfer passed; only a separate zero-output Qwen3 necessity/rescue preregistration is authorized."
            if causal_gate else
            "Controlled transfer did not pass; no necessity/rescue, component, head, or neuron target may be selected."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "run", "analyze", "finalize"))
    args = parser.parse_args()
    {"protocol": protocol_command, "run": run_command, "analyze": analyze_command, "finalize": finalize_command}[args.command]()


if __name__ == "__main__":
    main()
