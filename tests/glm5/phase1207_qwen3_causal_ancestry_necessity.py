#!/usr/bin/env python3
"""Qwen3 object-attribute causal onset, necessity, and conditional rescue.

The phase is downstream of the frozen Phase1205/1206 event.  It first scans
only residual depths 20--25 for the earliest repeated full-state causal
transfer.  At that frozen depth it replaces the active pair differential with
the matched surface differential.  Rescue is authorized only if that
contrast-removal operation passes the preregistered necessity gate.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
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
import phase1206_qwen3_object_attribute_causal_transfer as phase1206  # noqa: E402


PHASE = 1207
MODEL = "qwen3"
MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SOURCE1205 = ROOT / "tests/glm5/result/phase1205_qwen3_object_attribute_vertical_closure"
SOURCE1206 = ROOT / "tests/glm5/result/phase1206_qwen3_object_attribute_causal_transfer"
SOURCE1203 = ROOT / "tests/glm5/result/phase1203_object_attribute_behavior_protocol"
PAIR_PATH = SOURCE1205 / "protocol/pair_manifest.jsonl"
MANIFEST_PATH = SOURCE1203 / "protocol/model_manifests/qwen3.jsonl"
UPSTREAM_VECTOR_PATH = SOURCE1206 / "runs/captured_vectors.npz"
UPSTREAM_FINAL_PATH = SOURCE1206 / "analysis/final.json"
UPSTREAM_AUDIT_PATH = SOURCE1206 / "audit/independent_result_audit.json"

OUT_ROOT = ROOT / "tests/glm5/result/phase1207_qwen3_causal_ancestry_necessity"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/preexecution_audit.json"
CAPTURE_PATH = OUT_ROOT / "runs/captured_residuals.npz"
CAPTURE_SUMMARY_PATH = OUT_ROOT / "runs/capture_summary.json"
ONSET_RAW_PATH = OUT_ROOT / "runs/onset_scores.jsonl.gz"
ONSET_SUMMARY_PATH = OUT_ROOT / "runs/onset_summary.json"
ONSET_VERDICT_PATH = OUT_ROOT / "analysis/onset_verdict.json"
NECESSITY_RAW_PATH = OUT_ROOT / "runs/necessity_scores.jsonl.gz"
NECESSITY_SUMMARY_PATH = OUT_ROOT / "runs/necessity_summary.json"
NECESSITY_VERDICT_PATH = OUT_ROOT / "analysis/necessity_verdict.json"
RESCUE_RAW_PATH = OUT_ROOT / "runs/rescue_scores.jsonl.gz"
RESCUE_SUMMARY_PATH = OUT_ROOT / "runs/rescue_summary.json"
RESCUE_VERDICT_PATH = OUT_ROOT / "analysis/rescue_verdict.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

AUDIT_SCRIPT = TEST_ROOT / "phase1207_qwen3_causal_ancestry_necessity_audit.py"
RUNNER_SCRIPT = TEST_ROOT / "phase1207_run_sequential.py"
EXPECTED_1206_FINAL = "227a21405504cdac1967a2fb772c038bc6c051ab6470794dbbb091338603c00b"
EXPECTED_1206_AUDIT = "2d9b7ac8d8dae483ba37f3d55e4af368be6cdca485740e752b9b01a7aeb22d92"

DEPTHS = (20, 21, 22, 23, 24, 25)
RESCUE_DEPTH = 25
SIGNATURE_DEPTHS = (26, 30, 36)
CAPTURE_DEPTHS = tuple(sorted(set(DEPTHS + SIGNATURE_DEPTHS)))
SPLITS = phase1205.SPLITS
HIDDEN_SIZE = 2560
CAPTURE_BATCH_PAIRS = 8
PATCH_BATCH_SIZE = 16
RANDOM_SEED = 12070031
EPSILON = 1e-8
TIE_TOLERANCE = 1e-6

ONSET_CONDITIONS = (
    {"id": "active_full", "kind": "panel", "panel": "active", "evidence": "primary"},
    {"id": "matched_null_full", "kind": "panel", "panel": "matched_null", "evidence": "primary_control"},
    {"id": "surface_only_full", "kind": "panel", "panel": "surface_only", "evidence": "primary_control"},
    {"id": "semantic_neighbor_full", "kind": "panel", "panel": "semantic_neighbor", "evidence": "primary_control"},
    {"id": "random_r0", "kind": "random", "random_index": 0, "evidence": "primary_control"},
    {"id": "random_r1", "kind": "random", "random_index": 1, "evidence": "primary_control"},
    {"id": "random_r2", "kind": "random", "random_index": 2, "evidence": "primary_control"},
    {"id": "random_r3", "kind": "random", "random_index": 3, "evidence": "primary_control"},
    {"id": "zero", "kind": "zero", "evidence": "identity_control"},
)
ONSET_PRIMARY_CONTROLS = tuple(
    row["id"] for row in ONSET_CONDITIONS if row["evidence"] == "primary_control"
)

NECESSITY_CONDITIONS = (
    {"id": "active_vs_surface_remove", "kind": "active_contrast", "evidence": "primary"},
    {"id": "surface_common_control", "kind": "surface", "evidence": "primary_control"},
    {"id": "null_vs_surface_control", "kind": "null_contrast", "evidence": "primary_control"},
    {"id": "neighbor_vs_surface_control", "kind": "neighbor_contrast", "evidence": "primary_control"},
    {"id": "random_r0", "kind": "random", "random_index": 0, "evidence": "primary_control"},
    {"id": "random_r1", "kind": "random", "random_index": 1, "evidence": "primary_control"},
    {"id": "random_r2", "kind": "random", "random_index": 2, "evidence": "primary_control"},
    {"id": "random_r3", "kind": "random", "random_index": 3, "evidence": "primary_control"},
    {"id": "zero", "kind": "zero", "evidence": "identity_control"},
    {"id": "active_midpoint", "kind": "active_midpoint", "evidence": "dose_descriptive"},
)
NECESSITY_PRIMARY_CONTROLS = tuple(
    row["id"] for row in NECESSITY_CONDITIONS if row["evidence"] == "primary_control"
)

RESCUE_CONDITIONS = (
    {"id": "damage_only", "kind": "zero", "evidence": "damaged_baseline"},
    {"id": "specific_addback", "kind": "specific", "evidence": "primary"},
    {"id": "surface_addback", "kind": "surface", "evidence": "primary_control"},
    {"id": "null_addback", "kind": "null", "evidence": "primary_control"},
    {"id": "neighbor_addback", "kind": "neighbor", "evidence": "primary_control"},
    {"id": "opposite_specific", "kind": "opposite", "evidence": "primary_control"},
    {"id": "random_r0", "kind": "random", "random_index": 0, "evidence": "primary_control"},
    {"id": "random_r1", "kind": "random", "random_index": 1, "evidence": "primary_control"},
    {"id": "random_r2", "kind": "random", "random_index": 2, "evidence": "primary_control"},
    {"id": "random_r3", "kind": "random", "random_index": 3, "evidence": "primary_control"},
    {"id": "clean_state_clamp", "kind": "clean_clamp", "evidence": "positive_instrument"},
)
RESCUE_PRIMARY_CONTROLS = tuple(
    row["id"] for row in RESCUE_CONDITIONS if row["evidence"] == "primary_control"
)

ONSET_THRESHOLDS = {
    "finite_fraction": 1.0,
    "baseline_accuracy": 1.0,
    "donor_accuracy": 1.0,
    "positive_shift_fraction": 0.95,
    "donor_choice_fraction": 0.80,
    "median_transfer_fraction": 0.50,
    "beats_all_controls_fraction": 0.75,
    "median_advantage": 0.10,
    "minimum_each_direction_donor_choice": 0.75,
    "minimum_adjacent_discovery_depths": 2,
    "zero_max_abs_logit_drift": 1e-4,
}
NECESSITY_THRESHOLDS = {
    "finite_fraction": 1.0,
    "baseline_accuracy": 1.0,
    "positive_damage_fraction": 0.90,
    "behavior_damage_fraction": 0.10,
    "median_damage_fraction": 0.15,
    "beats_all_controls_fraction": 0.75,
    "median_normalized_advantage": 0.05,
    "minimum_each_direction_behavior_damage": 0.075,
    "zero_max_abs_logit_drift": 1e-4,
}
RESCUE_THRESHOLDS = {
    "finite_fraction": 1.0,
    "minimum_damaged_records_per_split": 12,
    "minimum_damaged_records_per_direction": 5,
    "behavior_restore_fraction": 0.75,
    "median_margin_recovery": 0.50,
    "positive_margin_recovery_fraction": 0.75,
    "median_response_recovery": 0.50,
    "margin_beats_all_controls_fraction": 0.75,
    "response_beats_all_controls_fraction": 0.75,
    "median_margin_advantage": 0.05,
    "minimum_each_direction_restore": 0.60,
    "clean_clamp_restore_fraction": 0.95,
    "clean_clamp_median_response_recovery": 0.95,
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


def validate_digest(value: dict[str, Any], key: str) -> None:
    if digest({name: item for name, item in value.items() if name != key}) != value.get(key):
        raise RuntimeError(f"digest mismatch: {key}")


def source_hashes() -> dict[str, str]:
    return {
        "main": sha256_file(Path(__file__).resolve()),
        "audit": sha256_file(AUDIT_SCRIPT),
        "runner": sha256_file(RUNNER_SCRIPT),
        "phase1206_main": sha256_file(Path(phase1206.__file__).resolve()),
    }


def load_material() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    pairs = read_jsonl(PAIR_PATH)
    manifest = {str(row["item_id"]): row for row in read_jsonl(MANIFEST_PATH)}
    by_group_panel = {(str(row["group_id"]), str(row["panel"])): row for row in pairs}
    active = [row for row in pairs if row["panel"] == "active"]
    return pairs, manifest, by_group_panel, active


def protocol_command() -> None:
    if PROTOCOL_PATH.exists() or (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to rewrite Phase1207 after protocol or output exists")
    final1206 = read_json(UPSTREAM_FINAL_PATH)
    audit1206 = read_json(UPSTREAM_AUDIT_PATH)
    validate_digest(final1206, "final_digest")
    validate_digest(audit1206, "audit_digest")
    upstream_checks = {
        "final_digest": final1206["final_digest"] == EXPECTED_1206_FINAL,
        "audit_digest": audit1206["audit_digest"] == EXPECTED_1206_AUDIT,
        "audit_pass": audit1206["gate_pass"] is True,
        "causal_transfer_gate": final1206["causal_transfer_gate"] is True,
        "preregistration_authorized": final1206["authorized_next"]["phase1207_qwen3_necessity_rescue_preregistration"] is True,
        "automatic_execution_previously_denied": final1206["authorized_next"]["automatic_phase1207_execution"] is False,
        "target_depth_25": final1206["target"]["depth"] == RESCUE_DEPTH,
    }
    if not all(upstream_checks.values()):
        raise RuntimeError(f"Phase1207 upstream failed: {upstream_checks}")
    pairs, _, _, active = load_material()
    counts = {
        "all_pairs": len(pairs),
        "active_pairs": len(active),
        "directions": 2,
        "onset_records": len(active) * 2 * len(DEPTHS) * len(ONSET_CONDITIONS),
        "necessity_records_if_authorized": len(active) * 2 * len(NECESSITY_CONDITIONS),
        "rescue_records_if_authorized": len(active) * 2 * len(RESCUE_CONDITIONS),
    }
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1207.qwen3_causal_ancestry_necessity.v1",
        "created_at": utc_now(),
        "objective": (
            "Separate earliest repeated full-state causal influence from active-specific necessity, then test a "
            "downstream same-sample rescue only if necessity passes."
        ),
        "scope": {
            "model": MODEL,
            "qwen3_only": True,
            "controlled_object_attribute_only": True,
            "causal_onset_claim": True,
            "active_vs_surface_necessity_claim": True,
            "rescue_claim_conditional": True,
            "head_or_neuron_claim": False,
            "natural_use_claim": False,
            "cross_model_claim": False,
            "brain_claim": False,
            "mechanism_closure_claim": False,
        },
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1206_final_digest": final1206["final_digest"],
            "phase1206_audit_digest": audit1206["audit_digest"],
            "pair_file_sha256": sha256_file(PAIR_PATH),
            "pair_digest": digest(pairs),
            "manifest_file_sha256": sha256_file(MANIFEST_PATH),
            "phase1206_vector_sha256": sha256_file(UPSTREAM_VECTOR_PATH),
        },
        "model": {
            "path": str(MODEL_PATH.resolve()),
            "precision": "FP16",
            "quantization": "none",
            "placement": "full_cuda",
            "hidden_size": HIDDEN_SIZE,
            "capture_batch_pairs": CAPTURE_BATCH_PAIRS,
            "patch_batch_size": PATCH_BATCH_SIZE,
        },
        "causal_onset": {
            "depths": list(DEPTHS),
            "role": "generation_boundary",
            "component": "residual",
            "conditions": list(ONSET_CONDITIONS),
            "primary_controls": list(ONSET_PRIMARY_CONTROLS),
            "thresholds": ONSET_THRESHOLDS,
            "selection": "earliest depth in earliest discovery passing run of at least two adjacent depths",
            "confirmation_rule": "confirmation and unseen must pass at the frozen discovery depth; no reselection",
        },
        "necessity": {
            "operation": (
                "For active pair states, replace the active pair differential with the matched surface-only "
                "differential while preserving the active pair midpoint: h_s' = h_s + (0.5-s)(d_A-d_S)."
            ),
            "conditions": list(NECESSITY_CONDITIONS),
            "primary_controls": list(NECESSITY_PRIMARY_CONTROLS),
            "control_norm": "all primary nonzero controls match ||d_A-d_S|| before the 0.5 signed shift",
            "thresholds": NECESSITY_THRESHOLDS,
            "claim_scope": "necessity of the sample-specific active-vs-surface contrast at the frozen onset depth",
        },
        "rescue": {
            "authorization": "run only if onset and necessity gates pass and selected depth is less than 25",
            "damage": "repeat frozen active-vs-surface contrast removal at selected onset depth",
            "rescue_depth": RESCUE_DEPTH,
            "signature_depths": list(SIGNATURE_DEPTHS),
            "conditions": list(RESCUE_CONDITIONS),
            "primary_controls": list(RESCUE_PRIMARY_CONTROLS),
            "primary_addback": "recipient-signed half of d_A-d_S at depth25",
            "positive_instrument": "clamp depth25 generation-boundary residual to the same sample clean state",
            "thresholds": RESCUE_THRESHOLDS,
        },
        "counts": counts,
        "random": {"seed": RANDOM_SEED, "directions": 4, "sample_specific_norm_matching": True},
        "authorization": {
            "capture_and_onset_after_zero_output_audit": True,
            "necessity_only_after_onset_gate": True,
            "rescue_only_after_necessity_gate": True,
            "component_decomposition": False,
            "automatic_phase1208": False,
        },
        "stop_rules": [
            "If the zero-output preaudit fails, do not load Qwen3.",
            "If no adjacent discovery onset band passes, do not run necessity or rescue.",
            "If heldout onset fails at the frozen depth, do not reselect another depth.",
            "If necessity fails any split, do not run rescue and do not claim the late state is necessary.",
            "If the selected onset depth is 25, do not perform a same-depth rescue and record mediation as untested.",
            "A rescue pass is still Qwen3-specific controlled evidence, not natural-language or brain closure.",
            "Do not begin head, neuron, or subspace search in this phase.",
        ],
        "upstream_checks": upstream_checks,
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(json.dumps({"phase": PHASE, "protocol_digest": protocol["protocol_digest"], "counts": counts}, ensure_ascii=False, indent=2))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_digest(protocol, "protocol_digest")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("Phase1207 source hash drift")
    if sha256_file(PAIR_PATH) != protocol["upstream"]["pair_file_sha256"]:
        raise RuntimeError("Phase1207 pair file drift")
    if digest(read_jsonl(PAIR_PATH)) != protocol["upstream"]["pair_digest"]:
        raise RuntimeError("Phase1207 pair semantic drift")
    return protocol


class CaptureResiduals:
    def __init__(self, layers: list[Any], depths: tuple[int, ...]):
        self.layers = layers
        self.depths = depths
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _hook(self, depth: int):
        def hook(module: Any, args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or self.positions is None:
                raise RuntimeError("capture state unavailable")
            batch = torch.arange(value.shape[0], device=value.device)
            self.values[depth] = value[batch, self.positions.to(value.device), :].detach()
            self.calls[depth] += 1
            return output
        return hook

    def register(self) -> None:
        for depth in self.depths:
            self.handles.append(self.layers[depth - 1].register_forward_hook(self._hook(depth)))

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.calls = defaultdict(int)

    def validate(self) -> None:
        if set(self.values) != set(self.depths) or any(self.calls[d] != 1 for d in self.depths):
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
            raise RuntimeError("patch layer output is not tensor")
        batch = torch.arange(value.shape[0], device=value.device)
        patched = value.clone()
        patched[batch, self.positions.to(value.device), :] = (
            value[batch, self.positions.to(value.device), :]
            + self.deltas.to(value.device, dtype=value.dtype)
        )
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


class CascadePatchCapture:
    def __init__(
        self,
        layers: list[Any],
        damage_depth: int,
        positions: torch.Tensor,
        damage_deltas: torch.Tensor,
        rescue_deltas: torch.Tensor | None,
        rescue_targets: torch.Tensor | None,
    ):
        self.layers = layers
        self.damage_depth = damage_depth
        self.positions = positions
        self.damage_deltas = damage_deltas
        self.rescue_deltas = rescue_deltas
        self.rescue_targets = rescue_targets
        self.handles: list[Any] = []
        self.damage_calls = 0
        self.rescue_calls = 0
        self.signature_calls: dict[int, int] = defaultdict(int)
        self.signature: dict[int, torch.Tensor] = {}

    def _damage(self, module: Any, args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        batch = torch.arange(value.shape[0], device=value.device)
        pos = self.positions.to(value.device)
        patched = value.clone()
        patched[batch, pos, :] = value[batch, pos, :] + self.damage_deltas.to(value.device, dtype=value.dtype)
        self.damage_calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def _rescue(self, module: Any, args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        batch = torch.arange(value.shape[0], device=value.device)
        pos = self.positions.to(value.device)
        patched = value.clone()
        if self.rescue_targets is not None:
            patched[batch, pos, :] = self.rescue_targets.to(value.device, dtype=value.dtype)
        elif self.rescue_deltas is not None:
            patched[batch, pos, :] = value[batch, pos, :] + self.rescue_deltas.to(value.device, dtype=value.dtype)
        self.rescue_calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def _capture(self, depth: int):
        def hook(module: Any, args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            batch = torch.arange(value.shape[0], device=value.device)
            self.signature[depth] = value[batch, self.positions.to(value.device), :].detach()
            self.signature_calls[depth] += 1
            return output
        return hook

    def __enter__(self):
        self.handles.append(self.layers[self.damage_depth - 1].register_forward_hook(self._damage))
        self.handles.append(self.layers[RESCUE_DEPTH - 1].register_forward_hook(self._rescue))
        for depth in SIGNATURE_DEPTHS:
            self.handles.append(self.layers[depth - 1].register_forward_hook(self._capture(depth)))
        return self

    def validate(self) -> None:
        if self.damage_calls != 1 or self.rescue_calls != 1 or any(self.signature_calls[d] != 1 for d in SIGNATURE_DEPTHS):
            raise RuntimeError(
                f"cascade drift damage={self.damage_calls} rescue={self.rescue_calls} signature={dict(self.signature_calls)}"
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def placement_audit(model: Any) -> dict[str, Any]:
    devices = {str(parameter.device) for parameter in model.parameters()}
    return {"placement": "full_cuda" if devices == {"cuda:0"} else "mixed", "devices": sorted(devices), "quantization": "none"}


def score_batch(logits: torch.Tensor, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    last = logits[:, -1, :].float()
    finite = torch.isfinite(last).all(dim=-1).cpu().numpy().astype(np.bool_)
    scores = np.empty((len(rows), 3), dtype=np.float32)
    for index, row in enumerate(rows):
        ids = [int(row["manifest"]["candidate_token_ids"][label][0]) for label in row["entities"]]
        scores[index] = last[index, ids].cpu().numpy()
    return scores, finite


def prediction(labels: list[str], scores: np.ndarray, finite: bool) -> str:
    if not finite or not np.isfinite(scores).all():
        return "NONFINITE"
    order = np.argsort(-scores, kind="stable")
    if float(scores[order[0]] - scores[order[1]]) <= TIE_TOLERANCE:
        return "UNRESOLVED_TIE"
    return labels[int(order[0])]


def random_vector(group_id: str, stage: str, index: int, norm: float) -> np.ndarray:
    seed_text = f"{RANDOM_SEED}|{stage}|{group_id}|{index}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(HIDDEN_SIZE).astype(np.float32)
    vector /= float(np.linalg.norm(vector)) + EPSILON
    return vector * float(norm)


def norm_match(vector: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return np.zeros(HIDDEN_SIZE, dtype=np.float32)
    return vector.astype(np.float32) * (float(target_norm) / norm)


def precision_ok(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    precision = quantization_audit(model)
    placement = placement_audit(model)
    ok = bool(
        precision["has_fp16_parameters"]
        and not precision["has_bf16_parameters"]
        and not precision["has_quantized_modules"]
        and set(precision["parameter_dtypes"]) == {"float16"}
        and placement["placement"] == "full_cuda"
    )
    if not ok:
        raise RuntimeError(f"precision/placement failed: {precision} {placement}")
    return precision, placement


def capture_command() -> None:
    protocol = verify_protocol()
    if CAPTURE_PATH.exists() or CAPTURE_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1207 capture output already exists")
    preaudit = read_json(PREAUDIT_PATH)
    validate_digest(preaudit, "audit_digest")
    if not preaudit["gate_pass"] or preaudit["protocol_digest"] != protocol["protocol_digest"]:
        raise RuntimeError("Phase1207 preaudit did not authorize capture")
    pairs, manifest, _, _ = load_material()
    depth_index = {depth: index for index, depth in enumerate(CAPTURE_DEPTHS)}
    vectors = np.empty((len(pairs), 2, len(CAPTURE_DEPTHS), HIDDEN_SIZE), dtype=np.float16)
    scores = np.empty((len(pairs), 2, 3), dtype=np.float32)
    finite = np.zeros((len(pairs), 2), dtype=np.bool_)
    started = time.time()
    model = None
    capture = None
    try:
        model, tokenizer, device, precision = load_fp16(MODEL)
        precision, placement = precision_ok(model)
        layers = list(get_layers(model))
        if len(layers) != 36:
            raise RuntimeError(f"layer count drift: {len(layers)}")
        capture = CaptureResiduals(layers, CAPTURE_DEPTHS)
        capture.register()
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_length[int(pair["input_length"])].append(pair)
        completed = 0
        with torch.inference_mode():
            for length in sorted(by_length):
                members = sorted(by_length[length], key=lambda row: int(row["pair_index"]))
                for start in range(0, len(members), CAPTURE_BATCH_PAIRS):
                    batch_pairs = members[start:start + CAPTURE_BATCH_PAIRS]
                    rows: list[dict[str, Any]] = []
                    input_rows: list[list[int]] = []
                    positions: list[int] = []
                    for pair in batch_pairs:
                        for state in (0, 1):
                            item = manifest[str(pair[f"state{state}_item_id"])]
                            rows.append({"pair": pair, "state": state, "manifest": item, "entities": list(pair["entities"])})
                            input_rows.append(list(item["input_ids"]))
                            positions.append(int(pair[f"state{state}_positions"]["generation_boundary"]))
                    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
                    mask = torch.ones_like(input_ids)
                    capture.begin(torch.tensor(positions, dtype=torch.long, device=device))
                    output = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
                    capture.validate()
                    batch_scores, batch_finite = score_batch(output.logits, rows)
                    for slot, row in enumerate(rows):
                        pair_index = int(row["pair"]["pair_index"])
                        state = int(row["state"])
                        scores[pair_index, state] = batch_scores[slot]
                        finite[pair_index, state] = batch_finite[slot]
                        for depth in CAPTURE_DEPTHS:
                            vectors[pair_index, state, depth_index[depth]] = capture.values[depth][slot].cpu().numpy()
                    completed += len(batch_pairs)
                    del output, input_ids, mask
                print(canonical({"phase": PHASE, "capture_length": length, "completed_pairs": completed}), flush=True)
        capture.close()
        capture = None
        correct: list[bool] = []
        for pair in pairs:
            labels = list(pair["entities"])
            index = int(pair["pair_index"])
            for state in (0, 1):
                correct.append(prediction(labels, scores[index, state], bool(finite[index, state])) == pair[f"state{state}_gold"])
        if not all(correct) or not bool(finite.all()):
            raise RuntimeError("Phase1207 capture baseline behavior drift")
        upstream = np.load(UPSTREAM_VECTOR_PATH, allow_pickle=False)
        d24 = vectors[:, :, depth_index[24]].astype(np.float32)
        d25 = vectors[:, :, depth_index[25]].astype(np.float32)
        replay24 = float(np.max(np.abs(d24 - upstream["d24_generation_boundary"].astype(np.float32))))
        replay25 = float(np.max(np.abs(d25 - upstream["d25_generation_boundary"].astype(np.float32))))
        replay_scores = float(np.max(np.abs(scores - upstream["baseline_scores"].astype(np.float32))))
        write_npz_atomic(
            CAPTURE_PATH,
            residuals=vectors,
            baseline_scores=scores,
            baseline_finite=finite,
            capture_depths=np.asarray(CAPTURE_DEPTHS, dtype=np.int16),
        )
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1207.capture.v1",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "preexecution_audit_digest": preaudit["audit_digest"],
            "pair_count": len(pairs),
            "residual_shape": list(vectors.shape),
            "scores_shape": list(scores.shape),
            "finite_fraction": float(finite.mean()),
            "baseline_accuracy": sum(correct) / len(correct),
            "upstream_replay_max_abs": {"depth24": replay24, "depth25": replay25, "scores": replay_scores},
            "capture_file_sha256": sha256_file(CAPTURE_PATH),
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
        }
        summary["summary_digest"] = digest(summary)
        write_json(CAPTURE_SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)
        gc.collect()


def load_capture() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    summary = read_json(CAPTURE_SUMMARY_PATH)
    validate_digest(summary, "summary_digest")
    if sha256_file(CAPTURE_PATH) != summary["capture_file_sha256"]:
        raise RuntimeError("capture file drift")
    arrays = np.load(CAPTURE_PATH, allow_pickle=False)
    depths = [int(value) for value in arrays["capture_depths"].tolist()]
    return arrays["residuals"], arrays["baseline_scores"], arrays["baseline_finite"], {d: i for i, d in enumerate(depths)}


def record_common(
    pair: dict[str, Any],
    recipient_state: int,
    condition: dict[str, Any],
    baseline_scores: np.ndarray,
    baseline_finite: np.ndarray,
    patched_scores: np.ndarray,
    patched_finite: bool,
    delta_l2: float,
    depth: int,
) -> dict[str, Any]:
    index = int(pair["pair_index"])
    donor_state = 1 - recipient_state
    labels = list(pair["entities"])
    return {
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
        "candidate_labels": labels,
        "recipient_gold": str(pair[f"state{recipient_state}_gold"]),
        "donor_gold": str(pair[f"state{donor_state}_gold"]),
        "recipient_scores": [float(value) for value in baseline_scores[index, recipient_state]],
        "donor_scores": [float(value) for value in baseline_scores[index, donor_state]],
        "patched_scores": [float(value) for value in patched_scores],
        "recipient_finite": bool(baseline_finite[index, recipient_state]),
        "donor_finite": bool(baseline_finite[index, donor_state]),
        "patched_finite": bool(patched_finite),
        "recipient_prediction": prediction(labels, baseline_scores[index, recipient_state], bool(baseline_finite[index, recipient_state])),
        "donor_prediction": prediction(labels, baseline_scores[index, donor_state], bool(baseline_finite[index, donor_state])),
        "patched_prediction": prediction(labels, patched_scores, bool(patched_finite)),
        "delta_l2": float(delta_l2),
    }


def run_onset_command() -> None:
    protocol = verify_protocol()
    if ONSET_RAW_PATH.exists() or ONSET_SUMMARY_PATH.exists() or ONSET_VERDICT_PATH.exists():
        raise RuntimeError("Phase1207 onset output already exists")
    vectors, baseline_scores, baseline_finite, depth_index = load_capture()
    pairs, manifest, panel_by_group, active = load_material()
    started = time.time()
    records: list[dict[str, Any]] = []
    model = None
    try:
        model, tokenizer, device, precision = load_fp16(MODEL)
        precision, placement = precision_ok(model)
        layers = list(get_layers(model))
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in active:
            by_length[int(pair["input_length"])].append(pair)
        with torch.inference_mode():
            for depth in DEPTHS:
                di = depth_index[depth]
                for condition in ONSET_CONDITIONS:
                    for length in sorted(by_length):
                        entries = [(pair, state) for pair in sorted(by_length[length], key=lambda row: int(row["pair_index"])) for state in (0, 1)]
                        for start in range(0, len(entries), PATCH_BATCH_SIZE):
                            batch = entries[start:start + PATCH_BATCH_SIZE]
                            input_rows: list[list[int]] = []
                            positions: list[int] = []
                            deltas: list[np.ndarray] = []
                            rows: list[dict[str, Any]] = []
                            for pair, state in batch:
                                item = manifest[str(pair[f"state{state}_item_id"])]
                                input_rows.append(list(item["input_ids"]))
                                positions.append(int(pair[f"state{state}_positions"]["generation_boundary"]))
                                rows.append({"manifest": item, "entities": list(pair["entities"])})
                                if condition["kind"] == "zero":
                                    delta = np.zeros(HIDDEN_SIZE, dtype=np.float32)
                                elif condition["kind"] == "random":
                                    index = int(pair["pair_index"])
                                    active_delta = vectors[index, 1, di].astype(np.float32) - vectors[index, 0, di].astype(np.float32)
                                    direction = 1.0 if state == 0 else -1.0
                                    delta = random_vector(str(pair["group_id"]), f"onset-d{depth}", int(condition["random_index"]), float(np.linalg.norm(active_delta))) * direction
                                else:
                                    source = panel_by_group[(str(pair["group_id"]), str(condition["panel"]))]
                                    source_index = int(source["pair_index"])
                                    delta = vectors[source_index, 1 - state, di].astype(np.float32) - vectors[source_index, state, di].astype(np.float32)
                                deltas.append(delta)
                            ids = torch.tensor(input_rows, dtype=torch.long, device=device)
                            mask = torch.ones_like(ids)
                            pos = torch.tensor(positions, dtype=torch.long, device=device)
                            delta_tensor = torch.from_numpy(np.stack(deltas)).to(device=device, dtype=torch.float16)
                            with DeltaPatch(layers[depth - 1], pos, delta_tensor) as patch:
                                output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
                            if patch.calls != 1:
                                raise RuntimeError("onset patch call drift")
                            scores, finite = score_batch(output.logits, rows)
                            for slot, (pair, state) in enumerate(batch):
                                row = record_common(pair, state, condition, baseline_scores, baseline_finite, scores[slot], bool(finite[slot]), float(np.linalg.norm(deltas[slot])), depth)
                                row["record_id"] = f"{pair['group_id']}|d{depth}|r{state}|{condition['id']}"
                                records.append(row)
                            del output, ids, mask, pos, delta_tensor
                    print(canonical({"phase": PHASE, "stage": "onset", "depth": depth, "condition": condition["id"], "records": len(records)}), flush=True)
        if len(records) != protocol["counts"]["onset_records"] or not all(row["patched_finite"] for row in records):
            raise RuntimeError("Phase1207 onset completeness/finiteness failed")
        write_jsonl_gz(ONSET_RAW_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1207.onset_run.v1",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "capture_summary_digest": read_json(CAPTURE_SUMMARY_PATH)["summary_digest"],
            "record_count": len(records),
            "raw_file_sha256": sha256_file(ONSET_RAW_PATH),
            "raw_digest": digest(records),
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = digest(summary)
        write_json(ONSET_SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()


def median(values: Iterable[float]) -> float:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    return float(np.median(data)) if data.size else 0.0


def enrich_transfer(row: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    recipient_index = labels.index(str(row["recipient_gold"]))
    donor_index = labels.index(str(row["donor_gold"]))
    recipient = np.asarray(row["recipient_scores"], dtype=np.float64)
    donor = np.asarray(row["donor_scores"], dtype=np.float64)
    patched = np.asarray(row["patched_scores"], dtype=np.float64)
    base = float(recipient[donor_index] - recipient[recipient_index])
    donor_margin = float(donor[donor_index] - donor[recipient_index])
    patched_margin = float(patched[donor_index] - patched[recipient_index])
    shift = patched_margin - base
    full = donor_margin - base
    return {
        **row,
        "donor_margin_shift": shift,
        "transfer_fraction": shift / (full + EPSILON),
        "positive_shift": shift > 0,
        "donor_choice": row["patched_prediction"] == row["donor_gold"],
        "recipient_correct": row["recipient_prediction"] == row["recipient_gold"],
        "donor_correct": row["donor_prediction"] == row["donor_gold"],
    }


def onset_metrics(rows: list[dict[str, Any]], split: str, depth: int) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split and int(row["depth"]) == depth]
    lookup = {(row["group_id"], int(row["recipient_state"]), row["condition"]): row for row in members}
    target = [row for row in members if row["condition"] == "active_full"]
    advantages: list[float] = []
    directions: dict[str, float] = {}
    for row in target:
        controls = [lookup[(row["group_id"], int(row["recipient_state"]), name)]["donor_margin_shift"] for name in ONSET_PRIMARY_CONTROLS]
        advantages.append(float(row["donor_margin_shift"]) - max(float(value) for value in controls))
    for state in (0, 1):
        subset = [row for row in target if int(row["recipient_state"]) == state]
        directions[f"state{state}_to_state{1-state}"] = sum(bool(row["donor_choice"]) for row in subset) / max(len(subset), 1)
    result = {
        "split": split,
        "depth": depth,
        "target_count": len(target),
        "finite_fraction": sum(bool(row["recipient_finite"] and row["donor_finite"] and row["patched_finite"]) for row in target) / max(len(target), 1),
        "baseline_accuracy": sum(bool(row["recipient_correct"]) for row in target) / max(len(target), 1),
        "donor_accuracy": sum(bool(row["donor_correct"]) for row in target) / max(len(target), 1),
        "positive_shift_fraction": sum(bool(row["positive_shift"]) for row in target) / max(len(target), 1),
        "donor_choice_fraction": sum(bool(row["donor_choice"]) for row in target) / max(len(target), 1),
        "median_shift": median(row["donor_margin_shift"] for row in target),
        "median_transfer_fraction": median(row["transfer_fraction"] for row in target),
        "beats_all_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_advantage": median(advantages),
        "direction_donor_choice": directions,
    }
    t = ONSET_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= t["finite_fraction"]
        and result["baseline_accuracy"] >= t["baseline_accuracy"]
        and result["donor_accuracy"] >= t["donor_accuracy"]
        and result["positive_shift_fraction"] >= t["positive_shift_fraction"]
        and result["donor_choice_fraction"] >= t["donor_choice_fraction"]
        and result["median_transfer_fraction"] >= t["median_transfer_fraction"]
        and result["beats_all_controls_fraction"] >= t["beats_all_controls_fraction"]
        and result["median_advantage"] >= t["median_advantage"]
        and min(directions.values()) >= t["minimum_each_direction_donor_choice"]
    )
    return result


def contiguous_runs(depths: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for depth in sorted(depths):
        if not runs or depth != runs[-1][-1] + 1:
            runs.append([depth])
        else:
            runs[-1].append(depth)
    return runs


def analyze_onset_command() -> None:
    protocol = verify_protocol()
    if ONSET_VERDICT_PATH.exists():
        raise RuntimeError("Phase1207 onset verdict exists")
    summary = read_json(ONSET_SUMMARY_PATH)
    validate_digest(summary, "summary_digest")
    if sha256_file(ONSET_RAW_PATH) != summary["raw_file_sha256"]:
        raise RuntimeError("onset raw drift")
    raw = read_jsonl_gz(ONSET_RAW_PATH)
    if digest(raw) != summary["raw_digest"]:
        raise RuntimeError("onset semantic drift")
    rows = [enrich_transfer(row) for row in raw]
    metrics = {split: {str(depth): onset_metrics(rows, split, depth) for depth in DEPTHS} for split in SPLITS}
    passing = [depth for depth in DEPTHS if metrics["discovery"][str(depth)]["pass"]]
    runs = contiguous_runs(passing)
    qualified = [run for run in runs if len(run) >= ONSET_THRESHOLDS["minimum_adjacent_discovery_depths"]]
    selected = qualified[0][0] if qualified else None
    heldout = bool(selected is not None and all(metrics[split][str(selected)]["pass"] for split in ("confirmation", "unseen_composition")))
    zero_rows = [row for row in raw if row["condition"] == "zero"]
    zero_drift = max(abs(float(a) - float(b)) for row in zero_rows for a, b in zip(row["patched_scores"], row["recipient_scores"]))
    identity = zero_drift <= ONSET_THRESHOLDS["zero_max_abs_logit_drift"]
    gate = bool(selected is not None and heldout and identity)
    verdict: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1207.onset_verdict.v1",
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "onset_summary_digest": summary["summary_digest"],
        "metrics": metrics,
        "discovery_passing_depths": passing,
        "discovery_runs": runs,
        "qualifying_runs": qualified,
        "selected_depth": selected,
        "heldout_pass_at_selected": heldout,
        "zero_max_abs_logit_drift": zero_drift,
        "identity_pass": identity,
        "onset_gate": gate,
        "status": "qwen3_causal_onset_qualified" if gate else "qwen3_causal_onset_not_qualified",
        "authorization": {"necessity_run": gate, "rescue_run": False, "component_search": False},
    }
    verdict["verdict_digest"] = digest(verdict)
    write_json(ONSET_VERDICT_PATH, verdict)
    print(json.dumps({"passing": passing, "selected": selected, "heldout": heldout, "gate": gate, "digest": verdict["verdict_digest"]}, ensure_ascii=False, indent=2))


def panel_delta(vectors: np.ndarray, depth_index: dict[int, int], pair: dict[str, Any], depth: int) -> np.ndarray:
    index = int(pair["pair_index"])
    di = depth_index[depth]
    return vectors[index, 1, di].astype(np.float32) - vectors[index, 0, di].astype(np.float32)


def necessity_direction(
    condition: dict[str, Any],
    group_id: str,
    d_active: np.ndarray,
    d_surface: np.ndarray,
    d_null: np.ndarray,
    d_neighbor: np.ndarray,
) -> np.ndarray:
    primary = d_active - d_surface
    target_norm = float(np.linalg.norm(primary))
    kind = str(condition["kind"])
    if kind == "active_contrast":
        return primary
    if kind == "surface":
        return norm_match(d_surface, target_norm)
    if kind == "null_contrast":
        return norm_match(d_null - d_surface, target_norm)
    if kind == "neighbor_contrast":
        return norm_match(d_neighbor - d_surface, target_norm)
    if kind == "random":
        return random_vector(group_id, "necessity", int(condition["random_index"]), target_norm)
    if kind == "active_midpoint":
        return d_active
    if kind == "zero":
        return np.zeros(HIDDEN_SIZE, dtype=np.float32)
    raise RuntimeError(f"unknown necessity kind {kind}")


def run_necessity_command() -> None:
    protocol = verify_protocol()
    if NECESSITY_RAW_PATH.exists() or NECESSITY_SUMMARY_PATH.exists() or NECESSITY_VERDICT_PATH.exists():
        raise RuntimeError("Phase1207 necessity output exists")
    onset = read_json(ONSET_VERDICT_PATH)
    validate_digest(onset, "verdict_digest")
    if not onset["onset_gate"]:
        raise RuntimeError("Phase1207 onset did not authorize necessity")
    depth = int(onset["selected_depth"])
    vectors, baseline_scores, baseline_finite, depth_index = load_capture()
    pairs, manifest, panel_by_group, active = load_material()
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in active:
        by_length[int(pair["input_length"])].append(pair)
    started = time.time()
    records: list[dict[str, Any]] = []
    model = None
    try:
        model, tokenizer, device, precision = load_fp16(MODEL)
        precision, placement = precision_ok(model)
        layers = list(get_layers(model))
        with torch.inference_mode():
            for condition in NECESSITY_CONDITIONS:
                for length in sorted(by_length):
                    entries = [(pair, state) for pair in sorted(by_length[length], key=lambda row: int(row["pair_index"])) for state in (0, 1)]
                    for start in range(0, len(entries), PATCH_BATCH_SIZE):
                        batch = entries[start:start + PATCH_BATCH_SIZE]
                        input_rows: list[list[int]] = []
                        positions: list[int] = []
                        deltas: list[np.ndarray] = []
                        rows: list[dict[str, Any]] = []
                        for pair, state in batch:
                            item = manifest[str(pair[f"state{state}_item_id"])]
                            input_rows.append(list(item["input_ids"]))
                            positions.append(int(pair[f"state{state}_positions"]["generation_boundary"]))
                            rows.append({"manifest": item, "entities": list(pair["entities"])})
                            group = str(pair["group_id"])
                            d_active = panel_delta(vectors, depth_index, pair, depth)
                            d_surface = panel_delta(vectors, depth_index, panel_by_group[(group, "surface_only")], depth)
                            d_null = panel_delta(vectors, depth_index, panel_by_group[(group, "matched_null")], depth)
                            d_neighbor = panel_delta(vectors, depth_index, panel_by_group[(group, "semantic_neighbor")], depth)
                            direction = necessity_direction(condition, group, d_active, d_surface, d_null, d_neighbor)
                            signed = 0.5 if state == 0 else -0.5
                            deltas.append(direction * signed)
                        ids = torch.tensor(input_rows, dtype=torch.long, device=device)
                        mask = torch.ones_like(ids)
                        pos = torch.tensor(positions, dtype=torch.long, device=device)
                        delta_tensor = torch.from_numpy(np.stack(deltas)).to(device=device, dtype=torch.float16)
                        with DeltaPatch(layers[depth - 1], pos, delta_tensor) as patch:
                            output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
                        if patch.calls != 1:
                            raise RuntimeError("necessity patch call drift")
                        scores, finite = score_batch(output.logits, rows)
                        for slot, (pair, state) in enumerate(batch):
                            row = record_common(pair, state, condition, baseline_scores, baseline_finite, scores[slot], bool(finite[slot]), float(np.linalg.norm(deltas[slot])), depth)
                            row["record_id"] = f"{pair['group_id']}|d{depth}|r{state}|{condition['id']}"
                            records.append(row)
                        del output, ids, mask, pos, delta_tensor
                print(canonical({"phase": PHASE, "stage": "necessity", "condition": condition["id"], "records": len(records)}), flush=True)
        if len(records) != protocol["counts"]["necessity_records_if_authorized"] or not all(row["patched_finite"] for row in records):
            raise RuntimeError("Phase1207 necessity completeness/finiteness failed")
        write_jsonl_gz(NECESSITY_RAW_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1207.necessity_run.v1",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "onset_verdict_digest": onset["verdict_digest"],
            "selected_depth": depth,
            "record_count": len(records),
            "raw_file_sha256": sha256_file(NECESSITY_RAW_PATH),
            "raw_digest": digest(records),
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = digest(summary)
        write_json(NECESSITY_SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()


def enrich_necessity(row: dict[str, Any], full_shift: float) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    recipient_index = labels.index(str(row["recipient_gold"]))
    donor_index = labels.index(str(row["donor_gold"]))
    recipient = np.asarray(row["recipient_scores"], dtype=np.float64)
    patched = np.asarray(row["patched_scores"], dtype=np.float64)
    base_margin = float(recipient[recipient_index] - recipient[donor_index])
    patched_margin = float(patched[recipient_index] - patched[donor_index])
    damage = base_margin - patched_margin
    return {
        **row,
        "recipient_margin": base_margin,
        "patched_recipient_margin": patched_margin,
        "margin_damage": damage,
        "damage_fraction": damage / (abs(float(full_shift)) + EPSILON),
        "positive_damage": damage > 0,
        "behavior_damage": row["patched_prediction"] != row["recipient_gold"],
        "recipient_correct": row["recipient_prediction"] == row["recipient_gold"],
    }


def necessity_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split]
    lookup = {(row["group_id"], int(row["recipient_state"]), row["condition"]): row for row in members}
    target = [row for row in members if row["condition"] == "active_vs_surface_remove"]
    advantages: list[float] = []
    directions: dict[str, float] = {}
    for row in target:
        controls = [lookup[(row["group_id"], int(row["recipient_state"]), name)]["damage_fraction"] for name in NECESSITY_PRIMARY_CONTROLS]
        advantages.append(float(row["damage_fraction"]) - max(float(value) for value in controls))
    for state in (0, 1):
        subset = [row for row in target if int(row["recipient_state"]) == state]
        directions[f"state{state}"] = sum(bool(row["behavior_damage"]) for row in subset) / max(len(subset), 1)
    result = {
        "split": split,
        "target_count": len(target),
        "finite_fraction": sum(bool(row["recipient_finite"] and row["patched_finite"]) for row in target) / max(len(target), 1),
        "baseline_accuracy": sum(bool(row["recipient_correct"]) for row in target) / max(len(target), 1),
        "positive_damage_fraction": sum(bool(row["positive_damage"]) for row in target) / max(len(target), 1),
        "behavior_damage_fraction": sum(bool(row["behavior_damage"]) for row in target) / max(len(target), 1),
        "median_margin_damage": median(row["margin_damage"] for row in target),
        "median_damage_fraction": median(row["damage_fraction"] for row in target),
        "beats_all_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_normalized_advantage": median(advantages),
        "direction_behavior_damage": directions,
    }
    t = NECESSITY_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= t["finite_fraction"]
        and result["baseline_accuracy"] >= t["baseline_accuracy"]
        and result["positive_damage_fraction"] >= t["positive_damage_fraction"]
        and result["behavior_damage_fraction"] >= t["behavior_damage_fraction"]
        and result["median_damage_fraction"] >= t["median_damage_fraction"]
        and result["beats_all_controls_fraction"] >= t["beats_all_controls_fraction"]
        and result["median_normalized_advantage"] >= t["median_normalized_advantage"]
        and min(directions.values()) >= t["minimum_each_direction_behavior_damage"]
    )
    return result


def analyze_necessity_command() -> None:
    protocol = verify_protocol()
    if NECESSITY_VERDICT_PATH.exists():
        raise RuntimeError("necessity verdict exists")
    onset = read_json(ONSET_VERDICT_PATH)
    summary = read_json(NECESSITY_SUMMARY_PATH)
    validate_digest(onset, "verdict_digest")
    validate_digest(summary, "summary_digest")
    if sha256_file(NECESSITY_RAW_PATH) != summary["raw_file_sha256"]:
        raise RuntimeError("necessity raw drift")
    raw = read_jsonl_gz(NECESSITY_RAW_PATH)
    if digest(raw) != summary["raw_digest"]:
        raise RuntimeError("necessity semantic drift")
    onset_rows = [enrich_transfer(row) for row in read_jsonl_gz(ONSET_RAW_PATH)]
    depth = int(onset["selected_depth"])
    full = {
        (row["group_id"], int(row["recipient_state"])): float(row["donor_margin_shift"])
        for row in onset_rows if int(row["depth"]) == depth and row["condition"] == "active_full"
    }
    rows = [enrich_necessity(row, full[(row["group_id"], int(row["recipient_state"]))]) for row in raw]
    metrics = {split: necessity_metrics(rows, split) for split in SPLITS}
    zero_rows = [row for row in raw if row["condition"] == "zero"]
    zero_drift = max(abs(float(a) - float(b)) for row in zero_rows for a, b in zip(row["patched_scores"], row["recipient_scores"]))
    identity = zero_drift <= NECESSITY_THRESHOLDS["zero_max_abs_logit_drift"]
    gate = bool(identity and all(metrics[split]["pass"] for split in SPLITS))
    rescue_authorized = bool(gate and depth < RESCUE_DEPTH)
    verdict: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1207.necessity_verdict.v1",
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "onset_verdict_digest": onset["verdict_digest"],
        "necessity_summary_digest": summary["summary_digest"],
        "selected_depth": depth,
        "metrics": metrics,
        "zero_max_abs_logit_drift": zero_drift,
        "identity_pass": identity,
        "necessity_gate": gate,
        "status": "qwen3_active_surface_contrast_necessary" if gate else "qwen3_active_surface_contrast_necessity_not_qualified",
        "authorization": {"rescue_run": rescue_authorized, "same_depth_rescue_denied": gate and depth == RESCUE_DEPTH, "component_search": False},
        "claim_boundary": "sample-specific active-vs-surface differential at the frozen Qwen3 causal-onset depth",
    }
    verdict["verdict_digest"] = digest(verdict)
    write_json(NECESSITY_VERDICT_PATH, verdict)
    print(json.dumps({"metrics": metrics, "necessity_gate": gate, "rescue_authorized": rescue_authorized, "digest": verdict["verdict_digest"]}, ensure_ascii=False, indent=2))


def rescue_base_vector(
    kind: str,
    group: str,
    state: int,
    d_active: np.ndarray,
    d_surface: np.ndarray,
    d_null: np.ndarray,
    d_neighbor: np.ndarray,
    random_index: int | None,
) -> np.ndarray:
    primary = d_active - d_surface
    target_norm = float(np.linalg.norm(primary))
    restore_sign = -0.5 if state == 0 else 0.5
    if kind == "specific":
        return primary * restore_sign
    if kind == "surface":
        return norm_match(d_surface, target_norm) * restore_sign
    if kind == "null":
        return norm_match(d_null - d_surface, target_norm) * restore_sign
    if kind == "neighbor":
        return norm_match(d_neighbor - d_surface, target_norm) * restore_sign
    if kind == "opposite":
        return primary * (-restore_sign)
    if kind == "random":
        return random_vector(group, "rescue", int(random_index), target_norm) * restore_sign
    if kind == "zero":
        return np.zeros(HIDDEN_SIZE, dtype=np.float32)
    raise RuntimeError(f"unknown rescue kind {kind}")


def signature_error(observed: list[np.ndarray], clean: list[np.ndarray]) -> float:
    relative: list[float] = []
    for value, target in zip(observed, clean):
        relative.append(float(np.linalg.norm(value.astype(np.float32) - target.astype(np.float32))) / (float(np.linalg.norm(target.astype(np.float32))) + EPSILON))
    return median(relative)


def run_rescue_command() -> None:
    protocol = verify_protocol()
    if RESCUE_RAW_PATH.exists() or RESCUE_SUMMARY_PATH.exists() or RESCUE_VERDICT_PATH.exists():
        raise RuntimeError("Phase1207 rescue output exists")
    onset = read_json(ONSET_VERDICT_PATH)
    necessity = read_json(NECESSITY_VERDICT_PATH)
    validate_digest(onset, "verdict_digest")
    validate_digest(necessity, "verdict_digest")
    if not necessity["authorization"]["rescue_run"]:
        raise RuntimeError("Phase1207 necessity did not authorize rescue")
    damage_depth = int(onset["selected_depth"])
    vectors, baseline_scores, baseline_finite, depth_index = load_capture()
    pairs, manifest, panel_by_group, active = load_material()
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in active:
        by_length[int(pair["input_length"])].append(pair)
    records: list[dict[str, Any]] = []
    started = time.time()
    model = None
    try:
        model, tokenizer, device, precision = load_fp16(MODEL)
        precision, placement = precision_ok(model)
        layers = list(get_layers(model))
        with torch.inference_mode():
            for condition in RESCUE_CONDITIONS:
                for length in sorted(by_length):
                    entries = [(pair, state) for pair in sorted(by_length[length], key=lambda row: int(row["pair_index"])) for state in (0, 1)]
                    for start in range(0, len(entries), PATCH_BATCH_SIZE):
                        batch = entries[start:start + PATCH_BATCH_SIZE]
                        input_rows: list[list[int]] = []
                        positions: list[int] = []
                        damage_deltas: list[np.ndarray] = []
                        rescue_deltas: list[np.ndarray] = []
                        clean_targets: list[np.ndarray] = []
                        clean_signatures: list[list[np.ndarray]] = []
                        rows: list[dict[str, Any]] = []
                        for pair, state in batch:
                            item = manifest[str(pair[f"state{state}_item_id"])]
                            input_rows.append(list(item["input_ids"]))
                            positions.append(int(pair[f"state{state}_positions"]["generation_boundary"]))
                            rows.append({"manifest": item, "entities": list(pair["entities"])})
                            group = str(pair["group_id"])
                            d_active_damage = panel_delta(vectors, depth_index, pair, damage_depth)
                            d_surface_damage = panel_delta(vectors, depth_index, panel_by_group[(group, "surface_only")], damage_depth)
                            damage_sign = 0.5 if state == 0 else -0.5
                            damage_deltas.append((d_active_damage - d_surface_damage) * damage_sign)
                            d_active = panel_delta(vectors, depth_index, pair, RESCUE_DEPTH)
                            d_surface = panel_delta(vectors, depth_index, panel_by_group[(group, "surface_only")], RESCUE_DEPTH)
                            d_null = panel_delta(vectors, depth_index, panel_by_group[(group, "matched_null")], RESCUE_DEPTH)
                            d_neighbor = panel_delta(vectors, depth_index, panel_by_group[(group, "semantic_neighbor")], RESCUE_DEPTH)
                            if condition["kind"] == "clean_clamp":
                                rescue_deltas.append(np.zeros(HIDDEN_SIZE, dtype=np.float32))
                            else:
                                rescue_deltas.append(rescue_base_vector(str(condition["kind"]), group, state, d_active, d_surface, d_null, d_neighbor, condition.get("random_index")))
                            index = int(pair["pair_index"])
                            clean_targets.append(vectors[index, state, depth_index[RESCUE_DEPTH]].astype(np.float32))
                            clean_signatures.append([vectors[index, state, depth_index[d]].astype(np.float32) for d in SIGNATURE_DEPTHS])
                        ids = torch.tensor(input_rows, dtype=torch.long, device=device)
                        mask = torch.ones_like(ids)
                        pos = torch.tensor(positions, dtype=torch.long, device=device)
                        damage_tensor = torch.from_numpy(np.stack(damage_deltas)).to(device=device, dtype=torch.float16)
                        rescue_tensor = None if condition["kind"] == "clean_clamp" else torch.from_numpy(np.stack(rescue_deltas)).to(device=device, dtype=torch.float16)
                        target_tensor = torch.from_numpy(np.stack(clean_targets)).to(device=device, dtype=torch.float16) if condition["kind"] == "clean_clamp" else None
                        with CascadePatchCapture(layers, damage_depth, pos, damage_tensor, rescue_tensor, target_tensor) as hooks:
                            output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, logits_to_keep=1)
                            hooks.validate()
                        scores, finite = score_batch(output.logits, rows)
                        for slot, (pair, state) in enumerate(batch):
                            observed = [hooks.signature[d][slot].cpu().numpy() for d in SIGNATURE_DEPTHS]
                            row = record_common(pair, state, condition, baseline_scores, baseline_finite, scores[slot], bool(finite[slot]), float(np.linalg.norm(rescue_deltas[slot])), RESCUE_DEPTH)
                            row.update({
                                "record_id": f"{pair['group_id']}|damage{damage_depth}|r{state}|{condition['id']}",
                                "damage_depth": damage_depth,
                                "rescue_depth": RESCUE_DEPTH,
                                "damage_delta_l2": float(np.linalg.norm(damage_deltas[slot])),
                                "response_error_to_clean": signature_error(observed, clean_signatures[slot]),
                            })
                            records.append(row)
                        del output, ids, mask, pos, damage_tensor, rescue_tensor, target_tensor
                print(canonical({"phase": PHASE, "stage": "rescue", "condition": condition["id"], "records": len(records)}), flush=True)
        if len(records) != protocol["counts"]["rescue_records_if_authorized"] or not all(row["patched_finite"] for row in records):
            raise RuntimeError("Phase1207 rescue completeness/finiteness failed")
        write_jsonl_gz(RESCUE_RAW_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1207.rescue_run.v1",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "necessity_verdict_digest": necessity["verdict_digest"],
            "damage_depth": damage_depth,
            "rescue_depth": RESCUE_DEPTH,
            "record_count": len(records),
            "raw_file_sha256": sha256_file(RESCUE_RAW_PATH),
            "raw_digest": digest(records),
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = digest(summary)
        write_json(RESCUE_SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()


def enrich_rescue(row: dict[str, Any], damage: dict[str, Any]) -> dict[str, Any]:
    labels = list(row["candidate_labels"])
    recipient_index = labels.index(str(row["recipient_gold"]))
    donor_index = labels.index(str(row["donor_gold"]))
    base_scores = np.asarray(row["recipient_scores"], dtype=np.float64)
    damage_scores = np.asarray(damage["patched_scores"], dtype=np.float64)
    rescued_scores = np.asarray(row["patched_scores"], dtype=np.float64)
    base_margin = float(base_scores[recipient_index] - base_scores[donor_index])
    damaged_margin = float(damage_scores[recipient_index] - damage_scores[donor_index])
    rescued_margin = float(rescued_scores[recipient_index] - rescued_scores[donor_index])
    lost = base_margin - damaged_margin
    recovery = (rescued_margin - damaged_margin) / (lost + EPSILON)
    damage_error = float(damage["response_error_to_clean"])
    response_recovery = 1.0 - float(row["response_error_to_clean"]) / (damage_error + EPSILON)
    return {
        **row,
        "damage_prediction": damage["patched_prediction"],
        "damage_behavior": damage["patched_prediction"] != row["recipient_gold"],
        "behavior_restored": row["patched_prediction"] == row["recipient_gold"],
        "base_margin": base_margin,
        "damaged_margin": damaged_margin,
        "rescued_margin": rescued_margin,
        "margin_recovery": recovery,
        "positive_margin_recovery": recovery > 0,
        "damage_response_error": damage_error,
        "response_recovery": response_recovery,
    }


def rescue_metrics(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    members = [row for row in rows if row["split"] == split]
    lookup = {(row["group_id"], int(row["recipient_state"]), row["condition"]): row for row in members}
    target = [row for row in members if row["condition"] == "specific_addback"]
    damaged = [row for row in target if row["damage_behavior"]]
    margin_advantage: list[float] = []
    response_advantage: list[float] = []
    for row in target:
        controls = [lookup[(row["group_id"], int(row["recipient_state"]), name)] for name in RESCUE_PRIMARY_CONTROLS]
        margin_advantage.append(float(row["margin_recovery"]) - max(float(control["margin_recovery"]) for control in controls))
        response_advantage.append(float(row["response_recovery"]) - max(float(control["response_recovery"]) for control in controls))
    directions: dict[str, dict[str, float]] = {}
    for state in (0, 1):
        subset = [row for row in damaged if int(row["recipient_state"]) == state]
        directions[f"state{state}"] = {"count": len(subset), "restore_fraction": sum(bool(row["behavior_restored"]) for row in subset) / max(len(subset), 1)}
    clamps = [row for row in members if row["condition"] == "clean_state_clamp"]
    result = {
        "split": split,
        "target_count": len(target),
        "damaged_count": len(damaged),
        "finite_fraction": sum(bool(row["patched_finite"]) for row in target) / max(len(target), 1),
        "behavior_restore_fraction": sum(bool(row["behavior_restored"]) for row in damaged) / max(len(damaged), 1),
        "median_margin_recovery": median(row["margin_recovery"] for row in target),
        "positive_margin_recovery_fraction": sum(bool(row["positive_margin_recovery"]) for row in target) / max(len(target), 1),
        "median_response_recovery": median(row["response_recovery"] for row in target),
        "margin_beats_all_controls_fraction": sum(value > 0 for value in margin_advantage) / max(len(margin_advantage), 1),
        "response_beats_all_controls_fraction": sum(value > 0 for value in response_advantage) / max(len(response_advantage), 1),
        "median_margin_advantage": median(margin_advantage),
        "direction_restore": directions,
        "clean_clamp_restore_fraction": sum(row["patched_prediction"] == row["recipient_gold"] for row in clamps) / max(len(clamps), 1),
        "clean_clamp_median_response_recovery": median(row["response_recovery"] for row in clamps),
    }
    t = RESCUE_THRESHOLDS
    result["pass"] = bool(
        result["finite_fraction"] >= t["finite_fraction"]
        and result["damaged_count"] >= t["minimum_damaged_records_per_split"]
        and all(value["count"] >= t["minimum_damaged_records_per_direction"] for value in directions.values())
        and result["behavior_restore_fraction"] >= t["behavior_restore_fraction"]
        and result["median_margin_recovery"] >= t["median_margin_recovery"]
        and result["positive_margin_recovery_fraction"] >= t["positive_margin_recovery_fraction"]
        and result["median_response_recovery"] >= t["median_response_recovery"]
        and result["margin_beats_all_controls_fraction"] >= t["margin_beats_all_controls_fraction"]
        and result["response_beats_all_controls_fraction"] >= t["response_beats_all_controls_fraction"]
        and result["median_margin_advantage"] >= t["median_margin_advantage"]
        and min(value["restore_fraction"] for value in directions.values()) >= t["minimum_each_direction_restore"]
        and result["clean_clamp_restore_fraction"] >= t["clean_clamp_restore_fraction"]
        and result["clean_clamp_median_response_recovery"] >= t["clean_clamp_median_response_recovery"]
    )
    return result


def analyze_rescue_command() -> None:
    protocol = verify_protocol()
    if RESCUE_VERDICT_PATH.exists():
        raise RuntimeError("rescue verdict exists")
    necessity = read_json(NECESSITY_VERDICT_PATH)
    summary = read_json(RESCUE_SUMMARY_PATH)
    validate_digest(necessity, "verdict_digest")
    validate_digest(summary, "summary_digest")
    if sha256_file(RESCUE_RAW_PATH) != summary["raw_file_sha256"]:
        raise RuntimeError("rescue raw drift")
    raw = read_jsonl_gz(RESCUE_RAW_PATH)
    if digest(raw) != summary["raw_digest"]:
        raise RuntimeError("rescue semantic drift")
    damage = {(row["group_id"], int(row["recipient_state"])): row for row in raw if row["condition"] == "damage_only"}
    rows = [enrich_rescue(row, damage[(row["group_id"], int(row["recipient_state"]))]) for row in raw]
    metrics = {split: rescue_metrics(rows, split) for split in SPLITS}
    gate = all(metrics[split]["pass"] for split in SPLITS)
    verdict: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1207.rescue_verdict.v1",
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "necessity_verdict_digest": necessity["verdict_digest"],
        "rescue_summary_digest": summary["summary_digest"],
        "metrics": metrics,
        "rescue_gate": gate,
        "status": "qwen3_same_sample_rescue_qualified" if gate else "qwen3_same_sample_rescue_not_qualified",
        "authorization": {"component_search": False, "phase1208_preregistration": gate},
    }
    verdict["verdict_digest"] = digest(verdict)
    write_json(RESCUE_VERDICT_PATH, verdict)
    print(json.dumps({"metrics": metrics, "rescue_gate": gate, "digest": verdict["verdict_digest"]}, ensure_ascii=False, indent=2))


def finalize_command() -> None:
    protocol = verify_protocol()
    audit = read_json(RESULT_AUDIT_PATH)
    validate_digest(audit, "audit_digest")
    if not audit["gate_pass"]:
        raise RuntimeError("Phase1207 independent audit failed")
    onset = read_json(ONSET_VERDICT_PATH)
    validate_digest(onset, "verdict_digest")
    necessity = read_json(NECESSITY_VERDICT_PATH) if NECESSITY_VERDICT_PATH.exists() else None
    rescue = read_json(RESCUE_VERDICT_PATH) if RESCUE_VERDICT_PATH.exists() else None
    if necessity is not None:
        validate_digest(necessity, "verdict_digest")
    if rescue is not None:
        validate_digest(rescue, "verdict_digest")
    onset_gate = bool(onset["onset_gate"])
    necessity_gate = bool(necessity and necessity["necessity_gate"])
    rescue_gate = bool(rescue and rescue["rescue_gate"])
    if not onset_gate:
        status = "causal_onset_not_qualified"
        statement = "No preregistered adjacent causal-onset band repeated across all splits; necessity and rescue were not tested."
    elif not necessity_gate:
        status = "causal_onset_qualified_necessity_failed"
        statement = (
            f"The earliest repeated full-state causal onset was depth {onset['selected_depth']}, but replacing the "
            "active pair differential with the surface-only differential did not satisfy the cross-split necessity gate; rescue was not tested."
        )
    elif rescue is None:
        status = "necessity_qualified_rescue_not_testable"
        statement = "Active-vs-surface contrast necessity qualified, but same-depth rescue was structurally untestable under the frozen protocol."
    elif not rescue_gate:
        status = "necessity_qualified_rescue_failed"
        statement = "Active-vs-surface contrast necessity qualified, but the frozen depth25 specific addback did not satisfy the same-sample behavior, margin, response, and control gates."
    else:
        status = "causal_onset_necessity_rescue_qualified"
        statement = "The frozen Qwen3 causal onset, active-vs-surface contrast necessity, and downstream same-sample rescue all repeated across the three splits under matched controls."
    final: dict[str, Any] = {
        "phase": PHASE,
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "independent_result_audit_digest": audit["audit_digest"],
        "onset": {"gate": onset_gate, "selected_depth": onset["selected_depth"], "verdict_digest": onset["verdict_digest"]},
        "necessity": None if necessity is None else {"gate": necessity_gate, "verdict_digest": necessity["verdict_digest"]},
        "rescue": None if rescue is None else {"gate": rescue_gate, "verdict_digest": rescue["verdict_digest"]},
        "new_k_item": {"id": "K187", "scope": "Qwen3 controlled causal ancestry and necessity boundary", "statement": statement},
        "evidence_scope": {
            "qwen3_only": True,
            "controlled_object_attribute": True,
            "full_state_causal_onset": onset_gate,
            "active_surface_contrast_necessity": necessity_gate,
            "same_sample_rescue": rescue_gate,
            "minimal_implementation": False,
            "natural_use": False,
            "cross_model": False,
            "brain": False,
            "mechanism_closure": False,
        },
        "authorized_next": {
            "phase1208_causal_ancestry_preregistration": rescue_gate,
            "automatic_phase1208_execution": False,
            "head_or_neuron_search": False,
            "natural_use": False,
            "cross_model": False,
        },
        "stop_rule": (
            "Only a separate zero-output causal-ancestry mediation preregistration is authorized; no automatic component search."
            if rescue_gate else
            "The failed or untested downstream gate closes automatic continuation on this operation; do not retune depth, contrast, threshold, or rescue."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "protocol", "capture", "run-onset", "analyze-onset", "run-necessity",
            "analyze-necessity", "run-rescue", "analyze-rescue", "finalize",
        ),
    )
    args = parser.parse_args()
    commands = {
        "protocol": protocol_command,
        "capture": capture_command,
        "run-onset": run_onset_command,
        "analyze-onset": analyze_onset_command,
        "run-necessity": run_necessity_command,
        "analyze-necessity": analyze_necessity_command,
        "run-rescue": run_rescue_command,
        "analyze-rescue": analyze_rescue_command,
        "finalize": finalize_command,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
