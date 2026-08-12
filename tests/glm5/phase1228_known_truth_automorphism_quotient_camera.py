#!/usr/bin/env python3
"""Phase 1228: known-truth automorphism-aware functional quotient camera.

The phase fixes the missing Phase1226 calibration.  Gauge variants are
physically different modules and hidden states while remaining exactly equal
under the registered public response basis.  Other systems contain exact role
symmetries, near-but-distinguishable roles, and hidden variants that become
distinguishable only after a sealed basis extension.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1228
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1228_known_truth_automorphism_quotient_camera_audit.py"
SOURCE_ROOT = TEST_ROOT / "result/phase1227_qwen3_teacher_forced_role_coalition"
SOURCE_FINAL = SOURCE_ROOT / "analysis/final.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit/independent_result_audit.json"
EXPECTED_SOURCE_FINAL = "127d50a4b991d755cc4307535ac7c46327267723009ea63543145e687228b298"
EXPECTED_SOURCE_AUDIT = "412bc4ef43f65e2b97d6d3f3a6d4f051091cbfa60d881d51db9f146b2cbe1d14"

ABORTED_ROOT = TEST_ROOT / "result/phase1228_known_truth_automorphism_quotient_camera"
ABORTED_PREAUDIT = ABORTED_ROOT / "audit/independent_preaudit.json"
EXPECTED_ABORTED_PREAUDIT = "b8b394364ff760aa2eb19b6aae75d9f43650a0979e50f9cff8860ac3df989861"
OUT_ROOT = TEST_ROOT / "result/phase1228_known_truth_automorphism_quotient_camera_revision1"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MANIFEST_PATH = OUT_ROOT / "protocol/system_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
DISCOVERY_PUBLIC = OUT_ROOT / "discovery/public_responses.jsonl"
DISCOVERY_TRUTH = OUT_ROOT / "discovery/truth.jsonl"
CAMERA_PATH = OUT_ROOT / "protocol/frozen_camera.json"
CONFIRMATION_PUBLIC = OUT_ROOT / "confirmation/public_responses.jsonl"
CONFIRMATION_SEALED = OUT_ROOT / "confirmation/sealed_truth.jsonl"
PREDICTION_PATH = OUT_ROOT / "confirmation/predictions.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

ROLES = ("R", "Q", "B")
ALLIANCES = ("R", "Q", "B", "RQ", "RB", "QB", "RQB")
PERMUTATIONS = tuple(itertools.permutations(ROLES))
FAMILIES = (
    "r_gate",
    "q_gate",
    "b_gate",
    "cardinality_symmetric",
    "rq_joint",
    "fully_asymmetric",
    "near_qb_distinguishable",
)
GAUGES = ("u", "v")
HIDDEN_VARIANTS = ("h0", "h1")
SPLITS = ("discovery", "confirmation")
REPLICATES = 48
WIDTH = 8
EQUIVALENCE_TOLERANCE = 0.004
SUFFICIENT_THRESHOLD = 0.90
SEALED_HIDDEN_THRESHOLD = 0.125
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "baseline_max_abs_max": 1e-4,
    "full_response_min": 0.999,
    "structure_accuracy_min": 1.0,
    "stabilizer_accuracy_min": 1.0,
    "orbit_accuracy_min": 1.0,
    "minimum_alliance_accuracy_min": 1.0,
    "near_distinguishable_accuracy_min": 1.0,
    "gauge_public_abstention_min": 1.0,
    "hidden_public_abstention_min": 1.0,
    "sealed_hidden_accuracy_min": 1.0,
    "sealed_gauge_abstention_min": 1.0,
    "physical_state_dict_difference_fraction_min": 1.0,
    "physical_hidden_difference_fraction_min": 1.0,
    "gauge_public_profile_max_abs_max": 0.0,
    "gauge_sealed_profile_max_abs_max": 0.0,
    "hidden_public_profile_max_abs_max": 0.0,
    "hidden_sealed_difference_min": 0.20,
    "metadata_null_accuracy_exact": 0.50,
    "leaky_sentinel_accuracy_exact": 1.0,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def tensor_dict_digest(values: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(values):
        tensor = values[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(str(tuple(tensor.shape)).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(str(tuple(tensor.shape)).encode("ascii"))
    hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def validate_source() -> None:
    final = read_json(SOURCE_FINAL)
    audit = read_json(SOURCE_AUDIT)
    if final.get("final_digest") != EXPECTED_SOURCE_FINAL:
        raise RuntimeError("Phase1227 final digest drift")
    if audit.get("audit_digest") != EXPECTED_SOURCE_AUDIT or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1227 audit drift")


def alliance_bits(name: str) -> tuple[int, int, int]:
    return tuple(int(role in name) for role in ROLES)  # type: ignore[return-value]


def canonical_alliance(roles: Iterable[str]) -> str:
    chosen = set(roles)
    return "".join(role for role in ROLES if role in chosen)


def permutation_id(permutation: tuple[str, str, str]) -> str:
    return "".join(permutation)


def permute_alliance(name: str, permutation: tuple[str, str, str]) -> str:
    mapping = dict(zip(ROLES, permutation))
    return canonical_alliance(mapping[role] for role in name)


def base_value(family: str, r: int, q: int, b: int) -> float:
    if family == "r_gate":
        return float(r)
    if family == "q_gate":
        return float(q)
    if family == "b_gate":
        return float(b)
    if family == "cardinality_symmetric":
        return float(r + q + b) / 3.0
    if family == "rq_joint":
        return float(r * q)
    if family == "fully_asymmetric":
        return 0.55 * r + 0.30 * q + 0.15 * b
    if family == "near_qb_distinguishable":
        return 0.58 * r + 0.215 * q + 0.205 * b
    raise ValueError(family)


def base_profile(family: str) -> dict[str, float]:
    return {name: base_value(family, *alliance_bits(name)) for name in ALLIANCES}


def exact_stabilizer(family: str) -> tuple[str, ...]:
    profile = base_profile(family)
    values: list[str] = []
    for permutation in PERMUTATIONS:
        error = max(abs(profile[name] - profile[permute_alliance(name, permutation)]) for name in ALLIANCES)
        if error <= 1e-12:
            values.append(permutation_id(permutation))
    return tuple(sorted(values))


def role_orbits(stabilizer: Iterable[str]) -> list[list[str]]:
    permutations = [tuple(value) for value in stabilizer]
    remaining = set(ROLES)
    orbits: list[list[str]] = []
    while remaining:
        seed = min(remaining, key=ROLES.index)
        orbit = {seed}
        changed = True
        while changed:
            changed = False
            for role in list(orbit):
                index = ROLES.index(role)
                for permutation in permutations:
                    image = permutation[index]
                    if image not in orbit:
                        orbit.add(image)
                        changed = True
        ordered = sorted(orbit, key=ROLES.index)
        orbits.append(ordered)
        remaining -= orbit
    return sorted(orbits, key=lambda values: tuple(ROLES.index(value) for value in values))


def minimum_alliances(profile: dict[str, float]) -> list[str]:
    sufficient = [name for name in ALLIANCES if profile[name] >= SUFFICIENT_THRESHOLD]
    if not sufficient:
        return []
    minimum = min(len(name) for name in sufficient)
    return sorted((name for name in sufficient if len(name) == minimum), key=ALLIANCES.index)


def transformed_table(family: str, curvature: float) -> np.ndarray:
    table = np.zeros((2, 2, 2), dtype=np.float32)
    for r, q, b in itertools.product((0, 1), repeat=3):
        value = base_value(family, r, q, b)
        table[r, q, b] = value + curvature * value * (1.0 - value)
    return table


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    split: str
    family: str
    replicate: int
    gauge_variant: str
    hidden_variant: str
    curvature: float
    slot_by_role: dict[str, int]
    channel_permutation: tuple[int, ...]
    channel_signs: tuple[int, ...]
    public_nonce: int


def system_spec(split: str, family: str, replicate: int, gauge: str, hidden: str) -> SystemSpec:
    split_offset = 0 if split == "discovery" else 100_003
    family_index = FAMILIES.index(family)
    gauge_index = GAUGES.index(gauge)
    hidden_index = HIDDEN_VARIANTS.index(hidden)
    slot_permutations = list(itertools.permutations(range(3)))
    slot_index = (replicate + 2 * family_index + 3 * gauge_index + split_offset) % len(slot_permutations)
    slot_order = slot_permutations[slot_index]
    slot_by_role = {role: int(slot_order[index]) for index, role in enumerate(ROLES)}
    seed = 12280019 + split_offset + 1009 * replicate + 97 * family_index + 7919 * gauge_index
    rng = np.random.default_rng(seed)
    channel_permutation = tuple(int(value) for value in rng.permutation(WIDTH))
    channel_signs = tuple(int(value) for value in rng.choice(np.asarray([-1, 1]), size=WIDTH))
    curvature_grid = (-0.08, -0.04, 0.00, 0.04, 0.08) if split == "discovery" else (-0.07, -0.03, 0.01, 0.05, 0.09)
    curvature = float(curvature_grid[(replicate + family_index) % len(curvature_grid)])
    identity = {
        "phase": PHASE,
        "split": split,
        "family": family,
        "replicate": replicate,
        "gauge": gauge,
        "hidden": hidden,
    }
    return SystemSpec(
        system_id=digest(identity)[:24],
        split=split,
        family=family,
        replicate=replicate,
        gauge_variant=gauge,
        hidden_variant=hidden,
        curvature=curvature,
        slot_by_role=slot_by_role,
        channel_permutation=channel_permutation,
        channel_signs=channel_signs,
        public_nonce=replicate % 2,
    )


def all_specs(split: str) -> list[SystemSpec]:
    return [
        system_spec(split, family, replicate, gauge, hidden)
        for family in FAMILIES
        for replicate in range(REPLICATES)
        for gauge in GAUGES
        for hidden in HIDDEN_VARIANTS
    ]


class PhysicalGaugeResponseTransformer(nn.Module):
    """Exact role router and ReLU response table with a cancellable gauge."""

    def __init__(self, spec: SystemSpec, device: torch.device) -> None:
        super().__init__()
        routing = torch.zeros((3, 3), dtype=torch.float16, device=device)
        for role_index, role in enumerate(ROLES):
            routing[role_index, spec.slot_by_role[role]] = 1.0
        permutation = torch.tensor(spec.channel_permutation, dtype=torch.long, device=device)
        signs = torch.tensor(spec.channel_signs, dtype=torch.float16, device=device)
        table = torch.tensor(transformed_table(spec.family, spec.curvature), dtype=torch.float16, device=device)
        sealed_head = torch.tensor(0.0 if spec.hidden_variant == "h0" else 0.25, dtype=torch.float16, device=device)
        self.register_buffer("routing", routing)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.register_buffer("signs", signs)
        self.register_buffer("public_table", table)
        self.register_buffer("sealed_head", sealed_head)

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        logical = torch.nn.functional.one_hot(bits.long(), WIDTH).to(torch.float16)
        return logical[:, self.permutation] * self.signs[None, :]

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return (encoded * self.signs[None, :])[:, self.inverse_permutation]

    def bundle(self, logical_bits: torch.Tensor) -> torch.Tensor:
        batch = logical_bits.shape[0]
        physical = torch.zeros((batch, 3, WIDTH), dtype=torch.float16, device=logical_bits.device)
        for role_index, role in enumerate(ROLES):
            physical[:, self.spec_slot(role), :] = self.encode(logical_bits[:, role_index])
        return physical

    def spec_slot(self, role: str) -> int:
        role_index = ROLES.index(role)
        return int(torch.argmax(self.routing[role_index]).item())

    def public_response(self, physical_state: torch.Tensor) -> torch.Tensor:
        logical_roles = torch.einsum("rs,bsw->brw", self.routing, physical_state)
        decoded = [self.decode(logical_roles[:, index])[:, :2] for index in range(3)]
        conjunction = torch.relu(
            decoded[0][:, :, None, None]
            + decoded[1][:, None, :, None]
            + decoded[2][:, None, None, :]
            - 2.0
        )
        return torch.einsum("bijk,ijk->b", conjunction, self.public_table)

    def sealed_probe_response(self) -> torch.Tensor:
        return self.sealed_head.clone()


def logical_masks(device: torch.device) -> torch.Tensor:
    values = [[0, 0, 0]] + [list(alliance_bits(name)) for name in ALLIANCES]
    return torch.tensor(values, dtype=torch.long, device=device)


def evaluate_spec(spec: SystemSpec, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    model = PhysicalGaugeResponseTransformer(spec, device).eval()
    masks = logical_masks(device)
    with torch.inference_mode():
        states = model.bundle(masks)
        responses = model.public_response(states).float()
        sealed = float(model.sealed_probe_response().float().item())
    baseline = float(responses[0].item())
    full = float(responses[-1].item())
    denominator = full - baseline
    if abs(denominator) <= EPSILON:
        raise RuntimeError("zero public response range")
    profile = {
        name: float((responses[index + 1].item() - baseline) / denominator)
        for index, name in enumerate(ALLIANCES)
    }
    public = {
        "schema_version": "phase1228.public.v1",
        "phase": PHASE,
        "system_id": spec.system_id,
        "split": spec.split,
        "public_nonce": spec.public_nonce,
        "baseline": baseline,
        "full_response": full,
        "profile": profile,
        "finite": all(math.isfinite(value) for value in (baseline, full, sealed, *profile.values())),
    }
    public["public_digest"] = digest(public)
    state_dict_hash = tensor_dict_digest(dict(model.state_dict()))
    hidden_hash = tensor_digest(states)
    truth = {
        "schema_version": "phase1228.truth.v1",
        "phase": PHASE,
        "system_id": spec.system_id,
        "split": spec.split,
        "family": spec.family,
        "replicate": spec.replicate,
        "gauge_variant": spec.gauge_variant,
        "hidden_variant": spec.hidden_variant,
        "curvature": spec.curvature,
        "slot_by_role": spec.slot_by_role,
        "state_dict_digest": state_dict_hash,
        "hidden_state_digest": hidden_hash,
        "truth_stabilizer": list(exact_stabilizer(spec.family)),
        "truth_orbits": role_orbits(exact_stabilizer(spec.family)),
        "truth_minimum_alliances": minimum_alliances(profile),
        "sealed_probe_response": sealed,
        "leaky_sentinel_feature": 0 if spec.gauge_variant == "u" else 1,
    }
    truth["truth_digest"] = digest(truth)
    return public, truth


def infer_camera(profile: dict[str, float]) -> dict[str, Any]:
    errors: dict[str, float] = {}
    stabilizer: list[str] = []
    for permutation in PERMUTATIONS:
        identifier = permutation_id(permutation)
        error = max(abs(profile[name] - profile[permute_alliance(name, permutation)]) for name in ALLIANCES)
        errors[identifier] = float(error)
        if error <= EQUIVALENCE_TOLERANCE:
            stabilizer.append(identifier)
    stabilizer = sorted(stabilizer)
    return {
        "inferred_stabilizer": stabilizer,
        "inferred_orbits": role_orbits(stabilizer),
        "minimum_alliances": minimum_alliances(profile),
        "permutation_errors": errors,
        "gauge_variant_decision": "ABSTAIN",
        "hidden_variant_decision_public": "ABSTAIN",
    }


def manifest_row(spec: SystemSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "phase1228.system-manifest.v1",
        "phase": PHASE,
        "system_id": spec.system_id,
        "split": spec.split,
        "family": spec.family,
        "replicate": spec.replicate,
        "gauge_variant": spec.gauge_variant,
        "hidden_variant": spec.hidden_variant,
        "curvature": spec.curvature,
        "slot_by_role": spec.slot_by_role,
        "channel_permutation": list(spec.channel_permutation),
        "channel_signs": list(spec.channel_signs),
        "public_nonce": spec.public_nonce,
    }
    row["row_digest"] = digest(row)
    return row


def spec_from_row(row: dict[str, Any]) -> SystemSpec:
    return SystemSpec(
        system_id=row["system_id"],
        split=row["split"],
        family=row["family"],
        replicate=int(row["replicate"]),
        gauge_variant=row["gauge_variant"],
        hidden_variant=row["hidden_variant"],
        curvature=float(row["curvature"]),
        slot_by_role={key: int(value) for key, value in row["slot_by_role"].items()},
        channel_permutation=tuple(int(value) for value in row["channel_permutation"]),
        channel_signs=tuple(int(value) for value in row["channel_signs"]),
        public_nonce=int(row["public_nonce"]),
    )


def materialize() -> None:
    validate_source()
    aborted = read_json(ABORTED_PREAUDIT)
    if aborted.get("all_checks_passed") or aborted.get("audit_digest") != EXPECTED_ABORTED_PREAUDIT:
        raise RuntimeError("Phase1228 aborted preaudit provenance drift")
    if OUT_ROOT.exists():
        raise RuntimeError(f"formal output already exists: {OUT_ROOT}")
    manifest = [manifest_row(spec) for split in SPLITS for spec in all_specs(split)]
    protocol: dict[str, Any] = {
        "schema_version": "phase1228.revision1.preregistration.v1",
        "phase": PHASE,
        "revision": 1,
        "created_at_utc": utc_now(),
        "objective": (
            "Calibrate an automorphism-aware response camera on physically distinct exact-equivalent gauges, "
            "exact role symmetries, near-but-distinguishable roles, and basis-dependent hidden variants."
        ),
        "authorization": "Revision 1 after the original preaudit stopped at 30/31 on an AST false positive",
        "revision_reason": (
            "The original audit rejected any mention of CONFIRMATION_SEALED, including the isolation guard "
            "CONFIRMATION_SEALED.exists(). Revision 1 forbids only read_json/read_jsonl access to sealed truth "
            "inside prediction and retains the aborted 30/31 artifact unchanged."
        ),
        "aborted_preflight": {
            "path": str(ABORTED_PREAUDIT.relative_to(ROOT)),
            "audit_digest": EXPECTED_ABORTED_PREAUDIT,
            "passed": 30,
            "total": 31,
        },
        "numerical_type": {
            "device": "CUDA required",
            "dtype": "FP16 buffers and hidden states; FP32 metric extraction",
            "width": WIDTH,
            "fixed_batch_geometry": [8, 3, WIDTH],
            "free_training": False,
            "pretrained_language_model": False,
        },
        "roles": list(ROLES),
        "alliances": list(ALLIANCES),
        "families": list(FAMILIES),
        "gauge_variants": list(GAUGES),
        "hidden_variants": list(HIDDEN_VARIANTS),
        "splits": list(SPLITS),
        "replicates": REPLICATES,
        "systems_per_split": len(all_specs("discovery")),
        "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "tolerance_origin": (
            "fixed before any Phase1228 response; independent of the revealed Phase1227 Q/B difference 0.0021"
        ),
        "sufficient_threshold": SUFFICIENT_THRESHOLD,
        "sealed_hidden_threshold": SEALED_HIDDEN_THRESHOLD,
        "camera_contract": {
            "exact_stabilizer": "all role permutations with max seven-alliance error <= fixed tolerance",
            "identifiable_object": "role orbits under the inferred stabilizer",
            "public_gauge_identity": "ABSTAIN",
            "public_hidden_identity": "ABSTAIN",
            "sealed_hidden_rule": "h1 iff sealed probe response > 0.125, else h0",
            "sealed_gauge_identity": "ABSTAIN",
            "near_family": "near_qb_distinguishable must have identity stabilizer",
        },
        "thresholds": THRESHOLDS,
        "split_discipline": {
            "discovery_truth_may_be_used_only_for_camera qualification": True,
            "confirmation_prediction_reads_only public responses and frozen camera": True,
            "confirmation sealed truth is opened only by reveal stage": True,
        },
        "claim_scope": [
            "This is a constructed known-truth camera calibration, not a language-mechanism result.",
            "Exact functional equivalence is relative to the registered public response basis.",
            "The sealed probe demonstrates that public equivalence need not survive basis extension.",
            "A pass authorizes deconfounded material design, not immediate hidden-state claims in Qwen3.",
        ],
        "prohibited": [
            "change tolerance after observing responses",
            "identify gauge from system id or state_dict digest",
            "call approximate similarity strict equivalence",
            "read confirmation sealed truth during prediction",
            "load Qwen3, GLM4, or DS7B",
        ],
        "source": {
            "phase1227_final_digest": EXPECTED_SOURCE_FINAL,
            "phase1227_audit_digest": EXPECTED_SOURCE_AUDIT,
            "phase1227_final_sha256": file_sha256(SOURCE_FINAL),
            "phase1227_audit_sha256": file_sha256(SOURCE_AUDIT),
        },
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
        },
        "material": {"count": len(manifest), "digest": digest(manifest)},
    }
    protocol["protocol_digest"] = digest(protocol)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"status": "materialized", "systems": len(manifest), "protocol_digest": protocol["protocol_digest"]}))


def verify_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_source()
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("preaudit did not pass")
    if protocol["protocol_digest"] != digest({key: value for key, value in protocol.items() if key != "protocol_digest"}):
        raise RuntimeError("protocol digest drift")
    if protocol["source_hashes"]["main"] != file_sha256(SCRIPT) or protocol["source_hashes"]["audit"] != file_sha256(AUDIT_SCRIPT):
        raise RuntimeError("source changed after freeze")
    if protocol["material"]["digest"] != digest(manifest):
        raise RuntimeError("manifest digest drift")
    return protocol, manifest


def run_split(split: str) -> None:
    protocol, manifest = verify_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    public_path = DISCOVERY_PUBLIC if split == "discovery" else CONFIRMATION_PUBLIC
    truth_path = DISCOVERY_TRUTH if split == "discovery" else CONFIRMATION_SEALED
    if public_path.exists() or truth_path.exists():
        raise RuntimeError(f"{split} outputs already exist")
    device = torch.device("cuda:0")
    public_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    selected = [row for row in manifest if row["split"] == split]
    started = time.time()
    for index, row in enumerate(selected):
        public, truth = evaluate_spec(spec_from_row(row), device)
        public_rows.append(public)
        truth_rows.append(truth)
        if (index + 1) % 224 == 0 or index + 1 == len(selected):
            print(canonical_json({"split": split, "completed": index + 1, "total": len(selected)}), flush=True)
    write_jsonl(public_path, public_rows)
    if split == "discovery":
        write_jsonl(truth_path, truth_rows)
    summary_path = OUT_ROOT / "runs" / f"{split}_summary.json"
    summary: dict[str, Any] = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "count": len(public_rows),
        "public_digest": digest(public_rows),
        "truth_digest": digest(truth_rows),
        "truth_storage": "written" if split == "discovery" else "commitment_only_until_reveal",
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
    summary["summary_digest"] = digest(summary)
    write_json(summary_path, summary)
    torch.cuda.empty_cache()


def score_rows(public_rows: list[dict[str, Any]], truth_rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth_by_id = {row["system_id"]: row for row in truth_rows}
    predictions = [{"system_id": row["system_id"], **infer_camera(row["profile"])} for row in public_rows]
    stabilizer_ok: list[bool] = []
    orbit_ok: list[bool] = []
    minimum_ok: list[bool] = []
    near_ok: list[bool] = []
    metadata_correct: list[bool] = []
    leaky_sentinel_correct: list[bool] = []
    for public, prediction in zip(public_rows, predictions):
        truth = truth_by_id[public["system_id"]]
        stabilizer_ok.append(prediction["inferred_stabilizer"] == truth["truth_stabilizer"])
        orbit_ok.append(prediction["inferred_orbits"] == truth["truth_orbits"])
        minimum_ok.append(prediction["minimum_alliances"] == truth["truth_minimum_alliances"])
        if truth["family"] == "near_qb_distinguishable":
            near_ok.append(prediction["inferred_stabilizer"] == ["RQB"])
        metadata_guess = "u" if int(public["public_nonce"]) == 0 else "v"
        metadata_correct.append(metadata_guess == truth["gauge_variant"])
        leaky_guess = "u" if int(truth["leaky_sentinel_feature"]) == 0 else "v"
        leaky_sentinel_correct.append(leaky_guess == truth["gauge_variant"])
    finite_fraction = float(np.mean([row["finite"] for row in public_rows]))
    return {
        "count": len(public_rows),
        "finite_fraction": finite_fraction,
        "baseline_max_abs": float(max(abs(float(row["baseline"])) for row in public_rows)),
        "full_response_min": float(min(float(row["full_response"]) for row in public_rows)),
        "structure_accuracy": float(np.mean([a and b and c for a, b, c in zip(stabilizer_ok, orbit_ok, minimum_ok)])),
        "stabilizer_accuracy": float(np.mean(stabilizer_ok)),
        "orbit_accuracy": float(np.mean(orbit_ok)),
        "minimum_alliance_accuracy": float(np.mean(minimum_ok)),
        "near_distinguishable_accuracy": float(np.mean(near_ok)),
        "gauge_public_abstention": 1.0,
        "hidden_public_abstention": 1.0,
        "metadata_null_accuracy": float(np.mean(metadata_correct)),
        "leaky_sentinel_accuracy": float(np.mean(leaky_sentinel_correct)),
        "predictions": predictions,
    }


def physical_pair_metrics(public_rows: list[dict[str, Any]], truth_rows: list[dict[str, Any]]) -> dict[str, Any]:
    public_by_id = {row["system_id"]: row for row in public_rows}
    grouped: dict[tuple[Any, ...], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for truth in truth_rows:
        key = (truth["split"], truth["family"], truth["replicate"])
        grouped[key][(truth["gauge_variant"], truth["hidden_variant"])] = truth
    gauge_state_diff: list[bool] = []
    gauge_hidden_diff: list[bool] = []
    gauge_public_diff: list[float] = []
    gauge_sealed_diff: list[float] = []
    hidden_state_diff: list[bool] = []
    hidden_public_diff: list[float] = []
    hidden_sealed_diff: list[float] = []
    for values in grouped.values():
        for hidden in HIDDEN_VARIANTS:
            left = values[("u", hidden)]
            right = values[("v", hidden)]
            gauge_state_diff.append(left["state_dict_digest"] != right["state_dict_digest"])
            gauge_hidden_diff.append(left["hidden_state_digest"] != right["hidden_state_digest"])
            lp = public_by_id[left["system_id"]]["profile"]
            rp = public_by_id[right["system_id"]]["profile"]
            gauge_public_diff.append(max(abs(float(lp[name]) - float(rp[name])) for name in ALLIANCES))
            gauge_sealed_diff.append(abs(float(left["sealed_probe_response"]) - float(right["sealed_probe_response"])))
        for gauge in GAUGES:
            left = values[(gauge, "h0")]
            right = values[(gauge, "h1")]
            hidden_state_diff.append(left["state_dict_digest"] != right["state_dict_digest"])
            lp = public_by_id[left["system_id"]]["profile"]
            rp = public_by_id[right["system_id"]]["profile"]
            hidden_public_diff.append(max(abs(float(lp[name]) - float(rp[name])) for name in ALLIANCES))
            hidden_sealed_diff.append(abs(float(left["sealed_probe_response"]) - float(right["sealed_probe_response"])))
    return {
        "gauge_pair_count": len(gauge_state_diff),
        "hidden_pair_count": len(hidden_state_diff),
        "physical_state_dict_difference_fraction": float(np.mean(gauge_state_diff)),
        "physical_hidden_difference_fraction": float(np.mean(gauge_hidden_diff)),
        "gauge_public_profile_max_abs": float(max(gauge_public_diff)),
        "gauge_sealed_profile_max_abs": float(max(gauge_sealed_diff)),
        "hidden_state_dict_difference_fraction": float(np.mean(hidden_state_diff)),
        "hidden_public_profile_max_abs": float(max(hidden_public_diff)),
        "hidden_sealed_difference_min": float(min(hidden_sealed_diff)),
    }


def metric_gates(metrics: dict[str, Any], physical: dict[str, Any] | None = None) -> dict[str, bool]:
    gates = {
        "finite": metrics["finite_fraction"] >= THRESHOLDS["finite_fraction_min"],
        "baseline": metrics["baseline_max_abs"] <= THRESHOLDS["baseline_max_abs_max"],
        "full_response": metrics["full_response_min"] >= THRESHOLDS["full_response_min"],
        "structure": metrics["structure_accuracy"] >= THRESHOLDS["structure_accuracy_min"],
        "stabilizer": metrics["stabilizer_accuracy"] >= THRESHOLDS["stabilizer_accuracy_min"],
        "orbits": metrics["orbit_accuracy"] >= THRESHOLDS["orbit_accuracy_min"],
        "minimum_alliance": metrics["minimum_alliance_accuracy"] >= THRESHOLDS["minimum_alliance_accuracy_min"],
        "near_distinguishable": metrics["near_distinguishable_accuracy"] >= THRESHOLDS["near_distinguishable_accuracy_min"],
        "gauge_abstention": metrics["gauge_public_abstention"] >= THRESHOLDS["gauge_public_abstention_min"],
        "hidden_abstention": metrics["hidden_public_abstention"] >= THRESHOLDS["hidden_public_abstention_min"],
        "metadata_null": metrics["metadata_null_accuracy"] == THRESHOLDS["metadata_null_accuracy_exact"],
        "leak_sentinel": metrics["leaky_sentinel_accuracy"] == THRESHOLDS["leaky_sentinel_accuracy_exact"],
    }
    if physical is not None:
        gates.update({
            "physical_state_dict_difference": physical["physical_state_dict_difference_fraction"] >= THRESHOLDS["physical_state_dict_difference_fraction_min"],
            "physical_hidden_difference": physical["physical_hidden_difference_fraction"] >= THRESHOLDS["physical_hidden_difference_fraction_min"],
            "gauge_public_equivalence": physical["gauge_public_profile_max_abs"] <= THRESHOLDS["gauge_public_profile_max_abs_max"],
            "gauge_sealed_equivalence": physical["gauge_sealed_profile_max_abs"] <= THRESHOLDS["gauge_sealed_profile_max_abs_max"],
            "hidden_public_equivalence": physical["hidden_public_profile_max_abs"] <= THRESHOLDS["hidden_public_profile_max_abs_max"],
            "hidden_sealed_separation": physical["hidden_sealed_difference_min"] >= THRESHOLDS["hidden_sealed_difference_min"],
        })
    return gates


def fit_discovery() -> None:
    protocol, _manifest = verify_inputs()
    if CAMERA_PATH.exists():
        raise RuntimeError("camera already frozen")
    public_rows = read_jsonl(DISCOVERY_PUBLIC)
    truth_rows = read_jsonl(DISCOVERY_TRUTH)
    scored = score_rows(public_rows, truth_rows)
    physical = physical_pair_metrics(public_rows, truth_rows)
    gates = metric_gates(scored, physical)
    camera: dict[str, Any] = {
        "schema_version": "phase1228.frozen-camera.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "sufficient_threshold": SUFFICIENT_THRESHOLD,
        "sealed_hidden_threshold": SEALED_HIDDEN_THRESHOLD,
        "discovery_metrics": {key: value for key, value in scored.items() if key != "predictions"},
        "discovery_physical_metrics": physical,
        "discovery_gates": gates,
        "qualified": all(gates.values()),
        "confirmation_prediction_rule": (
            "infer stabilizer, role orbits, and minimum alliances from public profile only; abstain on gauge and hidden identity"
        ),
    }
    camera["camera_digest"] = digest(camera)
    write_json(CAMERA_PATH, camera)
    print(canonical_json({"status": "camera_frozen", "qualified": camera["qualified"], "camera_digest": camera["camera_digest"]}))


def predict_confirmation() -> None:
    protocol, _manifest = verify_inputs()
    camera = read_json(CAMERA_PATH)
    if not camera.get("qualified"):
        raise RuntimeError("discovery camera did not qualify")
    if PREDICTION_PATH.exists():
        raise RuntimeError("predictions already exist")
    if CONFIRMATION_SEALED.exists():
        raise RuntimeError("confirmation truth was exposed before prediction")
    public_rows = read_jsonl(CONFIRMATION_PUBLIC)
    predictions: list[dict[str, Any]] = []
    for public in public_rows:
        prediction: dict[str, Any] = {
            "schema_version": "phase1228.confirmation-prediction.v1",
            "phase": PHASE,
            "system_id": public["system_id"],
            "public_digest": public["public_digest"],
            **infer_camera(public["profile"]),
            "metadata_gauge_guess": "u" if int(public["public_nonce"]) == 0 else "v",
            "sealed_hidden_rule": "h1 if sealed_probe_response > 0.125 else h0",
            "sealed_gauge_decision": "ABSTAIN",
        }
        prediction["prediction_digest"] = digest(prediction)
        predictions.append(prediction)
    write_jsonl(PREDICTION_PATH, predictions)
    marker = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "camera_digest": camera["camera_digest"],
        "public_digest": digest(public_rows),
        "prediction_digest": digest(predictions),
        "count": len(predictions),
    }
    marker["marker_digest"] = digest(marker)
    write_json(OUT_ROOT / "confirmation/prediction_marker.json", marker)
    print(canonical_json({"status": "confirmation_predicted", "count": len(predictions), "prediction_digest": marker["prediction_digest"]}))


def reveal() -> None:
    protocol, _manifest = verify_inputs()
    if FINAL_PATH.exists():
        raise RuntimeError("final already exists")
    if CONFIRMATION_SEALED.exists():
        raise RuntimeError("confirmation truth already revealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deterministic confirmation reveal")
    reveal_started_at = utc_now()
    camera = read_json(CAMERA_PATH)
    marker = read_json(OUT_ROOT / "confirmation/prediction_marker.json")
    public_rows = read_jsonl(CONFIRMATION_PUBLIC)
    predictions = read_jsonl(PREDICTION_PATH)
    confirmation_summary = read_json(OUT_ROOT / "runs/confirmation_summary.json")
    device = torch.device("cuda:0")
    regenerated_public: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for row in _manifest:
        if row["split"] != "confirmation":
            continue
        public, truth = evaluate_spec(spec_from_row(row), device)
        regenerated_public.append(public)
        truth_rows.append(truth)
    if digest(regenerated_public) != confirmation_summary["public_digest"]:
        raise RuntimeError("confirmation public response changed before reveal")
    if digest(truth_rows) != confirmation_summary["truth_digest"]:
        raise RuntimeError("confirmation truth commitment mismatch")
    write_jsonl(CONFIRMATION_SEALED, truth_rows)
    torch.cuda.empty_cache()
    truth_by_id = {row["system_id"]: row for row in truth_rows}
    prediction_by_id = {row["system_id"]: row for row in predictions}
    scored = score_rows(public_rows, truth_rows)
    physical = physical_pair_metrics(public_rows, truth_rows)
    sealed_hidden_correct: list[bool] = []
    metadata_correct: list[bool] = []
    prediction_matches: list[bool] = []
    for public in public_rows:
        truth = truth_by_id[public["system_id"]]
        prediction = prediction_by_id[public["system_id"]]
        expected = infer_camera(public["profile"])
        prediction_matches.append(all(prediction[key] == expected[key] for key in expected))
        hidden_guess = "h1" if float(truth["sealed_probe_response"]) > SEALED_HIDDEN_THRESHOLD else "h0"
        sealed_hidden_correct.append(hidden_guess == truth["hidden_variant"])
        metadata_correct.append(prediction["metadata_gauge_guess"] == truth["gauge_variant"])
    confirmation_metrics = {key: value for key, value in scored.items() if key != "predictions"}
    confirmation_metrics.update({
        "prediction_reproduction_accuracy": float(np.mean(prediction_matches)),
        "sealed_hidden_accuracy": float(np.mean(sealed_hidden_correct)),
        "sealed_gauge_abstention": 1.0,
        "metadata_null_accuracy": float(np.mean(metadata_correct)),
    })
    gates = metric_gates(confirmation_metrics, physical)
    gates.update({
        "prediction_reproduction": confirmation_metrics["prediction_reproduction_accuracy"] == 1.0,
        "sealed_hidden": confirmation_metrics["sealed_hidden_accuracy"] >= THRESHOLDS["sealed_hidden_accuracy_min"],
        "sealed_gauge_abstention": confirmation_metrics["sealed_gauge_abstention"] >= THRESHOLDS["sealed_gauge_abstention_min"],
        "prediction_precedes_reveal": marker["created_at_utc"] < reveal_started_at,
    })
    camera_gate = bool(camera["qualified"] and all(gates.values()))
    status = "automorphism_quotient_camera_passed" if camera_gate else "automorphism_quotient_camera_failed"
    run_summaries = {
        split: read_json(OUT_ROOT / "runs" / f"{split}_summary.json") for split in SPLITS
    }
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "camera_digest": camera["camera_digest"],
        "prediction_marker_digest": marker["marker_digest"],
        "run_summary_digests": {split: summary["summary_digest"] for split, summary in run_summaries.items()},
        "result": {
            "camera_gate": camera_gate,
            "discovery_metrics": camera["discovery_metrics"],
            "discovery_physical_metrics": camera["discovery_physical_metrics"],
            "confirmation_metrics": confirmation_metrics,
            "confirmation_physical_metrics": physical,
            "confirmation_gates": gates,
        },
        "k_item": {
            "identifier": "K205",
            "evidence_grade": "E3-KT" if camera_gate else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "Under a fixed CUDA-FP16 known-truth response basis, an automorphism-aware camera recovers exact "
                "role stabilizers, identifiable role orbits, and minimum alliances; distinguishes near symmetries; "
                "abstains on genuinely different exact-equivalent gauges; and splits public-equivalent hidden variants "
                "only after a sealed basis extension."
                if camera_gate else
                "The automorphism-aware camera failed at least one preregistered known-truth calibration gate."
            ),
            "scope": "constructed exact-router/ReLU-table micro-systems only; no pretrained-language claim",
        },
        "mathematics": {
            "new_mathematics_required": False,
            "strict_equivalence": "zero response distance under a fixed registered basis",
            "empirical_rule": "bounded tolerance is a camera decision region, not a globally transitive equivalence relation",
        },
        "claim_boundary": list(protocol["claim_scope"]),
        "authorization": {
            "automatic_execution": False,
            "auto_continue": 0,
            "reason": (
                "A known-truth camera pass authorizes a separately frozen deconfounded material contract. "
                "It does not authorize reusing revealed Phase1227 samples or silently running Qwen3."
            ),
            "next_experiment": "zero-model de-answer-load object-relation-value material construction and audit",
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    run_summary: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "discovery_count": run_summaries["discovery"]["count"],
        "confirmation_count": run_summaries["confirmation"]["count"],
        "camera_gate": camera_gate,
        "final_digest": final["final_digest"],
    }
    run_summary["summary_digest"] = digest(run_summary)
    write_json(RUN_SUMMARY_PATH, run_summary)
    print(canonical_json({"status": status, "camera_gate": camera_gate, "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=(
        "materialize", "run-discovery", "fit", "run-confirmation", "predict", "reveal"
    ))
    args = parser.parse_args()
    if args.stage == "materialize":
        materialize()
    elif args.stage == "run-discovery":
        run_split("discovery")
    elif args.stage == "fit":
        fit_discovery()
    elif args.stage == "run-confirmation":
        camera = read_json(CAMERA_PATH)
        if not camera.get("qualified"):
            raise RuntimeError("discovery camera did not qualify")
        run_split("confirmation")
    elif args.stage == "predict":
        predict_confirmation()
    else:
        reveal()


if __name__ == "__main__":
    main()
