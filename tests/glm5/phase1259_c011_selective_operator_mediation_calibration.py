"""Phase1259: known-truth selective operator and path-mediation camera.

This is an instrument-calibration experiment, not a language-model mechanism
claim.  It asks whether a response camera can distinguish three cases without
seeing implementation labels:

1. one global linear operator separates target from contextual changes;
2. separation exists only after conditioning on the public control state; or
3. the registered linear camera must abstain because target and context
   response spans collide.

The camera is tested under independent random gauges, held-out factorial
combinations, wrong donors, matched nulls, on-manifold checks, and an explicit
upstream -> mediator -> output block/rescue chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1259_c011_selective_operator_mediation_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PUBLIC = OUT / "material/public_systems.jsonl"
TRUTH = OUT / "material/private_mechanism_truth.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/system_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
AUDITOR = ROOT / "tests/glm5/phase1259_c011_selective_operator_mediation_calibration_audit.py"
PROBE = ROOT / "tests/glm5_temp/phase1259_c011_selective_operator_probe.json"

PHASE = 1259
CAMPAIGN = "C011"
K = 5
CONTROLS = 4
DIMENSION = 24
TASKS = 4
REPLICATES = 8
REGISTRY_SPLITS = ("discovery", "confirmation")
FAMILIES = (
    "global_orthogonal",
    "global_oblique",
    "global_redundant",
    "conditioned_swap",
    "conditioned_cycle",
    "collision_weighted",
)
EXPECTED_CAMERA = {
    "global_orthogonal": "global",
    "global_oblique": "global",
    "global_redundant": "global",
    "conditioned_swap": "conditioned",
    "conditioned_cycle": "conditioned",
    "collision_weighted": "abstain",
}
THRESHOLDS = {
    "selection_target_relative_error_max": 0.04,
    "selection_null_leakage_max": 0.04,
    "selection_idempotence_error_max": 0.04,
    "confirmation_target_state_relative_error_max": 0.05,
    "confirmation_null_state_fraction_max": 0.05,
    "confirmation_block_remaining_fraction_max": 0.05,
    "confirmation_on_manifold_distance_max": 0.05,
    "confirmation_accuracy_min": 0.999,
    "wrong_false_target_rate_max": 0.001,
    "camera_type_accuracy_min": 1.0,
    "abstention_accuracy_min": 1.0,
    "full_state_null_leakage_min": 0.99,
    "random_target_relative_error_min": 0.25,
    "oblique_orthogonal_null_leakage_min": 0.08,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_system_id(split: str, task: int, replicate: int, family: str) -> str:
    raw = f"phase1259|{split}|{task}|{replicate}|{family}"
    return "S" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:15]


def seed_for(split: str, task: int, replicate: int, family: str) -> int:
    payload = f"1259|{split}|{task}|{replicate}|{family}".encode("utf-8")
    return 1_259_000 + int(hashlib.sha256(payload).hexdigest()[:8], 16) % 50_000_000


def make_system_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for split in REGISTRY_SPLITS:
        for task in range(TASKS):
            for replicate in range(REPLICATES):
                for family in FAMILIES:
                    system_id = opaque_system_id(split, task, replicate, family)
                    seed = seed_for(split, task, replicate, family)
                    public.append({
                        "system_id": system_id,
                        "registry_split": split,
                        "task_id": task,
                        "replicate": replicate,
                        "state_dimension": DIMENSION,
                        "content_classes": K,
                        "context_classes": K,
                        "public_controls": CONTROLS,
                    })
                    truth.append({
                        "system_id": system_id,
                        "family": family,
                        "expected_camera": EXPECTED_CAMERA[family],
                        "seed": seed,
                    })
    return public, truth


def factorial_cases() -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for control in range(CONTROLS):
        for content in range(K):
            for target in range(K):
                if target == content:
                    continue
                wrong = next(value for value in range(K) if value not in (content, target))
                for context in range(K):
                    for context_alt in range(K):
                        if context_alt == context:
                            continue
                        slot = (content + 2 * target + 3 * context + context_alt + control) % 4
                        partition = "discovery" if slot in (0, 1) else ("selection" if slot == 2 else "confirmation")
                        rows.append({
                            "control": control,
                            "content": content,
                            "target": target,
                            "wrong": wrong,
                            "context": context,
                            "context_alt": context_alt,
                            "partition": partition,
                        })
    return rows


def protocol_payload(public: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    cases = factorial_cases()
    partition_counts = {name: sum(row["partition"] == name for row in cases) for name in ("discovery", "selection", "confirmation")}
    return {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema_version": "phase1259.c011.selective_operator_mediation.protocol.v1",
        "created_at_utc": utc_now(),
        "claim_type": "known_truth_instrument_calibration",
        "question": "Can a typed response camera select a global operator, a control-conditioned operator, or abstention, then close target rescue, context preservation, wrong rejection and path mediation on held-out combinations?",
        "systems": len(public),
        "systems_per_family": len(public) // len(FAMILIES),
        "registry_splits": list(REGISTRY_SPLITS),
        "tasks": TASKS,
        "replicates": REPLICATES,
        "families_hidden_until_analysis": True,
        "expected_camera_counts": {camera: sum(row["expected_camera"] == camera for row in truth) for camera in ("global", "conditioned", "abstain")},
        "factorial_case_count_per_system": len(cases),
        "factorial_partition_counts": partition_counts,
        "state_events": ["upstream", "mediator"],
        "camera_candidates": ["global_oblique_projector", "control_conditioned_oblique_projector", "typed_abstention"],
        "selection_order": "choose global if it passes; otherwise choose conditioned if it passes every pooled selection gate; otherwise abstain",
        "thresholds": THRESHOLDS,
        "gates": [
            "camera_type_recovery",
            "target_rescue",
            "context_preservation",
            "wrong_identity_rejection",
            "matched_null_rejection",
            "path_block_and_rescue",
            "on_manifold_confirmation",
            "heldout_factorial_prediction",
            "negative_control_separation",
        ],
        "controls": ["full_state_patch", "orthogonal_projector", "random_projector", "public_truth_leak_audit"],
        "stopping_rule": "If known-truth type recovery or any conjunctive gate fails, stop this camera and deny free-network extrapolation. No family, threshold or candidate may be changed after preregistration.",
        "authorized_next_step": "A pass authorizes one free-trained small-Transformer cross-depth external-validity phase. It does not authorize Qwen3 or a language-mechanism claim.",
        "forbidden_claims": [
            "natural-language semantic subspace",
            "Qwen3 mechanism",
            "unique physical implementation",
            "global low-dimensional Euclidean ontology",
            "new mathematics",
        ],
        "public_digest": digest(public),
        "truth_digest": digest(truth),
        "case_digest": digest(cases),
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "float64 deterministic known-truth tensor algebra",
    }


def preregister(force: bool) -> None:
    if PROTOCOL.exists() and not force:
        raise RuntimeError(f"protocol already exists: {PROTOCOL}")
    public, truth = make_system_rows()
    write_jsonl(PUBLIC, public)
    write_jsonl(TRUTH, truth)
    atomic_json(ENVIRONMENT, environment_snapshot())
    atomic_json(PROTOCOL, protocol_payload(public, truth))
    print(canonical_json({"status": "preregistered", "systems": len(public), "cases_per_system": len(factorial_cases())}))


def verify_protocol() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL)
    public = read_jsonl(PUBLIC)
    truth = read_jsonl(TRUTH)
    if digest(public) != protocol["public_digest"] or digest(truth) != protocol["truth_digest"]:
        raise RuntimeError("frozen material digest mismatch")
    if digest(factorial_cases()) != protocol["case_digest"]:
        raise RuntimeError("factorial case digest mismatch")
    if protocol["thresholds"] != THRESHOLDS:
        raise RuntimeError("threshold drift")
    return protocol, public, truth


def orthogonal_matrix(seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(DIMENSION, DIMENSION))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    return torch.tensor(q, dtype=torch.float64, device=device)


@dataclass
class KnownTruthSystem:
    family: str
    seed: int
    task_id: int
    device: torch.device

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed + 31 * self.task_id)
        self.content_permutation = torch.tensor(rng.permutation(K), dtype=torch.long, device=self.device)
        self.context_permutation = torch.tensor(rng.permutation(K), dtype=torch.long, device=self.device)
        self.q_up = orthogonal_matrix(self.seed + 101, self.device)
        self.q_med = orthogonal_matrix(self.seed + 211, self.device)
        self.advance = self.q_med @ self.q_up.T
        self.eye = torch.eye(K, dtype=torch.float64, device=self.device)

    def latent(self, control: torch.Tensor, content: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        c = self.eye.index_select(0, self.content_permutation.index_select(0, content))
        n = self.eye.index_select(0, self.context_permutation.index_select(0, context))
        z = torch.zeros((content.shape[0], DIMENSION), dtype=torch.float64, device=self.device)
        if self.family == "global_orthogonal":
            z[:, 0:K] = c
            z[:, K:2 * K] = n
        elif self.family == "global_oblique":
            z[:, 0:K] = c + 0.55 * n
            z[:, K:2 * K] = 0.25 * c + n
        elif self.family == "global_redundant":
            scale = 1.0 / math.sqrt(2.0)
            z[:, 0:K] = scale * c
            z[:, K:2 * K] = scale * n
            z[:, 2 * K:3 * K] = scale * c
            z[:, 3 * K:4 * K] = scale * n
        elif self.family == "conditioned_swap":
            even = control.remainder(2) == 0
            z[even, 0:K] = c[even]
            z[even, K:2 * K] = n[even]
            z[~even, 0:K] = n[~even]
            z[~even, K:2 * K] = c[~even]
        elif self.family == "conditioned_cycle":
            for g in range(CONTROLS):
                mask = control == g
                z[mask, g * K:(g + 1) * K] = c[mask]
                null_block = (g + 1) % CONTROLS
                z[mask, null_block * K:(null_block + 1) * K] = n[mask]
        elif self.family == "collision_weighted":
            z[:, 0:K] = c + 2.0 * n
        else:
            raise ValueError(self.family)
        return z

    def state(self, event: str, control: torch.Tensor, content: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latent = self.latent(control, content, context)
        gauge = self.q_up if event == "upstream" else self.q_med
        return latent @ gauge.T

    def prototypes(self, event: str) -> torch.Tensor:
        chunks = []
        for g in range(CONTROLS):
            control = torch.full((K * K,), g, dtype=torch.long, device=self.device)
            content = torch.arange(K, device=self.device).repeat_interleave(K)
            context = torch.arange(K, device=self.device).repeat(K)
            chunks.append(self.state(event, control, content, context))
        return torch.stack(chunks, dim=0)

    def decode(self, state: torch.Tensor, control: torch.Tensor, event: str = "mediator") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prototypes = self.prototypes(event).index_select(0, control)
        distances = torch.sum((state.unsqueeze(1) - prototypes) ** 2, dim=-1)
        best_distance, best = torch.min(distances, dim=1)
        return best.div(K, rounding_mode="floor"), best.remainder(K), torch.sqrt(best_distance.clamp_min(0.0))


def case_tensors(cases: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("control", "content", "target", "wrong", "context", "context_alt")
    result = {key: torch.tensor([int(row[key]) for row in cases], dtype=torch.long, device=device) for key in keys}
    result["partition"] = torch.tensor([
        {"discovery": 0, "selection": 1, "confirmation": 2}[str(row["partition"])] for row in cases
    ], dtype=torch.long, device=device)
    return result


def orthonormal_basis(samples: torch.Tensor) -> torch.Tensor:
    if samples.numel() == 0:
        return torch.zeros((DIMENSION, 0), dtype=samples.dtype, device=samples.device)
    _u, singular, vh = torch.linalg.svd(samples, full_matrices=False)
    if singular.numel() == 0 or float(singular[0].item()) <= 1.0e-12:
        return torch.zeros((DIMENSION, 0), dtype=samples.dtype, device=samples.device)
    rank = int(torch.sum(singular > singular[0] * 1.0e-8).item())
    return vh[:rank].T.contiguous()


def fit_oblique_operator(target_deltas: torch.Tensor, null_deltas: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    target_basis = orthonormal_basis(target_deltas)
    null_basis = orthonormal_basis(null_deltas)
    combined = torch.cat((target_basis, null_basis), dim=1)
    selector = torch.cat((
        torch.eye(target_basis.shape[1], dtype=target_deltas.dtype, device=target_deltas.device),
        torch.zeros((target_basis.shape[1], null_basis.shape[1]), dtype=target_deltas.dtype, device=target_deltas.device),
    ), dim=1)
    operator = target_basis @ selector @ torch.linalg.pinv(combined)
    combined_rank = int(torch.linalg.matrix_rank(combined, rtol=1.0e-8, atol=1.0e-10).item())
    return operator, {
        "target_rank": int(target_basis.shape[1]),
        "null_rank": int(null_basis.shape[1]),
        "combined_rank": combined_rank,
        "rank_deficit": int(target_basis.shape[1] + null_basis.shape[1] - combined_rank),
    }


def apply_operator(delta: torch.Tensor, operators: dict[int, torch.Tensor] | torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    if isinstance(operators, dict):
        stack = torch.stack([operators[int(value)] for value in control.tolist()], dim=0)
        return torch.bmm(stack, delta.unsqueeze(-1)).squeeze(-1)
    return delta @ operators.T


def operator_metrics(operator: dict[int, torch.Tensor] | torch.Tensor, target: torch.Tensor, null: torch.Tensor, control: torch.Tensor) -> dict[str, float]:
    target_prediction = apply_operator(target, operator, control)
    null_prediction = apply_operator(null, operator, control)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1.0e-12)
    null_norm = torch.linalg.vector_norm(null).clamp_min(1.0e-12)
    if isinstance(operator, dict):
        idempotence = max(
            float((torch.linalg.vector_norm(value @ value - value) / torch.linalg.vector_norm(value).clamp_min(1.0e-12)).item())
            for value in operator.values()
        )
    else:
        idempotence = float((torch.linalg.vector_norm(operator @ operator - operator) / torch.linalg.vector_norm(operator).clamp_min(1.0e-12)).item())
    return {
        "target_relative_error": float((torch.linalg.vector_norm(target_prediction - target) / target_norm).item()),
        "null_leakage": float((torch.linalg.vector_norm(null_prediction) / null_norm).item()),
        "idempotence_error": idempotence,
    }


def operator_passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["target_relative_error"] <= THRESHOLDS["selection_target_relative_error_max"]
        and metrics["null_leakage"] <= THRESHOLDS["selection_null_leakage_max"]
        and metrics["idempotence_error"] <= THRESHOLDS["selection_idempotence_error_max"]
    )


def fit_camera(states: dict[str, torch.Tensor], tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    discovery = tensors["partition"] == 0
    selection = tensors["partition"] == 1
    target_delta = states["target"] - states["base"]
    null_delta = states["null"] - states["base"]
    global_operator, global_ranks = fit_oblique_operator(target_delta[discovery], null_delta[discovery])
    conditioned: dict[int, torch.Tensor] = {}
    conditioned_ranks: dict[str, Any] = {}
    for control in range(CONTROLS):
        mask = discovery & (tensors["control"] == control)
        conditioned[control], ranks = fit_oblique_operator(target_delta[mask], null_delta[mask])
        conditioned_ranks[str(control)] = ranks
    global_metrics = operator_metrics(global_operator, target_delta[selection], null_delta[selection], tensors["control"][selection])
    conditioned_metrics = operator_metrics(conditioned, target_delta[selection], null_delta[selection], tensors["control"][selection])
    if operator_passes(global_metrics):
        selected_type = "global"
        selected: torch.Tensor | dict[int, torch.Tensor] | None = global_operator
    elif operator_passes(conditioned_metrics):
        selected_type = "conditioned"
        selected = conditioned
    else:
        selected_type = "abstain"
        selected = None
    target_basis = orthonormal_basis(target_delta[discovery])
    orthogonal_projector = target_basis @ target_basis.T
    generator = torch.Generator(device=target_delta.device)
    generator.manual_seed(913_579)
    random_matrix = torch.randn((DIMENSION, max(1, target_basis.shape[1])), generator=generator, dtype=torch.float64, device=target_delta.device)
    random_basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
    random_projector = random_basis @ random_basis.T
    controls = {
        "full_state": operator_metrics(torch.eye(DIMENSION, dtype=torch.float64, device=target_delta.device), target_delta[selection], null_delta[selection], tensors["control"][selection]),
        "orthogonal": operator_metrics(orthogonal_projector, target_delta[selection], null_delta[selection], tensors["control"][selection]),
        "random": operator_metrics(random_projector, target_delta[selection], null_delta[selection], tensors["control"][selection]),
    }
    return {
        "selected_type": selected_type,
        "selected": selected,
        "global_metrics": global_metrics,
        "conditioned_metrics": conditioned_metrics,
        "global_ranks": global_ranks,
        "conditioned_ranks": conditioned_ranks,
        "controls": controls,
    }


def relative_error(predicted: torch.Tensor, target: torch.Tensor, reference: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(predicted - target) / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)).item())


def evaluate_confirmation(
    system: KnownTruthSystem,
    states_up: dict[str, torch.Tensor],
    states_med: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
    camera_up: dict[str, Any],
    camera_med: dict[str, Any],
) -> dict[str, Any] | None:
    if camera_up["selected_type"] == "abstain" or camera_med["selected_type"] == "abstain":
        return None
    mask = tensors["partition"] == 2
    control = tensors["control"][mask]
    content = tensors["content"][mask]
    target = tensors["target"][mask]
    wrong = tensors["wrong"][mask]
    context = tensors["context"][mask]
    base_up = states_up["base"][mask]
    target_up = states_up["target"][mask]
    wrong_up = states_up["wrong"][mask]
    null_up = states_up["null"][mask]
    base_med = states_med["base"][mask]
    target_med = states_med["target"][mask]
    wrong_med = states_med["wrong"][mask]
    target_effect = target_med - base_med
    op_up = camera_up["selected"]
    op_med = camera_med["selected"]
    assert op_up is not None and op_med is not None

    patched_target_up = base_up + apply_operator(target_up - base_up, op_up, control)
    patched_wrong_up = base_up + apply_operator(wrong_up - base_up, op_up, control)
    patched_null_up = base_up + apply_operator(null_up - base_up, op_up, control)
    propagated_target = patched_target_up @ system.advance.T
    propagated_wrong = patched_wrong_up @ system.advance.T
    propagated_null = patched_null_up @ system.advance.T

    blocked = propagated_target + apply_operator(base_med - propagated_target, op_med, control)
    rescued = blocked + apply_operator(target_med - blocked, op_med, control)
    wrong_rescue = blocked + apply_operator(wrong_med - blocked, op_med, control)

    pred_target, ctx_target, dist_target = system.decode(propagated_target, control)
    pred_wrong, ctx_wrong, dist_wrong = system.decode(propagated_wrong, control)
    pred_null, ctx_null, dist_null = system.decode(propagated_null, control)
    pred_block, ctx_block, dist_block = system.decode(blocked, control)
    pred_rescue, ctx_rescue, dist_rescue = system.decode(rescued, control)
    pred_wrong_rescue, ctx_wrong_rescue, dist_wrong_rescue = system.decode(wrong_rescue, control)

    target_norm = torch.linalg.vector_norm(target_effect).clamp_min(1.0e-12)
    manifold_scale = torch.linalg.vector_norm(target_effect, dim=1).mean().clamp_min(1.0e-12)
    all_distances = torch.cat((dist_target, dist_wrong, dist_null, dist_block, dist_rescue, dist_wrong_rescue))
    return {
        "cases": int(mask.sum().item()),
        "target_state_relative_error": relative_error(propagated_target, target_med, target_effect),
        "wrong_state_relative_error": relative_error(propagated_wrong, wrong_med, wrong_med - base_med),
        "null_state_fraction": float((torch.linalg.vector_norm(propagated_null - base_med) / target_norm).item()),
        "target_accuracy": float((pred_target == target).double().mean().item()),
        "target_context_preservation": float((ctx_target == context).double().mean().item()),
        "wrong_identity_accuracy": float((pred_wrong == wrong).double().mean().item()),
        "wrong_false_target_rate": float((pred_wrong == target).double().mean().item()),
        "wrong_context_preservation": float((ctx_wrong == context).double().mean().item()),
        "null_content_preservation": float((pred_null == content).double().mean().item()),
        "null_context_preservation": float((ctx_null == context).double().mean().item()),
        "block_remaining_fraction": float((torch.linalg.vector_norm(blocked - base_med) / target_norm).item()),
        "block_base_accuracy": float((pred_block == content).double().mean().item()),
        "block_context_preservation": float((ctx_block == context).double().mean().item()),
        "rescue_target_accuracy": float((pred_rescue == target).double().mean().item()),
        "rescue_context_preservation": float((ctx_rescue == context).double().mean().item()),
        "wrong_rescue_identity_accuracy": float((pred_wrong_rescue == wrong).double().mean().item()),
        "wrong_rescue_false_target_rate": float((pred_wrong_rescue == target).double().mean().item()),
        "wrong_rescue_context_preservation": float((ctx_wrong_rescue == context).double().mean().item()),
        "on_manifold_distance_max": float((all_distances.max() / manifold_scale).item()),
    }


def confirmation_passes(metrics: dict[str, Any]) -> bool:
    accuracy_keys = (
        "target_accuracy", "target_context_preservation", "wrong_identity_accuracy", "wrong_context_preservation",
        "null_content_preservation", "null_context_preservation", "block_base_accuracy", "block_context_preservation",
        "rescue_target_accuracy", "rescue_context_preservation", "wrong_rescue_identity_accuracy", "wrong_rescue_context_preservation",
    )
    return (
        metrics["target_state_relative_error"] <= THRESHOLDS["confirmation_target_state_relative_error_max"]
        and metrics["null_state_fraction"] <= THRESHOLDS["confirmation_null_state_fraction_max"]
        and metrics["block_remaining_fraction"] <= THRESHOLDS["confirmation_block_remaining_fraction_max"]
        and metrics["on_manifold_distance_max"] <= THRESHOLDS["confirmation_on_manifold_distance_max"]
        and metrics["wrong_false_target_rate"] <= THRESHOLDS["wrong_false_target_rate_max"]
        and metrics["wrong_rescue_false_target_rate"] <= THRESHOLDS["wrong_false_target_rate_max"]
        and all(metrics[key] >= THRESHOLDS["confirmation_accuracy_min"] for key in accuracy_keys)
    )


def build_states(system: KnownTruthSystem, tensors: dict[str, torch.Tensor], event: str) -> dict[str, torch.Tensor]:
    control = tensors["control"]
    return {
        "base": system.state(event, control, tensors["content"], tensors["context"]),
        "target": system.state(event, control, tensors["target"], tensors["context"]),
        "wrong": system.state(event, control, tensors["wrong"], tensors["context"]),
        "null": system.state(event, control, tensors["content"], tensors["context_alt"]),
    }


def run_system(public: dict[str, Any], truth: dict[str, Any], tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    system = KnownTruthSystem(str(truth["family"]), int(truth["seed"]), int(public["task_id"]), device)
    states_up = build_states(system, tensors, "upstream")
    states_med = build_states(system, tensors, "mediator")
    camera_up = fit_camera(states_up, tensors)
    camera_med = fit_camera(states_med, tensors)
    confirmation = evaluate_confirmation(system, states_up, states_med, tensors, camera_up, camera_med)
    expected = str(truth["expected_camera"])
    selected = camera_up["selected_type"] if camera_up["selected_type"] == camera_med["selected_type"] else "event_mismatch"
    controls = camera_med["controls"]
    return {
        "system_id": public["system_id"],
        "registry_split": public["registry_split"],
        "task_id": public["task_id"],
        "replicate": public["replicate"],
        "selected_camera": selected,
        "expected_camera": expected,
        "camera_type_correct": selected == expected,
        "upstream_selection": {
            "selected_type": camera_up["selected_type"],
            "global_metrics": camera_up["global_metrics"],
            "conditioned_metrics": camera_up["conditioned_metrics"],
            "global_ranks": camera_up["global_ranks"],
            "conditioned_ranks": camera_up["conditioned_ranks"],
        },
        "mediator_selection": {
            "selected_type": camera_med["selected_type"],
            "global_metrics": camera_med["global_metrics"],
            "conditioned_metrics": camera_med["conditioned_metrics"],
            "global_ranks": camera_med["global_ranks"],
            "conditioned_ranks": camera_med["conditioned_ranks"],
        },
        "controls": controls,
        "confirmation": confirmation,
        "confirmation_passed": confirmation_passes(confirmation) if confirmation is not None else expected == "abstain",
    }


def run(device_name: str) -> None:
    protocol, public_rows, truth_rows = verify_protocol()
    if not PREAUDIT.exists() or not read_json(PREAUDIT).get("all_checks_passed"):
        raise RuntimeError("independent preaudit must pass before formal run")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    truth_by_id = {row["system_id"]: row for row in truth_rows}
    cases = factorial_cases()
    tensors = case_tensors(cases, device)
    rows = []
    for index, public in enumerate(public_rows, start=1):
        rows.append(run_system(public, truth_by_id[public["system_id"]], tensors, device))
        if index % 32 == 0:
            print(canonical_json({"completed": index, "total": len(public_rows)}), flush=True)
    write_jsonl(RAW, rows)
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "systems": len(rows),
        "cases_per_system": protocol["factorial_case_count_per_system"],
        "raw_digest": digest(rows),
        "protocol_hash": file_sha256(PROTOCOL),
    }
    atomic_json(SUMMARY, summary)
    atomic_json(COMPLETE, {"status": "formal_run_complete", "created_at_utc": utc_now(), "summary_hash": file_sha256(SUMMARY), "raw_hash": file_sha256(RAW)})
    print(canonical_json({"status": "formal_run_complete", "systems": len(rows)}))


def minimum(rows: list[dict[str, Any]], key: str) -> float:
    return min(float(row["confirmation"][key]) for row in rows if row["confirmation"] is not None)


def maximum(rows: list[dict[str, Any]], key: str) -> float:
    return max(float(row["confirmation"][key]) for row in rows if row["confirmation"] is not None)


def analyze() -> None:
    protocol, public, truth = verify_protocol()
    if not COMPLETE.exists():
        raise RuntimeError("formal run is incomplete")
    rows = read_jsonl(RAW)
    truth_by_id = {row["system_id"]: row for row in truth}
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    for row in rows:
        by_family[truth_by_id[row["system_id"]]["family"]].append(row)
    per_family: dict[str, Any] = {}
    for family, members in by_family.items():
        confirmations = [row for row in members if row["confirmation"] is not None]
        per_family[family] = {
            "systems": len(members),
            "expected_camera": EXPECTED_CAMERA[family],
            "selected_counts": {camera: sum(row["selected_camera"] == camera for row in members) for camera in ("global", "conditioned", "abstain", "event_mismatch")},
            "camera_type_accuracy": sum(row["camera_type_correct"] for row in members) / len(members),
            "confirmation_pass_fraction": sum(row["confirmation_passed"] for row in members) / len(members),
            "confirmation_worst": None if not confirmations else {
                "target_state_relative_error_max": maximum(confirmations, "target_state_relative_error"),
                "null_state_fraction_max": maximum(confirmations, "null_state_fraction"),
                "block_remaining_fraction_max": maximum(confirmations, "block_remaining_fraction"),
                "on_manifold_distance_max": maximum(confirmations, "on_manifold_distance_max"),
                "target_accuracy_min": minimum(confirmations, "target_accuracy"),
                "context_preservation_min": min(
                    minimum(confirmations, "target_context_preservation"),
                    minimum(confirmations, "null_context_preservation"),
                    minimum(confirmations, "rescue_context_preservation"),
                ),
                "wrong_identity_accuracy_min": minimum(confirmations, "wrong_identity_accuracy"),
                "wrong_false_target_rate_max": maximum(confirmations, "wrong_false_target_rate"),
                "rescue_target_accuracy_min": minimum(confirmations, "rescue_target_accuracy"),
            },
        }
    camera_type_accuracy = sum(row["camera_type_correct"] for row in rows) / len(rows)
    abstention_rows = [row for row in rows if row["expected_camera"] == "abstain"]
    abstention_accuracy = sum(row["selected_camera"] == "abstain" for row in abstention_rows) / len(abstention_rows)
    actionable = [row for row in rows if row["expected_camera"] != "abstain"]
    control_full_null_min = min(row["controls"]["full_state"]["null_leakage"] for row in rows)
    control_random_target_min = min(row["controls"]["random"]["target_relative_error"] for row in rows)
    oblique_members = by_family["global_oblique"]
    oblique_orth_null_min = min(row["controls"]["orthogonal"]["null_leakage"] for row in oblique_members)
    gates = {
        "G-CAMERA-TYPE": camera_type_accuracy >= THRESHOLDS["camera_type_accuracy_min"],
        "G-ABSTENTION": abstention_accuracy >= THRESHOLDS["abstention_accuracy_min"],
        "G-TARGET-RESCUE": all(row["confirmation"] is not None and row["confirmation"]["target_state_relative_error"] <= THRESHOLDS["confirmation_target_state_relative_error_max"] and row["confirmation"]["target_accuracy"] >= THRESHOLDS["confirmation_accuracy_min"] for row in actionable),
        "G-CONTEXT-PRESERVATION": all(row["confirmation"] is not None and min(row["confirmation"]["target_context_preservation"], row["confirmation"]["null_context_preservation"], row["confirmation"]["rescue_context_preservation"]) >= THRESHOLDS["confirmation_accuracy_min"] for row in actionable),
        "G-WRONG-REJECTION": all(row["confirmation"] is not None and row["confirmation"]["wrong_identity_accuracy"] >= THRESHOLDS["confirmation_accuracy_min"] and row["confirmation"]["wrong_false_target_rate"] <= THRESHOLDS["wrong_false_target_rate_max"] for row in actionable),
        "G-MATCHED-NULL": all(row["confirmation"] is not None and row["confirmation"]["null_state_fraction"] <= THRESHOLDS["confirmation_null_state_fraction_max"] for row in actionable),
        "G-PATH-MEDIATION": all(row["confirmation"] is not None and row["confirmation"]["block_remaining_fraction"] <= THRESHOLDS["confirmation_block_remaining_fraction_max"] and row["confirmation"]["rescue_target_accuracy"] >= THRESHOLDS["confirmation_accuracy_min"] for row in actionable),
        "G-ON-MANIFOLD": all(row["confirmation"] is not None and row["confirmation"]["on_manifold_distance_max"] <= THRESHOLDS["confirmation_on_manifold_distance_max"] for row in actionable),
        "G-CONTROLS": control_full_null_min >= THRESHOLDS["full_state_null_leakage_min"] and control_random_target_min >= THRESHOLDS["random_target_relative_error_min"] and oblique_orth_null_min >= THRESHOLDS["oblique_orthogonal_null_leakage_min"],
        "G-BREADTH": all(len(members) == len(rows) // len(FAMILIES) for members in by_family.values()) and all(row["confirmation_passed"] for row in rows),
    }
    passed = all(gates.values())
    adjudication = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": utc_now(),
        "verdict": "selective_operator_mediation_camera_calibrated" if passed else "selective_operator_mediation_camera_not_calibrated",
        "passed": passed,
        "systems": len(rows),
        "camera_type_accuracy": camera_type_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "per_family": per_family,
        "negative_controls": {
            "full_state_null_leakage_min": control_full_null_min,
            "random_target_relative_error_min": control_random_target_min,
            "oblique_orthogonal_null_leakage_min": oblique_orth_null_min,
        },
        "gates": gates,
        "claim_boundary": "Known-truth response-selective operator and mediation-camera calibration only. No natural-network or language-mechanism claim.",
        "authorization": {
            "free_transformer_cross_depth": passed,
            "qwen3": False,
            "language_mechanism": False,
            "new_mathematics": False,
        },
    }
    atomic_json(ANALYSIS, adjudication)
    final = {
        **adjudication,
        "artifact_hashes": {
            "protocol": file_sha256(PROTOCOL),
            "environment": file_sha256(ENVIRONMENT),
            "public": file_sha256(PUBLIC),
            "truth": file_sha256(TRUTH),
            "preaudit": file_sha256(PREAUDIT),
            "raw": file_sha256(RAW),
            "summary": file_sha256(SUMMARY),
            "complete": file_sha256(COMPLETE),
            "analysis": file_sha256(ANALYSIS),
        },
    }
    final["final_digest"] = digest(final)
    atomic_json(FINAL, final)
    print(canonical_json({"verdict": final["verdict"], "gates": gates, "camera_type_accuracy": camera_type_accuracy}))


def run_auditor(mode: str) -> None:
    result = subprocess.run([sys.executable, str(AUDITOR), "--mode", mode], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"independent auditor failed in {mode} mode")


def probe(device_name: str) -> None:
    device = torch.device(device_name)
    public, truth = make_system_rows()
    tensors = case_tensors(factorial_cases(), device)
    truth_by_id = {row["system_id"]: row for row in truth}
    selected = []
    seen: set[str] = set()
    for row in public:
        family = truth_by_id[row["system_id"]]["family"]
        if family in seen:
            continue
        seen.add(family)
        selected.append(run_system(row, truth_by_id[row["system_id"]], tensors, device))
    atomic_json(PROBE, {"created_at_utc": utc_now(), "device": str(device), "rows": selected})
    print(canonical_json({row["expected_camera"] + ":" + row["system_id"]: row["selected_camera"] for row in selected}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preregister", "preaudit", "run", "analyze", "audit", "all", "probe"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "preregister":
        preregister(args.force)
    elif args.stage == "preaudit":
        run_auditor("pre")
    elif args.stage == "run":
        run(args.device)
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "audit":
        run_auditor("final")
    elif args.stage == "probe":
        probe(args.device)
    else:
        preregister(args.force)
        run_auditor("pre")
        run(args.device)
        analyze()
        run_auditor("final")


if __name__ == "__main__":
    main()
