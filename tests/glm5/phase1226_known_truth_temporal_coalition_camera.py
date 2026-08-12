#!/usr/bin/env python3
"""Phase1226: fixed-domain known-truth temporal coalition camera.

This is a one-shot instrument qualification, not a language-mechanism claim.
Three transformer-shaped causal systems have identical clean behavior and
fixed CUDA geometry but different known mechanisms:

* ``boundary_store``: the answer-boundary state is sufficient and persistent;
* ``source_query_joint``: source and query must be patched jointly;
* ``sustained_recompute``: source/query are reset each rollout step, so a
  sustained intervention is required.

The camera receives only opaque intervention-response tables.  Confirmation
predictions are written before confirmation truth and held-out responses are
read.  Two latent variants per mechanism are response-equivalent, so the
camera must identify the functional quotient and abstain on latent identity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


PHASE = 1226
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1226_known_truth_temporal_coalition_camera_audit.py")
OUT_ROOT = ROOT / "tests/glm5/result/phase1226_known_truth_temporal_coalition_camera"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1225_qwen3_fp16_numerical_applicability"

WIDTH = 16
BATCH_SIZE = WIDTH * WIDTH
ROLL_OUT_STEPS = 4
SYSTEMS_PER_LATENT = 60
ROLES = ("source", "query", "boundary")
MECHANISMS = ("boundary_store", "source_query_joint", "sustained_recompute")
LATENT_VARIANTS = ("u", "v")
TEMPORAL_REGIMES = ("single", "sustained")
SLOT_COALITIONS = tuple(
    tuple(int(value) for value in coalition)
    for size in range(1, len(ROLES) + 1)
    for coalition in itertools.combinations(range(len(ROLES)), size)
)

SPLITS = {
    "discovery": {
        "seed": 1_226_101,
        "task_coefficients": ((1, 1), (1, 3), (3, 5), (5, 7)),
    },
    "confirmation": {
        "seed": 1_226_901,
        "task_coefficients": ((1, 5), (3, 1), (5, 3), (7, 5)),
    },
}

CAMERA_THRESHOLDS = {
    "sufficient_min": 1.0 - 1.0e-7,
    "inactive_max": 1.0 / ROLL_OUT_STEPS + 1.0e-7,
}

GATES = {
    "finite_fraction_min": 1.0,
    "clean_accuracy_min": 1.0,
    "donor_accuracy_min": 1.0,
    "null_donor_fraction_max": 1.0e-7,
    "discovery_class_accuracy_min": 1.0,
    "confirmation_class_accuracy_min": 1.0,
    "confirmation_min_class_accuracy_min": 1.0,
    "confirmation_structure_accuracy_min": 1.0,
    "confirmation_abstention_accuracy_min": 1.0,
    "holdout_mae_max": 1.0e-7,
    "holdout_max_abs_error_max": 1.0e-7,
    "metadata_null_accuracy_max": 1.0 / len(MECHANISMS) + 1.0e-12,
    "leaky_sentinel_accuracy_min": 1.0,
}

NUMERICAL_TYPE = {
    "device": "cuda:0",
    "dtype": "float16",
    "backend": "torch_eager",
    "batch_size": BATCH_SIZE,
    "role_slots": len(ROLES),
    "channel_width": WIDTH,
    "rollout_steps": ROLL_OUT_STEPS,
    "cache_policy": "none",
    "shape_changes_forbidden": True,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_digest(value: dict[str, Any], field: str) -> None:
    clean = dict(value)
    stored = clean.pop(field)
    if digest(clean) != stored:
        raise RuntimeError(f"digest mismatch for {field}")


def flatten_numeric(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        yield float(value)
    elif isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_numeric(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_numeric(item)


def source_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    final = read_json(SOURCE_ROOT / "analysis/final.json")
    audit = read_json(SOURCE_ROOT / "audit/independent_result_audit.json")
    validate_digest(final, "final_digest")
    validate_digest(audit, "audit_digest")
    return final, audit


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    split: str
    replicate: int
    mechanism: str
    latent_variant: str
    task_id: int
    coefficients: tuple[int, int]
    source_shift: int
    query_shift: int
    alternate_source_shift: int
    alternate_query_shift: int
    slot_by_role: dict[str, int]
    channel_permutation: tuple[int, ...]
    channel_signs: tuple[int, ...]
    channel_gauge_id: int


def alternate_shifts(coefficients: tuple[int, int]) -> tuple[int, int]:
    a, b = coefficients
    correct_delta = (a * 1 + b * 2) % WIDTH
    for source_shift in range(1, WIDTH):
        for query_shift in range(1, WIDTH):
            delta = (a * source_shift + b * query_shift) % WIDTH
            if delta not in (0, correct_delta):
                return source_shift, query_shift
    raise RuntimeError("no alternate shift found")


def system_spec(split: str, replicate: int, mechanism: str, latent_variant: str) -> SystemSpec:
    config = SPLITS[split]
    task_id = replicate % len(config["task_coefficients"])
    coefficients = tuple(int(value) for value in config["task_coefficients"][task_id])
    permutations = list(itertools.permutations(range(len(ROLES))))
    slot_permutation = permutations[replicate % len(permutations)]
    slot_by_role = {role: int(slot_permutation[index]) for index, role in enumerate(ROLES)}
    channel_gauge_id = replicate % 12
    rng = np.random.default_rng(int(config["seed"]) + 101 * channel_gauge_id)
    channel_permutation = tuple(int(value) for value in rng.permutation(WIDTH))
    channel_signs = tuple(int(value) for value in rng.choice(np.asarray([-1, 1]), size=WIDTH))
    alternate_source_shift, alternate_query_shift = alternate_shifts(coefficients)
    identity = {
        "phase": PHASE,
        "split": split,
        "replicate": replicate,
        "mechanism": mechanism,
        "latent_variant": latent_variant,
        "seed": config["seed"],
    }
    return SystemSpec(
        system_id=digest(identity)[:24],
        split=split,
        replicate=replicate,
        mechanism=mechanism,
        latent_variant=latent_variant,
        task_id=task_id,
        coefficients=coefficients,
        source_shift=1,
        query_shift=2,
        alternate_source_shift=alternate_source_shift,
        alternate_query_shift=alternate_query_shift,
        slot_by_role=slot_by_role,
        channel_permutation=channel_permutation,
        channel_signs=channel_signs,
        channel_gauge_id=channel_gauge_id,
    )


def all_specs(split: str) -> list[SystemSpec]:
    return [
        system_spec(split, replicate, mechanism, latent_variant)
        for replicate in range(SYSTEMS_PER_LATENT)
        for mechanism in MECHANISMS
        for latent_variant in LATENT_VARIANTS
    ]


def logical_table(coefficients: tuple[int, int], device: torch.device) -> torch.Tensor:
    source = torch.arange(WIDTH, device=device, dtype=torch.long)[:, None]
    query = torch.arange(WIDTH, device=device, dtype=torch.long)[None, :]
    a, b = coefficients
    return (a * source + b * query) % WIDTH


class KnownTruthRoleTransformer(nn.Module):
    """Exact-attention, exact-MLP micro-system with sealed causal morphology.

    Three opaque physical slots are routed to logical roles by one-key masked
    attention.  Source/query composition is an exact ReLU pair detector plus a
    fixed task table, matching a Transformer attention-to-MLP computation while
    retaining known intervention truth.
    """

    def __init__(self, spec: SystemSpec, device: torch.device) -> None:
        super().__init__()
        self.spec = spec
        routing = torch.zeros((len(ROLES), len(ROLES)), dtype=torch.float16, device=device)
        for role_index, role in enumerate(ROLES):
            routing[role_index, spec.slot_by_role[role]] = 1.0
        table = logical_table(spec.coefficients, device)
        table_one_hot = torch.nn.functional.one_hot(table, WIDTH).to(torch.float16)
        permutation = torch.tensor(spec.channel_permutation, dtype=torch.long, device=device)
        inverse = torch.argsort(permutation)
        signs = torch.tensor(spec.channel_signs, dtype=torch.float16, device=device)
        self.register_buffer("routing", routing)
        self.register_buffer("table_one_hot", table_one_hot)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", inverse)
        self.register_buffer("signs", signs)

    def encode(self, indices: torch.Tensor) -> torch.Tensor:
        logical = torch.nn.functional.one_hot(indices, WIDTH).to(torch.float16)
        return logical[:, self.permutation] * self.signs[None, :]

    def encode_vector(self, logical: torch.Tensor) -> torch.Tensor:
        return logical[:, self.permutation] * self.signs[None, :]

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        return (encoded * self.signs[None, :])[:, self.inverse_permutation]

    def attend_roles(self, physical_state: torch.Tensor) -> torch.Tensor:
        return torch.einsum("rs,bsw->brw", self.routing, physical_state)

    def compose(self, encoded_source: torch.Tensor, encoded_query: torch.Tensor) -> torch.Tensor:
        source = self.decode(encoded_source)
        query = self.decode(encoded_query)
        pair = torch.relu(source[:, :, None] + query[:, None, :] - 1.0)
        logical_output = torch.einsum("bij,ijc->bc", pair, self.table_one_hot)
        return self.encode_vector(logical_output)

    def logits_from_state(self, physical_state: torch.Tensor) -> torch.Tensor:
        logical_roles = self.attend_roles(physical_state)
        if self.spec.mechanism == "boundary_store":
            encoded_output = logical_roles[:, ROLES.index("boundary"), :]
        else:
            encoded_output = self.compose(
                logical_roles[:, ROLES.index("source"), :],
                logical_roles[:, ROLES.index("query"), :],
            )
        return 8.0 * self.decode(encoded_output)

    def bundle(self, source: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        output = logical_table(self.spec.coefficients, source.device)[source, query]
        logical = {
            "source": self.encode(source),
            "query": self.encode(query),
            "boundary": self.encode(output),
        }
        physical = torch.zeros(
            (len(source), len(ROLES), WIDTH), dtype=torch.float16, device=source.device
        )
        for role, slot in self.spec.slot_by_role.items():
            physical[:, slot, :] = logical[role]
        return physical

    def zero_bundle(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            (batch_size, len(ROLES), WIDTH), dtype=torch.float16, device=self.routing.device
        )

    def rollout(
        self,
        receiver: torch.Tensor,
        patch_bundle: torch.Tensor,
        selected_slots: tuple[int, ...],
        schedule: str,
    ) -> torch.Tensor:
        state = receiver.clone()
        outputs: list[torch.Tensor] = []
        for step in range(ROLL_OUT_STEPS):
            if self.spec.mechanism == "sustained_recompute":
                state = receiver.clone()
            active = (
                (schedule == "single" and step == 0)
                or schedule == "sustained"
                or (schedule == "delayed" and step == 2)
                or (schedule == "alternating" and step % 2 == 0)
            )
            if active and selected_slots:
                state[:, list(selected_slots), :] = patch_bundle[:, list(selected_slots), :]
            outputs.append(self.logits_from_state(state))
        return torch.stack(outputs, dim=0)


def task_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.cartesian_prod(
        torch.arange(WIDTH, device=device, dtype=torch.long),
        torch.arange(WIDTH, device=device, dtype=torch.long),
    )
    if len(pairs) != BATCH_SIZE:
        raise RuntimeError("fixed batch geometry changed")
    return pairs[:, 0], pairs[:, 1]


def target_for(coefficients: tuple[int, int], source: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    a, b = coefficients
    return (a * source + b * query) % WIDTH


def response_metrics(
    logits: torch.Tensor,
    recipient_target: torch.Tensor,
    donor_target: torch.Tensor,
) -> dict[str, Any]:
    predictions = torch.argmax(logits, dim=-1)
    donor_match = (predictions == donor_target[None, :]).to(torch.float32)
    recipient_match = (predictions == recipient_target[None, :]).to(torch.float32)
    donor_scores = logits.gather(
        2, donor_target[None, :, None].expand(ROLL_OUT_STEPS, -1, 1)
    ).squeeze(2)
    alternatives = logits.clone()
    alternatives.scatter_(
        2, donor_target[None, :, None].expand(ROLL_OUT_STEPS, -1, 1), -torch.inf
    )
    donor_margin = donor_scores - torch.max(alternatives, dim=2).values
    return {
        "donor_fraction": float(donor_match.mean().item()),
        "recipient_fraction": float(recipient_match.mean().item()),
        "trajectory_donor_fraction": [float(value) for value in donor_match.mean(dim=1).tolist()],
        "mean_donor_margin": float(donor_margin.float().mean().item()),
        "finite": bool(torch.isfinite(logits).all().item()),
    }


def role_for_slot(spec: SystemSpec, slot: int) -> str:
    return next(role for role, physical in spec.slot_by_role.items() if physical == slot)


def bundles_for(
    model: KnownTruthRoleTransformer,
    source: torch.Tensor,
    query: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    spec = model.spec
    donor_source = (source + spec.source_shift) % WIDTH
    donor_query = (query + spec.query_shift) % WIDTH
    alternate_source = (source + spec.alternate_source_shift) % WIDTH
    alternate_query = (query + spec.alternate_query_shift) % WIDTH
    bundles = {
        "correct": model.bundle(donor_source, donor_query),
        "wrong_object": model.bundle(donor_source, query),
        "wrong_relation": model.bundle(source, donor_query),
        "equal_norm_wrong": model.bundle(alternate_source, alternate_query),
        "identity": model.bundle(source, query),
        "zero": model.zero_bundle(len(source)),
    }
    recipient_target = target_for(spec.coefficients, source, query)
    donor_target = target_for(spec.coefficients, donor_source, donor_query)
    if bool(torch.any(recipient_target == donor_target).item()):
        raise RuntimeError("donor target must differ from recipient target for every row")
    return bundles, recipient_target, donor_target


def evaluate_intervention(
    model: KnownTruthRoleTransformer,
    receiver: torch.Tensor,
    patch_bundle: torch.Tensor,
    slots: tuple[int, ...],
    schedule: str,
    recipient_target: torch.Tensor,
    donor_target: torch.Tensor,
) -> dict[str, Any]:
    logits = model.rollout(receiver, patch_bundle, slots, schedule)
    return response_metrics(logits, recipient_target, donor_target)


def response_record(
    spec: SystemSpec, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = KnownTruthRoleTransformer(spec, device).eval()
    source, query = task_batch(device)
    receiver = model.bundle(source, query)
    bundles, recipient_target, donor_target = bundles_for(model, source, query)
    donor_receiver = bundles["correct"]
    baseline = response_metrics(
        model.rollout(receiver, receiver, (), "single"), recipient_target, donor_target
    )
    donor_clean = response_metrics(
        model.rollout(donor_receiver, donor_receiver, (), "single"), donor_target, donor_target
    )

    correct_responses: list[dict[str, Any]] = []
    for regime in TEMPORAL_REGIMES:
        for coalition in SLOT_COALITIONS:
            correct_responses.append({
                "slots": list(coalition),
                "regime": regime,
                **evaluate_intervention(
                    model, receiver, bundles["correct"], coalition, regime,
                    recipient_target, donor_target,
                ),
            })

    identity_control = evaluate_intervention(
        model, receiver, bundles["identity"], tuple(range(len(ROLES))), "sustained",
        recipient_target, donor_target,
    )
    state_norm = float(receiver.float().square().sum(dim=-1).sqrt().mean().item())
    public = {
        "system_id": spec.system_id,
        "split": spec.split,
        "task_id": spec.task_id,
        "opaque_slot_permutation_id": spec.replicate % 6,
        "channel_gauge_id": spec.channel_gauge_id,
        "latent_public_bit": 0 if spec.latent_variant == "u" else 1,
        "state_norm": state_norm,
        "width": WIDTH,
        "rollout_steps": ROLL_OUT_STEPS,
        "baseline_recipient_accuracy": baseline["recipient_fraction"],
        "baseline_donor_fraction": baseline["donor_fraction"],
        "donor_clean_accuracy": donor_clean["donor_fraction"],
        "identity_control_donor_fraction": identity_control["donor_fraction"],
        "correct_donor_responses": correct_responses,
    }

    minimal_roles = (
        ("boundary",)
        if spec.mechanism == "boundary_store"
        else ("source", "query")
    )
    minimal_slots = tuple(sorted(spec.slot_by_role[role] for role in minimal_roles))
    required_regime = "sustained" if spec.mechanism == "sustained_recompute" else "single"
    heldout_responses: list[dict[str, Any]] = []
    for donor_kind in ("wrong_object", "wrong_relation", "equal_norm_wrong", "identity", "zero"):
        heldout_responses.append({
            "name": donor_kind,
            **evaluate_intervention(
                model, receiver, bundles[donor_kind], minimal_slots, required_regime,
                recipient_target, donor_target,
            ),
        })
    for deleted_slot in minimal_slots:
        reduced = tuple(slot for slot in minimal_slots if slot != deleted_slot)
        heldout_responses.append({
            "name": f"delete_slot_{deleted_slot}",
            **evaluate_intervention(
                model, receiver, bundles["correct"], reduced, required_regime,
                recipient_target, donor_target,
            ),
        })
    for schedule in ("delayed", "alternating"):
        heldout_responses.append({
            "name": schedule,
            **evaluate_intervention(
                model, receiver, bundles["correct"], minimal_slots, schedule,
                recipient_target, donor_target,
            ),
        })

    holdout = {
        "system_id": spec.system_id,
        "split": spec.split,
        "responses": heldout_responses,
    }
    truth = {
        "system_id": spec.system_id,
        "split": spec.split,
        "replicate": spec.replicate,
        "mechanism_class": spec.mechanism,
        "latent_variant": spec.latent_variant,
        "latent_variant_identifiable": False,
        "slot_by_role": spec.slot_by_role,
        "minimal_sufficient_slots": list(minimal_slots),
        "required_temporal_regime": required_regime,
        "task_coefficients": list(spec.coefficients),
        "source_shift": spec.source_shift,
        "query_shift": spec.query_shift,
    }
    return public, holdout, truth


def protocol_payload() -> dict[str, Any]:
    source_final, source_audit = source_documents()
    contracts = source_final["result"]["contracts"]
    checks = {
        "phase1225_final_digest_valid": len(source_final["final_digest"]) == 64,
        "phase1225_audit_passed": source_audit["all_checks_passed"] is True,
        "phase1225_contracts_frozen": contracts == {"C0": True, "C1": False, "C2": True, "C3": True},
        "phase1225_auto_continue_false_preserved": source_final["authorization"]["automatic_execution"] is False,
        "new_user_turn_explicitly_requests_systematic_r_and_d": True,
        "fixed_numerical_type": NUMERICAL_TYPE["shape_changes_forbidden"] is True,
        "three_mechanism_classes": len(MECHANISMS) == 3,
        "seven_nonempty_coalitions": len(SLOT_COALITIONS) == 7,
        "two_temporal_regimes": len(TEMPORAL_REGIMES) == 2,
        "discovery_confirmation_tasks_disjoint": set(SPLITS["discovery"]["task_coefficients"]).isdisjoint(
            SPLITS["confirmation"]["task_coefficients"]
        ),
        "confirmation_truth_forbidden_during_prediction": True,
        "confirmation_holdout_forbidden_during_prediction": True,
        "latent_abstention_required": True,
        "qwen3_execution_forbidden": True,
        "cuda_required": True,
    }
    payload = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "title": "fixed-domain known-truth source-query-boundary temporal coalition camera",
        "authorization_type": "new explicit user turn; not Phase1225 auto-continuation",
        "scripts": {
            "main_sha256": sha256_file(SCRIPT),
            "audit_sha256": sha256_file(AUDIT_SCRIPT),
        },
        "source_phase1225_final_digest": source_final["final_digest"],
        "source_phase1225_audit_digest": source_audit["audit_digest"],
        "numerical_type_eta": NUMERICAL_TYPE,
        "run_instance_lambda": "one formal process instance plus independent audit regeneration",
        "splits": SPLITS,
        "systems_per_latent": SYSTEMS_PER_LATENT,
        "mechanism_classes": list(MECHANISMS),
        "latent_variants": list(LATENT_VARIANTS),
        "roles": list(ROLES),
        "slot_coalitions": [list(value) for value in SLOT_COALITIONS],
        "temporal_regimes": list(TEMPORAL_REGIMES),
        "camera_thresholds": CAMERA_THRESHOLDS,
        "gates": GATES,
        "public_intervention_basis": [
            "correct donor over all seven nonempty physical-slot coalitions",
            "single pulse at step zero",
            "sustained patch at every rollout step",
            "matched identity and clean receiver controls",
        ],
        "sealed_heldout_interventions": [
            "wrong-object donor on inferred minimum coalition",
            "wrong-relation donor on inferred minimum coalition",
            "equal-norm wrong donor",
            "identity donor",
            "zero donor",
            "delete-one controls",
            "delayed pulse",
            "alternating pulse",
        ],
        "primary_endpoints": [
            "blind functional mechanism-class recovery on disjoint confirmation tasks",
            "exact minimum coalition and temporal-regime recovery",
            "abstention on response-equivalent latent implementation variant",
            "pre-reveal prediction of held-out donor-response trajectories",
            "metadata null at one-third chance and leaky sentinel at one",
        ],
        "hard_stops": [
            "This is a known-truth fixed-geometry instrument qualification, not a Qwen3 or language mechanism discovery.",
            "Phase1225 C1 remains failed; cross-geometry coordinate identity is not claimed.",
            "Failure ends this camera path and forbids Qwen3 external-validity execution.",
            "Pass authorizes only a separately preregistered finite Qwen3 transfer on Phase1222 objects and exact geometry.",
            "No extra mechanism class, coalition, position, head, or threshold may be selected after reveal.",
        ],
        "execution_order": [
            "preregister",
            "independent zero-output preaudit",
            "discovery CUDA run",
            "discovery score",
            "confirmation CUDA run",
            "sealed confirmation prediction",
            "confirmation reveal and score",
            "finalize",
            "independent exact-regeneration audit",
        ],
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    payload["protocol_digest"] = digest(payload)
    return payload


def preregister() -> dict[str, Any]:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to overwrite Phase1226 formal artifacts")
    protocol = protocol_payload()
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    return protocol


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    validate_digest(protocol, "protocol_digest")
    if protocol["scripts"]["main_sha256"] != sha256_file(SCRIPT):
        raise RuntimeError("main script changed after preregistration")
    if protocol["scripts"]["audit_sha256"] != sha256_file(AUDIT_SCRIPT):
        raise RuntimeError("audit script changed after preregistration")
    return protocol


def run_split(split: str, device: torch.device) -> dict[str, Any]:
    protocol = verify_protocol()
    preaudit_path = OUT_ROOT / "protocol/independent_preaudit.json"
    if not preaudit_path.exists():
        raise RuntimeError("independent zero-output preaudit required")
    preaudit = read_json(preaudit_path)
    validate_digest(preaudit, "audit_digest")
    if not preaudit["all_checks_passed"]:
        raise RuntimeError("preaudit failed")
    if split == "confirmation":
        discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
        validate_digest(discovery, "score_digest")
        if not discovery["confirmation_authorized"]:
            raise RuntimeError("discovery did not authorize confirmation")
    run_root = OUT_ROOT / f"runs/{split}"
    if run_root.exists():
        raise RuntimeError(f"refusing to overwrite {run_root}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase1226 execution requires CUDA")

    public_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    specs = all_specs(split)
    with torch.inference_mode():
        for index, spec in enumerate(specs):
            public, holdout, truth = response_record(spec, device)
            public_rows.append(public)
            holdout_rows.append(holdout)
            truth_rows.append(truth)
            if (index + 1) % 60 == 0:
                print(canonical({"split": split, "completed": index + 1, "total": len(specs)}), flush=True)

    write_jsonl_gz(run_root / "public_camera_inputs.jsonl.gz", public_rows)
    write_jsonl_gz(run_root / "sealed_holdout_responses.jsonl.gz", holdout_rows)
    write_jsonl_gz(run_root / "sealed_truth.jsonl.gz", truth_rows)
    summary = {
        "phase": PHASE,
        "split": split,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(public_rows),
        "mechanism_counts": dict(sorted(Counter(row["mechanism_class"] for row in truth_rows).items())),
        "latent_counts": dict(sorted(Counter(row["latent_variant"] for row in truth_rows).items())),
        "task_counts": dict(sorted(Counter(str(row["task_id"]) for row in public_rows).items())),
        "finite_fraction": float(np.mean([
            all(math.isfinite(value) for value in flatten_numeric(row)) for row in public_rows
        ])),
        "clean_accuracy_min": min(row["baseline_recipient_accuracy"] for row in public_rows),
        "donor_accuracy_min": min(row["donor_clean_accuracy"] for row in public_rows),
        "null_donor_fraction_max": max(row["baseline_donor_fraction"] for row in public_rows),
        "identity_control_donor_fraction_max": max(row["identity_control_donor_fraction"] for row in public_rows),
        "public_digest": digest(public_rows),
        "holdout_digest": digest(holdout_rows),
        "truth_digest": digest(truth_rows),
    }
    summary["summary_digest"] = digest(summary)
    write_json(run_root / "summary.json", summary)
    return summary


def response_lookup(row: dict[str, Any], regime: str) -> list[dict[str, Any]]:
    return [item for item in row["correct_donor_responses"] if item["regime"] == regime]


def infer_camera(row: dict[str, Any]) -> dict[str, Any]:
    threshold = CAMERA_THRESHOLDS["sufficient_min"]
    single = response_lookup(row, "single")
    sustained = response_lookup(row, "sustained")
    sufficient_single = [item for item in single if item["donor_fraction"] >= threshold]
    sufficient_sustained = [item for item in sustained if item["donor_fraction"] >= threshold]

    if sufficient_single:
        minimum_size = min(len(item["slots"]) for item in sufficient_single)
        minimum = sorted(
            tuple(item["slots"]) for item in sufficient_single if len(item["slots"]) == minimum_size
        )[0]
        if minimum_size == 1:
            mechanism = "boundary_store"
        elif minimum_size == 2:
            mechanism = "source_query_joint"
        else:
            mechanism = "ABSTAIN"
        regime = "single"
    elif sufficient_sustained:
        minimum_size = min(len(item["slots"]) for item in sufficient_sustained)
        minimum = sorted(
            tuple(item["slots"]) for item in sufficient_sustained if len(item["slots"]) == minimum_size
        )[0]
        mechanism = "sustained_recompute" if minimum_size == 2 else "ABSTAIN"
        regime = "sustained"
    else:
        minimum = ()
        mechanism = "ABSTAIN"
        regime = "ABSTAIN"

    steps = int(row["rollout_steps"])
    if mechanism in ("boundary_store", "source_query_joint"):
        delayed = [0.0, 0.0, 1.0, 1.0]
        alternating = [1.0] * steps
    elif mechanism == "sustained_recompute":
        delayed = [0.0, 0.0, 1.0, 0.0]
        alternating = [1.0 if step % 2 == 0 else 0.0 for step in range(steps)]
    else:
        delayed = [0.0] * steps
        alternating = [0.0] * steps

    heldout_predictions: list[dict[str, Any]] = []
    for name in ("wrong_object", "wrong_relation", "equal_norm_wrong", "identity"):
        heldout_predictions.append({
            "name": name,
            "donor_fraction": 0.0,
            "trajectory_donor_fraction": [0.0] * steps,
        })
    zero_value = 1.0 / int(row["width"])
    heldout_predictions.append({
        "name": "zero",
        "donor_fraction": zero_value,
        "trajectory_donor_fraction": [zero_value] * steps,
    })
    for slot in minimum:
        heldout_predictions.append({
            "name": f"delete_slot_{slot}",
            "donor_fraction": 0.0,
            "trajectory_donor_fraction": [0.0] * steps,
        })
    heldout_predictions.extend((
        {
            "name": "delayed",
            "donor_fraction": float(np.mean(delayed)),
            "trajectory_donor_fraction": delayed,
        },
        {
            "name": "alternating",
            "donor_fraction": float(np.mean(alternating)),
            "trajectory_donor_fraction": alternating,
        },
    ))
    return {
        "system_id": row["system_id"],
        "predicted_mechanism_class": mechanism,
        "predicted_minimal_sufficient_slots": list(minimum),
        "predicted_required_temporal_regime": regime,
        "latent_variant_decision": "ABSTAIN",
        "heldout_predictions": heldout_predictions,
    }


def score_structure(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    truth_by_id = {row["system_id"]: row for row in truth}
    classes: list[bool] = []
    structures: list[bool] = []
    abstentions: list[bool] = []
    per_class: dict[str, list[bool]] = defaultdict(list)
    for prediction in predictions:
        expected = truth_by_id[prediction["system_id"]]
        class_ok = prediction["predicted_mechanism_class"] == expected["mechanism_class"]
        structure_ok = (
            prediction["predicted_minimal_sufficient_slots"] == expected["minimal_sufficient_slots"]
            and prediction["predicted_required_temporal_regime"] == expected["required_temporal_regime"]
        )
        abstain_ok = prediction["latent_variant_decision"] == "ABSTAIN"
        classes.append(class_ok)
        structures.append(structure_ok)
        abstentions.append(abstain_ok)
        per_class[expected["mechanism_class"]].append(class_ok)
    return {
        "class_accuracy": float(np.mean(classes)),
        "min_class_accuracy": float(min(np.mean(values) for values in per_class.values())),
        "per_class_accuracy": {key: float(np.mean(values)) for key, values in sorted(per_class.items())},
        "structure_accuracy": float(np.mean(structures)),
        "abstention_accuracy": float(np.mean(abstentions)),
    }


def heldout_error(predictions: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> dict[str, float]:
    holdout_by_id = {row["system_id"]: row for row in holdout}
    errors: list[float] = []
    for prediction in predictions:
        actual = {
            row["name"]: row for row in holdout_by_id[prediction["system_id"]]["responses"]
        }
        for predicted in prediction["heldout_predictions"]:
            expected = actual[predicted["name"]]
            errors.append(abs(predicted["donor_fraction"] - expected["donor_fraction"]))
            errors.extend(
                abs(left - right)
                for left, right in zip(
                    predicted["trajectory_donor_fraction"], expected["trajectory_donor_fraction"]
                )
            )
    return {
        "mae": float(np.mean(errors)),
        "max_abs_error": float(max(errors)),
        "comparison_count": len(errors),
    }


def majority_lookup_accuracy(
    discovery_public: list[dict[str, Any]],
    discovery_truth: list[dict[str, Any]],
    confirmation_public: list[dict[str, Any]],
    confirmation_truth: list[dict[str, Any]],
    signature: Any,
) -> float:
    discovery_truth_by_id = {row["system_id"]: row for row in discovery_truth}
    labels_by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    for row in discovery_public:
        labels_by_signature[canonical(signature(row))][
            discovery_truth_by_id[row["system_id"]]["mechanism_class"]
        ] += 1
    global_default = sorted(
        Counter(row["mechanism_class"] for row in discovery_truth).items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]
    lookup = {
        key: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, counts in labels_by_signature.items()
    }
    confirmation_truth_by_id = {row["system_id"]: row for row in confirmation_truth}
    outcomes = [
        lookup.get(canonical(signature(row)), global_default)
        == confirmation_truth_by_id[row["system_id"]]["mechanism_class"]
        for row in confirmation_public
    ]
    return float(np.mean(outcomes))


def score_discovery() -> dict[str, Any]:
    verify_protocol()
    public = read_jsonl_gz(OUT_ROOT / "runs/discovery/public_camera_inputs.jsonl.gz")
    truth = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_truth.jsonl.gz")
    holdout = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_holdout_responses.jsonl.gz")
    predictions = [infer_camera(row) for row in public]
    structure = score_structure(predictions, truth)
    heldout = heldout_error(predictions, holdout)
    checks = {
        "class_accuracy": structure["class_accuracy"] >= GATES["discovery_class_accuracy_min"],
        "min_class_accuracy": structure["min_class_accuracy"] >= GATES["discovery_class_accuracy_min"],
        "structure_accuracy": structure["structure_accuracy"] >= 1.0,
        "abstention_accuracy": structure["abstention_accuracy"] >= 1.0,
        "heldout_mae": heldout["mae"] <= GATES["holdout_mae_max"],
        "heldout_max_abs_error": heldout["max_abs_error"] <= GATES["holdout_max_abs_error_max"],
    }
    result = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "structure_metrics": structure,
        "heldout_metrics": heldout,
        "checks": checks,
        "confirmation_authorized": all(checks.values()),
    }
    result["score_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/discovery_score.json", result)
    return result


def predict_confirmation() -> dict[str, Any]:
    verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
    validate_digest(discovery, "score_digest")
    if not discovery["confirmation_authorized"]:
        raise RuntimeError("discovery did not authorize prediction")
    prediction_path = OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz"
    manifest_path = OUT_ROOT / "analysis/confirmation_prediction_manifest.json"
    if prediction_path.exists() or manifest_path.exists():
        raise RuntimeError("refusing to overwrite confirmation predictions")
    public = read_jsonl_gz(OUT_ROOT / "runs/confirmation/public_camera_inputs.jsonl.gz")
    predictions = [infer_camera(row) for row in public]
    write_jsonl_gz(prediction_path, predictions)
    manifest = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "prediction_count": len(predictions),
        "public_digest": digest(public),
        "prediction_digest": digest(predictions),
        "truth_read": False,
        "holdout_response_read": False,
        "camera_rule": "frozen source-query-boundary coalition and temporal quotient classifier",
    }
    manifest["manifest_digest"] = digest(manifest)
    write_json(manifest_path, manifest)
    return manifest


def score_confirmation() -> dict[str, Any]:
    verify_protocol()
    manifest = read_json(OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
    validate_digest(manifest, "manifest_digest")
    predictions = read_jsonl_gz(OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz")
    truth = read_jsonl_gz(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl.gz")
    holdout = read_jsonl_gz(OUT_ROOT / "runs/confirmation/sealed_holdout_responses.jsonl.gz")
    structure = score_structure(predictions, truth)
    heldout = heldout_error(predictions, holdout)

    discovery_public = read_jsonl_gz(OUT_ROOT / "runs/discovery/public_camera_inputs.jsonl.gz")
    discovery_truth = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_truth.jsonl.gz")
    confirmation_public = read_jsonl_gz(OUT_ROOT / "runs/confirmation/public_camera_inputs.jsonl.gz")
    metadata_accuracy = majority_lookup_accuracy(
        discovery_public, discovery_truth, confirmation_public, truth,
        lambda row: (
            row["task_id"], row["opaque_slot_permutation_id"], row["channel_gauge_id"],
            row["latent_public_bit"], round(row["state_norm"], 6),
        ),
    )
    leaky_accuracy = majority_lookup_accuracy(
        discovery_public, discovery_truth, confirmation_public, truth,
        lambda row: next(
            item["mechanism_class"] for item in discovery_truth + truth if item["system_id"] == row["system_id"]
        ),
    )
    checks = {
        "class_accuracy": structure["class_accuracy"] >= GATES["confirmation_class_accuracy_min"],
        "min_class_accuracy": structure["min_class_accuracy"] >= GATES["confirmation_min_class_accuracy_min"],
        "structure_accuracy": structure["structure_accuracy"] >= GATES["confirmation_structure_accuracy_min"],
        "abstention_accuracy": structure["abstention_accuracy"] >= GATES["confirmation_abstention_accuracy_min"],
        "heldout_mae": heldout["mae"] <= GATES["holdout_mae_max"],
        "heldout_max_abs_error": heldout["max_abs_error"] <= GATES["holdout_max_abs_error_max"],
        "metadata_null": metadata_accuracy <= GATES["metadata_null_accuracy_max"],
        "leaky_sentinel": leaky_accuracy >= GATES["leaky_sentinel_accuracy_min"],
    }
    result = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "prediction_manifest_digest": manifest["manifest_digest"],
        "structure_metrics": structure,
        "heldout_metrics": heldout,
        "metadata_null_accuracy": metadata_accuracy,
        "leaky_sentinel_accuracy": leaky_accuracy,
        "checks": checks,
        "camera_gate": all(checks.values()),
    }
    result["score_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/confirmation_score.json", result)
    return result


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    discovery = read_json(OUT_ROOT / "analysis/discovery_score.json")
    confirmation = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    validate_digest(discovery, "score_digest")
    validate_digest(confirmation, "score_digest")
    summaries = {
        split: read_json(OUT_ROOT / f"runs/{split}/summary.json") for split in SPLITS
    }
    for summary in summaries.values():
        validate_digest(summary, "summary_digest")
    gate = bool(discovery["confirmation_authorized"] and confirmation["camera_gate"])
    result = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": (
            "known_truth_temporal_coalition_camera_passed"
            if gate else "known_truth_temporal_coalition_camera_failed"
        ),
        "protocol_digest": protocol["protocol_digest"],
        "split_summary_digests": {key: value["summary_digest"] for key, value in summaries.items()},
        "discovery_score_digest": discovery["score_digest"],
        "confirmation_score_digest": confirmation["score_digest"],
        "result": {
            "camera_gate": gate,
            "structure_metrics": confirmation["structure_metrics"],
            "heldout_metrics": confirmation["heldout_metrics"],
            "metadata_null_accuracy": confirmation["metadata_null_accuracy"],
            "leaky_sentinel_accuracy": confirmation["leaky_sentinel_accuracy"],
        },
        "k_item": {
            "identifier": "K203" if gate else "K203-NEG",
            "evidence_grade": "E3-KT" if gate else "E3-KT-NEG",
            "statement": (
                "Under fixed CUDA-FP16 geometry, the frozen response camera exactly separates boundary-store, "
                "source-query-joint, and sustained-recompute functional quotients, recovers their minimum "
                "coalition/time regime, abstains on response-equivalent latent variants, and predicts sealed interventions."
                if gate else
                "The frozen camera failed at least one known-truth functional-quotient or held-out prediction gate."
            ),
            "scope": "constructed transformer-shaped known-truth micro-systems only; no pretrained-language claim",
        },
        "authorization": {
            "automatic_execution": False,
            "next_experiment": (
                "separately preregistered finite Qwen3 Phase1222-role external-validity transfer"
                if gate else None
            ),
            "qwen3_execution_now": False,
            "reason": (
                "known-truth pass authorizes protocol design but the Qwen transfer contract is not yet frozen"
                if gate else "known-truth camera failed; Qwen transfer forbidden"
            ),
        },
        "claim_boundary": [
            "No Qwen3, GLM4, or DS7B model was loaded in this phase.",
            "The micro-system uses exact one-key role attention and an exact ReLU composition MLP; it is transformer-shaped but not a freely trained language model.",
            "Latent variants are deliberately intervention-equivalent; abstention is required and does not identify their physical implementation.",
            "Phase1225 cross-geometry C1 remains failed and is not repaired by this result.",
            "An independent audit verifies artifact integrity and exact regeneration, not semantic independence from every shared implementation assumption.",
        ],
        "new_mathematics_required": False,
    }
    result["final_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/final.json", result)
    return result


def selftest() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    with torch.inference_mode():
        for mechanism in MECHANISMS:
            spec = system_spec("discovery", 0, mechanism, "u")
            public, holdout, truth = response_record(spec, device)
            prediction = infer_camera(public)
            structure = score_structure([prediction], [truth])
            error = heldout_error([prediction], [holdout])
            rows.append({
                "mechanism": mechanism,
                "structure": structure,
                "heldout": error,
            })
    passed = all(
        row["structure"]["class_accuracy"] == 1.0
        and row["structure"]["structure_accuracy"] == 1.0
        and row["heldout"]["max_abs_error"] <= GATES["holdout_max_abs_error_max"]
        for row in rows
    )
    result = {"phase": PHASE, "device": str(device), "passed": passed, "rows": rows}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("selftest failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "selftest", "preregister", "run-discovery", "score-discovery",
            "run-confirmation", "predict-confirmation", "score-confirmation", "finalize",
        ),
    )
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "preregister":
        print(json.dumps(preregister(), ensure_ascii=False, indent=2))
    elif args.stage == "run-discovery":
        print(json.dumps(run_split("discovery", torch.device("cuda")), ensure_ascii=False, indent=2))
    elif args.stage == "score-discovery":
        print(json.dumps(score_discovery(), ensure_ascii=False, indent=2))
    elif args.stage == "run-confirmation":
        print(json.dumps(run_split("confirmation", torch.device("cuda")), ensure_ascii=False, indent=2))
    elif args.stage == "predict-confirmation":
        print(json.dumps(predict_confirmation(), ensure_ascii=False, indent=2))
    elif args.stage == "score-confirmation":
        print(json.dumps(score_confirmation(), ensure_ascii=False, indent=2))
    elif args.stage == "finalize":
        print(json.dumps(finalize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
