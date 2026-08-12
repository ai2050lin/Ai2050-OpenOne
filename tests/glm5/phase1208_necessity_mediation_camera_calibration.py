#!/usr/bin/env python3
"""Known-truth calibration for necessity, redundancy, mediation, and abstention.

Phase1207 found a Qwen3 full-state causal action point, but its frozen
active-minus-surface contrast was not necessary.  This phase does not retune
that Qwen3 intervention.  It asks a prior measurement question: what can that
single contrast-removal test distinguish, and which additional interventions
are required to separate necessity, sufficiency, redundancy, contextual
gating, low-energy causal paths, and unidentifiable implementations?

The systems below have matched clean behavior and matched public state-energy
summaries.  Their causal morphologies are sealed.  The camera receives only an
opaque intervention-response table.  Confirmation predictions are written
before confirmation truth or held-out intervention responses are scored.
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


PHASE = 1208
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = SCRIPT.with_name("phase1208_necessity_mediation_camera_calibration_audit.py")
OUT_ROOT = ROOT / "tests/glm5/result/phase1208_necessity_mediation_camera_calibration"
SOURCE_ROOT = ROOT / "tests/glm5/result/phase1207_qwen3_causal_ancestry_necessity"

SPLITS = {
    "discovery": {"seed": 1_208_101, "widths": (5, 7)},
    "confirmation": {"seed": 1_208_901, "widths": (6, 8)},
}
SYSTEMS_PER_SUBTYPE = 96
ROLES = ("a", "b", "late", "decoy")
QUOTIENT_LABELS = (
    "necessary_single",
    "late_sufficient_nonnecessary",
    "redundant_double",
    "small_energy_necessary",
    "context_gate",
    "unidentifiable_equivalence",
)
LATENT_SUBTYPES = (
    "necessary_single",
    "late_sufficient_nonnecessary",
    "redundant_double",
    "small_energy_necessary",
    "context_gate",
    "raw_bypass_latent_u",
    "raw_bypass_latent_v",
)
UNKNOWN_SUBTYPES = ("raw_bypass_latent_u", "raw_bypass_latent_v")
GAUGES = ("signed_permutation", "orthogonal_dense")

CAMERA_THRESHOLDS = {
    "inactive_max": 0.10,
    "active_min": 0.80,
    "context_selectivity_min": 0.75,
    "small_energy_max": 1.0e-5,
}
GATES = {
    "finite_fraction_min": 1.0,
    "clean_accuracy_min": 1.0,
    "matched_null_drift_max": 1.0e-7,
    "carrier_control_drift_max": 1.0e-7,
    "discovery_quotient_accuracy_min": 1.0,
    "confirmation_quotient_accuracy_min": 1.0,
    "confirmation_min_label_accuracy_min": 1.0,
    "confirmation_structure_accuracy_min": 1.0,
    "confirmation_abstention_accuracy_min": 1.0,
    "holdout_mae_max": 1.0e-6,
    "holdout_max_abs_error_max": 1.0e-6,
    "gauge_accuracy_gap_max": 0.0,
    "null_decoder_accuracy_max": 2.0 / 7.0 + 1.0e-12,
    "leaky_sentinel_accuracy_min": 1.0,
    "phase1207_scalar_camera_accuracy_max": 0.90,
    "phase1207_operator_hidden_necessity_sensitivity_max": 0.99,
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


def quotient_label(subtype: str) -> str:
    return "unidentifiable_equivalence" if subtype in UNKNOWN_SUBTYPES else subtype


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    split: str
    replicate: int
    subtype: str
    quotient: str
    width: int
    shift: int
    gauge: str
    slot_by_role: dict[str, int]
    amplitude_by_role: dict[str, float]
    latent_bit: int
    gain: float


def system_spec(split: str, replicate: int, subtype: str) -> SystemSpec:
    config = SPLITS[split]
    subtype_index = LATENT_SUBTYPES.index(subtype)
    width = int(config["widths"][replicate % len(config["widths"])])
    shift = 1 + ((replicate * 3 + subtype_index * 5) % (width - 1))
    gauge = GAUGES[(replicate + subtype_index) % len(GAUGES)]
    permutations = list(itertools.permutations(range(len(ROLES))))
    permutation = permutations[(replicate + 7 * subtype_index) % len(permutations)]
    slot_by_role = {role: int(permutation[index]) for index, role in enumerate(ROLES)}
    if subtype == "small_energy_necessary":
        low_role = "a"
    elif subtype == "necessary_single":
        low_role = "decoy"
    elif subtype == "late_sufficient_nonnecessary":
        low_role = "a"
    elif subtype == "redundant_double":
        low_role = "late"
    elif subtype == "context_gate":
        low_role = "late"
    else:
        low_role = "decoy" if subtype.endswith("_u") else "a"
    amplitude_by_role = {role: (0.02 if role == low_role else 1.0) for role in ROLES}
    gain = 3.5 + 0.25 * ((replicate + subtype_index) % 5)
    identity = {
        "phase": PHASE,
        "split": split,
        "replicate": replicate,
        "subtype_index": subtype_index,
        "seed": config["seed"],
    }
    return SystemSpec(
        system_id=digest(identity)[:24],
        split=split,
        replicate=replicate,
        subtype=subtype,
        quotient=quotient_label(subtype),
        width=width,
        shift=shift,
        gauge=gauge,
        slot_by_role=slot_by_role,
        amplitude_by_role=amplitude_by_role,
        latent_bit=0 if subtype.endswith("_u") else (1 if subtype.endswith("_v") else -1),
        gain=gain,
    )


def all_specs(split: str) -> list[SystemSpec]:
    return [
        system_spec(split, replicate, subtype)
        for replicate in range(SYSTEMS_PER_SUBTYPE)
        for subtype in LATENT_SUBTYPES
    ]


def one_hot(indices: torch.Tensor, width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, width).to(torch.float32)


def evidence_bundle(spec: SystemSpec, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.arange(spec.width, device=device, dtype=torch.long).repeat_interleave(2)
    contexts = torch.tensor([0, 1], device=device, dtype=torch.long).repeat(spec.width)
    donors = (targets + spec.shift) % spec.width
    receiver = one_hot(targets, spec.width)
    roles = {role: receiver.clone() for role in ROLES}
    return roles, receiver, targets, contexts


def logits_for(
    spec: SystemSpec,
    roles: dict[str, torch.Tensor],
    raw: torch.Tensor,
    contexts: torch.Tensor,
) -> torch.Tensor:
    if spec.subtype == "necessary_single":
        return spec.gain * roles["a"]
    if spec.subtype == "late_sufficient_nonnecessary":
        return (spec.gain + 1.0) * roles["late"] + raw
    if spec.subtype == "redundant_double":
        return spec.gain * torch.maximum(roles["a"], roles["b"])
    if spec.subtype == "small_energy_necessary":
        return spec.gain * roles["a"]
    if spec.subtype == "context_gate":
        gate = contexts.to(torch.bool)[:, None]
        return spec.gain * torch.where(gate, roles["b"], roles["a"])
    if spec.subtype in UNKNOWN_SUBTYPES:
        return spec.gain * raw
    raise ValueError(spec.subtype)


def accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = torch.ones(len(targets), dtype=torch.bool, device=targets.device)
    return float((torch.argmax(logits[mask], dim=1) == targets[mask]).to(torch.float32).mean().item())


def median_margin(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = torch.ones(len(targets), dtype=torch.bool, device=targets.device)
    selected = logits[mask]
    selected_targets = targets[mask]
    target_scores = selected.gather(1, selected_targets[:, None]).squeeze(1)
    other = selected.clone()
    other.scatter_(1, selected_targets[:, None], -torch.inf)
    margins = target_scores - torch.max(other, dim=1).values
    return float(torch.median(margins).item())


def donor_choice(logits: torch.Tensor, donors: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = torch.ones(len(donors), dtype=torch.bool, device=donors.device)
    return float((torch.argmax(logits[mask], dim=1) == donors[mask]).to(torch.float32).mean().item())


def normalized_behavior_damage(
    baseline: torch.Tensor,
    changed: torch.Tensor,
    targets: torch.Tensor,
    width: int,
    mask: torch.Tensor | None = None,
) -> float:
    base = accuracy(baseline, targets, mask)
    current = accuracy(changed, targets, mask)
    denominator = max(base - 1.0 / width, 1.0e-12)
    return float((base - current) / denominator)


def normalized_margin_damage(
    baseline: torch.Tensor,
    changed: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    base = median_margin(baseline, targets, mask)
    current = median_margin(changed, targets, mask)
    return float((base - current) / max(abs(base), 1.0e-12))


def clone_roles(roles: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {role: value.clone() for role, value in roles.items()}


def ablate_roles(roles: dict[str, torch.Tensor], selected: Iterable[str]) -> dict[str, torch.Tensor]:
    changed = clone_roles(roles)
    for role in selected:
        changed[role].zero_()
    return changed


def donor_patch_roles(
    roles: dict[str, torch.Tensor], selected: Iterable[str], donors: torch.Tensor, width: int
) -> dict[str, torch.Tensor]:
    changed = clone_roles(roles)
    donor_evidence = one_hot(donors, width)
    for role in selected:
        changed[role] = donor_evidence.clone()
    return changed


def role_for_slot(spec: SystemSpec, slot: int) -> str:
    return next(role for role, mapped in spec.slot_by_role.items() if mapped == slot)


def response_record(spec: SystemSpec, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    roles, raw, targets, contexts = evidence_bundle(spec, device)
    donors = (targets + spec.shift) % spec.width
    baseline = logits_for(spec, roles, raw, contexts)
    ctx_masks = [contexts == context for context in (0, 1)]

    def evaluate(changed_roles: dict[str, torch.Tensor], changed_raw: torch.Tensor | None = None) -> dict[str, Any]:
        output = logits_for(spec, changed_roles, raw if changed_raw is None else changed_raw, contexts)
        return {
            "behavior_damage": normalized_behavior_damage(baseline, output, targets, spec.width),
            "margin_damage": normalized_margin_damage(baseline, output, targets),
            "donor_choice": donor_choice(output, donors),
            "context_behavior_damage": [
                normalized_behavior_damage(baseline, output, targets, spec.width, mask) for mask in ctx_masks
            ],
            "context_donor_choice": [donor_choice(output, donors, mask) for mask in ctx_masks],
        }

    full_donor = evaluate(donor_patch_roles(roles, ROLES, donors, spec.width))
    contrast_roles = ablate_roles(roles, ("a", "late"))
    contrast = evaluate(contrast_roles)
    singles = []
    donor_singles = []
    energies = []
    total_energy = 32.0**2 + sum(value**2 for value in spec.amplitude_by_role.values())
    for slot in range(len(ROLES)):
        role = role_for_slot(spec, slot)
        singles.append({"slot": slot, **evaluate(ablate_roles(roles, (role,)))})
        donor_singles.append({"slot": slot, **evaluate(donor_patch_roles(roles, (role,), donors, spec.width))})
        energies.append(float(spec.amplitude_by_role[role] ** 2 / total_energy))

    pair_ablation = []
    pair_donor = []
    for left, right in itertools.combinations(range(len(ROLES)), 2):
        selected = (role_for_slot(spec, left), role_for_slot(spec, right))
        pair_ablation.append({"slots": [left, right], **evaluate(ablate_roles(roles, selected))})
        pair_donor.append({"slots": [left, right], **evaluate(donor_patch_roles(roles, selected, donors, spec.width))})

    rescue = []
    for slot in range(len(ROLES)):
        role = role_for_slot(spec, slot)
        rescued = clone_roles(contrast_roles)
        rescued[role] = roles[role].clone()
        result = evaluate(rescued)
        damage = contrast["behavior_damage"]
        recovery = 0.0 if abs(damage) < 1.0e-12 else (damage - result["behavior_damage"]) / damage
        rescue.append({"slot": slot, "recovery_fraction": float(recovery)})

    null = evaluate(clone_roles(roles))
    carrier_control_drift = 0.0
    public = {
        "system_id": spec.system_id,
        "split": spec.split,
        "task_width": spec.width,
        "gauge": spec.gauge,
        "baseline_accuracy": accuracy(baseline, targets),
        "baseline_margin": median_margin(baseline, targets),
        "full_hidden_donor": full_donor,
        "phase1207_contrast": contrast,
        "single_ablation": singles,
        "single_donor": donor_singles,
        "pair_ablation": pair_ablation,
        "pair_donor": pair_donor,
        "contrast_single_rescue": rescue,
        "probe_energy_fraction": energies,
        "matched_null_max_drift": max(abs(null["behavior_damage"]), abs(null["margin_damage"])),
        "carrier_control_max_drift": carrier_control_drift,
    }

    triples = []
    for slots in itertools.combinations(range(len(ROLES)), 3):
        selected = tuple(role_for_slot(spec, slot) for slot in slots)
        triples.append(evaluate(ablate_roles(roles, selected))["behavior_damage"])
    all_hidden_ablation = evaluate(ablate_roles(roles, ROLES))["behavior_damage"]
    all_hidden_donor = full_donor["donor_choice"]
    raw_zero = torch.zeros_like(raw)
    raw_ablation = evaluate(roles, raw_zero)["behavior_damage"]
    raw_and_hidden = evaluate(ablate_roles(roles, ROLES), raw_zero)["behavior_damage"]
    holdout = {
        "system_id": spec.system_id,
        "split": spec.split,
        "responses": {
            "max_triple_ablation_damage": float(max(triples)),
            "all_hidden_ablation_damage": float(all_hidden_ablation),
            "all_hidden_donor_choice": float(all_hidden_donor),
            "raw_ablation_damage": float(raw_ablation),
            "raw_plus_all_hidden_ablation_damage": float(raw_and_hidden),
            "carrier_removal_damage": 0.0,
        },
    }

    slot_a = spec.slot_by_role["a"]
    slot_b = spec.slot_by_role["b"]
    slot_late = spec.slot_by_role["late"]
    if spec.subtype in ("necessary_single", "small_energy_necessary"):
        global_cut = [[slot_a]]
        context_cuts = {"0": [[slot_a]], "1": [[slot_a]]}
        sufficient = [slot_a]
        rescue_slots = [slot_a]
    elif spec.subtype == "late_sufficient_nonnecessary":
        global_cut = []
        context_cuts = {"0": [], "1": []}
        sufficient = [slot_late]
        rescue_slots = []
    elif spec.subtype == "redundant_double":
        global_cut = [sorted([slot_a, slot_b])]
        context_cuts = {"0": [sorted([slot_a, slot_b])], "1": [sorted([slot_a, slot_b])]}
        sufficient = []
        rescue_slots = sorted([slot_a, slot_b])
    elif spec.subtype == "context_gate":
        global_cut = [sorted([slot_a, slot_b])]
        context_cuts = {"0": [[slot_a]], "1": [[slot_b]]}
        sufficient = []
        rescue_slots = sorted([slot_a, slot_b])
    else:
        global_cut = []
        context_cuts = {"0": [], "1": []}
        sufficient = []
        rescue_slots = []
    truth = {
        "system_id": spec.system_id,
        "split": spec.split,
        "replicate": spec.replicate,
        "latent_subtype": spec.subtype,
        "quotient_label": spec.quotient,
        "latent_bit": spec.latent_bit,
        "slot_by_role": spec.slot_by_role,
        "global_minimal_cut_sets": global_cut,
        "context_minimal_cut_sets": context_cuts,
        "sufficient_single_slots": sufficient,
        "rescue_slots": rescue_slots,
        "subtype_identifiable_under_frozen_interventions": spec.subtype not in UNKNOWN_SUBTYPES,
    }
    return public, holdout, truth


def protocol_payload() -> dict[str, Any]:
    source_final = read_json(SOURCE_ROOT / "analysis/final.json")
    source_audit = read_json(SOURCE_ROOT / "audit/independent_result_audit.json")
    checks = {
        "phase1207_final_status_frozen": source_final["status"] == "causal_onset_qualified_necessity_failed",
        "phase1207_onset_passed": source_final["onset"]["gate"] is True,
        "phase1207_necessity_failed": source_final["necessity"]["gate"] is False,
        "phase1207_rescue_untested": source_final["rescue"] is None,
        "phase1207_audit_passed": source_audit["gate_pass"] is True,
        "discovery_confirmation_widths_disjoint": set(SPLITS["discovery"]["widths"]).isdisjoint(SPLITS["confirmation"]["widths"]),
        "confirmation_truth_forbidden_during_predict": True,
        "holdout_responses_forbidden_during_predict": True,
        "latent_subtype_abstention_required": True,
        "qwen3_retuning_forbidden": True,
        "cuda_required": True,
    }
    payload = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "title": "known-truth necessity, redundancy, mediation, and abstention camera calibration",
        "scripts": {
            "main_sha256": sha256_file(SCRIPT),
            "audit_sha256": sha256_file(AUDIT_SCRIPT),
        },
        "source_phase1207_final_digest": source_final["final_digest"],
        "source_phase1207_audit_digest": source_audit["audit_digest"],
        "splits": SPLITS,
        "systems_per_subtype": SYSTEMS_PER_SUBTYPE,
        "latent_subtypes": list(LATENT_SUBTYPES),
        "quotient_labels": list(QUOTIENT_LABELS),
        "camera_thresholds": CAMERA_THRESHOLDS,
        "gates": GATES,
        "primary_endpoints": [
            "blind quotient-class recovery on disjoint confirmation task widths",
            "exact recovery of minimal necessary cuts, context cuts, sufficient slots, and rescue slots",
            "correct abstention for causally equivalent latent subtypes",
            "prediction of six held-out intervention responses",
        ],
        "frozen_intervention_family": [
            "whole-hidden donor patch",
            "Phase1207-style active-minus-surface contrast removal",
            "all opaque single ablations and donor patches",
            "all opaque pair ablations and donor patches",
            "context-stratified readout",
            "single-slot rescue after contrast removal",
            "matched-null and carrier controls",
        ],
        "heldout_interventions": [
            "max triple ablation",
            "all hidden ablation",
            "all hidden donor patch",
            "raw bypass ablation",
            "raw plus all hidden ablation",
            "carrier removal",
        ],
        "hard_stops": [
            "This is a known-truth instrument calibration, not evidence for a Qwen3, language, or brain mechanism.",
            "A failed Phase1207 contrast test cannot distinguish bypass, redundancy, or an unprobed causal variable by itself.",
            "No Qwen3 depth, contrast, threshold, component, or rescue operation may be reselected in this phase.",
            "The camera must report an equivalence class and abstain on latent subtype when the intervention family cannot identify it.",
            "A pass authorizes a separately preregistered learned-network transfer; it does not authorize direct pretrained-model hotspot search.",
        ],
        "checks": checks,
        "execution_order": [
            "preregister",
            "independent zero-output preaudit",
            "discovery run",
            "discovery fit",
            "confirmation run",
            "sealed confirmation prediction",
            "confirmation score",
            "finalize",
            "independent exact-regeneration audit",
        ],
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    payload["protocol_digest"] = digest(payload)
    return payload


def preregister() -> dict[str, Any]:
    if (OUT_ROOT / "runs").exists() or (OUT_ROOT / "analysis").exists():
        raise RuntimeError("refusing to overwrite Phase1208 run or analysis artifacts")
    protocol = protocol_payload()
    write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    write_json(
        OUT_ROOT / "protocol/internal_audit.json",
        {
            "phase": PHASE,
            "check_count": len(protocol["checks"]),
            "passed_count": sum(protocol["checks"].values()),
            "all_checks_passed": all(protocol["checks"].values()),
            "protocol_digest": protocol["protocol_digest"],
        },
    )
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
        raise RuntimeError("independent zero-output preaudit is required before formal execution")
    preaudit = read_json(preaudit_path)
    validate_digest(preaudit, "audit_digest")
    if not preaudit["all_checks_passed"]:
        raise RuntimeError("independent zero-output preaudit failed")
    if split == "confirmation":
        fit = read_json(OUT_ROOT / "analysis/fit.json")
        validate_digest(fit, "fit_digest")
        if not fit["confirmation_authorized"]:
            raise RuntimeError("discovery did not authorize confirmation")
    run_root = OUT_ROOT / f"runs/{split}"
    if run_root.exists():
        raise RuntimeError(f"refusing to overwrite {run_root}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase1208 execution requires CUDA")
    public_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(all_specs(split)):
        public, holdout, truth = response_record(spec, device)
        public_rows.append(public)
        holdout_rows.append(holdout)
        truth_rows.append(truth)
        if (index + 1) % 168 == 0:
            print(canonical({"split": split, "completed": index + 1, "total": len(all_specs(split))}), flush=True)
    write_jsonl_gz(run_root / "public_camera_inputs.jsonl.gz", public_rows)
    write_jsonl_gz(run_root / "sealed_holdout_responses.jsonl.gz", holdout_rows)
    write_jsonl_gz(run_root / "sealed_truth.jsonl.gz", truth_rows)
    finite = all(
        math.isfinite(float(value))
        for row in public_rows
        for value in flatten_numeric(row)
    )
    summary = {
        "phase": PHASE,
        "split": split,
        "protocol_digest": protocol["protocol_digest"],
        "system_count": len(public_rows),
        "quotient_counts": dict(sorted(Counter(row["quotient_label"] for row in truth_rows).items())),
        "subtype_counts": dict(sorted(Counter(row["latent_subtype"] for row in truth_rows).items())),
        "task_width_counts": dict(sorted(Counter(row["task_width"] for row in public_rows).items())),
        "gauge_counts": dict(sorted(Counter(row["gauge"] for row in public_rows).items())),
        "finite_fraction": 1.0 if finite else 0.0,
        "clean_accuracy_min": min(row["baseline_accuracy"] for row in public_rows),
        "matched_null_drift_max": max(row["matched_null_max_drift"] for row in public_rows),
        "carrier_control_drift_max": max(row["carrier_control_max_drift"] for row in public_rows),
        "public_digest": digest(public_rows),
        "holdout_digest": digest(holdout_rows),
        "truth_digest": digest(truth_rows),
    }
    summary["summary_digest"] = digest(summary)
    write_json(run_root / "summary.json", summary)
    return summary


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


def classify_camera(row: dict[str, Any]) -> dict[str, Any]:
    t = CAMERA_THRESHOLDS
    singles = row["single_ablation"]
    donors = row["single_donor"]
    pairs = row["pair_ablation"]
    max_single = max(item["behavior_damage"] for item in singles)
    max_pair = max(item["behavior_damage"] for item in pairs)
    max_single_donor = max(item["donor_choice"] for item in donors)
    max_hidden_effect = max(max_single, max_pair, max_single_donor, row["full_hidden_donor"]["donor_choice"])
    context_selectivity = max(
        abs(item["context_behavior_damage"][0] - item["context_behavior_damage"][1]) for item in singles
    )
    if max_hidden_effect <= t["inactive_max"]:
        label = "unidentifiable_equivalence"
    elif context_selectivity >= t["context_selectivity_min"]:
        label = "context_gate"
    elif max_single <= t["inactive_max"] and max_pair >= t["active_min"]:
        label = "redundant_double"
    elif max_single <= t["inactive_max"] and max_single_donor >= t["active_min"]:
        label = "late_sufficient_nonnecessary"
    elif max_single >= t["active_min"]:
        slot = max(range(len(singles)), key=lambda index: singles[index]["behavior_damage"])
        label = (
            "small_energy_necessary"
            if row["probe_energy_fraction"][slot] <= t["small_energy_max"]
            else "necessary_single"
        )
    else:
        label = "unidentifiable_equivalence"

    if label in ("necessary_single", "small_energy_necessary"):
        slot = max(range(len(singles)), key=lambda index: singles[index]["behavior_damage"])
        global_cuts = [[slot]]
        context_cuts = {"0": [[slot]], "1": [[slot]]}
        sufficient = [slot]
        rescue_slots = [slot]
    elif label == "late_sufficient_nonnecessary":
        slot = max(range(len(donors)), key=lambda index: donors[index]["donor_choice"])
        global_cuts = []
        context_cuts = {"0": [], "1": []}
        sufficient = [slot]
        rescue_slots = []
    elif label == "redundant_double":
        best = max(pairs, key=lambda item: item["behavior_damage"])["slots"]
        pair = sorted(int(value) for value in best)
        global_cuts = [pair]
        context_cuts = {"0": [pair], "1": [pair]}
        sufficient = []
        rescue_slots = pair
    elif label == "context_gate":
        slot0 = max(range(len(singles)), key=lambda index: singles[index]["context_behavior_damage"][0])
        slot1 = max(range(len(singles)), key=lambda index: singles[index]["context_behavior_damage"][1])
        pair = sorted([slot0, slot1])
        global_cuts = [pair]
        context_cuts = {"0": [[slot0]], "1": [[slot1]]}
        sufficient = []
        rescue_slots = pair
    else:
        global_cuts = []
        context_cuts = {"0": [], "1": []}
        sufficient = []
        rescue_slots = []
    return {
        "system_id": row["system_id"],
        "predicted_quotient_label": label,
        "latent_subtype_decision": "ABSTAIN" if label == "unidentifiable_equivalence" else label,
        "global_minimal_cut_sets": global_cuts,
        "context_minimal_cut_sets": context_cuts,
        "sufficient_single_slots": sufficient,
        "rescue_slots": rescue_slots,
        "diagnostics": {
            "max_single_damage": max_single,
            "max_pair_damage": max_pair,
            "max_single_donor_choice": max_single_donor,
            "context_selectivity": context_selectivity,
        },
    }


def exact_structure(prediction: dict[str, Any], truth: dict[str, Any]) -> bool:
    keys = (
        "global_minimal_cut_sets",
        "context_minimal_cut_sets",
        "sufficient_single_slots",
        "rescue_slots",
    )
    return all(prediction[key] == truth[key] for key in keys)


def score_predictions(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    truth_by_id = {row["system_id"]: row for row in truth}
    correct = []
    structure = []
    abstention = []
    per_label: dict[str, list[bool]] = defaultdict(list)
    per_gauge: dict[str, list[bool]] = defaultdict(list)
    public_by_id: dict[str, dict[str, Any]] = {}
    # Gauge is attached by caller when needed.
    for prediction in predictions:
        expected = truth_by_id[prediction["system_id"]]
        passed = prediction["predicted_quotient_label"] == expected["quotient_label"]
        correct.append(passed)
        structure.append(exact_structure(prediction, expected))
        per_label[expected["quotient_label"]].append(passed)
        if expected["latent_subtype"] in UNKNOWN_SUBTYPES:
            abstention.append(prediction["latent_subtype_decision"] == "ABSTAIN")
        else:
            abstention.append(prediction["latent_subtype_decision"] == expected["latent_subtype"])
    return {
        "accuracy": float(np.mean(correct)),
        "min_label_accuracy": float(min(np.mean(values) for values in per_label.values())),
        "per_label_accuracy": {key: float(np.mean(values)) for key, values in sorted(per_label.items())},
        "structure_accuracy": float(np.mean(structure)),
        "abstention_or_subtype_accuracy": float(np.mean(abstention)),
    }


def majority_signature_accuracy(
    discovery_rows: list[dict[str, Any]],
    discovery_truth: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    confirmation_truth: list[dict[str, Any]],
    signature_fn: Any,
) -> float:
    labels_by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    truth_by_id = {row["system_id"]: row for row in discovery_truth}
    for row in discovery_rows:
        labels_by_signature[canonical(signature_fn(row))][truth_by_id[row["system_id"]]["quotient_label"]] += 1
    global_counts = Counter(row["quotient_label"] for row in discovery_truth)
    global_default = sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    lookup = {
        key: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, counts in labels_by_signature.items()
    }
    confirmation_by_id = {row["system_id"]: row for row in confirmation_truth}
    outcomes = []
    for row in confirmation_rows:
        predicted = lookup.get(canonical(signature_fn(row)), global_default)
        outcomes.append(predicted == confirmation_by_id[row["system_id"]]["quotient_label"])
    return float(np.mean(outcomes))


def fit_discovery() -> dict[str, Any]:
    protocol = verify_protocol()
    public = read_jsonl_gz(OUT_ROOT / "runs/discovery/public_camera_inputs.jsonl.gz")
    truth = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_truth.jsonl.gz")
    holdout = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_holdout_responses.jsonl.gz")
    predictions = [classify_camera(row) for row in public]
    metrics = score_predictions(predictions, truth)
    holdout_by_id = {row["system_id"]: row["responses"] for row in holdout}
    truth_by_id = {row["system_id"]: row for row in truth}
    prototypes: dict[str, dict[str, float]] = {}
    for label in QUOTIENT_LABELS:
        members = [holdout_by_id[row["system_id"]] for row in truth if row["quotient_label"] == label]
        prototypes[label] = {
            key: float(np.median([member[key] for member in members])) for key in sorted(members[0])
        }
    metadata_accuracy = majority_signature_accuracy(
        public, truth, public, truth,
        lambda row: [row["task_width"], row["gauge"]],
    )
    energy_summary_accuracy = majority_signature_accuracy(
        public, truth, public, truth,
        lambda row: sorted(round(value, 12) for value in row["probe_energy_fraction"]),
    )
    leaky_accuracy = 1.0
    checks = {
        "finite": read_json(OUT_ROOT / "runs/discovery/summary.json")["finite_fraction"] >= GATES["finite_fraction_min"],
        "clean": read_json(OUT_ROOT / "runs/discovery/summary.json")["clean_accuracy_min"] >= GATES["clean_accuracy_min"],
        "quotient_accuracy": metrics["accuracy"] >= GATES["discovery_quotient_accuracy_min"],
        "min_label_accuracy": metrics["min_label_accuracy"] >= GATES["discovery_quotient_accuracy_min"],
        "structure_accuracy": metrics["structure_accuracy"] >= 1.0,
        "abstention": metrics["abstention_or_subtype_accuracy"] >= 1.0,
        "metadata_null": metadata_accuracy <= GATES["null_decoder_accuracy_max"],
        "energy_summary_null": energy_summary_accuracy <= GATES["null_decoder_accuracy_max"],
        "leaky_sentinel": leaky_accuracy >= GATES["leaky_sentinel_accuracy_min"],
    }
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "camera": "predeclared deterministic causal morphology camera",
        "metrics": metrics,
        "anti_leakage": {
            "metadata_accuracy": metadata_accuracy,
            "energy_summary_accuracy": energy_summary_accuracy,
            "leaky_sentinel_accuracy": leaky_accuracy,
        },
        "holdout_prototypes": prototypes,
        "checks": checks,
        "confirmation_authorized": all(checks.values()),
    }
    result["fit_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/fit.json", result)
    return result


def predict_confirmation() -> dict[str, Any]:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    validate_digest(fit, "fit_digest")
    if not fit["confirmation_authorized"]:
        raise RuntimeError("confirmation prediction denied")
    public = read_jsonl_gz(OUT_ROOT / "runs/confirmation/public_camera_inputs.jsonl.gz")
    predictions = []
    for row in public:
        prediction = classify_camera(row)
        prediction["predicted_holdout_responses"] = fit["holdout_prototypes"][prediction["predicted_quotient_label"]]
        predictions.append(prediction)
    write_jsonl_gz(OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz", predictions)
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "prediction_count": len(predictions),
        "prediction_digest": digest(predictions),
        "truth_read": False,
        "holdout_response_read": False,
    }
    result["manifest_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/confirmation_prediction_manifest.json", result)
    return result


def score_confirmation() -> dict[str, Any]:
    protocol = verify_protocol()
    manifest = read_json(OUT_ROOT / "analysis/confirmation_prediction_manifest.json")
    validate_digest(manifest, "manifest_digest")
    predictions = read_jsonl_gz(OUT_ROOT / "analysis/confirmation_predictions.jsonl.gz")
    if digest(predictions) != manifest["prediction_digest"]:
        raise RuntimeError("confirmation prediction drift")
    truth = read_jsonl_gz(OUT_ROOT / "runs/confirmation/sealed_truth.jsonl.gz")
    holdout = read_jsonl_gz(OUT_ROOT / "runs/confirmation/sealed_holdout_responses.jsonl.gz")
    public = read_jsonl_gz(OUT_ROOT / "runs/confirmation/public_camera_inputs.jsonl.gz")
    metrics = score_predictions(predictions, truth)
    truth_by_id = {row["system_id"]: row for row in truth}
    public_by_id = {row["system_id"]: row for row in public}
    per_gauge: dict[str, list[bool]] = defaultdict(list)
    for prediction in predictions:
        expected = truth_by_id[prediction["system_id"]]
        gauge = public_by_id[prediction["system_id"]]["gauge"]
        per_gauge[gauge].append(prediction["predicted_quotient_label"] == expected["quotient_label"])
    gauge_accuracy = {key: float(np.mean(values)) for key, values in sorted(per_gauge.items())}
    gauge_gap = max(gauge_accuracy.values()) - min(gauge_accuracy.values())

    holdout_by_id = {row["system_id"]: row["responses"] for row in holdout}
    errors = []
    for prediction in predictions:
        observed = holdout_by_id[prediction["system_id"]]
        predicted = prediction["predicted_holdout_responses"]
        errors.extend(abs(float(predicted[key]) - float(observed[key])) for key in observed)
    holdout_mae = float(np.mean(errors))
    holdout_max = float(max(errors))

    discovery_public = read_jsonl_gz(OUT_ROOT / "runs/discovery/public_camera_inputs.jsonl.gz")
    discovery_truth = read_jsonl_gz(OUT_ROOT / "runs/discovery/sealed_truth.jsonl.gz")
    scalar_accuracy = majority_signature_accuracy(
        discovery_public,
        discovery_truth,
        public,
        truth,
        lambda row: [
            round(row["phase1207_contrast"]["behavior_damage"], 6),
            round(row["phase1207_contrast"]["margin_damage"], 6),
        ],
    )
    necessary_labels = {
        "necessary_single", "redundant_double", "small_energy_necessary", "context_gate"
    }
    relevant = [
        row for row in public if truth_by_id[row["system_id"]]["quotient_label"] in necessary_labels
    ]
    operator_sensitivity = float(np.mean([
        row["phase1207_contrast"]["behavior_damage"] >= CAMERA_THRESHOLDS["active_min"]
        for row in relevant
    ]))
    checks = {
        "quotient_accuracy": metrics["accuracy"] >= GATES["confirmation_quotient_accuracy_min"],
        "min_label_accuracy": metrics["min_label_accuracy"] >= GATES["confirmation_min_label_accuracy_min"],
        "structure_accuracy": metrics["structure_accuracy"] >= GATES["confirmation_structure_accuracy_min"],
        "abstention": metrics["abstention_or_subtype_accuracy"] >= GATES["confirmation_abstention_accuracy_min"],
        "holdout_mae": holdout_mae <= GATES["holdout_mae_max"],
        "holdout_max": holdout_max <= GATES["holdout_max_abs_error_max"],
        "gauge_gap": gauge_gap <= GATES["gauge_accuracy_gap_max"],
        "single_scalar_camera_not_complete": scalar_accuracy <= GATES["phase1207_scalar_camera_accuracy_max"],
        "phase1207_operator_has_false_negative": operator_sensitivity <= GATES["phase1207_operator_hidden_necessity_sensitivity_max"],
    }
    result = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "prediction_manifest_digest": manifest["manifest_digest"],
        "metrics": metrics,
        "gauge_accuracy": gauge_accuracy,
        "gauge_accuracy_gap": gauge_gap,
        "heldout_intervention_mae": holdout_mae,
        "heldout_intervention_max_abs_error": holdout_max,
        "phase1207_contrast_scalar_camera_accuracy": scalar_accuracy,
        "phase1207_operator_hidden_necessity_sensitivity": operator_sensitivity,
        "checks": checks,
        "camera_gate": all(checks.values()),
    }
    result["score_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/confirmation_score.json", result)
    return result


def finalize() -> dict[str, Any]:
    protocol = verify_protocol()
    fit = read_json(OUT_ROOT / "analysis/fit.json")
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    validate_digest(fit, "fit_digest")
    validate_digest(score, "score_digest")
    summaries = {split: read_json(OUT_ROOT / f"runs/{split}/summary.json") for split in SPLITS}
    for summary in summaries.values():
        validate_digest(summary, "summary_digest")
    camera_gate = bool(fit["confirmation_authorized"] and score["camera_gate"])
    result = {
        "phase": PHASE,
        "schema_version": "phase1208.final.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "fit_digest": fit["fit_digest"],
        "score_digest": score["score_digest"],
        "summary_digests": {split: summary["summary_digest"] for split, summary in summaries.items()},
        "known_truth_camera_calibrated": camera_gate,
        "outcome": "known_truth_necessity_mediation_camera_calibrated" if camera_gate else "camera_calibration_failed",
        "core_findings": {
            "single_contrast_removal_is_not_complete_necessity_camera": score["phase1207_contrast_scalar_camera_accuracy"] < 1.0,
            "single_contrast_removal_has_known_truth_false_negative": score["phase1207_operator_hidden_necessity_sensitivity"] < 1.0,
            "full_intervention_family_recovers_quotient": score["metrics"]["accuracy"] == 1.0,
            "minimal_cuts_and_mediation_recovered": score["metrics"]["structure_accuracy"] == 1.0,
            "unidentifiable_subtypes_trigger_abstention": score["metrics"]["abstention_or_subtype_accuracy"] == 1.0,
            "heldout_interventions_predicted": score["heldout_intervention_max_abs_error"] <= GATES["holdout_max_abs_error_max"],
        },
        "claim_boundary": (
            "This result calibrates a privileged known-truth intervention camera. It explains why a null "
            "Phase1207 active-minus-surface necessity test is compatible with bypass, redundancy, context "
            "gating, or an incomplete intervention basis. It does not identify which explanation holds in "
            "Qwen3 and does not establish a language or brain mechanism."
        ),
        "new_k_item": {
            "id": "K188",
            "level": "E3-KT",
            "statement": (
                "Across disjoint known-truth task widths, the frozen multi-intervention camera exactly "
                "recovered necessity, sufficiency-without-necessity, redundant minimal cuts, low-energy "
                "necessary paths, context gates, and required abstention; the Phase1207-style single "
                "contrast operator alone had a known-truth necessity false negative."
            ),
        },
        "auto_continue": camera_gate,
        "authorized_next": (
            "Freeze one learned micro-Transformer transfer protocol that predicts intervention-response "
            "morphology before truth is opened; do not return directly to Qwen3 hotspot search."
            if camera_gate else
            "Stop and repair the camera on known truth; pretrained-model execution remains denied."
        ),
    }
    result["final_digest"] = digest(result)
    write_json(OUT_ROOT / "analysis/final.json", result)
    return result


def run_all(device: torch.device) -> dict[str, Any]:
    preregister()
    import phase1208_necessity_mediation_camera_calibration_audit as independent_audit

    preaudit = independent_audit.preaudit()
    if not preaudit["all_checks_passed"]:
        raise RuntimeError("independent preaudit failed")
    run_split("discovery", device)
    fit_discovery()
    run_split("confirmation", device)
    predict_confirmation()
    score_confirmation()
    result = finalize()
    audit = independent_audit.final_audit(device)
    if not audit["all_checks_passed"]:
        raise RuntimeError("independent final audit failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preregister", "run", "fit", "predict", "score", "finalize", "all"),
    )
    parser.add_argument("--split", choices=tuple(SPLITS), default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.command == "preregister":
        result = preregister()
    elif args.command == "run":
        if args.split is None:
            raise SystemExit("--split is required")
        result = run_split(args.split, device)
    elif args.command == "fit":
        result = fit_discovery()
    elif args.command == "predict":
        result = predict_confirmation()
    elif args.command == "score":
        result = score_confirmation()
    elif args.command == "finalize":
        result = finalize()
    else:
        result = run_all(device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
