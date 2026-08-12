#!/usr/bin/env python3
"""Phase 1227: Qwen3 teacher-forced role-coalition external-validity test.

This phase has two jobs.  First, it corrects Phase1226's latent-variant claim:
u/v are labels on the same physical micro-system, not two implementations.
Second, it transfers only the *spatial* R/Q/B coalition camera to one frozen
Qwen3 scope and one architecture-defined residual depth.  It deliberately
does not combine teacher forcing, full-context regeneration, and KV-cache
generation into one temporal construct.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1224_final_layer_patch_construct_audit as p1224
import phase1226_known_truth_temporal_coalition_camera as p1226
from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1227
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1227_qwen3_teacher_forced_role_coalition_audit.py"

SOURCE_1223 = TEST_ROOT / "result/phase1223_passed_atom_physical_trajectory"
SOURCE_1224 = TEST_ROOT / "result/phase1224_final_layer_patch_construct_audit"
SOURCE_1225 = TEST_ROOT / "result/phase1225_qwen3_fp16_numerical_applicability"
SOURCE_1226 = TEST_ROOT / "result/phase1226_known_truth_temporal_coalition_camera"
SOURCE_PAIRS = SOURCE_1223 / "protocol/pair_manifest.jsonl"
SOURCE_STATES = SOURCE_1223 / "protocol/state_manifest.jsonl"

EXPECTED_FINAL_DIGESTS = {
    1223: "b1973184747d83a665b6dc3fd61bff4164e21aab0e2e14cfb2f5a69a57ab9304",
    1224: "f1ef7ef3669d4b3838b331e72b2135ab6548ecefdf0d30fe30eba99bb05f0edb",
    1225: "57b0a44926fb58c9555da5c8ffb6ff50b58c2bca3c83a0c6a92e51310f1d483a",
    1226: "8bab328b242007e77a3b29cdced126d0daea9af36f9ced759670a474d26903d1",
}
EXPECTED_AUDIT_DIGESTS = {
    1223: "d9c8a50299997ecdb1955419758121223118e7e568ecbbe49113bb71a6383542",
    1224: "a94373f4c84e9fbd338b9959034935b3be0afda3d80bde36f0f5ba57cced9805",
    1225: "e97b797ec6919f89e8c45cd5dc39da3590f94907ada293f7fdd96ddbc5063697",
    1226: "930fe07c78e052e838148e3ba14bef91595c229e4b60a4ad5c71844941c72c22",
}

OUT_ROOT = TEST_ROOT / "result/phase1227_qwen3_teacher_forced_role_coalition"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MANIFEST_PATH = OUT_ROOT / "protocol/anchor_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RECORD_PATH = OUT_ROOT / "runs/coalition_records.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
HOLDOUT_SPLITS = ("confirmation", "natural_use", "sealed")
PRIMARY_SCOPE = "query_relation|natural"
LAYER_COUNT = 36
FROZEN_DEPTH = LAYER_COUNT // 2
EPSILON = 1e-8

ROLE_GROUPS = {
    "R": ("record_object", "record_relation", "record_value"),
    "Q": ("query_subject", "query_relation"),
    "B": ("generation_boundary",),
}
ALLIANCES = {
    "R": ("R",),
    "Q": ("Q",),
    "B": ("B",),
    "RQ": ("R", "Q"),
    "RB": ("R", "B"),
    "QB": ("Q", "B"),
    "RQB": ("R", "Q", "B"),
}
CORRECT_CONDITIONS = tuple(f"correct:{name}" for name in ALLIANCES)
CONTROL_CONDITIONS = (
    "record_order:RQB",
    "paraphrase:RQB",
    "equal_norm_reverse:RQB",
    "identity:RQB",
    "zero:RQB",
)
CONDITIONS = CORRECT_CONDITIONS + CONTROL_CONDITIONS

THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "patch_calls_exact": 1,
    "hook_write_max_abs_max": 0.0,
    "causal_prefix_row_spread_max": 1e-4,
    "identity_score_drift_max": 1e-4,
    "recipient_accuracy_min": 0.75,
    "donor_accuracy_min": 0.75,
    "positive_target_shift_fraction_min": 0.75,
    "median_abs_target_shift_min": 1.0,
    "sufficient_completion_median_min": 0.50,
    "sufficient_positive_fraction_min": 0.75,
    "profile_range_min": 0.10,
    "discovery_leave_one_out_cosine_min": 0.50,
    "role_label_retrieval_fraction_min": 0.625,
    "full_correct_over_controls_min": 0.10,
    "holdout_profile_cosine_min": 0.50,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def median(values: Iterable[float]) -> float:
    data = list(values)
    return float(np.median(np.asarray(data, dtype=np.float64))) if data else 0.0


def source_root(phase: int) -> Path:
    return {1223: SOURCE_1223, 1224: SOURCE_1224, 1225: SOURCE_1225, 1226: SOURCE_1226}[phase]


def verify_sources() -> None:
    for phase in (1223, 1224, 1225, 1226):
        final = read_json(source_root(phase) / "analysis/final.json")
        audit = read_json(source_root(phase) / "audit/independent_result_audit.json")
        if final.get("final_digest") != EXPECTED_FINAL_DIGESTS[phase]:
            raise RuntimeError(f"Phase{phase} final digest drift")
        if audit.get("audit_digest") != EXPECTED_AUDIT_DIGESTS[phase] or not audit.get("all_checks_passed"):
            raise RuntimeError(f"Phase{phase} independent audit drift")


def phase1226_latent_variant_audit() -> dict[str, Any]:
    """Show that u/v differ only in metadata under the released implementation."""
    device = torch.device("cpu")
    checks: list[dict[str, Any]] = []
    for mechanism in p1226.MECHANISMS:
        left_spec = p1226.system_spec("discovery", 0, mechanism, "u")
        right_spec = p1226.system_spec("discovery", 0, mechanism, "v")
        left = p1226.KnownTruthRoleTransformer(left_spec, device).eval()
        right = p1226.KnownTruthRoleTransformer(right_spec, device).eval()
        physical_differences = 0
        max_abs = 0.0
        for key, left_value in left.state_dict().items():
            right_value = right.state_dict()[key]
            difference = float((left_value.float() - right_value.float()).abs().max().item())
            max_abs = max(max_abs, difference)
            physical_differences += int(difference != 0.0)
        left_public, left_heldout, _left_truth = p1226.response_record(left_spec, device)
        right_public, right_heldout, _right_truth = p1226.response_record(right_spec, device)
        response_equal = (
            left_public["correct_donor_responses"] == right_public["correct_donor_responses"]
            and left_heldout["responses"] == right_heldout["responses"]
        )
        checks.append({
            "mechanism": mechanism,
            "state_dict_tensor_difference_count": physical_differences,
            "state_dict_max_abs_difference": max_abs,
            "registered_response_equal": bool(response_equal),
            "system_id_differs": left_spec.system_id != right_spec.system_id,
            "latent_label_differs": left_spec.latent_variant != right_spec.latent_variant,
        })
    return {
        "scope": "released Phase1226 code, discovery replicate 0, all three mechanism families",
        "checks": checks,
        "all_physical_tensors_identical": all(row["state_dict_tensor_difference_count"] == 0 for row in checks),
        "all_registered_responses_identical": all(row["registered_response_equal"] for row in checks),
        "correction": (
            "Phase1226 abstention was tested against an arbitrary nonmechanistic u/v label attached "
            "to an identical physical system. It did not test two physically distinct, response-equivalent implementations."
        ),
    }


def build_manifest() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_sources()
    pairs = [row for row in read_jsonl(SOURCE_PAIRS) if row["scope"] == PRIMARY_SCOPE]
    states = read_jsonl(SOURCE_STATES)
    state_by_id = {row["state_id"]: row for row in states}
    if len(pairs) != 32 or Counter(row["split"] for row in pairs) != Counter({split: 8 for split in SPLITS}):
        raise RuntimeError("primary anchor cardinality drift")
    rows: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda row: (SPLITS.index(row["split"]), row["local_pair"])):
        panels = {name: state_by_id[state_id] for name, state_id in pair["panel_states"].items()}
        canonical = panels["canonical"]
        candidate_ids = {
            key: [int(value) for value in canonical["candidate_token_ids"][key]]
            for key in canonical["candidates"]
        }
        if pair["recipient_gold"] == pair["donor_gold"]:
            raise RuntimeError("recipient and donor gold must differ")
        for panel_name, state in panels.items():
            if state["candidate_token_ids"] != canonical["candidate_token_ids"]:
                raise RuntimeError(f"candidate tokenization drift in {panel_name}")
            if state["gold"] not in candidate_ids:
                raise RuntimeError(f"unknown panel gold in {panel_name}")
            for role in tuple(role for values in ROLE_GROUPS.values() for role in values):
                position = int(state["role_positions"][role])
                if not 0 <= position < len(state["input_ids"]):
                    raise RuntimeError(f"invalid role position {role} in {state['state_id']}")
        row: dict[str, Any] = {
            "schema_version": "phase1227.anchor-manifest.v1",
            "phase": PHASE,
            "pair_id": pair["pair_id"],
            "split": pair["split"],
            "scope": pair["scope"],
            "panel_state_ids": dict(pair["panel_states"]),
            "recipient_gold": pair["recipient_gold"],
            "donor_gold": pair["donor_gold"],
            "candidates": list(canonical["candidates"]),
            "candidate_token_ids": candidate_ids,
            "continuation_length": len(next(iter(candidate_ids.values()))),
            "panel_prompt_lengths": {name: len(state["input_ids"]) for name, state in panels.items()},
            "panel_role_positions": {name: dict(state["role_positions"]) for name, state in panels.items()},
        }
        row["row_digest"] = digest(row)
        rows.append(row)
    return rows, state_by_id


def materialize() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"formal output directory already exists: {OUT_ROOT}")
    manifest, _states = build_manifest()
    correction = phase1226_latent_variant_audit()
    protocol: dict[str, Any] = {
        "schema_version": "phase1227.preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "Test whether the Phase1226 spatial R/Q/B coalition response camera has a stable, "
            "role-identifiable teacher-forced signature in one frozen Qwen3 semantic scope."
        ),
        "primary_scope": PRIMARY_SCOPE,
        "splits": list(SPLITS),
        "holdout_splits": list(HOLDOUT_SPLITS),
        "pairs_per_split": 8,
        "numerical_type": {
            "model": "qwen3",
            "precision": "pure FP16 parameters on CUDA",
            "execution": "one teacher-forced full-candidate forward per intervention",
            "use_cache": False,
            "attention_mask": "all ones over the causal sequence",
            "candidate_rows": 4,
            "depth": FROZEN_DEPTH,
            "depth_selection": "architecture midpoint 36//2; no outcome search",
            "module": "output of decoder layer index 17 (residual_d18)",
            "scope_exclusions": [
                "full-context autoregressive regeneration",
                "incremental KV-cache generation",
                "temporal mechanism-family classification",
            ],
        },
        "role_groups": {key: list(value) for key, value in ROLE_GROUPS.items()},
        "alliances": {key: list(value) for key, value in ALLIANCES.items()},
        "conditions": list(CONDITIONS),
        "controls": {
            "record_order": "same semantics, reordered records",
            "paraphrase": "same semantics, alternative surface template",
            "equal_norm_reverse": "recipient minus the correct donor displacement",
            "identity": "recipient role state rewritten onto itself",
            "zero": "off-manifold zero role state",
        },
        "response": (
            "completion=(patched donor-vs-recipient margin - recipient margin) / "
            "(clean donor margin - clean recipient margin)"
        ),
        "discovery_rule": {
            "profile": "median completion for the seven correct-donor alliances",
            "sufficient_alliance": (
                "median completion >= 0.50 and positive completion fraction >= 0.75"
            ),
            "minimum_signature": "all sufficient alliances of minimum cardinality",
            "role_identification": "identity labeling must beat all five nonidentity R/Q/B permutations",
            "allowed_decisions": ["SPATIAL_SIGNATURE", "ABSTAIN"],
            "forbidden_decisions": ["boundary_store", "source_query_joint", "sustained_recompute"],
        },
        "holdout_rule": (
            "Freeze the discovery profile and minimum signature; require all three held-out splits "
            "to reproduce behavior, instrumentation, profile geometry, role labeling, minimum signature, and controls."
        ),
        "thresholds": THRESHOLDS,
        "phase1226_correction": correction,
        "claim_scope": [
            "A positive result is Qwen3-only, teacher-forced, one scope, one frozen depth, and spatial only.",
            "A negative result does not imply no role coalition exists at another depth or numerical type.",
            "No depth, role, scope, threshold, or control may be changed after materialization.",
            "The Phase1226 u/v abstention clause is downgraded; K203 retains only its mechanism/coalition/time and held-out prediction clauses.",
        ],
        "source_digests": {
            str(phase): {
                "final": EXPECTED_FINAL_DIGESTS[phase],
                "audit": EXPECTED_AUDIT_DIGESTS[phase],
            }
            for phase in (1223, 1224, 1225, 1226)
        },
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1223_pairs": file_sha256(SOURCE_PAIRS),
            "phase1223_states": file_sha256(SOURCE_STATES),
            "phase1226_main": file_sha256(TEST_ROOT / "phase1226_known_truth_temporal_coalition_camera.py"),
        },
        "material": {"count": len(manifest), "manifest_digest": digest(manifest)},
        "prohibited": [
            "depth search",
            "scope replacement",
            "post-hoc threshold changes",
            "calling teacher forcing a temporal mechanism test",
            "claiming u/v are physically distinct",
            "running GLM4 or DS7B in this Qwen3-only phase",
        ],
    }
    protocol["protocol_digest"] = digest(protocol)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"status": "materialized", "pairs": len(manifest), "protocol_digest": protocol["protocol_digest"]}))


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_sources()
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    claimed = protocol["protocol_digest"]
    if claimed != digest({key: value for key, value in protocol.items() if key != "protocol_digest"}):
        raise RuntimeError("protocol digest drift")
    if protocol["source_hashes"]["main"] != file_sha256(SCRIPT):
        raise RuntimeError("main script changed after freeze")
    if protocol["source_hashes"]["audit"] != file_sha256(AUDIT_SCRIPT):
        raise RuntimeError("audit script changed after freeze")
    if protocol["material"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest digest drift")
    states = {row["state_id"]: row for row in read_jsonl(SOURCE_STATES)}
    return protocol, manifest, states


def roles_for_alliance(name: str) -> tuple[str, ...]:
    return tuple(role for group in ALLIANCES[name] for role in ROLE_GROUPS[group])


def role_row_spread(hidden: torch.Tensor, state: dict[str, Any]) -> float:
    maximum = 0.0
    for role in tuple(role for values in ROLE_GROUPS.values() for role in values):
        position = int(state["role_positions"][role])
        maximum = max(maximum, float((hidden[:, position].float() - hidden[0:1, position].float()).abs().max().item()))
    return maximum


def replacement_for(
    recipient_hidden: torch.Tensor,
    recipient_state: dict[str, Any],
    source_hidden: torch.Tensor,
    source_state: dict[str, Any],
    roles: tuple[str, ...],
    mode: str = "source",
) -> tuple[list[int], torch.Tensor]:
    replacement = recipient_hidden.clone()
    positions: list[int] = []
    for role in roles:
        target = int(recipient_state["role_positions"][role])
        source = int(source_state["role_positions"][role])
        recipient_vector = recipient_hidden[0, target]
        source_vector = source_hidden[0, source]
        if mode == "source":
            vector = source_vector
        elif mode == "reverse":
            vector = recipient_vector - (source_vector - recipient_vector)
        elif mode == "identity":
            vector = recipient_vector
        elif mode == "zero":
            vector = torch.zeros_like(recipient_vector)
        else:
            raise ValueError(mode)
        replacement[:, target, :] = vector[None, :]
        positions.append(target)
    return positions, replacement


def scores_and_margin(
    logits: torch.Tensor,
    candidates: list[str],
    candidate_ids: dict[str, list[int]],
    donor_gold: str,
    recipient_gold: str,
) -> tuple[dict[str, float], float, bool]:
    scores, _tokens, finite = p1224.score_from_logits(logits, candidates, candidate_ids)
    return scores, p1224.fixed_margin(scores, donor_gold, recipient_gold), finite


def patched_condition(
    model: Any,
    module: Any,
    recipient_input: torch.Tensor,
    continuation_length: int,
    candidates: list[str],
    candidate_ids: dict[str, list[int]],
    donor_gold: str,
    recipient_gold: str,
    recipient_margin: float,
    donor_margin: float,
    positions: list[int],
    replacement: torch.Tensor,
) -> dict[str, Any]:
    logits, calls, write_max_abs = p1224.forward_patch(
        model, module, recipient_input, continuation_length, positions, replacement
    )
    scores, margin, finite = scores_and_margin(logits, candidates, candidate_ids, donor_gold, recipient_gold)
    target_shift = donor_margin - recipient_margin
    completion = (margin - recipient_margin) / target_shift if abs(target_shift) > EPSILON else 0.0
    result = {
        "finite": bool(finite),
        "patch_calls": int(calls),
        "hook_write_max_abs": float(write_max_abs),
        "margin": float(margin),
        "shift": float(margin - recipient_margin),
        "completion": float(completion),
        "scores": scores,
    }
    del logits
    return result


def run() -> None:
    protocol, manifest, state_by_id = verify_formal_inputs()
    if RECORD_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("formal outputs already exist")
    started = time.time()
    records: list[dict[str, Any]] = []
    model = None
    try:
        model, _tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 is not pure FP16")
        layers = get_layers(model)
        if len(layers) != LAYER_COUNT:
            raise RuntimeError("Qwen3 layer count drift")
        module = layers[FROZEN_DEPTH - 1]

        for item_index, item in enumerate(manifest):
            panels = {name: state_by_id[state_id] for name, state_id in item["panel_state_ids"].items()}
            recipient = panels["canonical"]
            donor = panels["binding_permutation"]
            candidates = list(item["candidates"])
            candidate_ids = {key: [int(value) for value in item["candidate_token_ids"][key]] for key in candidates}

            inputs: dict[str, torch.Tensor] = {}
            hidden: dict[str, torch.Tensor] = {}
            logits: dict[str, torch.Tensor] = {}
            continuation_length = int(item["continuation_length"])
            for panel_name in ("canonical", "binding_permutation", "record_order", "paraphrase"):
                panel_input, panel_continuation, _boundary = p1224.make_batch(panels[panel_name], candidates, device)
                if panel_continuation != continuation_length:
                    raise RuntimeError("continuation geometry drift")
                panel_logits, panel_hidden = p1224.forward_capture(model, module, panel_input, continuation_length)
                inputs[panel_name] = panel_input
                logits[panel_name] = panel_logits
                hidden[panel_name] = panel_hidden

            recipient_scores, recipient_margin, recipient_finite = scores_and_margin(
                logits["canonical"], candidates, candidate_ids, item["donor_gold"], item["recipient_gold"]
            )
            donor_scores, donor_margin, donor_finite = scores_and_margin(
                logits["binding_permutation"], candidates, candidate_ids, item["donor_gold"], item["recipient_gold"]
            )
            target_shift = donor_margin - recipient_margin
            conditions: dict[str, dict[str, Any]] = {}

            for alliance in ALLIANCES:
                positions, replacement = replacement_for(
                    hidden["canonical"], recipient, hidden["binding_permutation"], donor, roles_for_alliance(alliance)
                )
                conditions[f"correct:{alliance}"] = patched_condition(
                    model, module, inputs["canonical"], continuation_length, candidates, candidate_ids,
                    item["donor_gold"], item["recipient_gold"], recipient_margin, donor_margin,
                    positions, replacement,
                )

            for panel_name in ("record_order", "paraphrase"):
                positions, replacement = replacement_for(
                    hidden["canonical"], recipient, hidden[panel_name], panels[panel_name], roles_for_alliance("RQB")
                )
                conditions[f"{panel_name}:RQB"] = patched_condition(
                    model, module, inputs["canonical"], continuation_length, candidates, candidate_ids,
                    item["donor_gold"], item["recipient_gold"], recipient_margin, donor_margin,
                    positions, replacement,
                )

            for control_name, mode in (("equal_norm_reverse", "reverse"), ("identity", "identity"), ("zero", "zero")):
                positions, replacement = replacement_for(
                    hidden["canonical"], recipient, hidden["binding_permutation"], donor,
                    roles_for_alliance("RQB"), mode=mode,
                )
                conditions[f"{control_name}:RQB"] = patched_condition(
                    model, module, inputs["canonical"], continuation_length, candidates, candidate_ids,
                    item["donor_gold"], item["recipient_gold"], recipient_margin, donor_margin,
                    positions, replacement,
                )

            recipient_prediction = max(recipient_scores, key=lambda key: (recipient_scores[key], key))
            donor_prediction = max(donor_scores, key=lambda key: (donor_scores[key], key))
            record: dict[str, Any] = {
                "schema_version": "phase1227.coalition-record.v1",
                "phase": PHASE,
                "pair_id": item["pair_id"],
                "split": item["split"],
                "scope": item["scope"],
                "depth": FROZEN_DEPTH,
                "recipient_gold": item["recipient_gold"],
                "donor_gold": item["donor_gold"],
                "recipient_prediction": recipient_prediction,
                "donor_prediction": donor_prediction,
                "recipient_correct": recipient_prediction == item["recipient_gold"],
                "donor_correct": donor_prediction == item["donor_gold"],
                "recipient_finite": bool(recipient_finite),
                "donor_finite": bool(donor_finite),
                "recipient_scores": recipient_scores,
                "donor_scores": donor_scores,
                "recipient_margin": float(recipient_margin),
                "donor_margin": float(donor_margin),
                "target_shift": float(target_shift),
                "causal_prefix_row_spread": {
                    name: role_row_spread(hidden[name], panels[name]) for name in hidden
                },
                "conditions": conditions,
            }
            record["record_digest"] = digest(record)
            records.append(record)
            print(canonical_json({
                "phase": PHASE,
                "pair": item_index + 1,
                "total": len(manifest),
                "split": item["split"],
                "target_shift": round(target_shift, 4),
                "full_completion": round(conditions["correct:RQB"]["completion"], 4),
            }), flush=True)

            del inputs, hidden, logits, conditions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_jsonl(RECORD_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "record_count": len(records),
            "record_digest": digest(records),
            "model": "qwen3",
            "device": str(device),
            "placement": placement,
            "precision_audit": precision,
            "depth": FROZEN_DEPTH,
            "elapsed_seconds": float(time.time() - started),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def centered_cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > EPSILON else 0.0


def canonical_alliance(groups: Iterable[str]) -> str:
    selected = set(groups)
    return "".join(role for role in ("R", "Q", "B") if role in selected)


def permuted_profile(profile: dict[str, float], permutation: tuple[str, str, str]) -> list[float]:
    mapping = dict(zip(("R", "Q", "B"), permutation))
    transformed: dict[str, float] = {}
    for alliance, value in profile.items():
        transformed[canonical_alliance(mapping[role] for role in ALLIANCES[alliance])] = value
    return [float(transformed[name]) for name in ALLIANCES]


def role_retrieval(observed: dict[str, float], prototype: dict[str, float]) -> tuple[bool, float, float]:
    observed_vector = [observed[name] for name in ALLIANCES]
    permutations = list(itertools.permutations(("R", "Q", "B")))
    scores = [centered_cosine(observed_vector, permuted_profile(prototype, permutation)) for permutation in permutations]
    identity_index = permutations.index(("R", "Q", "B"))
    alternatives = [score for index, score in enumerate(scores) if index != identity_index]
    identity = scores[identity_index]
    best_alternative = max(alternatives)
    return bool(identity > best_alternative + 1e-6), float(identity), float(best_alternative)


def pair_profile(record: dict[str, Any]) -> dict[str, float]:
    return {name: float(record["conditions"][f"correct:{name}"]["completion"]) for name in ALLIANCES}


def aggregate_split(rows: list[dict[str, Any]], prototype: dict[str, float] | None = None) -> dict[str, Any]:
    profiles = [pair_profile(row) for row in rows]
    profile = {name: median(item[name] for item in profiles) for name in ALLIANCES}
    positive = {
        name: float(np.mean([item[name] > 0.0 for item in profiles])) for name in ALLIANCES
    }
    sufficient = [
        name for name in ALLIANCES
        if profile[name] >= THRESHOLDS["sufficient_completion_median_min"]
        and positive[name] >= THRESHOLDS["sufficient_positive_fraction_min"]
    ]
    if sufficient:
        minimum_size = min(len(ALLIANCES[name]) for name in sufficient)
        minimum_signature = sorted(name for name in sufficient if len(ALLIANCES[name]) == minimum_size)
    else:
        minimum_signature = []

    retrieval_rows: list[dict[str, float | bool]] = []
    if prototype is None:
        for index, observed in enumerate(profiles):
            other = [item for other_index, item in enumerate(profiles) if other_index != index]
            leave_one_out = {name: median(item[name] for item in other) for name in ALLIANCES}
            passed, identity, alternative = role_retrieval(observed, leave_one_out)
            retrieval_rows.append({"passed": passed, "identity_cosine": identity, "best_permuted_cosine": alternative})
    else:
        for observed in profiles:
            passed, identity, alternative = role_retrieval(observed, prototype)
            retrieval_rows.append({"passed": passed, "identity_cosine": identity, "best_permuted_cosine": alternative})

    control_names = ("record_order:RQB", "paraphrase:RQB", "equal_norm_reverse:RQB", "identity:RQB", "zero:RQB")
    control_medians = {
        name: median(row["conditions"][name]["completion"] for row in rows) for name in control_names
    }
    full_correct = profile["RQB"]
    full_over_controls = full_correct - max(control_medians.values())
    identity_drifts = []
    for row in rows:
        baseline = row["recipient_scores"]
        identity = row["conditions"]["identity:RQB"]["scores"]
        identity_drifts.append(max(abs(float(identity[key]) - float(baseline[key])) for key in baseline))
    write_values = [
        float(condition["hook_write_max_abs"])
        for row in rows for condition in row["conditions"].values()
    ]
    calls = [int(condition["patch_calls"]) for row in rows for condition in row["conditions"].values()]
    finite = [
        bool(row["recipient_finite"] and row["donor_finite"])
        and all(condition["finite"] for condition in row["conditions"].values())
        for row in rows
    ]
    row_spreads = [float(value) for row in rows for value in row["causal_prefix_row_spread"].values()]
    target_shifts = [float(row["target_shift"]) for row in rows]
    result: dict[str, Any] = {
        "count": len(rows),
        "recipient_accuracy": float(np.mean([row["recipient_correct"] for row in rows])),
        "donor_accuracy": float(np.mean([row["donor_correct"] for row in rows])),
        "finite_fraction": float(np.mean(finite)),
        "positive_target_shift_fraction": float(np.mean([value > 0.0 for value in target_shifts])),
        "median_abs_target_shift": median(abs(value) for value in target_shifts),
        "profile": profile,
        "positive_completion_fraction": positive,
        "profile_range": float(max(profile.values()) - min(profile.values())),
        "sufficient_alliances": sufficient,
        "minimum_signature": minimum_signature,
        "role_retrieval_fraction": float(np.mean([row["passed"] for row in retrieval_rows])),
        "role_identity_cosine_median": median(float(row["identity_cosine"]) for row in retrieval_rows),
        "role_best_permuted_cosine_median": median(float(row["best_permuted_cosine"]) for row in retrieval_rows),
        "control_medians": control_medians,
        "full_correct_over_max_control": float(full_over_controls),
        "max_identity_score_drift": float(max(identity_drifts)),
        "max_hook_write_abs": float(max(write_values)),
        "patch_call_values": sorted(set(calls)),
        "max_causal_prefix_row_spread": float(max(row_spreads)),
    }
    if prototype is not None:
        result["profile_cosine_to_discovery"] = centered_cosine(
            [profile[name] for name in ALLIANCES], [prototype[name] for name in ALLIANCES]
        )
    return result


def instrumentation_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "finite": summary["finite_fraction"] >= THRESHOLDS["finite_fraction_min"],
        "patch_calls": summary["patch_call_values"] == [THRESHOLDS["patch_calls_exact"]],
        "hook_write": summary["max_hook_write_abs"] <= THRESHOLDS["hook_write_max_abs_max"],
        "causal_prefix": summary["max_causal_prefix_row_spread"] <= THRESHOLDS["causal_prefix_row_spread_max"],
        "identity": summary["max_identity_score_drift"] <= THRESHOLDS["identity_score_drift_max"],
    }


def behavior_gates(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "recipient": summary["recipient_accuracy"] >= THRESHOLDS["recipient_accuracy_min"],
        "donor": summary["donor_accuracy"] >= THRESHOLDS["donor_accuracy_min"],
        "target_sign": summary["positive_target_shift_fraction"] >= THRESHOLDS["positive_target_shift_fraction_min"],
        "target_size": summary["median_abs_target_shift"] >= THRESHOLDS["median_abs_target_shift_min"],
    }


def analyze() -> None:
    protocol, manifest, _states = verify_formal_inputs()
    records = read_jsonl(RECORD_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)
    if len(records) != len(manifest) or run_summary["record_digest"] != digest(records):
        raise RuntimeError("record integrity drift")
    by_split = {split: [row for row in records if row["split"] == split] for split in SPLITS}
    discovery = aggregate_split(by_split["discovery"])
    discovery_instrument = instrumentation_gates(discovery)
    discovery_behavior = behavior_gates(discovery)
    discovery_signature_gates = {
        "has_sufficient_alliance": bool(discovery["minimum_signature"]),
        "profile_range": discovery["profile_range"] >= THRESHOLDS["profile_range_min"],
        "leave_one_out_cosine": discovery["role_identity_cosine_median"] >= THRESHOLDS["discovery_leave_one_out_cosine_min"],
        "role_label_retrieval": discovery["role_retrieval_fraction"] >= THRESHOLDS["role_label_retrieval_fraction_min"],
        "controls": discovery["full_correct_over_max_control"] >= THRESHOLDS["full_correct_over_controls_min"],
    }
    discovery_gate = all(discovery_instrument.values()) and all(discovery_behavior.values()) and all(discovery_signature_gates.values())
    decision = "SPATIAL_SIGNATURE" if discovery_gate else "ABSTAIN"

    holdouts: dict[str, Any] = {}
    holdout_passes: list[bool] = []
    for split in HOLDOUT_SPLITS:
        summary = aggregate_split(by_split[split], prototype=discovery["profile"])
        instrument = instrumentation_gates(summary)
        behavior = behavior_gates(summary)
        prediction = {
            "profile_cosine": summary["profile_cosine_to_discovery"] >= THRESHOLDS["holdout_profile_cosine_min"],
            "role_label_retrieval": summary["role_retrieval_fraction"] >= THRESHOLDS["role_label_retrieval_fraction_min"],
            "minimum_signature": summary["minimum_signature"] == discovery["minimum_signature"],
            "controls": summary["full_correct_over_max_control"] >= THRESHOLDS["full_correct_over_controls_min"],
        }
        passed = bool(discovery_gate and all(instrument.values()) and all(behavior.values()) and all(prediction.values()))
        holdout_passes.append(passed)
        holdouts[split] = {
            "summary": summary,
            "instrumentation_gates": instrument,
            "behavior_gates": behavior,
            "prediction_gates": prediction,
            "passed": passed,
        }

    external_validity_gate = bool(discovery_gate and all(holdout_passes))
    if external_validity_gate:
        status = "qwen3_teacher_forced_spatial_signature_confirmed"
        k_statement = (
            "At Qwen3 residual_d18 in query_relation|natural teacher-forced scoring, a frozen R/Q/B "
            "coalition response signature and its minimum sufficient alliance repeat across three held-out splits "
            "and exceed registered controls."
        )
        evidence_grade = "E2-QWEN-SINGLE"
    else:
        status = "qwen3_teacher_forced_spatial_signature_not_confirmed"
        k_statement = (
            "The Phase1226 spatial coalition camera did not close a role-identifiable, control-exceeding response "
            "signature at Qwen3 residual_d18 in the frozen query_relation|natural teacher-forced scope. This is a "
            "typed local boundary, not evidence against all depths or autoregressive numerical types."
        )
        evidence_grade = "E3-NEGATIVE-BOUNDARY"

    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": status,
        "protocol_digest": protocol["protocol_digest"],
        "run_summary_digest": run_summary["summary_digest"],
        "record_digest": digest(records),
        "phase1226_correction": protocol["phase1226_correction"],
        "result": {
            "decision": decision,
            "external_validity_gate": external_validity_gate,
            "discovery": {
                "summary": discovery,
                "instrumentation_gates": discovery_instrument,
                "behavior_gates": discovery_behavior,
                "signature_gates": discovery_signature_gates,
                "passed": discovery_gate,
            },
            "holdouts": holdouts,
        },
        "k_items": [
            {
                "identifier": "K203-correction",
                "evidence_grade": "E3-CODE-AUDIT",
                "statement": (
                    "Phase1226 u/v were physically identical under the released code; its latent-abstention clause "
                    "is withdrawn. K203 retains its known-truth mechanism/coalition/time and held-out prediction result."
                ),
            },
            {"identifier": "K204", "evidence_grade": evidence_grade, "statement": k_statement},
        ],
        "claim_boundary": list(protocol["claim_scope"]),
        "mathematics": {
            "new_mathematics_required": False,
            "interpretation": (
                "This phase tests a finite typed response quotient with ordinary margins, normalized responses, "
                "permutation controls, and holdout prediction. It does not identify a global semantic state space."
            ),
        },
        "authorization": {
            "automatic_execution": False,
            "auto_continue": 0,
            "reason": (
                "Teacher forcing, full-context regeneration, and KV-cache generation require separate intervention "
                "contracts. This phase neither authorizes a depth search nor silently promotes a spatial result into a temporal claim."
            ),
            "next_if_positive": "freeze one full-context no-cache rollout contract before execution",
            "next_if_negative": "stop local depth search; redesign the state object as a trajectory/response signature",
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": status, "decision": decision, "external_validity_gate": external_validity_gate, "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("materialize", "run", "analyze"))
    args = parser.parse_args()
    if args.stage == "materialize":
        materialize()
    elif args.stage == "run":
        run()
    else:
        analyze()


if __name__ == "__main__":
    main()
