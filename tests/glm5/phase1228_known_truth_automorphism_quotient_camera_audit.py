#!/usr/bin/env python3
"""Independent audit for Phase 1228.

The audit intentionally does not import the experiment module.  It rebuilds
the manifest, response tables, stabilizers, orbits, minimum alliances, and
physical-pair claims from the frozen files and an independent implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1228
SCRIPT = TEST_ROOT / "phase1228_known_truth_automorphism_quotient_camera.py"
AUDIT_SCRIPT = Path(__file__).resolve()
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
PREDICTION_MARKER = OUT_ROOT / "confirmation/prediction_marker.json"
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def alliance_bits(name: str) -> tuple[int, int, int]:
    return tuple(int(role in name) for role in ROLES)  # type: ignore[return-value]


def canonical_alliance(roles: Iterable[str]) -> str:
    selected = set(roles)
    return "".join(role for role in ROLES if role in selected)


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


def expected_profile(family: str, curvature: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ALLIANCES:
        base = base_value(family, *alliance_bits(name))
        values[name] = float(np.float16(base + curvature * base * (1.0 - base)))
    return values


def exact_stabilizer(family: str) -> list[str]:
    profile = base_profile(family)
    answer = []
    for permutation in PERMUTATIONS:
        error = max(abs(profile[name] - profile[permute_alliance(name, permutation)]) for name in ALLIANCES)
        if error <= 1e-12:
            answer.append(permutation_id(permutation))
    return sorted(answer)


def inferred_stabilizer(profile: dict[str, float], tolerance: float) -> tuple[list[str], dict[str, float]]:
    answer: list[str] = []
    errors: dict[str, float] = {}
    for permutation in PERMUTATIONS:
        identifier = permutation_id(permutation)
        error = max(abs(float(profile[name]) - float(profile[permute_alliance(name, permutation)])) for name in ALLIANCES)
        errors[identifier] = float(error)
        if error <= tolerance:
            answer.append(identifier)
    return sorted(answer), errors


def role_orbits(stabilizer: Iterable[str]) -> list[list[str]]:
    permutations = [tuple(value) for value in stabilizer]
    remaining = set(ROLES)
    answer: list[list[str]] = []
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
        answer.append(ordered)
        remaining -= orbit
    return sorted(answer, key=lambda values: tuple(ROLES.index(value) for value in values))


def minimum_alliances(profile: dict[str, float], threshold: float) -> list[str]:
    sufficient = [name for name in ALLIANCES if float(profile[name]) >= threshold]
    if not sufficient:
        return []
    minimum = min(len(name) for name in sufficient)
    return sorted((name for name in sufficient if len(name) == minimum), key=ALLIANCES.index)


def expected_manifest_row(split: str, family: str, replicate: int, gauge: str, hidden: str) -> dict[str, Any]:
    split_offset = 0 if split == "discovery" else 100_003
    family_index = FAMILIES.index(family)
    gauge_index = GAUGES.index(gauge)
    slot_permutations = list(itertools.permutations(range(3)))
    slot_index = (replicate + 2 * family_index + 3 * gauge_index + split_offset) % len(slot_permutations)
    slot_order = slot_permutations[slot_index]
    slot_by_role = {role: int(slot_order[index]) for index, role in enumerate(ROLES)}
    seed = 12280019 + split_offset + 1009 * replicate + 97 * family_index + 7919 * gauge_index
    rng = np.random.default_rng(seed)
    channel_permutation = [int(value) for value in rng.permutation(WIDTH)]
    channel_signs = [int(value) for value in rng.choice(np.asarray([-1, 1]), size=WIDTH)]
    grid = (-0.08, -0.04, 0.00, 0.04, 0.08) if split == "discovery" else (-0.07, -0.03, 0.01, 0.05, 0.09)
    curvature = float(grid[(replicate + family_index) % len(grid)])
    identity = {
        "phase": PHASE,
        "split": split,
        "family": family,
        "replicate": replicate,
        "gauge": gauge,
        "hidden": hidden,
    }
    row: dict[str, Any] = {
        "schema_version": "phase1228.system-manifest.v1",
        "phase": PHASE,
        "system_id": digest(identity)[:24],
        "split": split,
        "family": family,
        "replicate": replicate,
        "gauge_variant": gauge,
        "hidden_variant": hidden,
        "curvature": curvature,
        "slot_by_role": slot_by_role,
        "channel_permutation": channel_permutation,
        "channel_signs": channel_signs,
        "public_nonce": replicate % 2,
    }
    row["row_digest"] = digest(row)
    return row


def expected_manifest() -> list[dict[str, Any]]:
    return [
        expected_manifest_row(split, family, replicate, gauge, hidden)
        for split in SPLITS
        for family in FAMILIES
        for replicate in range(REPLICATES)
        for gauge in GAUGES
        for hidden in HIDDEN_VARIANTS
    ]


def forbidden_truth_reads(source: str, names: set[str]) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    selected: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
    combined = ast.Module(body=selected, type_ignores=[])
    found: set[tuple[str, str]] = set()
    for node in ast.walk(combined):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"read_json", "read_jsonl"} or not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Name) and argument.id in {"CONFIRMATION_SEALED", "DISCOVERY_TRUTH"}:
            found.add((node.func.id, argument.id))
    return found


def preaudit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    main_source = SCRIPT.read_text(encoding="utf-8")
    truth_reads = forbidden_truth_reads(main_source, {"predict_confirmation", "infer_camera"})
    expected = expected_manifest()
    counts = Counter((row["split"], row["family"], row["gauge_variant"], row["hidden_variant"]) for row in manifest)
    grouped: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest:
        grouped[(row["split"], row["family"], int(row["replicate"]), row["hidden_variant"])][row["gauge_variant"]] = row
    gauge_configs_differ = all(
        (values["u"]["slot_by_role"], values["u"]["channel_permutation"], values["u"]["channel_signs"])
        != (values["v"]["slot_by_role"], values["v"]["channel_permutation"], values["v"]["channel_signs"])
        for values in grouped.values()
    )
    hidden_groups: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest:
        hidden_groups[(row["split"], row["family"], int(row["replicate"]), row["gauge_variant"])][row["hidden_variant"]] = row
    hidden_public_configs_equal = all(
        all(values["h0"][key] == values["h1"][key] for key in (
            "curvature", "slot_by_role", "channel_permutation", "channel_signs", "public_nonce"
        ))
        for values in hidden_groups.values()
    )
    near = base_profile("near_qb_distinguishable")
    near_nonidentity_errors = [
        max(abs(near[name] - near[permute_alliance(name, permutation)]) for name in ALLIANCES)
        for permutation in PERMUTATIONS
        if permutation_id(permutation) != "RQB"
    ]
    expected_stabilizers = {family: exact_stabilizer(family) for family in FAMILIES}
    output_paths = (
        DISCOVERY_PUBLIC, DISCOVERY_TRUTH, CAMERA_PATH, CONFIRMATION_PUBLIC,
        CONFIRMATION_SEALED, PREDICTION_PATH, PREDICTION_MARKER, FINAL_PATH, RESULT_AUDIT_PATH,
    )
    source_final = read_json(SOURCE_FINAL)
    source_audit = read_json(SOURCE_AUDIT)
    aborted = read_json(ABORTED_PREAUDIT)
    checks = {
        "phase": protocol.get("phase") == PHASE,
        "revision": protocol.get("revision") == 1,
        "protocol_digest": protocol.get("protocol_digest") == digest(strip_digest(protocol, "protocol_digest")),
        "main_hash": protocol["source_hashes"]["main"] == file_sha256(SCRIPT),
        "audit_hash": protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT),
        "source_final_digest": source_final.get("final_digest") == EXPECTED_SOURCE_FINAL,
        "source_audit_digest": source_audit.get("audit_digest") == EXPECTED_SOURCE_AUDIT,
        "source_audit_pass": bool(source_audit.get("all_checks_passed")),
        "source_sha256": (
            protocol["source"]["phase1227_final_sha256"] == file_sha256(SOURCE_FINAL)
            and protocol["source"]["phase1227_audit_sha256"] == file_sha256(SOURCE_AUDIT)
        ),
        "manifest_count": len(manifest) == 2 * 7 * 48 * 2 * 2,
        "manifest_digest": protocol["material"]["digest"] == digest(manifest),
        "manifest_exact_rebuild": manifest == expected,
        "manifest_row_digests": all(row["row_digest"] == digest(strip_digest(row, "row_digest")) for row in manifest),
        "system_ids_unique": len({row["system_id"] for row in manifest}) == len(manifest),
        "split_ids_disjoint": not (
            {row["system_id"] for row in manifest if row["split"] == "discovery"}
            & {row["system_id"] for row in manifest if row["split"] == "confirmation"}
        ),
        "balanced_cells": len(counts) == 2 * 7 * 2 * 2 and set(counts.values()) == {48},
        "gauge_configs_differ": gauge_configs_differ,
        "hidden_public_configs_equal": hidden_public_configs_equal,
        "fixed_cuda_fp16": (
            protocol["numerical_type"]["device"] == "CUDA required"
            and protocol["numerical_type"]["dtype"].startswith("FP16")
            and protocol["numerical_type"]["fixed_batch_geometry"] == [8, 3, 8]
        ),
        "fixed_tolerance": protocol["equivalence_tolerance"] == EQUIVALENCE_TOLERANCE,
        "tolerance_origin_declared": "independent" in protocol["tolerance_origin"],
        "near_separation": min(near_nonidentity_errors) > EQUIVALENCE_TOLERANCE,
        "symbolic_stabilizers_nonempty": all(expected_stabilizers[family] for family in FAMILIES),
        "near_identity_only": expected_stabilizers["near_qb_distinguishable"] == ["RQB"],
        "asymmetric_identity_only": expected_stabilizers["fully_asymmetric"] == ["RQB"],
        "symmetric_positive_controls": (
            len(expected_stabilizers["cardinality_symmetric"]) == 6
            and len(expected_stabilizers["r_gate"]) == 2
            and len(expected_stabilizers["rq_joint"]) == 2
        ),
        "aborted_preflight_preserved": (
            aborted.get("audit_digest") == EXPECTED_ABORTED_PREAUDIT
            and aborted.get("all_checks_passed") is False
            and aborted.get("passed_count") == 30
            and aborted.get("check_count") == 31
        ),
        "prediction_source_no_truth_reads": not truth_reads,
        "no_formal_outputs_before_preaudit": not any(path.exists() for path in output_paths),
        "claim_scope": "not a language-mechanism result" in " ".join(protocol["claim_scope"]),
        "approximation_boundary": "call approximate similarity strict equivalence" in protocol["prohibited"],
        "confirmation_commitment_contract": protocol["split_discipline"]["confirmation sealed truth is opened only by reveal stage"],
        "no_pretrained_models": protocol["numerical_type"]["pretrained_language_model"] is False,
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_preaudit",
        "created_at_utc": datetime.now().astimezone().isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "manifest_count": len(manifest),
        "expected_stabilizers": expected_stabilizers,
        "near_min_nonidentity_error": float(min(near_nonidentity_errors)),
        "protocol_digest": protocol["protocol_digest"],
    }
    result["audit_digest"] = digest(result)
    write_json(PREAUDIT_PATH, result)
    return result


def compare_dict_floats(left: dict[str, float], right: dict[str, float], tolerance: float = 1e-12) -> bool:
    return set(left) == set(right) and all(abs(float(left[key]) - float(right[key])) <= tolerance for key in left)


def score_independent(
    public_rows: list[dict[str, Any]], truth_rows: list[dict[str, Any]], manifest_by_id: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], float]:
    truth_by_id = {row["system_id"]: row for row in truth_rows}
    predictions: dict[str, dict[str, Any]] = {}
    stabilizer_ok: list[bool] = []
    orbit_ok: list[bool] = []
    minimum_ok: list[bool] = []
    near_ok: list[bool] = []
    metadata_ok: list[bool] = []
    leak_ok: list[bool] = []
    expected_error = 0.0
    for public in public_rows:
        system_id = public["system_id"]
        truth = truth_by_id[system_id]
        manifest = manifest_by_id[system_id]
        stabilizer, errors = inferred_stabilizer(public["profile"], EQUIVALENCE_TOLERANCE)
        orbits = role_orbits(stabilizer)
        minimum = minimum_alliances(public["profile"], SUFFICIENT_THRESHOLD)
        predictions[system_id] = {
            "inferred_stabilizer": stabilizer,
            "inferred_orbits": orbits,
            "minimum_alliances": minimum,
            "permutation_errors": errors,
            "gauge_variant_decision": "ABSTAIN",
            "hidden_variant_decision_public": "ABSTAIN",
        }
        stabilizer_ok.append(stabilizer == truth["truth_stabilizer"])
        orbit_ok.append(orbits == truth["truth_orbits"])
        minimum_ok.append(minimum == truth["truth_minimum_alliances"])
        if manifest["family"] == "near_qb_distinguishable":
            near_ok.append(stabilizer == ["RQB"])
        metadata_guess = "u" if int(public["public_nonce"]) == 0 else "v"
        metadata_ok.append(metadata_guess == truth["gauge_variant"])
        leak_guess = "u" if int(truth["leaky_sentinel_feature"]) == 0 else "v"
        leak_ok.append(leak_guess == truth["gauge_variant"])
        expected = expected_profile(manifest["family"], float(manifest["curvature"]))
        expected_error = max(expected_error, max(abs(float(public["profile"][name]) - expected[name]) for name in ALLIANCES))
    metrics = {
        "count": len(public_rows),
        "finite_fraction": float(np.mean([bool(row["finite"]) for row in public_rows])),
        "baseline_max_abs": float(max(abs(float(row["baseline"])) for row in public_rows)),
        "full_response_min": float(min(float(row["full_response"]) for row in public_rows)),
        "structure_accuracy": float(np.mean([a and b and c for a, b, c in zip(stabilizer_ok, orbit_ok, minimum_ok)])),
        "stabilizer_accuracy": float(np.mean(stabilizer_ok)),
        "orbit_accuracy": float(np.mean(orbit_ok)),
        "minimum_alliance_accuracy": float(np.mean(minimum_ok)),
        "near_distinguishable_accuracy": float(np.mean(near_ok)),
        "gauge_public_abstention": 1.0,
        "hidden_public_abstention": 1.0,
        "metadata_null_accuracy": float(np.mean(metadata_ok)),
        "leaky_sentinel_accuracy": float(np.mean(leak_ok)),
    }
    return metrics, predictions, expected_error


def physical_independent(
    public_rows: list[dict[str, Any]], truth_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    public_by_id = {row["system_id"]: row for row in public_rows}
    grouped: dict[tuple[str, str, int], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for truth in truth_rows:
        key = (truth["split"], truth["family"], int(truth["replicate"]))
        grouped[key][(truth["gauge_variant"], truth["hidden_variant"])] = truth
    gauge_state: list[bool] = []
    gauge_hidden: list[bool] = []
    gauge_public: list[float] = []
    gauge_sealed: list[float] = []
    hidden_state: list[bool] = []
    hidden_public: list[float] = []
    hidden_sealed: list[float] = []
    for values in grouped.values():
        for hidden in HIDDEN_VARIANTS:
            left, right = values[("u", hidden)], values[("v", hidden)]
            gauge_state.append(left["state_dict_digest"] != right["state_dict_digest"])
            gauge_hidden.append(left["hidden_state_digest"] != right["hidden_state_digest"])
            lp, rp = public_by_id[left["system_id"]]["profile"], public_by_id[right["system_id"]]["profile"]
            gauge_public.append(max(abs(float(lp[name]) - float(rp[name])) for name in ALLIANCES))
            gauge_sealed.append(abs(float(left["sealed_probe_response"]) - float(right["sealed_probe_response"])))
        for gauge in GAUGES:
            left, right = values[(gauge, "h0")], values[(gauge, "h1")]
            hidden_state.append(left["state_dict_digest"] != right["state_dict_digest"])
            lp, rp = public_by_id[left["system_id"]]["profile"], public_by_id[right["system_id"]]["profile"]
            hidden_public.append(max(abs(float(lp[name]) - float(rp[name])) for name in ALLIANCES))
            hidden_sealed.append(abs(float(left["sealed_probe_response"]) - float(right["sealed_probe_response"])))
    return {
        "gauge_pair_count": len(gauge_state),
        "hidden_pair_count": len(hidden_state),
        "physical_state_dict_difference_fraction": float(np.mean(gauge_state)),
        "physical_hidden_difference_fraction": float(np.mean(gauge_hidden)),
        "gauge_public_profile_max_abs": float(max(gauge_public)),
        "gauge_sealed_profile_max_abs": float(max(gauge_sealed)),
        "hidden_state_dict_difference_fraction": float(np.mean(hidden_state)),
        "hidden_public_profile_max_abs": float(max(hidden_public)),
        "hidden_sealed_difference_min": float(min(hidden_sealed)),
    }


def metrics_match(observed: dict[str, Any], expected: dict[str, Any], tolerance: float = 1e-12) -> bool:
    for key, value in expected.items():
        if key not in observed:
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if abs(float(observed[key]) - float(value)) > tolerance:
                return False
        elif observed[key] != value:
            return False
    return True


def result_audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    pre = read_json(PREAUDIT_PATH)
    discovery_public = read_jsonl(DISCOVERY_PUBLIC)
    discovery_truth = read_jsonl(DISCOVERY_TRUTH)
    confirmation_public = read_jsonl(CONFIRMATION_PUBLIC)
    confirmation_truth = read_jsonl(CONFIRMATION_SEALED)
    predictions = read_jsonl(PREDICTION_PATH)
    marker = read_json(PREDICTION_MARKER)
    camera = read_json(CAMERA_PATH)
    final = read_json(FINAL_PATH)
    overall = read_json(RUN_SUMMARY_PATH)
    discovery_summary = read_json(OUT_ROOT / "runs/discovery_summary.json")
    confirmation_summary = read_json(OUT_ROOT / "runs/confirmation_summary.json")
    manifest_by_id = {row["system_id"]: row for row in manifest}
    d_metrics, d_predictions, d_response_error = score_independent(discovery_public, discovery_truth, manifest_by_id)
    c_metrics, c_predictions, c_response_error = score_independent(confirmation_public, confirmation_truth, manifest_by_id)
    d_physical = physical_independent(discovery_public, discovery_truth)
    c_physical = physical_independent(confirmation_public, confirmation_truth)
    truth_by_id = {row["system_id"]: row for row in confirmation_truth}
    prediction_by_id = {row["system_id"]: row for row in predictions}
    prediction_exact: list[bool] = []
    sealed_hidden: list[bool] = []
    sealed_gauge: list[bool] = []
    for public in confirmation_public:
        system_id = public["system_id"]
        observed = prediction_by_id[system_id]
        expected = c_predictions[system_id]
        prediction_exact.append(
            observed["inferred_stabilizer"] == expected["inferred_stabilizer"]
            and observed["inferred_orbits"] == expected["inferred_orbits"]
            and observed["minimum_alliances"] == expected["minimum_alliances"]
            and compare_dict_floats(observed["permutation_errors"], expected["permutation_errors"])
            and observed["gauge_variant_decision"] == "ABSTAIN"
            and observed["hidden_variant_decision_public"] == "ABSTAIN"
        )
        truth = truth_by_id[system_id]
        hidden_guess = "h1" if float(truth["sealed_probe_response"]) > SEALED_HIDDEN_THRESHOLD else "h0"
        sealed_hidden.append(hidden_guess == truth["hidden_variant"])
        sealed_gauge.append(observed["sealed_gauge_decision"] == "ABSTAIN")
    c_metrics_extended = dict(c_metrics)
    c_metrics_extended.update({
        "prediction_reproduction_accuracy": float(np.mean(prediction_exact)),
        "sealed_hidden_accuracy": float(np.mean(sealed_hidden)),
        "sealed_gauge_abstention": float(np.mean(sealed_gauge)),
    })
    all_public = discovery_public + confirmation_public
    all_truth = discovery_truth + confirmation_truth
    public_digests = all(row["public_digest"] == digest(strip_digest(row, "public_digest")) for row in all_public)
    truth_digests = all(row["truth_digest"] == digest(strip_digest(row, "truth_digest")) for row in all_truth)
    prediction_digests = all(row["prediction_digest"] == digest(strip_digest(row, "prediction_digest")) for row in predictions)
    public_ids = {row["system_id"] for row in all_public}
    truth_ids = {row["system_id"] for row in all_truth}
    manifest_ids = {row["system_id"] for row in manifest}
    prediction_ids = {row["system_id"] for row in predictions}
    near_confirmation_separation = min(
        min(value for key, value in c_predictions[row["system_id"]]["permutation_errors"].items() if key != "RQB")
        for row in confirmation_public
        if manifest_by_id[row["system_id"]]["family"] == "near_qb_distinguishable"
    )
    marker_time = datetime.fromisoformat(marker["created_at_utc"])
    final_time = datetime.fromisoformat(final["created_at_utc"])
    checks = {
        "preaudit_pass": bool(pre.get("all_checks_passed")),
        "preaudit_digest": pre["audit_digest"] == digest(strip_digest(pre, "audit_digest")),
        "protocol_digest": protocol["protocol_digest"] == digest(strip_digest(protocol, "protocol_digest")),
        "source_immutability": (
            protocol["source_hashes"]["main"] == file_sha256(SCRIPT)
            and protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT)
        ),
        "manifest_immutability": protocol["material"]["digest"] == digest(manifest),
        "manifest_exact_rebuild": manifest == expected_manifest(),
        "counts": (
            len(discovery_public) == len(discovery_truth) == 1344
            and len(confirmation_public) == len(confirmation_truth) == len(predictions) == 1344
        ),
        "id_alignment": public_ids == truth_ids == manifest_ids and prediction_ids == {row["system_id"] for row in confirmation_public},
        "public_self_digests": public_digests,
        "truth_self_digests": truth_digests,
        "prediction_self_digests": prediction_digests,
        "camera_self_digest": camera["camera_digest"] == digest(strip_digest(camera, "camera_digest")),
        "marker_self_digest": marker["marker_digest"] == digest(strip_digest(marker, "marker_digest")),
        "final_self_digest": final["final_digest"] == digest(strip_digest(final, "final_digest")),
        "overall_self_digest": overall["summary_digest"] == digest(strip_digest(overall, "summary_digest")),
        "split_summary_digests": (
            discovery_summary["summary_digest"] == digest(strip_digest(discovery_summary, "summary_digest"))
            and confirmation_summary["summary_digest"] == digest(strip_digest(confirmation_summary, "summary_digest"))
        ),
        "run_commitments": (
            discovery_summary["public_digest"] == digest(discovery_public)
            and discovery_summary["truth_digest"] == digest(discovery_truth)
            and confirmation_summary["public_digest"] == digest(confirmation_public)
            and confirmation_summary["truth_digest"] == digest(confirmation_truth)
        ),
        "truth_storage_contract": (
            discovery_summary["truth_storage"] == "written"
            and confirmation_summary["truth_storage"] == "commitment_only_until_reveal"
        ),
        "independent_public_response_discovery": d_response_error == 0.0,
        "independent_public_response_confirmation": c_response_error == 0.0,
        "discovery_metrics": metrics_match(camera["discovery_metrics"], d_metrics),
        "discovery_physical": metrics_match(camera["discovery_physical_metrics"], d_physical),
        "confirmation_metrics": metrics_match(final["result"]["confirmation_metrics"], c_metrics_extended),
        "confirmation_physical": metrics_match(final["result"]["confirmation_physical_metrics"], c_physical),
        "discovery_qualified": bool(camera["qualified"]),
        "confirmation_predictions_exact": all(prediction_exact),
        "structure_exact": d_metrics["structure_accuracy"] == c_metrics["structure_accuracy"] == 1.0,
        "near_distinguishable": (
            d_metrics["near_distinguishable_accuracy"] == c_metrics["near_distinguishable_accuracy"] == 1.0
            and near_confirmation_separation > EQUIVALENCE_TOLERANCE
        ),
        "minimum_alliances_exact": d_metrics["minimum_alliance_accuracy"] == c_metrics["minimum_alliance_accuracy"] == 1.0,
        "gauge_physical_difference": (
            d_physical["physical_state_dict_difference_fraction"] == 1.0
            and c_physical["physical_state_dict_difference_fraction"] == 1.0
            and d_physical["physical_hidden_difference_fraction"] == 1.0
            and c_physical["physical_hidden_difference_fraction"] == 1.0
        ),
        "gauge_public_exact_equivalence": d_physical["gauge_public_profile_max_abs"] == c_physical["gauge_public_profile_max_abs"] == 0.0,
        "gauge_sealed_exact_equivalence": d_physical["gauge_sealed_profile_max_abs"] == c_physical["gauge_sealed_profile_max_abs"] == 0.0,
        "hidden_public_exact_equivalence": d_physical["hidden_public_profile_max_abs"] == c_physical["hidden_public_profile_max_abs"] == 0.0,
        "hidden_sealed_separation": d_physical["hidden_sealed_difference_min"] >= 0.20 and c_physical["hidden_sealed_difference_min"] >= 0.20,
        "public_abstention": (
            all(row["gauge_variant_decision"] == "ABSTAIN" for row in predictions)
            and all(row["hidden_variant_decision_public"] == "ABSTAIN" for row in predictions)
        ),
        "sealed_hidden_exact": all(sealed_hidden),
        "sealed_gauge_abstention": all(sealed_gauge),
        "metadata_null": d_metrics["metadata_null_accuracy"] == c_metrics["metadata_null_accuracy"] == 0.5,
        "leak_positive_control": d_metrics["leaky_sentinel_accuracy"] == c_metrics["leaky_sentinel_accuracy"] == 1.0,
        "prediction_commitment": (
            marker["prediction_digest"] == digest(predictions)
            and marker["public_digest"] == digest(confirmation_public)
            and marker["count"] == len(predictions)
        ),
        "prediction_before_reveal": (
            marker_time < final_time
            and PREDICTION_PATH.stat().st_mtime_ns <= CONFIRMATION_SEALED.stat().st_mtime_ns <= FINAL_PATH.stat().st_mtime_ns
        ),
        "formal_gate": final["result"]["camera_gate"] is True and final["status"] == "automorphism_quotient_camera_passed",
        "k205_scope": (
            final["k_item"]["identifier"] == "K205"
            and final["k_item"]["evidence_grade"] == "E3-KT"
            and "no pretrained-language claim" in final["k_item"]["scope"]
        ),
        "math_boundary": (
            final["mathematics"]["new_mathematics_required"] is False
            and "not a globally transitive equivalence relation" in final["mathematics"]["empirical_rule"]
        ),
        "auto_continue_zero": final["authorization"]["auto_continue"] == 0,
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_result_audit",
        "created_at_utc": datetime.now().astimezone().isoformat(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "independent_metrics": {
            "discovery": d_metrics,
            "confirmation": c_metrics_extended,
            "discovery_physical": d_physical,
            "confirmation_physical": c_physical,
            "max_public_response_error": max(d_response_error, c_response_error),
            "near_confirmation_min_nonidentity_error": float(near_confirmation_separation),
        },
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    result["audit_digest"] = digest(result)
    write_json(RESULT_AUDIT_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result"))
    args = parser.parse_args()
    result = preaudit() if args.stage == "preaudit" else result_audit()
    print(canonical_json({
        "stage": args.stage,
        "all_checks_passed": result["all_checks_passed"],
        "passed": result["passed_count"],
        "total": result["check_count"],
        "audit_digest": result["audit_digest"],
    }))


if __name__ == "__main__":
    main()
