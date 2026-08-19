#!/usr/bin/env python3
"""Independent Phase1322 audit; deliberately does not import the executor."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1322_c037_isomorphic_full_state_field"
PARENT = T / "result/phase1321_c037_qwen3_behavior"
CONTRACT = T / "result/phase1320_c037_event_isomorphism_boundary_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_field_manifest.jsonl"
PROJ = OUT / "protocol/fixed_signed_projection.npz"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/full_state_field_arrays.npz"
META = OUT / "raw/field_metadata.json"
S = OUT / "analysis/field_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1322_c037_isomorphic_full_state_field.py"
SELF = Path(__file__).resolve()
MATERIAL = CONTRACT / "material/frozen_isomorphic_lookup_pairs.jsonl"
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("department", "region", "mode", "level", "channel", "status")
SURFACES = ("registry_narrative", "registry_table")
EPS = 1e-10


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    l = left.astype(np.float64).ravel()
    r = right.astype(np.float64).ravel()
    denominator = np.linalg.norm(l) * np.linalg.norm(r)
    return float(np.dot(l, r) / denominator) if denominator > EPS else 0.0


def gram(vectors: np.ndarray) -> np.ndarray:
    flat = vectors.reshape(vectors.shape[0], -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = flat / np.where(norms > EPS, norms, 1.0)
    return unit @ unit.T


def recompute(role: np.ndarray, norms: np.ndarray, answers: np.ndarray,
              metadata: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    index = {(m["partition"], m["profile_index"], m["surface"], m["attribute"], m["panel"]): i
             for i, m in enumerate(metadata)}
    active = [i for i, item in enumerate(metadata) if item["panel"] == "active"]
    null = [i for i, item in enumerate(metadata) if item["panel"] == "matched_null"]
    repeated = [i for i, item in enumerate(metadata) if item["panel"] == "self_repeat"]
    gates = {"finite": bool(np.isfinite(role).all() and np.isfinite(norms).all()),
             "behavior_replay": float(np.mean(answers)) >= thresholds["behavior_replay_accuracy_min"],
             "active_nonzero": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS))
                               >= thresholds["active_nonzero_fraction_min"]}
    cells: dict[str, dict[str, float]] = {}
    for partition in PARTITIONS:
        emb, own_values, gap_values, wins, grams, perms = [], [], [], [], [], []
        for profile in range(4):
            field: dict[str, np.ndarray] = {}
            for surface in SURFACES:
                values = np.stack([role[index[(partition, profile, surface, attr, "active")]] for attr in ATTRS])
                field[surface] = values - values.mean(axis=0, keepdims=True)
            for attr_index, attr in enumerate(ATTRS):
                left, right = field[SURFACES[0]][attr_index], field[SURFACES[1]][attr_index]
                own = cosine(left, right)
                wrong = [cosine(left, field[SURFACES[1]][j]) for j in range(6) if j != attr_index]
                own_values.append(own)
                gap_values.append(own - max(wrong))
                wins.append(own > max(wrong))
                li = index[(partition, profile, SURFACES[0], attr, "active")]
                ri = index[(partition, profile, SURFACES[1], attr, "active")]
                emb.append(cosine(role[li, 0, 7:10], role[ri, 0, 7:10]))
            for surface in SURFACES:
                centered = field[surface]
                eg, dg = gram(centered[:, 0, 7:10]), gram(centered[:, 1:])
                upper = np.triu_indices(6, 1)
                grams.append(cosine(eg[upper], dg[upper]))
                order = np.roll(np.arange(6), 1)
                pg = dg[order][:, order]
                perms.append(cosine(eg[upper], pg[upper]))
        cell = {"surface_embedding_cosine_median": float(np.median(emb)),
                "typed_cross_surface_cosine_median": float(np.median(own_values)),
                "typed_cross_surface_gap_median": float(np.median(gap_values)),
                "typed_cross_surface_own_win_fraction": float(np.mean(wins)),
                "embedding_downstream_gram_cosine_median": float(np.median(grams)),
                "embedding_downstream_permuted_cosine_median": float(np.median(perms)),
                "embedding_downstream_over_permuted_gap": float(np.median(np.asarray(grams) - np.asarray(perms)))}
        cells[partition] = cell
        if partition in {"confirmation", "holdout"}:
            gates[f"{partition}_embedding_surface"] = cell["surface_embedding_cosine_median"] >= thresholds["surface_embedding_cosine_median_min"]
            gates[f"{partition}_typed_cosine"] = cell["typed_cross_surface_cosine_median"] >= thresholds["typed_cross_surface_cosine_median_min"]
            gates[f"{partition}_typed_gap"] = cell["typed_cross_surface_gap_median"] >= thresholds["typed_cross_surface_gap_median_min"]
            gates[f"{partition}_typed_win"] = cell["typed_cross_surface_own_win_fraction"] >= thresholds["typed_cross_surface_own_win_fraction_min"]
            gates[f"{partition}_gram"] = cell["embedding_downstream_gram_cosine_median"] >= thresholds["embedding_downstream_gram_cosine_median_min"]
            gates[f"{partition}_gram_control"] = cell["embedding_downstream_over_permuted_gap"] >= thresholds["embedding_downstream_over_permuted_gap_min"]
    metrics = {"finite_fraction": float(np.mean(np.isfinite(role))),
               "behavior_replay_accuracy": float(np.mean(answers)),
               "active_nonzero_fraction": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS)),
               "active_total_energy_median": float(np.median(np.sum(norms[active] ** 2, axis=(1, 2)))),
               "matched_null_total_energy_median": float(np.median(np.sum(norms[null] ** 2, axis=(1, 2)))),
               "self_repeat_total_energy_max": float(np.max(np.sum(norms[repeated] ** 2, axis=(1, 2))))}
    return {"metrics": metrics, "partitions": cells, "gates": gates, "all_gates_passed": all(gates.values())}


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SELF)}, protocol["source_hashes"])
    parent_ok = load(PARENT / "analysis/final.json").get("authorization") == "phase1322_isomorphic_field_only"
    parent_ok &= load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed", False)
    add(checks, "parent_authorized", parent_ok, "Phase1321")
    expected = {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                "parent_final": sha(PARENT / "analysis/final.json"),
                "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                "contract_protocol": sha(CONTRACT / "protocol/preregistration.json"),
                "material": sha(MATERIAL), "manifest": sha(M), "projection": sha(PROJ)}
    add(checks, "dependencies", protocol["dependencies"] == expected, protocol["dependencies"])
    manifest = rows(M)
    add(checks, "manifest_count_hash", len(manifest) == 432 and protocol["manifest"]["sha256"] == sha(M), len(manifest))
    add(checks, "manifest_balance", Counter(x["panel"] for x in manifest)
        == Counter({"active": 144, "matched_null": 144, "self_repeat": 144}), dict(Counter(x["panel"] for x in manifest)))
    phi_ok = all(len(state["slot_positions"]) == 10 and state["slot_positions"][3] == state["true_boundary"]
                 and state["true_boundary"] == len(state["ids"]) - 1 for pair in manifest for state in pair["states"])
    add(checks, "identity_phi_and_boundary", phi_ok, "10 canonical slots and compiled final boundary")
    signs = np.load(PROJ)["signs"]
    add(checks, "projection_frozen", signs.shape == (2560, 64) and set(np.unique(signs)) == {-1, 1}
        and protocol["projection"]["sha256"] == sha(PROJ), signs.shape)
    capture = protocol["capture"]
    add(checks, "capture_scope", capture["all_positions"] is True and capture["layers_including_embedding"] == 37
        and capture["exact_full_residual_depth"] == 15 and capture["slot_order_source"].startswith("state.slot_positions"), capture)
    add(checks, "no_component_scan", "No component/head/attention/probe read" in protocol["hard_stops"]
        and protocol["formal_run_budget"] == 1, protocol["hard_stops"])
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(item["passed"] for item in checks)
    value = {"phase": 1322, "campaign": "C037", "audit_stage": stage,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
             "checks": checks, "passed_count": sum(item["passed"] for item in checks), "total_count": len(checks),
             "all_checks_passed": passed, "authorization": authorization if passed else "none",
             "protocol_digest": load(P)["protocol_digest"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": value["passed_count"], "total": value["total_count"],
                     "authorization": value["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    checks = base(load(P))
    add(checks, "formal_outputs_absent", not any(path.exists() for path in (A, META, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1322_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    arrays, meta, summary = np.load(A), load(META), load(S)
    shapes = {"all_position_sketch": (432, 37, meta["max_length"], 64),
              "all_position_norm": (432, 37, meta["max_length"]), "role_sketch": (432, 37, 10, 64),
              "exact_layer15_role_delta": (432, 10, 2560), "embedding_record_value_delta": (432, 3, 2560),
              "behavior_correct": (432, 2), "lengths": (432,)}
    add(checks, "array_shapes", all(arrays[key].shape == shape for key, shape in shapes.items()),
        {key: arrays[key].shape for key in arrays.files})
    add(checks, "hash_chain", meta["arrays_sha256"] == sha(A) and summary["arrays_sha256"] == sha(A)
        and meta["manifest_sha256"] == sha(M) and meta["projection_sha256"] == sha(PROJ),
        {"arrays": meta["arrays_sha256"], "manifest": meta["manifest_sha256"]})
    result = recompute(arrays["role_sketch"].astype(np.float32), arrays["all_position_norm"],
                       arrays["behavior_correct"], meta["metadata"], protocol["thresholds"])
    add(checks, "independent_metrics", result["metrics"] == summary["metrics"]
        and result["partitions"] == summary["partitions"], result)
    add(checks, "independent_gates", result["gates"] == summary["gates"]
        and result["all_gates_passed"] == summary["all_gates_passed"], result["gates"])
    authorization = "phase1323_shared_typed_causal_only" if result["all_gates_passed"] else "close_c037_at_isomorphic_field_boundary"
    final = load(F)
    add(checks, "verdict", final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"], final)
    qa = meta["model_audit"]
    add(checks, "fp16_cuda", qa["has_fp16_parameters"] and not qa["has_quantized_modules"]
        and meta["cuda_peak_allocated_bytes"] > 0, {"qa": qa, "peak": meta["cuda_peak_allocated_bytes"]})
    complete = load(C)
    add(checks, "budget", complete["formal_runs_consumed"] == 1
        and complete["protocol_digest"] == protocol["protocol_digest"], complete)
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
