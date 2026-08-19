#!/usr/bin/env python3
"""Independent Phase1326 pre/post audit; never imports the field executor."""
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
OUT = T / "result/phase1326_c039_composition_field"
PARENT = T / "result/phase1325_c039_qwen3_behavior"
CONTRACT = T / "result/phase1324_c039_exact_truth_scope_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_field_manifest.jsonl"
PROJ = OUT / "protocol/fixed_signed_projection.npz"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/full_state_composition_field.npz"
META = OUT / "raw/field_metadata.json"
S = OUT / "analysis/field_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1326_c039_composition_field.py"
SELF = Path(__file__).resolve()
MATERIAL = CONTRACT / "material/frozen_truth_scope_pairs.jsonl"
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("prefix_scope", "reported_statement")
PANELS = ("active_single", "active_outer_context_true", "active_outer_context_false",
          "active_inner_context_true", "active_inner_context_false", "wrong_scope", "lexical_null", "self_repeat")
ACTIVE, NESTED = PANELS[:5], PANELS[1:5]
ROLES = ("proposition_entity", "proposition_property", "active_operator", "context_operator",
         "query_entity", "query_property", "query_end", "assistant_boundary")
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
    l, r = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    denominator = np.linalg.norm(l) * np.linalg.norm(r)
    return float(np.dot(l, r) / denominator) if denominator > EPS else 0.0


def recompute(role: np.ndarray, norms: np.ndarray, masks: np.ndarray, answers: np.ndarray, margins: np.ndarray,
              metadata: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    index = {(m["partition"], m["profile_index"], m["property"], m["surface"], m["panel"]): i
             for i, m in enumerate(metadata)}
    active = [i for i, m in enumerate(metadata) if m["panel"] in ACTIVE]
    repeated = [i for i, m in enumerate(metadata) if m["panel"] == "self_repeat"]
    gates = {"finite": bool(np.isfinite(role).all() and np.isfinite(norms).all() and np.isfinite(margins).all()),
             "behavior_replay": float(np.mean(answers)) >= thresholds["behavior_replay_accuracy_min"],
             "active_nonzero": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS))
                               >= thresholds["active_nonzero_fraction_min"],
             "self_repeat": float(np.max(np.sum(norms[repeated] ** 2, axis=(1, 2)))) <= thresholds["self_repeat_energy_max"]}
    discovery = {"outer": {0: [], 1: []}, "inner": {0: [], 1: []}}
    for i, item in enumerate(metadata):
        if item["partition"] == "discovery" and item["panel"] in NESTED:
            discovery[item["active_role"]][int(item["parity"])].append(role[i, 1:].ravel())
    prototypes = {kind: {parity: np.mean(values, axis=0) for parity, values in groups.items()}
                  for kind, groups in discovery.items()}
    cells: dict[str, dict[str, float]] = {}
    for partition in PARTITIONS:
        emb: list[float] = []
        own_values: list[float] = []
        wins: list[bool] = []
        sign: list[bool] = []
        active_abs: list[float] = []
        wrong_abs: list[float] = []
        lexical_abs: list[float] = []
        parity_correct: list[bool] = []
        parity_gaps: list[float] = []
        properties = sorted({m["property"] for m in metadata if m["partition"] == partition})
        for profile in range(4):
            for prop in properties:
                fields: dict[str, np.ndarray] = {}
                for surface in SURFACES:
                    values = np.stack([role[index[(partition, profile, prop, surface, panel)]] for panel in ACTIVE])
                    fields[surface] = values - values.mean(axis=0, keepdims=True)
                for panel_index, panel in enumerate(ACTIVE):
                    left, right = fields[SURFACES[0]][panel_index], fields[SURFACES[1]][panel_index]
                    own = cosine(left, right)
                    wrong = [cosine(left, fields[SURFACES[1]][j]) for j in range(len(ACTIVE)) if j != panel_index]
                    own_values.append(own)
                    wins.append(own > max(wrong))
                    li, ri = index[(partition, profile, prop, SURFACES[0], panel)], index[(partition, profile, prop, SURFACES[1], panel)]
                    emb.append(cosine(role[li, 0, 2], role[ri, 0, 2]))
        for i, item in enumerate(metadata):
            if item["partition"] != partition:
                continue
            delta = float(margins[i, 1] - margins[i, 0])
            if item["panel"] in ACTIVE:
                expected = 1.0 if int(item["parity"] or 0) == 1 else -1.0
                sign.append(delta * expected > 0)
                active_abs.append(abs(delta))
            elif item["panel"] == "wrong_scope":
                wrong_abs.append(abs(delta))
            elif item["panel"] == "lexical_null":
                lexical_abs.append(abs(delta))
            if partition in {"confirmation", "holdout"} and item["panel"] in NESTED:
                target_kind = "outer" if item["active_role"] == "inner" else "inner"
                vector = role[i, 1:].ravel()
                scores = [cosine(vector, prototypes[target_kind][parity]) for parity in (0, 1)]
                truth = int(item["parity"])
                parity_correct.append(int(np.argmax(scores)) == truth)
                parity_gaps.append(scores[truth] - scores[1 - truth])
        cell = {"surface_operator_embedding_cosine_median": float(np.median(emb)),
                "cross_surface_panel_cosine_median": float(np.median(own_values)),
                "cross_surface_panel_own_win_fraction": float(np.mean(wins)),
                "active_margin_sign_accuracy": float(np.mean(sign)),
                "active_abs_margin_delta_median": float(np.median(active_abs)),
                "wrong_scope_abs_margin_delta_median": float(np.median(wrong_abs)),
                "lexical_null_abs_margin_delta_median": float(np.median(lexical_abs)),
                "cross_role_parity_accuracy": float(np.mean(parity_correct)) if parity_correct else 0.0,
                "cross_role_parity_gap_median": float(np.median(parity_gaps)) if parity_gaps else 0.0}
        cells[partition] = cell
        if partition in {"confirmation", "holdout"}:
            gates[f"{partition}_operator_embedding"] = cell["surface_operator_embedding_cosine_median"] >= thresholds["surface_operator_embedding_cosine_median_min"]
            gates[f"{partition}_panel_cosine"] = cell["cross_surface_panel_cosine_median"] >= thresholds["cross_surface_panel_cosine_median_min"]
            gates[f"{partition}_panel_win"] = cell["cross_surface_panel_own_win_fraction"] >= thresholds["cross_surface_panel_own_win_fraction_min"]
            gates[f"{partition}_margin_sign"] = cell["active_margin_sign_accuracy"] >= thresholds["active_margin_sign_accuracy_min"]
            gates[f"{partition}_margin_size"] = cell["active_abs_margin_delta_median"] >= thresholds["active_abs_margin_delta_median_min"]
            gates[f"{partition}_wrong_scope"] = cell["wrong_scope_abs_margin_delta_median"] <= thresholds["wrong_scope_abs_margin_delta_median_max"]
            gates[f"{partition}_lexical_null"] = cell["lexical_null_abs_margin_delta_median"] <= thresholds["lexical_null_abs_margin_delta_median_max"]
            gates[f"{partition}_parity_accuracy"] = cell["cross_role_parity_accuracy"] >= thresholds["cross_role_parity_accuracy_min"]
            gates[f"{partition}_parity_gap"] = cell["cross_role_parity_gap_median"] >= thresholds["cross_role_parity_gap_median_min"]
    metrics = {"finite_fraction": float(np.mean(np.isfinite(role))), "behavior_replay_accuracy": float(np.mean(answers)),
               "active_nonzero_fraction": float(np.mean(np.sum(norms[active] ** 2, axis=(1, 2)) > EPS)),
               "active_total_energy_median": float(np.median(np.sum(norms[active] ** 2, axis=(1, 2)))),
               "self_repeat_total_energy_max": float(np.max(np.sum(norms[repeated] ** 2, axis=(1, 2)))),
               "role_presence_fraction": float(np.mean(masks))}
    return {"metrics": metrics, "partitions": cells, "gates": gates, "all_gates_passed": all(gates.values())}


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SELF)}, protocol["source_hashes"])
    parent_ok = load(PARENT / "analysis/final.json").get("authorization") == "phase1326_c039_composition_field_only"
    parent_ok &= load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed", False)
    add(checks, "parent_authorized", parent_ok, "Phase1325")
    expected = {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                "parent_final": sha(PARENT / "analysis/final.json"), "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                "contract_protocol": sha(CONTRACT / "protocol/preregistration.json"),
                "material": sha(MATERIAL), "manifest": sha(M), "projection": sha(PROJ)}
    add(checks, "dependencies", protocol["dependencies"] == expected, protocol["dependencies"])
    manifest = rows(M)
    add(checks, "manifest_count_hash", len(manifest) == 1152 and protocol["manifest"]["sha256"] == sha(M), len(manifest))
    add(checks, "manifest_balance", Counter(x["panel"] for x in manifest) == Counter({panel: 144 for panel in PANELS}),
        dict(Counter(x["panel"] for x in manifest)))
    roles_ok = all(state["true_boundary"] == len(state["ids"]) - 1
        and state["positions"]["assistant_boundary"] == [state["true_boundary"]]
        and state["positions"]["query_entity"] and state["positions"]["query_property"] and state["positions"]["query_end"]
        for pair in manifest for state in pair["states"])
    add(checks, "role_phi_boundary", roles_ok, "compiled spans and assistant boundary")
    signs = np.load(PROJ)["signs"]
    add(checks, "projection_frozen", signs.shape == (2560, 64) and set(np.unique(signs)) == {-1, 1}
        and protocol["projection"]["sha256"] == sha(PROJ), signs.shape)
    add(checks, "capture_scope", protocol["capture"]["all_positions"] is True
        and protocol["capture"]["layers_including_embedding"] == 37 and protocol["capture"]["roles"] == list(ROLES)
        and protocol["capture"]["exact_full_residual_depth"] == 15, protocol["capture"])
    add(checks, "analysis_frozen", protocol["analysis"]["gate_application"].startswith("global")
        and "no fitted alignment" in protocol["analysis"]["parity_transfer"], protocol["analysis"])
    add(checks, "hard_stops", protocol["formal_run_budget"] == 1
        and "No attention/MLP/head/probe read" in protocol["hard_stops"], protocol["hard_stops"])
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(item["passed"] for item in checks)
    value = {"phase": 1326, "campaign": "C039", "audit_stage": stage,
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
    write(PRE, checks, "pre_model", "run_phase1326_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    arrays, meta, summary = np.load(A), load(META), load(S)
    shapes = {"all_position_sketch": (1152, 37, meta["max_length"], 64),
              "all_position_norm": (1152, 37, meta["max_length"]), "role_sketch": (1152, 37, 8, 64),
              "role_mask": (1152, 2, 8), "exact_layer15_role_delta": (1152, 8, 2560),
              "behavior_correct": (1152, 2), "yes_no_margin": (1152, 2), "lengths": (1152,)}
    add(checks, "array_shapes", all(arrays[key].shape == shape for key, shape in shapes.items()),
        {key: arrays[key].shape for key in arrays.files})
    add(checks, "hash_chain", meta["arrays_sha256"] == sha(A) and summary["arrays_sha256"] == sha(A)
        and meta["manifest_sha256"] == sha(M) and meta["projection_sha256"] == sha(PROJ), meta["arrays_sha256"])
    result = recompute(arrays["role_sketch"].astype(np.float32), arrays["all_position_norm"], arrays["role_mask"],
                       arrays["behavior_correct"], arrays["yes_no_margin"], meta["metadata"], protocol["thresholds"])
    add(checks, "independent_metrics", result["metrics"] == summary["metrics"]
        and result["partitions"] == summary["partitions"], result)
    add(checks, "independent_gates", result["gates"] == summary["gates"]
        and result["all_gates_passed"] == summary["all_gates_passed"], result["gates"])
    authorization = "phase1327_c039_composition_causal_only" if result["all_gates_passed"] \
        else "close_c039_at_descriptive_composition_boundary"
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
