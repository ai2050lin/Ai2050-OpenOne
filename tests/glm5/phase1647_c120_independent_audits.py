#!/usr/bin/env python3
"""Independent recomputation audits for C120."""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1647_c120_controlled_comparison_observation_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1647_c120_controlled_comparison_common as c120


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def support(values: np.ndarray, k: int = 256) -> set[int]:
    return {int(value) for value in np.argpartition(np.abs(values), -k)[-k:]}


def save(name: str, phase: int, checks: dict, authorization: str) -> None:
    report = {
        "phase": phase,
        "campaign": "C120",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": authorization,
    }
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / f"audit/{name}.json", report)
    print(json.dumps(report, indent=2))


def contract() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    internal = core.load(OUT / "audit/internal_contract_audit.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    cells = Counter(
        (row["partition"], row["dimension"], row["truth_factor"], row["gap_factor"], row["surface_factor"], row["output_format"])
        for row in cases
    )
    checks = {
        "internal": internal["all_checks_passed"],
        "producer": protocol["producer_sha256"] == core.sha(TESTS / "phase1647_c120_controlled_comparison_common.py"),
        "digest": protocol["material_digest"] == core.digest([*units, *cases]),
        "counts": (len(units), len(cases), len(compiled), len(manifest)) == (24, 1152, 1152, protocol["occurrences"]),
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in c120.PARTITIONS},
        "factorial": cells == {
            (partition, dimension, truth, gap, surface, output_format): 8
            for partition in c120.PARTITIONS
            for dimension, truth, gap, surface, output_format in itertools.product(
                c120.DIMENSIONS, (1, -1), (1, -1), (1, -1), (1, -1)
            )
        },
        "truth_balance": all(
            sum(row["truth_factor"] for row in cases if row["partition"] == partition and row["dimension"] == dimension) == 0
            for partition in c120.PARTITIONS for dimension in c120.DIMENSIONS
        ),
        "scores": all(
            row["truth_factor"] == (1 if row["scores"]["A"][row["dimension"]] > row["scores"]["B"][row["dimension"]] else -1)
            for row in cases
        ),
        "roles": all(set(row["role_positions"]) == set(c120.ROLES) for row in compiled),
        "candidates": all(len(candidate) == 1 for row in compiled for candidate in row["candidate_ids"]),
        "zero_models": all(value == 0.5 for key, value in protocol["zero_models"].items() if key != "score_comparison_oracle"),
        "boundary": all(term in protocol["claim_boundary"] for term in ("attention/MLP", "orthogonal", "topology", "new-mathematics")),
        "authorization": protocol["authorization"] == "execute_phase1648_c120_cuda_capture",
    }
    save("independent_contract_audit", 1647, checks, protocol["authorization"])


def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/capture_summary.json")
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    logits = np.load(OUT / "raw/qwen3_candidate_logits.float32.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    accuracy = sum(row["correct"] for row in index) / len(index)
    by_dimension = {
        name: sum(row["correct"] for row in index if row["dimension"] == name) / 384
        for name in c120.DIMENSIONS
    }
    checks = {
        "contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "shape": list(raw.shape) == protocol["archive"]["shape"] and raw.dtype == np.uint16,
        "hash": core.sha(OUT / protocol["archive"]["path"]) == report["raw_sha256"],
        "logits": list(logits.shape) == [1152, 2] and bool(np.isfinite(logits).all()) and core.sha(OUT / "raw/qwen3_candidate_logits.float32.npy") == report["logits_sha256"],
        "index": len(index) == 1152 and core.sha(OUT / "raw/qwen3_behavior_index.jsonl") == report["index_sha256"],
        "behavior": abs(accuracy - report["behavior"]["overall"]) < 1e-12 and by_dimension == report["behavior"]["by_dimension"],
        "gate": report["behavior_gate_passed"] == all(report["behavior_gate_checks"].values()),
        "repeat": all(value == 0 for value in report["numeric"].values()),
        "bf16": report["runtime"]["quantization"]["has_bf16_parameters"] and not report["runtime"]["quantization"]["has_quantized_modules"],
        "authorization": report["authorization"] == ("execute_phase1649_c120_discovery_observation" if report["behavior_gate_passed"] else "seal_hidden_state_analysis_and_close_C120_behavior_boundary"),
    }
    save("independent_capture_audit", 1648, checks, report["authorization"])


def discover() -> None:
    report = core.load(OUT / "analysis/discovery_freeze.json")
    frozen = core.load(OUT / "protocol/frozen_dimension_nominations.json")
    fields = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    candidates = core.rows(OUT / "analysis/discovery_candidate_table.jsonl")
    winner_checks = []
    for nomination in frozen["nominations"]:
        local = [row for row in candidates if row["dimension"] == nomination["dimension"] and row["score"] is not None]
        winner = sorted(local, key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], c120.ROLES.index(row["role"])))[0]
        e = c120.EFFECTS.index(nomination["effect"])
        r = c120.ROLES.index(nomination["role"])
        state = int(nomination["state"])
        mean = np.mean(fields[:, e, r, state], axis=0, dtype=np.float32)
        winner_checks.append(
            winner["role"] == nomination["role"]
            and winner["state"] == nomination["state"]
            and support(mean) == set(nomination["support"])
        )
    checks = {
        "capture": core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"],
        "shape": list(fields.shape) == [8, 7, 10, 37, 2560],
        "hash": core.sha(OUT / "analysis/discovery_unit_effect_role_state.float32.npy") == report["field_sha256"] == frozen["field_sha256"],
        "candidates": len(candidates) == 930,
        "nominations": len(frozen["nominations"]) == 3 and all(winner_checks),
        "freeze_hash": core.sha(OUT / "protocol/frozen_dimension_nominations.json") == report["nomination_sha256"],
        "authorization": report["authorization"] == "execute_phase1650_c120_confirmation_lockbox_validation",
    }
    save("independent_discovery_audit", 1649, checks, report["authorization"])


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/validation_adjudication.json")
    nominations = core.load(OUT / "protocol/frozen_dimension_nominations.json")["nominations"]
    discovery = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    fields = np.load(OUT / "analysis/validation_unit_effect_role_state.float32.npy", mmap_mode="r")
    timing = core.rows(OUT / "analysis/three_dimension_role_state_timing_atlas.jsonl")
    metrics_ok = []
    for nomination, registered in zip(nominations, report["dimension_metrics"], strict=True):
        e = c120.EFFECTS.index(nomination["effect"])
        r = c120.ROLES.index(nomination["role"])
        state = int(nomination["state"])
        d = np.mean(discovery[:, e, r, state], axis=0, dtype=np.float32)
        c = np.mean(fields[:8, e, r, state], axis=0, dtype=np.float32)
        l = np.mean(fields[8:, e, r, state], axis=0, dtype=np.float32)
        expected = {
            "confirmation_lockbox_cosine": cosine(c, l),
            "confirmation_to_discovery_cosine": cosine(c, d),
            "lockbox_to_discovery_cosine": cosine(l, d),
            "confirmation_support_overlap": len(support(c) & set(nomination["support"])) / 256,
            "lockbox_support_overlap": len(support(l) & set(nomination["support"])) / 256,
        }
        metrics_ok.append(all(abs(registered[key] - value) < 1e-7 for key, value in expected.items()))
    checks = {
        "discovery": core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"],
        "shape": list(fields.shape) == [16, 7, 10, 37, 2560],
        "hash": core.sha(OUT / "analysis/validation_unit_effect_role_state.float32.npy") == report["field_sha256"],
        "metrics": len(metrics_ok) == 3 and all(metrics_ok),
        "passed_count": report["passed_dimensions"] == sum(row["passed"] for row in report["dimension_metrics"]),
        "timing": len(timing) == 1110 and core.sha(OUT / "analysis/three_dimension_role_state_timing_atlas.jsonl") == report["timing_sha256"],
        "finite": all(math.isfinite(row["confirmation_lockbox_cosine"]) for row in timing),
        "authorization": report["authorization"] == "execute_phase1651_c120_synthesis_heatmap_and_closure",
    }
    save("independent_validation_audit", 1650, checks, report["authorization"])


def closure() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    internal = core.load(OUT / "audit/internal_closure_audit.json")
    payload = core.load(PUBLIC)
    effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C120"]
    raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C120"]
    support_rows = [row for row in payload["support_rows"] if row.get("dataset") == "C120"]
    checks = {
        "internal": internal["all_checks_passed"],
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == internal["asset_sha256"],
        "effects": len(effects) == closure["heatmap"]["c120_effect_rows"] and len(effects) > 900,
        "raw": len(raw) == 240 and len(raw) == closure["heatmap"]["c120_raw_rows"],
        "coordinates": all(len(row["values"]) == 2560 for row in [*effects, *raw, *support_rows]),
        "embedding": any(row["state"] == 0 for row in effects) and any(row["state"] == 0 for row in raw),
        "supports": len(support_rows) == 3 and all(row["k"] == 256 for row in support_rows),
        "batch": payload["campaign"] == "C109-C120" and "c120_batch" in payload,
        "boundary": all(term in closure["claim_boundary"] for term in ("does not identify weights", "attention/MLP", "orthogonal", "new mathematics")),
        "authorization": closure["next_authorization"].startswith("C121"),
    }
    save("independent_closure_audit", 1651, checks, "append_C119R_C120_memo_and_verify_client")


STAGES = {
    "contract": contract,
    "capture": capture,
    "discover": discover,
    "validate": validate,
    "closure": closure,
}


if __name__ == "__main__":
    STAGES[sys.argv[1]]()
