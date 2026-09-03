#!/usr/bin/env python3
"""C142: coordinate Mobius atlas with explicit output-code separation."""
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1676_c142_mobius_output_code_separation"
C141 = RESULT / "phase1675_c141_multifamily_full_coordinate_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1675_c141_multifamily_full_coordinate_atlas as c141

PHASE, CAMPAIGN = 1676, "C142"
ARMS, ROLES, CHECKPOINTS, DIM = c141.ARMS, c141.ROLES, c127.CHECKPOINTS, 2560
SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
SUBSET_NAMES = ("f1", "f2", "f3", "f1xf2", "f1xf3", "f2xf3", "f1xf2xf3")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den <= 1e-12 else float(np.dot(a.ravel(), b.ravel()) / den)


def top_overlap(a: np.ndarray, b: np.ndarray, k: int = 256) -> float:
    left = set(np.argpartition(np.abs(a), -k)[-k:].tolist())
    right = set(np.argpartition(np.abs(b), -k)[-k:].tolist())
    return len(left & right) / k


def build_partition(partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = core.rows(C141 / "compiled/qwen3.jsonl")
    raw = np.load(C141 / "raw/qwen3_six_role_field.bf16.npy", mmap_mode="r")
    coeff = np.zeros((5, 4, 7, 6, 38, DIM), np.float32)
    code = np.zeros((5, 4, 6, 38, DIM), np.float32)
    surface = np.zeros_like(code)
    for i, row in enumerate(rows):
        if row["partition"] != partition:
            continue
        ai = ARMS.index(row["arm"])
        absolute_unit = int(row["unit_id"].rsplit("-", 1)[1])
        ui = absolute_unit if partition == "discovery" else absolute_unit - 4
        factors = np.asarray([row["factors"][f"f{j}"] for j in range(1, 4)], np.float32)
        signs = np.asarray([np.prod(factors[list(subset)]) for subset in SUBSETS], np.float32)
        value = c127.decode(raw[i])
        coeff[ai, ui] += signs[:, None, None, None] * value[None] / 32.0
        code[ai, ui] += float(row["codebook_factor"]) * value / 32.0
        surface[ai, ui] += float(row["surface_factor"]) * value / 32.0
        if (i + 1) % 160 == 0:
            print(f"[C142 {partition}] scanned {i + 1}/1280", flush=True)
    return coeff, code, surface


def discover() -> None:
    if (OUT / "protocol/frozen_nominees.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(C141 / "audit/independent_closure_audit.json")
    if not parent["all_checks_passed"] or parent["authorization"] != "start_C142":
        raise RuntimeError(parent)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(exist_ok=True)
    (OUT / "protocol").mkdir(exist_ok=True)
    (OUT / "audit").mkdir(exist_ok=True)
    discovery, code, surface = build_partition("discovery")
    np.save(OUT / "analysis/discovery_mobius.float32.npy", discovery)
    np.save(OUT / "analysis/discovery_code.float32.npy", code)
    np.save(OUT / "analysis/discovery_surface.float32.npy", surface)
    nominees = {}
    for ai, arm in enumerate(ARMS):
        nominees[arm] = {}
        for si, name in enumerate(SUBSET_NAMES):
            candidates = []
            for ri, role in enumerate(ROLES):
                for q in range(0, 29):
                    left = discovery[ai, :2, si, ri, q].mean(0)
                    right = discovery[ai, 2:, si, ri, q].mean(0)
                    stability = cosine(left, right)
                    norm = min(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
                    candidates.append((max(stability, 0.0) * norm, stability, norm, ri, q))
            score, stability, norm, ri, q = max(candidates)
            vector = discovery[ai, :, si, ri, q].mean(0)
            vector_path = OUT / f"protocol/{arm}_{name}.float32.npy"
            vector_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vector_path, vector)
            nominees[arm][name] = {
                "role": ROLES[ri],
                "role_index": ri,
                "checkpoint": CHECKPOINTS[q],
                "checkpoint_index": q,
                "split_half_cosine": stability,
                "split_half_min_norm": norm,
                "score": score,
                "support": sorted(np.argpartition(np.abs(vector), -256)[-256:].tolist()),
                "vector_sha256": core.sha(vector_path),
            }
        code_candidates = []
        for ri, role in enumerate(ROLES):
            for q in range(38):
                left = code[ai, :2, ri, q].mean(0)
                right = code[ai, 2:, ri, q].mean(0)
                stability = cosine(left, right)
                norm = min(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
                code_candidates.append((max(stability, 0.0) * norm, stability, norm, ri, q))
        score, stability, norm, ri, q = max(code_candidates)
        vector = code[ai, :, ri, q].mean(0)
        path = OUT / f"protocol/{arm}_output_code.float32.npy"
        np.save(path, vector)
        nominees[arm]["output_code"] = {"role": ROLES[ri], "role_index": ri, "checkpoint": CHECKPOINTS[q], "checkpoint_index": q, "split_half_cosine": stability, "split_half_min_norm": norm, "score": score, "support": sorted(np.argpartition(np.abs(vector), -256)[-256:].tolist()), "vector_sha256": core.sha(path)}
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "discovery_mobius_nominees_frozen",
        "definition": "M_S=2^(-|S|) sum product(epsilon_j) Z(epsilon), additionally averaged over two surfaces and two output codebooks",
        "subsets": list(SUBSET_NAMES),
        "nominees": nominees,
        "early_mid_search": "checkpoint indices 0..28 inclusive",
        "confirmation_gate": {"cosine_min": 0.70, "top256_overlap_min": 0.30},
        "claim_boundary": "researcher-defined factorial effects; stable coordinates are not semantic neurons or natural model modules",
        "source_paths": {"C141_role": str(C141 / "raw/qwen3_six_role_field.bf16.npy")},
        "source_hashes": {"C141_role": core.sha(C141 / "raw/qwen3_six_role_field.bf16.npy")},
        "producer_sha256": core.sha(Path(__file__)),
        "confirmation_unread": True,
        "authorization": "validate_C142_confirmation",
    }
    core.save(OUT / "protocol/frozen_nominees.json", protocol)
    checks = {"shape": list(discovery.shape) == [5, 4, 7, 6, 38, DIM], "code_shape": list(code.shape) == [5, 4, 6, 38, DIM], "finite": bool(np.isfinite(discovery).all() and np.isfinite(code).all()), "nominees": sum(len(v) for v in nominees.values()) == 40, "source": protocol["source_hashes"]["C141_role"] == core.load(C141 / "analysis/authoritative_run.json")["capture"]["role_sha256"]}
    core.save(OUT / "audit/internal_discovery_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": protocol["authorization"]})
    print(json.dumps({"checks": checks, "nominees": nominees}, indent=2))
    del discovery, code, surface
    gc.collect()


def validate() -> None:
    freeze = core.load(OUT / "protocol/frozen_nominees.json")
    if freeze["authorization"] != "validate_C142_confirmation":
        raise RuntimeError("unauthorized")
    confirmation, code, surface = build_partition("confirmation")
    np.save(OUT / "analysis/confirmation_mobius.float32.npy", confirmation)
    np.save(OUT / "analysis/confirmation_code.float32.npy", code)
    np.save(OUT / "analysis/confirmation_surface.float32.npy", surface)
    results, hierarchy, code_results = {}, {}, {}
    all_pass = 0
    for ai, arm in enumerate(ARMS):
        results[arm] = {}
        energy = []
        for si, name in enumerate(SUBSET_NAMES):
            nominee = freeze["nominees"][arm][name]
            disc = np.load(OUT / f"protocol/{arm}_{name}.float32.npy")
            conf = confirmation[ai, :, si, nominee["role_index"], nominee["checkpoint_index"]].mean(0)
            co = cosine(disc, conf)
            overlap = top_overlap(disc, conf)
            passed = co >= freeze["confirmation_gate"]["cosine_min"] and overlap >= freeze["confirmation_gate"]["top256_overlap_min"]
            all_pass += int(passed)
            results[arm][name] = {"cosine": co, "top256_overlap": overlap, "passed": passed, "role": nominee["role"], "checkpoint": nominee["checkpoint"]}
            energy.append(float(np.mean(np.square(confirmation[ai, :, si], dtype=np.float64))))
        total = sum(energy)
        hierarchy[arm] = {name: value / max(total, 1e-30) for name, value in zip(SUBSET_NAMES, energy)}
        nominee = freeze["nominees"][arm]["output_code"]
        disc = np.load(OUT / f"protocol/{arm}_output_code.float32.npy")
        conf = code[ai, :, nominee["role_index"], nominee["checkpoint_index"]].mean(0)
        norms = np.linalg.norm(code[ai].mean(0), axis=-1)
        late_fraction = float(norms[:, 32:].sum() / max(norms.sum(), 1e-12))
        code_results[arm] = {"cosine": cosine(disc, conf), "top256_overlap": top_overlap(disc, conf), "role": nominee["role"], "checkpoint": nominee["checkpoint"], "late_checkpoint_norm_fraction": late_fraction}
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "confirmation_adjudicated",
        "semantic_results": results,
        "passing_semantic_nominees": all_pass,
        "total_semantic_nominees": 35,
        "confirmation_energy_hierarchy": hierarchy,
        "output_code_results": code_results,
        "claim_boundary": "code-averaged coordinate effects and separate output-code field; no natural operator or causal use inferred",
        "authorization": "close_C142_start_C143",
    }
    core.save(OUT / "analysis/confirmation.json", report)
    checks = {"shape": list(confirmation.shape) == [5, 4, 7, 6, 38, DIM], "finite": bool(np.isfinite(confirmation).all() and np.isfinite(code).all()), "results": sum(len(v) for v in results.values()) == 35, "code": len(code_results) == 5}
    core.save(OUT / "audit/internal_confirmation_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "scientific_replication_count": all_pass, "authorization": report["authorization"]})
    print(json.dumps(report, indent=2))


def close() -> None:
    report = core.load(OUT / "analysis/confirmation.json")
    checks = {"discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"], "typed": report["total_semantic_nominees"] == 35}
    closure = {"phase": PHASE, "campaign": CAMPAIGN, "status": "mobius_atlas_closed", "headline": {"passing": report["passing_semantic_nominees"], "total": report["total_semantic_nominees"]}, "theory_update": "semantic-factor effects survive explicit output-code averaging to a degree measured per arm; output-code fields remain separately typed", "problems": ["controlled three-factor programs", "four independent units per partition", "Mobius effects are researcher contrasts", "readability is not transport or causation"], "next_authorization": "C143 transition model competition"}
    core.save(OUT / "analysis/closure.json", closure)
    core.save(OUT / "audit/internal_closure_audit.json", {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_final_then_C143"})
    print(json.dumps(closure, indent=2))


def main() -> None:
    modes = {"discover": discover, "validate": validate, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit("discover|validate|close")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
