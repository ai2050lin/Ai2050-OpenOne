#!/usr/bin/env python3
"""C193: decompose C192 failure into program scaffold and family residual."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1727_c193_program_scaffold_failure_decomposition"
C192 = RESULT / "phase1726_c192_multi_program_response_equivalence"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c193_program_centered_response_residual.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1726_c192_multi_program_response_equivalence as c192

PHASE, CAMPAIGN = 1727, "C193"


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C192 / "audit/independent_final_audit.json"); final = core.load(C192 / "analysis/final.json")
    checks = {"authorization": parent["all_checks_passed"] and "C193_failure_decomposition" in parent["authorization"], "c192_failed": not final["headline"]["constrained_nearest_neighbor"]["passed"], "complete": final["headline"]["observed_cells"] == 112 and not final["headline"]["registered_missing"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "program_scaffold_failure_decomposition_frozen",
        "data": "all 112 C192 response profiles; no new model run",
        "analyses": ["unconstrained nearest-label atlas", "factor-specific exclusion atlas", "leave-one-family-out program centering", "strict cross-program/unit/phrase family nearest neighbor after centering"],
        "centering": "for each cell subtract the mean profile of the same program among other relation families, then L1-normalize the signed residual",
        "residual_gate": {"same_family_rate_min": 0.50, "advantage_over_available_peer_baseline_min": 0.25, "support_min": 100},
        "claim_boundary": "exploratory failure decomposition after C192 rejection; cannot rescue or rewrite C192 primary prediction",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "imputation", "threshold search", "claiming residual confirmation"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "build_failure_decomposition_and_residual_heatmap",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks}, indent=2))


def profiles_and_index():
    raw = np.load(C192 / "raw/multi_program_relation_response.float16.npy", mmap_mode="r"); index = core.rows(C192 / "raw/response_index.jsonl")
    profiles = np.stack([c192.profile(np.asarray(raw[row["anchor_index"]], dtype=np.float32)) for row in index])
    return profiles, index


def sim(left, right):
    return float(1.0 - 0.5 * np.abs(left.astype(np.float64) - right.astype(np.float64)).sum())


def nearest_summary(values, index, candidate_filter):
    rows = []
    for i, row in enumerate(index):
        candidates = [j for j, other in enumerate(index) if i != j and candidate_filter(row, other)]
        if not candidates:
            continue
        scores = [sim(values[i], values[j]) for j in candidates]; neighbor = candidates[int(np.argmax(scores))]
        baseline = sum(index[j]["family"] == row["family"] for j in candidates) / len(candidates)
        rows.append({"cell_index": i, "neighbor_index": neighbor, "similarity": float(max(scores)), "same_family": index[neighbor]["family"] == row["family"], "same_program": index[neighbor]["program"] == row["program"], "same_unit": index[neighbor]["unit"] == row["unit"], "same_phrase_variant": index[neighbor]["phrase_variant"] == row["phrase_variant"], "family_baseline": baseline, "candidate_count": len(candidates)})
    return {"support": len(rows), "same_family_rate": float(np.mean([row["same_family"] for row in rows])), "same_program_rate": float(np.mean([row["same_program"] for row in rows])), "same_unit_rate": float(np.mean([row["same_unit"] for row in rows])), "same_phrase_rate": float(np.mean([row["same_phrase_variant"] for row in rows])), "family_baseline": float(np.mean([row["family_baseline"] for row in rows])), "family_advantage": float(np.mean([row["same_family"] for row in rows])) - float(np.mean([row["family_baseline"] for row in rows])), "rows": rows}


def build():
    profiles, index = profiles_and_index()
    raw_atlases = {
        "unconstrained": nearest_summary(profiles, index, lambda _row, _other: True),
        "cross_program": nearest_summary(profiles, index, lambda row, other: row["program"] != other["program"]),
        "cross_unit": nearest_summary(profiles, index, lambda row, other: row["unit"] != other["unit"]),
        "cross_phrase": nearest_summary(profiles, index, lambda row, other: row["phrase_variant"] != other["phrase_variant"]),
        "within_program_cross_unit_phrase": nearest_summary(profiles, index, lambda row, other: row["program"] == other["program"] and row["unit"] != other["unit"] and row["phrase_variant"] != other["phrase_variant"]),
        "strict_cross_all": nearest_summary(profiles, index, lambda row, other: row["program"] != other["program"] and row["unit"] != other["unit"] and row["phrase_variant"] != other["phrase_variant"]),
    }
    residuals = np.empty_like(profiles)
    for i, row in enumerate(index):
        reference = [j for j, other in enumerate(index) if other["program"] == row["program"] and other["family"] != row["family"]]
        centered = profiles[i].astype(np.float64) - profiles[reference].astype(np.float64).mean(axis=0)
        norm = np.abs(centered).sum()
        residuals[i] = (centered / max(norm, 1e-30)).astype(np.float32)
    residual_strict = nearest_summary(residuals, index, lambda row, other: row["program"] != other["program"] and row["unit"] != other["unit"] and row["phrase_variant"] != other["phrase_variant"])
    gate = core.load(OUT / "protocol/preregistration.json")["residual_gate"]
    residual_passed = residual_strict["support"] >= gate["support_min"] and residual_strict["same_family_rate"] >= gate["same_family_rate_min"] and residual_strict["family_advantage"] >= gate["advantage_over_available_peer_baseline_min"]
    summary = {name: {key: value for key, value in atlas.items() if key != "rows"} for name, atlas in raw_atlases.items()}
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "program_scaffold_failure_decomposed", "raw_nearest_atlases": summary, "program_centered_strict": {**{key: value for key, value in residual_strict.items() if key != "rows"}, "passed_exploratory_gate": residual_passed}, "interpretation": "C192 remains failed. Improvement after family-excluded program centering would locate a smaller family-conditioned residual beneath a dominant program scaffold, but is exploratory until prospectively replicated.", "next_authorization": "C194_prospective_program_centered_residual_replication_if_residual_gate_passes_else_C194_change_measurement_object"}
    core.save(OUT / "analysis/failure_decomposition.json", report); core.write_rows(OUT / "analysis/residual_strict_nearest.jsonl", residual_strict["rows"])
    variation = np.var(residuals, axis=0)
    payload = {"schema": "c193_program_centered_response_residual.v1", "result_type": "program_centered_response_residual_heatmap", "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B", "title": "C193 Family-Excluded Program-Centered Response Residual", "dimensions": list(range(2560)), "default_coordinates": np.argsort(-variation)[:64].astype(int).tolist(), "rows": [{**row, "label": f"{row['family']} / {row['program']} / unit{row['unit']} / phrase{row['phrase_variant']}", "values": residuals[i].tolist()} for i, row in enumerate(index)], "raw_summary": summary, "residual_result": report["program_centered_strict"], "coordinate_semantics": "Each row is a signed, L1-normalized residual after subtracting the same-program mean formed from other relation families; columns are q25 physical activation coordinates.", "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"]}
    PUBLIC.parent.mkdir(parents=True, exist_ok=True); PUBLIC.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    asset = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size, "rows": len(index), "schema": payload["schema"]}; core.save(OUT / "analysis/public_asset.json", asset)
    checks = {"profiles": profiles.shape == (112, 2560), "residuals": residuals.shape == (112, 2560), "six_atlases": len(summary) == 6, "strict_support": residual_strict["support"] == 112, "finite": bool(np.isfinite(residuals).all())}
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"raw": summary, "residual": report["program_centered_strict"], "asset": asset, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/failure_decomposition.json"); asset = core.load(OUT / "analysis/public_asset.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "build": core.load(OUT / "audit/internal_build_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"], "asset_hash": core.sha(PUBLIC) == asset["sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "asset": asset, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "build", "close")); args = parser.parse_args(); {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__":
    main()
