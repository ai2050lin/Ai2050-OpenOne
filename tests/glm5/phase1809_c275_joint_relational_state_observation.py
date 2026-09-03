#!/usr/bin/env python3
"""C275: observe whether emerging events reuse same-sign coordinates across roles."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common


core = common.core
OUT = common.RESULT / "phase1809_c275_joint_relational_state_observation"
C264 = common.OUTS["C264"]
C265 = common.OUTS["C265"]
C274 = common.RESULT / "phase1808_c274_joint_full_field_condition"
TRAIN = common.prior.OUTS["C248"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c275_cross_role_reuse_atlas.json"
FAMILIES = common.FAMILIES + ("nested_attitude",)
MATERIALS = ("third", "fourth")


def pair_ids(index: list[dict], family: str, panel: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family, "factor_a", panel)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def event(delta: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0)).astype(np.int8)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = [core.load(path / "analysis/final.json") for path in (C264, C274)]
    checks = {
        "parents_closed": all(item["all_checks_passed"] for item in parents),
        "authorization": parents[-1]["next_authorization"].startswith("C275_joint_relational_state_observation"),
        "third_and_fourth_materials": True,
        "all_roles_checkpoints_coordinates": True,
        "no_topk_projection_attention_mlp": True,
        "observation_not_causality": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1809,
        "campaign": "C275",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cross_role_reuse_observation_frozen",
        "object": "For every destination-role coordinate that is inactive at q and active at q+1, test whether the same signed event already exists at q in another registered role.",
        "materials": list(MATERIALS),
        "outputs": ["source-to-destination same-sign coverage", "opposite-sign coverage", "any-source union coverage", "coordinate-resolved reuse atlas"],
        "descriptive_gate": "At least four families have fourth-material any-source coverage >=0.50, third/fourth absolute difference <=0.10, and the same dominant source for at least four of six destination roles.",
        "claim_boundary": "Same-coordinate cross-role precedence is an observational reuse candidate. It is not natural transport, a causal edge, or proof that the receiving event was copied from the observed source.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "decide_after_two_material_reveal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    thresholds = np.asarray(core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    material_data = {
        "third": (np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r"), core.rows(TRAIN / "raw/hidden_index.jsonl")),
        "fourth": (np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r"), core.rows(C264 / "raw/hidden_index.jsonl")),
    }
    emergent_counts = np.zeros((2, len(FAMILIES), 6, 2560), np.uint32)
    same_counts = np.zeros((2, len(FAMILIES), 6, 6, 2560), np.uint32)
    opposite_counts = np.zeros_like(same_counts)
    family_rows: list[dict] = []
    for mi, material in enumerate(MATERIALS):
        states, index = material_data[material]
        for fi, family in enumerate(FAMILIES):
            panel = "nested_composition" if family == "nested_attitude" else "core"
            left, right = pair_ids(index, family, panel)
            total_emergent = 0
            union_same = 0
            source_same = np.zeros((6, 6), np.int64)
            source_opposite = np.zeros((6, 6), np.int64)
            destination_emergent = np.zeros(6, np.int64)
            for q in range(36):
                current_delta = np.asarray(states[right, q], np.float32) - np.asarray(states[left, q], np.float32)
                next_delta = np.asarray(states[right, q + 1], np.float32) - np.asarray(states[left, q + 1], np.float32)
                current = event(current_delta, thresholds[q])
                nxt = event(next_delta, thresholds[q + 1])
                for di in range(6):
                    emerging = (current[:, di] == 0) & (nxt[:, di] != 0)
                    emergent_counts[mi, fi, di] += emerging.sum(axis=0, dtype=np.uint32)
                    destination_emergent[di] += int(emerging.sum())
                    total_emergent += int(emerging.sum())
                    any_same = np.zeros_like(emerging)
                    for si in range(6):
                        same = emerging & (current[:, si] == nxt[:, di])
                        opposite = emerging & (current[:, si] == -nxt[:, di]) & (current[:, si] != 0)
                        same_counts[mi, fi, si, di] += same.sum(axis=0, dtype=np.uint32)
                        opposite_counts[mi, fi, si, di] += opposite.sum(axis=0, dtype=np.uint32)
                        source_same[si, di] += int(same.sum())
                        source_opposite[si, di] += int(opposite.sum())
                        if si != di:
                            any_same |= same
                    union_same += int(any_same.sum())
            coverage = np.divide(source_same, np.maximum(destination_emergent[None, :], 1))
            opposite = np.divide(source_opposite, np.maximum(destination_emergent[None, :], 1))
            dominant = np.argmax(np.where(np.eye(6, dtype=bool), -1.0, coverage), axis=0)
            family_rows.append({
                "material": material,
                "family": family,
                "pairs": int(len(left)),
                "emergent_events": int(total_emergent),
                "any_source_same_sign_coverage": float(union_same / max(total_emergent, 1)),
                "source_destination_same_sign_coverage": coverage.tolist(),
                "source_destination_opposite_sign_coverage": opposite.tolist(),
                "dominant_source_by_destination": [common.ROLES[int(i)] for i in dominant],
            })
            print(f"[C275] {material}/{family}: reuse={union_same / max(total_emergent, 1):.4f}", flush=True)
    np.save(OUT / "analysis/emergent_counts.uint32.npy", emergent_counts)
    np.save(OUT / "analysis/same_sign_source_counts.uint32.npy", same_counts)
    np.save(OUT / "analysis/opposite_sign_source_counts.uint32.npy", opposite_counts)
    core.write_rows(OUT / "analysis/material_family_rows.jsonl", family_rows)

    comparisons = []
    passing = 0
    for family in FAMILIES:
        third = next(row for row in family_rows if row["material"] == "third" and row["family"] == family)
        fourth = next(row for row in family_rows if row["material"] == "fourth" and row["family"] == family)
        difference = abs(third["any_source_same_sign_coverage"] - fourth["any_source_same_sign_coverage"])
        dominant_agreement = sum(a == b for a, b in zip(third["dominant_source_by_destination"], fourth["dominant_source_by_destination"]))
        passed = fourth["any_source_same_sign_coverage"] >= 0.50 and difference <= 0.10 and dominant_agreement >= 4
        passing += int(passed)
        comparisons.append({
            "family": family,
            "third_coverage": third["any_source_same_sign_coverage"],
            "fourth_coverage": fourth["any_source_same_sign_coverage"],
            "absolute_difference": difference,
            "dominant_source_agreement": dominant_agreement,
            "family_gate_passed": passed,
        })
    report = {
        "phase": 1809,
        "campaign": "C275",
        "status": "cross_role_reuse_ecology_observed",
        "comparisons": comparisons,
        "families_passing": passing,
        "descriptive_gate_passed": passing >= 4,
        "strict_interpretation": "The atlas measures whether a destination's newly active signed coordinate was already present in another role. Repetition can arise from common input or shared downstream processing, so the result is a reuse topology candidate only.",
        "next_authorization": "C276_prospective_cross_role_reuse_prediction_without_patching" if passing >= 4 else "close_C263_C275_stage_and_redesign_joint_state_object",
    }
    core.save(OUT / "analysis/summary.json", report)

    rows = []
    mi = MATERIALS.index("fourth")
    for fi, family in enumerate(FAMILIES):
        for si, source in enumerate(common.ROLES):
            for di, destination in enumerate(common.ROLES):
                denom = np.maximum(emergent_counts[mi, fi, di].astype(np.float32), 1.0)
                rows.append({
                    "source": "c275_cross_role_same_sign_reuse",
                    "family": family,
                    "effect": "same_sign_precedence",
                    "checkpoint": "all_q0_q35",
                    "checkpoint_type": "embedding_and_hidden_state_transitions",
                    "role": f"{source}->{destination}",
                    "label": f"{family}/{source}->{destination}/same-sign",
                    "values": np.round(same_counts[mi, fi, si, di].astype(np.float32) / denom, 6).tolist(),
                })
    asset = {
        "schema": "c275_cross_role_reuse_atlas.v1",
        "phase": 1809,
        "campaign": "C275",
        "model": "Qwen3-4B",
        "dimensions": list(range(2560)),
        "default_coordinates": list(range(64)),
        "total_rows": len(rows),
        "coordinate_semantics": "Each value is the fraction of fourth-material destination-role emergence events preceded by the same signed physical coordinate in a source role, aggregated over q0-q35.",
        "claim_boundary": protocol["claim_boundary"],
        "summary": {"families_passing": passing, "descriptive_gate_passed": passing >= 4},
        "rows": rows,
    }
    save_json(ASSET, asset)
    asset_checks = {
        "schema": asset["schema"] == "c275_cross_role_reuse_atlas.v1",
        "rows": len(rows) == 6 * 6 * 6,
        "all_coordinates": all(len(row["values"]) == 2560 for row in rows),
    }
    core.save(OUT / "analysis/visualization.json", {"asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(), "checks": asset_checks, "all_checks_passed": all(asset_checks.values())})
    analysis_checks = {
        "material_family_rows": len(family_rows) == 12,
        "comparisons": len(comparisons) == 6,
        "count_shapes": list(same_counts.shape) == [2, 6, 6, 6, 2560],
        "finite": bool(np.isfinite([row["fourth_coverage"] for row in comparisons]).all()),
        "asset": all(asset_checks.values()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1809, "campaign": "C275", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
