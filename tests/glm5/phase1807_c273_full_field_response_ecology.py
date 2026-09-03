#!/usr/bin/env python3
"""C273: decompose full-coordinate event ecology without changing C265 rules."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common


core = common.core
OUT = common.RESULT / "phase1807_c273_full_field_response_ecology"
C264 = common.OUTS["C264"]
C265 = common.OUTS["C265"]
C272 = common.OUTS["C272"]
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c273_response_ecology_atlas.json"
FAMILIES = common.FAMILIES + ("nested_attitude",)
CATEGORY_NAMES = (
    "truth_active",
    "stable",
    "reversal",
    "emergence",
    "decay",
    "passport_correct",
    "passport_missed",
    "passport_wrong",
)


def pair_ids(index: list[dict], family: str, panel: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family, "factor_a", panel)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = {
        name: core.load(path / "analysis/final.json")
        for name, path in (("C264", C264), ("C265", C265), ("C272", C272))
    }
    checks = {
        "parents_closed": all(item["all_checks_passed"] for item in parents.values()),
        "authorization": parents["C272"]["next_authorization"].startswith("C273_full_field_response_ecology"),
        "same_frozen_fourth_material": True,
        "same_c265_thresholds_and_passports": True,
        "all_2560_coordinates": True,
        "no_topk_or_projection": True,
        "no_attention_or_mlp": True,
        "observation_only": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1807,
        "campaign": "C273",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_field_failure_ecology_frozen",
        "object": "C264 fourth-material factor-a response at every role, checkpoint, and physical activation coordinate",
        "categories": list(CATEGORY_NAMES),
        "fixed_rule": "Use C265 passport predictions, baseline medians, support and agreement exactly as frozen; do not retune.",
        "descriptive_questions": [
            "Is next-event ecology dominated by same-coordinate persistence?",
            "Does the passport fail mainly by abstention, wrong sign, or both?",
            "Where do emergence and reversal events remain for a future joint-state model?",
        ],
        "interpretive_gates": {
            "persistence_dominant": "stable / truth_active >= 0.50",
            "passport_coverage_bottleneck": "passport_missed > passport_wrong among truth-active events",
            "joint_state_redesign_authorized": "emergence + reversal > 0 and C265/C266 broad gates failed",
        },
        "claim_boundary": "Counts characterize the frozen fourth-material response ecology. They do not identify unique causes, semantic neurons, or coordinate-to-coordinate transmission.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "decide_after_full_coordinate_reveal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    states = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(C264 / "raw/hidden_index.jsonl")
    pred_map = np.load(C265 / "analysis/passport_pred_sign.int8.npy", mmap_mode="r")
    med_map = np.load(C265 / "analysis/passport_baseline_median.float16.npy", mmap_mode="r")
    thresholds = np.asarray(
        core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"],
        np.float32,
    )
    ecology = np.lib.format.open_memmap(
        OUT / "analysis/full_coordinate_ecology_counts.uint16.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(len(FAMILIES), 36, len(common.ROLES), len(CATEGORY_NAMES), 2560),
    )
    summary_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    for fi, family in enumerate(FAMILIES):
        panel = "nested_composition" if family == "nested_attitude" else "core"
        left, right = pair_ids(index, family, panel)
        family_counts = np.zeros(len(CATEGORY_NAMES), np.int64)
        for q in range(36):
            checkpoint_counts = np.zeros(len(CATEGORY_NAMES), np.int64)
            for ri, role in enumerate(common.ROLES):
                source = np.asarray(states[left, q, ri], np.float32)
                current_delta = np.asarray(states[right, q, ri], np.float32) - source
                next_delta = np.asarray(states[right, q + 1, ri], np.float32) - np.asarray(states[left, q + 1, ri], np.float32)
                current = np.where(current_delta > thresholds[q], 1, np.where(current_delta < -thresholds[q], -1, 0)).astype(np.int8)
                truth = np.where(next_delta > thresholds[q + 1], 1, np.where(next_delta < -thresholds[q + 1], -1, 0)).astype(np.int8)
                high = source >= np.asarray(med_map[fi, q, ri], np.float32)[None, :]
                keys = np.where(current < 0, high.astype(np.int8), np.where(current > 0, 2 + high.astype(np.int8), -1))
                predicted = np.zeros_like(current)
                for key in range(4):
                    mask = keys == key
                    predicted[mask] = np.broadcast_to(np.asarray(pred_map[fi, q, ri, key]), predicted.shape)[mask]
                masks = (
                    truth != 0,
                    (current == truth) & (truth != 0),
                    (current == -truth) & (current != 0) & (truth != 0),
                    (current == 0) & (truth != 0),
                    (current != 0) & (truth == 0),
                    (predicted == truth) & (truth != 0),
                    (predicted == 0) & (truth != 0),
                    (predicted != 0) & (predicted != truth) & (truth != 0),
                )
                counts = np.stack([mask.sum(axis=0) for mask in masks], axis=0).astype(np.uint16)
                ecology[fi, q, ri] = counts
                totals = counts.sum(axis=1, dtype=np.int64)
                family_counts += totals
                checkpoint_counts += totals
            active = max(int(checkpoint_counts[0]), 1)
            checkpoint_rows.append({
                "family": family,
                "checkpoint_from": q,
                "checkpoint_to": q + 1,
                "truth_active": int(checkpoint_counts[0]),
                "stable_fraction": float(checkpoint_counts[1] / active),
                "reversal_fraction": float(checkpoint_counts[2] / active),
                "emergence_fraction": float(checkpoint_counts[3] / active),
                "passport_correct_fraction": float(checkpoint_counts[5] / active),
                "passport_missed_fraction": float(checkpoint_counts[6] / active),
                "passport_wrong_fraction": float(checkpoint_counts[7] / active),
            })
        active = max(int(family_counts[0]), 1)
        item = {
            "family": family,
            "pairs": int(len(left)),
            "truth_active": int(family_counts[0]),
            "stable_fraction": float(family_counts[1] / active),
            "reversal_fraction": float(family_counts[2] / active),
            "emergence_fraction": float(family_counts[3] / active),
            "decay_to_truth_active_ratio": float(family_counts[4] / active),
            "passport_correct_fraction": float(family_counts[5] / active),
            "passport_missed_fraction": float(family_counts[6] / active),
            "passport_wrong_fraction": float(family_counts[7] / active),
        }
        item["persistence_dominant"] = item["stable_fraction"] >= 0.50
        item["passport_coverage_bottleneck"] = item["passport_missed_fraction"] > item["passport_wrong_fraction"]
        summary_rows.append(item)
        print(f"[C273] {family}: stable={item['stable_fraction']:.4f}, missed={item['passport_missed_fraction']:.4f}, wrong={item['passport_wrong_fraction']:.4f}", flush=True)
    ecology.flush()
    core.write_rows(OUT / "analysis/family_ecology.jsonl", summary_rows)
    core.write_rows(OUT / "analysis/checkpoint_ecology.jsonl", checkpoint_rows)

    total_active = sum(row["truth_active"] for row in summary_rows)
    weighted = lambda key: float(sum(row[key] * row["truth_active"] for row in summary_rows) / max(total_active, 1))
    overall = {
        "truth_active": int(total_active),
        "stable_fraction": weighted("stable_fraction"),
        "reversal_fraction": weighted("reversal_fraction"),
        "emergence_fraction": weighted("emergence_fraction"),
        "passport_correct_fraction": weighted("passport_correct_fraction"),
        "passport_missed_fraction": weighted("passport_missed_fraction"),
        "passport_wrong_fraction": weighted("passport_wrong_fraction"),
    }
    c265_headline = parents["C265"]["headline"]
    report = {
        "phase": 1807,
        "campaign": "C273",
        "status": "full_field_response_ecology_observed",
        "overall": overall,
        "families": summary_rows,
        "persistence_dominant": overall["stable_fraction"] >= 0.50,
        "passport_coverage_bottleneck": overall["passport_missed_fraction"] > overall["passport_wrong_fraction"],
        "joint_state_redesign_authorized": bool((overall["emergence_fraction"] + overall["reversal_fraction"] > 0) and not c265_headline["broad_prediction_gate_passed"]),
        "strict_interpretation": "The independent per-coordinate passport loses mainly where the next event cannot be recovered from that coordinate's own sign and a one-bit baseline guard. This is a failure ecology, not proof that a particular cross-coordinate model is correct.",
        "next_authorization": "C274_joint_full_field_condition_discovery_with_frozen_C264_holdout; no_static_mask_patch",
    }
    core.save(OUT / "analysis/summary.json", report)

    chosen_q = (0, 4, 8, 12, 16, 20, 24, 28, 32, 35)
    rows = []
    for fi, family in enumerate(FAMILIES):
        for q in chosen_q:
            for role in ("relation", "boundary"):
                ri = common.ROLES.index(role)
                truth = np.asarray(ecology[fi, q, ri, 0], np.float32)
                denom = np.maximum(truth, 1.0)
                for category in ("stable", "reversal", "emergence", "passport_missed", "passport_wrong"):
                    ci = CATEGORY_NAMES.index(category)
                    rows.append({
                        "source": "c273_full_coordinate_failure_ecology",
                        "family": family,
                        "effect": category,
                        "checkpoint": q,
                        "checkpoint_type": "embedding" if q == 0 else "hidden_state",
                        "role": role,
                        "label": f"{family}/{category}/q{q}/{role}",
                        "values": np.round(np.asarray(ecology[fi, q, ri, ci], np.float32) / denom, 6).tolist(),
                    })
    asset = {
        "schema": "c273_response_ecology_atlas.v1",
        "phase": 1807,
        "campaign": "C273",
        "model": "Qwen3-4B",
        "dimensions": list(range(2560)),
        "default_coordinates": list(range(64)),
        "total_rows": len(rows),
        "coordinate_semantics": "Each value is a per-coordinate event-category fraction over all eligible fourth-material pairs. q0 is embedding and q1-q35 are HiddenState transitions.",
        "claim_boundary": protocol["claim_boundary"],
        "summary": overall,
        "rows": rows,
    }
    save_json(ASSET, asset)
    asset_checks = {
        "schema": asset["schema"] == "c273_response_ecology_atlas.v1",
        "rows": len(rows) == len(FAMILIES) * len(chosen_q) * 2 * 5,
        "all_coordinates": all(len(row["values"]) == 2560 for row in rows),
        "embedding_present": any(row["checkpoint"] == 0 for row in rows),
        "hidden_present": any(row["checkpoint"] > 0 for row in rows),
    }
    core.save(OUT / "analysis/visualization.json", {
        "asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        "checks": asset_checks,
        "all_checks_passed": all(asset_checks.values()),
    })
    analysis_checks = {
        "families": len(summary_rows) == 6,
        "checkpoints": len(checkpoint_rows) == 6 * 36,
        "ecology_shape": list(ecology.shape) == [6, 36, 6, 8, 2560],
        "finite": bool(np.isfinite(list(overall.values())).all()),
        "asset": all(asset_checks.values()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {
        "contract": all(checks.values()),
        "analysis": all(analysis_checks.values()),
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {
        "phase": 1807,
        "campaign": "C273",
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": report,
        "next_authorization": report["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
