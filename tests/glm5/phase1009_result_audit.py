#!/usr/bin/env python3
"""Audit Phase1009 protocol, behavior, atlas, and optional causal outputs."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1009_crossfamily_response_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    NATURAL_STATES,
    OUT_ROOT,
    PAIR_OPERATIONS,
    PHASE,
    ROLE_CLASSES,
    read_json,
    read_jsonl,
    write_json,
)


OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def protocol_audit(protocol: dict[str, Any]) -> dict[str, Any]:
    model_rows = {}
    for model_name in MODELS:
        root = OUT_ROOT / "protocol" / model_name
        cases = read_jsonl(root / "cases.jsonl")
        units = read_jsonl(root / "units.jsonl")
        require(len(cases) == 1152, f"{model_name}: case count")
        require(len(units) == 144, f"{model_name}: unit count")
        case_by_id = {case["record_id"]: case for case in cases}
        require(len(case_by_id) == len(cases), f"{model_name}: duplicate case")
        family_counts = Counter(unit["family"] for unit in units)
        require(
            family_counts == Counter({family: 48 for family in FAMILIES}),
            f"{model_name}: family unit counts {family_counts}",
        )
        split_counts = Counter(
            (unit["family"], unit["split"]) for unit in units
        )
        require(
            all(
                split_counts[(family, split)] == 24
                for family in FAMILIES
                for split in ("discovery", "confirmation")
            ),
            f"{model_name}: split counts {split_counts}",
        )
        for unit in units:
            base = case_by_id[unit["case_ids"]["base"]]
            for state in NATURAL_STATES:
                case = case_by_id[unit["case_ids"][state]]
                require(
                    set(case["role_positions"])
                    == set(ROLE_CLASSES[case["family"]]),
                    f"{case['record_id']}: role schema",
                )
                require(
                    all(
                        0 <= int(position) < len(case["input_ids"])
                        for position in case["role_positions"].values()
                    ),
                    f"{case['record_id']}: role bounds",
                )
            require(
                case_by_id[unit["case_ids"]["FQ"]]["gold"] == base["gold"],
                f"{unit['unit_id']}: FQ answer invariant",
            )
            for state in ("E", "O", "N", "S"):
                require(
                    case_by_id[unit["case_ids"][state]]["gold"]
                    == base["gold"],
                    f"{unit['unit_id']}: {state} answer invariant",
                )
            for state in ("F", "Q"):
                require(
                    case_by_id[unit["case_ids"][state]]["gold"]
                    != base["gold"],
                    f"{unit['unit_id']}: {state} answer flip",
                )
        model_rows[model_name] = {
            "case_count": len(cases),
            "unit_count": len(units),
            "pair_count": len(units) * len(PAIR_OPERATIONS),
            "family_unit_counts": dict(sorted(family_counts.items())),
        }
    return {
        "protocol_revision": int(protocol["protocol_revision"]),
        "protocol_digest": protocol["preregistration_digest"],
        "models": model_rows,
    }


def behavior_audit() -> dict[str, Any]:
    result = {}
    for model_name in MODELS:
        root = OUT_ROOT / "behavior" / model_name
        rows = read_jsonl(root / "rows.jsonl")
        pairs = read_jsonl(root / "pair_qualification.jsonl")
        summary = read_json(root / "summary.json")
        require(len(rows) == 1152, f"{model_name}: behavior rows")
        require(len(pairs) == 1152, f"{model_name}: behavior pairs")
        require(
            summary["case_count"] == len(rows),
            f"{model_name}: behavior summary coverage",
        )
        family_rates = {
            row["family"]: float(row["semantic_panel_rate"])
            for row in summary["family_rates"]
        }
        result[model_name] = {
            "semantic_panel_rate": float(
                summary["overall_semantic_panel_rate"]
            ),
            "strict_teacher_rate": float(
                summary["overall_strict_teacher_rate"]
            ),
            "rollout_rate": float(summary["overall_rollout_case_rate"]),
            "family_semantic_panel_rates": family_rates,
            "all_families_above_0_8": bool(
                all(value >= 0.8 for value in family_rates.values())
            ),
        }
    return result


def scan_audit() -> dict[str, Any]:
    result = {}
    scalar_total = 0
    for model_name in MODELS:
        model_summary = read_json(
            OUT_ROOT / "scan" / model_name / "summary.json"
        )
        family_rows = {}
        model_scalar = 0
        for family in FAMILIES:
            root = OUT_ROOT / "scan" / model_name / family
            summary = read_json(root / "summary.json")
            events = read_jsonl(root / "events.jsonl")
            units = read_jsonl(root / "units.jsonl")
            arrays = np.load(root / "response_scalars.npz")
            directions = np.load(root / "direction_consistency.npz")
            normalized = arrays["normalized_magnitude"]
            raw = arrays["raw_magnitude"]
            expected_shape = (
                len(units),
                len(ANALYSIS_OPERATIONS),
                len(events),
            )
            require(
                normalized.shape == expected_shape,
                f"{model_name}/{family}: normalized shape",
            )
            require(
                raw.shape == expected_shape,
                f"{model_name}/{family}: raw shape",
            )
            require(
                np.all(np.isfinite(normalized)),
                f"{model_name}/{family}: nonfinite normalized",
            )
            require(
                np.all(np.isfinite(raw)),
                f"{model_name}/{family}: nonfinite raw",
            )
            identity = normalized[:, OP_INDEX["I"], :]
            require(
                float(np.max(identity)) == 0.0,
                f"{model_name}/{family}: identity floor",
            )
            require(
                directions["direction_consistency"].shape
                == (
                    len(ANALYSIS_OPERATIONS),
                    2,
                    len(events),
                ),
                f"{model_name}/{family}: direction shape",
            )
            expected_scalar = int(np.prod(expected_shape))
            require(
                int(summary["scalar_measurement_count"])
                == expected_scalar,
                f"{model_name}/{family}: scalar count",
            )
            require(
                int(summary["raw_hidden_tensors_persisted"]) == 0,
                f"{model_name}/{family}: raw persistence",
            )
            model_scalar += expected_scalar
            family_rows[family] = {
                "unit_count": len(units),
                "event_count": len(events),
                "scalar_measurement_count": expected_scalar,
                "identity_maximum": float(np.max(identity)),
            }
        require(
            int(model_summary["scalar_measurement_count"]) == model_scalar,
            f"{model_name}: aggregate scalar count",
        )
        scalar_total += model_scalar
        result[model_name] = {
            "scalar_measurement_count": model_scalar,
            "families": family_rows,
        }
    result["total_scalar_measurement_count"] = scalar_total
    return result


def final_audit() -> dict[str, Any]:
    root = OUT_ROOT / "final"
    summary = read_json(root / "summary.json")
    motifs = read_jsonl(root / "all_trajectory_motifs.jsonl")
    cross = read_jsonl(root / "cross_family_motifs.jsonl")
    require(
        len(motifs) == int(summary["trajectory_count"]),
        "final motif count",
    )
    require(
        sum(row["repeated_candidate"] for row in motifs)
        == int(summary["repeated_candidate_count"]),
        "final repeated count",
    )
    require(
        all(
            not row["direct_surface_confounded"]
            for row in motifs
            if row["refinement_eligible"]
        ),
        "surface-confounded motif marked eligible",
    )
    motif_ids = {row["motif_id"] for row in motifs}
    for row in cross:
        require(
            row["family_count"] >= 2,
            f"{row['cross_family_motif_id']}: family support",
        )
        require(
            all(
                motif_id in motif_ids
                for motif_id in row["member_motif_ids"]
            ),
            f"{row['cross_family_motif_id']}: missing member",
        )
        require(
            float(row["relative_depth_spread"]) <= 0.1500001,
            f"{row['cross_family_motif_id']}: depth spread",
        )
    return {
        "trajectory_count": len(motifs),
        "repeated_candidate_count": int(
            summary["repeated_candidate_count"]
        ),
        "refinement_eligible_count": int(
            summary["refinement_eligible_count"]
        ),
        "cross_family_motif_count": len(cross),
        "strong_cross_family_cross_model_count": int(
            summary["strong_cross_family_cross_model_count"]
        ),
        "late_semantic0_shared_decision_candidate_count": int(
            summary["late_semantic0_shared_decision_candidate_count"]
        ),
        "automatic_next_step_eligible": bool(
            summary["automatic_next_step_rule"]["eligible"]
        ),
    }


def causal_audit() -> dict[str, Any]:
    result = {}
    for model_name in ("qwen3", "glm4"):
        path = OUT_ROOT / "causal_replication" / model_name / "summary.json"
        if not path.exists():
            result[model_name] = {"status": "not_run"}
            continue
        summary = read_json(path)
        rows = read_jsonl(
            OUT_ROOT
            / "causal_replication"
            / model_name
            / "units.jsonl"
        )
        require(
            not summary["selection_used_phase1009_data"],
            f"{model_name}: causal selection leakage",
        )
        require(
            summary["no_op_audit_pass"],
            f"{model_name}: causal no-op",
        )
        require(
            max(row["noop_max_logit_error"] for row in rows) <= 1e-5,
            f"{model_name}: causal no-op row",
        )
        result[model_name] = {
            "status": "complete",
            "unit_operation_count": len(rows),
            "positive_cell_count": int(summary["positive_cell_count"]),
            "positive_families": summary["positive_families"],
            "cross_family_local_replication": bool(
                summary["cross_family_local_replication"]
            ),
        }
    aggregate_path = OUT_ROOT / "causal_replication" / "summary.json"
    if aggregate_path.exists():
        aggregate = read_json(aggregate_path)
        require(
            aggregate["all_no_op_audits_pass"],
            "aggregate causal no-op",
        )
        require(
            int(aggregate["positive_cell_count"])
            == sum(
                row.get("positive_cell_count", 0)
                for row in result.values()
                if row["status"] == "complete"
            ),
            "aggregate causal positive-cell count",
        )
        require(
            aggregate["models_with_cross_family_local_replication"]
            == [
                model_name
                for model_name in ("qwen3", "glm4")
                if result[model_name].get(
                    "cross_family_local_replication", False
                )
            ],
            "aggregate causal cross-family support",
        )
        result["aggregate"] = {
            "cell_count": int(aggregate["cell_count"]),
            "positive_cell_count": int(
                aggregate["positive_cell_count"]
            ),
            "models_with_cross_family_local_replication": aggregate[
                "models_with_cross_family_local_replication"
            ],
        }
    return result


def supplementary_audit(final: dict[str, Any]) -> dict[str, Any]:
    rollout = read_json(
        OUT_ROOT / "behavior" / "rollout_surface_audit_summary.json"
    )
    require(len(rollout["models"]) == len(MODELS), "rollout model count")
    rollout_rows = {}
    for row in rollout["models"]:
        model_name = row["model"]
        require(model_name in MODELS, f"rollout unknown model {model_name}")
        require(int(row["case_count"]) == 1152, f"{model_name}: rollout cases")
        require(
            float(row["frozen_strict_exact_rate"]) == 0.0,
            f"{model_name}: frozen strict rollout changed",
        )
        require(
            0.0 <= float(row["flexible_full_protocol_rate"]) <= 1.0,
            f"{model_name}: flexible rollout bounds",
        )
        rollout_rows[model_name] = {
            "name_case_insensitive_rate": float(
                row["name_case_insensitive_rate"]
            ),
            "flexible_full_protocol_rate": float(
                row["flexible_full_protocol_rate"]
            ),
            "frozen_strict_exact_rate": float(
                row["frozen_strict_exact_rate"]
            ),
        }
    require(
        set(rollout_rows) == set(MODELS),
        "rollout model coverage",
    )

    threshold = read_json(OUT_ROOT / "audit" / "threshold_stability.json")
    frozen = threshold["runs"]["0.90"]
    require(
        int(frozen["repeated_candidate_count"])
        == int(final["repeated_candidate_count"]),
        "threshold audit repeated count",
    )
    require(
        int(frozen["refinement_eligible_count"])
        == int(final["refinement_eligible_count"]),
        "threshold audit eligible count",
    )
    require(
        int(frozen["cross_family_motif_count"])
        == int(final["cross_family_motif_count"]),
        "threshold audit cross-family count",
    )
    require(
        int(frozen["strong_cross_family_cross_model_count"])
        == int(final["strong_cross_family_cross_model_count"]),
        "threshold audit strong count",
    )
    require(
        0.0 <= float(threshold["eligible_0_85_vs_0_95_jaccard"]) <= 1.0,
        "eligible threshold Jaccard",
    )
    require(
        0.0 <= float(threshold["strong_0_85_vs_0_95_jaccard"]) <= 1.0,
        "strong threshold Jaccard",
    )

    prefix_root = (
        OUT_ROOT / "causal_replication" / "glm4_prefix_audit"
    )
    prefix = read_json(prefix_root / "summary.json")
    common = read_json(prefix_root / "common_denominator_summary.json")
    require(prefix["no_op_audit_pass"], "prefix audit no-op")
    require(
        not prefix["selection_used_phase1009_or_surface_data"],
        "prefix audit selection leakage",
    )
    require(
        set(prefix["surfaces"]) == {"answer", "result", "choice"},
        "prefix surface coverage",
    )
    require(
        len(prefix["cell_summaries"]) == 18,
        "prefix cell coverage",
    )
    require(
        all(
            row["localized_directional_contribution"]
            for row in prefix["cell_summaries"]
            if row["status"] == "complete"
        ),
        "prefix complete-cell contribution",
    )
    require(
        int(common["complete_cell_count"]) == 15,
        "common-denominator complete cells",
    )
    require(
        int(common["positive_cell_count"])
        == int(common["complete_cell_count"]),
        "common-denominator positive cells",
    )
    require(
        common["all_complete_cells_positive"],
        "common-denominator contribution",
    )
    require(
        int(common["original_vs_rebatched_answer_state_count"]) == 288,
        "rebatched state coverage",
    )
    require(
        int(common["original_vs_rebatched_answer_hit_mismatch_count"]) == 8,
        "rebatched mismatch count",
    )

    return {
        "rollout_surface_diagnostic": rollout_rows,
        "threshold_stability": {
            "thresholds": threshold["thresholds"],
            "eligible_0_85_vs_0_95_jaccard": float(
                threshold["eligible_0_85_vs_0_95_jaccard"]
            ),
            "strong_0_85_vs_0_95_jaccard": float(
                threshold["strong_0_85_vs_0_95_jaccard"]
            ),
            "late_candidate_count_by_threshold": {
                key: int(
                    row["late_semantic0_attention_candidate_count"]
                )
                for key, row in threshold["runs"].items()
            },
        },
        "glm4_output_prefix_specificity": {
            "surface_summaries": prefix["surface_summaries"],
            "common_complete_cell_count": int(
                common["complete_cell_count"]
            ),
            "common_positive_cell_count": int(
                common["positive_cell_count"]
            ),
            "rebatched_answer_hit_mismatch_count": int(
                common["original_vs_rebatched_answer_hit_mismatch_count"]
            ),
            "rebatched_answer_hit_mismatch_rate": float(
                common["original_vs_rebatched_answer_hit_mismatch_rate"]
            ),
            "remaining_confound": (
                "All tested outputs remain person names from the same "
                "candidate type; literal prefix invariance is not "
                "answer-type invariance."
            ),
        },
    }


def inventory() -> list[dict[str, Any]]:
    files = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        if path.relative_to(OUT_ROOT).as_posix() == "audit/summary.json":
            continue
        row = {
            "relative_path": path.relative_to(OUT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix in {".json", ".jsonl"}:
            row["sha256"] = sha256(path)
        files.append(row)
    return files


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    final = final_audit()
    payload = {
        "schema_version": "phase1009_result_audit.v2",
        "phase": PHASE,
        "protocol": protocol_audit(protocol),
        "behavior": behavior_audit(),
        "scan": scan_audit(),
        "final": final,
        "causal": causal_audit(),
        "supplementary": supplementary_audit(final),
        "integrity_failures": [],
        "scientific_claim_ceiling": (
            "repeated response structure plus an independently replicated "
            "GLM4 late local head contribution across three synthetic "
            "families and three literal prefixes; never a shared transport "
            "path, answer-type-general mechanism, or language formula"
        ),
    }
    payload["inventory"] = inventory()
    payload["audit_pass"] = True
    write_json(OUT_ROOT / "audit" / "summary.json", payload)
    print(json.dumps({
        "phase": PHASE,
        "audit_pass": True,
        "total_scalar_measurement_count": payload["scan"][
            "total_scalar_measurement_count"
        ],
        "final": payload["final"],
        "causal": payload["causal"],
        "file_count": len(payload["inventory"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
