#!/usr/bin/env python3
"""Compare Phase569 correct and relation-confusion coarse event traces."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase569_relation_competition"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("path_discovery", "path_confirmation")
PHENOTYPES = ("stable_correct", "stable_relation_confusion")
EXPECTED_CASES_PER_CELL = 96
NORMALIZED_GAP_MIN = 0.005
DIRECT_POSITIVE_RATE_GAP_MIN = 0.15
DECODED_POSITIVE_RATE_GAP_MIN = 0.20
ANALYSIS_PATH = OUT_DIR / "phase569_coarse_trace_analysis.json"
CANDIDATE_PATH = OUT_DIR / "phase569_coarse_trace_candidate_registry.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def same_nonzero_sign(left: float, right: float) -> bool:
    return left * right > 0.0


def depth_band(relative_depth: float) -> int:
    return min(7, int(relative_depth * 8.0))


def analyze_model(model: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_path = OUT_DIR / f"phase569_{model}_coarse_trace_rows.jsonl"
    summary_path = OUT_DIR / f"phase569_{model}_coarse_trace_summary.json"
    contract_path = OUT_DIR / f"phase569_{model}_coarse_trace_contract.json"
    for path in (rows_path, summary_path, contract_path):
        if not path.exists():
            raise RuntimeError(f"Missing Phase569 trace artifact: {path}")
    summary = read_json(summary_path)
    contract = read_json(contract_path)
    rows = list(iter_jsonl(rows_path))
    if summary["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"Phase569 {model} trace hash drift")
    expected = summary["layer_count"] * 4 * 10 * 4
    if len(rows) != expected or summary["event_row_count"] != expected:
        raise RuntimeError(f"Phase569 {model} trace denominator drift")
    if any(
        row["model"] != model
        or row["case_count"] != EXPECTED_CASES_PER_CELL
        or row["sealed"]
        or row["causal"]
        for row in rows
    ):
        raise RuntimeError(f"Phase569 {model} trace identity/evidence drift")
    groups: dict[tuple[int, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        coord = (row["layer"], row["component"], row["semantic_role"])
        groups[coord][(row["phenotype"], row["split"])] = row
    comparisons = []
    candidates = []
    for (layer, component, role), values in sorted(groups.items()):
        if len(values) != 4:
            raise RuntimeError(f"Phase569 {model} comparison cell is incomplete")
        normalized_gaps = {}
        direct_rate_gaps = {}
        decoded_margin_gaps = {}
        decoded_rate_gaps = {}
        for split in SPLITS:
            correct = values[("stable_correct", split)]
            confusion = values[("stable_relation_confusion", split)]
            normalized_gaps[split] = (
                correct["mean_normalized_target_other_projection"]
                - confusion["mean_normalized_target_other_projection"]
            )
            direct_rate_gaps[split] = (
                correct["direct_target_minus_other_positive_rate"]
                - confusion["direct_target_minus_other_positive_rate"]
            )
            if correct["decoded_margin_available"]:
                decoded_margin_gaps[split] = (
                    correct["mean_decoded_target_minus_other_margin"]
                    - confusion["mean_decoded_target_minus_other_margin"]
                )
                decoded_rate_gaps[split] = (
                    correct["decoded_target_minus_other_positive_rate"]
                    - confusion["decoded_target_minus_other_positive_rate"]
                )
        normalized_replicates = (
            same_nonzero_sign(*normalized_gaps.values())
            and min(abs(value) for value in normalized_gaps.values()) >= NORMALIZED_GAP_MIN
        )
        direct_rate_replicates = (
            same_nonzero_sign(*direct_rate_gaps.values())
            and min(abs(value) for value in direct_rate_gaps.values())
            >= DIRECT_POSITIVE_RATE_GAP_MIN
        )
        decoded_rate_replicates = bool(
            decoded_rate_gaps
            and all(value > 0.0 for value in decoded_rate_gaps.values())
            and min(decoded_rate_gaps.values()) >= DECODED_POSITIVE_RATE_GAP_MIN
        )
        component_candidate = normalized_replicates and direct_rate_replicates
        residual_candidate = normalized_replicates and decoded_rate_replicates
        replication_score = (
            min(abs(value) for value in normalized_gaps.values())
            + min(abs(value) for value in direct_rate_gaps.values())
            + (
                min(abs(value) for value in decoded_rate_gaps.values())
                if decoded_rate_gaps else 0.0
            )
        )
        comparison = {
            "model": model,
            "layer": layer,
            "layer_count": summary["layer_count"],
            "relative_depth": layer / max(1, summary["layer_count"] - 1),
            "depth_band_8": depth_band(layer / max(1, summary["layer_count"] - 1)),
            "component": component,
            "semantic_role": role,
            "normalized_projection_gaps_correct_minus_confusion": normalized_gaps,
            "direct_positive_rate_gaps_correct_minus_confusion": direct_rate_gaps,
            "decoded_margin_gaps_correct_minus_confusion": decoded_margin_gaps,
            "decoded_positive_rate_gaps_correct_minus_confusion": decoded_rate_gaps,
            "normalized_projection_gap_replicates": normalized_replicates,
            "direct_positive_rate_gap_replicates": direct_rate_replicates,
            "decoded_positive_rate_gap_replicates": decoded_rate_replicates,
            "coarse_component_candidate": component_candidate,
            "coarse_residual_candidate": residual_candidate,
            "coarse_observer_candidate": component_candidate or residual_candidate,
            "replication_score": replication_score,
            "observer_only": True,
            "causal": False,
        }
        comparisons.append(comparison)
        if comparison["coarse_observer_candidate"]:
            candidates.append(comparison)
    candidates.sort(
        key=lambda row: (row["relative_depth"], -row["replication_score"], row["component"], row["semantic_role"])
    )
    top_by_strength = sorted(
        candidates, key=lambda row: (-row["replication_score"], row["relative_depth"])
    )[:64]
    report = {
        "model": model,
        "layer_count": summary["layer_count"],
        "trace_case_count": summary["case_count"],
        "comparison_count": len(comparisons),
        "coarse_observer_candidate_count": len(candidates),
        "first_32_candidates_by_depth": candidates[:32],
        "top_32_candidates_by_strength": top_by_strength[:32],
        "max_component_ledger_relative_error": summary["max_component_ledger_relative_error"],
        "mean_component_ledger_relative_error": summary["mean_component_ledger_relative_error"],
        "full_vectors_persisted": summary["full_vectors_persisted"],
        "causal_intervention_executed": False,
        "sealed_split_read": False,
    }
    return report, top_by_strength


def cross_model_topology(candidates_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for model, candidates in candidates_by_model.items():
        for row in candidates:
            discovery_gap = row["normalized_projection_gaps_correct_minus_confusion"][
                "path_discovery"
            ]
            sign = 1 if discovery_gap > 0.0 else -1
            groups[(row["component"], row["semantic_role"], row["depth_band_8"], sign)].append({
                "model": model,
                "layer": row["layer"],
                "relative_depth": row["relative_depth"],
                "replication_score": row["replication_score"],
            })
    output = []
    for (component, role, band, sign), rows in sorted(groups.items()):
        models = sorted({row["model"] for row in rows})
        if len(models) < 2:
            continue
        best_by_model = {
            model: max(
                (row for row in rows if row["model"] == model),
                key=lambda row: row["replication_score"],
            )
            for model in models
        }
        output.append({
            "component": component,
            "semantic_role": role,
            "depth_band_8": band,
            "normalized_gap_sign": sign,
            "model_count": len(models),
            "models": models,
            "best_event_by_model": best_by_model,
            "minimum_replication_score_across_models": min(
                row["replication_score"] for row in best_by_model.values()
            ),
            "observer_only": True,
            "causal": False,
        })
    return sorted(
        output,
        key=lambda row: (
            -row["model_count"], row["depth_band_8"],
            -row["minimum_replication_score_across_models"],
        ),
    )


def analyze() -> dict[str, Any]:
    reports = []
    candidates_by_model = {}
    for model in MODELS:
        report, candidates = analyze_model(model)
        reports.append(report)
        candidates_by_model[model] = candidates
    shared = cross_model_topology(candidates_by_model)
    analysis = {
        "schema_version": "phase569_coarse_trace_analysis.v1",
        "phase_id": "Phase569",
        "created_at": now(),
        "status": "complete",
        "screening_thresholds": {
            "minimum_cases_per_phenotype_split_coordinate": EXPECTED_CASES_PER_CELL,
            "minimum_replicated_absolute_normalized_projection_gap": NORMALIZED_GAP_MIN,
            "minimum_replicated_absolute_direct_positive_rate_gap": DIRECT_POSITIVE_RATE_GAP_MIN,
            "minimum_replicated_positive_decoded_rate_gap": DECODED_POSITIVE_RATE_GAP_MIN,
            "thresholds_are_screening_only_not_mechanism_proof": True,
        },
        "model_reports": reports,
        "cross_model_shared_topology_count": len(shared),
        "cross_model_shared_topology": shared[:128],
        "observer_only": True,
        "causal_intervention_executed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }
    candidate_registry = {
        "schema_version": "phase569_coarse_trace_candidate_registry.v1",
        "phase_id": "Phase569",
        "created_at": now(),
        "analysis_sha256_before_registry_write": None,
        "candidates_by_model": candidates_by_model,
        "cross_model_shared_topology": shared,
        "observer_only": True,
        "causal_validation_required": True,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, analysis)
    candidate_registry["analysis_sha256_before_registry_write"] = sha256_file(ANALYSIS_PATH)
    write_json(CANDIDATE_PATH, candidate_registry)
    print(json.dumps({
        "models": [
            {
                "model": report["model"],
                "candidate_count": report["coarse_observer_candidate_count"],
                "first_candidate": (
                    report["first_32_candidates_by_depth"][0]
                    if report["first_32_candidates_by_depth"] else None
                ),
            }
            for report in reports
        ],
        "cross_model_shared_topology_count": len(shared),
        "first_shared_topology": shared[0] if shared else None,
    }, ensure_ascii=False, indent=2))
    return analysis


if __name__ == "__main__":
    analyze()
