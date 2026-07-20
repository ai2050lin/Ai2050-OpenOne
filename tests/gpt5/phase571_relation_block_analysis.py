#!/usr/bin/env python3
"""Discover and independently confirm Phase571 contiguous signed-write blocks."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase571_relation_block"
MODELS = ("qwen3", "glm4", "deepseek7b")
POOLS = ("block_discovery", "block_confirmation")
PHENOTYPES = ("stable_correct", "stable_relation_confusion")
ANALYSIS_PATH = OUT_DIR / "phase571_continuous_block_analysis.json"
REGISTRY_PATH = OUT_DIR / "phase571_continuous_block_registry.json"
PROTOCOL_PATH = OUT_DIR / "phase571_frozen_protocol.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def band(layer: int, layer_count: int, band_count: int) -> int:
    return min(band_count - 1, (layer * band_count) // layer_count)


def layer_bounds(start_band: int, end_band: int, layer_count: int, band_count: int) -> tuple[int, int]:
    members = [
        layer for layer in range(layer_count)
        if start_band <= band(layer, layer_count, band_count) <= end_band
    ]
    if not members:
        raise RuntimeError("Phase571 selected an empty depth interval")
    return min(members), max(members)


def wrong_interval(start: int, end: int, band_count: int) -> tuple[int, int]:
    width = end - start + 1
    options = []
    for candidate_start in range(0, band_count - width + 1):
        candidate_end = candidate_start + width - 1
        overlap = not (candidate_end < start or candidate_start > end)
        distance = max(0, max(start - candidate_end, candidate_start - end))
        options.append((overlap, -distance, candidate_start, candidate_end))
    selected = min(options)
    return selected[2], selected[3]


def compact_metrics(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "mean_absolute": mean([abs(value) for value in values]),
        "positive_rate": rate(sum(value > 0.0 for value in values), len(values)),
    }


def analyze_model(model: str, frozen: dict[str, Any]) -> dict[str, Any]:
    rows_path = OUT_DIR / f"phase571_{model}_signed_write_rows.jsonl.gz"
    summary_path = OUT_DIR / f"phase571_{model}_signed_write_summary.json"
    behavior_path = OUT_DIR / f"phase571_{model}_matched_behavior_summary.json"
    for path in (rows_path, summary_path, behavior_path):
        if not path.exists():
            raise RuntimeError(f"Missing Phase571 artifact: {path}")
    summary = read_json(summary_path)
    behavior = read_json(behavior_path)
    if summary["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"Phase571 {model} trace hash drift")
    rows = list(iter_jsonl(rows_path))
    expected_rows = summary["case_count"] * summary["semantic_role_count"]
    if len(rows) != expected_rows or any(row["sealed"] or row["causal"] for row in rows):
        raise RuntimeError(f"Phase571 {model} trace denominator/evidence drift")
    layer_count = int(summary["layer_count"])
    band_count = int(frozen["depth_band_count"])
    rule = frozen["block_discovery_rule"]
    role_priority = frozen["causal_role_priority"]
    max_width = int(rule["maximum_interval_width_in_bands"])
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["semantic_role"], row["pool"], row["phenotype"])].append(row)
    candidates = []
    comparisons = []
    for role in role_priority:
        for start_band in range(band_count):
            for end_band in range(start_band, min(band_count, start_band + max_width)):
                values: dict[str, dict[str, list[float]]] = {
                    pool: {phenotype: [] for phenotype in PHENOTYPES} for pool in POOLS
                }
                for pool in POOLS:
                    for phenotype in PHENOTYPES:
                        for row in grouped[(role, pool, phenotype)]:
                            attention = row["attention_signed_unit_projection"]
                            mlp = row["mlp_signed_unit_projection"]
                            total = sum(
                                attention[layer] + mlp[layer]
                                for layer in range(layer_count)
                                if start_band <= band(layer, layer_count, band_count) <= end_band
                            )
                            values[pool][phenotype].append(float(total))
                pool_reports = {}
                gaps = {}
                relative_gaps = {}
                positive_rate_gaps = {}
                for pool in POOLS:
                    correct = compact_metrics(values[pool]["stable_correct"])
                    confusion = compact_metrics(values[pool]["stable_relation_confusion"])
                    gap = correct["mean"] - confusion["mean"]
                    scale = mean([
                        abs(value)
                        for phenotype in PHENOTYPES for value in values[pool][phenotype]
                    ])
                    relative = gap / max(scale, 1e-12)
                    positive_gap = correct["positive_rate"] - confusion["positive_rate"]
                    pool_reports[pool] = {
                        "stable_correct": correct,
                        "stable_relation_confusion": confusion,
                    }
                    gaps[pool] = gap
                    relative_gaps[pool] = relative
                    positive_rate_gaps[pool] = positive_gap
                same_gap_sign = gaps[POOLS[0]] * gaps[POOLS[1]] > 0.0
                rate_tracks_mean = all(
                    gaps[pool] * positive_rate_gaps[pool] > 0.0 for pool in POOLS
                )
                same_rate_sign = positive_rate_gaps[POOLS[0]] * positive_rate_gaps[POOLS[1]] > 0.0
                discovery_pass = (
                    abs(relative_gaps["block_discovery"])
                    >= rule["minimum_absolute_relative_gap_each_split"]
                    and abs(positive_rate_gaps["block_discovery"])
                    >= rule["minimum_absolute_positive_rate_gap_each_split"]
                )
                confirmation_pass = (
                    discovery_pass
                    and same_gap_sign
                    and same_rate_sign
                    and rate_tracks_mean
                    and abs(relative_gaps["block_confirmation"])
                    >= rule["minimum_absolute_relative_gap_each_split"]
                    and abs(positive_rate_gaps["block_confirmation"])
                    >= rule["minimum_absolute_positive_rate_gap_each_split"]
                    and abs(gaps["block_confirmation"])
                    >= rule["confirmation_to_discovery_absolute_gap_ratio_min"]
                    * abs(gaps["block_discovery"])
                )
                report = {
                    "model": model,
                    "semantic_role": role,
                    "start_band": start_band,
                    "end_band": end_band,
                    "band_width": end_band - start_band + 1,
                    "pool_metrics": pool_reports,
                    "correct_minus_confusion_mean_gap": gaps,
                    "relative_mean_gap": relative_gaps,
                    "positive_rate_gap": positive_rate_gaps,
                    "same_nonzero_gap_sign": same_gap_sign,
                    "positive_rate_tracks_mean_gap": rate_tracks_mean,
                    "same_nonzero_positive_rate_gap_sign": same_rate_sign,
                    "discovery_pass": discovery_pass,
                    "independent_confirmation_pass": confirmation_pass,
                    "observer_only": True,
                    "causal": False,
                }
                comparisons.append(report)
                if confirmation_pass:
                    candidates.append(report)
    priority = {role: index for index, role in enumerate(role_priority)}
    candidates.sort(key=lambda row: (
        row["end_band"],
        row["band_width"],
        row["start_band"],
        -min(abs(value) for value in row["relative_mean_gap"].values()),
        priority[row["semantic_role"]],
    ))
    selected = None
    if candidates:
        chosen = candidates[0]
        start_layer, end_layer = layer_bounds(
            chosen["start_band"], chosen["end_band"], layer_count, band_count
        )
        wrong_start_band, wrong_end_band = wrong_interval(
            chosen["start_band"], chosen["end_band"], band_count
        )
        wrong_start_layer, wrong_end_layer = layer_bounds(
            wrong_start_band, wrong_end_band, layer_count, band_count
        )
        role_index = priority[chosen["semantic_role"]]
        wrong_role = role_priority[(role_index + 1) % len(role_priority)]
        selected = {
            **chosen,
            "layer_count": layer_count,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "wrong_start_band": wrong_start_band,
            "wrong_end_band": wrong_end_band,
            "wrong_start_layer": wrong_start_layer,
            "wrong_end_layer": wrong_end_layer,
            "wrong_role_control": wrong_role,
            "selection_frozen_before_causal_execution": True,
        }
    answer_candidates = [
        row for row in candidates if row["semantic_role"] == "answer_boundary"
    ]
    return {
        "model": model,
        "layer_count": layer_count,
        "trace_case_count": summary["case_count"],
        "case_count_per_pool_phenotype": summary["case_count_per_pool_phenotype"],
        "comparison_count": len(comparisons),
        "confirmed_candidate_count": len(candidates),
        "first_32_confirmed_candidates": candidates[:32],
        "first_16_confirmed_answer_boundary_candidates": answer_candidates[:16],
        "selected_continuous_block": selected,
        "authorized_for_coarse_block_causal": bool(
            selected and behavior["qualified_for_coarse_block_causal"]
        ),
        "full_vectors_persisted": False,
        "causal_intervention_executed": False,
        "sealed_split_read": False,
    }


def analyze() -> dict[str, Any]:
    frozen = read_json(PROTOCOL_PATH)
    reports = [analyze_model(model, frozen) for model in MODELS]
    topology_groups: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for report in reports:
        selected = report["selected_continuous_block"]
        if selected:
            topology_groups[(
                selected["semantic_role"], selected["start_band"], selected["end_band"]
            )].append(report["model"])
    shared = [
        {
            "semantic_role": key[0],
            "start_band": key[1],
            "end_band": key[2],
            "models": sorted(models),
            "model_count": len(models),
        }
        for key, models in sorted(topology_groups.items()) if len(models) >= 2
    ]
    selected_by_model = {
        report["model"]: report["selected_continuous_block"] for report in reports
    }
    authorized_models = [
        report["model"] for report in reports
        if report["authorized_for_coarse_block_causal"]
    ]
    analysis = {
        "schema_version": "phase571_continuous_block_analysis.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "status": "complete",
        "block_discovery_rule": frozen["block_discovery_rule"],
        "model_reports": reports,
        "shared_selected_topologies": shared,
        "authorized_models_for_coarse_block_causal": authorized_models,
        "authorized_model_count": len(authorized_models),
        "observer_only": True,
        "causal_intervention_executed": False,
        "closure_claimed": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, analysis)
    registry = {
        "schema_version": "phase571_continuous_block_registry.v1",
        "phase_id": "Phase571",
        "created_at": now(),
        "analysis_sha256": sha256_file(ANALYSIS_PATH),
        "selected_block_by_model": selected_by_model,
        "authorized_models": authorized_models,
        "coarse_block_conditions": frozen["coarse_block_conditions"],
        "coarse_block_gate": frozen["coarse_block_gate"],
        "donor_stage_authorized": False,
        "causal_execution_required_before_donor_stage": True,
        "sealed_split_read": False,
    }
    write_json(REGISTRY_PATH, registry)
    print(json.dumps({
        "authorized_models": authorized_models,
        "shared_selected_topologies": shared,
        "models": [
            {
                "model": report["model"],
                "candidate_count": report["confirmed_candidate_count"],
                "selected": report["selected_continuous_block"],
            }
            for report in reports
        ],
    }, ensure_ascii=False, indent=2))
    return analysis


if __name__ == "__main__":
    analyze()
