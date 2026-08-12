#!/usr/bin/env python3
"""Independent structural and arithmetic audit for Phase1014."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
OUTPUT_MODES = ("entity", "property", "binary")
OPERATIONS = ("F", "Q", "FQ", "E", "O", "N", "L", "I", "X")
SPLITS = ("discovery", "confirmation")
EXPECTED_PROTOCOL_DIGEST = (
    "37b5cfb4a369d68e01098739824aa46844b562cd61e259247e77a795c7328a89"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def recompute_consistency(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    flat_sums = sums.reshape(-1, sums.shape[-2], sums.shape[-1])
    flat_counts = counts.reshape(-1, counts.shape[-1])
    flat_result = result.reshape(-1, result.shape[-1])
    for index in range(flat_sums.shape[0]):
        count = flat_counts[index].astype(np.float64)
        squared = np.einsum(
            "ed,ed->e",
            flat_sums[index].astype(np.float64, copy=False),
            flat_sums[index].astype(np.float64, copy=False),
        )
        valid = count >= 2
        flat_result[index, valid] = (
            (squared[valid] - count[valid])
            / (count[valid] * (count[valid] - 1.0))
        )
    return result


def audit() -> dict[str, Any]:
    protocol_path = OUT_ROOT / "protocol" / "protocol.json"
    protocol = read_json(protocol_path)
    require(
        protocol["preregistration_digest"] == EXPECTED_PROTOCOL_DIGEST,
        "protocol digest drift",
    )
    analysis = read_json(OUT_ROOT / "analysis" / "summary.json")
    require(
        analysis["protocol_digest"] == EXPECTED_PROTOCOL_DIGEST,
        "analysis protocol digest drift",
    )
    require(
        analysis["selection_contract"]["confirmation_never_selects"],
        "confirmation selection leakage",
    )
    require(
        not analysis["selection_contract"]["weighted_mechanism_score_used"],
        "weighted mechanism score unexpectedly used",
    )

    model_audits = {}
    precision_audits = {}
    arithmetic_checks = []
    key_paths = [protocol_path, OUT_ROOT / "analysis" / "summary.json"]
    for model in MODELS:
        protocol_units = read_jsonl(
            OUT_ROOT / "protocol" / model / "units.jsonl"
        )
        protocol_cases = read_jsonl(
            OUT_ROOT / "protocol" / model / "cases.jsonl"
        )
        case_by_id = {
            row["record_id"]: row for row in protocol_cases
        }
        require(len(protocol_units) == 720, f"{model}: protocol units")
        require(len(protocol_cases) == 5760, f"{model}: protocol cases")
        require(
            all(len(row["edit_positions"]["L"]) == 1 for row in protocol_units),
            f"{model}: lexical control edit drift",
        )
        require(
            all(
                all(
                    len(case_by_id[case_id]["input_ids"])
                    == len(
                        case_by_id[
                            row["case_ids"]["base"]
                        ]["input_ids"]
                    )
                    for case_id in row["case_ids"].values()
                )
                for row in protocol_units
            ),
            f"{model}: equal-length state drift",
        )
        counterbalance = Counter(
            row["counterbalance_cell"] for row in protocol_units
        )
        require(
            counterbalance
            == Counter({
                "f0q0": 180,
                "f1q0": 180,
                "f0q1": 180,
                "f1q1": 180,
            }),
            f"{model}: counterbalance drift",
        )

        behavior = read_json(
            OUT_ROOT / "behavior" / model / "summary.json"
        )
        require(behavior["case_count"] == 5760, f"{model}: behavior cases")
        scan_root = OUT_ROOT / "scan" / model
        scan = read_json(scan_root / "summary.json")
        require(scan["formal_scan"], f"{model}: non-formal scan")
        require(scan["unit_count"] == 720, f"{model}: scan units")
        require(
            scan["singleton_forward_count"] == 6480,
            f"{model}: singleton forward count",
        )
        require(
            scan["raw_hidden_tensors_persisted"] == 0,
            f"{model}: raw hidden tensor persistence",
        )
        require(
            scan["identity_maximum"] == 0.0,
            f"{model}: identity nonzero",
        )
        events = read_jsonl(scan_root / "events.jsonl")
        require(
            [row["event_index"] for row in events]
            == list(range(len(events))),
            f"{model}: event index drift",
        )

        panel_count = 0
        panel_unit_count = 0
        scalar_count = 0
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel_root = scan_root / family / output_mode
                panel = read_json(panel_root / "summary.json")
                units = read_jsonl(panel_root / "units.jsonl")
                require(len(units) == 48, f"{model}: panel unit count")
                require(
                    Counter(row["split"] for row in units)
                    == Counter({"discovery": 24, "confirmation": 24}),
                    f"{model}: panel split balance",
                )
                for split in SPLITS:
                    require(
                        Counter(
                            row["counterbalance_cell"]
                            for row in units
                            if row["split"] == split
                        )
                        == Counter({
                            "f0q0": 6,
                            "f1q0": 6,
                            "f0q1": 6,
                            "f1q1": 6,
                        }),
                        f"{model}: panel counterbalance {split}",
                    )
                scalar = np.load(panel_root / "response_scalars.npz")
                raw = scalar["raw_magnitude"]
                normalized = scalar["normalized_magnitude"]
                require(
                    raw.shape == (48, len(OPERATIONS), len(events)),
                    f"{model}: raw shape",
                )
                require(
                    normalized.shape == raw.shape,
                    f"{model}: normalized shape",
                )
                require(np.isfinite(raw).all(), f"{model}: raw nonfinite")
                require(
                    np.isfinite(normalized).all(),
                    f"{model}: normalized nonfinite",
                )
                require(
                    float(np.max(np.abs(raw[:, OPERATIONS.index("I")])))
                    == 0.0,
                    f"{model}: identity scalar drift",
                )
                scalar.close()

                direction = np.load(
                    panel_root / "direction_consistency.npz"
                )
                consistency = direction["direction_consistency"]
                counts = direction["direction_count"]
                require(
                    consistency.shape
                    == (2, 3, 2, 2, len(events)),
                    f"{model}: direction shape",
                )
                require(
                    counts.shape == (3, 2, 2, len(events)),
                    f"{model}: count shape",
                )
                finite_values = consistency[np.isfinite(consistency)]
                require(
                    np.all(finite_values >= -1.00001)
                    and np.all(finite_values <= 1.00001),
                    f"{model}: direction bounds",
                )

                sums = np.load(
                    panel_root / "canonical_direction_sums.npz"
                )
                whole_recomputed = recompute_consistency(
                    sums["whole"], sums["whole_count"]
                )
                head_recomputed = recompute_consistency(
                    sums["head"], sums["head_count"]
                )
                recomputed = np.concatenate(
                    (whole_recomputed, head_recomputed), axis=-1
                )
                saved = consistency[
                    1,
                    0,
                ].astype(np.float64, copy=False)
                require(
                    np.allclose(
                        recomputed,
                        saved,
                        rtol=1e-5,
                        atol=1e-5,
                        equal_nan=True,
                    ),
                    f"{model}: canonical arithmetic mismatch",
                )
                arithmetic_checks.append({
                    "model": model,
                    "family": family,
                    "output_mode": output_mode,
                    "event_count": len(events),
                    "canonical_all_units_recomputed": True,
                })
                sums.close()
                direction.close()
                panel_count += 1
                panel_unit_count += panel["unit_count"]
                scalar_count += panel["scalar_measurement_count"]
                key_paths.extend([
                    panel_root / "summary.json",
                    panel_root / "response_scalars.npz",
                    panel_root / "direction_consistency.npz",
                    panel_root / "canonical_direction_sums.npz",
                ])
        require(panel_count == 15, f"{model}: panel count")
        require(panel_unit_count == 720, f"{model}: panel unit total")
        require(
            scalar_count == scan["scalar_measurement_count"],
            f"{model}: scalar count mismatch",
        )
        model_analysis = analysis["models"][model]
        require(
            model_analysis["unit_count"] == 720,
            f"{model}: analysis unit count",
        )
        model_audits[model] = {
            "protocol_unit_count": len(protocol_units),
            "protocol_case_count": len(protocol_cases),
            "behavior_case_count": behavior["case_count"],
            "scan_event_count": len(events),
            "scan_panel_count": panel_count,
            "singleton_forward_count": scan[
                "singleton_forward_count"
            ],
            "identity_maximum": scan["identity_maximum"],
            "arithmetic_panels_recomputed": 15,
        }
        key_paths.extend([
            OUT_ROOT / "behavior" / model / "summary.json",
            scan_root / "summary.json",
            scan_root / "events.jsonl",
        ])

    precision_protocol_path = (
        OUT_ROOT / "precision_protocol" / "protocol.json"
    )
    precision_protocol = read_json(precision_protocol_path)
    require(
        not precision_protocol["selection_used_confirmation"],
        "precision protocol confirmation leakage",
    )
    precision_summary_path = (
        OUT_ROOT / "precision_bf16" / "summary.json"
    )
    precision_summary = read_json(precision_summary_path)
    require(
        precision_summary["precision_protocol_digest"]
        == precision_protocol["precision_protocol_digest"],
        "precision summary digest drift",
    )
    for model in MODELS:
        selections = read_jsonl(
            OUT_ROOT
            / "precision_protocol"
            / model
            / "events.jsonl"
        )
        units = read_jsonl(
            OUT_ROOT
            / "precision_protocol"
            / model
            / "units.jsonl"
        )
        require(
            len(selections) <= 8,
            f"{model}: precision event cap",
        )
        require(
            all(
                not row["selection_used_confirmation"]
                for row in selections
            ),
            f"{model}: precision selection leakage",
        )
        require(len(units) == 120, f"{model}: precision units")
        require(
            Counter(int(row["world_index"]) for row in units)
            == Counter({0: 30, 1: 30, 2: 30, 3: 30}),
            f"{model}: precision world balance",
        )
        require(
            all(row["split"] == "confirmation" for row in units),
            f"{model}: precision split drift",
        )
        precision_model_root = OUT_ROOT / "precision_bf16" / model
        model_summary = read_json(
            precision_model_root / "summary.json"
        )
        candidate_rows = read_jsonl(
            precision_model_root / "candidate_summary.jsonl"
        )
        cells = read_jsonl(precision_model_root / "cells.jsonl")
        unit_metrics = read_jsonl(
            precision_model_root / "unit_metrics.jsonl"
        )
        require(
            model_summary["source_precision_protocol_digest"]
            == precision_protocol["precision_protocol_digest"],
            f"{model}: precision digest drift",
        )
        require(
            model_summary["selected_event_count"] == len(selections),
            f"{model}: precision event count",
        )
        require(
            model_summary["singleton_forward_count"]
            == len(units) * len(model_summary["state_forward_order"]),
            f"{model}: precision forward count",
        )
        require(
            model_summary["identity_maximum"] == 0.0,
            f"{model}: BF16 identity drift",
        )
        require(
            len(candidate_rows) == len(selections),
            f"{model}: precision candidate rows",
        )
        require(
            len(cells) == len(selections) * 15,
            f"{model}: precision cell rows",
        )
        require(
            len(unit_metrics) == len(selections) * len(units),
            f"{model}: precision unit metric rows",
        )
        require(
            all(
                row[
                    "eight_bit_bf16_median_direction_cosine"
                ] is None
                or -1.00001
                <= row[
                    "eight_bit_bf16_median_direction_cosine"
                ]
                <= 1.00001
                for row in candidate_rows
            ),
            f"{model}: precision cosine bounds",
        )
        precision_audits[model] = {
            "selected_event_count": len(selections),
            "confirmation_unit_count": len(units),
            "singleton_forward_count": model_summary[
                "singleton_forward_count"
            ],
            "cell_count": len(cells),
            "identity_maximum": model_summary["identity_maximum"],
            "precision_supported_event_count": model_summary[
                "precision_supported_event_count"
            ],
        }
        key_paths.extend([
            OUT_ROOT / "precision_protocol" / model / "events.jsonl",
            OUT_ROOT / "precision_protocol" / model / "units.jsonl",
            precision_model_root / "summary.json",
            precision_model_root / "candidate_summary.jsonl",
            precision_model_root / "cells.jsonl",
            precision_model_root / "unit_metrics.jsonl",
        ])
    require(
        precision_summary["selected_event_count"]
        == sum(
            value["selected_event_count"]
            for value in precision_audits.values()
        ),
        "precision aggregate selected count",
    )
    key_paths.extend([precision_protocol_path, precision_summary_path])

    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
        for path in key_paths
    }
    result = {
        "schema_version": "phase1014_result_audit.v1",
        "phase": 1014,
        "passed": True,
        "protocol_digest": EXPECTED_PROTOCOL_DIGEST,
        "model_audits": model_audits,
        "precision_audits": precision_audits,
        "arithmetic_check_count": len(arithmetic_checks),
        "arithmetic_checks": arithmetic_checks,
        "key_file_sha256": hashes,
        "claim_limit": (
            "audit establishes data integrity and arithmetic "
            "reproducibility, not mechanism validity"
        ),
    }
    path = OUT_ROOT / "audit" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    audit()
