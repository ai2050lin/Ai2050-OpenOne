#!/usr/bin/env python3
"""Independent artifact and arithmetic audit for Phase1119."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import phase1119_qwen3_scale_protocol as protocol


def pair_panel(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[row["pair_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for pair_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["sense"])
        if len(rows) != 2 or [row["sense"] for row in rows] != [0, 1]:
            raise RuntimeError(f"malformed pair: {pair_id}")
        finite = all(row["finite"] for row in rows)
        pairs.append(
            {
                "concept_id": rows[0]["concept_id"],
                "split": rows[0]["split"],
                "template": rows[0]["template"],
                "finite": finite,
                "true_d": rows[0]["true_z"] - rows[1]["true_z"] if finite else None,
                "control_d": rows[0]["control_z"] - rows[1]["control_z"] if finite else None,
                "bidirectional": finite and rows[0]["true_z"] > 0.0 and rows[1]["true_z"] < 0.0,
            }
        )
    return pairs


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    finite = [row for row in rows if row["finite"]]
    count = max(len(finite), 1)
    return {
        "finite_fraction": len(finite) / max(len(rows), 1),
        "direction_accuracy": sum(row["true_d"] > 0.0 for row in finite) / count,
        "control_direction_accuracy": sum(row["control_d"] > 0.0 for row in finite) / count,
        "control_advantage": (
            sum(row["true_d"] > 0.0 for row in finite)
            - sum(row["control_d"] > 0.0 for row in finite)
        )
        / count,
        "bidirectional_accuracy": sum(row["bidirectional"] for row in finite) / count,
    }


def concept_fraction(pairs: list[dict[str, Any]]) -> tuple[float, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["concept_id"]].append(row)
    positive: dict[str, bool] = {}
    splits: dict[str, str] = {}
    for concept_id, rows in grouped.items():
        values = [row["true_d"] for row in rows if row["finite"]]
        positive[concept_id] = bool(values) and statistics.median(values) > 0.0
        splits[concept_id] = rows[0]["split"]
    by_split = {
        split: sum(positive[key] for key in positive if splits[key] == split)
        for split in protocol.SPLITS
    }
    return sum(positive.values()) / max(len(positive), 1), by_split


def close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-12


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    cases = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")

    prereg_core = dict(prereg)
    prereg_digest = prereg_core.pop("protocol_digest")
    final_core = dict(final)
    final_digest = final_core.pop("final_digest")
    checks: dict[str, bool] = {
        "protocol_digest": protocol.digest(prereg_core) == prereg_digest,
        "protocol_audit": protocol_audit["all_checks_passed"] is True,
        "case_digest": protocol.digest(cases) == prereg["case_digest"],
        "case_count": len(cases) == 684,
        "final_digest": protocol.digest(final_core) == final_digest,
        "final_protocol_link": final["protocol_digest"] == prereg_digest,
        "hidden_not_authorized": final["hidden_or_causal_authorized"] is False,
    }
    case_by_index = {row["case_index"]: row for row in cases}
    recomputed: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for model_name in prereg["models"]:
        root = protocol.OUT_ROOT / "behavior" / model_name
        details = list(protocol.read_jsonl(root / "candidate_detail.jsonl"))
        summary = protocol.read_json(root / "summary.json")
        summary_core = dict(summary)
        summary_digest = summary_core.pop("summary_digest")
        detail_by_index = {row["case_index"]: row for row in details}
        prefix = model_name
        checks[f"{prefix}_summary_digest"] = protocol.digest(summary_core) == summary_digest
        checks[f"{prefix}_detail_digest"] = protocol.digest(details) == summary["detail_digest"]
        checks[f"{prefix}_counts"] = len(details) == len(detail_by_index) == len(cases) == 684
        checks[f"{prefix}_indices"] = set(detail_by_index) == set(case_by_index)
        checks[f"{prefix}_links"] = all(
            detail_by_index[index]["record_id"] == case["record_id"]
            and detail_by_index[index]["pair_id"] == case["pair_id"]
            and detail_by_index[index]["sense"] == case["sense"]
            for index, case in case_by_index.items()
        )
        checks[f"{prefix}_score_arithmetic"] = all(
            (not row["finite"])
            or (
                close(row["true_z"], row["true_scores"][0] - row["true_scores"][1])
                and close(
                    row["control_z"],
                    row["control_scores"][0] - row["control_scores"][1],
                )
            )
            for row in details
        )
        checks[f"{prefix}_precision"] = (
            summary["precision"]["has_fp16_parameters"] is True
            and summary["precision"]["has_bf16_parameters"] is False
            and summary["precision"]["has_quantized_modules"] is False
            and summary["parameter_count"] == prereg["expected_parameter_counts"][model_name]
        )
        checks[f"{prefix}_finite"] = summary["finite_fraction"] >= 0.99
        pairs = pair_panel(details)
        overall = summarize(pairs)
        by_split = {
            split: summarize([row for row in pairs if row["split"] == split])
            for split in protocol.SPLITS
        }
        concept, positive_by_split = concept_fraction(pairs)
        reported = final["models"][model_name]
        checks[f"{prefix}_overall_metrics"] = all(
            close(overall[key], reported["overall"][key])
            for key in overall
        )
        checks[f"{prefix}_split_metrics"] = all(
            close(by_split[split][key], reported["by_split"][split][key])
            for split in protocol.SPLITS
            for key in by_split[split]
        )
        checks[f"{prefix}_concept_metrics"] = (
            close(concept, reported["concept_summary"]["positive_median_fraction"])
            and positive_by_split == reported["concept_summary"]["positive_by_split"]
        )
        recomputed[model_name] = {
            "overall": overall,
            "by_split": by_split,
            "concept_positive_fraction": concept,
        }
        summaries[model_name] = summary

    small = recomputed["qwen3_4b"]
    large = recomputed["qwen3_14b"]
    reported_gain = final["gains_14b_minus_4b"]
    checks["scale_gain_arithmetic"] = (
        close(
            large["overall"]["direction_accuracy"] - small["overall"]["direction_accuracy"],
            reported_gain["direction_accuracy"],
        )
        and close(
            large["overall"]["control_advantage"] - small["overall"]["control_advantage"],
            reported_gain["control_advantage"],
        )
        and close(
            large["overall"]["bidirectional_accuracy"]
            - small["overall"]["bidirectional_accuracy"],
            reported_gain["bidirectional_accuracy"],
        )
        and close(
            large["concept_positive_fraction"] - small["concept_positive_fraction"],
            reported_gain["concept_positive_fraction"],
        )
    )
    forbidden = ("hidden", "head", "neuron", "causal", "activation")
    analysis_suffixes = {".json", ".jsonl", ".npz", ".pt", ".pth"}
    generated = [
        path.name.casefold()
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file() and path.suffix.casefold() in analysis_suffixes
    ]
    allowed_hidden_names = {"final_summary.json", "preregistration.json"}
    checks["no_hidden_scan_artifacts"] = not any(
        marker in name
        for name in generated
        if name not in allowed_hidden_names
        for marker in forbidden
    )

    core = {
        "schema_version": "phase1119_qwen3_scale_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg_digest,
        "final_digest": final_digest,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(core)
    audit["audit_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
