#!/usr/bin/env python3
"""Independent result audit for Phase1139."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1139_matched_path_residual_interpolation"
MODELS = ("qwen3_4b", "qwen3_14b")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_CURVES = 78
EXPECTED_RECORDS = 390
EPSILON = 1e-8


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def median(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def independently_summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["curve_id"])].append(row)
    identity = []
    main = []
    controls = []
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        margins = [row["oriented_margin"] for row in ordered]
        if ordered[0]["identity_margin_drift"] is not None:
            identity.append(abs(float(ordered[0]["identity_margin_drift"])))
        finite = all(row["finite"] and value is not None for row, value in zip(ordered, margins))
        if not finite:
            continue
        base = float(margins[0])
        endpoint = float(margins[-1])
        span = endpoint - base
        if ordered[0]["curve_kind"] == "same_answer_temporal_control":
            controls.append(abs(span))
            continue
        valid = base < 0.0 and endpoint > 0.0 and span > EPSILON
        if not valid:
            continue
        normalized = [(float(value) - base) / span for value in margins]
        deviations = [abs(value - alpha) for value, alpha in zip(normalized[1:-1], ALPHAS[1:-1])]
        steps = [normalized[index + 1] - normalized[index] for index in range(4)]
        max_index = max(range(4), key=lambda index: steps[index])
        main.append({
            "span": span,
            "deviation": statistics.median(deviations),
            "max_step": steps[max_index],
            "monotonic": sum(step >= -0.02 for step in steps) / 4,
            "max_step_end": ALPHAS[max_index + 1],
        })
    intervals = Counter(row["max_step_end"] for row in main)
    dominant = sorted(intervals.items(), key=lambda pair: (-pair[1], pair[0]))[0][0] if intervals else None
    main_span = median(row["span"] for row in main)
    control_span = median(controls)
    return {
        "curve_count": len(grouped),
        "identity_max": max(identity) if identity else None,
        "valid_main_count": len(main),
        "main_span": main_span,
        "control_span": control_span,
        "span_ratio": main_span / max(control_span, EPSILON) if main_span is not None and control_span is not None else None,
        "linear_deviation": median(row["deviation"] for row in main),
        "max_step": median(row["max_step"] for row in main),
        "monotonic": median(row["monotonic"] for row in main),
        "dominant_interval": dominant,
    }


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks: dict[str, bool] = {
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_matches_audit": protocol_audit["protocol_digest"] == prereg["protocol_digest"],
        "selected_fraction_frozen_0_7": prereg["selection"]["selected_requested_fraction"] == 0.7,
        "confirmation_cohort_13": prereg["cohort"]["item_count"] == 13,
        "alpha_grid_exact": tuple(prereg["intervention"]["alphas"]) == ALPHAS,
        "phase1138_not_reopened": final["phase1138_reopened"] is False,
        "attractor_claim_denied": final["attractor_claim_authorized"] is False,
        "component_scan_denied": final["component_scan_authorized"] is False,
        "auto_continue_false": final["auto_continue"] is False,
    }
    recomputed = {}
    for model_name in MODELS:
        records = read_jsonl(OUT_ROOT / "runs" / model_name / "records.jsonl")
        summary = read_json(OUT_ROOT / "runs" / model_name / "summary.json")
        metrics = final["models"][model_name]
        independent = independently_summarize(records)
        recomputed[model_name] = independent
        prefix = model_name
        checks[f"{prefix}_record_count"] = len(records) == EXPECTED_RECORDS == summary["record_count"]
        checks[f"{prefix}_curve_count"] = independent["curve_count"] == EXPECTED_CURVES == summary["curve_count"]
        checks[f"{prefix}_record_digest"] = summary["record_digest"] == digest(records)
        checks[f"{prefix}_summary_protocol_digest"] = summary["protocol_digest"] == prereg["protocol_digest"]
        checks[f"{prefix}_alphas_complete"] = all(
            sorted(float(row["alpha"]) for row in records if row["curve_id"] == curve_id) == list(ALPHAS)
            for curve_id in {str(row["curve_id"]) for row in records}
        )
        checks[f"{prefix}_all_finite"] = all(bool(row["finite"]) for row in records)
        checks[f"{prefix}_identity_recomputed"] = math.isclose(
            float(independent["identity_max"]),
            float(metrics["identity_max_abs_margin_drift"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        checks[f"{prefix}_valid_main_recomputed"] = independent["valid_main_count"] == round(
            float(metrics["main_valid_fraction"]) * int(metrics["main_curve_count"])
        )
        checks[f"{prefix}_deviation_recomputed"] = math.isclose(
            float(independent["linear_deviation"]),
            float(metrics["median_linear_deviation"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        checks[f"{prefix}_max_step_recomputed"] = math.isclose(
            float(independent["max_step"]),
            float(metrics["median_max_step"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        checks[f"{prefix}_span_ratio_recomputed"] = math.isclose(
            float(independent["span_ratio"]),
            float(metrics["main_to_same_answer_span_ratio"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        checks[f"{prefix}_dominant_interval_recomputed"] = (
            independent["dominant_interval"] == metrics["dominant_max_step_end_alpha"]
        )

    core = {
        "schema_version": "phase1139_independent_result_audit.v1",
        "phase": 1139,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "independent_recomputation": recomputed,
    }
    audit = dict(core)
    audit["audit_digest"] = digest(core)
    write_json(OUT_ROOT / "audit/independent_result_audit.json", audit)
    if not audit["all_checks_passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Phase1139 audit failed: {failed}")
    print(json.dumps({
        "phase": 1139,
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "all_checks_passed": True,
        "audit_digest": audit["audit_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
