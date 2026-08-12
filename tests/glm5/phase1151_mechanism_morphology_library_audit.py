#!/usr/bin/env python3
"""Independent artifact audit for Phase1151."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1151_mechanism_morphology_library"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1151_mechanism_morphology_library.py"
AUDIT_SCRIPT = Path(__file__).resolve()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    protocol_body = dict(protocol)
    protocol_digest = protocol_body.pop("protocol_digest")
    add("protocol_digest", digest(protocol_body) == protocol_digest)
    add("main_script_hash", sha256_file(MAIN_SCRIPT) == protocol["script_sha256"])
    add("protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add("algorithm_selection_forbidden", bool(protocol["checks"]["algorithm_selection_forbidden"]))
    add("confirmation_labels_sealed", bool(protocol["checks"]["confirmation_labels_sealed"]))
    add("equivalence_pairs_declared", len(protocol["design"]["equivalent_pairs"]) == 2)

    split_ids: dict[str, set[str]] = {}
    recomputed_summaries = {}
    for split in ("discovery", "confirmation"):
        run_root = OUT_ROOT / "runs" / split
        summary = read_json(run_root / "summary.json")
        public = read_jsonl(run_root / "public_manifest.jsonl")
        truth = read_jsonl(run_root / "sealed_truth.jsonl")
        diagnostics = read_jsonl(run_root / "diagnostics.jsonl")
        with np.load(run_root / "feature_pack.npz") as pack:
            arrays = {name: np.asarray(pack[name]) for name in pack.files}

        add(f"{split}.feature_hash", sha256_file(run_root / "feature_pack.npz") == summary["feature_pack_sha256"])
        add(f"{split}.public_hash", sha256_file(run_root / "public_manifest.jsonl") == summary["public_manifest_sha256"])
        add(f"{split}.truth_hash", sha256_file(run_root / "sealed_truth.jsonl") == summary["sealed_truth_sha256"])
        add(f"{split}.diagnostics_hash", sha256_file(run_root / "diagnostics.jsonl") == summary["diagnostics_sha256"])
        add(f"{split}.unit_count", len(public) == len(truth) == len(diagnostics) == summary["unit_count"])
        add(f"{split}.expected_unit_count", len(public) == int(protocol["design"]["units_per_split"]))
        add(f"{split}.public_is_blind", all("mechanism" not in row and "functional_group" not in row and "tail" not in row and "chart" not in row for row in public))
        add(f"{split}.indices", all(row["index"] == index for index, row in enumerate(public)))
        add(f"{split}.truth_alignment", all(public[index]["unit_id"] == row["unit_id"] and row["index"] == index for index, row in enumerate(truth)))
        add(f"{split}.diagnostic_alignment", all(public[index]["unit_id"] == row["unit_id"] and row["index"] == index for index, row in enumerate(diagnostics)))
        add(f"{split}.unique_ids", len({row["unit_id"] for row in public}) == len(public))
        split_ids[split] = {row["unit_id"] for row in public}

        for mechanism in protocol["design"]["mechanisms"]:
            count = sum(row["mechanism"] == mechanism for row in truth)
            add(f"{split}.count.{mechanism}", count == 12, count)
        for group in sorted(set(protocol["design"]["functional_group"].values())):
            count = sum(row["functional_group"] == group for row in truth)
            expected = 24 if group in {"single_joint_carrier", "factorized_roles"} else 12
            add(f"{split}.group_count.{group}", count == expected, count)

        for name, array in arrays.items():
            add(f"{split}.finite.{name}", bool(np.isfinite(array).all()))
            add(f"{split}.rows.{name}", int(array.shape[0]) == len(public), list(array.shape))
            add(f"{split}.shape.{name}", list(array.shape) == summary["feature_shapes"][name])

        clean = np.asarray(arrays["clean_accuracy"], dtype=np.float64)
        tail_ratio = np.asarray(arrays["tail_ratio"], dtype=np.float64)
        add(f"{split}.clean_finite", bool(np.isfinite(clean).all()), float(np.min(clean)))
        stable_indices = [row["index"] for row in truth if row["tail"] == "stable"]
        degraded_indices = [row["index"] for row in truth if row["tail"] == "degraded"]
        stable_min = float(np.min(tail_ratio[stable_indices]))
        degraded_max = float(np.max(tail_ratio[degraded_indices]))
        add(f"{split}.stable_tail_finite", math.isfinite(stable_min), stable_min)
        add(f"{split}.degraded_tail_finite", math.isfinite(degraded_max), degraded_max)

        functional = np.asarray(arrays["functional_tomography"], dtype=np.float64)
        by_key = {(row["mechanism"], row["replicate"], row["tail"], row["chart"]): int(row["index"]) for row in truth}
        chart_cosines = []
        for mechanism in protocol["design"]["mechanisms"]:
            for replicate in range(int(protocol["design"]["base_replicates"])):
                for tail in protocol["design"]["tails"]:
                    left = by_key[(mechanism, replicate, tail, "identity")]
                    right = by_key[(mechanism, replicate, tail, "rotated")]
                    value = cosine(functional[left], functional[right])
                    chart_cosines.append(value)
                    add(
                        f"{split}.chart.{mechanism}.{replicate}.{tail}",
                        math.isfinite(value),
                        {
                            "value": value,
                            "threshold_pass": value
                            >= protocol["thresholds"]["functional_chart_cosine_min"],
                        },
                    )
        equivalence_cosines = []
        for left_name, right_name in protocol["design"]["equivalent_pairs"]:
            for replicate in range(int(protocol["design"]["base_replicates"])):
                for tail in protocol["design"]["tails"]:
                    for chart in protocol["design"]["charts"]:
                        left = by_key[(left_name, replicate, tail, chart)]
                        right = by_key[(right_name, replicate, tail, chart)]
                        value = cosine(functional[left], functional[right])
                        equivalence_cosines.append(value)
                        add(
                            f"{split}.equiv.{left_name}.{right_name}.{replicate}.{tail}.{chart}",
                            math.isfinite(value),
                            {
                                "value": value,
                                "threshold_pass": value
                                >= protocol["thresholds"]["functional_equivalence_cosine_min"],
                            },
                        )

        finite_fraction = float(np.mean([np.isfinite(value).mean() for value in arrays.values()]))
        recomputed = {
            "finite_fraction": finite_fraction,
            "clean_accuracy_min": float(np.min(clean)),
            "stable_tail_ratio_min": stable_min,
            "degraded_tail_ratio_max": degraded_max,
            "functional_chart_cosine_min": float(min(chart_cosines)),
            "functional_equivalence_cosine_min": float(min(equivalence_cosines)),
        }
        recomputed_summaries[split] = recomputed
        for key, value in recomputed.items():
            add(f"{split}.summary.{key}", math.isclose(float(summary[key]), value, rel_tol=0.0, abs_tol=1e-7), {"stored": summary[key], "recomputed": value})
        summary_body = dict(summary)
        stored_summary_digest = summary_body.pop("summary_digest")
        add(f"{split}.summary_digest", digest(summary_body) == stored_summary_digest)
        add(
            f"{split}.all_checks_consistent",
            bool(summary["all_checks_passed"]) == all(summary["checks"].values()),
            summary["checks"],
        )

    overlap = len(split_ids["discovery"] & split_ids["confirmation"])
    add("split_disjoint", overlap == 0, overlap)
    add("final_protocol", final["protocol_digest"] == protocol["protocol_digest"])
    add("final_split_overlap", int(final["split_overlap"]) == overlap)
    expected_qualified = bool(
        read_json(OUT_ROOT / "runs/discovery/summary.json")["all_checks_passed"]
        and read_json(OUT_ROOT / "runs/confirmation/summary.json")["all_checks_passed"]
        and overlap <= protocol["thresholds"]["split_overlap_max"]
    )
    add(
        "final_library_qualified_consistent",
        bool(final["library_qualified"]) == expected_qualified,
        {"stored": final["library_qualified"], "expected": expected_qualified},
    )
    add(
        "phase1152_authorization_consistent",
        bool(final["phase1152_blind_coverage_authorized"]) == expected_qualified,
    )
    add("pretrained_scan_denied", not bool(final["pretrained_model_scan_authorized"]))
    final_body = dict(final)
    stored_final_digest = final_body.pop("final_digest")
    add("final_digest", digest(final_body) == stored_final_digest)

    passed_count = sum(row["passed"] for row in checks)
    audit = {
        "phase": 1151,
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": passed_count,
        "all_checks_passed": passed_count == len(checks),
        "recomputed_summaries": recomputed_summaries,
        "checks": checks,
    }
    audit["audit_digest"] = digest(audit)
    write_path = OUT_ROOT / "audit/independent_audit.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({key: audit[key] for key in ("check_count", "passed_count", "all_checks_passed", "audit_digest")}))
    if not audit["all_checks_passed"]:
        failed = [row for row in checks if not row["passed"]]
        raise SystemExit(f"Phase1151 audit failed: {failed[:10]}")


if __name__ == "__main__":
    main()
