#!/usr/bin/env python3
"""Independent audit for the Phase1152 tie-aware library."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1152_tie_aware_morphology_library"
MAIN_SCRIPT = ROOT / "tests/glm5/phase1152_tie_aware_morphology_library.py"
AUDIT_SCRIPT = Path(__file__).resolve()
EXPECTED_DIMS = {
    "raw_coordinates": 2400,
    "state_gram": 2325,
    "single_site_patch": 40,
    "pairwise_patch": 160,
    "factorial_interaction": 60,
    "exhaustive_coalition": 92,
    "functional_tomography": 312,
}


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

    body = dict(protocol)
    stored_protocol_digest = body.pop("protocol_digest")
    add("protocol_digest", digest(body) == stored_protocol_digest)
    add("script_hash", sha256_file(MAIN_SCRIPT) == protocol["script_sha256"])
    add("protocol_checks", all(protocol["checks"].values()), protocol["checks"])
    add("top1_fields_removed", protocol["projection"]["single_removed_columns"] == [3, 9] and protocol["projection"]["pair_removed_columns"] == [3, 9])
    add("row_recanonicalized", bool(protocol["projection"]["row_recanonicalization"]))

    split_ids: dict[str, set[str]] = {}
    scientific_pass = {}
    for split in ("discovery", "confirmation"):
        root = OUT_ROOT / "runs" / split
        summary = read_json(root / "summary.json")
        public = read_jsonl(root / "public_manifest.jsonl")
        truth = read_jsonl(root / "sealed_truth.jsonl")
        diagnostics = read_jsonl(root / "diagnostics.jsonl")
        with np.load(root / "feature_pack.npz") as pack:
            arrays = {name: np.asarray(pack[name]) for name in pack.files}
        add(f"{split}.feature_hash", sha256_file(root / "feature_pack.npz") == summary["feature_pack_sha256"])
        add(f"{split}.public_hash", sha256_file(root / "public_manifest.jsonl") == summary["public_manifest_sha256"])
        add(f"{split}.truth_hash", sha256_file(root / "sealed_truth.jsonl") == summary["sealed_truth_sha256"])
        add(f"{split}.diagnostic_hash", sha256_file(root / "diagnostics.jsonl") == summary["diagnostics_sha256"])
        add(f"{split}.count", len(public) == len(truth) == len(diagnostics) == 128)
        add(f"{split}.blind_public", all("mechanism" not in row and "functional_group" not in row and "chart" not in row and "tail" not in row for row in public))
        add(f"{split}.aligned", all(public[index]["unit_id"] == truth[index]["unit_id"] == diagnostics[index]["unit_id"] and public[index]["index"] == truth[index]["index"] == diagnostics[index]["index"] == index for index in range(len(public))))
        add(f"{split}.unique", len({row["unit_id"] for row in public}) == len(public))
        split_ids[split] = {row["unit_id"] for row in public}

        for mechanism in protocol["design"]["mechanisms"]:
            value = sum(row["mechanism"] == mechanism for row in truth)
            add(f"{split}.mechanism.{mechanism}", value == 16, value)

        for name, expected_dim in EXPECTED_DIMS.items():
            array = arrays[name]
            add(f"{split}.shape.{name}", list(array.shape) == [128, expected_dim], list(array.shape))
            add(f"{split}.finite.{name}", bool(np.isfinite(array).all()))
            add(f"{split}.nonempty.{name}", bool(np.any(np.abs(array) > 0)))
        add(f"{split}.tail_shape", list(arrays["tail_ratio"].shape) == [128])
        add(f"{split}.clean_shape", list(arrays["clean_accuracy"].shape) == [128])

        functional = arrays["functional_tomography"].astype(np.float64)
        by_key = {(row["mechanism"], row["replicate"], row["tail"], row["chart"]): int(row["index"]) for row in truth}
        chart_values = []
        for mechanism in protocol["design"]["mechanisms"]:
            for replicate in range(int(protocol["design"]["base_replicates"])):
                for tail in protocol["design"]["tails"]:
                    left = by_key[(mechanism, replicate, tail, "identity")]
                    right = by_key[(mechanism, replicate, tail, "rotated")]
                    value = cosine(functional[left], functional[right])
                    chart_values.append(value)
                    add(f"{split}.chart.{mechanism}.{replicate}.{tail}", math.isfinite(value), value)
        equivalence_values = []
        for left_name, right_name in protocol["design"]["equivalent_pairs"]:
            for replicate in range(int(protocol["design"]["base_replicates"])):
                for tail in protocol["design"]["tails"]:
                    for chart in protocol["design"]["charts"]:
                        left = by_key[(left_name, replicate, tail, chart)]
                        right = by_key[(right_name, replicate, tail, chart)]
                        value = cosine(functional[left], functional[right])
                        equivalence_values.append(value)
                        add(f"{split}.equiv.{left_name}.{right_name}.{replicate}.{tail}.{chart}", math.isfinite(value), value)
        stable_indices = [row["index"] for row in truth if row["tail"] == "stable"]
        degraded_indices = [row["index"] for row in truth if row["tail"] == "degraded"]
        recomputed = {
            "finite_fraction": float(np.mean([np.isfinite(value).mean() for value in arrays.values()])),
            "clean_accuracy_min": float(np.min(arrays["clean_accuracy"])),
            "stable_tail_ratio_min": float(np.min(arrays["tail_ratio"][stable_indices])),
            "degraded_tail_ratio_max": float(np.max(arrays["tail_ratio"][degraded_indices])),
            "functional_chart_cosine_min": float(min(chart_values)),
            "functional_equivalence_cosine_min": float(min(equivalence_values)),
        }
        for key, value in recomputed.items():
            add(f"{split}.summary.{key}", math.isclose(float(summary[key]), value, rel_tol=0.0, abs_tol=1e-7), {"stored": summary[key], "recomputed": value})
        summary_body = dict(summary)
        stored_summary_digest = summary_body.pop("summary_digest")
        add(f"{split}.summary_digest", digest(summary_body) == stored_summary_digest)
        expected_science = all(summary["checks"].values())
        add(f"{split}.science_consistent", bool(summary["all_checks_passed"]) == expected_science, summary["checks"])
        scientific_pass[split] = expected_science

    overlap = len(split_ids["discovery"] & split_ids["confirmation"])
    add("split_disjoint", overlap == 0, overlap)
    expected_final = bool(scientific_pass["discovery"] and scientific_pass["confirmation"] and overlap == 0)
    add("final_qualified_consistent", bool(final["library_qualified"]) == expected_final)
    add("phase1153_authorization_consistent", bool(final["phase1153_blind_coverage_authorized"]) == expected_final)
    add("pretrained_scan_denied", not bool(final["pretrained_model_scan_authorized"]))
    final_body = dict(final)
    stored_final_digest = final_body.pop("final_digest")
    add("final_digest", digest(final_body) == stored_final_digest)

    passed_count = sum(row["passed"] for row in checks)
    result = {
        "phase": 1152,
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": passed_count,
        "all_checks_passed": passed_count == len(checks),
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    path = OUT_ROOT / "audit/independent_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({key: result[key] for key in ("check_count", "passed_count", "all_checks_passed", "audit_digest")}))
    if not result["all_checks_passed"]:
        raise SystemExit([row for row in checks if not row["passed"]][:10])


if __name__ == "__main__":
    main()
