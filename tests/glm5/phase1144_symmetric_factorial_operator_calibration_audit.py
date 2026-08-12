#!/usr/bin/env python3
"""Independent raw-array audit for Phase1144."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from phase1143_ground_truth_mechanism_calibration_audit import recompute_metrics


PHASE = 1144
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1144_symmetric_factorial_operator_calibration"
PRIMARY = ROOT / "tests/glm5/phase1144_symmetric_factorial_operator_calibration.py"
SPLITS = ("discovery", "confirmation")
N_ITEMS = 32


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close(left: Any, right: Any, tolerance: float = 2e-6) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def mapping(metadata: dict[str, Any], config_id: str, context: int, scenario: str) -> list[int] | None:
    if scenario in {"linear_shared_only", "nonlinear_shared_only"}:
        return None
    if scenario != "linear_permuted_payload":
        return list(range(N_ITEMS))
    permutation = metadata["permutations"][f"{config_id}.context{context}"]
    inverse = [0] * N_ITEMS
    for donor, payload_id in enumerate(permutation):
        inverse[int(payload_id)] = donor
    return inverse


def audit_split(split: str, prereg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = OUT_ROOT / "runs" / split
    metadata = read_json(out / "metadata.json")
    summary = read_json(out / "summary.json")
    rows = read_jsonl(out / "metrics.jsonl")
    lookup = {(row["config_id"], int(row["context"]), row["scenario"], row["method"], row["panel"], int(row["layer"])): row for row in rows}
    raw = np.load(out / "raw_matrices.npz")
    effects, endpoints, positives = raw["effects"], raw["endpoints"], raw["positive_endpoints"]
    dims = metadata["dimensions"]
    mismatches = []
    for ci, config in enumerate(dims["configs"]):
        for xi, context in enumerate(dims["contexts"]):
            for si, scenario in enumerate(dims["scenarios"]):
                expected = mapping(metadata, config["config_id"], int(context), scenario)
                for mi, method in enumerate(dims["methods"]):
                    for pi, panel in enumerate(dims["panels"]):
                        for li, layer in enumerate(dims["layers"]):
                            actual = recompute_metrics(effects[ci, xi, si, mi, pi, li], endpoints[ci, xi, si, mi, pi, li], positives[ci, xi, si, pi], expected)
                            stored = lookup[(config["config_id"], int(context), scenario, method, panel, int(layer))]
                            for name, value in actual.items():
                                if not close(value, stored[name]):
                                    mismatches.append({"key": [config["config_id"], context, scenario, method, panel, layer], "metric": name, "stored": stored[name], "actual": value})
    expected_rows = len(dims["configs"]) * len(dims["contexts"]) * len(dims["scenarios"]) * len(dims["methods"]) * len(dims["panels"]) * len(dims["layers"])
    named = {
        "protocol_digest": metadata["protocol_digest"] == prereg["protocol_digest"],
        "raw_hash": sha256_file(out / "raw_matrices.npz") == summary["raw_sha256"],
        "metrics_hash": sha256_file(out / "metrics.jsonl") == summary["metrics_sha256"],
        "metadata_hash": sha256_file(out / "metadata.json") == summary["metadata_sha256"],
        "row_count": len(rows) == expected_rows == summary["metric_row_count"],
        "raw_shape": list(effects.shape) == metadata["raw_shape"],
        "raw_finite": bool(np.isfinite(effects).all() and np.isfinite(endpoints).all()),
        "metrics_recomputed": not mismatches,
        "twelve_cells": summary["cell_count"] == 12,
        "all_cells_pass": bool(summary["all_cells_pass"]),
    }
    return ([{"split": split, "name": name, "passed": bool(value)} for name, value in named.items()], {"split": split, "named_checks": named, "mismatches": mismatches[:20]})


def main() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(prereg)
    stored = body.pop("protocol_digest")
    checks = [
        {"name": "protocol_digest", "passed": digest(body) == stored},
        {"name": "primary_hash", "passed": sha256_file(PRIMARY) == prereg["script_sha256"]},
        {"name": "protocol_audit", "passed": bool(read_json(OUT_ROOT / "protocol/audit.json")["all_checks_passed"])},
    ]
    details = []
    for split in SPLITS:
        more, detail = audit_split(split, prereg)
        checks.extend(more)
        details.append(detail)
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checks.extend([
        {"name": "selection_qualified", "passed": bool(selection["candidate_qualified"])},
        {"name": "calibration_confirmed", "passed": bool(final["calibration_passed"])},
        {"name": "natural_protocol_only", "passed": bool(final["natural_discovery_protocol_authorized"] and not final["natural_hidden_scan_authorized"])},
        {"name": "components_denied", "passed": not bool(final["component_search_authorized"])},
    ])
    audit = {"phase": PHASE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "check_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks), "all_checks_passed": all(bool(row["passed"]) for row in checks), "checks": checks, "details": details, "protocol_digest": stored, "final_digest": final["final_digest"]}
    audit["audit_digest"] = digest(audit)
    write_json(OUT_ROOT / "audit/independent_result_audit.json", audit)
    print(canonical(audit))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
