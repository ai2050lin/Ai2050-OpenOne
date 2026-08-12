from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1191_prefix_future_formation_identity as p1191  # noqa: E402


def add(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def compare_summary(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = [
        horizon + suffix
        for horizon in ("middle", "late", "endpoint")
        for suffix in ("_true_cosine_mean", "_null_cosine_mean", "_advantage_mean", "_positive_fraction")
    ]
    errors = [abs(float(left[key]) - float(right[key])) for key in keys]
    for key in ("system_count", "task_count", "positive_task_count"):
        errors.append(float(left[key] != right[key]))
    errors.append(float(bool(left["positive_gate_pass"]) != bool(right["positive_gate_pass"])))
    errors.append(float(bool(left["negative_boundary_pass"]) != bool(right["negative_boundary_pass"])))
    return max(errors)


def replay_error(observed: dict[str, Any], replayed: dict[str, Any]) -> float:
    errors = []
    for field in ("prefix", "middle", "late", "endpoint"):
        errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(observed[field], dtype=np.float64)
                        - np.asarray(replayed[field], dtype=np.float64)
                    )
                )
            )
        )
    for horizon in ("middle", "late", "endpoint"):
        errors.append(abs(observed[horizon + "_true_cosine"] - replayed[horizon + "_true_cosine"]))
    return max(errors)


def audit() -> None:
    checks: list[dict[str, Any]] = []
    protocol = p1191.read_json(p1191.PROTOCOL_PATH)
    summary = p1191.read_json(p1191.SUMMARY_PATH)
    claims = p1191.read_json(p1191.CLAIMS_PATH)
    rows = p1191.read_jsonl(p1191.RAW_ROWS)

    add(
        checks,
        "protocol_digest",
        p1191.digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
        == protocol["protocol_digest"],
    )
    add(checks, "source_hashes", p1191.source_hashes() == protocol["source_hashes"])
    add(
        checks,
        "formal_manifest",
        p1191.formal_manifest(p1191.endpoints(p1191.FORMAL_SOURCE)) == protocol["formal_manifest"],
    )
    add(checks, "raw_hash", p1191.file_sha256(p1191.RAW_ROWS) == summary["raw_rows_sha256"])
    add(checks, "row_count", len(rows) == 64, len(rows))
    add(checks, "unique_trajectories", len({row["trajectory_id"] for row in rows}) == 64)
    add(checks, "task_count", len({row["task_name"] for row in rows}) == 8)
    add(
        checks,
        "split_counts",
        {split: sum(row["split"] == split for row in rows) for split in ("discovery", "confirmation")}
        == {"discovery": 32, "confirmation": 32},
    )
    add(
        checks,
        "vector_lengths",
        all(len(row[field]) == 128 for row in rows for field in ("prefix", "middle", "late", "endpoint")),
    )
    add(
        checks,
        "finite_vectors",
        all(np.isfinite(np.asarray(row[field], dtype=np.float64)).all() for row in rows for field in ("prefix", "middle", "late", "endpoint")),
    )

    lookup = {(row["task_name"], row["replicate"]): row for row in rows}
    maximum_error = 0.0
    mapping_errors = 0
    for row in rows:
        prefix = np.asarray(row["prefix"], dtype=np.float64)
        null = lookup[(row["task_name"], (row["replicate"] + 1) % 8)]
        mapping_errors += int(row["null_trajectory_id"] != null["trajectory_id"])
        maximum_error = max(maximum_error, abs(float(np.linalg.norm(prefix)) - row["prefix_norm"]))
        for horizon in ("middle", "late", "endpoint"):
            future = np.asarray(row[horizon], dtype=np.float64)
            null_future = np.asarray(null[horizon], dtype=np.float64)
            true_cosine = p1191.cosine(prefix, future)
            null_cosine = p1191.cosine(prefix, null_future)
            maximum_error = max(
                maximum_error,
                abs(float(np.linalg.norm(future)) - row[horizon + "_norm"]),
                abs(true_cosine - row[horizon + "_true_cosine"]),
                abs(null_cosine - row[horizon + "_null_cosine"]),
                abs((true_cosine - null_cosine) - row[horizon + "_advantage"]),
            )
    add(checks, "metric_recompute", maximum_error <= 1e-12, maximum_error)
    add(checks, "null_mapping", mapping_errors == 0, mapping_errors)

    for split in ("discovery", "confirmation"):
        independent = p1191.summarize(rows, split)
        error = compare_summary(independent, summary[split])
        add(checks, split + ".summary", error <= 1e-12, error)

    contract = p1191.read_json(p1187.CONTRACT_PATH)
    for family in ("positive", "negative"):
        recompiled = {
            name: p1187.compile_claim(raw, contract) for name, raw in claims[family]["raw"].items()
        }
        add(checks, family + ".typed_recompile", recompiled == claims[family]["compiled"])
        add(checks, family + ".typed_accept", all(claim["accepted"] for claim in recompiled.values()))
        add(
            checks,
            family + ".typed_gate",
            bool(all(claim["authorizes"] for claim in recompiled.values()))
            == bool(claims[family]["gate_pass"]),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    endpoint_map = {
        p1191.p1190.trajectory_id(p1191.p1189.load_payload(path)): path
        for path in p1191.endpoints(p1191.FORMAL_SOURCE)
    }
    replay_details = []
    for task_name in sorted({row["task_name"] for row in rows}):
        observed = next(row for row in rows if row["task_name"] == task_name and row["replicate"] == 0)
        replayed = p1191.build_rows([endpoint_map[observed["trajectory_id"]]], "formal", torch.device("cuda"))[0]
        error = replay_error(observed, replayed)
        replay_details.append({"task_name": task_name, "max_error": error})
        torch.cuda.empty_cache()
    replay_max = max(item["max_error"] for item in replay_details)
    add(checks, "eight_task_cuda_replay", replay_max <= 1e-8, replay_details)

    first = rows[0]
    prefix = np.asarray(first["prefix"], dtype=np.float64)
    original = np.asarray(first["endpoint"], dtype=np.float64)
    corrupted = original[::-1].copy()
    corruption = abs(p1191.cosine(prefix, original) - p1191.cosine(prefix, corrupted))
    add(checks, "rank_corruption_positive_sentinel", corruption >= 0.05, corruption)
    add(checks, "decision_exclusive", not (summary["positive_gate_pass"] and summary["negative_boundary_pass"]))
    add(checks, "decision_consistent", summary["decision"] in ("positive", "negative_boundary", "ambiguous"))

    gate_pass = all(check["pass"] for check in checks)
    result = {
        "phase": p1191.PHASE,
        "audit_kind": "independent_digest_vector_null_type_and_cuda_replay",
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "gate_pass": gate_pass,
        "audit_digest": None,
    }
    result["audit_digest"] = p1191.digest({key: value for key, value in result.items() if key != "audit_digest"})
    p1191.write_json(p1191.AUDIT_PATH, result)
    if not gate_pass:
        raise RuntimeError("Phase1191 audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    parser.parse_args()
    audit()


if __name__ == "__main__":
    main()
