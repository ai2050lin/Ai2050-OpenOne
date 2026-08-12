#!/usr/bin/env python3
"""Post-hoc tie-aware adjudication for the single Phase 1221 audit mismatch.

This does not replace the frozen audit. It preserves that 20/21 result and
tests the narrower diagnosis that canonical JSON key sorting changed only the
arbitrary winner among exactly tied mean scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
SCRIPT = Path(__file__).resolve()
OUT_ROOT = TEST_ROOT / "result/phase1221_typed_operation_behavior_and_error_fingerprints"
MATERIAL_PATH = OUT_ROOT / "material/typed_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FROZEN_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
OUTPUT_PATH = OUT_ROOT / "audit/posthoc_tie_aware_result_audit.json"
TOLERANCE = 1e-7


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"output already exists: {OUTPUT_PATH}")
    materials = {row["item_id"]: row for row in read_jsonl(MATERIAL_PATH)}
    manifests = {row["item_id"]: row for row in read_jsonl(MANIFEST_PATH)}
    raw = read_jsonl(RAW_PATH)
    final = read_json(FINAL_PATH)
    frozen = read_json(FROZEN_AUDIT_PATH)

    row_checks = []
    tied_sum_rows = 0
    tied_mean_rows = 0
    serialized_tie_winner_mismatches = 0
    for row in raw:
        material = materials[row["item_id"]]
        manifest = manifests[row["item_id"]]
        scores = row["candidate_scores"]
        max_sum = max(value["sum_log_probability"] for value in scores.values())
        max_mean = max(value["mean_log_probability"] for value in scores.values())
        sum_maximizers = {
            candidate for candidate, value in scores.items()
            if abs(value["sum_log_probability"] - max_sum) <= TOLERANCE
        }
        mean_maximizers = {
            candidate for candidate, value in scores.items()
            if abs(value["mean_log_probability"] - max_mean) <= TOLERANCE
        }
        tied_sum_rows += len(sum_maximizers) > 1
        tied_mean_rows += len(mean_maximizers) > 1
        sum_prediction_valid = (
            row["sum_prediction"] is None
            if len(sum_maximizers) > 1
            else row["sum_prediction"] in sum_maximizers
        )
        mean_prediction_valid = row["mean_prediction"] in mean_maximizers
        serialized_first = sorted(
            scores, key=lambda candidate: scores[candidate]["mean_log_probability"], reverse=True
        )[0]
        serialized_tie_winner_mismatches += (
            len(mean_maximizers) > 1 and serialized_first != row["mean_prediction"]
        )
        prediction_position = (
            material["candidate_order"].index(row["sum_prediction"])
            if row["sum_prediction"] in material["candidate_order"]
            else None
        )
        expected_fingerprint = (
            material["fingerprint_by_candidate"].get(row["sum_prediction"], "unregistered_candidate")
            if row["sum_prediction"]
            else "tie"
        )
        checks = {
            "candidate_set": set(scores) == set(manifest["candidates"]),
            "sum_prediction_equivalence_class": sum_prediction_valid,
            "mean_prediction_equivalence_class": mean_prediction_valid,
            "agreement": row["sum_mean_winner_agreement"] == (row["sum_prediction"] == row["mean_prediction"]),
            "correct": row["candidate_correct"] == (row["sum_prediction"] == material["gold"]),
            "position": row["prediction_position"] == prediction_position,
            "fingerprint": row["error_fingerprint"] == expected_fingerprint,
            "finite": row["all_candidate_scores_finite"] == all(
                value["all_vocab_logits_finite"] and math.isfinite(value["sum_log_probability"])
                and math.isfinite(value["mean_log_probability"])
                for value in scores.values()
            ),
        }
        row_checks.append(all(checks.values()))

    only_expected_frozen_failure = (
        frozen["check_count"] == 21
        and frozen["passed_count"] == 20
        and not frozen["all_checks_passed"]
        and [key for key, value in frozen["checks"].items() if not value] == ["candidate_scores_recomputed"]
    )
    checks = {
        "frozen_audit_preserved": only_expected_frozen_failure,
        "all_rows_tie_aware_valid": len(raw) == 15360 and all(row_checks),
        "at_least_one_mean_tie": tied_mean_rows > 0,
        "mismatch_explained_by_serialized_tie_order": serialized_tie_winner_mismatches > 0,
        "primary_authorization_remains_denied": (
            final["status"] == "typed_behavior_no_family_authorized"
            and not final["authorized_next"]["automatic_execution"]
            and not final["behavior"]["authorized_family_tracks"]
        ),
    }
    result = {
        "phase": 1221,
        "mode": "posthoc_tie_aware_adjudication",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "supplemental_only_does_not_replace_frozen_audit",
        "script_sha256": file_sha256(SCRIPT),
        "frozen_audit_digest": frozen["audit_digest"],
        "final_digest": final["final_digest"],
        "row_count": len(raw),
        "tied_sum_row_count": tied_sum_rows,
        "tied_mean_row_count": tied_mean_rows,
        "serialized_tie_winner_mismatch_count": serialized_tie_winner_mismatches,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_boundary": {
            "posthoc": True,
            "changes_primary_predictions": False,
            "changes_authorization": False,
            "upgrades_frozen_audit_to_preregistered_pass": False,
        },
    }
    result["audit_digest"] = digest(result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
