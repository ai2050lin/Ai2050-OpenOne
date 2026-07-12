#!/usr/bin/env python3
"""Audit and hash all blind contrast rows before semantic condition mapping."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
CONTRAST = PHASE371 / "phase371c_blind_vector_contrast"
FREEZE = PHASE371 / "phase371c_blind_contrast_execution_freeze.json"
ROUTES = CONTRAST / "private/phase371c_blind_route_contrasts.jsonl"
VOCAB = CONTRAST / "private/phase371c_blind_vocab_contrasts.jsonl"
OUT = CONTRAST / "phase371c_blind_contrast_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_rows(path: Path, expected: int, route_rows: bool) -> tuple[int, list[str], Counter]:
    errors = []
    count = 0
    cells = Counter()
    forbidden = {"family_id", "mechanism_id", "contrast_condition", "target", "target_aliases", "distractors"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        count += 1
        row = json.loads(line)
        if forbidden & set(row):
            errors.append(f"forbidden:{line_number}")
        if row.get("semantic_labels_available") is not False:
            errors.append(f"semantic_boundary:{line_number}")
        if row.get("candidate_selected") is not False:
            errors.append(f"candidate_boundary:{line_number}")
        for key, value in row.get("indices", {}).items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"nonfinite:{line_number}:{key}")
        cells[(row["model"], row["anonymous_group_id"], row["anonymous_pair_id"])] += 1
    if count != expected:
        errors.append(f"row_count:{count}:{expected}")
    expected_per_pair = 756 if route_rows else 3
    if set(cells.values()) != {expected_per_pair}:
        errors.append(f"pair_cell_count:{sorted(set(cells.values()))}")
    if len(cells) != 396:
        errors.append(f"pair_count:{len(cells)}")
    return count, errors, cells


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    route_count, route_errors, route_cells = audit_rows(ROUTES, 299376, True)
    vocab_count, vocab_errors, vocab_cells = audit_rows(VOCAB, 1188, False)
    errors = [*route_errors, *vocab_errors]
    valid = bool(freeze["valid"]) and not errors
    summary = {
        "schema_version": "47.18.0",
        "phase_id": "Phase371C-Contrast",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "seal_complete_blind_contrast_rows_before_semantic_condition_mapping",
        "valid": valid,
        "denominator": {
            "route_row_count": route_count,
            "vocab_row_count": vocab_count,
            "anonymous_pair_count": len(route_cells),
            "route_rows_per_pair": sorted(set(route_cells.values())),
            "vocab_rows_per_pair": sorted(set(vocab_cells.values())),
        },
        "quality": {
            "error_count": len(errors),
            "errors": errors[:100],
            "semantic_labels_available": False,
            "candidate_selected_count": 0,
            "all_indices_finite": not any(error.startswith("nonfinite") for error in errors),
        },
        "sealed_hashes": {
            "route_rows": sha256_file(ROUTES),
            "vocab_rows": sha256_file(VOCAB),
            "extractor_freeze": sha256_file(FREEZE),
        },
        "authorization": {
            "freeze_semantic_discovery_gate": valid,
            "map_discovery_condition_semantics_before_gate_freeze": False,
            "open_calibration": False,
            "open_physical": False,
        },
        "next_decision": "freeze_exact_route_discovery_gate_then_map_A_B_and_C_D_pairs",
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
