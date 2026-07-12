#!/usr/bin/env python3
"""Audit Phase388 intervention mechanics before opening the causal denominator."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P388 = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rows = [
        row
        for model in MODELS
        for row in read_jsonl(
            P388 / "collection/instrument_audit" / model / "direction_rows.jsonl"
        )
    ]
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 instrument directions, found {len(rows)}")
    failures: list[str] = []
    finite_scalar_count = 0
    for row in rows:
        if row["identity_query_max_abs_error"] != 0.0:
            failures.append(f"identity_query:{row['direction_id']}")
        if row["identity_margin_shift_abs_error"] != 0.0:
            failures.append(f"identity_margin:{row['direction_id']}")
        for scenario in row["scenario_rows"]:
            audit = scenario["patch_audit"]
            condition = scenario["intervention"]
            expected_key = condition in {
                "identity_source_kv",
                "donor_source_k_only",
                "donor_source_kv",
                "donor_wrong_source_kv",
                "donor_source_kv_at_terminal_control_depth",
            }
            expected_value = condition in {
                "identity_source_kv",
                "donor_source_v_only",
                "donor_source_kv",
                "donor_wrong_source_kv",
                "donor_source_kv_at_terminal_control_depth",
            }
            if audit["key_patch_call_count"] != int(expected_key):
                failures.append(f"key_count:{row['direction_id']}:{condition}")
            if audit["value_patch_call_count"] != int(expected_value):
                failures.append(f"value_count:{row['direction_id']}:{condition}")
            for key in (
                "key_max_patch_error",
                "value_max_patch_error",
                "key_max_outside_error",
                "value_max_outside_error",
            ):
                if audit[key] != 0.0:
                    failures.append(f"{key}:{row['direction_id']}:{condition}")
            for key in (
                "query_projection_toward_donor",
                "query_shift_cosine_to_donor_direction",
                "query_off_axis_ratio",
                "query_shift_norm",
                "donor_vs_recipient_logit_margin",
                "donor_direction_margin_shift",
                "normalized_margin_mediation",
            ):
                if math.isfinite(float(scenario[key])):
                    finite_scalar_count += 1
                else:
                    failures.append(f"nonfinite:{row['direction_id']}:{condition}:{key}")
        generation = row["main_generation"]["audit"]
        if generation["key_patch_call_count"] != 1:
            failures.append(f"generation_key_count:{row['direction_id']}")
        if generation["value_patch_call_count"] != 1:
            failures.append(f"generation_value_count:{row['direction_id']}")
        if generation["key_max_outside_error"] != 0.0:
            failures.append(f"generation_key_outside:{row['direction_id']}")
        if generation["value_max_outside_error"] != 0.0:
            failures.append(f"generation_value_outside:{row['direction_id']}")

    valid = not failures
    summary = {
        "schema_version": "62.4.0",
        "phase_id": "Phase388-InstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": len(MODELS),
            "instrument_group_count": 2,
            "direction_count": len(rows),
            "scenario_count": len(rows) * 7,
            "finite_scalar_count": finite_scalar_count,
        },
        "results": {
            "identity_query_exact_count": sum(
                row["identity_query_max_abs_error"] == 0.0 for row in rows
            ),
            "identity_margin_exact_count": sum(
                row["identity_margin_shift_abs_error"] == 0.0 for row in rows
            ),
            "patch_failure_count": len(failures),
            "failures": failures,
            "valid": valid,
        },
        "authorization": {
            "causal_test": valid,
            "replace_instrument_or_causal_groups": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "instrument_validity_is_causal_evidence": False,
            "instrument_pilot_scientific_effect_used_for_threshold_tuning": False,
        },
    }
    write_json(P388 / "phase388_instrument_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
