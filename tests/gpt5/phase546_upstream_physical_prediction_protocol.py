#!/usr/bin/env python3
"""Freeze an upstream-only event search and a fresh physical confirmation set.

Phase545 showed that every nominally successful global event was either measured
after generation or at the layer-zero input.  This protocol therefore freezes a
mechanical repair: select the strongest prompt-end event, excluding layer-zero
input, using Phase545 discovery rows only, then test it on untouched physical
pairs 24-72 from the independent-confirmation behavior split.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase545_natural_entry_physical_analysis import event_table  # noqa: E402


PHASE544 = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
PHASE545 = ROOT / "tests/gpt5/result/phase545_natural_entry_physical_path"
OUT_DIR = ROOT / "tests/gpt5/result/phase546_upstream_physical_prediction"
MATRIX_PATH = PHASE544 / "phase544_model_mechanism_qualification.jsonl"
CASES_PATH = PHASE544 / "phase544_registered_cases.jsonl"
OLD_PAIRS_PATH = PHASE545 / "phase545_registered_physical_pairs.jsonl"
EVENTS_PATH = OUT_DIR / "phase546_frozen_upstream_events.jsonl"
PAIRS_PATH = OUT_DIR / "phase546_registered_confirmation_pairs.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase546_upstream_protocol.json"
AUDIT_PATH = OUT_DIR / "phase546_static_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
PAIR_INDEX_START = 24
PAIR_INDEX_STOP = 73


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_events(eligible: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for model in MODELS:
        rows_path = PHASE545 / f"phase545_{model}_pair_layer_rows.jsonl"
        if not rows_path.exists():
            continue
        for row in iter_jsonl(rows_path):
            key = (row["model"], row["family_id"], row["mechanism_id"])
            if row["split"] == "discovery" and key in eligible:
                groups[key].append(row)

    frozen = []
    for (model, family, mechanism), rows in sorted(groups.items()):
        table = event_table(rows)
        allowed = [
            value for value in table.values()
            if value["stage"] == "prompt_end"
            and not (value["component"] == "layer_input" and value["layer"] == 0)
        ]
        if not allowed:
            raise RuntimeError(f"No upstream event candidates for {(model, family, mechanism)}")
        selected = max(
            allowed,
            key=lambda value: (
                value["selection_score"],
                value["median_pair_direction_alignment"],
                -value["layer"],
            ),
        )
        frozen.append({
            "schema_version": "phase546_frozen_upstream_event.v1",
            "phase_id": "Phase546",
            "created_at": now(),
            "event_id": f"phase546_{model}_{family}_{mechanism}_upstream_event",
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "selection_source_phase": "Phase545",
            "selection_split": "discovery",
            "selection_pair_indices": list(range(0, 24)),
            "stage": selected["stage"],
            "component": selected["component"],
            "role": selected["role"],
            "layer": selected["layer"],
            "layer_count": selected["layer_count"],
            "relative_depth": selected["relative_depth"],
            "discovery_statistics": selected,
            "generated_token_state_excluded": True,
            "layer_zero_input_excluded": True,
            "observer_only": True,
            "predictive": False,
            "compute_edge": False,
            "causal": False,
            "single_neuron": False,
            "sealed": False,
        })
    return frozen


def register() -> dict[str, Any]:
    matrix = read_jsonl(MATRIX_PATH)
    eligible = {
        (row["model"], row["family_id"], row["mechanism_id"])
        for row in matrix if row["behavior_entry_eligible"]
    }
    frozen_events = freeze_events(eligible)
    frozen_by_cell = {
        (row["model"], row["family_id"], row["mechanism_id"]): row
        for row in frozen_events
    }
    write_jsonl(EVENTS_PATH, frozen_events)

    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["surface"] == "direct_natural"
        and row["split"] == "independent_confirmation"
        and PAIR_INDEX_START <= row["pair_index"] < PAIR_INDEX_STOP
        and (row["model"], row["family_id"], row["mechanism_id"]) in eligible
    ]
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        groups[(row["model"], row["family_id"], row["mechanism_id"], row["pair_index"])].append(row)

    pairs = []
    for (model, family, mechanism, pair_index), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["world_condition"])
        if len(rows) != 2 or {row["world_condition"] for row in rows} != {"world_a", "world_b"}:
            raise RuntimeError((model, family, mechanism, pair_index, len(rows)))
        event = frozen_by_cell[(model, family, mechanism)]
        pairs.append({
            "schema_version": "phase546_upstream_confirmation_pair.v1",
            "phase_id": "Phase546",
            "created_at": now(),
            "physical_pair_id": f"phase546_{model}_{family}_{mechanism}_confirmation_{pair_index:03d}",
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "split": "fresh_physical_confirmation",
            "source_behavior_split": "independent_confirmation",
            "pair_index": pair_index,
            "world_a_case_id": rows[0]["case_id"],
            "world_b_case_id": rows[1]["case_id"],
            "world_a_target": rows[0]["target"],
            "world_b_target": rows[1]["target"],
            "surface": "direct_natural",
            "frozen_event_id": event["event_id"],
            "frozen_stage": event["stage"],
            "frozen_component": event["component"],
            "frozen_role": event["role"],
            "frozen_layer": event["layer"],
            "all_layers_on_frozen_axis_collected": True,
            "observer_only": True,
            "predictive": False,
            "compute_edge": False,
            "causal": False,
            "single_neuron": False,
            "sealed": False,
        })
    write_jsonl(PAIRS_PATH, pairs)

    old_keys = {
        (row["model"], row["family_id"], row["mechanism_id"], row["split"], row["pair_index"])
        for row in read_jsonl(OLD_PAIRS_PATH)
    }
    new_keys = {
        (row["model"], row["family_id"], row["mechanism_id"], "independent_confirmation", row["pair_index"])
        for row in pairs
    }
    counts = Counter(row["model"] for row in pairs)
    cell_counts = Counter((row["model"], row["family_id"], row["mechanism_id"]) for row in pairs)
    audit = {
        "schema_version": "phase546_upstream_static_audit.v1",
        "phase_id": "Phase546",
        "created_at": now(),
        "behavior_eligible_cell_count": len(eligible),
        "frozen_upstream_event_count": len(frozen_events),
        "registered_confirmation_pair_count": len(pairs),
        "registered_pair_count_by_model": dict(counts),
        "pair_count_per_model_mechanism": sorted(set(cell_counts.values())),
        "confirmation_pair_index_range": [PAIR_INDEX_START, PAIR_INDEX_STOP - 1],
        "phase545_physical_pair_overlap_count": len(old_keys & new_keys),
        "target_identity_error_count": sum(row["world_a_target"] == row["world_b_target"] for row in pairs),
        "generated_stage_event_count": sum(row["stage"] != "prompt_end" for row in frozen_events),
        "layer_zero_input_event_count": sum(
            row["component"] == "layer_input" and row["layer"] == 0 for row in frozen_events
        ),
        "sealed_pair_count": sum(row["sealed"] for row in pairs),
        "strict_preregistration": False,
        "repair_rule_frozen_before_fresh_physical_collection": True,
        "fresh_physical_holdout": True,
    }
    audit["valid"] = (
        len(eligible) == 9
        and len(frozen_events) == 9
        and len(pairs) == 441
        and dict(counts) == {"glm4": 245, "qwen3": 196}
        and sorted(set(cell_counts.values())) == [49]
        and audit["phase545_physical_pair_overlap_count"] == 0
        and audit["target_identity_error_count"] == 0
        and audit["generated_stage_event_count"] == 0
        and audit["layer_zero_input_event_count"] == 0
        and audit["sealed_pair_count"] == 0
    )
    audit["status"] = "static_pass_no_fresh_hidden_state_read" if audit["valid"] else "static_fail"
    write_json(AUDIT_PATH, audit)

    protocol = {
        "schema_version": "phase546_upstream_physical_prediction_protocol.v1",
        "phase_id": "Phase546",
        "created_at": now(),
        "title": "Fresh prediction of prompt-end upstream physical observers",
        "models_in_required_order": list(MODELS),
        "gate_skipped_models": ["deepseek7b"],
        "repair_context": (
            "Phase545 terminal-identity audit was observed before this rule was frozen; therefore this is "
            "an independently confirmed repair, not a strict preregistration."
        ),
        "frozen_search_rule": {
            "selection_source": "Phase545 discovery rows only",
            "allowed_stage": "prompt_end",
            "excluded_event": "layer_input at layer zero",
            "score": "median normalized world-pair delta multiplied by positive direction-support fraction",
            "axes_frozen_before_confirmation": ["component", "role", "layer"],
        },
        "fresh_confirmation": {
            "source_behavior_split": "independent_confirmation",
            "physical_pair_indices": [PAIR_INDEX_START, PAIR_INDEX_STOP - 1],
            "independent_pairs_per_model_mechanism": PAIR_INDEX_STOP - PAIR_INDEX_START,
            "no_phase545_physical_overlap": True,
        },
        "prediction_gates": read_json(PHASE545 / "phase545_physical_protocol.json")["prediction_gates"],
        "claim_boundaries": {
            "upstream_observer_is_compute_edge": False,
            "upstream_observer_is_causal": False,
            "world_pair_delta_is_abstract_semantics": False,
            "single_neuron_scan_allowed": False,
            "sealed_split_read": False,
        },
        "frozen_events_path": str(EVENTS_PATH.relative_to(ROOT)),
        "frozen_events_sha256": sha256_file(EVENTS_PATH),
        "registered_pairs_path": str(PAIRS_PATH.relative_to(ROOT)),
        "registered_pairs_sha256": sha256_file(PAIRS_PATH),
        "source_behavior_cases_sha256": sha256_file(CASES_PATH),
        "source_behavior_matrix_sha256": sha256_file(MATRIX_PATH),
        "source_phase545_protocol_sha256": sha256_file(PHASE545 / "phase545_physical_protocol.json"),
        "static_audit": audit,
    }
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    register()
