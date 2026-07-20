#!/usr/bin/env python3
"""Open and execute the frozen Phase573 sealed behavior and causal validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase573_natural_transition_behavior import balanced_worlds, stable_expected  # noqa: E402
import phase573_natural_transition_protocol as protocol  # noqa: E402
import phase573_coarse_message_causal_protocol as causal_protocol  # noqa: E402
from phase573_coarse_message_causal import (  # noqa: E402
    causal_batch,
    deterministic_sign_flip_count,
    generate_batch,
)
import phase573_sealed_validation_protocol as sealed_protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
MODEL = causal_protocol.MODEL


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    import gzip
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def iter_sealed(path: Path) -> list[dict[str, Any]]:
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def behavior_rows_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_sealed_behavior_rows.jsonl.gz"


def causal_rows_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_sealed_causal_rows.jsonl.gz"


def registry_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_sealed_registry.json"


def summary_path() -> Path:
    return OUT_DIR / f"phase573_{MODEL}_sealed_validation_summary.json"


def decision_path() -> Path:
    return OUT_DIR / "phase573_sealed_validation_decision.json"


def receipt_path() -> Path:
    return OUT_DIR / "phase573_sealed_execution_receipt.json"


def run(max_new_tokens: int, restart: bool) -> Path:
    frozen = read_json(sealed_protocol.SEALED_PROTOCOL)
    paths = (
        behavior_rows_path(), causal_rows_path(), registry_path(), summary_path(),
        decision_path(), receipt_path(),
    )
    if restart:
        for path in paths:
            path.unlink(missing_ok=True)
    if sha256_file(protocol.SEALED_CASES_PATH) != frozen[
        "sealed_cases_sha256_committed"
    ]:
        raise RuntimeError("Phase573 sealed case commitment mismatch")
    cases = iter_sealed(protocol.SEALED_CASES_PATH)
    if (
        len(cases) != frozen["sealed_case_count_committed"]
        or len(cases) != 4096
        or any(not row["sealed"] or row["split"] != "sealed" for row in cases)
    ):
        raise RuntimeError("Phase573 sealed case denominator or identity drift")
    case_bank = {row["case_id"]: row for row in cases}
    by_world_variant = {
        (row["base_case_id"], row["variant"]): row for row in cases
    }
    base_rows = {
        row["base_case_id"]: row for row in cases if row["variant"] == "base"
    }
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError("Phase573 sealed model identity drift")
        if getattr(loaded.model.config, "_attn_implementation", None) != "eager":
            raise RuntimeError("Phase573 sealed causal accounting requires eager attention")

        output_rows: list[dict[str, Any]] = []
        relation_cases = sorted(
            [row for row in cases if row["variant"] in ("base", "relation_swap")],
            key=lambda row: row["case_id"],
        )
        batch_size = int(frozen["behavior_batch_size"])
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(relation_cases), batch_size):
                output_rows.extend(generate_batch(
                    loaded, relation_cases[start:start + batch_size], repeat,
                    max_new_tokens,
                ))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase573 sealed "
                f"relation/{repeat} {len(relation_cases)}/{len(relation_cases)}",
                flush=True,
            )
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        world_ids = sorted(base_rows)
        relation_eligible = [
            base_id for base_id in world_ids
            if stable_expected(by_case_repeat, f"{base_id}_base")
            and stable_expected(by_case_repeat, f"{base_id}_relation_swap")
        ]
        relation_minimum = frozen["behavior_gate"][
            "minimum_relation_qualified_worlds_each_split"
        ]
        if len(relation_eligible) < relation_minimum:
            raise RuntimeError(
                f"Phase573 sealed relation behavior gate failed: {len(relation_eligible)}"
            )
        controls_selected = balanced_worlds(
            base_rows,
            relation_eligible,
            int(frozen["behavior_gate"]["control_screen_cap_each_split"]),
        )
        controls = sorted(
            [
                by_world_variant[(base_id, variant)]
                for base_id in controls_selected
                for variant in ("object_swap", "order_swap")
            ],
            key=lambda row: row["case_id"],
        )
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(controls), batch_size):
                output_rows.extend(generate_batch(
                    loaded, controls[start:start + batch_size], repeat, max_new_tokens
                ))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase573 sealed "
                f"controls/{repeat} {len(controls)}/{len(controls)}",
                flush=True,
            )
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        all_axis = [
            base_id for base_id in controls_selected
            if stable_expected(by_case_repeat, f"{base_id}_object_swap")
            and stable_expected(by_case_repeat, f"{base_id}_order_swap")
        ]
        final_count = int(frozen["behavior_gate"]["final_worlds_each_split"])
        selected = balanced_worlds(base_rows, all_axis, final_count)
        if len(selected) != final_count:
            raise RuntimeError(
                f"Phase573 sealed all-axis behavior gate failed: {len(selected)}"
            )
        write_jsonl(behavior_rows_path(), output_rows)
        write_json(registry_path(), {
            "schema_version": "phase573_sealed_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "relation_qualified_world_count": len(relation_eligible),
            "control_screen_world_count": len(controls_selected),
            "all_axis_qualified_world_count": len(all_axis),
            "selected_world_count": len(selected),
            "selected_base_case_ids": selected,
            "selection_uses_internal_state": False,
            "sealed_split_read": True,
        })

        causal_rows: list[dict[str, Any]] = []
        max_reconstruction = 0.0
        baseline_mismatch = 0
        causal_batch_worlds = int(frozen["causal_batch_worlds"])
        for start in range(0, len(selected), causal_batch_worlds):
            batch_worlds = selected[start:start + causal_batch_worlds]
            rows = sorted(
                [
                    case_bank[f"{base_id}_{variant}"]
                    for base_id in batch_worlds
                    for variant in ("base", "relation_swap")
                ],
                key=lambda row: (row["base_case_id"], row["variant"]),
            )
            batch_rows, reconstruction, mismatch = causal_batch(
                loaded,
                layers,
                rows,
                tuple(frozen["conditions"]),
                int(frozen["candidate_layer"]),
                int(frozen["wrong_depth_control_layer"]),
            )
            causal_rows.extend(batch_rows)
            max_reconstruction = max(max_reconstruction, reconstruction)
            baseline_mismatch += mismatch
            if reconstruction > float(frozen["reconstruction_relative_error_max"]):
                raise RuntimeError(
                    f"Phase573 sealed reconstruction failed: {reconstruction}"
                )
        write_jsonl(causal_rows_path(), causal_rows)

        by_condition = {
            condition: [row for row in causal_rows if row["condition"] == condition]
            for condition in frozen["conditions"]
        }
        expected = final_count * 2
        if any(len(rows) != expected for rows in by_condition.values()):
            raise RuntimeError("Phase573 sealed causal denominator drift")
        metrics = {}
        for condition, rows in by_condition.items():
            effects = [float(row["donor_switch_effect"]) for row in rows]
            metrics[condition] = {
                "case_count": len(rows),
                "mean_donor_switch_effect": finite(sum(effects) / len(effects)),
                "positive_effect_rate": sum(value > 0.0 for value in effects) / len(effects),
                "donor_candidate_win_rate": sum(
                    row["intervention_donor_wins"] for row in rows
                ) / len(rows),
                "sign_flip_audit": deterministic_sign_flip_count(
                    effects, int(frozen["permutations"])
                ),
            }
        gate = frozen["causal_gate"]
        remove = metrics["selected_edge_remove"]
        replace = metrics["paired_relation_selected_replace"]
        nonselected = metrics["nonselected_edge_remove"]
        control_means = [
            metrics[condition]["mean_donor_switch_effect"]
            for condition in (
                "channel_roll_donor_replace", "wrong_depth_donor_replace",
                "wrong_position_donor_replace",
            )
        ]
        remove_gap = (
            remove["mean_donor_switch_effect"]
            - nonselected["mean_donor_switch_effect"]
        )
        replace_gap = replace["mean_donor_switch_effect"] - max(control_means)
        sealed_pass = (
            remove["positive_effect_rate"] >= gate["minimum_positive_effect_rate"]
            and replace["positive_effect_rate"] >= gate["minimum_positive_effect_rate"]
            and remove["mean_donor_switch_effect"]
            > gate["minimum_mean_donor_switch_effect"]
            and replace["mean_donor_switch_effect"]
            > gate["minimum_mean_donor_switch_effect"]
            and remove_gap >= gate["minimum_mean_gap_vs_control"]
            and replace_gap >= gate["minimum_mean_gap_vs_control"]
            and replace["donor_candidate_win_rate"]
            >= gate["minimum_donor_candidate_win_rate"]
        )
        summary = {
            "schema_version": "phase573_sealed_validation_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "relation_qualified_world_count": len(relation_eligible),
            "control_screen_world_count": len(controls_selected),
            "all_axis_qualified_world_count": len(all_axis),
            "selected_world_count": len(selected),
            "behavior_row_count": len(output_rows),
            "causal_row_count": len(causal_rows),
            "condition_count": len(frozen["conditions"]),
            "candidate_layer": frozen["candidate_layer"],
            "candidate_receiver": frozen["candidate_receiver"],
            "condition_metrics": metrics,
            "selected_vs_nonselected_removal_mean_gap": finite(remove_gap),
            "paired_replace_vs_strongest_control_mean_gap": finite(replace_gap),
            "sealed_causal_gate_pass": sealed_pass,
            "maximum_reconstruction_relative_error": finite(max_reconstruction),
            "same_shape_baseline_mismatch_count": baseline_mismatch,
            "runtime_seconds": time.monotonic() - started,
            "behavior_rows_sha256": sha256_file(behavior_rows_path()),
            "causal_rows_sha256": sha256_file(causal_rows_path()),
            "post_softmax_value_contribution_intervention": True,
            "key_effect_identified": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": True,
        }
        write_json(summary_path(), summary)
        decision = {
            "schema_version": "phase573_sealed_validation_decision.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "sealed_causal_gate_pass": sealed_pass,
            "claim_allowed": (
                "Qwen3 local all-head selected semantic-fact value message at layer24 "
                "answer boundary is necessary and replaceable on this synthetic relation task"
                if sealed_pass else None
            ),
            "claim_not_allowed": [
                "relation selection rule closure",
                "attention key/query mechanism",
                "head, channel, neuron, or parameter mechanism",
                "cross-model portability",
                "natural-language category encoding closure",
                "72-mechanism closure",
            ],
            "next_step": (
                "decompose how the query-conditioned selected message is formed before "
                "layer24, without reopening the sealed set"
                if sealed_pass else
                "downgrade the coarse edge to dual-open-split evidence and stop this route"
            ),
            "sealed_split_read": True,
        }
        write_json(decision_path(), decision)
        receipt = {
            "schema_version": "phase573_sealed_execution_receipt.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "sealed_protocol_sha256": sha256_file(sealed_protocol.SEALED_PROTOCOL),
            "sealed_cases_sha256_verified": sha256_file(protocol.SEALED_CASES_PATH),
            "sealed_case_count_read": len(cases),
            "sealed_validation_complete": True,
            "sealed_causal_gate_pass": sealed_pass,
            "summary_sha256": sha256_file(summary_path()),
            "decision_sha256": sha256_file(decision_path()),
        }
        write_json(receipt_path(), receipt)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary_path()
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
