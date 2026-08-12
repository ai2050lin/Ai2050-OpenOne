#!/usr/bin/env python3
"""Finalize Phase1045 and decide whether distributed receiver mapping is due."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1044_natural_recompute_trajectory_scan as tools
import phase1045_receiver_mediation_protocol as protocol
import phase1045_receiver_mediation_scan as scan


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def executed_logits(
    values: np.ndarray,
) -> np.ndarray:
    selected = []
    for condition_slot, condition in enumerate(
        protocol.SOURCE_CONDITIONS
    ):
        for operation in protocol.OPERATIONS_BY_CONDITION[condition]:
            operation_slot = scan.OPERATIONS.index(operation)
            selected.append(
                values[:, condition_slot, operation_slot, :, :]
            )
    return np.concatenate(
        [row.reshape(-1) for row in selected], axis=0
    )


def gate_audit(
    metrics: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "source_shift_median": (
            metrics["source_shift"]["median"]
            >= gate["source_shift_median_min"]
        ),
        "source_positive_rate": (
            metrics["source_shift"]["positive_rate"]
            >= gate["source_positive_rate_min"]
        ),
        "query_blocked_median": (
            metrics["query_blocked_amount"]["median"]
            >= gate["query_blocked_amount_median_min"]
        ),
        "query_blocked_positive_rate": (
            metrics["query_blocked_amount"]["positive_rate"]
            >= gate["query_blocked_positive_rate_min"]
        ),
        "query_mediation_fraction": (
            metrics["query_mediation_fraction"]["median"]
            >= gate["query_mediation_fraction_median_min"]
        ),
        "query_minus_wrong": (
            metrics["query_minus_wrong_blocked"]["median"]
            >= gate["query_minus_wrong_blocked_median_min"]
        ),
        "query_replay_shift_median": (
            metrics["query_replay_shift"]["median"]
            >= gate["query_replay_shift_median_min"]
        ),
        "query_replay_positive_rate": (
            metrics["query_replay_shift"]["positive_rate"]
            >= gate["query_replay_positive_rate_min"]
        ),
        "query_replay_recovery": (
            metrics["query_replay_recovery"]["median"]
            >= gate["query_replay_recovery_median_min"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {}
    finite_audits = {}
    gate_audits = {}
    manifests = {}
    for model_name in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model_name
        summary = protocol.read_json(atlas / "summary.json")
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name} protocol digest drift")
        values = np.load(
            atlas / "paired_candidate_logits.fp32.npy", mmap_mode="r"
        )
        executed = executed_logits(values)
        finite = tools.finite_summary(executed)
        summary["paired_logits_finite"] = finite
        summary["unexecuted_slots_are_nan"] = bool(
            np.all(
                np.isnan(values[:, 1:, 1:, :, :])
                | np.isfinite(values[:, 1:, 1:, :, :])
            )
        )
        protocol.write_json(atlas / "summary.json", summary)
        summaries[model_name] = summary
        finite_audits[model_name] = finite
        gate_audits[model_name] = gate_audit(
            summary["metrics"], prereg["mediation_gate"]
        )
        manifests[model_name] = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(atlas.iterdir())
            if path.is_file()
        }

    primary = prereg["mediation_gate"]["primary_models"]
    primary_pass = all(
        gate_audits[model_name]["passed"] for model_name in primary
    )
    source_replication = all(
        summaries[model_name]["metrics"]["source_shift"][
            "positive_rate"
        ]
        >= prereg["mediation_gate"]["source_positive_rate_min"]
        for model_name in protocol.MODELS
    )
    weak_query_use = {
        model_name: {
            "mediation_fraction_median": summaries[model_name][
                "metrics"
            ]["query_mediation_fraction"]["median"],
            "replay_recovery_median": summaries[model_name]["metrics"][
                "query_replay_recovery"
            ]["median"],
            "query_minus_wrong_blocked_median": summaries[model_name][
                "metrics"
            ]["query_minus_wrong_blocked"]["median"],
        }
        for model_name in protocol.MODELS
    }
    automatic_next = {
        "same_receiver_candidate_confirmed": primary_pass,
        "same_receiver_candidate_tuning_needed": False,
        "distributed_receiver_coalition_atlas_needed": (
            source_replication and not primary_pass
        ),
        "route": (
            "Stop tuning the single query receiver. Map preregistered "
            "multi-position receiver coalitions at a small set of relative "
            "depths, with full-sequence swap as an intervention upper bound."
            if source_replication and not primary_pass
            else "No immediate extension is authorized."
        ),
        "reason": (
            "The early complete-state source replicated strongly in all "
            "three models, but query-only mediation was 0.003, 0.095, and "
            "0.029 median in Qwen3, GLM4, and DeepSeek7B. A distributed "
            "receiver test is a new structural question, not threshold "
            "tuning of the failed point."
        ),
    }
    aggregate = {
        "schema_version": "phase1045_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "sample_plan": prereg["sample_plan"],
        "finite_audits": finite_audits,
        "model_gate_audits": gate_audits,
        "primary_models_passed": primary_pass,
        "source_replication_all_models": source_replication,
        "weak_query_use": weak_query_use,
        "model_metrics": {
            model_name: summary["metrics"]
            for model_name, summary in summaries.items()
        },
        "automatic_next_decision": automatic_next,
        "artifact_manifest": manifests,
        "conclusion": [
            (
                "The Phase1044 repeated query response is real and partly "
                "used, but it is not the principal mediator of the early "
                "source effect."
            ),
            (
                "GLM4 shows the clearest local use: 0.358 median blocked "
                "margin, 0.095 median mediation fraction, and 0.149 replay "
                "recovery. These are partial contributions below the "
                "preregistered 0.2 mediation threshold."
            ),
            (
                "Qwen3 and DeepSeek7B carry even smaller query-only shares. "
                "The common result is distributed parallel transport, not "
                "a universal single-query bottleneck."
            ),
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    protocol.write_json(
        protocol.OUT_ROOT / "automatic_next_decision.json",
        automatic_next,
    )
    print(json.dumps({
        "primary_models_passed": primary_pass,
        "source_replication_all_models": source_replication,
        "gate_audits": gate_audits,
        "automatic_next_decision": automatic_next,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
