#!/usr/bin/env python3
"""Aggregate and audit the completed Phase1032 atlas."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1032_span_alliance_protocol as protocol


def condition_map(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["condition"]): row
        for row in metrics["conditions"]
    }


def metric(
    conditions: dict[str, dict[str, Any]],
    condition: str,
    readout: str,
    scope: str,
) -> dict[str, Any]:
    values = conditions[condition]["metrics"]
    if readout == "within_template":
        return values["prototype_readout"]["within_template"][scope]
    if readout == "next_token":
        return values["next_token_candidate_logits"][scope]
    raise ValueError(readout)


def both_templates(
    predicate,
) -> bool:
    return all(predicate(template) for template in range(2))


def model_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    conditions = condition_map(metrics)
    result: dict[str, Any] = {}
    for readout in ("within_template", "next_token"):
        selected = both_templates(
            lambda template: (
                metric(
                    conditions,
                    "selected_source_span_b",
                    readout,
                    f"template_{template}",
                )["alternate_top1"]
                - metric(
                    conditions,
                    "unselected_source_span_b",
                    readout,
                    f"template_{template}",
                )["alternate_top1"]
                >= 0.30
            )
        )
        double_gain = both_templates(
            lambda template: (
                metric(
                    conditions,
                    "selected_source_span_b",
                    readout,
                    f"template_{template}/bank_double",
                )["alternate_top1"]
                - metric(
                    conditions,
                    "selected_source_endpoint_b",
                    readout,
                    f"template_{template}/bank_double",
                )["alternate_top1"]
                >= 0.10
            )
        )
        query_gain_q = both_templates(
            lambda template: (
                metric(
                    conditions,
                    "query_clause_span_q",
                    readout,
                    f"template_{template}",
                )["alternate_top1"]
                - metric(
                    conditions,
                    "query_nonce_span_q",
                    readout,
                    f"template_{template}",
                )["alternate_top1"]
                >= 0.10
                and metric(
                    conditions,
                    "query_clause_span_q",
                    readout,
                    f"template_{template}",
                )["alternate_top1"]
                - metric(
                    conditions,
                    "query_clause_span_scrambled",
                    readout,
                    f"template_{template}",
                )["alternate_top1"]
                >= 0.10
            )
        )
        query_gain_bq = both_templates(
            lambda template: (
                metric(
                    conditions,
                    "query_clause_span_bq",
                    readout,
                    f"template_{template}",
                )["base_top1"]
                - metric(
                    conditions,
                    "query_nonce_span_bq",
                    readout,
                    f"template_{template}",
                )["base_top1"]
                >= 0.10
                and metric(
                    conditions,
                    "query_clause_span_bq",
                    readout,
                    f"template_{template}",
                )["base_top1"]
                - metric(
                    conditions,
                    "query_clause_span_scrambled",
                    readout,
                    f"template_{template}",
                )["base_top1"]
                >= 0.10
            )
        )

        composition = {}
        for query_kind in (
            "query_nonce_q",
            "query_clause_q",
            "query_nonce_bq",
            "query_clause_bq",
        ):
            combined = f"source_pair_span_plus_{query_kind}"
            query_condition = f"{query_kind}_span"
            # Condition names place ``span`` before q/bq.
            query_condition = {
                "query_nonce_q": "query_nonce_span_q",
                "query_clause_q": "query_clause_span_q",
                "query_nonce_bq": "query_nonce_span_bq",
                "query_clause_bq": "query_clause_span_bq",
            }[query_kind]
            composition[query_kind] = both_templates(
                lambda template, combined=combined,
                query_condition=query_condition: (
                    metric(
                        conditions,
                        combined,
                        readout,
                        f"template_{template}",
                    )["base_top1"]
                    - max(
                        metric(
                            conditions,
                            "source_pair_span_b",
                            readout,
                            f"template_{template}",
                        )["base_top1"],
                        metric(
                            conditions,
                            query_condition,
                            readout,
                            f"template_{template}",
                        )["base_top1"],
                    )
                    >= 0.10
                )
            )

        result[readout] = {
            "selected_source_repetition_both_templates": selected,
            "double_span_gain_both_templates": double_gain,
            "query_clause_gain_q_both_templates": query_gain_q,
            "query_clause_gain_bq_both_templates": query_gain_bq,
            "composition_restoration_both_templates": composition,
        }
    return result


def compact_model(
    metrics: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    conditions = condition_map(metrics)
    selected_rows = {}
    for condition in (
        "selected_source_endpoint_b",
        "selected_source_span_b",
        "unselected_source_span_b",
        "source_pair_endpoint_b",
        "source_pair_span_b",
        "source_pair_span_scrambled",
        "source_pair_span_wrong_position",
        "query_endpoint_q",
        "query_nonce_span_q",
        "query_clause_span_q",
        "query_endpoint_bq",
        "query_nonce_span_bq",
        "query_clause_span_bq",
        "query_clause_span_scrambled",
        "source_pair_span_plus_query_nonce_q",
        "source_pair_span_plus_query_clause_q",
        "source_pair_span_plus_query_nonce_bq",
        "source_pair_span_plus_query_clause_bq",
    ):
        selected_rows[condition] = {
            "within_template_all": metric(
                conditions, condition, "within_template", "all"
            ),
            "next_token_all": metric(
                conditions, condition, "next_token", "all"
            ),
            "within_template_by_template": {
                str(template): metric(
                    conditions,
                    condition,
                    "within_template",
                    f"template_{template}",
                )
                for template in range(2)
            },
            "next_token_by_template": {
                str(template): metric(
                    conditions,
                    condition,
                    "next_token",
                    f"template_{template}",
                )
                for template in range(2)
            },
        }
    return {
        "model": metrics["model"],
        "selected_depths": metrics["selected_depths"],
        "clean": metrics["clean"],
        "key_conditions": selected_rows,
        "paired_comparisons": metrics["paired_comparisons"],
        "evidence": model_evidence(metrics),
        "precision": {
            "precision": summary["precision"],
            "quantization": summary["quantization"],
            "candidate_logit_source": summary["candidate_logit_source"],
            "runtime_precision_audit": summary[
                "runtime_precision_audit"
            ],
            "manual_vs_full_model_candidate_logit_max_abs": summary[
                "manual_vs_full_model_candidate_logit_max_abs"
            ],
        },
        "finiteness": summary["finiteness"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def cross_model_evidence(
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for readout in ("within_template", "next_token"):
        selected_models = [
            name for name, row in models.items()
            if row["evidence"][readout][
                "selected_source_repetition_both_templates"
            ]
        ]
        span_models = [
            name for name, row in models.items()
            if row["evidence"][readout][
                "double_span_gain_both_templates"
            ]
        ]
        query_q_models = [
            name for name, row in models.items()
            if row["evidence"][readout][
                "query_clause_gain_q_both_templates"
            ]
        ]
        query_bq_models = [
            name for name, row in models.items()
            if row["evidence"][readout][
                "query_clause_gain_bq_both_templates"
            ]
        ]
        composition = {}
        for kind in (
            "query_nonce_q",
            "query_clause_q",
            "query_nonce_bq",
            "query_clause_bq",
        ):
            passing = [
                name for name, row in models.items()
                if row["evidence"][readout][
                    "composition_restoration_both_templates"
                ][kind]
            ]
            composition[kind] = {
                "models": passing,
                "repeated_at_least_two_models": len(passing) >= 2,
            }
        result[readout] = {
            "selected_source": {
                "models": selected_models,
                "repeated_at_least_two_models": (
                    len(selected_models) >= 2
                ),
            },
            "double_span_gain": {
                "models": span_models,
                "repeated_at_least_two_models": len(span_models) >= 2,
            },
            "query_clause_gain_q": {
                "models": query_q_models,
                "repeated_at_least_two_models": (
                    len(query_q_models) >= 2
                ),
            },
            "query_clause_gain_bq": {
                "models": query_bq_models,
                "repeated_at_least_two_models": (
                    len(query_bq_models) >= 2
                ),
            },
            "composition_restoration": composition,
        }
    return result


def recommendation(cross: dict[str, Any]) -> dict[str, Any]:
    query_repeated = any(
        cross[readout][kind]["repeated_at_least_two_models"]
        for readout in cross
        for kind in ("query_clause_gain_q", "query_clause_gain_bq")
    )
    span_repeated = any(
        cross[readout]["double_span_gain"][
            "repeated_at_least_two_models"
        ]
        for readout in cross
    )
    composition_repeated = any(
        row["repeated_at_least_two_models"]
        for readout in cross.values()
        for row in readout["composition_restoration"].values()
    )
    if composition_repeated:
        route = (
            "Freeze the repeated source/query alliance and run an "
            "independent new-template replication before component "
            "decomposition."
        )
    elif query_repeated:
        route = (
            "Replicate the distributed query-clause effect on new "
            "templates, then localize which query subspan carries the gain."
        )
    elif span_repeated:
        route = (
            "Replicate the double-token source-span gain with new words, "
            "then decompose source transport into attention and MLP paths."
        )
    else:
        route = (
            "Do not repeat broader residual-span patching. Preserve the "
            "conditional source-selection result and switch to a "
            "component-resolved routing atlas that compares queried and "
            "unqueried sources without selecting highest activations."
        )
    return {
        "composition_repeated": composition_repeated,
        "query_span_repeated": query_repeated,
        "source_span_gain_repeated": span_repeated,
        "automatic_next_route": route,
    }


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def artifact_manifest() -> dict[str, Any]:
    manifest_path = protocol.OUT_ROOT / "artifact_manifest.json"
    final_audit_path = protocol.OUT_ROOT / "final_audit.json"
    files = []
    for path in sorted(
        item
        for item in protocol.OUT_ROOT.rglob("*")
        if item.is_file()
        and item not in {manifest_path, final_audit_path}
    ):
        files.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        })
    result = {
        "schema_version": "phase1032_artifact_manifest.v1",
        "phase": protocol.PHASE,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }
    protocol.write_json(manifest_path, result)
    return result


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {}
    checks = {
        "all_models_present": True,
        "protocol_digest_consistent": True,
        "all_fp16_no_quantization": True,
        "all_state_arrays_finite": True,
        "candidate_logit_min_finite_row_rate_95": True,
        "manual_logits_match_full_model": True,
        "candidate_logit_path_valid": True,
    }
    for model_name in protocol.MODELS:
        atlas_dir = protocol.OUT_ROOT / "atlas" / model_name
        metrics_path = atlas_dir / "metrics.json"
        summary_path = atlas_dir / "summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            checks["all_models_present"] = False
            continue
        metrics = protocol.read_json(metrics_path)
        summary = protocol.read_json(summary_path)
        checks["protocol_digest_consistent"] &= (
            summary["protocol_digest"] == prereg["protocol_digest"]
        )
        checks["all_fp16_no_quantization"] &= (
            summary["precision"] == "fp16"
            and summary["quantization"] == "none"
            and not summary["runtime_precision_audit"][
                "has_quantized_modules"
            ]
            and summary["runtime_precision_audit"][
                "has_fp16_parameters"
            ]
        )
        checks["all_state_arrays_finite"] &= bool(
            summary["finiteness"].get(
                "all_state_arrays_finite",
                summary["finiteness"]["all_arrays_finite"],
            )
        )
        candidate_rates = summary["finiteness"].get(
            "candidate_logit_finite_row_rates",
            {"legacy_all_finite": 1.0},
        )
        checks["candidate_logit_min_finite_row_rate_95"] &= all(
            float(rate) >= 0.95
            for rate in candidate_rates.values()
        )
        manual_error = summary[
            "manual_vs_full_model_candidate_logit_max_abs"
        ]
        if model_name == "qwen3":
            checks["manual_logits_match_full_model"] &= (
                manual_error is not None
                and float(manual_error) <= 1e-5
            )
            checks["candidate_logit_path_valid"] &= (
                summary["candidate_logit_source"]
                == "base_model_final_state_plus_output_head"
            )
        else:
            checks["candidate_logit_path_valid"] &= (
                summary["candidate_logit_source"]
                == "full_causal_lm_forward"
            )
        models[model_name] = compact_model(metrics, summary)

    if not checks["all_models_present"]:
        raise RuntimeError("not all model outputs are present")
    cross = cross_model_evidence(models)
    next_route = recommendation(cross)
    report = {
        "schema_version": "phase1032_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "models": models,
        "cross_model_evidence": cross,
        "automatic_next_decision": next_route,
        "claim_limit": prereg["claim_limit"],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", report)
    manifest = artifact_manifest()
    protocol.write_json(
        protocol.OUT_ROOT / "final_audit.json",
        {
            "schema_version": "phase1032_final_audit.v1",
            "phase": protocol.PHASE,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "artifact_manifest": {
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
            },
        },
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "checks": checks,
        "cross_model_evidence": cross,
        "automatic_next_decision": next_route,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
