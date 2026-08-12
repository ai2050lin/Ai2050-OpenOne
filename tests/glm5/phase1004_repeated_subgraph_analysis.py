#!/usr/bin/env python3
"""Aggregate Phase1004 without changing any preregistered gate."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PHASE = 1004
ROOT = (
    Path(__file__).resolve().parent
    / "result"
    / "phase1004_blind_causal_state_basis"
)
OUT_ROOT = ROOT / "analysis"
MODELS = ("qwen3", "glm4", "deepseek7b")
ROLE_ORDER = (
    "query_name",
    "slot0_entity",
    "slot0_value",
    "slot1_entity",
    "slot1_value",
)
PRECISION_RUNS = (
    {
        "precision": "8bit",
        "source_root": "blind_source",
        "receiver_root": "blind_receiver",
        "rollout_root": "blind_rollout",
        "models": MODELS,
    },
    {
        "precision": "bf16",
        "source_root": "blind_source_bf16",
        "receiver_root": "blind_receiver_bf16",
        "rollout_root": "blind_rollout_bf16",
        "models": ("qwen3",),
    },
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def role_signature(role_coverage: dict[str, float]) -> str:
    return "|".join(
        f"{role}={float(role_coverage.get(role, 0.0)):.2f}"
        for role in ROLE_ORDER
    )


def source_rows_for_run(
    precision: str,
    source_root: str,
    models: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for model in models:
        summary = read_json(ROOT / source_root / model / "summary.json")
        behavior_n = sum(
            int(item["n"]) for item in summary["behavior"].values()
        )
        behavior_correct = sum(
            float(item["candidate_accuracy"]) * int(item["n"])
            for item in summary["behavior"].values()
        )
        cells = summary["cells"]
        model_rows = []
        for cell in cells:
            semantic = cell["semantic_reconstruction_audit"]
            source = cell["final_conditions"]["frozen_source"]
            same = cell["final_conditions"][
                "frozen_same_answer_control"
            ]
            row = {
                "schema_version": "phase1004_source_cell_aggregate.v1",
                "phase": PHASE,
                "precision": precision,
                "model": model,
                "domain": cell["domain"],
                "split": cell["split"],
                "template": int(cell["template"]),
                "n": int(cell["n"]),
                "source_gate_pass": bool(
                    cell["final_source_gate_pass"]
                ),
                "frozen_position_count": int(
                    cell["frozen_position_count"]
                ),
                "frozen_physical_positions": cell[
                    "frozen_physical_positions"
                ],
                "donor_rate": float(source["donor_rate"]),
                "median_normalized_transfer": float(
                    source["median_normalized_transfer"]
                ),
                "same_answer_target_rate": float(
                    same["target_rate"]
                ),
                "noop_prediction_agreement": float(
                    cell["noop_prediction_agreement"]
                ),
                "anchor_precision": float(
                    semantic["mean_anchor_precision"]
                ),
                "anchor_recall": float(
                    semantic["mean_anchor_recall"]
                ),
                "exact_five_anchor_rate": float(
                    semantic["exact_five_anchor_position_set_rate"]
                ),
                "role_coverage_rate": semantic["role_coverage_rate"],
                "role_signature": role_signature(
                    semantic["role_coverage_rate"]
                ),
            }
            rows.append(row)
            model_rows.append(row)
        model_summaries.append({
            "schema_version": "phase1004_source_model_aggregate.v1",
            "phase": PHASE,
            "precision": precision,
            "model": model,
            "behavior_group_count": len(summary["behavior"]),
            "behavior_group_pass_count": sum(
                bool(item["gate_pass"])
                for item in summary["behavior"].values()
            ),
            "behavior_case_count": behavior_n,
            "behavior_candidate_accuracy": (
                behavior_correct / behavior_n
                if behavior_n
                else None
            ),
            "source_cell_count": len(model_rows),
            "source_gate_pass_count": sum(
                row["source_gate_pass"] for row in model_rows
            ),
            "exact_five_anchor_cell_count": sum(
                row["exact_five_anchor_rate"] >= 0.99
                for row in model_rows
            ),
            "mean_frozen_position_count": mean([
                float(row["frozen_position_count"])
                for row in model_rows
            ]),
            "mean_anchor_precision": mean([
                row["anchor_precision"] for row in model_rows
            ]),
            "mean_anchor_recall": mean([
                row["anchor_recall"] for row in model_rows
            ]),
        })
    return rows, model_summaries


def summarize_source_signatures(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["precision"] == "8bit":
            grouped[row["role_signature"]].append(row)
    result = []
    for signature, values in grouped.items():
        models = sorted({item["model"] for item in values})
        domains = sorted({item["domain"] for item in values})
        splits = sorted({item["split"] for item in values})
        result.append({
            "role_signature": signature,
            "occurrence_count": len(values),
            "source_gate_pass_count": sum(
                item["source_gate_pass"] for item in values
            ),
            "models": models,
            "domains": domains,
            "splits": splits,
            "templates": sorted({
                int(item["template"]) for item in values
            }),
            "repeated_across_models_domains_and_splits": (
                len(models) >= 2
                and len(domains) >= 2
                and {"discovery", "confirmation"}.issubset(splits)
            ),
        })
    return sorted(
        result,
        key=lambda item: (
            not item["repeated_across_models_domains_and_splits"],
            -item["occurrence_count"],
            item["role_signature"],
        ),
    )


def receiver_rows_for_run(
    precision: str,
    receiver_root: str,
    models: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for model in models:
        summary = read_json(
            ROOT / receiver_root / model / "summary.json"
        )
        for domain, domain_summary in summary["domains"].items():
            domain_row = {
                "schema_version": (
                    "phase1004_receiver_domain_aggregate.v1"
                ),
                "phase": PHASE,
                "precision": precision,
                "model": model,
                "domain": domain,
                "status": domain_summary["status"],
                "source_parent_gate_pass": bool(
                    domain_summary.get(
                        "source_parent_gate_pass", False
                    )
                ),
                "repeated_event_count": int(
                    domain_summary.get("repeated_event_count", 0)
                ),
                "repeated_attention_event_count": int(
                    domain_summary.get(
                        "repeated_attention_event_count", 0
                    )
                ),
                "head_subspace_parent_authorized": bool(
                    domain_summary.get(
                        "head_subspace_parent_authorized", False
                    )
                ),
            }
            domain_rows.append(domain_row)
            if domain_summary["status"] != "complete":
                continue
            repeated_ids = {
                item["event_id"]
                for item in domain_summary["repeated_events"]
            }
            confirmation = domain_summary["confirmation_metrics"]
            for discovery in domain_summary["frozen_events"]:
                event_id = discovery["event_id"]
                confirmed = confirmation[event_id]
                rows.append({
                    "schema_version": (
                        "phase1004_receiver_event_aggregate.v1"
                    ),
                    "phase": PHASE,
                    "precision": precision,
                    "model": model,
                    "domain": domain,
                    "event_id": event_id,
                    "component": discovery["component"],
                    "checkpoint_index": int(
                        discovery["checkpoint_index"]
                    ),
                    "relative_depth": float(
                        discovery["relative_depth"]
                    ),
                    "depth_half": discovery["depth_half"],
                    "discovery_rank": int(
                        discovery["discovery_rank"]
                    ),
                    "repeated_event": event_id in repeated_ids,
                    "discovery_median_mediation": float(
                        discovery["median_mediation_fraction"]
                    ),
                    "discovery_mean_sufficiency": float(
                        discovery["mean_sufficiency_transfer"]
                    ),
                    "confirmation_median_mediation": float(
                        confirmed["median_mediation_fraction"]
                    ),
                    "confirmation_mean_sufficiency": float(
                        confirmed["mean_sufficiency_transfer"]
                    ),
                    "confirmation_sufficiency_flip_rate": float(
                        confirmed["sufficiency_flip_rate"]
                    ),
                    "confirmation_restored_to_target_rate": float(
                        confirmed["restored_to_target_rate"]
                    ),
                })
    return rows, domain_rows


def repeated_receiver_classes(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row["precision"] == "8bit" and row["repeated_event"]:
            grouped[(row["event_id"], row["component"])].append(row)
    result = []
    for (event_id, component), values in grouped.items():
        models = sorted({item["model"] for item in values})
        domains = sorted({item["domain"] for item in values})
        result.append({
            "event_id": event_id,
            "component": component,
            "models": models,
            "domains": domains,
            "occurrence_count": len(values),
            "cross_model_repeated": len(models) >= 2,
            "cross_domain_repeated": len(domains) >= 2,
        })
    return sorted(result, key=lambda item: item["event_id"])


def trajectory_records(
    precision: str,
    source_root: str,
    models: Iterable[str],
) -> list[dict[str, Any]]:
    output = []
    for model in models:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in (ROOT / source_root / model).rglob(
            "diagnostic_trajectory_rows.jsonl"
        ):
            for row in read_jsonl(path):
                grouped[row["record_id"]].append(row)
        for record_id, rows in grouped.items():
            rows.sort(key=lambda item: int(item["depth"]))
            stable_depth = None
            for index, row in enumerate(rows):
                if (
                    int(row["target_rank"]) == 1
                    and all(
                        int(item["target_rank"]) == 1
                        for item in rows[index:]
                    )
                ):
                    stable_depth = float(row["relative_depth"])
                    break
            entropy_changes = [
                (
                    float(rows[index - 1]["candidate_panel_entropy"])
                    - float(rows[index]["candidate_panel_entropy"]),
                    float(rows[index]["relative_depth"]),
                )
                for index in range(1, len(rows))
            ]
            largest_drop, largest_drop_depth = max(
                entropy_changes,
                key=lambda item: item[0],
            )
            top_token_switches = sum(
                int(rows[index]["top_token_id"])
                != int(rows[index - 1]["top_token_id"])
                for index in range(1, len(rows))
            )
            first = rows[0]
            output.append({
                "schema_version": (
                    "phase1004_trajectory_record_aggregate.v1"
                ),
                "phase": PHASE,
                "precision": precision,
                "model": model,
                "domain": first["domain"],
                "split": first["split"],
                "template": int(first["template"]),
                "record_id": record_id,
                "depth_count": len(rows),
                "earliest_stable_target_top1_relative_depth": (
                    stable_depth
                ),
                "largest_candidate_entropy_drop": largest_drop,
                "largest_candidate_entropy_drop_relative_depth": (
                    largest_drop_depth
                ),
                "top_token_switch_count": top_token_switches,
                "final_target_rank": int(rows[-1]["target_rank"]),
                "final_candidate_panel_entropy": float(
                    rows[-1]["candidate_panel_entropy"]
                ),
                "observer": first["observer"],
                "observer_allowed_for_causal_selection": bool(
                    first["observer_allowed_for_causal_selection"]
                ),
                "observer_is_native_intermediate_probability": bool(
                    first["observer_is_native_intermediate_probability"]
                ),
            })
    return output


def trajectory_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(row["precision"], row["model"])].append(row)
    result = []
    for (precision, model), values in sorted(grouped.items()):
        stable = [
            item["earliest_stable_target_top1_relative_depth"]
            for item in values
            if item[
                "earliest_stable_target_top1_relative_depth"
            ]
            is not None
        ]
        result.append({
            "precision": precision,
            "model": model,
            "record_count": len(values),
            "stable_target_top1_rate": len(stable) / len(values),
            "median_earliest_stable_target_top1_relative_depth": (
                median(stable)
            ),
            "median_largest_entropy_drop_relative_depth": median([
                item[
                    "largest_candidate_entropy_drop_relative_depth"
                ]
                for item in values
            ]),
            "median_top_token_switch_count": median([
                float(item["top_token_switch_count"])
                for item in values
            ]),
            "observer_allowed_for_causal_selection": False,
            "interpretation_boundary": (
                "Raw normalized logit-lens trajectories are "
                "descriptive observers, not native intermediate "
                "probabilities or causal selectors."
            ),
        })
    return result


def rollout_rows_for_run(
    precision: str,
    rollout_root: str,
    models: Iterable[str],
) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        summary = read_json(
            ROOT / rollout_root / model / "summary.json"
        )
        for cell in summary["cells"]:
            row = {
                "schema_version": (
                    "phase1004_rollout_cell_aggregate.v1"
                ),
                "phase": PHASE,
                "precision": precision,
                "model": model,
                "domain": cell["domain"],
                "template": int(cell["template"]),
                "status": cell.get("status", "tested"),
                "rollout_gate_pass": bool(
                    cell["rollout_gate_pass"]
                ),
            }
            if "conditions" in cell:
                conditions = cell["conditions"]
                row.update({
                    "clean_target_semantic_rate": float(
                        conditions["clean_target"][
                            "target_semantic_rate"
                        ]
                    ),
                    "source_donor_semantic_rate": float(
                        conditions["frozen_source"][
                            "donor_semantic_rate"
                        ]
                    ),
                    "source_donor_eos_boundary_rate": float(
                        conditions["frozen_source"][
                            "donor_eos_boundary_rate"
                        ]
                    ),
                    "same_answer_target_semantic_rate": float(
                        conditions["frozen_same_answer_control"][
                            "target_semantic_rate"
                        ]
                    ),
                    "noop_sequence_agreement": float(
                        conditions["frozen_target_noop"][
                            "noop_sequence_agreement"
                        ]
                    ),
                })
            rows.append(row)
    return rows


def rollout_repeats(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row["precision"] == "8bit":
            grouped[(row["domain"], row["template"])].append(row)
    result = []
    for (domain, template), values in sorted(grouped.items()):
        tested = [
            item for item in values if item["status"] == "tested"
        ]
        passing = [
            item for item in tested if item["rollout_gate_pass"]
        ]
        result.append({
            "domain": domain,
            "template": template,
            "model_count": len(values),
            "tested_model_count": len(tested),
            "passing_model_count": len(passing),
            "passing_models": sorted({
                item["model"] for item in passing
            }),
            "cross_model_repeat": len({
                item["model"] for item in passing
            }) >= 2,
        })
    return result


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def main() -> None:
    all_source_rows: list[dict[str, Any]] = []
    source_model_summaries: list[dict[str, Any]] = []
    all_receiver_rows: list[dict[str, Any]] = []
    receiver_domain_rows: list[dict[str, Any]] = []
    all_trajectory_rows: list[dict[str, Any]] = []
    all_rollout_rows: list[dict[str, Any]] = []

    for run in PRECISION_RUNS:
        source_rows, model_summaries = source_rows_for_run(
            run["precision"],
            run["source_root"],
            run["models"],
        )
        all_source_rows.extend(source_rows)
        source_model_summaries.extend(model_summaries)
        receiver_rows, domain_rows = receiver_rows_for_run(
            run["precision"],
            run["receiver_root"],
            run["models"],
        )
        all_receiver_rows.extend(receiver_rows)
        receiver_domain_rows.extend(domain_rows)
        all_trajectory_rows.extend(trajectory_records(
            run["precision"],
            run["source_root"],
            run["models"],
        ))
        all_rollout_rows.extend(rollout_rows_for_run(
            run["precision"],
            run["rollout_root"],
            run["models"],
        ))

    source_signatures = summarize_source_signatures(
        all_source_rows
    )
    receiver_classes = repeated_receiver_classes(
        all_receiver_rows
    )
    trajectory_summaries = trajectory_summary(
        all_trajectory_rows
    )
    rollout_repeat_rows = rollout_repeats(all_rollout_rows)

    formal_8bit_source = [
        row for row in all_source_rows if row["precision"] == "8bit"
    ]
    formal_8bit_receiver_domains = [
        row
        for row in receiver_domain_rows
        if row["precision"] == "8bit"
    ]
    formal_8bit_rollout = [
        row for row in all_rollout_rows if row["precision"] == "8bit"
    ]
    tested_rollout = [
        row
        for row in formal_8bit_rollout
        if row["status"] == "tested"
    ]
    repeated_role_signatures = [
        item
        for item in source_signatures
        if item["repeated_across_models_domains_and_splits"]
    ]
    cross_model_receiver = [
        item
        for item in receiver_classes
        if item["cross_model_repeated"]
    ]
    cross_model_rollout = [
        item
        for item in rollout_repeat_rows
        if item["cross_model_repeat"]
    ]
    qwen_cross_precision_events = sorted(
        set(
            row["event_id"]
            for row in all_receiver_rows
            if (
                row["model"] == "qwen3"
                and row["repeated_event"]
                and row["precision"] == "8bit"
            )
        )
        & set(
            row["event_id"]
            for row in all_receiver_rows
            if (
                row["model"] == "qwen3"
                and row["repeated_event"]
                and row["precision"] == "bf16"
            )
        )
    )

    summary = {
        "schema_version": "phase1004_repeated_subgraph_analysis.v1",
        "phase": PHASE,
        "analysis_order": (
            "Discover repeated physical intervention structures "
            "first; interpret roles and formulas only afterward."
        ),
        "source_8bit": {
            "behavior_group_count": 12,
            "behavior_group_pass_count": sum(
                item["behavior_group_pass_count"]
                for item in source_model_summaries
                if item["precision"] == "8bit"
            ),
            "behavior_case_count": sum(
                item["behavior_case_count"]
                for item in source_model_summaries
                if item["precision"] == "8bit"
            ),
            "source_cell_count": len(formal_8bit_source),
            "source_gate_pass_count": sum(
                row["source_gate_pass"]
                for row in formal_8bit_source
            ),
            "exact_five_anchor_cell_count": sum(
                row["exact_five_anchor_rate"] >= 0.99
                for row in formal_8bit_source
            ),
            "mean_anchor_precision": mean([
                row["anchor_precision"]
                for row in formal_8bit_source
            ]),
            "mean_anchor_recall": mean([
                row["anchor_recall"] for row in formal_8bit_source
            ]),
            "mean_frozen_position_count": mean([
                float(row["frozen_position_count"])
                for row in formal_8bit_source
            ]),
            "repeated_role_signatures": repeated_role_signatures,
        },
        "source_model_summaries": source_model_summaries,
        "source_role_signatures": source_signatures,
        "receiver_8bit": {
            "domain_count": len(formal_8bit_receiver_domains),
            "source_parent_qualified_domain_count": sum(
                row["source_parent_gate_pass"]
                for row in formal_8bit_receiver_domains
            ),
            "repeated_event_domain_count": sum(
                row["repeated_event_count"] > 0
                for row in formal_8bit_receiver_domains
            ),
            "repeated_attention_domain_count": sum(
                row["repeated_attention_event_count"] > 0
                for row in formal_8bit_receiver_domains
            ),
            "head_subspace_authorized_domain_count": sum(
                row["head_subspace_parent_authorized"]
                for row in formal_8bit_receiver_domains
            ),
            "repeated_event_classes": receiver_classes,
            "cross_model_repeated_event_classes": (
                cross_model_receiver
            ),
            "qwen_cross_precision_repeated_event_ids": (
                qwen_cross_precision_events
            ),
        },
        "rollout_8bit": {
            "cell_count": len(formal_8bit_rollout),
            "tested_cell_count": len(tested_rollout),
            "gate_pass_count": sum(
                row["rollout_gate_pass"] for row in tested_rollout
            ),
            "template_repeats": rollout_repeat_rows,
            "cross_model_repeats": cross_model_rollout,
        },
        "trajectory_diagnostic": {
            "summaries": trajectory_summaries,
            "claim_boundary": (
                "The trajectory observer did not select causal "
                "events. Stable top-1 depth, entropy-drop depth, "
                "and token-switch counts are descriptive only."
            ),
        },
        "evidence_classification": {
            "repeated_label_blind_source_role_topology_found": bool(
                repeated_role_signatures
            ),
            "cross_precision_late_residual_chain_found": bool(
                qwen_cross_precision_events
            ),
            "cross_model_internal_receiver_subgraph_found": bool(
                cross_model_receiver
            ),
            "repeated_attention_parent_found": any(
                row["repeated_attention_event_count"] > 0
                for row in receiver_domain_rows
            ),
            "head_or_subspace_decomposition_authorized": any(
                row["head_subspace_parent_authorized"]
                for row in receiver_domain_rows
            ),
            "cross_model_natural_rollout_repeat_found": bool(
                cross_model_rollout
            ),
            "mechanism_formula_authorized": False,
            "complete_language_mechanism_claim_authorized": False,
            "current_result": (
                "A repeated prompt-state transport skeleton was "
                "found. A cross-model internal causal subgraph was "
                "not found, and the late residual chain is not yet "
                "a component-level mechanism."
            ),
            "automatic_next_step": (
                "Do not decompose attention heads. Test whether the "
                "4-5 input-anchor state compresses into a smaller "
                "label-blind causal source at later layers while "
                "holding surface controls fixed."
            ),
        },
    }
    if not finite_tree(summary):
        raise RuntimeError("Non-finite value in aggregate summary")

    write_jsonl(OUT_ROOT / "source_cell_rows.jsonl", all_source_rows)
    write_jsonl(
        OUT_ROOT / "receiver_event_rows.jsonl", all_receiver_rows
    )
    write_jsonl(
        OUT_ROOT / "receiver_domain_rows.jsonl", receiver_domain_rows
    )
    write_jsonl(
        OUT_ROOT / "trajectory_record_rows.jsonl",
        all_trajectory_rows,
    )
    write_jsonl(OUT_ROOT / "rollout_cell_rows.jsonl", all_rollout_rows)
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
