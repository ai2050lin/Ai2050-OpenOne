#!/usr/bin/env python3
"""Mechanical and gate-consistency audit for formal Phase1004 results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PHASE = 1004
ROOT = (
    Path(__file__).resolve().parent
    / "result"
    / "phase1004_blind_causal_state_basis"
)
OUT_ROOT = ROOT / "audit"
MODELS = ("qwen3", "glm4", "deepseek7b")
FORMAL_ROOTS = (
    ROOT / "protocol",
    ROOT / "blind_source",
    ROOT / "blind_receiver",
    ROOT / "blind_rollout",
    ROOT / "blind_source_bf16",
    ROOT / "blind_receiver_bf16",
    ROOT / "blind_rollout_bf16",
    ROOT / "analysis",
)
RUNS = (
    {
        "precision": "8bit",
        "source": "blind_source",
        "receiver": "blind_receiver",
        "rollout": "blind_rollout",
        "models": MODELS,
    },
    {
        "precision": "bf16",
        "source": "blind_source_bf16",
        "receiver": "blind_receiver_bf16",
        "rollout": "blind_rollout_bf16",
        "models": ("qwen3",),
    },
)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: {error}"
                ) from error
    return result


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def walk_values(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_values(item)


def add_error(
    errors: list[dict[str, str]],
    check: bool,
    code: str,
    detail: str,
) -> None:
    if not check:
        errors.append({"code": code, "detail": detail})


def audit_value_tree(
    value: Any,
    path: Path,
    errors: list[dict[str, str]],
) -> tuple[int, int]:
    phase_fields = 0
    finite_numbers = 0
    for item in walk_values(value):
        if isinstance(item, float):
            finite_numbers += 1
            add_error(
                errors,
                math.isfinite(item),
                "non_finite_number",
                str(path),
            )
        if isinstance(item, dict) and "phase" in item:
            phase_fields += 1
            add_error(
                errors,
                item["phase"] == PHASE,
                "phase_mismatch",
                f"{path}: {item['phase']}",
            )
    return phase_fields, finite_numbers


def audit_protocol(
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    path = ROOT / "preregistered_protocol.json"
    protocol = read_json(path)
    expected_digest = digest({
        key: value
        for key, value in protocol.items()
        if key != "preregistration_digest"
    })
    add_error(
        errors,
        protocol["protocol_revision"] == 3,
        "protocol_revision",
        str(protocol["protocol_revision"]),
    )
    add_error(
        errors,
        protocol["preregistration_digest"] == expected_digest,
        "preregistration_digest",
        (
            f"stored={protocol['preregistration_digest']} "
            f"computed={expected_digest}"
        ),
    )
    add_error(
        errors,
        protocol["models_in_required_execution_order"]
        == list(MODELS),
        "model_order",
        str(protocol["models_in_required_execution_order"]),
    )

    model_rows = []
    for model in MODELS:
        model_root = ROOT / "protocol" / model
        cases = read_jsonl(model_root / "cases.jsonl")
        pairs = read_jsonl(model_root / "pairs.jsonl")
        audit = read_json(model_root / "protocol_audit.json")
        add_error(
            errors,
            len(cases) == 2048 == audit["case_count"],
            "case_count",
            f"{model}: {len(cases)} vs {audit['case_count']}",
        )
        add_error(
            errors,
            len(pairs) == 1024 == audit["pair_count"],
            "pair_count",
            f"{model}: {len(pairs)} vs {audit['pair_count']}",
        )
        add_error(
            errors,
            digest(cases) == audit["case_digest"],
            "case_digest",
            model,
        )
        add_error(
            errors,
            digest(pairs) == audit["pair_digest"],
            "pair_digest",
            model,
        )
        selection_rows = 0
        for domain in ("color", "shape"):
            for split in ("discovery", "confirmation"):
                selected_path = (
                    model_root
                    / f"{domain}_{split}_selected_pairs.jsonl"
                )
                selected = read_jsonl(selected_path)
                selection_rows += len(selected)
                add_error(
                    errors,
                    len(selected) == 32,
                    "selected_pair_count",
                    (
                        f"{model}/{domain}/{split}: "
                        f"{len(selected)}"
                    ),
                )
                key = f"{domain}:{split}"
                add_error(
                    errors,
                    digest(selected)
                    == audit["selection_digests"][key],
                    "selection_digest",
                    f"{model}/{key}",
                )
        add_error(
            errors,
            not audit["discovery_confirmation_name_overlap"],
            "name_split_overlap",
            model,
        )
        add_error(
            errors,
            not audit["prior_phase_name_overlap"],
            "prior_name_overlap",
            model,
        )
        add_error(
            errors,
            not audit["template_overlap"],
            "template_overlap",
            model,
        )
        add_error(
            errors,
            all(
                not values
                for values in audit[
                    "value_overlap_by_domain"
                ].values()
            ),
            "value_overlap",
            model,
        )
        add_error(
            errors,
            audit[
                "all_pair_counterfactuals_change_only_two_entity_tokens"
            ],
            "counterfactual_contract",
            model,
        )
        add_error(
            errors,
            audit["all_pairs_swap_gold_and_foil"],
            "gold_foil_contract",
            model,
        )
        model_rows.append({
            "model": model,
            "case_count": len(cases),
            "pair_count": len(pairs),
            "selected_pair_count": selection_rows,
            "case_digest_match": digest(cases)
            == audit["case_digest"],
            "pair_digest_match": digest(pairs)
            == audit["pair_digest"],
        })
    return {
        "protocol_revision": protocol["protocol_revision"],
        "stored_digest": protocol["preregistration_digest"],
        "computed_digest": expected_digest,
        "digest_match": protocol["preregistration_digest"]
        == expected_digest,
        "models": model_rows,
    }


def audit_donor(
    value: dict[str, Any],
    context: str,
    errors: list[dict[str, str]],
) -> None:
    add_error(
        errors,
        value["candidate_pool_source"]
        == "complete_frozen_protocol_model_domain_split_template",
        "donor_pool_source",
        context,
    )
    add_error(
        errors,
        int(value["candidate_pool_count"]) == 256,
        "donor_pool_count",
        f"{context}: {value['candidate_pool_count']}",
    )
    add_error(
        errors,
        int(value["recipient_count"]) == 32,
        "donor_recipient_count",
        f"{context}: {value['recipient_count']}",
    )
    add_error(
        errors,
        int(value["unique_donor_count"]) == 32,
        "unique_donor_count",
        f"{context}: {value['unique_donor_count']}",
    )
    add_error(
        errors,
        float(value["unique_donor_fraction"]) == 1.0,
        "unique_donor_fraction",
        context,
    )
    add_error(
        errors,
        int(value["maximum_donor_reuse"]) == 1,
        "maximum_donor_reuse",
        context,
    )
    add_error(
        errors,
        bool(value["all_cross_world"]),
        "cross_world_donor",
        context,
    )
    add_error(
        errors,
        bool(value["all_answer_contracts_hold"]),
        "answer_donor_contract",
        context,
    )
    if not value["same_answer_control"]:
        add_error(
            errors,
            bool(value["different_answer_value_sets_disjoint"]),
            "different_answer_value_disjoint",
            context,
        )


def recompute_source_gate(cell: dict[str, Any]) -> bool:
    source = cell["final_conditions"]["frozen_source"]
    return (
        float(source["donor_rate"]) >= 0.80
        and float(source["median_normalized_transfer"]) >= 0.50
        and float(cell["noop_prediction_agreement"]) >= 0.99
        and float(cell["same_answer_control_target_rate"]) >= 0.95
    )


def audit_source_run(
    source_root: str,
    model: str,
    precision: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    root = ROOT / source_root / model
    summary = read_json(root / "summary.json")
    add_error(
        errors,
        summary["precision"] == precision,
        "source_precision",
        f"{model}: {summary['precision']} vs {precision}",
    )
    expected_cells = 2 * sum(
        bool(value["gate_pass"])
        for value in summary["behavior"].values()
    )
    add_error(
        errors,
        int(summary["cell_count"]) == expected_cells,
        "source_cell_count_from_behavior",
        (
            f"{precision}/{model}: {summary['cell_count']} "
            f"vs {expected_cells}"
        ),
    )
    for key, behavior in summary["behavior"].items():
        recomputed = float(behavior["candidate_accuracy"]) >= 0.95
        add_error(
            errors,
            bool(behavior["gate_pass"]) == recomputed,
            "behavior_gate_consistency",
            f"{precision}/{model}/{key}",
        )
        add_error(
            errors,
            int(behavior["n"]) == 64,
            "behavior_n",
            f"{precision}/{model}/{key}: {behavior['n']}",
        )

    pass_count = 0
    exact_count = 0
    trajectory_rows = 0
    for cell in summary["cells"]:
        context = (
            f"{precision}/{model}/{cell['domain']}/"
            f"{cell['split']}/t{cell['template']}"
        )
        add_error(
            errors,
            int(cell["n"]) == 32,
            "source_cell_n",
            f"{context}: {cell['n']}",
        )
        add_error(
            errors,
            int(cell["source_depth"]) == 1,
            "source_depth",
            context,
        )
        add_error(
            errors,
            not cell["selection_uses_semantic_labels"],
            "semantic_selection_leak",
            context,
        )
        add_error(
            errors,
            not cell["selection_uses_confirmation_to_tune_rule"],
            "confirmation_selection_leak",
            context,
        )
        add_error(
            errors,
            all(
                0 <= int(position) < int(cell["position_count"])
                for position in cell["frozen_physical_positions"]
            ),
            "source_position_outside_prompt",
            context,
        )
        add_error(
            errors,
            len(cell["frozen_physical_positions"])
            == int(cell["frozen_position_count"]),
            "frozen_position_count",
            context,
        )
        audit_donor(cell["donor_audit"], context, errors)
        audit_donor(
            cell["same_answer_donor_audit"],
            context + "/same",
            errors,
        )
        gate = recompute_source_gate(cell)
        add_error(
            errors,
            bool(cell["final_source_gate_pass"]) == gate,
            "source_gate_consistency",
            context,
        )
        pass_count += gate
        semantic = cell["semantic_reconstruction_audit"]
        add_error(
            errors,
            semantic["revealed_after_selection"]
            and not semantic["selection_uses_this_audit"],
            "semantic_reveal_order",
            context,
        )
        exact_count += (
            float(semantic["exact_five_anchor_position_set_rate"])
            >= 0.99
        )
        trajectory_path = (
            root
            / cell["domain"]
            / cell["split"]
            / f"template_{cell['template']}"
            / "diagnostic_trajectory_rows.jsonl"
        )
        actual_trajectory_rows = count_jsonl(trajectory_path)
        trajectory_rows += actual_trajectory_rows
        add_error(
            errors,
            actual_trajectory_rows == int(cell["trajectory_row_count"]),
            "trajectory_row_count",
            (
                f"{context}: {actual_trajectory_rows} vs "
                f"{cell['trajectory_row_count']}"
            ),
        )
        final_path = trajectory_path.with_name("final_rows.jsonl")
        add_error(
            errors,
            count_jsonl(final_path) == 3 * int(cell["n"]),
            "final_source_row_count",
            context,
        )
    add_error(
        errors,
        pass_count == int(summary["source_gate_pass_count"]),
        "source_pass_count",
        f"{precision}/{model}: {pass_count}",
    )
    add_error(
        errors,
        exact_count
        == int(summary["semantic_exact_reconstruction_count"]),
        "semantic_exact_count",
        f"{precision}/{model}: {exact_count}",
    )
    return {
        "precision": precision,
        "model": model,
        "behavior_group_count": len(summary["behavior"]),
        "behavior_group_pass_count": sum(
            bool(value["gate_pass"])
            for value in summary["behavior"].values()
        ),
        "source_cell_count": int(summary["cell_count"]),
        "source_gate_pass_count": pass_count,
        "semantic_exact_count": exact_count,
        "trajectory_row_count": trajectory_rows,
    }


def event_repeats(
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
) -> bool:
    templates_positive = all(
        float(metric["median_mediation_fraction"]) >= 0.10
        and float(metric["mean_sufficiency_transfer"]) >= 0.10
        for metric in confirmation["template_metrics"].values()
    )
    return (
        float(discovery["median_mediation_fraction"]) >= 0.10
        and float(discovery["mean_sufficiency_transfer"]) >= 0.10
        and float(confirmation["median_mediation_fraction"]) >= 0.10
        and float(confirmation["mean_sufficiency_transfer"]) >= 0.10
        and templates_positive
    )


def audit_receiver_run(
    receiver_root: str,
    model: str,
    precision: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    summary = read_json(
        ROOT / receiver_root / model / "summary.json"
    )
    add_error(
        errors,
        summary["precision"] == precision,
        "receiver_precision",
        f"{model}: {summary['precision']} vs {precision}",
    )
    repeated_domains = 0
    qualified_domains = 0
    repeated_events = 0
    repeated_attention = 0
    for domain, value in summary["domains"].items():
        context = f"{precision}/{model}/{domain}"
        if value["status"] != "complete":
            add_error(
                errors,
                not value["receiver_scan_run"],
                "failed_parent_receiver_scan",
                context,
            )
            add_error(
                errors,
                any(
                    not cell["final_source_gate_pass"]
                    for cell in value["source_cells"].values()
                ),
                "receiver_skip_without_failed_source",
                context,
            )
            continue
        qualified_domains += 1
        add_error(
            errors,
            bool(value["source_parent_gate_pass"]),
            "complete_receiver_without_parent",
            context,
        )
        expected_repeated = []
        for discovery in value["frozen_events"]:
            event_id = discovery["event_id"]
            confirmation = value["confirmation_metrics"][event_id]
            if event_repeats(discovery, confirmation):
                expected_repeated.append(event_id)
            for metrics in (discovery, confirmation):
                add_error(
                    errors,
                    float(
                        metrics[
                            "receiver_noop_prediction_agreement"
                        ]
                    )
                    >= 0.99,
                    "receiver_noop_agreement",
                    f"{context}/{event_id}",
                )
                add_error(
                    errors,
                    float(
                        metrics[
                            "maximum_receiver_noop_candidate_logit_error"
                        ]
                    )
                    <= 1e-6,
                    "receiver_noop_logit_error",
                    f"{context}/{event_id}",
                )
            add_error(
                errors,
                not discovery["selection_uses_confirmation"]
                and not discovery["selection_uses_semantic_labels"],
                "receiver_selection_leak",
                f"{context}/{event_id}",
            )
        stored_repeated = [
            item["event_id"] for item in value["repeated_events"]
        ]
        add_error(
            errors,
            expected_repeated == stored_repeated,
            "repeated_event_consistency",
            (
                f"{context}: expected={expected_repeated} "
                f"stored={stored_repeated}"
            ),
        )
        stored_attention = [
            item["event_id"]
            for item in value["repeated_events"]
            if item["component"] == "attn"
        ]
        add_error(
            errors,
            int(value["repeated_attention_event_count"])
            == len(stored_attention),
            "repeated_attention_count",
            context,
        )
        add_error(
            errors,
            bool(value["head_subspace_parent_authorized"])
            == bool(stored_attention),
            "head_subspace_authorization",
            context,
        )
        repeated_events += len(stored_repeated)
        repeated_attention += len(stored_attention)
        repeated_domains += bool(stored_repeated)
    add_error(
        errors,
        repeated_domains == int(summary["repeated_domain_count"]),
        "repeated_domain_count",
        f"{precision}/{model}: {repeated_domains}",
    )
    return {
        "precision": precision,
        "model": model,
        "qualified_domain_count": qualified_domains,
        "repeated_domain_count": repeated_domains,
        "repeated_event_count": repeated_events,
        "repeated_attention_event_count": repeated_attention,
    }


def recompute_rollout_gate(cell: dict[str, Any]) -> bool:
    conditions = cell["conditions"]
    return (
        float(
            conditions["clean_target"]["target_semantic_rate"]
        )
        >= 0.95
        and float(
            conditions["frozen_source"]["donor_semantic_rate"]
        )
        >= 0.70
        and float(
            conditions["frozen_source"][
                "donor_eos_boundary_rate"
            ]
        )
        >= 0.95
        and float(
            conditions["frozen_target_noop"][
                "noop_sequence_agreement"
            ]
        )
        >= 0.99
        and float(
            conditions["frozen_same_answer_control"][
                "target_semantic_rate"
            ]
        )
        >= 0.95
    )


def audit_rollout_run(
    rollout_root: str,
    model: str,
    precision: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    summary = read_json(
        ROOT / rollout_root / model / "summary.json"
    )
    add_error(
        errors,
        summary["precision"] == precision,
        "rollout_precision",
        f"{model}: {summary['precision']} vs {precision}",
    )
    add_error(
        errors,
        int(summary["cell_count"]) == 4 == len(summary["cells"]),
        "rollout_cell_count",
        f"{precision}/{model}: {summary['cell_count']}",
    )
    tested = 0
    passed = 0
    for cell in summary["cells"]:
        context = (
            f"{precision}/{model}/{cell['domain']}/"
            f"t{cell['template']}"
        )
        if cell.get("status") == "source_parent_gate_failed":
            add_error(
                errors,
                not cell["rollout_gate_pass"]
                and "conditions" not in cell,
                "skipped_rollout_shape",
                context,
            )
            continue
        tested += 1
        audit_donor(cell["donor_audit"], context, errors)
        audit_donor(
            cell["same_answer_donor_audit"],
            context + "/same",
            errors,
        )
        gate = recompute_rollout_gate(cell)
        passed += gate
        add_error(
            errors,
            bool(cell["rollout_gate_pass"]) == gate,
            "rollout_gate_consistency",
            context,
        )
        add_error(
            errors,
            int(cell["n"]) == 32,
            "rollout_n",
            f"{context}: {cell['n']}",
        )
    add_error(
        errors,
        passed == int(summary["rollout_gate_pass_count"]),
        "rollout_pass_count",
        f"{precision}/{model}: {passed}",
    )
    return {
        "precision": precision,
        "model": model,
        "tested_cell_count": tested,
        "rollout_gate_pass_count": passed,
    }


def audit_formal_files(
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    json_files = []
    jsonl_files = []
    for root in FORMAL_ROOTS:
        if not root.exists():
            errors.append({
                "code": "missing_formal_root",
                "detail": str(root),
            })
            continue
        json_files.extend(root.rglob("*.json"))
        jsonl_files.extend(root.rglob("*.jsonl"))
    json_files.append(ROOT / "preregistered_protocol.json")
    json_files = sorted(set(json_files))
    jsonl_files = sorted(set(jsonl_files))

    phase_fields = 0
    finite_numbers = 0
    jsonl_rows = 0
    for path in json_files:
        value = read_json(path)
        phases, numbers = audit_value_tree(value, path, errors)
        phase_fields += phases
        finite_numbers += numbers
    for path in jsonl_files:
        for value in read_jsonl(path):
            jsonl_rows += 1
            phases, numbers = audit_value_tree(value, path, errors)
            phase_fields += phases
            finite_numbers += numbers
    return {
        "json_file_count": len(json_files),
        "jsonl_file_count": len(jsonl_files),
        "jsonl_row_count": jsonl_rows,
        "phase_field_count": phase_fields,
        "finite_float_count": finite_numbers,
    }


def main() -> None:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    protocol = audit_protocol(errors)
    source_runs = []
    receiver_runs = []
    rollout_runs = []
    for run in RUNS:
        for model in run["models"]:
            source_runs.append(audit_source_run(
                run["source"],
                model,
                run["precision"],
                errors,
            ))
            receiver_runs.append(audit_receiver_run(
                run["receiver"],
                model,
                run["precision"],
                errors,
            ))
            rollout_runs.append(audit_rollout_run(
                run["rollout"],
                model,
                run["precision"],
                errors,
            ))

    archived_roots = sorted(
        path.name
        for path in ROOT.iterdir()
        if (
            path.is_dir()
            and (
                "_pre_" in path.name
                or path.name.startswith("blind_source_pre")
            )
        )
    )
    if archived_roots:
        warnings.append({
            "code": "archived_nonformal_attempts_excluded",
            "detail": ", ".join(archived_roots),
        })
    formal_files = audit_formal_files(errors)
    summary = {
        "schema_version": "phase1004_result_audit.v1",
        "phase": PHASE,
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "protocol": protocol,
        "source_runs": source_runs,
        "receiver_runs": receiver_runs,
        "rollout_runs": rollout_runs,
        "formal_files": formal_files,
        "excluded_archived_roots": archived_roots,
        "claim_boundary": (
            "A mechanical pass establishes file, protocol, donor, "
            "gate, and no-op consistency. It does not establish a "
            "language theory or prove that an intervention is the "
            "native algorithm used by the model."
        ),
    }
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
