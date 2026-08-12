from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase1148_mandatory_mediation_calibration as p1148
import phase1150_role_factorized_independent_replication as p1150


PHASE = 1151
ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1151_counterfactual_role_state_transplant"
PREREG_PATH = OUT_ROOT / "protocol" / "preregistration.json"


def protocol_body() -> dict[str, Any]:
    parent = p1148.read_json(p1150.PREREG_PATH)
    p1150.verify_preregistration(parent)
    confirmation = p1148.read_json(p1150.OUT_ROOT / "analysis" / "confirmation_selection.json")
    final = p1148.read_json(p1150.OUT_ROOT / "analysis" / "final.json")
    if not confirmation["phase1151_authorized"] or not final["auto_continue"]:
        raise RuntimeError("Phase1151 is not authorized by Phase1150")
    return {
        "phase": PHASE,
        "title": "Counterfactual transplant of planted entity-role and field-role states",
        "claim_scope": (
            "Tests whether role-position states from an independently confirmed planted mediator are "
            "sufficient to transplant row and column addresses with a predicted double dissociation. "
            "It does not test a free Transformer or a pretrained language model."
        ),
        "parent_phase1150_protocol_digest": parent["protocol_digest"],
        "parent_phase1150_confirmation_digest": confirmation["selection_digest"],
        "parent_phase1150_final_digest": final["final_digest"],
        "source_hashes": {
            "primary_script": p1148.file_sha256(Path(__file__).resolve()),
            "phase1150_dependency": p1148.file_sha256(Path(p1150.__file__).resolve()),
        },
        "replicates": p1150.split_replicates(parent, "confirmation"),
        "condition": "role_factorized",
        "splits": list(parent["contingent_phase1151"]["splits"]),
        "positions": {
            "entity_role": -3,
            "field_role": -2,
            "answer_boundary": -1,
        },
        "interventions": {
            "normal": "recipient entity-role plus recipient field-role",
            "row_role": "different-row same-column donor entity-role plus recipient field-role",
            "row_cross_role": "same donor field-role passed to row head plus recipient field-role",
            "column_role": "recipient entity-role plus same-row different-column donor field-role",
            "column_cross_role": "recipient entity-role plus same donor entity-role passed to column head",
            "both_role": "different-row different-column donor entity-role and field-role",
            "both_answer": "same donor answer-boundary state passed to both heads",
            "same_address_role": "different example with same row and column address",
        },
        "thresholds": dict(parent["contingent_phase1151"]),
        "primary_claims": {
            "counterfactual_sufficiency": (
                "normal, row-role, column-role, both-role, and same-address modes pass all absolute gates "
                "in every confirmation replicate and split"
            ),
            "position_specificity": (
                "role-aligned counterfactual answer accuracy exceeds matched cross-role controls by the "
                "frozen advantage in every replicate and split"
            ),
        },
        "forbidden": [
            "No donor, split, threshold, position, or metric changes after protocol creation",
            "No discovery-model reuse",
            "No claim that the intervention discovers a naturally learned mediator",
            "No causal necessity claim outside the planted readout architecture",
        ],
    }


def verify_preregistration(prereg: dict[str, Any]) -> None:
    body = dict(prereg)
    digest = body.pop("protocol_digest")
    if p1148.canonical_digest(body) != digest:
        raise RuntimeError("Phase1151 protocol digest mismatch")
    if p1148.file_sha256(Path(__file__).resolve()) != prereg["source_hashes"]["primary_script"]:
        raise RuntimeError("Phase1151 primary script changed after preregistration")
    if p1148.file_sha256(Path(p1150.__file__).resolve()) != prereg["source_hashes"][
        "phase1150_dependency"
    ]:
        raise RuntimeError("Phase1150 dependency changed after Phase1151 preregistration")
    parent = p1148.read_json(p1150.PREREG_PATH)
    if parent["protocol_digest"] != prereg["parent_phase1150_protocol_digest"]:
        raise RuntimeError("Phase1150 parent protocol changed")
    confirmation = p1148.read_json(p1150.OUT_ROOT / "analysis" / "confirmation_selection.json")
    if confirmation["selection_digest"] != prereg["parent_phase1150_confirmation_digest"]:
        raise RuntimeError("Phase1150 confirmation result changed")


def create_protocol() -> dict[str, Any]:
    body = protocol_body()
    prereg = dict(body)
    prereg["protocol_digest"] = p1148.canonical_digest(body)
    if PREREG_PATH.exists():
        if p1148.read_json(PREREG_PATH) != prereg:
            raise RuntimeError("Existing Phase1151 protocol differs from current script")
    else:
        p1148.write_json(PREREG_PATH, prereg)
    checks = {
        "four_confirmation_replicates": len(prereg["replicates"]) == 4,
        "two_evaluation_splits": prereg["splits"] == ["holdout", "quartet"],
        "role_positions_match_parent": prereg["positions"]
        == {"entity_role": -3, "field_role": -2, "answer_boundary": -1},
        "thresholds_frozen_by_parent": prereg["thresholds"]
        == p1148.read_json(p1150.PREREG_PATH)["contingent_phase1151"],
    }
    audit = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = p1148.canonical_digest(audit)
    p1148.write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1151 protocol audit failed: {checks}")
    return prereg


def choose_donors(rows: np.ndarray, columns: np.ndarray) -> dict[str, np.ndarray]:
    modes = {
        "row": lambda i, j: rows[j] != rows[i] and columns[j] == columns[i],
        "column": lambda i, j: rows[j] == rows[i] and columns[j] != columns[i],
        "both": lambda i, j: rows[j] != rows[i] and columns[j] != columns[i],
        "same": lambda i, j: rows[j] == rows[i] and columns[j] == columns[i],
    }
    donors: dict[str, np.ndarray] = {}
    for mode, predicate in modes.items():
        selected = np.empty(len(rows), dtype=np.int64)
        for index in range(len(rows)):
            candidates = [
                candidate
                for candidate in range(len(rows))
                if candidate != index and predicate(index, candidate)
            ]
            if not candidates:
                raise RuntimeError(f"No {mode} donor for item {index}")
            selected[index] = candidates[(index * 17 + len(mode)) % len(candidates)]
        donors[mode] = selected
    return donors


def capture_role_states(
    model: Any,
    inputs: np.ndarray,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    captured = {"entity": [], "field": [], "answer": []}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            ids = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, states = model.backbone(ids, return_states=True)
                normalized = model.backbone.final_norm(states[-1]).float()
            captured["entity"].append(normalized[:, -3, :].cpu())
            captured["field"].append(normalized[:, -2, :].cpu())
            captured["answer"].append(normalized[:, -1, :].cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in captured.items()}


def evaluate_mode(
    model: Any,
    dataset: dict[str, np.ndarray],
    states: dict[str, torch.Tensor],
    donors: dict[str, np.ndarray],
    mode: str,
    batch_size: int,
) -> dict[str, float]:
    rows = dataset["row_targets"]
    columns = dataset["column_targets"]
    if mode == "normal":
        row_source = states["entity"]
        column_source = states["field"]
        expected_rows = rows
        expected_columns = columns
    elif mode in ("row_role", "row_cross_role"):
        donor = donors["row"]
        row_source = states["entity" if mode == "row_role" else "field"][donor]
        column_source = states["field"]
        expected_rows = rows[donor]
        expected_columns = columns
    elif mode in ("column_role", "column_cross_role"):
        donor = donors["column"]
        row_source = states["entity"]
        column_source = states["field" if mode == "column_role" else "entity"][donor]
        expected_rows = rows
        expected_columns = columns[donor]
    elif mode in ("both_role", "both_answer"):
        donor = donors["both"]
        source_key = "answer" if mode == "both_answer" else None
        row_source = states[source_key][donor] if source_key else states["entity"][donor]
        column_source = states[source_key][donor] if source_key else states["field"][donor]
        expected_rows = rows[donor]
        expected_columns = columns[donor]
    elif mode == "same_address_role":
        donor = donors["same"]
        row_source = states["entity"][donor]
        column_source = states["field"][donor]
        expected_rows = rows
        expected_columns = columns
    else:
        raise ValueError(mode)

    predicted_rows: list[int] = []
    predicted_columns: list[int] = []
    predicted_answers: list[int] = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = start + batch_size
            row_state = row_source[start:stop].to(device)
            column_state = column_source[start:stop].to(device)
            grids = torch.from_numpy(dataset["grid_values"][start:stop]).to(device)
            row_logits = model.row_head(row_state)
            column_logits = model.column_head(column_state)
            distribution = p1148.mediated_distribution(row_logits, column_logits, grids)
            predicted_rows.extend(row_logits.argmax(-1).cpu().tolist())
            predicted_columns.extend(column_logits.argmax(-1).cpu().tolist())
            predicted_answers.extend(distribution.argmax(-1).cpu().tolist())

    predicted_rows_array = np.asarray(predicted_rows)
    predicted_columns_array = np.asarray(predicted_columns)
    expected_answers = dataset["grid_values"][
        np.arange(len(rows)), expected_rows, expected_columns
    ]
    original_answers = dataset["targets"]
    return {
        "row_transport_accuracy": float(np.mean(predicted_rows_array == expected_rows)),
        "column_transport_accuracy": float(
            np.mean(predicted_columns_array == expected_columns)
        ),
        "counterfactual_answer_accuracy": float(
            np.mean(np.asarray(predicted_answers) == expected_answers)
        ),
        "counterfactual_changed_fraction": float(np.mean(expected_answers != original_answers)),
    }


def evaluate_replicate(replicate: str, prereg: dict[str, Any]) -> dict[str, Any]:
    parent = p1148.read_json(p1150.PREREG_PATH)
    spec = parent["replicates"][replicate]
    summary = p1150.load_summary(replicate, "role_factorized", parent)
    model = p1150.load_model(summary)
    datasets, _ = p1148.build_evaluation_sets(spec, parent, "formal")
    per_split: dict[str, Any] = {}
    thresholds = prereg["thresholds"]
    for split_name, dataset in datasets:
        if split_name not in prereg["splits"]:
            continue
        donors = choose_donors(dataset["row_targets"], dataset["column_targets"])
        states = capture_role_states(
            model, dataset["inputs"], int(spec["training"]["evaluation_batch_size"])
        )
        modes = {
            mode: evaluate_mode(
                model,
                dataset,
                states,
                donors,
                mode,
                int(spec["training"]["evaluation_batch_size"]),
            )
            for mode in (
                "normal",
                "row_role",
                "row_cross_role",
                "column_role",
                "column_cross_role",
                "both_role",
                "both_answer",
                "same_address_role",
            )
        }
        minimum_normal = float(thresholds["minimum_normal_accuracy"])
        minimum_transport = float(thresholds["minimum_address_transport_accuracy"])
        minimum_answer = float(thresholds["minimum_counterfactual_answer_accuracy"])
        minimum_preservation = float(thresholds["minimum_orthogonal_role_preservation"])
        advantage = float(thresholds["minimum_cross_role_specificity_advantage"])
        sufficiency_gate = {
            "normal": modes["normal"]["counterfactual_answer_accuracy"] >= minimum_normal,
            "row_address": modes["row_role"]["row_transport_accuracy"] >= minimum_transport,
            "row_preserves_column": modes["row_role"]["column_transport_accuracy"]
            >= minimum_preservation,
            "row_answer": modes["row_role"]["counterfactual_answer_accuracy"] >= minimum_answer,
            "column_address": modes["column_role"]["column_transport_accuracy"]
            >= minimum_transport,
            "column_preserves_row": modes["column_role"]["row_transport_accuracy"]
            >= minimum_preservation,
            "column_answer": modes["column_role"]["counterfactual_answer_accuracy"]
            >= minimum_answer,
            "both_row": modes["both_role"]["row_transport_accuracy"] >= minimum_transport,
            "both_column": modes["both_role"]["column_transport_accuracy"] >= minimum_transport,
            "both_answer": modes["both_role"]["counterfactual_answer_accuracy"] >= minimum_answer,
            "same_address_rescue": modes["same_address_role"]["counterfactual_answer_accuracy"]
            >= minimum_answer,
        }
        specificity_advantages = {
            "row_answer": modes["row_role"]["counterfactual_answer_accuracy"]
            - modes["row_cross_role"]["counterfactual_answer_accuracy"],
            "column_answer": modes["column_role"]["counterfactual_answer_accuracy"]
            - modes["column_cross_role"]["counterfactual_answer_accuracy"],
            "both_answer": modes["both_role"]["counterfactual_answer_accuracy"]
            - modes["both_answer"]["counterfactual_answer_accuracy"],
        }
        specificity_gate = {
            name: value >= advantage for name, value in specificity_advantages.items()
        }
        donor_digest = p1148.array_digest(
            donors["row"], donors["column"], donors["both"], donors["same"]
        )
        per_split[split_name] = {
            "dataset_digest": p1148.dataset_digest(dataset),
            "donor_digest": donor_digest,
            "modes": modes,
            "sufficiency_gate": sufficiency_gate,
            "sufficiency_passed": all(sufficiency_gate.values()),
            "specificity_advantages": specificity_advantages,
            "specificity_gate": specificity_gate,
            "specificity_passed": all(specificity_gate.values()),
        }
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "replicate": replicate,
        "splits": per_split,
        "sufficiency_passed": all(item["sufficiency_passed"] for item in per_split.values()),
        "specificity_passed": all(item["specificity_passed"] for item in per_split.values()),
        "model_sha256": summary["model_sha256"],
    }
    result["result_digest"] = p1148.canonical_digest(result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(prereg: dict[str, Any]) -> dict[str, Any]:
    per_replicate = {
        replicate: evaluate_replicate(replicate, prereg)
        for replicate in prereg["replicates"]
    }
    sufficiency = all(item["sufficiency_passed"] for item in per_replicate.values())
    specificity = all(item["specificity_passed"] for item in per_replicate.values())
    result = {
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "per_replicate": per_replicate,
        "counterfactual_sufficiency_confirmed": sufficiency,
        "cross_role_position_specificity_confirmed": specificity,
        "outcome": (
            "planted_role_states_counterfactually_sufficient_and_position_specific"
            if sufficiency and specificity
            else "planted_role_states_counterfactually_sufficient_without_full_position_specificity"
            if sufficiency
            else "counterfactual_role_state_sufficiency_not_confirmed"
        ),
        "claim_boundary": (
            "The intervention concerns a planted mediator whose answer is forced through row and column heads. "
            "It calibrates a causal test but does not discover a natural free-network mechanism."
        ),
        "auto_continue": False,
        "auto_continue_reason": (
            "The next free-network functional-equivalence phase requires a separately calibrated and frozen "
            "mechanism-family protocol; no automatic location search is authorized."
        ),
    }
    result["final_digest"] = p1148.canonical_digest(result)
    p1148.write_json(OUT_ROOT / "analysis" / "final.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("create-protocol", "run"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "create-protocol":
        result = create_protocol()
    else:
        prereg = p1148.read_json(PREREG_PATH)
        verify_preregistration(prereg)
        result = run(prereg)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
