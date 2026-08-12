#!/usr/bin/env python3
"""Exhaustively map the five Phase 1003 semantic-anchor roles.

All 32 subsets are intervened with real cross-world donor states.  The same
factorial table supports sufficiency and complement-based restoration; no
activation ranking or weighted score is used.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1003_crossparadigm_protocol import (
    ANCHOR_ROLES,
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    digest,
    read_json,
    selected_directional_rows,
    stable_order,
    write_json,
    write_jsonl,
)


SUBSETS = tuple(
    tuple(
        role
        for role_index, role in enumerate(ANCHOR_ROLES)
        if mask & (1 << role_index)
    )
    for mask in range(2 ** len(ANCHOR_ROLES))
)


def subset_id(roles: Iterable[str]) -> str:
    role_set = set(roles)
    mask = sum(
        (1 << index)
        for index, role in enumerate(ANCHOR_ROLES)
        if role in role_set
    )
    return f"s{mask:02d}"


def complement(roles: Iterable[str]) -> tuple[str, ...]:
    role_set = set(roles)
    return tuple(role for role in ANCHOR_ROLES if role not in role_set)


def semantic_case(case: dict[str, Any]) -> dict[str, Any]:
    semantic_step = int(case["semantic_step"])
    result = dict(case)
    result["input_ids"] = (
        list(case["input_ids"])
        + list(case["answer_token_ids"][:semantic_step])
    )
    result["input_token_count"] = len(result["input_ids"])
    result["role_positions"] = dict(case["role_positions"])
    result["role_positions"]["answer_boundary"] = (
        result["input_token_count"] - 1
    )
    return result


def case_tensors(rows: list[dict[str, Any]], device):
    widths = {len(row["input_ids"]) for row in rows}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift {widths}")
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def candidate_logits(
    logits: torch.Tensor,
    candidate_ids: dict[str, int],
) -> torch.Tensor:
    labels = list(candidate_ids)
    ids = torch.tensor(
        [candidate_ids[label] for label in labels],
        dtype=torch.long,
        device=logits.device,
    )
    return logits.index_select(-1, ids).float().detach()


def capture_depth(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention = case_tensors(cases, device)
    captured: list[torch.Tensor] = []

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach())

    handle = layers[depth - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        if len(captured) != 1:
            raise RuntimeError(f"depth capture count {len(captured)}")
        candidates = candidate_logits(
            output.logits[:, -1, :],
            cases[0]["candidate_token_ids"],
        )
        return candidates, captured[0]
    finally:
        handle.remove()
        del input_ids, attention


def expanded_subset_forward(
    model,
    layers,
    device,
    target_cases: list[dict[str, Any]],
    target_hidden: torch.Tensor,
    donor_cases: list[dict[str, Any]],
    donor_hidden: torch.Tensor,
    depth: int,
    subsets: list[tuple[str, ...]],
) -> torch.Tensor:
    target_input, target_attention = case_tensors(target_cases, device)
    condition_count = len(subsets)
    input_ids = target_input.repeat((condition_count, 1))
    attention = target_attention.repeat((condition_count, 1))
    batch_size = len(target_cases)
    base_index = torch.arange(batch_size, device=device)
    role_positions = {}
    role_vectors = {}
    for role in ANCHOR_ROLES:
        target_positions = torch.tensor(
            [
                int(row["role_positions"][role])
                for row in target_cases
            ],
            dtype=torch.long,
            device=device,
        )
        donor_positions = torch.tensor(
            [
                int(row["role_positions"][role])
                for row in donor_cases
            ],
            dtype=torch.long,
            device=device,
        )
        target_vectors = target_hidden[
            base_index, target_positions, :
        ]
        donor_vectors = donor_hidden[
            base_index, donor_positions, :
        ]
        role_positions[role] = target_positions.repeat(condition_count)
        role_vectors[role] = torch.cat([
            donor_vectors if role in subset else target_vectors
            for subset in subsets
        ])

    count = [0]

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        patched = value.clone()
        expanded_index = torch.arange(
            patched.shape[0], device=patched.device
        )
        for role in ANCHOR_ROLES:
            patched[
                expanded_index,
                role_positions[role],
                :,
            ] = role_vectors[role].to(
                device=patched.device, dtype=patched.dtype
            )
        count[0] += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = layers[depth - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        if count[0] != 1:
            raise RuntimeError(f"subset patch count {count[0]}")
        logits = candidate_logits(
            output.logits[:, -1, :],
            target_cases[0]["candidate_token_ids"],
        )
        return logits.reshape(condition_count, batch_size, -1)
    finally:
        handle.remove()
        del input_ids, attention


def choose_donors(
    rows: list[dict[str, Any]],
    model_name: str,
    domain: str,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [row["target"] for row in rows]
    usage: Counter[str] = Counter()
    donors = []
    for recipient_index, row in enumerate(rows):
        target = row["target"]
        eligible = [
            candidate
            for candidate in candidates
            if candidate["world_id"] != target["world_id"]
            and candidate["gold"] != target["gold"]
            and candidate["template"] == target["template"]
            and candidate["input_token_count"] == target["input_token_count"]
        ]
        if not eligible:
            raise RuntimeError(
                f"{model_name}/{domain}/{split}: no donor for "
                f"{target['record_id']}"
            )
        eligible.sort(key=lambda candidate: (
            usage[candidate["record_id"]],
            stable_order(
                candidate["record_id"],
                f"donor:{model_name}:{domain}:{split}:"
                f"{recipient_index}:{target['record_id']}",
            ),
        ))
        donor = eligible[0]
        usage[donor["record_id"]] += 1
        donors.append(donor)
    return donors, {
        "recipient_count": len(rows),
        "unique_donor_count": len(usage),
        "unique_donor_fraction": len(usage) / max(len(rows), 1),
        "maximum_donor_reuse": max(usage.values()),
        "minimum_used_donor_reuse": min(usage.values()),
        "all_cross_world": all(
            donor["world_id"] != row["target"]["world_id"]
            for row, donor in zip(rows, donors)
        ),
        "all_donor_answers_differ_from_target": all(
            donor["gold"] != row["target"]["gold"]
            for row, donor in zip(rows, donors)
        ),
        "donor_assignment_digest": digest([
            {
                "recipient": row["target"]["record_id"],
                "donor": donor["record_id"],
            }
            for row, donor in zip(rows, donors)
        ]),
    }


def row_batches(
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
):
    groups: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, donor in zip(rows, donors):
        groups[int(row["template"])].append((row, donor))
    for template, values in sorted(groups.items()):
        values.sort(key=lambda item: (
            item[0]["pair_id"],
            item[0]["direction"],
        ))
        for start in range(0, len(values), batch_size):
            chunk = values[start : start + batch_size]
            yield (
                template,
                [item[0] for item in chunk],
                [item[1] for item in chunk],
            )


def prediction_labels(
    logits: torch.Tensor,
    labels: list[str],
) -> list[str]:
    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    return [labels[int(index)] for index in indices]


def contrast_margin(
    logits: torch.Tensor,
    labels: list[str],
    donors: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> torch.Tensor:
    index = {label: position for position, label in enumerate(labels)}
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    donor_index = torch.tensor(
        [index[row["gold"]] for row in donors],
        dtype=torch.long,
        device=logits.device,
    )
    target_index = torch.tensor(
        [index[row["gold"]] for row in targets],
        dtype=torch.long,
        device=logits.device,
    )
    return (
        logits[batch_index, donor_index]
        - logits[batch_index, target_index]
    )


def summarize_subsets(
    rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    values = [row for row in rows if row["split"] == split]
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_lookup = {}
    for row in values:
        by_subset[row["subset_id"]].append(row)
        row_lookup[(
            row["pair_id"],
            row["direction"],
            row["subset_id"],
        )] = row
    result = {}
    for subset in SUBSETS:
        current_id = subset_id(subset)
        complement_id = subset_id(complement(subset))
        current = by_subset[current_id]
        complement_rows = [
            row_lookup[
                (
                    row["pair_id"],
                    row["direction"],
                    complement_id,
                )
            ]
            for row in current
        ]
        restoration_mediation = [
            (
                row["full_anchor_margin"]
                - complement_row["patched_margin"]
            )
            / max(
                abs(
                    row["full_anchor_margin"]
                    - row["target_margin"]
                ),
                1e-8,
            )
            for row, complement_row in zip(
                current, complement_rows
            )
        ]
        qualified = [
            row
            for row in current
            if row["target_clean_correct"]
            and row["donor_clean_correct"]
        ]
        result[current_id] = {
            "subset_id": current_id,
            "roles": list(subset),
            "cardinality": len(subset),
            "n": len(current),
            "behavior_qualified_n": len(qualified),
            "donor_rate": float(np.mean([
                row["donor_prediction"] for row in current
            ])),
            "target_rate": float(np.mean([
                row["target_prediction"] for row in current
            ])),
            "clean_prediction_agreement": float(np.mean([
                (
                    row["prediction"]
                    == row["target_clean_prediction"]
                )
                for row in current
            ])),
            "maximum_candidate_logit_difference_from_target": float(
                max(
                    row[
                        "candidate_max_abs_difference_from_target"
                    ]
                    for row in current
                )
            ),
            "median_normalized_transfer": float(np.median([
                row["normalized_transfer"] for row in current
            ])),
            "mean_normalized_transfer": float(np.mean([
                row["normalized_transfer"] for row in current
            ])),
            "qualified_donor_rate": (
                float(np.mean([
                    row["donor_prediction"] for row in qualified
                ]))
                if qualified
                else None
            ),
            "restored_roles": list(subset),
            "remaining_donor_subset_id": complement_id,
            "restoration_target_rate": float(np.mean([
                row["target_prediction"] for row in complement_rows
            ])),
            "median_restoration_mediation": float(np.median(
                restoration_mediation
            )),
        }
    return result


def role_interactions(
    subset_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    empty = subset_summary[subset_id(())][
        "mean_normalized_transfer"
    ]
    result = []
    for role_a, role_b in itertools.combinations(ANCHOR_ROLES, 2):
        effect_a = subset_summary[subset_id((role_a,))][
            "mean_normalized_transfer"
        ]
        effect_b = subset_summary[subset_id((role_b,))][
            "mean_normalized_transfer"
        ]
        effect_ab = subset_summary[subset_id((role_a, role_b))][
            "mean_normalized_transfer"
        ]
        result.append({
            "role_a": role_a,
            "role_b": role_b,
            "factorial_interaction": (
                effect_ab - effect_a - effect_b + empty
            ),
            "descriptive_only": True,
        })
    return result


def discovery_selection(
    subset_summary: dict[str, Any],
    donor_threshold: float,
    transfer_threshold: float,
) -> dict[str, Any]:
    passing = [
        value
        for value in subset_summary.values()
        if value["donor_rate"] >= donor_threshold
        and value["median_normalized_transfer"] >= transfer_threshold
    ]
    if not passing:
        return {
            "status": "NO_DISCOVERY_SUBSET_PASSES",
            "minimum_cardinality": None,
            "selected_subset_ids": [],
            "selected_subsets": [],
        }
    minimum = min(value["cardinality"] for value in passing)
    selected = sorted(
        (
            value for value in passing
            if value["cardinality"] == minimum
        ),
        key=lambda value: value["subset_id"],
    )
    return {
        "status": "FROZEN_FROM_DISCOVERY",
        "minimum_cardinality": minimum,
        "selected_subset_ids": [
            value["subset_id"] for value in selected
        ],
        "selected_subsets": [
            value["roles"] for value in selected
        ],
        "selection_uses_confirmation": False,
    }


def run_domain(
    model,
    layers,
    device,
    model_name: str,
    domain: str,
    source_depth: int,
    batch_size: int,
    condition_batch_size: int,
) -> dict[str, Any]:
    result_rows = []
    donor_audits = {}
    for split in ("discovery", "confirmation"):
        rows = selected_directional_rows(
            model_name, domain, split
        )
        donors, donor_audit = choose_donors(
            rows, model_name, domain, split
        )
        donor_audits[split] = donor_audit
        batches = list(row_batches(rows, donors, batch_size))
        for batch_number, (_, batch, donor_batch) in enumerate(
            batches, 1
        ):
            target_cases = [
                semantic_case(row["target"]) for row in batch
            ]
            donor_cases = [
                semantic_case(row) for row in donor_batch
            ]
            expanded_target_cases = (
                target_cases * condition_batch_size
            )
            expanded_donor_cases = (
                donor_cases * condition_batch_size
            )
            target_logits_all, target_hidden_all = capture_depth(
                model,
                layers,
                device,
                expanded_target_cases,
                source_depth,
            )
            donor_logits_all, donor_hidden_all = capture_depth(
                model,
                layers,
                device,
                expanded_donor_cases,
                source_depth,
            )
            batch_count = len(target_cases)
            target_logits = target_logits_all.reshape(
                condition_batch_size, batch_count, -1
            )[0]
            donor_logits = donor_logits_all.reshape(
                condition_batch_size, batch_count, -1
            )[0]
            target_hidden = target_hidden_all.reshape(
                condition_batch_size,
                batch_count,
                target_hidden_all.shape[1],
                target_hidden_all.shape[2],
            )[0]
            donor_hidden = donor_hidden_all.reshape(
                condition_batch_size,
                batch_count,
                donor_hidden_all.shape[1],
                donor_hidden_all.shape[2],
            )[0]
            labels = list(target_cases[0]["candidate_token_ids"])
            donor_predictions = prediction_labels(
                donor_logits, labels
            )
            donor_margin = contrast_margin(
                donor_logits, labels, donor_cases, target_cases
            )
            full_margin = None
            condition_outputs = {}
            for start in range(0, len(SUBSETS), condition_batch_size):
                condition_subsets = list(
                    SUBSETS[start : start + condition_batch_size]
                )
                condition_logits = expanded_subset_forward(
                    model,
                    layers,
                    device,
                    target_cases,
                    target_hidden,
                    donor_cases,
                    donor_hidden,
                    source_depth,
                    condition_subsets,
                )
                for condition_index, subset in enumerate(
                    condition_subsets
                ):
                    condition_outputs[subset_id(subset)] = (
                        condition_logits[condition_index]
                    )
            clean_logits = target_logits
            target_predictions = prediction_labels(
                clean_logits, labels
            )
            target_margin = contrast_margin(
                clean_logits, labels, donor_cases, target_cases
            )
            full_id = subset_id(ANCHOR_ROLES)
            full_margin = contrast_margin(
                condition_outputs[full_id],
                labels,
                donor_cases,
                target_cases,
            )
            for subset in SUBSETS:
                current_id = subset_id(subset)
                logits = condition_outputs[current_id]
                predictions = prediction_labels(logits, labels)
                patched_margin = contrast_margin(
                    logits, labels, donor_cases, target_cases
                )
                for index, row in enumerate(batch):
                    normalizer = max(
                        abs(float(
                            full_margin[index] - target_margin[index]
                        )),
                        1e-8,
                    )
                    result_rows.append({
                        "schema_version": (
                            "phase1003_anchor_subset_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "domain": domain,
                        "split": split,
                        "pair_id": row["pair_id"],
                        "direction": row["direction"],
                        "template": row["template"],
                        "target_record_id": row["target"]["record_id"],
                        "donor_record_id": donor_batch[index]["record_id"],
                        "target_world_id": row["target"]["world_id"],
                        "donor_world_id": donor_batch[index]["world_id"],
                        "target_gold": row["target"]["gold"],
                        "donor_gold": donor_batch[index]["gold"],
                        "subset_id": current_id,
                        "subset_roles": list(subset),
                        "subset_cardinality": len(subset),
                        "prediction": predictions[index],
                        "donor_prediction": (
                            predictions[index]
                            == donor_batch[index]["gold"]
                        ),
                        "target_prediction": (
                            predictions[index]
                            == row["target"]["gold"]
                        ),
                        "target_clean_prediction": (
                            target_predictions[index]
                        ),
                        "donor_clean_prediction": (
                            donor_predictions[index]
                        ),
                        "target_clean_correct": (
                            target_predictions[index]
                            == row["target"]["gold"]
                        ),
                        "donor_clean_correct": (
                            donor_predictions[index]
                            == donor_batch[index]["gold"]
                        ),
                        "target_margin": float(target_margin[index]),
                        "donor_margin": float(donor_margin[index]),
                        "full_anchor_margin": float(full_margin[index]),
                        "patched_margin": float(patched_margin[index]),
                        "normalized_transfer": float(
                            (
                                patched_margin[index]
                                - target_margin[index]
                            )
                            / normalizer
                        ),
                        "candidate_max_abs_difference_from_target": float(
                            torch.max(torch.abs(
                                logits[index] - clean_logits[index]
                            ))
                        ),
                    })
            del (
                target_logits,
                donor_logits,
                target_logits_all,
                donor_logits_all,
                target_hidden,
                donor_hidden,
                target_hidden_all,
                donor_hidden_all,
                condition_outputs,
                clean_logits,
            )
            print(
                f"[{model_name}/{domain}/{split}] "
                f"{batch_number}/{len(batches)}",
                flush=True,
            )

    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    split_summary = {
        split: summarize_subsets(result_rows, split)
        for split in ("discovery", "confirmation")
    }
    selection = discovery_selection(
        split_summary["discovery"],
        thresholds["subset_donor_rate"],
        thresholds["subset_median_normalized_transfer"],
    )
    selected_confirmation = []
    for current_id in selection["selected_subset_ids"]:
        item = split_summary["confirmation"][current_id]
        selected_confirmation.append({
            "subset_id": current_id,
            "roles": item["roles"],
            "cardinality": item["cardinality"],
            "donor_rate": item["donor_rate"],
            "median_normalized_transfer": item[
                "median_normalized_transfer"
            ],
            "confirmation_gate": (
                item["donor_rate"]
                >= thresholds["subset_donor_rate"]
                and item["median_normalized_transfer"]
                >= thresholds["subset_median_normalized_transfer"]
            ),
        })
    empty_id = subset_id(())
    full_id = subset_id(ANCHOR_ROLES)
    controls = {
        split: {
            "empty_noop_prediction_agreement": (
                split_summary[split][empty_id][
                    "clean_prediction_agreement"
                ]
            ),
            "empty_noop_max_candidate_logit_difference": (
                split_summary[split][empty_id][
                    "maximum_candidate_logit_difference_from_target"
                ]
            ),
            "full_anchor_donor_rate": (
                split_summary[split][full_id]["donor_rate"]
            ),
            "full_anchor_median_transfer": (
                split_summary[split][full_id][
                    "median_normalized_transfer"
                ]
            ),
        }
        for split in ("discovery", "confirmation")
    }
    domain_pass = all(
        values["empty_noop_prediction_agreement"]
        >= thresholds["noop_prediction_agreement"]
        and values["full_anchor_donor_rate"]
        >= thresholds["full_anchor_donor_rate"]
        for values in controls.values()
    )
    confirmation_selection_pass = (
        bool(selected_confirmation)
        and all(
            item["confirmation_gate"]
            for item in selected_confirmation
        )
    )
    summary = {
        "schema_version": "phase1003_anchor_subset_summary.v1",
        "phase": PHASE,
        "implementation_revision": 4,
        "model": model_name,
        "domain": domain,
        "status": "complete",
        "source_depth": source_depth,
        "case_batch_size": batch_size,
        "condition_batch_size": condition_batch_size,
        "condition_isolation": condition_batch_size == 1,
        "instrument_revision_audit": {
            "revision_1": (
                "No-op was initially conflated with target correctness."
            ),
            "revision_2": (
                "No-op was compared with a clean forward at a different "
                "batch size; 8bit numerical drift remained."
            ),
            "revision_3": (
                "State capture and intervention used the same expanded "
                "batch size, but mixed intervention conditions still "
                "showed large candidate-logit drift."
            ),
            "revision_4": (
                "Formal execution isolates one intervention condition per "
                "forward call and matches capture/intervention batch size."
            ),
            "scientific_results_used_to_select_subsets": False,
        },
        "subset_count": len(SUBSETS),
        "direction_count_per_split": 64,
        "donor_audits": donor_audits,
        "controls": controls,
        "split_summary": split_summary,
        "pairwise_interactions": {
            split: role_interactions(split_summary[split])
            for split in ("discovery", "confirmation")
        },
        "discovery_selection": selection,
        "selected_confirmation": selected_confirmation,
        "parent_instrument_pass": domain_pass,
        "frozen_subset_confirmation_pass": (
            confirmation_selection_pass
        ),
        "claim_boundary": (
            "The exhaustive table identifies causal semantic-role subsets "
            "at depth 1 for this controlled task. Roles are prompt "
            "positions, not neurons, and minimality is relative to the "
            "five-role universe only."
        ),
    }
    root = OUT_ROOT / "anchor_subsets" / model_name / domain
    write_jsonl(root / "rows.jsonl", result_rows)
    write_json(root / "summary.json", summary)
    return summary


def run_model(
    model_name: str,
    batch_size: int,
    condition_batch_size: int,
) -> dict[str, Any]:
    behavior = read_json(
        OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][model_name])
    passing_domains = [
        domain
        for domain, passed in behavior["domain_gates"].items()
        if passed
    ]
    if len(passing_domains) < prereg["primary_thresholds"][
        "cross_domain_minimum_pass_count"
    ]:
        raise RuntimeError(
            f"{model_name}: fewer than two behavior-qualified domains"
        )
    model = tokenizer = None
    started = time.time()
    summaries = {}
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        for domain in passing_domains:
            summaries[domain] = run_domain(
                model,
                layers,
                device,
                model_name,
                domain,
                source_depth,
                batch_size,
                condition_batch_size,
            )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "phase1003_anchor_subset_model.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "behavior_qualified_domains": passing_domains,
        "domains": summaries,
        "parent_instrument_pass_count": sum(
            summary["parent_instrument_pass"]
            for summary in summaries.values()
        ),
        "frozen_subset_confirmation_pass_count": sum(
            summary["frozen_subset_confirmation_pass"]
            for summary in summaries.values()
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        OUT_ROOT / "anchor_subsets" / model_name / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            OUT_ROOT
            / "anchor_subsets"
            / model_name
            / "summary.json"
        )
        if path.exists():
            summaries[model_name] = read_json(path)
    cross_domain = {}
    for domain in DOMAINS:
        available = [
            summary["domains"][domain]
            for summary in summaries.values()
            if domain in summary["domains"]
        ]
        cross_domain[domain] = {
            "tested_model_count": len(available),
            "parent_instrument_pass_count": sum(
                item["parent_instrument_pass"] for item in available
            ),
            "frozen_subset_confirmation_pass_count": sum(
                item["frozen_subset_confirmation_pass"]
                for item in available
            ),
        }
    payload = {
        "schema_version": "phase1003_anchor_subset_aggregate.v1",
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_domain": cross_domain,
    }
    write_json(
        OUT_ROOT / "anchor_subsets" / "summary.json", payload
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--condition-batch-size", type=int, default=1
    )
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(
            args.model,
            args.batch_size,
            args.condition_batch_size,
        )
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
