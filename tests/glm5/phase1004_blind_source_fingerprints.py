#!/usr/bin/env python3
"""Discover label-blind residual source sets from causal fingerprints.

The selection path never reads ``sealed_semantic_role_positions``. Those
labels are revealed only in ``semantic_reconstruction_audit`` after the
physical position set has been frozen.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1004_blind_causal_basis_protocol import (
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    canonical,
    digest,
    read_json,
    read_jsonl,
    selected_directional_rows,
    semantic_case,
    stable_order,
    write_json,
    write_jsonl,
)


SOURCE_DEPTH = 1
BATCH_SIZE = 16


def case_tensors(rows: list[dict[str, Any]], device):
    widths = {len(row["input_ids"]) for row in rows}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift: {widths}")
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def candidate_logits(
    logits: torch.Tensor,
    candidate_ids: dict[str, int],
) -> tuple[list[str], torch.Tensor]:
    labels = list(candidate_ids)
    ids = torch.tensor(
        [candidate_ids[label] for label in labels],
        dtype=torch.long,
        device=logits.device,
    )
    return labels, logits.index_select(-1, ids).float().detach()


def prediction_labels(logits: torch.Tensor, labels: list[str]) -> list[str]:
    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    return [labels[int(index)] for index in indices]


def contrast_margin(
    logits: torch.Tensor,
    labels: list[str],
    donor_cases: list[dict[str, Any]],
    target_cases: list[dict[str, Any]],
) -> torch.Tensor:
    index = {label: position for position, label in enumerate(labels)}
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    donor_index = torch.tensor(
        [index[row["gold"]] for row in donor_cases],
        dtype=torch.long,
        device=logits.device,
    )
    target_index = torch.tensor(
        [index[row["gold"]] for row in target_cases],
        dtype=torch.long,
        device=logits.device,
    )
    return (
        logits[batch_index, donor_index]
        - logits[batch_index, target_index]
    )


def final_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise RuntimeError(f"cannot find final norm for {type(model).__name__}")


def capture(
    model,
    device,
    cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    *,
    trajectory: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    input_ids, attention = case_tensors(cases, device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    labels, candidates = candidate_logits(
        output.logits[:, -1, :], candidate_ids
    )
    depth_hidden = output.hidden_states[SOURCE_DEPTH].detach()
    trajectory_rows = []
    if trajectory:
        norm = final_norm_module(model)
        lm_head = model.get_output_embeddings()
        batch_index = torch.arange(len(cases), device=device)
        label_index = {label: index for index, label in enumerate(labels)}
        for depth, hidden in enumerate(output.hidden_states):
            with torch.inference_mode():
                normalized = norm(hidden[:, -1, :])
                lens_logits = lm_head(normalized).float()
                log_partition = torch.logsumexp(
                    lens_logits, dim=-1, keepdim=True
                )
                log_prob = lens_logits - log_partition
                probability = torch.exp(log_prob)
                full_entropy = -torch.sum(
                    probability * log_prob, dim=-1
                )
                _, panel = candidate_logits(lens_logits, candidate_ids)
                panel_log_prob = torch.log_softmax(panel, dim=-1)
                panel_entropy = -torch.sum(
                    torch.exp(panel_log_prob) * panel_log_prob,
                    dim=-1,
                )
                top_token = lens_logits.argmax(dim=-1)
                for row_index, case in enumerate(cases):
                    target_token_id = int(
                        candidate_ids[case["gold"]]
                    )
                    target_logit = lens_logits[
                        row_index, target_token_id
                    ]
                    target_rank = int(
                        torch.sum(
                            lens_logits[row_index] > target_logit
                        ).item()
                    ) + 1
                    target_panel_index = label_index[case["gold"]]
                    trajectory_rows.append({
                        "schema_version": (
                            "phase1004_diagnostic_trajectory_row.v1"
                        ),
                        "phase": PHASE,
                        "model": case["model"],
                        "record_id": case["record_id"],
                        "domain": case["domain"],
                        "split": case["split"],
                        "template": case["template"],
                        "observer": "final_norm_raw_logit_lens",
                        "observer_is_native_intermediate_probability": False,
                        "observer_allowed_for_causal_selection": False,
                        "depth": depth,
                        "relative_depth": (
                            depth / max(len(output.hidden_states) - 1, 1)
                        ),
                        "full_vocab_entropy": float(
                            full_entropy[row_index].item()
                        ),
                        "candidate_panel_entropy": float(
                            panel_entropy[row_index].item()
                        ),
                        "target_rank": target_rank,
                        "target_panel_logit": float(
                            panel[row_index, target_panel_index].item()
                        ),
                        "top_token_id": int(
                            top_token[row_index].item()
                        ),
                    })
            del (
                normalized,
                lens_logits,
                log_partition,
                log_prob,
                probability,
                full_entropy,
                panel,
                panel_log_prob,
                panel_entropy,
                top_token,
            )
    del output, input_ids, attention
    return candidates, depth_hidden, trajectory_rows


def forward_patch(
    model,
    layers,
    device,
    target_cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    target_hidden: torch.Tensor,
    source_hidden: torch.Tensor,
    donor_positions: Iterable[int],
) -> torch.Tensor:
    positions = sorted({int(value) for value in donor_positions})
    patch = target_hidden.clone()
    if positions:
        patch[:, positions, :] = source_hidden[:, positions, :]
    input_ids, attention = case_tensors(target_cases, device)
    count = [0]

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if value.shape[1] != patch.shape[1]:
            return output
        count[0] += 1
        replacement = patch.to(device=value.device, dtype=value.dtype)
        return (
            (replacement,) + output[1:]
            if isinstance(output, tuple)
            else replacement
        )

    handle = layers[SOURCE_DEPTH - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if count[0] != 1:
            raise RuntimeError(f"source patch count drift: {count[0]}")
        _, result = candidate_logits(
            output.logits[:, -1, :], candidate_ids
        )
        return result
    finally:
        handle.remove()
        del input_ids, attention


def choose_donors(
    rows: list[dict[str, Any]],
    *,
    same_answer: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_name = rows[0]["model"]
    domain = rows[0]["domain"]
    split = rows[0]["split"]
    templates = {
        int(row["target"]["template"]) for row in rows
    }
    candidates = [
        semantic_case(case)
        for case in read_jsonl(
            OUT_ROOT / "protocol" / model_name / "cases.jsonl"
        )
        if case["domain"] == domain
        and case["split"] == split
        and int(case["template"]) in templates
    ]
    usage: Counter[str] = Counter()
    donors = []
    for recipient_index, row in enumerate(rows):
        target = semantic_case(row["target"])
        target_values = set(target["base_values"])
        eligible = []
        for candidate in candidates:
            if candidate["world_id"] == target["world_id"]:
                continue
            if candidate["template"] != target["template"]:
                continue
            if candidate["input_token_count"] != target["input_token_count"]:
                continue
            if same_answer:
                if candidate["gold"] != target["gold"]:
                    continue
            else:
                if candidate["gold"] == target["gold"]:
                    continue
                if set(candidate["base_values"]) & target_values:
                    continue
            eligible.append(candidate)
        if not eligible:
            raise RuntimeError(
                f"no {'same' if same_answer else 'different'} donor for "
                f"{target['record_id']}"
            )
        eligible.sort(key=lambda candidate: (
            usage[candidate["record_id"]],
            stable_order(
                candidate["record_id"],
                f"donor:{same_answer}:{recipient_index}:"
                f"{target['record_id']}",
            ),
        ))
        donor = eligible[0]
        usage[donor["record_id"]] += 1
        donors.append(donor)
    return donors, {
        "same_answer_control": same_answer,
        "candidate_pool_source": (
            "complete_frozen_protocol_model_domain_split_template"
        ),
        "candidate_pool_count": len(candidates),
        "recipient_count": len(rows),
        "unique_donor_count": len(usage),
        "unique_donor_fraction": len(usage) / max(len(rows), 1),
        "maximum_donor_reuse": max(usage.values()),
        "all_cross_world": all(
            donor["world_id"] != semantic_case(row["target"])["world_id"]
            for row, donor in zip(rows, donors)
        ),
        "all_answer_contracts_hold": all(
            (donor["gold"] == semantic_case(row["target"])["gold"])
            == same_answer
            for row, donor in zip(rows, donors)
        ),
        "different_answer_value_sets_disjoint": (
            None
            if same_answer
            else all(
                not (
                    set(donor["base_values"])
                    & set(semantic_case(row["target"])["base_values"])
                )
                for row, donor in zip(rows, donors)
            )
        ),
        "assignment_digest": digest([
            {
                "recipient": semantic_case(row["target"])["record_id"],
                "donor": donor["record_id"],
            }
            for row, donor in zip(rows, donors)
        ]),
    }


def grouped_rows(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["target"]["template"])].append(row)
    return {
        template: sorted(
            values,
            key=lambda item: (item["pair_id"], item["direction"]),
        )
        for template, values in sorted(groups.items())
    }


def batches(
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    same_donors: list[dict[str, Any]],
    batch_size: int,
):
    if not (len(rows) == len(donors) == len(same_donors)):
        raise RuntimeError("batch input length mismatch")
    for start in range(0, len(rows), batch_size):
        stop = start + batch_size
        yield (
            rows[start:stop],
            donors[start:stop],
            same_donors[start:stop],
        )


def condition_rows(
    batch: list[dict[str, Any]],
    donor_batch: list[dict[str, Any]],
    target_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    labels: list[str],
    condition: str,
    positions: list[int],
) -> list[dict[str, Any]]:
    target_cases = [semantic_case(row["target"]) for row in batch]
    target_margin = contrast_margin(
        target_logits, labels, donor_batch, target_cases
    )
    donor_margin = contrast_margin(
        donor_logits, labels, donor_batch, target_cases
    )
    patched_margin = contrast_margin(
        patched_logits, labels, donor_batch, target_cases
    )
    predictions = prediction_labels(patched_logits, labels)
    rows = []
    for index, item in enumerate(batch):
        denominator = float(
            donor_margin[index] - target_margin[index]
        )
        transfer = float(
            (patched_margin[index] - target_margin[index])
            / max(abs(denominator), 1e-8)
        )
        rows.append({
            "schema_version": "phase1004_source_condition_row.v1",
            "phase": PHASE,
            "model": item["model"],
            "domain": item["domain"],
            "split": item["split"],
            "template": int(item["target"]["template"]),
            "pair_id": item["pair_id"],
            "direction": item["direction"],
            "target_record_id": item["target"]["record_id"],
            "donor_record_id": donor_batch[index]["record_id"],
            "condition": condition,
            "physical_positions": positions,
            "position_count": len(positions),
            "target_gold": target_cases[index]["gold"],
            "donor_gold": donor_batch[index]["gold"],
            "prediction": predictions[index],
            "predicted_target": (
                predictions[index] == target_cases[index]["gold"]
            ),
            "predicted_donor": (
                predictions[index] == donor_batch[index]["gold"]
            ),
            "target_margin": float(target_margin[index]),
            "donor_margin": float(donor_margin[index]),
            "patched_margin": float(patched_margin[index]),
            "normalized_transfer": transfer,
            "candidate_logits": [
                float(value)
                for value in patched_logits[index].detach().cpu().tolist()
            ],
        })
    return rows


def run_masks(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    same_donors: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    masks: list[dict[str, Any]],
    batch_size: int,
    *,
    include_trajectory: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows = []
    trajectory_rows = []
    labels = list(candidate_ids)
    all_batches = list(
        batches(rows, donors, same_donors, batch_size)
    )
    for batch_number, (batch, donor_batch, same_batch) in enumerate(
        all_batches, 1
    ):
        target_cases = [semantic_case(row["target"]) for row in batch]
        target_logits, target_hidden, target_trajectory = capture(
            model,
            device,
            target_cases,
            candidate_ids,
            trajectory=include_trajectory,
        )
        donor_logits, donor_hidden, _ = capture(
            model,
            device,
            donor_batch,
            candidate_ids,
            trajectory=False,
        )
        same_logits, same_hidden, _ = capture(
            model,
            device,
            same_batch,
            candidate_ids,
            trajectory=False,
        )
        trajectory_rows.extend(target_trajectory)
        for mask_number, mask in enumerate(masks, 1):
            source_kind = mask.get("source_kind", "different_answer")
            if source_kind == "target":
                source_hidden = target_hidden
                source_logits = target_logits
                donor_for_margin = donor_batch
            elif source_kind == "same_answer":
                source_hidden = same_hidden
                source_logits = same_logits
                donor_for_margin = same_batch
            else:
                source_hidden = donor_hidden
                source_logits = donor_logits
                donor_for_margin = donor_batch
            patched = forward_patch(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                target_hidden,
                source_hidden,
                mask["positions"],
            )
            output_rows.extend(condition_rows(
                batch,
                donor_for_margin,
                target_logits,
                source_logits,
                patched,
                labels,
                mask["condition"],
                list(mask["positions"]),
            ))
            del patched
            if mask_number % 16 == 0:
                print(
                    f"[mask] batch {batch_number}/{len(all_batches)} "
                    f"condition {mask_number}/{len(masks)}",
                    flush=True,
                )
        del (
            target_logits,
            target_hidden,
            donor_logits,
            donor_hidden,
            same_logits,
            same_hidden,
        )
    return output_rows, trajectory_rows


def summarize_condition(
    rows: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    values = [row for row in rows if row["condition"] == condition]
    if not values:
        raise RuntimeError(f"missing condition {condition}")
    return {
        "condition": condition,
        "n": len(values),
        "position_count": values[0]["position_count"],
        "physical_positions": values[0]["physical_positions"],
        "donor_rate": float(np.mean([
            row["predicted_donor"] for row in values
        ])),
        "target_rate": float(np.mean([
            row["predicted_target"] for row in values
        ])),
        "mean_normalized_transfer": float(np.mean([
            row["normalized_transfer"] for row in values
        ])),
        "median_normalized_transfer": float(np.median([
            row["normalized_transfer"] for row in values
        ])),
    }


def position_fingerprints(
    rows: list[dict[str, Any]],
    width: int,
) -> list[dict[str, Any]]:
    full = {
        (
            row["pair_id"],
            row["direction"],
        ): row
        for row in rows
        if row["condition"] == "different_answer_full"
    }
    result = []
    for position in range(width):
        event_id = f"p{position:03d}"
        single_values = [
            row
            for row in rows
            if row["condition"] == f"single:{event_id}"
        ]
        loo_values = [
            row
            for row in rows
            if row["condition"] == f"loo:{event_id}"
        ]
        mediation = []
        for row in loo_values:
            full_row = full[(row["pair_id"], row["direction"])]
            mediation.append(
                (
                    full_row["patched_margin"]
                    - row["patched_margin"]
                )
                / max(
                    abs(
                        full_row["patched_margin"]
                        - full_row["target_margin"]
                    ),
                    1e-8,
                )
            )
        result.append({
            "event_id": event_id,
            "physical_position": position,
            "n": len(single_values),
            "single_donor_rate": float(np.mean([
                row["predicted_donor"] for row in single_values
            ])),
            "single_target_rate": float(np.mean([
                row["predicted_target"] for row in single_values
            ])),
            "single_mean_transfer": float(np.mean([
                row["normalized_transfer"] for row in single_values
            ])),
            "single_median_transfer": float(np.median([
                row["normalized_transfer"] for row in single_values
            ])),
            "loo_restored_target_rate": float(np.mean([
                row["predicted_target"] for row in loo_values
            ])),
            "loo_donor_rate": float(np.mean([
                row["predicted_donor"] for row in loo_values
            ])),
            "loo_median_mediation": float(np.median(mediation)),
            "loo_mean_mediation": float(np.mean(mediation)),
            "causal_signature": {
                "single_switch": (
                    float(np.mean([
                        row["predicted_donor"]
                        for row in single_values
                    ])) >= 0.50
                ),
                "leaveout_restore": (
                    float(np.mean([
                        row["predicted_target"]
                        for row in loo_values
                    ])) >= 0.50
                ),
                "positive_mediation": (
                    float(np.median(mediation)) >= 0.10
                ),
            },
        })
    return result


def rank_positions(
    fingerprints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        fingerprints,
        key=lambda item: (
            -item["loo_restored_target_rate"],
            -item["loo_median_mediation"],
            -item["single_donor_rate"],
            item["physical_position"],
        ),
    )
    return [
        {
            **item,
            "causal_rank": rank,
            "selection_uses_semantic_labels": False,
            "selection_uses_confirmation_to_tune_rule": False,
        }
        for rank, item in enumerate(ranked, 1)
    ]


def passes_source_gate(summary: dict[str, Any]) -> bool:
    return (
        summary["donor_rate"] >= 0.80
        and summary["median_normalized_transfer"] >= 0.50
    )


def reveal_semantic_roles(
    frozen_positions: list[int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen = set(frozen_positions)
    exact = []
    precisions = []
    recalls = []
    role_coverage: dict[str, list[bool]] = defaultdict(list)
    selected_role_names: dict[str, int] = Counter()
    for row in rows:
        roles = row["target"]["sealed_semantic_role_positions"]
        anchor_positions = set(int(value) for value in roles.values())
        intersection = frozen & anchor_positions
        exact.append(frozen == anchor_positions)
        precisions.append(
            len(intersection) / max(len(frozen), 1)
        )
        recalls.append(
            len(intersection) / max(len(anchor_positions), 1)
        )
        for role, position in roles.items():
            included = int(position) in frozen
            role_coverage[role].append(included)
            if included:
                selected_role_names[role] += 1
    return {
        "revealed_after_selection": True,
        "selection_uses_this_audit": False,
        "frozen_physical_positions": sorted(frozen),
        "exact_five_anchor_position_set_rate": float(np.mean(exact)),
        "mean_anchor_precision": float(np.mean(precisions)),
        "mean_anchor_recall": float(np.mean(recalls)),
        "role_coverage_rate": {
            role: float(np.mean(values))
            for role, values in sorted(role_coverage.items())
        },
        "selected_role_occurrence_counts": dict(
            sorted(selected_role_names.items())
        ),
    }


def run_cell(
    model,
    layers,
    device,
    model_name: str,
    domain: str,
    split: str,
    template: int,
    rows: list[dict[str, Any]],
    batch_size: int,
    output_root: Path,
) -> dict[str, Any]:
    donor_all, donor_audit = choose_donors(rows, same_answer=False)
    same_all, same_audit = choose_donors(rows, same_answer=True)
    candidate_ids = semantic_case(rows[0]["target"])[
        "candidate_token_ids"
    ]
    semantic_width = len(
        semantic_case(rows[0]["target"])["input_ids"]
    )
    prompt_width = len(rows[0]["target"]["input_ids"])
    if prompt_width >= semantic_width:
        raise RuntimeError(
            f"expected teacher-forced prefix: {prompt_width}/{semantic_width}"
        )
    all_positions = list(range(prompt_width))
    masks = [
        {
            "condition": "target_noop",
            "positions": all_positions,
            "source_kind": "target",
        },
        {
            "condition": "different_answer_full",
            "positions": all_positions,
            "source_kind": "different_answer",
        },
        {
            "condition": "same_answer_full",
            "positions": all_positions,
            "source_kind": "same_answer",
        },
    ]
    for position in all_positions:
        event_id = f"p{position:03d}"
        masks.append({
            "condition": f"single:{event_id}",
            "positions": [position],
            "source_kind": "different_answer",
        })
        masks.append({
            "condition": f"loo:{event_id}",
            "positions": [
                value for value in all_positions if value != position
            ],
            "source_kind": "different_answer",
        })
    fingerprint_rows, trajectory_rows = run_masks(
        model,
        layers,
        device,
        rows,
        donor_all,
        same_all,
        candidate_ids,
        masks,
        batch_size,
        include_trajectory=True,
    )
    fingerprints = position_fingerprints(
        fingerprint_rows, prompt_width
    )
    ranked = rank_positions(fingerprints)

    prefix_masks = []
    ordered_positions = [
        int(item["physical_position"]) for item in ranked
    ]
    for count in range(1, prompt_width + 1):
        prefix_masks.append({
            "condition": f"prefix:{count:03d}",
            "positions": ordered_positions[:count],
            "source_kind": "different_answer",
        })
    prefix_rows, _ = run_masks(
        model,
        layers,
        device,
        rows,
        donor_all,
        same_all,
        candidate_ids,
        prefix_masks,
        batch_size,
    )
    prefix_summary = [
        summarize_condition(
            prefix_rows, f"prefix:{count:03d}"
        )
        for count in range(1, prompt_width + 1)
    ]
    eligible = [
        item for item in prefix_summary
        if passes_source_gate(item)
    ]
    if eligible:
        current = set(eligible[0]["physical_positions"])
        prefix_gate_pass = True
    else:
        current = set(all_positions)
        prefix_gate_pass = False

    reverse_audit = []
    for position in reversed(ordered_positions):
        if position not in current:
            continue
        trial = sorted(current - {position})
        if not trial:
            continue
        condition = f"reverse_delete:p{position:03d}"
        trial_rows, _ = run_masks(
            model,
            layers,
            device,
            rows,
            donor_all,
            same_all,
            candidate_ids,
            [{
                "condition": condition,
                "positions": trial,
                "source_kind": "different_answer",
            }],
            batch_size,
        )
        trial_summary = summarize_condition(trial_rows, condition)
        passed = passes_source_gate(trial_summary)
        reverse_audit.append({
            "removed_position": position,
            "trial_positions": trial,
            "passed": passed,
            "summary": trial_summary,
        })
        if passed:
            current.remove(position)

    frozen_positions = sorted(current)
    final_masks = [
        {
            "condition": "frozen_source",
            "positions": frozen_positions,
            "source_kind": "different_answer",
        },
        {
            "condition": "frozen_same_answer_control",
            "positions": frozen_positions,
            "source_kind": "same_answer",
        },
        {
            "condition": "frozen_target_noop",
            "positions": frozen_positions,
            "source_kind": "target",
        },
    ]
    final_rows, _ = run_masks(
        model,
        layers,
        device,
        rows,
        donor_all,
        same_all,
        candidate_ids,
        final_masks,
        batch_size,
    )
    final_summary = {
        mask["condition"]: summarize_condition(
            final_rows, mask["condition"]
        )
        for mask in final_masks
    }
    noops = [
        row
        for row in final_rows
        if row["condition"] == "frozen_target_noop"
    ]
    target_clean_correct = float(np.mean([
        row["target_margin"] < 0 for row in noops
    ]))
    noop_prediction_agreement = final_summary[
        "frozen_target_noop"
    ]["target_rate"]
    same_answer_target_rate = final_summary[
        "frozen_same_answer_control"
    ]["target_rate"]
    final_gate = (
        passes_source_gate(final_summary["frozen_source"])
        and noop_prediction_agreement >= 0.99
        and same_answer_target_rate >= 0.95
    )
    semantic_audit = reveal_semantic_roles(
        frozen_positions, rows
    )
    cell_root = (
        output_root / domain / split / f"template_{template}"
    )
    write_jsonl(cell_root / "fingerprint_rows.jsonl", fingerprint_rows)
    write_jsonl(cell_root / "prefix_rows.jsonl", prefix_rows)
    write_jsonl(cell_root / "final_rows.jsonl", final_rows)
    write_jsonl(cell_root / "diagnostic_trajectory_rows.jsonl", trajectory_rows)
    write_json(cell_root / "position_fingerprints.json", {
        "schema_version": "phase1004_position_fingerprints.v1",
        "phase": PHASE,
        "model": model_name,
        "domain": domain,
        "split": split,
        "template": template,
        "position_count": prompt_width,
        "semantic_input_position_count": semantic_width,
        "ranked_events": ranked,
    })
    summary = {
        "schema_version": "phase1004_blind_source_cell_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "domain": domain,
        "split": split,
        "template": template,
        "n": len(rows),
        "source_depth": SOURCE_DEPTH,
        "position_count": prompt_width,
        "semantic_input_position_count": semantic_width,
        "selection_uses_semantic_labels": False,
        "selection_uses_confirmation_to_tune_rule": False,
        "donor_audit": donor_audit,
        "same_answer_donor_audit": same_audit,
        "instrument_conditions": {
            condition: summarize_condition(
                fingerprint_rows, condition
            )
            for condition in (
                "target_noop",
                "different_answer_full",
                "same_answer_full",
            )
        },
        "prefix_gate_pass": prefix_gate_pass,
        "first_passing_prefix": eligible[0] if eligible else None,
        "reverse_delete_audit": reverse_audit,
        "frozen_physical_positions": frozen_positions,
        "frozen_position_count": len(frozen_positions),
        "final_conditions": final_summary,
        "target_clean_correct_rate_proxy": target_clean_correct,
        "noop_prediction_agreement": noop_prediction_agreement,
        "same_answer_control_target_rate": same_answer_target_rate,
        "final_source_gate_pass": final_gate,
        "semantic_reconstruction_audit": semantic_audit,
        "trajectory_row_count": len(trajectory_rows),
    }
    write_json(cell_root / "summary.json", summary)
    return summary


def behavior_gate(
    model,
    device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    result = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        cases = [semantic_case(row["target"]) for row in batch]
        candidate_ids = cases[0]["candidate_token_ids"]
        logits, _, _ = capture(
            model,
            device,
            cases,
            candidate_ids,
            trajectory=False,
        )
        labels = list(candidate_ids)
        predictions = prediction_labels(logits, labels)
        for item, case, prediction in zip(batch, cases, predictions):
            result.append({
                "record_id": case["record_id"],
                "template": case["template"],
                "gold": case["gold"],
                "prediction": prediction,
                "correct": prediction == case["gold"],
            })
        del logits
    accuracy = float(np.mean([row["correct"] for row in result]))
    return {
        "n": len(result),
        "candidate_accuracy": accuracy,
        "gate_threshold": 0.95,
        "gate_pass": accuracy >= 0.95,
        "template_accuracy": {
            str(template): float(np.mean([
                row["correct"]
                for row in result
                if int(row["template"]) == template
            ]))
            for template in sorted({
                int(row["template"]) for row in result
            })
        },
    }


def run_model(
    model_name: str,
    batch_size: int,
    *,
    use_8bit: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1004 requires CUDA")
    precision_root = "blind_source" if use_8bit else "blind_source_bf16"
    output_root = OUT_ROOT / precision_root / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name,
            dtype=torch.bfloat16,
            use_8bit=use_8bit,
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        cell_summaries = []
        behavior = {}
        for domain in DOMAINS:
            for split in ("discovery", "confirmation"):
                all_rows = selected_directional_rows(
                    model_name, domain, split
                )
                behavior[f"{domain}:{split}"] = behavior_gate(
                    model, device, all_rows, batch_size
                )
                if not behavior[f"{domain}:{split}"]["gate_pass"]:
                    continue
                groups = grouped_rows(all_rows)
                for template, rows in groups.items():
                    print(
                        f"[cell] {model_name}/{domain}/{split}/t{template}",
                        flush=True,
                    )
                    summary = run_cell(
                        model,
                        layers,
                        device,
                        model_name,
                        domain,
                        split,
                        template,
                        rows,
                        batch_size,
                        output_root,
                    )
                    cell_summaries.append(summary)
        summary = {
            "schema_version": "phase1004_blind_source_model_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "precision": "8bit" if use_8bit else "bf16",
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "behavior": behavior,
            "cell_count": len(cell_summaries),
            "source_gate_pass_count": sum(
                item["final_source_gate_pass"]
                for item in cell_summaries
            ),
            "semantic_exact_reconstruction_count": sum(
                item["semantic_reconstruction_audit"][
                    "exact_five_anchor_position_set_rate"
                ] >= 0.99
                for item in cell_summaries
            ),
            "cells": cell_summaries,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 instead of the 8-bit cross-model screen.",
    )
    args = parser.parse_args()
    summary = run_model(
        args.model,
        args.batch_size,
        use_8bit=not args.bf16,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
