#!/usr/bin/env python3
"""Map blind component receivers downstream of Phase1004 source sets."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    selected_directional_rows,
    semantic_case,
    stable_order,
    write_json,
    write_jsonl,
)
from phase1004_blind_source_fingerprints import (
    SOURCE_DEPTH,
    candidate_logits,
    capture,
    case_tensors,
    choose_donors,
    contrast_margin,
    prediction_labels,
)


CHECKPOINT_COUNT = 12
RECEIVER_LIMIT = 12
BATCH_SIZE = 8
COMPONENTS = ("attn", "mlp", "residual")


def checkpoint_blocks(n_layers: int) -> list[int]:
    values = np.linspace(
        SOURCE_DEPTH,
        n_layers - 1,
        CHECKPOINT_COUNT,
    )
    result = sorted({int(round(value)) for value in values})
    if len(result) != CHECKPOINT_COUNT:
        raise RuntimeError(
            f"checkpoint collision {n_layers}: {result}"
        )
    return result


def event_definitions(n_layers: int) -> list[dict[str, Any]]:
    events = []
    for checkpoint_index, block_index in enumerate(
        checkpoint_blocks(n_layers)
    ):
        relative_depth = (block_index + 1) / n_layers
        for component in COMPONENTS:
            events.append({
                "event_id": (
                    f"r{checkpoint_index:02d}.{component}"
                ),
                "checkpoint_index": checkpoint_index,
                "block_index": block_index,
                "layer_number": block_index + 1,
                "relative_depth": relative_depth,
                "depth_half": (
                    "early_mid"
                    if relative_depth < 0.5
                    else "mid_late"
                ),
                "component": component,
                "position_interface": "answer_boundary",
                "semantic_label": None,
            })
    return events


def event_module(layers, event: dict[str, Any]):
    layer = layers[int(event["block_index"])]
    if event["component"] == "attn":
        return layer.self_attn
    if event["component"] == "mlp":
        return layer.mlp
    if event["component"] == "residual":
        return layer
    raise KeyError(event["component"])


def register_source_patch(
    layers,
    target_hidden: torch.Tensor,
    donor_hidden: torch.Tensor,
    positions: list[int],
):
    patch = target_hidden.clone()
    if positions:
        patch[:, positions, :] = donor_hidden[:, positions, :]
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
    return handle, count


def capture_events(
    model,
    layers,
    device,
    target_cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    events: list[dict[str, Any]],
    *,
    target_hidden: torch.Tensor | None = None,
    donor_hidden: torch.Tensor | None = None,
    source_positions: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    input_ids, attention = case_tensors(target_cases, device)
    captured: dict[str, torch.Tensor] = {}
    event_counts: dict[str, int] = defaultdict(int)
    handles = []
    source_handle = None
    source_count = [0]
    positions = torch.tensor(
        [int(case["answer_boundary"]) for case in target_cases],
        dtype=torch.long,
        device=device,
    )
    try:
        if donor_hidden is not None:
            if target_hidden is None or source_positions is None:
                raise RuntimeError("incomplete source patch")
            source_handle, source_count = register_source_patch(
                layers,
                target_hidden,
                donor_hidden,
                source_positions,
            )
        for event in events:
            def make_hook(event_id: str):
                def hook(module, args, output):
                    value = (
                        output[0] if isinstance(output, tuple) else output
                    )
                    batch_index = torch.arange(
                        value.shape[0], device=value.device
                    )
                    captured[event_id] = value[
                        batch_index,
                        positions.to(value.device),
                        :,
                    ].detach()
                    event_counts[event_id] += 1
                    return output

                return hook

            handles.append(
                event_module(layers, event).register_forward_hook(
                    make_hook(event["event_id"])
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if donor_hidden is not None and source_count[0] != 1:
            raise RuntimeError(
                f"source capture count drift: {source_count[0]}"
            )
        missing = [
            event["event_id"]
            for event in events
            if event_counts[event["event_id"]] != 1
        ]
        if missing:
            raise RuntimeError(f"event capture drift: {missing[:5]}")
        _, logits = candidate_logits(
            output.logits[:, -1, :], candidate_ids
        )
        return logits, captured
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()
        del input_ids, attention


def forward_receiver(
    model,
    layers,
    device,
    target_cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    event: dict[str, Any],
    receiver_vectors: torch.Tensor,
    *,
    target_hidden: torch.Tensor | None = None,
    donor_hidden: torch.Tensor | None = None,
    source_positions: list[int] | None = None,
) -> torch.Tensor:
    input_ids, attention = case_tensors(target_cases, device)
    answer_positions = torch.tensor(
        [int(case["answer_boundary"]) for case in target_cases],
        dtype=torch.long,
        device=device,
    )
    source_handle = None
    source_count = [0]
    receiver_count = [0]

    def receiver_hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        patched = value.clone()
        batch_index = torch.arange(
            patched.shape[0], device=patched.device
        )
        patched[
            batch_index,
            answer_positions.to(patched.device),
            :,
        ] = receiver_vectors.to(
            device=patched.device, dtype=patched.dtype
        )
        receiver_count[0] += 1
        return (
            (patched,) + output[1:]
            if isinstance(output, tuple)
            else patched
        )

    receiver_handle = event_module(
        layers, event
    ).register_forward_hook(receiver_hook)
    try:
        if donor_hidden is not None:
            if target_hidden is None or source_positions is None:
                raise RuntimeError("incomplete source patch")
            source_handle, source_count = register_source_patch(
                layers,
                target_hidden,
                donor_hidden,
                source_positions,
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if receiver_count[0] != 1:
            raise RuntimeError(
                f"receiver patch count drift: {receiver_count[0]}"
            )
        if donor_hidden is not None and source_count[0] != 1:
            raise RuntimeError(
                f"source patch count drift: {source_count[0]}"
            )
        _, result = candidate_logits(
            output.logits[:, -1, :], candidate_ids
        )
        return result
    finally:
        receiver_handle.remove()
        if source_handle is not None:
            source_handle.remove()
        del input_ids, attention


def source_summary(
    model_name: str,
    domain: str,
    split: str,
    template: int,
    precision_root: str = "blind_source",
) -> dict[str, Any]:
    path = (
        OUT_ROOT
        / precision_root
        / model_name
        / domain
        / split
        / f"template_{template}"
        / "summary.json"
    )
    if not path.exists():
        model_summary_path = (
            OUT_ROOT
            / precision_root
            / model_name
            / "summary.json"
        )
        model_summary = json.loads(
            model_summary_path.read_text(encoding="utf-8")
        )
        behavior = model_summary["behavior"][f"{domain}:{split}"]
        if behavior["gate_pass"]:
            raise FileNotFoundError(
                "Source cell artifact is missing despite a passing "
                f"behavior gate: {path}"
            )
        return {
            "schema_version": (
                "phase1004_skipped_source_cell_reference.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "domain": domain,
            "split": split,
            "template": template,
            "status": "upstream_behavior_gate_failed",
            "artifact_exists": False,
            "behavior_gate": behavior,
            "final_source_gate_pass": False,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def screen_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["target"]["template"])].append(row)
    selected = []
    for template, values in sorted(groups.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(
                f"{row['pair_id']}:{row['direction']}",
                f"receiver-screen:t{template}",
            ),
        )
        selected.extend(ordered[:8])
    if len(selected) != 16:
        raise RuntimeError(f"receiver screen size drift: {len(selected)}")
    return selected


def batches_by_template(
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
):
    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for row, donor in zip(rows, donors):
        grouped[int(row["target"]["template"])].append((row, donor))
    for template, values in sorted(grouped.items()):
        values.sort(
            key=lambda item: (
                item[0]["pair_id"], item[0]["direction"]
            )
        )
        for start in range(0, len(values), batch_size):
            chunk = values[start:start + batch_size]
            yield (
                template,
                [item[0] for item in chunk],
                [item[1] for item in chunk],
            )


def receiver_rows(
    model,
    layers,
    device,
    model_name: str,
    domain: str,
    split: str,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    batch_size: int,
    precision_source_root: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donors, donor_audit = choose_donors(
        rows, same_answer=False
    )
    result = []
    all_batches = list(
        batches_by_template(rows, donors, batch_size)
    )
    for batch_number, (template, batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        target_cases = [semantic_case(row["target"]) for row in batch]
        candidate_ids = target_cases[0]["candidate_token_ids"]
        frozen_source = source_summary(
            model_name,
            domain,
            split,
            template,
            precision_source_root,
        )["frozen_physical_positions"]
        target_logits, target_hidden, _ = capture(
            model,
            device,
            target_cases,
            candidate_ids,
            trajectory=False,
        )
        donor_logits, donor_hidden, _ = capture(
            model,
            device,
            donor_batch,
            candidate_ids,
            trajectory=False,
        )
        _, target_events = capture_events(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            events,
        )
        source_do_logits, source_events = capture_events(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            events,
            target_hidden=target_hidden,
            donor_hidden=donor_hidden,
            source_positions=frozen_source,
        )
        labels = list(candidate_ids)
        target_margin = contrast_margin(
            target_logits, labels, donor_batch, target_cases
        )
        donor_margin = contrast_margin(
            donor_logits, labels, donor_batch, target_cases
        )
        source_margin = contrast_margin(
            source_do_logits, labels, donor_batch, target_cases
        )
        source_predictions = prediction_labels(
            source_do_logits, labels
        )
        for event_number, event in enumerate(events, 1):
            event_id = event["event_id"]
            suff_logits = forward_receiver(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                event,
                source_events[event_id],
            )
            restore_logits = forward_receiver(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                event,
                target_events[event_id],
                target_hidden=target_hidden,
                donor_hidden=donor_hidden,
                source_positions=frozen_source,
            )
            noop_logits = forward_receiver(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                event,
                target_events[event_id],
            )
            suff_margin = contrast_margin(
                suff_logits, labels, donor_batch, target_cases
            )
            restore_margin = contrast_margin(
                restore_logits, labels, donor_batch, target_cases
            )
            suff_predictions = prediction_labels(
                suff_logits, labels
            )
            restore_predictions = prediction_labels(
                restore_logits, labels
            )
            noop_predictions = prediction_labels(
                noop_logits, labels
            )
            for index, item in enumerate(batch):
                denominator = float(
                    donor_margin[index] - target_margin[index]
                )
                source_effect = float(
                    source_margin[index] - target_margin[index]
                )
                result.append({
                    "schema_version": (
                        "phase1004_receiver_fingerprint_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "split": split,
                    "template": template,
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    **event,
                    "source_positions": frozen_source,
                    "target_gold": target_cases[index]["gold"],
                    "donor_gold": donor_batch[index]["gold"],
                    "source_transfer": (
                        source_effect / max(abs(denominator), 1e-8)
                    ),
                    "source_flipped": (
                        source_predictions[index]
                        == donor_batch[index]["gold"]
                    ),
                    "sufficiency_transfer": float(
                        (
                            suff_margin[index] - target_margin[index]
                        )
                        / max(abs(denominator), 1e-8)
                    ),
                    "sufficiency_flipped": (
                        suff_predictions[index]
                        == donor_batch[index]["gold"]
                    ),
                    "mediation_fraction": float(
                        (
                            source_margin[index]
                            - restore_margin[index]
                        )
                        / max(abs(source_effect), 1e-8)
                    ),
                    "restored_to_target": (
                        restore_predictions[index]
                        == target_cases[index]["gold"]
                    ),
                    "receiver_noop_prediction_agreement": (
                        noop_predictions[index]
                        == prediction_labels(
                            target_logits, labels
                        )[index]
                    ),
                    "receiver_noop_max_candidate_logit_error": float(
                        torch.max(
                            torch.abs(
                                noop_logits[index]
                                - target_logits[index]
                            )
                        ).item()
                    ),
                })
            del suff_logits, restore_logits, noop_logits
            if event_number % 6 == 0:
                print(
                    f"[receiver] {model_name}/{domain}/{split} "
                    f"batch {batch_number}/{len(all_batches)} "
                    f"event {event_number}/{len(events)}",
                    flush=True,
                )
        del (
            target_logits,
            target_hidden,
            donor_logits,
            donor_hidden,
            target_events,
            source_do_logits,
            source_events,
        )
    return result, donor_audit


def summarize_events(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    summary = {}
    for event_id, values in groups.items():
        first = values[0]
        summary[event_id] = {
            key: first[key]
            for key in (
                "event_id",
                "checkpoint_index",
                "block_index",
                "layer_number",
                "relative_depth",
                "depth_half",
                "component",
                "position_interface",
            )
        }
        summary[event_id].update({
            "n": len(values),
            "mean_source_transfer": float(np.mean([
                row["source_transfer"] for row in values
            ])),
            "source_flip_rate": float(np.mean([
                row["source_flipped"] for row in values
            ])),
            "mean_sufficiency_transfer": float(np.mean([
                row["sufficiency_transfer"] for row in values
            ])),
            "median_sufficiency_transfer": float(np.median([
                row["sufficiency_transfer"] for row in values
            ])),
            "sufficiency_flip_rate": float(np.mean([
                row["sufficiency_flipped"] for row in values
            ])),
            "median_mediation_fraction": float(np.median([
                row["mediation_fraction"] for row in values
            ])),
            "mean_mediation_fraction": float(np.mean([
                row["mediation_fraction"] for row in values
            ])),
            "positive_mediation_rate": float(np.mean([
                row["mediation_fraction"] > 0 for row in values
            ])),
            "restored_to_target_rate": float(np.mean([
                row["restored_to_target"] for row in values
            ])),
            "receiver_noop_prediction_agreement": float(np.mean([
                row["receiver_noop_prediction_agreement"]
                for row in values
            ])),
            "maximum_receiver_noop_candidate_logit_error": float(max(
                row["receiver_noop_max_candidate_logit_error"]
                for row in values
            )),
            "template_metrics": {
                str(template): {
                    "n": len([
                        row for row in values
                        if int(row["template"]) == template
                    ]),
                    "median_mediation_fraction": float(np.median([
                        row["mediation_fraction"]
                        for row in values
                        if int(row["template"]) == template
                    ])),
                    "mean_sufficiency_transfer": float(np.mean([
                        row["sufficiency_transfer"]
                        for row in values
                        if int(row["template"]) == template
                    ])),
                }
                for template in sorted({
                    int(row["template"]) for row in values
                })
            },
        })
    return summary


def rank_discovery_events(
    summary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        summary.values(),
        key=lambda item: (
            -item["median_mediation_fraction"],
            -item["mean_sufficiency_transfer"],
            -item["restored_to_target_rate"],
            item["relative_depth"],
            item["component"],
        ),
    )
    selected = []
    for rank, item in enumerate(ranked[:RECEIVER_LIMIT], 1):
        selected.append({
            **item,
            "discovery_rank": rank,
            "selection_uses_confirmation": False,
            "selection_uses_semantic_labels": False,
        })
    return selected


def run_domain(
    model,
    layers,
    device,
    model_name: str,
    domain: str,
    all_events: list[dict[str, Any]],
    batch_size: int,
    output_root: Path,
    precision_source_root: str,
) -> dict[str, Any]:
    source_cells = {
        f"{split}:t{template}": source_summary(
            model_name,
            domain,
            split,
            template,
            precision_source_root,
        )
        for split in ("discovery", "confirmation")
        for template in (
            (0, 1) if split == "discovery" else (2, 3)
        )
    }
    parent_gate = all(
        cell["final_source_gate_pass"]
        for cell in source_cells.values()
    )
    if not parent_gate:
        summary = {
            "schema_version": (
                "phase1004_receiver_domain_summary.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "domain": domain,
            "status": "source_parent_gate_failed",
            "source_cells": source_cells,
            "receiver_scan_run": False,
        }
        write_json(output_root / domain / "summary.json", summary)
        return summary

    discovery_all = selected_directional_rows(
        model_name, domain, "discovery"
    )
    discovery = screen_subset(discovery_all)
    discovery_rows, discovery_donor_audit = receiver_rows(
        model,
        layers,
        device,
        model_name,
        domain,
        "discovery",
        discovery,
        all_events,
        batch_size,
        precision_source_root,
    )
    discovery_metrics = summarize_events(discovery_rows)
    selected = rank_discovery_events(discovery_metrics)
    event_lookup = {
        event["event_id"]: event for event in all_events
    }
    frozen_events = [
        event_lookup[item["event_id"]] for item in selected
    ]

    confirmation = selected_directional_rows(
        model_name, domain, "confirmation"
    )
    confirmation_rows, confirmation_donor_audit = receiver_rows(
        model,
        layers,
        device,
        model_name,
        domain,
        "confirmation",
        confirmation,
        frozen_events,
        batch_size,
        precision_source_root,
    )
    confirmation_metrics = summarize_events(confirmation_rows)
    repeated_events = []
    for selected_item in selected:
        event_id = selected_item["event_id"]
        confirmation_item = confirmation_metrics[event_id]
        confirmation_templates_positive = all(
            metric["median_mediation_fraction"] >= 0.10
            and metric["mean_sufficiency_transfer"] >= 0.10
            for metric in confirmation_item[
                "template_metrics"
            ].values()
        )
        if (
            selected_item["median_mediation_fraction"] >= 0.10
            and selected_item["mean_sufficiency_transfer"] >= 0.10
            and confirmation_item[
                "median_mediation_fraction"
            ] >= 0.10
            and confirmation_item[
                "mean_sufficiency_transfer"
            ] >= 0.10
            and confirmation_templates_positive
        ):
            repeated_events.append({
                "event_id": event_id,
                "component": selected_item["component"],
                "relative_depth": selected_item["relative_depth"],
                "depth_half": selected_item["depth_half"],
                "discovery": selected_item,
                "confirmation": confirmation_item,
            })

    repeated_attention_events = [
        item
        for item in repeated_events
        if item["component"] == "attn"
    ]
    domain_root = output_root / domain
    write_jsonl(
        domain_root / "discovery_screen_rows.jsonl",
        discovery_rows,
    )
    write_jsonl(
        domain_root / "confirmation_rows.jsonl",
        confirmation_rows,
    )
    summary = {
        "schema_version": "phase1004_receiver_domain_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "domain": domain,
        "status": "complete",
        "source_parent_gate_pass": parent_gate,
        "source_cells": source_cells,
        "event_universe_count": len(all_events),
        "discovery_n": len(discovery),
        "confirmation_n": len(confirmation),
        "discovery_donor_audit": discovery_donor_audit,
        "confirmation_donor_audit": confirmation_donor_audit,
        "discovery_metrics": discovery_metrics,
        "frozen_events": selected,
        "confirmation_metrics": confirmation_metrics,
        "repeated_event_count": len(repeated_events),
        "repeated_events": repeated_events,
        "repeated_attention_event_count": len(
            repeated_attention_events
        ),
        "repeated_attention_events": repeated_attention_events,
        "head_subspace_parent_authorized": bool(
            repeated_attention_events
        ),
    }
    write_json(domain_root / "summary.json", summary)
    return summary


def run_model(
    model_name: str,
    batch_size: int,
    *,
    use_8bit: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1004 requires CUDA")
    precision_source_root = (
        "blind_source" if use_8bit else "blind_source_bf16"
    )
    receiver_root = (
        "blind_receiver" if use_8bit else "blind_receiver_bf16"
    )
    output_root = OUT_ROOT / receiver_root / model_name
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
        events = event_definitions(info.n_layers)
        domain_summaries = {
            domain: run_domain(
                model,
                layers,
                device,
                model_name,
                domain,
                events,
                batch_size,
                output_root,
                precision_source_root,
            )
            for domain in DOMAINS
        }
        summary = {
            "schema_version": "phase1004_receiver_model_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "precision": "8bit" if use_8bit else "bf16",
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "checkpoint_blocks": checkpoint_blocks(info.n_layers),
            "event_universe": events,
            "domains": domain_summaries,
            "repeated_domain_count": sum(
                item.get("repeated_event_count", 0) > 0
                for item in domain_summaries.values()
            ),
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
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    summary = run_model(
        args.model,
        args.batch_size,
        use_8bit=not args.bf16,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
