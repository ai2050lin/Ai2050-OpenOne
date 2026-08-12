#!/usr/bin/env python3
"""Search for later-layer single-position causal source compression.

Discovery ranks only physical (layer, end-offset) events. Semantic role
labels are revealed after the top event list is frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1004_blind_causal_basis_protocol import (
    DOMAINS,
    selected_directional_rows,
    semantic_case,
    stable_order,
)
from phase1004_blind_receiver_fingerprints import checkpoint_blocks
from phase1004_blind_source_fingerprints import choose_donors


PHASE = 1005
UPSTREAM_PHASE = 1004
UPSTREAM_DIGEST = (
    "de6dbd935417aee274e9a1e1d640d5af"
    "59371c0dfe98b2219f282e91931ade87"
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1005_blind_layerwise_source_compression"
)
PROTOCOL_PATH = OUT_ROOT / "preregistered_protocol.json"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
EVENT_LIMIT = 12


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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


def protocol_payload() -> dict[str, Any]:
    payload = {
        "schema_version": "phase1005_preregistered_protocol.v1",
        "phase": PHASE,
        "protocol_revision": 2,
        "title": (
            "Label-blind layerwise single-position source "
            "compression"
        ),
        "upstream_phase": UPSTREAM_PHASE,
        "upstream_preregistration_digest": UPSTREAM_DIGEST,
        "epistemic_order": [
            "Freeze the physical end-offset and layer universe.",
            "Screen discovery rows without semantic labels.",
            "Freeze the top 12 physical events per domain.",
            "Test them on disjoint names, values, and templates.",
            "Apply different-answer, same-answer, and no-op controls.",
            "Reveal semantic roles only after event selection.",
        ],
        "required_execution_order": [
            "qwen3:8bit",
            "glm4:8bit",
            "deepseek7b:8bit",
            "qwen3:bf16",
        ],
        "parent_behavior_gate": {
            "source": "Phase1004 behavior summaries",
            "required_discovery_accuracy": 0.95,
            "required_confirmation_accuracy": 0.95,
        },
        "measurement_boundary": (
            "The frozen Phase1004 semantic value-decision boundary: "
            "append only the fixed Answer prefix, then measure the "
            "next attribute-value token. Interventions remain "
            "restricted to original raw-prompt positions."
        ),
        "revision_audit": {
            "revision_1": (
                "The first implementation scanned raw-prompt next "
                "token logits instead of the upstream semantic "
                "value-decision boundary. Its no-op target rate was "
                "only 0.75, proving boundary mismatch before any "
                "result could be interpreted."
            ),
            "revision_1_result_used": False,
            "revision_1_artifacts_retained_at": (
                "invalid_pre_semantic_boundary_fix"
            ),
            "revision_2": (
                "Use semantic_case for every forward pass while "
                "computing physical offsets from the raw prompt, so "
                "Answer-prefix positions remain ineligible sources."
            ),
            "revision_2_chosen_from_scientific_result": False,
        },
        "discovery_rows": {
            "count_per_domain": 16,
            "count_per_template": 8,
            "selection": "stable hash only",
        },
        "confirmation_rows": {
            "count_per_domain": 64,
            "count_per_template": 32,
            "disjoint_from_discovery": True,
        },
        "physical_event_universe": {
            "layers": (
                "12 frozen relative residual checkpoints inherited "
                "from Phase1004"
            ),
            "positions": (
                "all common negative offsets from the end of the "
                "raw prompt across discovery templates"
            ),
            "semantic_labels_used": False,
        },
        "discovery_ranking": [
            "minimum donor rate over both discovery templates",
            "minimum median normalized transfer over both templates",
            "overall donor rate",
            "overall median normalized transfer",
            "earlier event id only as deterministic tie break",
        ],
        "frozen_event_limit": EVENT_LIMIT,
        "confirmation_controls": [
            "different-answer cross-world donor",
            "same-answer cross-world donor",
            "target-to-target no-op",
        ],
        "compressed_event_gate": {
            "discovery_each_template_donor_rate": 0.80,
            "discovery_each_template_median_transfer": 0.50,
            "confirmation_each_template_donor_rate": 0.80,
            "confirmation_each_template_median_transfer": 0.50,
            "confirmation_each_template_same_answer_target_rate": 0.95,
            "confirmation_each_template_noop_target_rate": 0.99,
        },
        "diagnostic_boundary": (
            "The gate is an operational sufficiency/control test. "
            "It is not a language law, native model equation, or "
            "proof that the patched event is the only route."
        ),
        "valid_no_go": [
            "No parent-qualified domain.",
            "No selected event passes confirmation.",
            "A discovery event fails on either confirmation template.",
            "Same-answer or no-op controls fail.",
        ],
        "forbidden_claims": [
            "A single position stores the whole concept.",
            "A selected residual coordinate is a neuron.",
            "Compression proves a complete language mechanism.",
            "Operational thresholds are a discovered formula.",
        ],
        "preregistration_digest": None,
    }
    payload["preregistration_digest"] = digest({
        key: value
        for key, value in payload.items()
        if key != "preregistration_digest"
    })
    return payload


def write_or_verify_protocol(*, write: bool) -> dict[str, Any]:
    expected = protocol_payload()
    if write:
        if PROTOCOL_PATH.exists():
            current = json.loads(
                PROTOCOL_PATH.read_text(encoding="utf-8")
            )
            if current != expected:
                raise RuntimeError(
                    "Refusing to overwrite a different Phase1005 "
                    "preregistration"
                )
        else:
            write_json(PROTOCOL_PATH, expected)
        return expected
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(
            "Phase1005 protocol is missing; run --write-protocol first"
        )
    current = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError("Phase1005 preregistration digest drift")
    return current


def source_root(use_8bit: bool) -> str:
    return "blind_source" if use_8bit else "blind_source_bf16"


def upstream_behavior(
    model_name: str,
    use_8bit: bool,
) -> dict[str, Any]:
    path = (
        ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1004_blind_causal_state_basis"
        / source_root(use_8bit)
        / model_name
        / "summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["behavior"]


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
                f"phase1005-compression-screen:t{template}",
            ),
        )
        selected.extend(ordered[:8])
    if len(selected) != 16:
        raise RuntimeError(
            f"Phase1005 discovery screen size drift: {len(selected)}"
        )
    return selected


def candidate_logits(
    logits: torch.Tensor,
    candidate_ids: dict[str, int],
) -> tuple[list[str], torch.Tensor]:
    labels = list(candidate_ids)
    token_ids = torch.tensor(
        [candidate_ids[label] for label in labels],
        dtype=torch.long,
        device=logits.device,
    )
    return labels, logits.index_select(-1, token_ids).float()


def case_tensors(cases: list[dict[str, Any]], device):
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"Input width drift: {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    attention = torch.ones_like(input_ids)
    return input_ids, attention


def capture(
    model,
    device,
    cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    layer_numbers: list[int],
) -> tuple[list[str], torch.Tensor, dict[int, torch.Tensor]]:
    input_ids, attention = case_tensors(cases, device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    labels, logits = candidate_logits(
        output.logits[:, -1, :], candidate_ids
    )
    hidden = {
        layer_number: output.hidden_states[layer_number].detach()
        for layer_number in layer_numbers
    }
    del output, input_ids, attention
    return labels, logits, hidden


def forward_patch(
    model,
    layers,
    device,
    target_cases: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    event: dict[str, Any],
    source_hidden: torch.Tensor,
    raw_prompt_width: int,
) -> torch.Tensor:
    input_ids, attention = case_tensors(target_cases, device)
    width = input_ids.shape[1]
    position = raw_prompt_width + int(event["end_offset"])
    if not 0 <= position < raw_prompt_width <= width:
        raise RuntimeError(
            "Invalid raw-prompt end offset "
            f"{event['end_offset']} for raw width "
            f"{raw_prompt_width} and semantic width {width}"
        )
    count = [0]

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        replacement = value.clone()
        replacement[:, position, :] = source_hidden[
            :, position, :
        ].to(device=value.device, dtype=value.dtype)
        count[0] += 1
        if isinstance(output, tuple):
            return (replacement,) + output[1:]
        return replacement

    handle = layers[int(event["block_index"])].register_forward_hook(
        hook
    )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if count[0] != 1:
            raise RuntimeError(
                f"Patch count drift for {event['event_id']}: {count[0]}"
            )
        _, result = candidate_logits(
            output.logits[:, -1, :], candidate_ids
        )
        return result
    finally:
        handle.remove()
        del input_ids, attention


def condition_rows(
    directional_rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    target_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    labels: list[str],
    event: dict[str, Any],
    condition: str,
) -> list[dict[str, Any]]:
    label_index = {label: index for index, label in enumerate(labels)}
    target_gold = [
        label_index[row["target"]["gold"]]
        for row in directional_rows
    ]
    donor_gold = [label_index[row["gold"]] for row in donors]
    target_gold_tensor = torch.tensor(
        target_gold,
        dtype=torch.long,
        device=patched_logits.device,
    )
    donor_gold_tensor = torch.tensor(
        donor_gold,
        dtype=torch.long,
        device=patched_logits.device,
    )
    row_index = torch.arange(
        len(directional_rows), device=patched_logits.device
    )
    target_margin = (
        target_logits[row_index, donor_gold_tensor]
        - target_logits[row_index, target_gold_tensor]
    )
    donor_margin = (
        donor_logits[row_index, donor_gold_tensor]
        - donor_logits[row_index, target_gold_tensor]
    )
    patched_margin = (
        patched_logits[row_index, donor_gold_tensor]
        - patched_logits[row_index, target_gold_tensor]
    )
    predictions = patched_logits.argmax(dim=-1)
    output = []
    for index, row in enumerate(directional_rows):
        denominator = float(
            donor_margin[index] - target_margin[index]
        )
        transfer = float(
            (patched_margin[index] - target_margin[index])
            / max(abs(denominator), 1e-8)
        )
        output.append({
            "schema_version": (
                "phase1005_layer_position_intervention_row.v1"
            ),
            "phase": PHASE,
            "upstream_phase": UPSTREAM_PHASE,
            "model": row["model"],
            "domain": row["domain"],
            "split": row["split"],
            "template": int(row["target"]["template"]),
            "pair_id": row["pair_id"],
            "direction": row["direction"],
            "target_record_id": row["target"]["record_id"],
            "donor_record_id": donors[index]["record_id"],
            "target_gold": row["target"]["gold"],
            "donor_gold": donors[index]["gold"],
            "event_id": event["event_id"],
            "checkpoint_index": int(event["checkpoint_index"]),
            "block_index": int(event["block_index"]),
            "layer_number": int(event["layer_number"]),
            "relative_depth": float(event["relative_depth"]),
            "end_offset": int(event["end_offset"]),
            "physical_position": (
                len(row["target"]["input_ids"])
                + int(event["end_offset"])
            ),
            "condition": condition,
            "target_margin": float(target_margin[index]),
            "donor_margin": float(donor_margin[index]),
            "patched_margin": float(patched_margin[index]),
            "normalized_transfer": transfer,
            "prediction": labels[int(predictions[index])],
            "predicted_target": (
                int(predictions[index]) == target_gold[index]
            ),
            "predicted_donor": (
                int(predictions[index]) == donor_gold[index]
            ),
        })
    return output


def grouped_batches(
    directional_rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    same_donors: list[dict[str, Any]],
    batch_size: int,
):
    triples = list(zip(directional_rows, donors, same_donors))
    triples.sort(
        key=lambda item: (
            int(item[0]["target"]["template"]),
            item[0]["pair_id"],
            item[0]["direction"],
        )
    )
    grouped: dict[int, list[tuple[Any, Any, Any]]] = defaultdict(list)
    for triple in triples:
        grouped[int(triple[0]["target"]["template"])].append(triple)
    for template, values in sorted(grouped.items()):
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            yield (
                template,
                [item[0] for item in batch],
                [item[1] for item in batch],
                [item[2] for item in batch],
            )


def run_events(
    model,
    layers,
    device,
    directional_rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    same_donors: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    events: list[dict[str, Any]],
    conditions: tuple[str, ...],
    batch_size: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    layer_numbers = sorted({
        int(event["layer_number"]) for event in events
    })
    batches = list(grouped_batches(
        directional_rows, donors, same_donors, batch_size
    ))
    for batch_number, (
        template,
        batch_rows,
        donor_batch,
        same_batch,
    ) in enumerate(batches, 1):
        target_cases = [
            semantic_case(row["target"]) for row in batch_rows
        ]
        raw_prompt_widths = {
            len(row["target"]["input_ids"]) for row in batch_rows
        }
        if len(raw_prompt_widths) != 1:
            raise RuntimeError(
                f"Raw prompt width drift: {raw_prompt_widths}"
            )
        raw_prompt_width = next(iter(raw_prompt_widths))
        labels, target_logits, target_hidden = capture(
            model,
            device,
            target_cases,
            candidate_ids,
            layer_numbers,
        )
        _, donor_logits, donor_hidden = capture(
            model,
            device,
            donor_batch,
            candidate_ids,
            layer_numbers,
        )
        same_logits = same_hidden = None
        if "same_answer" in conditions:
            _, same_logits, same_hidden = capture(
                model,
                device,
                same_batch,
                candidate_ids,
                layer_numbers,
            )
        for event_number, event in enumerate(events, 1):
            layer_number = int(event["layer_number"])
            for condition in conditions:
                if condition == "different_answer":
                    source = donor_hidden[layer_number]
                    margin_donors = donor_batch
                    margin_logits = donor_logits
                elif condition == "same_answer":
                    source = same_hidden[layer_number]
                    margin_donors = same_batch
                    margin_logits = same_logits
                elif condition == "target_noop":
                    source = target_hidden[layer_number]
                    margin_donors = donor_batch
                    margin_logits = donor_logits
                else:
                    raise RuntimeError(f"Unknown condition {condition}")
                patched = forward_patch(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    event,
                    source,
                    raw_prompt_width,
                )
                output_rows.extend(condition_rows(
                    batch_rows,
                    margin_donors,
                    target_logits,
                    margin_logits,
                    patched,
                    labels,
                    event,
                    condition,
                ))
                del patched
            if event_number % 60 == 0 or event_number == len(events):
                print(
                    f"[{progress_label}] template {template} "
                    f"batch {batch_number}/{len(batches)} "
                    f"event {event_number}/{len(events)}",
                    flush=True,
                )
        del (
            target_logits,
            donor_logits,
            target_hidden,
            donor_hidden,
        )
        if same_logits is not None:
            del same_logits, same_hidden
        gc.collect()
        torch.cuda.empty_cache()
    return output_rows


def summarize_rows(
    rows: list[dict[str, Any]],
    event_id: str,
    condition: str,
) -> dict[str, Any]:
    values = [
        row
        for row in rows
        if row["event_id"] == event_id
        and row["condition"] == condition
    ]
    if not values:
        raise RuntimeError(f"Missing {event_id}/{condition}")

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        transfers = [float(item["normalized_transfer"]) for item in items]
        return {
            "n": len(items),
            "donor_rate": float(np.mean([
                item["predicted_donor"] for item in items
            ])),
            "target_rate": float(np.mean([
                item["predicted_target"] for item in items
            ])),
            "mean_normalized_transfer": float(np.mean(transfers)),
            "median_normalized_transfer": float(
                np.median(transfers)
            ),
        }

    result = metrics(values)
    result["template_metrics"] = {
        str(template): metrics([
            item
            for item in values
            if int(item["template"]) == template
        ])
        for template in sorted({
            int(item["template"]) for item in values
        })
    }
    return result


def event_universe(
    n_layers: int,
    discovery_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    template_widths: dict[int, set[int]] = defaultdict(set)
    for row in discovery_rows:
        template_widths[int(row["target"]["template"])].add(
            len(row["target"]["input_ids"])
        )
    if any(len(widths) != 1 for widths in template_widths.values()):
        raise RuntimeError(f"Template width drift: {template_widths}")
    common_width = min(next(iter(widths)) for widths in template_widths.values())
    events = []
    for checkpoint_index, block_index in enumerate(
        checkpoint_blocks(n_layers)
    ):
        layer_number = block_index + 1
        for end_distance in range(common_width, 0, -1):
            end_offset = -end_distance
            events.append({
                "event_id": (
                    f"r{checkpoint_index:02d}.e{end_distance:03d}"
                ),
                "checkpoint_index": checkpoint_index,
                "block_index": block_index,
                "layer_number": layer_number,
                "relative_depth": layer_number / n_layers,
                "end_offset": end_offset,
                "selection_uses_semantic_labels": False,
            })
    return events


def rank_discovery(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summaries = {
        event["event_id"]: summarize_rows(
            rows, event["event_id"], "different_answer"
        )
        for event in events
    }
    event_lookup = {event["event_id"]: event for event in events}

    def ranking(event_id: str):
        summary = summaries[event_id]
        templates = list(summary["template_metrics"].values())
        return (
            -min(item["donor_rate"] for item in templates),
            -min(
                item["median_normalized_transfer"]
                for item in templates
            ),
            -summary["donor_rate"],
            -summary["median_normalized_transfer"],
            event_id,
        )

    selected = []
    for rank, event_id in enumerate(
        sorted(summaries, key=ranking)[:EVENT_LIMIT],
        1,
    ):
        selected.append({
            **event_lookup[event_id],
            "discovery_rank": rank,
            "discovery_metrics": summaries[event_id],
            "selection_uses_confirmation": False,
            "selection_uses_semantic_labels": False,
        })
    return selected, summaries


def template_gate(
    different: dict[str, Any],
    same: dict[str, Any],
    noop: dict[str, Any],
) -> bool:
    templates = sorted(different["template_metrics"])
    return all(
        different["template_metrics"][template]["donor_rate"] >= 0.80
        and different["template_metrics"][template][
            "median_normalized_transfer"
        ]
        >= 0.50
        and same["template_metrics"][template]["target_rate"] >= 0.95
        and noop["template_metrics"][template]["target_rate"] >= 0.99
        for template in templates
    )


def discovery_gate(metrics: dict[str, Any]) -> bool:
    return all(
        item["donor_rate"] >= 0.80
        and item["median_normalized_transfer"] >= 0.50
        for item in metrics["template_metrics"].values()
    )


def reveal_roles(
    event: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    unmatched = 0
    for row in rows:
        case = row["target"]
        position = len(case["input_ids"]) + int(event["end_offset"])
        matched = False
        for role, role_position in case[
            "sealed_semantic_role_positions"
        ].items():
            if int(role_position) == position:
                counts[role] += 1
                matched = True
        if not matched:
            unmatched += 1
    return {
        "revealed_after_selection": True,
        "selection_uses_this_audit": False,
        "event_id": event["event_id"],
        "end_offset": int(event["end_offset"]),
        "role_match_rate": {
            role: count / len(rows)
            for role, count in sorted(counts.items())
        },
        "unmatched_rate": unmatched / len(rows),
    }


def run_domain(
    model,
    layers,
    device,
    model_name: str,
    domain: str,
    n_layers: int,
    batch_size: int,
    output_root: Path,
) -> dict[str, Any]:
    discovery_all = selected_directional_rows(
        model_name, domain, "discovery"
    )
    discovery = screen_subset(discovery_all)
    confirmation = selected_directional_rows(
        model_name, domain, "confirmation"
    )
    discovery_donors, discovery_donor_audit = choose_donors(
        discovery, same_answer=False
    )
    discovery_same, discovery_same_audit = choose_donors(
        discovery, same_answer=True
    )
    confirmation_donors, confirmation_donor_audit = choose_donors(
        confirmation, same_answer=False
    )
    confirmation_same, confirmation_same_audit = choose_donors(
        confirmation, same_answer=True
    )
    discovery_candidate_ids = discovery[0]["target"][
        "candidate_token_ids"
    ]
    confirmation_candidate_ids = confirmation[0]["target"][
        "candidate_token_ids"
    ]
    events = event_universe(n_layers, discovery)
    print(
        f"[domain] {model_name}/{domain}: "
        f"{len(events)} discovery events",
        flush=True,
    )
    discovery_rows = run_events(
        model,
        layers,
        device,
        discovery,
        discovery_donors,
        discovery_same,
        discovery_candidate_ids,
        events,
        ("different_answer",),
        batch_size,
        f"discovery:{model_name}:{domain}",
    )
    selected, discovery_summaries = rank_discovery(
        discovery_rows, events
    )
    domain_root = output_root / domain
    write_jsonl(
        domain_root / "discovery_rows.jsonl", discovery_rows
    )
    write_json(
        domain_root / "discovery_event_summaries.json",
        discovery_summaries,
    )
    write_json(
        domain_root / "frozen_discovery_events.json",
        selected,
    )
    confirmation_events = [
        {
            key: value
            for key, value in event.items()
            if key
            not in {
                "discovery_metrics",
                "discovery_rank",
                "selection_uses_confirmation",
            }
        }
        for event in selected
    ]
    confirmation_rows = run_events(
        model,
        layers,
        device,
        confirmation,
        confirmation_donors,
        confirmation_same,
        confirmation_candidate_ids,
        confirmation_events,
        ("different_answer", "same_answer", "target_noop"),
        batch_size,
        f"confirmation:{model_name}:{domain}",
    )
    confirmed = []
    for event in selected:
        event_id = event["event_id"]
        different = summarize_rows(
            confirmation_rows, event_id, "different_answer"
        )
        same = summarize_rows(
            confirmation_rows, event_id, "same_answer"
        )
        noop = summarize_rows(
            confirmation_rows, event_id, "target_noop"
        )
        gate = (
            discovery_gate(event["discovery_metrics"])
            and template_gate(different, same, noop)
        )
        confirmed.append({
            **event,
            "confirmation_different_answer": different,
            "confirmation_same_answer": same,
            "confirmation_target_noop": noop,
            "compressed_event_gate_pass": gate,
            "semantic_reconstruction_audit": reveal_roles(
                event, confirmation
            ),
        })
    write_jsonl(
        domain_root / "confirmation_rows.jsonl", confirmation_rows
    )
    summary = {
        "schema_version": (
            "phase1005_source_compression_domain_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "domain": domain,
        "status": "complete",
        "selection_uses_semantic_labels": False,
        "selection_uses_confirmation": False,
        "discovery_n": len(discovery),
        "confirmation_n": len(confirmation),
        "event_universe_count": len(events),
        "frozen_event_count": len(selected),
        "discovery_donor_audit": discovery_donor_audit,
        "discovery_same_answer_donor_audit": discovery_same_audit,
        "confirmation_donor_audit": confirmation_donor_audit,
        "confirmation_same_answer_donor_audit": (
            confirmation_same_audit
        ),
        "frozen_events": confirmed,
        "compressed_event_pass_count": sum(
            item["compressed_event_gate_pass"]
            for item in confirmed
        ),
        "compressed_single_position_found": any(
            item["compressed_event_gate_pass"]
            for item in confirmed
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
    write_or_verify_protocol(write=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1005 requires CUDA")
    precision = "8bit" if use_8bit else "bf16"
    output_root = OUT_ROOT / precision / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    behavior = upstream_behavior(model_name, use_8bit)
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name,
            dtype=torch.bfloat16,
            use_8bit=use_8bit,
        )
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        domains = {}
        for domain in DOMAINS:
            parent = (
                behavior[f"{domain}:discovery"]["gate_pass"]
                and behavior[f"{domain}:confirmation"]["gate_pass"]
            )
            if not parent:
                value = {
                    "schema_version": (
                        "phase1005_source_compression_domain_summary.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "status": "upstream_behavior_parent_gate_failed",
                    "upstream_behavior": {
                        "discovery": behavior[
                            f"{domain}:discovery"
                        ],
                        "confirmation": behavior[
                            f"{domain}:confirmation"
                        ],
                    },
                    "compressed_event_pass_count": 0,
                    "compressed_single_position_found": False,
                }
                write_json(output_root / domain / "summary.json", value)
                domains[domain] = value
                continue
            domains[domain] = run_domain(
                model,
                layers,
                device,
                model_name,
                domain,
                info.n_layers,
                batch_size,
                output_root,
            )
        summary = {
            "schema_version": (
                "phase1005_source_compression_model_summary.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "precision": precision,
            "status": "complete",
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "domains": domains,
            "parent_qualified_domain_count": sum(
                item["status"] == "complete"
                for item in domains.values()
            ),
            "compressed_domain_count": sum(
                item["compressed_single_position_found"]
                for item in domains.values()
            ),
            "compressed_event_pass_count": sum(
                item["compressed_event_pass_count"]
                for item in domains.values()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        choices=MODEL_ORDER,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--write-protocol", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_protocol:
        protocol = write_or_verify_protocol(write=True)
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("model is required unless --write-protocol is used")
    summary = run_model(
        args.model,
        args.batch_size,
        use_8bit=not args.bf16,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
