#!/usr/bin/env python3
"""Blind targeted follow-up for Phase1017 contextual branch directions.

The target list is frozen from discovery data only. Discovery templates define
one reference interaction direction per word and physical event. Confirmation
templates then measure whether correct and failed units attach to, or depart
from, that reference direction. No raw activation tensor is written to disk.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model
from phase1014_bf16_precision_confirmation import load_bf16
from phase1017_semantic_niche_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    STATES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1017_semantic_niche_scan import contrast_values


STATE_INDEX = {state: index for index, state in enumerate(STATES)}
AMBIGUOUS_STATES = ("a0_l0", "a1_l0", "a0_l1", "a1_l1")
EPSILON = 1e-12


def unit_vector(value: torch.Tensor) -> torch.Tensor | None:
    norm = float(torch.linalg.vector_norm(value).item())
    if norm <= EPSILON:
        return None
    return value / norm


def safe_cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    denominator = float(
        torch.linalg.vector_norm(a).item()
        * torch.linalg.vector_norm(b).item()
    )
    if denominator <= EPSILON:
        return None
    return float(torch.dot(a, b).item() / denominator)


def direction_consistency(vectors: list[torch.Tensor]) -> float | None:
    units = [unit_vector(value) for value in vectors]
    valid = [value.double() for value in units if value is not None]
    count = len(valid)
    if count < 2:
        return None
    direction_sum = torch.stack(valid).sum(dim=0)
    squared = float(torch.dot(direction_sum, direction_sum).item())
    return float((squared - count) / (count * (count - 1)))


def mean_direction(vectors: list[torch.Tensor]) -> torch.Tensor | None:
    units = [unit_vector(value) for value in vectors]
    valid = [value for value in units if value is not None]
    if not valid:
        return None
    return unit_vector(torch.stack(valid).sum(dim=0))


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


class SelectiveEventCapture:
    """Capture only preregistered event/role pairs for a batched forward."""

    def __init__(self, model, layers, selections: list[dict[str, Any]]):
        self.model = model
        self.layers = layers
        self.selections = selections
        self.positions: dict[str, torch.Tensor] = {}
        self.values: dict[tuple[str, str], torch.Tensor] = {}
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.handles = []
        self.grouped: dict[tuple[str, int], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in selections:
            self.grouped[(row["component"], int(row["depth"]))].append(row)

    def _selected(self, value: torch.Tensor, role: str) -> torch.Tensor:
        if value.ndim != 3:
            raise RuntimeError(f"unexpected capture shape: {tuple(value.shape)}")
        positions = self.positions[role].to(value.device)
        batch_index = torch.arange(value.shape[0], device=value.device)
        return value[batch_index, positions, :]

    def _output_hook(self, rows: list[dict[str, Any]]):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("unexpected non-tensor module output")
            for row in rows:
                key = (row["event_id"], row["role"])
                self.values[key] = (
                    self._selected(value, row["role"]).detach().float().cpu()
                )
                self.counts[key] += 1
            return output

        return hook

    def _head_hook(self, rows: list[dict[str, Any]], head_count: int):
        def hook(module, args):
            value = args[0]
            if value.shape[-1] % head_count:
                raise RuntimeError("pre-o_proj width is not head aligned")
            reshaped = value.reshape(
                value.shape[0],
                value.shape[1],
                head_count,
                value.shape[-1] // head_count,
            )
            for row in rows:
                role = row["role"]
                positions = self.positions[role].to(value.device)
                batch_index = torch.arange(value.shape[0], device=value.device)
                selected = reshaped[
                    batch_index,
                    positions,
                    int(row["head"]),
                    :,
                ]
                key = (row["event_id"], role)
                self.values[key] = selected.detach().float().cpu()
                self.counts[key] += 1

        return hook

    def register(self, head_count: int) -> None:
        for (component, depth), rows in sorted(self.grouped.items()):
            if component == "residual" and depth == 0:
                module = self.model.get_input_embeddings()
                self.handles.append(
                    module.register_forward_hook(self._output_hook(rows))
                )
            elif component == "residual":
                self.handles.append(
                    self.layers[depth - 1].register_forward_hook(
                        self._output_hook(rows)
                    )
                )
            elif component == "attention_output":
                self.handles.append(
                    self.layers[depth - 1].self_attn.register_forward_hook(
                        self._output_hook(rows)
                    )
                )
            elif component == "mlp_output":
                self.handles.append(
                    self.layers[depth - 1].mlp.register_forward_hook(
                        self._output_hook(rows)
                    )
                )
            elif component == "attention_head_pre_o_proj":
                self.handles.append(
                    self.layers[
                        depth - 1
                    ].self_attn.o_proj.register_forward_pre_hook(
                        self._head_hook(rows, head_count)
                    )
                )
            else:
                raise RuntimeError(f"unsupported component {component}")

    def begin(self, positions: dict[str, torch.Tensor]) -> None:
        self.positions = positions
        self.values = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        expected = {
            (row["event_id"], row["role"]) for row in self.selections
        }
        missing = sorted(expected - set(self.values))
        repeated = {
            str(key): count for key, count in self.counts.items() if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"selective capture drift missing={missing} repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = {}
        self.values = {}


def behavior_by_unit(model_name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(OUT_ROOT / "behavior" / model_name / "formal.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["unit_id"]].append(row)
    result = {}
    for unit_id, unit_rows in grouped.items():
        state_rows = {
            row["state"]: row
            for row in unit_rows
            if row["state"] in AMBIGUOUS_STATES
        }
        if set(state_rows) != set(AMBIGUOUS_STATES):
            raise RuntimeError(f"incomplete behavior unit {unit_id}")
        ordered = [state_rows[state] for state in AMBIGUOUS_STATES]
        candidate_count = int(sum(row["candidate_hit"] for row in ordered))
        generation_count = int(sum(
            row["generation_first_word_hit"] for row in ordered
        ))
        result[unit_id] = {
            "candidate_hit_count": candidate_count,
            "candidate_stable_correct": candidate_count == 4,
            "generation_hit_count": generation_count,
            "generation_stable_correct": generation_count == 4,
        }
    return result


def capture_units(
    *,
    model,
    device,
    capture: SelectiveEventCapture,
    selections: list[dict[str, Any]],
    units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    behavior: dict[str, dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    observations = []
    for index, unit in enumerate(units):
        cases = [case_by_id[unit["record_ids"][state]] for state in STATES]
        lengths = {len(case["input_ids"]) for case in cases}
        if len(lengths) != 1:
            raise RuntimeError(f"state length drift in {unit['unit_id']}")
        input_ids = torch.tensor(
            [case["input_ids"] for case in cases],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        roles = sorted({row["role"] for row in selections})
        positions = {
            role: torch.tensor(
                [int(case["role_positions"][role]) for case in cases],
                dtype=torch.long,
                device=device,
            )
            for role in roles
        }
        capture.begin(positions)
        try:
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            capture.validate()
            unit_behavior = behavior[unit["unit_id"]]
            for selection in selections:
                key = (selection["event_id"], selection["role"])
                values = capture.values[key]
                deltas, scales = contrast_values(values)
                bt = deltas["BT"].double()
                scale = float(scales["BT"].item())
                magnitude = float(
                    torch.linalg.vector_norm(bt).item() / max(scale, EPSILON)
                )
                observations.append({
                    "model": model_name,
                    "word": unit["word"],
                    "split": unit["split"],
                    "template": int(unit["template"]),
                    "world": int(unit["world"]),
                    "unit_id": unit["unit_id"],
                    "event_id": selection["event_id"],
                    "component": selection["component"],
                    "depth": int(selection["depth"]),
                    "head": selection["head"],
                    "role": selection["role"],
                    "bt": bt,
                    "normalized_magnitude": magnitude,
                    **unit_behavior,
                })
        finally:
            capture.values = {}
            del input_ids, attention_mask, positions
        if (index + 1) % 24 == 0:
            print(
                f"[targeted] {model_name} {index + 1}/{len(units)} units",
                flush=True,
            )
    return observations


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["u"] for row in rows]
    residual = [
        row["residual_unit"]
        for row in rows
        if row["residual_unit"] is not None
    ]
    return {
        "count": len(rows),
        "word_count": len({row["word"] for row in rows}),
        "raw_direction_consistency": direction_consistency(raw),
        "residual_direction_consistency": direction_consistency(residual),
        "median_normalized_magnitude": median([
            row["normalized_magnitude"] for row in rows
        ]),
        "median_reference_attachment": median([
            row["reference_attachment"] for row in rows
        ]),
        "median_residual_fraction": median([
            row["residual_fraction"] for row in rows
        ]),
        "raw_mean_direction": mean_direction(raw),
        "residual_mean_direction": mean_direction(residual),
    }


def strip_directions(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"raw_mean_direction", "residual_mean_direction"}
    }


def analyze_model(
    model_name: str,
    selections: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    output_root: Path,
    elapsed_seconds: float,
    placement: dict[str, Any],
) -> dict[str, Any]:
    discovery = [row for row in observations if row["split"] == "discovery"]
    confirmation = [
        row for row in observations if row["split"] == "confirmation"
    ]
    discovery_groups: dict[tuple[str, str, str], list[torch.Tensor]] = (
        defaultdict(list)
    )
    for row in discovery:
        discovery_groups[
            (row["word"], row["event_id"], row["role"])
        ].append(row["bt"])

    references = {}
    reference_rows = []
    for key, vectors in sorted(discovery_groups.items()):
        reference = mean_direction(vectors)
        if reference is None:
            continue
        references[key] = reference
        reference_rows.append({
            "schema_version": "phase1017_discovery_reference.v1",
            "phase": PHASE,
            "model": model_name,
            "word": key[0],
            "event_id": key[1],
            "role": key[2],
            "unit_count": len(vectors),
            "direction_consistency": direction_consistency(vectors),
            "claim": "discovery-only contextual interaction reference",
        })

    scalar_rows = []
    processed = []
    for row in confirmation:
        key = (row["word"], row["event_id"], row["role"])
        reference = references.get(key)
        u = unit_vector(row["bt"])
        if reference is None or u is None:
            continue
        attachment = float(torch.dot(u, reference).item())
        residual = u - attachment * reference
        residual_fraction = float(torch.linalg.vector_norm(residual).item())
        residual_unit = unit_vector(residual)
        enriched = {
            **row,
            "u": u,
            "reference_attachment": attachment,
            "residual_fraction": residual_fraction,
            "residual_unit": residual_unit,
        }
        processed.append(enriched)
        scalar_rows.append({
            "schema_version": "phase1017_targeted_unit_scalar.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            **{
                key: value
                for key, value in row.items()
                if key != "bt"
            },
            "reference_attachment": attachment,
            "residual_fraction": residual_fraction,
            "reference_source": "discovery_split_only",
        })

    comparison_rows = []
    for selection in selections:
        target_rows = [
            row
            for row in processed
            if row["event_id"] == selection["event_id"]
            and row["role"] == selection["role"]
        ]
        for behavior_name, field in (
            ("candidate", "candidate_stable_correct"),
            ("generation", "generation_stable_correct"),
        ):
            correct = [row for row in target_rows if row[field]]
            failed = [row for row in target_rows if not row[field]]
            correct_summary = group_summary(correct)
            failed_summary = group_summary(failed)
            per_word = []
            for word in sorted({row["word"] for row in target_rows}):
                word_rows = [row for row in target_rows if row["word"] == word]
                word_correct = [row for row in word_rows if row[field]]
                word_failed = [row for row in word_rows if not row[field]]
                if len(word_correct) < 2 or len(word_failed) < 2:
                    continue
                wc = group_summary(word_correct)
                wf = group_summary(word_failed)
                per_word.append({
                    "word": word,
                    "correct_count": len(word_correct),
                    "failed_count": len(word_failed),
                    "attachment_difference_correct_minus_failed": (
                        wc["median_reference_attachment"]
                        - wf["median_reference_attachment"]
                    ),
                    "magnitude_difference_correct_minus_failed": (
                        wc["median_normalized_magnitude"]
                        - wf["median_normalized_magnitude"]
                    ),
                    "raw_correct_failed_cosine": safe_cosine(
                        wc["raw_mean_direction"],
                        wf["raw_mean_direction"],
                    ),
                    "residual_correct_failed_cosine": safe_cosine(
                        wc["residual_mean_direction"],
                        wf["residual_mean_direction"],
                    ),
                })
            attachment_differences = [
                row["attachment_difference_correct_minus_failed"]
                for row in per_word
            ]
            magnitude_differences = [
                row["magnitude_difference_correct_minus_failed"]
                for row in per_word
            ]
            positive_attachment = sum(
                value > 0 for value in attachment_differences
            )
            negative_attachment = sum(
                value < 0 for value in attachment_differences
            )
            comparison_rows.append({
                "schema_version": "phase1017_behavior_direction_comparison.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "event_id": selection["event_id"],
                "component": selection["component"],
                "depth": int(selection["depth"]),
                "head": selection["head"],
                "role": selection["role"],
                "behavior_measure": behavior_name,
                "correct": strip_directions(correct_summary),
                "failed": strip_directions(failed_summary),
                "pooled_raw_correct_failed_cosine": safe_cosine(
                    correct_summary["raw_mean_direction"],
                    failed_summary["raw_mean_direction"],
                ),
                "pooled_residual_correct_failed_cosine": safe_cosine(
                    correct_summary["residual_mean_direction"],
                    failed_summary["residual_mean_direction"],
                ),
                "eligible_word_count": len(per_word),
                "median_word_attachment_difference": median(
                    attachment_differences
                ),
                "attachment_positive_word_fraction": (
                    positive_attachment / len(per_word) if per_word else None
                ),
                "attachment_negative_word_fraction": (
                    negative_attachment / len(per_word) if per_word else None
                ),
                "median_word_magnitude_difference": median(
                    magnitude_differences
                ),
                "per_word": per_word,
                "claim": (
                    "held-out behavior association only; no causal or "
                    "neuron-localization claim"
                ),
            })

    model_root = output_root / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(model_root / "discovery_references.jsonl", reference_rows)
    write_jsonl(model_root / "confirmation_unit_scalars.jsonl", scalar_rows)
    write_jsonl(model_root / "behavior_direction_comparisons.jsonl", comparison_rows)

    comparable = [
        row for row in comparison_rows
        if row["eligible_word_count"] >= 3
        and row["correct"]["count"] >= 8
        and row["failed"]["count"] >= 8
    ]
    repeated_attachment = [
        row for row in comparable
        if max(
            row["attachment_positive_word_fraction"] or 0.0,
            row["attachment_negative_word_fraction"] or 0.0,
        ) >= 0.75
    ]
    summary = {
        "schema_version": "phase1017_targeted_behavior_model.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "selection_count": len(selections),
        "discovery_reference_count": len(reference_rows),
        "confirmation_unit_scalar_count": len(scalar_rows),
        "comparison_count": len(comparison_rows),
        "comparable_count": len(comparable),
        "repeated_attachment_direction_count": len(repeated_attachment),
        "repeated_attachment_rows": [
            {
                "event_id": row["event_id"],
                "behavior_measure": row["behavior_measure"],
                "eligible_word_count": row["eligible_word_count"],
                "median_word_attachment_difference": (
                    row["median_word_attachment_difference"]
                ),
                "attachment_positive_word_fraction": (
                    row["attachment_positive_word_fraction"]
                ),
                "attachment_negative_word_fraction": (
                    row["attachment_negative_word_fraction"]
                ),
            }
            for row in repeated_attachment
        ],
        "batched_forward_count": len({
            row["unit_id"] for row in observations
        }),
        "elapsed_seconds": elapsed_seconds,
        "placement": placement,
        "claim_limits": [
            "Targets were selected without behavior or confirmation data.",
            "Correct/failed differences are associations, not causes.",
            "Discovery references are contextual state directions, not fixed word encodings.",
        ],
    }
    write_json(model_root / "summary.json", summary)
    return summary


def run_model(model_name: str) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    selection_meta = read_json(
        OUT_ROOT / "targeted_behavior_scan" / "selection.json"
    )
    if not selection_meta["selection_used_discovery_only"]:
        raise RuntimeError("target selection is not discovery-only")
    if selection_meta["selection_used_behavior"]:
        raise RuntimeError("target selection used behavior")
    if selection_meta["selection_used_confirmation"]:
        raise RuntimeError("target selection used confirmation")
    selections = [
        row
        for row in read_jsonl(
            OUT_ROOT / "targeted_behavior_scan" / "selection.jsonl"
        )
        if row["model"] == model_name
    ]
    if len(selections) != 8:
        raise RuntimeError(f"expected 8 selections for {model_name}")
    behavior = behavior_by_unit(model_name)
    behavior_selection = read_json(
        OUT_ROOT / "behavior" / model_name / "selection.json"
    )
    prompt_mode = behavior_selection["selected_prompt_mode"]
    units = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"units.{model_name}.{prompt_mode}.jsonl"
    )
    units = sorted(
        units,
        key=lambda row: (
            0 if row["split"] == "discovery" else 1,
            row["word"],
            row["template"],
            row["world"],
        ),
    )
    cases = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    case_by_id = {row["record_id"]: row for row in cases}
    if prereg["protocol_digest"] != behavior_selection["protocol_digest"]:
        raise RuntimeError("protocol/behavior digest mismatch")

    model = tokenizer = device = None
    capture = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        info = get_model_info(model, model_name)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        for row in selections:
            if int(row["depth"]) > int(info.n_layers):
                raise RuntimeError("selection depth exceeds model")
        capture = SelectiveEventCapture(model, layers, selections)
        capture.register(head_count)
        observations = capture_units(
            model=model,
            device=device,
            capture=capture,
            selections=selections,
            units=units,
            case_by_id=case_by_id,
            behavior=behavior,
            model_name=model_name,
        )
        summary = analyze_model(
            model_name,
            selections,
            observations,
            OUT_ROOT / "targeted_behavior_scan",
            time.time() - started,
            placement,
        )
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_model(model)
        del model, tokenizer, device, capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize() -> dict[str, Any]:
    summaries = [
        read_json(
            OUT_ROOT / "targeted_behavior_scan" / model / "summary.json"
        )
        for model in MODELS
    ]
    repeated_by_model = {
        row["model"]: int(row["repeated_attachment_direction_count"])
        for row in summaries
    }
    qwen_glm_repeat = (
        repeated_by_model.get("qwen3", 0) > 0
        and repeated_by_model.get("glm4", 0) > 0
    )
    summary = {
        "schema_version": "phase1017_targeted_behavior_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model_count": len(summaries),
        "selection_count": int(sum(
            row["selection_count"] for row in summaries
        )),
        "batched_forward_count": int(sum(
            row["batched_forward_count"] for row in summaries
        )),
        "confirmation_unit_scalar_count": int(sum(
            row["confirmation_unit_scalar_count"] for row in summaries
        )),
        "comparable_count": int(sum(
            row["comparable_count"] for row in summaries
        )),
        "repeated_attachment_direction_by_model": repeated_by_model,
        "automatic_continuation_assessment": {
            "repeat_full_atlas": False,
            "continue_to_neuron_localization": False,
            "continue_to_causal_closure": False,
            "next_required_test": (
                "pre-register a larger lexical/context holdout for the "
                "behavior-associated branch"
                if qwen_glm_repeat
                else "expand contexts before any behavior-mechanism claim"
            ),
            "reason": (
                "The atlas is stable enough to retain, but association must "
                "replicate before neuron or causal localization."
            ),
        },
        "models": summaries,
    }
    write_json(OUT_ROOT / "targeted_behavior_scan" / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", choices=MODELS)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    elif args.model:
        run_model(args.model)
    else:
        raise SystemExit("provide a model or --finalize")


if __name__ == "__main__":
    main()
