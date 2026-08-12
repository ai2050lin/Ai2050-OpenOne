#!/usr/bin/env python3
"""Post-hoc GLM4 audit of Phase1009 causal effects across output prefixes.

This audit was motivated by the positive cross-family causal replication. It
cannot upgrade that preregistered result because the alternative prefixes were
chosen afterward. It can only diagnose whether the exact word "Answer" is
necessary for the observed local effect.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase548_shared_attention_compute_protocol import render_chat
from phase1009_crossfamily_heldout_causal_replication import (
    PHASE1008_ROOT,
    SOURCE_OPERATION,
    run_batch,
    summarize,
)
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODEL = "glm4"
OPERATIONS = ("F", "Q")
SURFACES = {
    "answer": {
        "instruction_label": "Answer",
        "assistant_prefix": "\nAnswer: ",
    },
    "result": {
        "instruction_label": "Result",
        "assistant_prefix": "\nResult: ",
    },
    "choice": {
        "instruction_label": "Choice",
        "assistant_prefix": "\nChoice: ",
    },
}
FROZEN_INSTRUCTION = (
    "Reply exactly as Answer: NAME done. Replace NAME with one listed "
    "person. Add nothing else."
)


def derived_case(
    tokenizer,
    case: dict[str, Any],
    surface_name: str,
) -> dict[str, Any]:
    surface = SURFACES[surface_name]
    instruction = (
        f"Reply exactly as {surface['instruction_label']}: NAME done. "
        "Replace NAME with one listed person. Add nothing else."
    )
    raw_prompt = case["raw_prompt"].replace(FROZEN_INSTRUCTION, instruction)
    if raw_prompt == case["raw_prompt"] and surface_name != "answer":
        raise RuntimeError("output instruction replacement failed")
    rendered = render_chat(tokenizer, MODEL, raw_prompt)
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    candidate_answers = {
        name: [
            int(value)
            for value in tokenizer.encode(
                f"{surface['assistant_prefix']}{name} done",
                add_special_tokens=False,
            )
        ]
        for name in case["candidate_names"]
    }
    widths = {len(values) for values in candidate_answers.values()}
    if len(widths) != 1:
        raise RuntimeError(
            f"{surface_name}/{case['record_id']}: answer width drift"
        )
    width = next(iter(widths))
    varying = [
        index
        for index in range(width)
        if len({
            values[index] for values in candidate_answers.values()
        }) > 1
    ]
    if len(varying) != 1:
        raise RuntimeError(
            f"{surface_name}/{case['record_id']}: semantic step {varying}"
        )
    semantic_step = varying[0]
    prefixes = {
        tuple(values[:semantic_step])
        for values in candidate_answers.values()
    }
    suffixes = {
        tuple(values[semantic_step + 1:])
        for values in candidate_answers.values()
    }
    if len(prefixes) != 1 or len(suffixes) != 1:
        raise RuntimeError(
            f"{surface_name}/{case['record_id']}: answer framing drift"
        )
    prefix_ids = list(next(iter(prefixes)))
    answer_text = (
        f"{surface['assistant_prefix']}{case['gold']} done"
    )
    answer_ids = candidate_answers[case["gold"]]
    if (
        answer_ids[:len(prefix_ids)] != prefix_ids
        or semantic_step != len(prefix_ids)
        or len(next(iter(suffixes))) != 1
    ):
        raise RuntimeError(
            f"{surface_name}/{case['record_id']}: answer framing drift"
        )
    extended = [
        int(value)
        for value in tokenizer.encode(
            rendered + answer_text,
            add_special_tokens=False,
        )
    ]
    if extended != input_ids + answer_ids:
        raise RuntimeError(
            f"{surface_name}/{case['record_id']}: answer boundary drift"
        )
    row = dict(case)
    row.update({
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "answer_text": answer_text,
        "answer_token_ids": answer_ids,
        "semantic_step": semantic_step,
        "function_step": semantic_step + 1,
        "protocol_prefix_ids": prefix_ids,
        "scan_role_positions": {},
        "output_surface": surface_name,
    })
    return row


def candidate_prediction(
    model,
    device,
    tokenizer,
    cases: list[dict[str, Any]],
) -> list[int]:
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"behavior width drift {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] + case["protocol_prefix_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
    logits = output.logits[:, -1, :]
    predictions = []
    for index, case in enumerate(cases):
        candidate_ids = [
            int(value)
            for value in case["candidate_name_ids"].values()
        ]
        local = logits[index, candidate_ids].argmax()
        predictions.append(candidate_ids[int(local.item())])
    del output, logits, input_ids, attention
    return predictions


def main() -> None:
    atlas = read_json(OUT_ROOT / "final" / "summary.json")
    if not atlas["automatic_next_step_rule"]["eligible"]:
        raise RuntimeError("atlas did not authorize causal work")
    original_cases = read_jsonl(
        OUT_ROOT / "protocol" / MODEL / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / MODEL / "units.jsonl"
    )
    case_by_id = {
        case["record_id"]: case for case in original_cases
    }
    selection_bundle = read_json(
        PHASE1008_ROOT
        / "refinement_final"
        / MODEL
        / "causal_selection.json"
    )
    selections = {
        selection["operation"]: selection
        for selection in selection_bundle["selections"]
    }
    started = time.time()
    model = tokenizer = device = None
    behavior_rows = []
    causal_rows = []
    try:
        model, tokenizer, device = load_model(MODEL, use_8bit=True)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        surface_case_by_id = {
            surface: {
                record_id: derived_case(tokenizer, case, surface)
                for record_id, case in case_by_id.items()
            }
            for surface in SURFACES
        }
        for surface in SURFACES:
            if surface == "answer":
                for record_id, case in case_by_id.items():
                    derived = surface_case_by_id[surface][record_id]
                    if derived["input_ids"] != case["input_ids"]:
                        raise RuntimeError(
                            f"{record_id}: original surface input drift"
                        )
            for family in FAMILIES:
                for operation in OPERATIONS:
                    selected_units = [
                        unit for unit in units
                        if unit["split"] == "confirmation"
                        and unit["family"] == family
                    ]
                    state_cases = []
                    descriptors = []
                    for unit in selected_units:
                        for state in ("base", operation):
                            case = surface_case_by_id[surface][
                                unit["case_ids"][state]
                            ]
                            state_cases.append(case)
                            descriptors.append((unit, state, case))
                    grouped: dict[int, list[int]] = defaultdict(list)
                    for index, case in enumerate(state_cases):
                        grouped[
                            len(case["input_ids"])
                            + len(case["protocol_prefix_ids"])
                        ].append(index)
                    prediction_by_index = {}
                    for indices in grouped.values():
                        batch = [state_cases[index] for index in indices]
                        predictions = candidate_prediction(
                            model,
                            device,
                            tokenizer,
                            batch,
                        )
                        for index, prediction in zip(indices, predictions):
                            prediction_by_index[index] = prediction
                    qualified_units = set()
                    per_unit_hits: dict[str, dict[str, bool]] = defaultdict(dict)
                    for index, (unit, state, case) in enumerate(descriptors):
                        gold_id = int(
                            case["answer_token_ids"][
                                int(case["semantic_step"])
                            ]
                        )
                        hit = prediction_by_index[index] == gold_id
                        per_unit_hits[unit["unit_id"]][state] = hit
                        behavior_rows.append({
                            "schema_version": (
                                "phase1009_output_prefix_behavior.v1"
                            ),
                            "phase": PHASE,
                            "model": MODEL,
                            "surface": surface,
                            "family": family,
                            "operation": operation,
                            "unit_id": unit["unit_id"],
                            "state": state,
                            "candidate_panel_hit": bool(hit),
                        })
                    for unit_id, hits in per_unit_hits.items():
                        if hits.get("base") and hits.get(operation):
                            qualified_units.add(unit_id)
                    items = []
                    for unit in selected_units:
                        if unit["unit_id"] not in qualified_units:
                            continue
                        items.append({
                            "unit": unit,
                            "base": surface_case_by_id[surface][
                                unit["case_ids"]["base"]
                            ],
                            "variant": surface_case_by_id[surface][
                                unit["case_ids"][operation]
                            ],
                        })
                    grouped_items: dict[
                        tuple[int, int, int],
                        list[dict[str, Any]],
                    ] = defaultdict(list)
                    for item in items:
                        base = item["base"]
                        variant = item["variant"]
                        grouped_items[(
                            int(item["unit"]["template"]),
                            len(base["input_ids"])
                            + len(base["protocol_prefix_ids"]),
                            len(variant["input_ids"])
                            + len(variant["protocol_prefix_ids"]),
                        )].append(item)
                    selection = selections[SOURCE_OPERATION[operation]]
                    layer = layers[int(selection["layer"]) - 1]
                    cell_rows = []
                    for batch in grouped_items.values():
                        cell_rows.extend(run_batch(
                            model=model,
                            layer=layer,
                            device=device,
                            head_count=head_count,
                            selection=selection,
                            family=family,
                            operation=operation,
                            items=batch,
                        ))
                    for row in cell_rows:
                        row["output_surface"] = surface
                        row["posthoc_specificity_audit"] = True
                    causal_rows.extend(cell_rows)
                    print(
                        f"[prefix-audit] {surface}/{family}/{operation} "
                        f"qualified={len(cell_rows)}",
                        flush=True,
                    )
        cell_summaries = []
        for surface in SURFACES:
            for family in FAMILIES:
                for operation in OPERATIONS:
                    rows = [
                        row for row in causal_rows
                        if row["output_surface"] == surface
                        and row["family"] == family
                        and row["operation"] == operation
                    ]
                    if len(rows) < 8:
                        cell_summaries.append({
                            "surface": surface,
                            "family": family,
                            "operation": operation,
                            "n": len(rows),
                            "status": "underfilled",
                            "localized_directional_contribution": False,
                        })
                        continue
                    row = summarize(MODEL, family, operation, rows)
                    row["surface"] = surface
                    row["status"] = "complete"
                    row["posthoc_specificity_audit"] = True
                    cell_summaries.append(row)
        no_op_pass = all(
            row["noop_max_logit_error"] <= 1e-5
            for row in causal_rows
        )
        if not no_op_pass:
            raise RuntimeError("output-prefix no-op audit failed")
        surface_summaries = {}
        for surface in SURFACES:
            cells = [
                row for row in cell_summaries
                if row["surface"] == surface
            ]
            behavior = [
                row for row in behavior_rows
                if row["surface"] == surface
            ]
            surface_summaries[surface] = {
                "candidate_panel_state_rate": float(np.mean([
                    row["candidate_panel_hit"] for row in behavior
                ])),
                "complete_cell_count": int(sum(
                    row["status"] == "complete" for row in cells
                )),
                "positive_cell_count": int(sum(
                    row["localized_directional_contribution"]
                    for row in cells
                )),
            }
        result = {
            "schema_version": "phase1009_output_prefix_causal_audit.v1",
            "phase": PHASE,
            "model": MODEL,
            "surfaces": list(SURFACES),
            "surface_summaries": surface_summaries,
            "cell_summaries": cell_summaries,
            "no_op_audit_pass": no_op_pass,
            "selection_used_phase1009_or_surface_data": False,
            "posthoc_specificity_audit": True,
            "interpretation": (
                "Persistence under Result/Choice means the literal prefix "
                "word Answer is not necessary. All surfaces still emit a "
                "person name, so answer-type and candidate-vocabulary "
                "confounds remain."
            ),
            "elapsed_seconds": time.time() - started,
        }
        root = OUT_ROOT / "causal_replication" / "glm4_prefix_audit"
        write_jsonl(root / "behavior_rows.jsonl", behavior_rows)
        write_jsonl(root / "causal_rows.jsonl", causal_rows)
        write_jsonl(root / "cell_summaries.jsonl", cell_summaries)
        write_json(root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
