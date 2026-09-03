#!/usr/bin/env python3
"""C120: observation-first controlled comparison-family campaign."""
from __future__ import annotations

import gc
import itertools
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1647_c120_controlled_comparison_observation_campaign"
C119 = RESULT / "phase1643_c119_identifiable_default_override_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1575_c101_dual_arm_contract as breadth_base
import phase1608_c110_exact_field_capture as capture_base

CAMPAIGN = "C120"
FAMILY = "controlled_numeric_comparison"
PARTITIONS = ("discovery", "confirmation", "lockbox")
DIMENSIONS = ("length", "width", "weight")
ROLES = (
    "focus_pre", "record_a_name", "record_a_value", "record_b_name",
    "record_b_value", "focus_post", "query_dimension", "query_focus",
    "query_other", "boundary",
)
EFFECTS = (
    "length_truth", "width_truth", "weight_truth", "shared_truth",
    "gap_truth_interaction", "record_order", "output_vocabulary",
)
STATES, DIM, WIDTH, BATCH = 37, 2560, 256, 8
KEY_STATES = (0, 4, 8, 12, 16, 19, 24, 28, 30, 32, 36)
RAW_STATES = (0, 8, 16, 19, 24, 30, 32, 36)
NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}
STEMS = (
    "azun", "belq", "corv", "daxi", "emur", "feln", "gost", "havi",
    "irun", "jexa", "kord", "lumi", "mawr", "nexi", "orun", "peld",
    "qavi", "relt", "suno", "tavi", "umek", "vorn", "wexa", "yori",
)
SYSTEM = (
    "Use only the explicit integer measurement scores in the local record. Compare only "
    "the requested dimension. Reply using exactly the requested vocabulary."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def topk(values: np.ndarray, k: int) -> list[int]:
    selected = np.argpartition(np.abs(values), -k)[-k:]
    return selected[np.argsort(-np.abs(values[selected]), kind="stable")].astype(int).tolist()


def inventory() -> list[tuple[str, str, str]]:
    return [(f"Meter{stem}A", f"Meter{stem}B", f"{stem}marker") for stem in STEMS]


def scores_for(unit_index: int, dimension: str, truth: int, gap: int) -> dict[str, dict[str, int]]:
    queried = (6, 5) if gap == 1 else (9, 2)
    a_query, b_query = queried if truth == 1 else queried[::-1]
    distractors = [1, 3, 8, 10] if gap == 1 else [1, 4, 7, 10]
    shift = (unit_index + DIMENSIONS.index(dimension)) % 4
    distractors = distractors[shift:] + distractors[:shift]
    other_dimensions = [name for name in DIMENSIONS if name != dimension]
    values = {
        "A": {dimension: a_query},
        "B": {dimension: b_query},
    }
    for index, name in enumerate(other_dimensions):
        values["A"][name] = distractors[index * 2]
        values["B"][name] = distractors[index * 2 + 1]
    return values


def prompt_for(
    unit_index: int,
    values: tuple[str, str, str],
    dimension: str,
    truth: int,
    gap: int,
    surface: int,
    output_format: int,
) -> tuple[str, dict]:
    focus, other, marker = values
    scores = scores_for(unit_index, dimension, truth, gap)
    names = {"A": focus, "B": other}
    record_parts = {}
    for side in ("A", "B"):
        record_parts[side] = (
            f"{names[side]} has length score {NUMBER_WORDS[scores[side]['length']]}, "
            f"width score {NUMBER_WORDS[scores[side]['width']]}, and "
            f"weight score {NUMBER_WORDS[scores[side]['weight']]}"
        )
    order = ("A", "B") if surface == 1 else ("B", "A")
    vocabulary = "yes or no" if output_format == 1 else "true or false"
    prompt = (
        f"Focus before record: {focus}. Measurement record: {record_parts[order[0]]}. "
        f"{record_parts[order[1]]}. Calibration note: {marker} is inactive. "
        f"Focus after record: {focus}. Requested dimension: {dimension}. "
        f"Is {focus}'s {dimension} score greater than {other}'s {dimension} score? "
        f"Reply exactly {vocabulary}."
    )
    return prompt, {
        "focus": focus,
        "other": other,
        "marker": marker,
        "scores": scores,
        "focus_value_word": NUMBER_WORDS[scores["A"][dimension]],
        "other_value_word": NUMBER_WORDS[scores["B"][dimension]],
        "truth_factor": truth,
        "output_labels": ["yes", "no"] if output_format == 1 else ["true", "false"],
    }


def build() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index, values in enumerate(inventory()):
        partition = PARTITIONS[unit_index // 8]
        unit = {
            "unit_id": f"c120-compare-{unit_index:02d}",
            "family": FAMILY,
            "world": "controlled_exact_integer_measurements",
            "partition": partition,
            "values": list(values),
        }
        units.append(unit)
        for dimension, truth, gap, surface, output_format in itertools.product(
            DIMENSIONS, (1, -1), (1, -1), (1, -1), (1, -1)
        ):
            prompt, metadata = prompt_for(
                unit_index, values, dimension, truth, gap, surface, output_format
            )
            cases.append({
                **unit,
                **metadata,
                "case_id": f"c120-{len(cases):04d}",
                "dimension": dimension,
                "gap_factor": gap,
                "surface_factor": surface,
                "output_format": output_format,
                "truth": truth == 1,
                "gold_position": 0 if truth == 1 else 1,
                "prompt": prompt,
            })
    return units, cases


def unique_span(tok, ids: list[int], text: str) -> list[int]:
    spans = breadth_base.find_spans(tok, ids, text)
    if len(spans) != 1:
        raise RuntimeError(("unique span", text, spans))
    return spans[0]


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    candidate_cache: dict[tuple[str, str], list[list[int]]] = {}
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        labels = tuple(row["output_labels"])
        if labels not in candidate_cache:
            encoded = [tok.encode(" " + label, add_special_tokens=False) for label in labels]
            if any(len(value) != 1 for value in encoded):
                raise RuntimeError(("candidate singleton", labels, encoded))
            candidate_cache[labels] = [[int(value[0])] for value in encoded]
        focus_spans = breadth_base.find_spans(tok, ids, row["focus"])
        other_spans = breadth_base.find_spans(tok, ids, row["other"])
        dimension_spans = breadth_base.find_spans(tok, ids, row["dimension"])
        if len(focus_spans) != 4 or len(other_spans) != 2 or len(dimension_spans) < 5:
            raise RuntimeError((row["case_id"], focus_spans, other_spans, dimension_spans))
        focus_pre, record_a_name, focus_post, query_focus = focus_spans
        record_b_name, query_other = other_spans
        query_dimension_candidates = [
            span for span in dimension_spans
            if min(span) > max(focus_post) and max(span) < min(query_focus)
        ]
        if len(query_dimension_candidates) != 1:
            raise RuntimeError(("query dimension", row["case_id"], query_dimension_candidates))
        roles = {
            "focus_pre": focus_pre,
            "record_a_name": record_a_name,
            "record_a_value": unique_span(tok, ids, row["focus_value_word"]),
            "record_b_name": record_b_name,
            "record_b_value": unique_span(tok, ids, row["other_value_word"]),
            "focus_post": focus_post,
            "query_dimension": query_dimension_candidates[0],
            "query_focus": query_focus,
            "query_other": query_other,
            "boundary": [len(ids) - 1],
        }
        occupied = [position for span in roles.values() for position in span]
        if len(occupied) != len(set(occupied)):
            raise RuntimeError(("overlapping roles", row["case_id"], roles))
        compiled.append({
            **row,
            "prompt_ids": ids,
            "candidate_ids": candidate_cache[labels],
            "role_positions": roles,
        })
    return compiled


def zero_models(rows: list[dict]) -> dict[str, float]:
    gold = np.asarray([row["truth_factor"] == 1 for row in rows])
    predictions = {
        "always_positive": np.ones(len(rows), dtype=bool),
        "always_negative": np.zeros(len(rows), dtype=bool),
        "length_only": np.asarray([row["dimension"] == "length" for row in rows]),
        "near_only": np.asarray([row["gap_factor"] == 1 for row in rows]),
        "surface_only": np.asarray([row["surface_factor"] == 1 for row in rows]),
        "format_only": np.asarray([row["output_format"] == 1 for row in rows]),
    }
    result = {name: float(np.mean(value == gold)) for name, value in predictions.items()}
    result["score_comparison_oracle"] = 1.0
    return result


def lookup_manifest() -> tuple[dict[int, list[dict]], dict[tuple[int, str], list[int]]]:
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    by_row: dict[int, list[dict]] = defaultdict(list)
    lookup: dict[tuple[int, str], list[int]] = defaultdict(list)
    for occurrence in manifest:
        row_index = int(occurrence["row_index"])
        by_row[row_index].append(occurrence)
        lookup[(row_index, occurrence["role"])].append(int(occurrence["occurrence_index"]))
    return by_row, lookup


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C120 already exists: {OUT}")
    parent = core.load(C119 / "analysis/paired_behavior_diagnostic.json")
    parent_audit = core.load(C119 / "audit/paired_behavior_diagnostic_audit.json")
    if not parent_audit["all_checks_passed"] or not parent["authorization"].startswith("execute_C120"):
        raise RuntimeError("C120 authorization missing")
    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = compile_rows(tok, cases)
    occurrences = []
    for row_index, row in enumerate(compiled):
        for role in ROLES:
            positions = row["role_positions"][role]
            for subtoken, position in enumerate(positions):
                occurrences.append({
                    "occurrence_index": len(occurrences),
                    "row_index": row_index,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "partition": row["partition"],
                    "dimension": row["dimension"],
                    "truth_factor": row["truth_factor"],
                    "gap_factor": row["gap_factor"],
                    "surface_factor": row["surface_factor"],
                    "output_format": row["output_format"],
                    "role": role,
                    "subtoken": subtoken,
                    "span_length": len(positions),
                    "token_position": int(position),
                    "token_id": int(row["prompt_ids"][position]),
                    "token_text": tok.convert_ids_to_tokens([int(row["prompt_ids"][position])])[0],
                })
    checks = {
        "units": len(units) == 24,
        "cases": len(cases) == len(compiled) == 1152,
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in PARTITIONS},
        "factorial": all(
            sum(
                row["unit_id"] == unit["unit_id"]
                and row["dimension"] == dimension
                and row["truth_factor"] == truth
                and row["gap_factor"] == gap
                and row["surface_factor"] == surface
                and row["output_format"] == output_format
                for row in cases
            ) == 1
            for unit in units
            for dimension, truth, gap, surface, output_format in itertools.product(
                DIMENSIONS, (1, -1), (1, -1), (1, -1), (1, -1)
            )
        ),
        "unique_prompts": len({row["prompt"] for row in cases}) == 1152,
        "truth_balance": all(
            sum(row["truth_factor"] for row in cases if row[field] == value) == 0
            for field, values in (
                ("partition", PARTITIONS), ("dimension", DIMENSIONS),
                ("gap_factor", (1, -1)), ("surface_factor", (1, -1)),
                ("output_format", (1, -1)),
            )
            for value in values
        ),
        "score_uniqueness": all(
            len({value for side in row["scores"].values() for value in side.values()}) == 6
            for row in cases
        ),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "candidates": all(len(candidate) == 1 for row in compiled for candidate in row["candidate_ids"]),
        "zero_models": all(value == 0.5 for key, value in zero_models(cases).items() if key != "score_comparison_oracle"),
        "oracle": zero_models(cases)["score_comparison_oracle"] == 1.0,
        "width": max(len(row["prompt_ids"]) for row in compiled) <= WIDTH,
        "semantic_unique": all(
            row["scores"]["A"][row["dimension"]] != row["scores"]["B"][row["dimension"]]
            for row in cases
        ),
        "machine_naturalness": all(
            phrase in row["prompt"]
            for row in cases
            for phrase in ("Measurement record:", "Requested dimension:", "score greater than")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": 1647,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "object": "explicit integer greater-than comparison across length, width and weight dimensions",
        "model": "Qwen3-4B local CUDA BF16 without quantization",
        "material": {
            "units": 24,
            "cases": 1152,
            "partitions": {name: 8 for name in PARTITIONS},
            "human_naturalness": "not available; controlled machine-audited English only",
        },
        "factors": ["dimension", "truth", "numeric_gap", "record_order", "output_vocabulary"],
        "roles": list(ROLES),
        "states": {"embedding": 0, "hidden_states": [1, 36], "activation_coordinates": 2560},
        "archive": {
            "path": "raw/qwen3_role_subtoken_all_states.uint16.npy",
            "dtype": "uint16_bfloat16_bits",
            "shape": [STATES, len(occurrences), DIM],
        },
        "zero_models": zero_models(cases),
        "behavior_gates": {
            "overall_min": 0.90,
            "each_partition_min": 0.85,
            "each_dimension_min": 0.85,
            "each_gap_min": 0.85,
            "each_output_format_min": 0.85,
        },
        "effects": list(EFFECTS),
        "discovery_rule": {
            "effects": [f"{name}_truth" for name in DIMENSIONS],
            "eligible_roles": list(ROLES),
            "eligible_states": list(range(31)),
            "minimum_half_norm": 1e-5,
            "support_k": 256,
            "score": "split_half_cosine * min(left_norm,right_norm), independently per comparison dimension",
        },
        "validation_gates": {
            "confirmation_lockbox_cosine_min": 0.80,
            "each_to_discovery_cosine_min": 0.70,
            "each_support_overlap_min": 0.35,
        },
        "observation_policy": (
            "capture every registered embedding/HiddenState activation coordinate; nominate each dimension "
            "independently on discovery; reveal confirmation and lockbox only after freezing; report missing "
            "dimensions without imposing shared, orthogonal, low-dimensional, manifold or topological structure"
        ),
        "stop_conditions": {
            "pre_model": "any material, balance, role, candidate, width or zero-model audit fails",
            "behavior": "seal HiddenState analysis if any registered behavioral gate fails",
            "post_reveal": "do not modify object, data, partitions, candidates, effects, thresholds or support",
        },
        "claim_boundary": (
            "controlled synthetic exact-score comparison in one Qwen3; no natural-language universality, "
            "weights, semantic neurons, attention/MLP, endogenous route, shared comparison module, orthogonal "
            "subspace, low-dimensional manifold, topology, algebraic closure or new-mathematics claim"
        ),
        "parent_hashes": {
            "c119_diagnostic": core.sha(C119 / "analysis/paired_behavior_diagnostic.json"),
            "c119_diagnostic_audit": core.sha(C119 / "audit/paired_behavior_diagnostic_audit.json"),
        },
        "material_digest": core.digest([*units, *cases]),
        "occurrences": len(occurrences),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1648_c120_cuda_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {
        "phase": 1647,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "occurrences": len(occurrences),
        "max_width": max(len(row["prompt_ids"]) for row in compiled),
        "zero_models": protocol["zero_models"],
        "authorization": protocol["authorization"],
    }
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def capture() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C120 independent contract audit missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    by_row, lookup = lookup_manifest()
    raw_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (raw_path, logits_path, index_path)):
        raise RuntimeError("C120 raw output already exists")
    field = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.uint16, shape=tuple(protocol["archive"]["shape"])
    )
    candidate_logits = np.lib.format.open_memmap(
        logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2)
    )
    behavior, model, first_rows = [], None, None
    repeat_hidden = repeat_logits = 0.0
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            output, logits, ids, mask, positions, lengths = capture_base.forward(
                model, batch, pad, device, WIDTH
            )
            for state_index, state in enumerate(output.hidden_states):
                if state.dtype != torch.bfloat16 or not bool(torch.isfinite(state).all()):
                    raise RuntimeError((state_index, state.dtype))
                for local in range(len(batch)):
                    row_index = start + local
                    occurrences = by_row[row_index]
                    indices = np.asarray(
                        [int(item["occurrence_index"]) for item in occurrences], dtype=np.int64
                    )
                    token_positions = [int(item["token_position"]) for item in occurrences]
                    field[state_index, indices] = (
                        state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
                    )
            for local, row in enumerate(batch):
                row_index = start + local
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior.append({
                    "row_index": row_index,
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "partition": row["partition"],
                    "dimension": row["dimension"],
                    "truth_factor": row["truth_factor"],
                    "gap_factor": row["gap_factor"],
                    "surface_factor": row["surface_factor"],
                    "output_format": row["output_format"],
                    "gold_position": row["gold_position"],
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "positive_minus_negative": scores[0] - scores[1],
                })
            if start == 0:
                first_rows = batch
            if (start // BATCH + 1) % 12 == 0:
                field.flush()
                candidate_logits.flush()
                print(f"[phase1648] captured {start + len(batch)}/{len(rows)}", flush=True)
            del output, logits, ids, mask, positions
        field.flush()
        candidate_logits.flush()
        output, logits, ids, mask, positions, lengths = capture_base.forward(
            model, first_rows, pad, device, WIDTH
        )
        for state_index, state in enumerate(output.hidden_states):
            for local in range(len(first_rows)):
                occurrences = by_row[local]
                indices = np.asarray(
                    [int(item["occurrence_index"]) for item in occurrences], dtype=np.int64
                )
                token_positions = [int(item["token_position"]) for item in occurrences]
                old = np.asarray(field[state_index, indices])
                new = state[local, token_positions].contiguous().view(torch.uint16).cpu().numpy()
                if not np.array_equal(old, new):
                    repeat_hidden = max(
                        repeat_hidden, float(np.max(np.abs(decode(old) - decode(new))))
                    )
        for local, row in enumerate(first_rows):
            scores = np.asarray(
                [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]],
                dtype=np.float32,
            )
            repeat_logits = max(
                repeat_logits, float(np.max(np.abs(scores - candidate_logits[local])))
            )
        del output, logits, ids, mask, positions
    finally:
        field.flush()
        candidate_logits.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
    core.write_rows(index_path, behavior)
    field = np.load(raw_path, mmap_mode="r")
    candidate_logits = np.load(logits_path, mmap_mode="r")
    causal_prefix = output_previsible = 0.0
    by_unit: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_unit[row["unit_id"]].append(row_index)
    for indices in by_unit.values():
        reference = indices[0]
        for row_index in indices[1:]:
            left = capture_base.role_bits(field, lookup, reference, "focus_pre")
            right = capture_base.role_bits(field, lookup, row_index, "focus_pre")
            if not np.array_equal(left, right):
                causal_prefix = max(
                    causal_prefix, float(np.max(np.abs(decode(left) - decode(right))))
                )
        unit_rows = [rows[index] for index in indices]
        for dimension, truth, gap, surface in itertools.product(
            DIMENSIONS, (1, -1), (1, -1), (1, -1)
        ):
            yes_index = next(
                index for index, row in zip(indices, unit_rows, strict=True)
                if (row["dimension"], row["truth_factor"], row["gap_factor"], row["surface_factor"], row["output_format"])
                == (dimension, truth, gap, surface, 1)
            )
            true_index = next(
                index for index, row in zip(indices, unit_rows, strict=True)
                if (row["dimension"], row["truth_factor"], row["gap_factor"], row["surface_factor"], row["output_format"])
                == (dimension, truth, gap, surface, -1)
            )
            for role in ROLES[:-1]:
                left = capture_base.role_bits(field, lookup, yes_index, role)
                right = capture_base.role_bits(field, lookup, true_index, role)
                if not np.array_equal(left, right):
                    output_previsible = max(
                        output_previsible, float(np.max(np.abs(decode(left) - decode(right))))
                    )

    def accuracy(selected: list[dict]) -> float:
        return float(np.mean([row["correct"] for row in selected]))

    summary = {
        "overall": accuracy(behavior),
        "by_partition": {
            name: accuracy([row for row in behavior if row["partition"] == name])
            for name in PARTITIONS
        },
        "by_dimension": {
            name: accuracy([row for row in behavior if row["dimension"] == name])
            for name in DIMENSIONS
        },
        "by_gap": {
            str(value): accuracy([row for row in behavior if row["gap_factor"] == value])
            for value in (1, -1)
        },
        "by_output_format": {
            str(value): accuracy([row for row in behavior if row["output_format"] == value])
            for value in (1, -1)
        },
        "by_truth": {
            str(value): accuracy([row for row in behavior if row["truth_factor"] == value])
            for value in (1, -1)
        },
    }
    gates = protocol["behavior_gates"]
    gate_checks = {
        "overall": summary["overall"] >= gates["overall_min"],
        "partitions": all(value >= gates["each_partition_min"] for value in summary["by_partition"].values()),
        "dimensions": all(value >= gates["each_dimension_min"] for value in summary["by_dimension"].values()),
        "gaps": all(value >= gates["each_gap_min"] for value in summary["by_gap"].values()),
        "formats": all(value >= gates["each_output_format_min"] for value in summary["by_output_format"].values()),
    }
    checks = {
        "shape": list(field.shape) == protocol["archive"]["shape"],
        "dtype": field.dtype == np.uint16,
        "logits": list(candidate_logits.shape) == [1152, 2] and bool(np.isfinite(candidate_logits).all()),
        "index": len(behavior) == 1152,
        "repeat": repeat_hidden == 0.0 and repeat_logits == 0.0,
        "causal_prefix": causal_prefix == 0.0,
        "output_previsible": output_previsible == 0.0,
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {
        "phase": 1648,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "cuda_capture_complete",
        "shape": list(field.shape),
        "raw_data_bytes": int(field.nbytes),
        "raw_file_bytes": raw_path.stat().st_size,
        "raw_sha256": core.sha(raw_path),
        "logits_sha256": core.sha(logits_path),
        "index_sha256": core.sha(index_path),
        "behavior": summary,
        "behavior_gate_checks": gate_checks,
        "behavior_gate_passed": all(gate_checks.values()),
        "numeric": {
            "repeat_hidden_max_abs": repeat_hidden,
            "repeat_logits_max_abs": repeat_logits,
            "causal_prefix_max_abs": causal_prefix,
            "output_previsible_max_abs": output_previsible,
        },
        "runtime": {"placement": placement, "quantization": quant},
        "checks": checks,
        "authorization": (
            "execute_phase1649_c120_discovery_observation"
            if all(gate_checks.values())
            else "seal_hidden_state_analysis_and_close_C120_behavior_boundary"
        ),
    }
    core.save(OUT / "analysis/capture_summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def effect_coefficients(row: dict) -> dict[str, float]:
    truth = float(row["truth_factor"])
    result = {
        "shared_truth": truth / 48.0,
        "gap_truth_interaction": truth * float(row["gap_factor"]) / 48.0,
        "record_order": float(row["surface_factor"]) / 48.0,
        "output_vocabulary": float(row["output_format"]) / 48.0,
    }
    for dimension in DIMENSIONS:
        result[f"{dimension}_truth"] = truth / 16.0 if row["dimension"] == dimension else 0.0
    return result


def derive_fields(partitions: set[str], path: Path) -> tuple[np.ndarray, list[dict]]:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = [row for row in core.rows(OUT / "material/units.jsonl") if row["partition"] in partitions]
    _, lookup = lookup_manifest()
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    unit_index = {row["unit_id"]: index for index, row in enumerate(units)}
    role_index = {role: index for index, role in enumerate(ROLES)}
    effect_index = {effect: index for index, effect in enumerate(EFFECTS)}
    fields = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(units), len(EFFECTS), len(ROLES), STATES, DIM),
    )
    fields[:] = 0.0
    for state in range(STATES):
        for row_index, row in enumerate(rows):
            if row["partition"] not in partitions:
                continue
            coefficients = effect_coefficients(row)
            u = unit_index[row["unit_id"]]
            for role in ROLES:
                values = np.mean(
                    decode(raw[state, lookup[(row_index, role)]]), axis=0, dtype=np.float32
                )
                for effect, coefficient in coefficients.items():
                    if coefficient:
                        fields[u, effect_index[effect], role_index[role], state] += coefficient * values
        if state % 6 == 0 or state == 36:
            fields.flush()
            print(f"[C120 fields] {sorted(partitions)} state {state}/36", flush=True)
    fields.flush()
    return fields, units


def discover() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    if not core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"]:
        raise RuntimeError("C120 capture audit missing")
    if not capture_report["behavior_gate_passed"]:
        raise RuntimeError("C120 behavior gate did not authorize HiddenState analysis")
    path = OUT / "analysis/discovery_unit_effect_role_state.float32.npy"
    fields, units = derive_fields({"discovery"}, path)
    candidates, nominations = [], []
    minimum = protocol["discovery_rule"]["minimum_half_norm"]
    for dimension in DIMENSIONS:
        effect = f"{dimension}_truth"
        effect_index = EFFECTS.index(effect)
        local = []
        for role_index, role in enumerate(ROLES):
            for state in protocol["discovery_rule"]["eligible_states"]:
                left = np.mean(fields[:4, effect_index, role_index, state], axis=0, dtype=np.float32)
                right = np.mean(fields[4:, effect_index, role_index, state], axis=0, dtype=np.float32)
                left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
                split = cosine(left, right)
                score = split * min(left_norm, right_norm) if min(left_norm, right_norm) >= minimum else None
                row = {
                    "dimension": dimension,
                    "effect": effect,
                    "role": role,
                    "state": int(state),
                    "split_half_cosine": split,
                    "left_norm": left_norm,
                    "right_norm": right_norm,
                    "score": score,
                }
                candidates.append(row)
                local.append(row)
        eligible = [row for row in local if row["score"] is not None]
        winner = sorted(
            eligible,
            key=lambda row: (-row["score"], -row["split_half_cosine"], row["state"], ROLES.index(row["role"])),
        )[0]
        mean = np.mean(
            fields[:, effect_index, ROLES.index(winner["role"]), winner["state"]],
            axis=0,
            dtype=np.float32,
        )
        nominations.append({
            **winner,
            "support_k": 256,
            "support": topk(mean, 256),
            "field_norm": float(np.linalg.norm(mean)),
            "discovery_units": [row["unit_id"] for row in units],
        })
    table_path = OUT / "analysis/discovery_candidate_table.jsonl"
    nomination_path = OUT / "protocol/frozen_dimension_nominations.json"
    core.write_rows(table_path, candidates)
    core.save(nomination_path, {
        "created_at_utc": now(),
        "field_sha256": core.sha(path),
        "candidate_table_sha256": core.sha(table_path),
        "nominations": nominations,
    })
    report = {
        "phase": 1649,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "three_dimension_candidates_frozen",
        "nominations": [
            {key: value for key, value in row.items() if key not in ("support", "discovery_units")}
            for row in nominations
        ],
        "checks": {
            "units": len(units) == 8,
            "shape": list(fields.shape) == [8, 7, 10, 37, 2560],
            "candidates": len(candidates) == 3 * 10 * 31,
            "nominations": len(nominations) == 3,
            "supports": all(len(set(row["support"])) == 256 for row in nominations),
        },
        "field_sha256": core.sha(path),
        "nomination_sha256": core.sha(nomination_path),
        "authorization": "execute_phase1650_c120_confirmation_lockbox_validation",
    }
    if not all(report["checks"].values()):
        raise RuntimeError(report)
    core.save(OUT / "analysis/discovery_freeze.json", report)
    print(json.dumps(report, indent=2))


def validate() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_discovery_audit.json")["all_checks_passed"]:
        raise RuntimeError("C120 discovery audit missing")
    nomination_payload = core.load(OUT / "protocol/frozen_dimension_nominations.json")
    nominations = nomination_payload["nominations"]
    path = OUT / "analysis/validation_unit_effect_role_state.float32.npy"
    fields, units = derive_fields({"confirmation", "lockbox"}, path)
    discovery = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    gates = protocol["validation_gates"]
    dimension_metrics = []
    for nomination in nominations:
        effect_index = EFFECTS.index(nomination["effect"])
        role_index = ROLES.index(nomination["role"])
        state = int(nomination["state"])
        d = np.mean(discovery[:, effect_index, role_index, state], axis=0, dtype=np.float32)
        c = np.mean(fields[:8, effect_index, role_index, state], axis=0, dtype=np.float32)
        l = np.mean(fields[8:, effect_index, role_index, state], axis=0, dtype=np.float32)
        support = set(nomination["support"])
        metrics = {
            "dimension": nomination["dimension"],
            "effect": nomination["effect"],
            "role": nomination["role"],
            "state": state,
            "confirmation_lockbox_cosine": cosine(c, l),
            "confirmation_to_discovery_cosine": cosine(c, d),
            "lockbox_to_discovery_cosine": cosine(l, d),
            "confirmation_support_overlap": len(set(topk(c, 256)) & support) / 256,
            "lockbox_support_overlap": len(set(topk(l, 256)) & support) / 256,
            "norms": {
                "discovery": float(np.linalg.norm(d)),
                "confirmation": float(np.linalg.norm(c)),
                "lockbox": float(np.linalg.norm(l)),
            },
        }
        metrics["checks"] = {
            "confirmation_lockbox": metrics["confirmation_lockbox_cosine"] >= gates["confirmation_lockbox_cosine_min"],
            "to_discovery": min(metrics["confirmation_to_discovery_cosine"], metrics["lockbox_to_discovery_cosine"]) >= gates["each_to_discovery_cosine_min"],
            "support_overlap": min(metrics["confirmation_support_overlap"], metrics["lockbox_support_overlap"]) >= gates["each_support_overlap_min"],
        }
        metrics["passed"] = all(metrics["checks"].values())
        dimension_metrics.append(metrics)

    timing_rows = []
    for effect in (f"{name}_truth" for name in DIMENSIONS):
        effect_index = EFFECTS.index(effect)
        for role_index, role in enumerate(ROLES):
            for state in range(STATES):
                c = np.mean(fields[:8, effect_index, role_index, state], axis=0, dtype=np.float32)
                l = np.mean(fields[8:, effect_index, role_index, state], axis=0, dtype=np.float32)
                timing_rows.append({
                    "effect": effect,
                    "role": role,
                    "state": state,
                    "confirmation_lockbox_cosine": cosine(c, l),
                    "confirmation_norm": float(np.linalg.norm(c)),
                    "lockbox_norm": float(np.linalg.norm(l)),
                })
    timing_path = OUT / "analysis/three_dimension_role_state_timing_atlas.jsonl"
    core.write_rows(timing_path, timing_rows)

    shared_geometry = {}
    for partition, values in (
        ("discovery", discovery), ("confirmation", fields[:8]), ("lockbox", fields[8:])
    ):
        means = {
            dimension: np.mean(
                values[:, EFFECTS.index(f"{dimension}_truth"), ROLES.index("boundary"), 30],
                axis=0,
                dtype=np.float32,
            )
            for dimension in DIMENSIONS
        }
        shared_geometry[partition] = {
            f"{left}_{right}": {
                "cosine": cosine(means[left], means[right]),
                "top256_overlap": len(set(topk(means[left], 256)) & set(topk(means[right], 256))) / 256,
            }
            for index, left in enumerate(DIMENSIONS)
            for right in DIMENSIONS[index + 1:]
        }
    report = {
        "phase": 1650,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "confirmation_lockbox_observation_complete",
        "dimension_metrics": dimension_metrics,
        "passed_dimensions": sum(row["passed"] for row in dimension_metrics),
        "shared_boundary_state30_geometry_descriptive": shared_geometry,
        "timing_rows": len(timing_rows),
        "checks": {
            "units": len(units) == 16,
            "shape": list(fields.shape) == [16, 7, 10, 37, 2560],
            "metrics": len(dimension_metrics) == 3,
            "timing": len(timing_rows) == 3 * 10 * 37,
            "finite": all(
                math.isfinite(row["confirmation_lockbox_cosine"])
                and math.isfinite(row["confirmation_norm"])
                and math.isfinite(row["lockbox_norm"])
                for row in timing_rows
            ),
        },
        "field_sha256": core.sha(path),
        "timing_sha256": core.sha(timing_path),
        "authorization": "execute_phase1651_c120_synthesis_heatmap_and_closure",
    }
    if not all(report["checks"].values()):
        raise RuntimeError(report)
    core.save(OUT / "analysis/validation_adjudication.json", report)
    print(json.dumps(report, indent=2))


def synthesize() -> None:
    if not core.load(OUT / "audit/independent_validation_audit.json")["all_checks_passed"]:
        raise RuntimeError("C120 validation audit missing")
    protocol = core.load(OUT / "protocol/preregistration.json")
    capture_report = core.load(OUT / "analysis/capture_summary.json")
    discovery_report = core.load(OUT / "analysis/discovery_freeze.json")
    validation = core.load(OUT / "analysis/validation_adjudication.json")
    nominations = core.load(OUT / "protocol/frozen_dimension_nominations.json")["nominations"]
    discovery = np.load(OUT / "analysis/discovery_unit_effect_role_state.float32.npy", mmap_mode="r")
    heldout = np.load(OUT / "analysis/validation_unit_effect_role_state.float32.npy", mmap_mode="r")
    means = {
        "discovery": np.mean(discovery, axis=0, dtype=np.float32),
        "confirmation": np.mean(heldout[:8], axis=0, dtype=np.float32),
        "lockbox": np.mean(heldout[8:], axis=0, dtype=np.float32),
    }
    nomination_by_effect = {row["effect"]: row for row in nominations}
    effect_rows, seen = [], set()
    for partition, mean in means.items():
        for effect in (f"{name}_truth" for name in DIMENSIONS):
            nomination = nomination_by_effect[effect]
            for role in ROLES:
                states = set(KEY_STATES)
                if role == nomination["role"]:
                    states.update(range(STATES))
                for state in sorted(states):
                    key = (partition, effect, role, state)
                    if key in seen:
                        continue
                    seen.add(key)
                    effect_rows.append({
                        "dataset": "C120",
                        "family": FAMILY,
                        "partition": partition,
                        "dimension": effect.removesuffix("_truth"),
                        "role": role,
                        "state": state,
                        "state_kind": "embedding" if state == 0 else "hidden_state",
                        "effect": effect,
                        "values": np.asarray(mean[EFFECTS.index(effect), ROLES.index(role), state], dtype=np.float32).tolist(),
                    })
        for effect in ("shared_truth", "gap_truth_interaction", "record_order", "output_vocabulary"):
            for state in (0, 16, 24, 30, 36):
                effect_rows.append({
                    "dataset": "C120",
                    "family": FAMILY,
                    "partition": partition,
                    "dimension": "aggregate",
                    "role": "boundary",
                    "state": state,
                    "state_kind": "embedding" if state == 0 else "hidden_state",
                    "effect": effect,
                    "values": np.asarray(mean[EFFECTS.index(effect), ROLES.index("boundary"), state], dtype=np.float32).tolist(),
                })

    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    manifest = core.rows(OUT / "protocol/role_occurrence_manifest.jsonl")
    raw = np.load(OUT / protocol["archive"]["path"], mmap_mode="r")
    occurrence_lookup: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for occurrence in manifest:
        occurrence_lookup[(int(occurrence["row_index"]), occurrence["role"])].append(occurrence)
    raw_rows = []
    for partition in PARTITIONS:
        row_index = next(
            index for index, row in enumerate(compiled)
            if row["partition"] == partition
            and row["dimension"] == "length"
            and row["truth_factor"] == 1
            and row["gap_factor"] == 1
            and row["surface_factor"] == 1
            and row["output_format"] == 1
        )
        row = compiled[row_index]
        for role in ROLES:
            occurrences = occurrence_lookup[(row_index, role)]
            for state in RAW_STATES:
                values = np.mean(
                    np.stack([decode(raw[state, int(item["occurrence_index"])]) for item in occurrences]),
                    axis=0,
                    dtype=np.float32,
                )
                raw_rows.append({
                    "dataset": "C120",
                    "case_id": row["case_id"],
                    "family": FAMILY,
                    "partition": partition,
                    "dimension": row["dimension"],
                    "truth_factor": row["truth_factor"],
                    "gap_factor": row["gap_factor"],
                    "surface_factor": row["surface_factor"],
                    "output_format": row["output_format"],
                    "role": role,
                    "subtoken": "role_mean",
                    "token_position": [int(item["token_position"]) for item in occurrences],
                    "token_id": [int(item["token_id"]) for item in occurrences],
                    "token_text": "|".join(item["token_text"] for item in occurrences),
                    "state": state,
                    "state_kind": "embedding" if state == 0 else "hidden_state",
                    "values": np.asarray(values, dtype=np.float32).tolist(),
                })

    payload = core.load(PUBLIC)
    payload["effect_rows"] = [row for row in payload["effect_rows"] if row.get("dataset") != "C120"] + effect_rows
    payload["raw_rows"] = [row for row in payload["raw_rows"] if row.get("dataset") != "C120"] + raw_rows
    payload["support_rows"] = [row for row in payload.get("support_rows", []) if row.get("dataset") != "C120"]
    for nomination in nominations:
        indicator = np.zeros(DIM, dtype=np.float32)
        indicator[np.asarray(nomination["support"], dtype=np.int64)] = 1.0
        payload["support_rows"].append({
            "dataset": "C120",
            "name": f"C120 {nomination['dimension']} discovery support",
            "family": FAMILY,
            "dimension": nomination["dimension"],
            "role": nomination["role"],
            "state": nomination["state"],
            "k": 256,
            "values": indicator.tolist(),
        })
    payload["scale"] = {
        "effect_symmetric_abs_q99": float(np.quantile(np.concatenate([
            np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["effect_rows"]
        ]), 0.99)),
        "raw_symmetric_abs_q99": float(np.quantile(np.concatenate([
            np.abs(np.asarray(row["values"], dtype=np.float32)) for row in payload["raw_rows"]
        ]), 0.99)),
    }
    payload.update({
        "phase": 1651,
        "campaign": "C109-C120",
        "title": "C109-C120 Relation-Role-State Activation Atlas",
        "c120_batch": {
            "contract": {
                "object": protocol["object"],
                "factors": protocol["factors"],
                "behavior_gates": protocol["behavior_gates"],
            },
            "capture": capture_report,
            "discovery": discovery_report,
            "nominations": [
                {key: value for key, value in row.items() if key not in ("support", "discovery_units")}
                for row in nominations
            ],
            "validation": validation,
        },
        "claim_boundary": (
            "C120 adds all 2560 activation coordinates for registered embedding/HiddenState observations "
            "of exact length, width and weight score comparisons. It reports independently discovered "
            "dimension-conditioned response fields and their held-out stability. It does not identify "
            "weights, attention/MLP, semantic neurons, endogenous routes, a shared comparison module, "
            "orthogonal subspaces, manifolds, topology, algebraic closure or new mathematics."
        ),
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c120_relation_role_state_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)
    closure = {
        "phase": 1651,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "controlled_comparison_observation_campaign_complete",
        "headline": {
            "behavior": capture_report["behavior"],
            "nominations": discovery_report["nominations"],
            "validation": validation["dimension_metrics"],
            "passed_dimensions": validation["passed_dimensions"],
            "shared_boundary_state30_geometry_descriptive": validation["shared_boundary_state30_geometry_descriptive"],
        },
        "strict_conclusion": (
            "C120 tests whether full-coordinate truth-response fields for three explicit comparison dimensions "
            "can be discovered and repeated in held-out lexical units. Passing fields are stable registered "
            "response objects, not proof of a shared comparator, semantic coordinates or a closed mechanism."
        ),
        "new_puzzles": {
            "K312-OBS": "registered three-dimension comparison truth fields and held-out stability result",
            "K313-OBS": "descriptive cross-dimension geometry at fixed boundary@state30 without a shared-subspace claim",
        },
        "theory_update": (
            "RDC retains the separation H for natural states, R for researcher-defined contrasts and Gamma "
            "for interventions. C120 adds dimension-indexed comparison response fields R_{dimension,truth,role,state}; "
            "no common object is asserted unless future independent replacement or perturbation tests support it."
        ),
        "unified_formula": (
            "H_s(x)=Phi_{theta,<s}(E(x),kappa); "
            "R_{m,r,s}=E_x[t(x) 1[m(x)=m] H_{r,s}(x)]; "
            "Gamma=(s,C,S,V); y_I=O(Phi_{theta,>=s}(I_Gamma(H_s)))"
        ),
        "problems": [
            "controlled exact-score English and one Qwen3",
            "machine naturalness only; no independent human blind review",
            "comparison words and integer scores remain explicit",
            "discovery searches role/state separately for each dimension and can prefer late decision states",
            "top256 support is descriptive and not minimal, necessary or causally sufficient",
            "cross-dimension cosine at boundary@state30 does not identify a common comparator",
            "no natural adjective comparisons, cross-model test, attention/MLP or weight analysis",
        ],
        "heatmap": {
            "path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"),
            "bytes": PUBLIC.stat().st_size,
            "sha256": core.sha(PUBLIC),
            "activation_coordinates": 2560,
            "includes_embedding": True,
            "c120_effect_rows": len(effect_rows),
            "c120_raw_rows": len(raw_rows),
        },
        "claim_boundary": payload["claim_boundary"],
        "next_authorization": (
            "C121 may continue the same comparison-family target with fresh adjective paraphrases and "
            "discovery-frozen coordinate perturbations only for C120 dimensions that passed all held-out gates"
        ),
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "validation": len(validation["dimension_metrics"]) == 3,
        "effect_rows": len(effect_rows) > 900,
        "raw_rows": len(raw_rows) == 3 * len(ROLES) * len(RAW_STATES),
        "coordinates": all(len(row["values"]) == 2560 for row in [*effect_rows, *raw_rows]),
        "embedding": any(row["state"] == 0 for row in effect_rows) and any(row["state"] == 0 for row in raw_rows),
        "asset": core.sha(canonical) == core.sha(PUBLIC),
        "batch": "c120_batch" in payload,
        "boundary": "does not identify weights" in payload["claim_boundary"] and "attention/MLP" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    audit = {
        "phase": 1651,
        "campaign": CAMPAIGN,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "asset_sha256": core.sha(PUBLIC),
        "authorization": "run_independent_closure_and_client_audits_then_append_memo",
    }
    core.save(OUT / "audit/internal_closure_audit.json", audit)
    print(json.dumps({"headline": closure["headline"], "checks": checks, "next": closure["next_authorization"]}, indent=2))


STAGES = {
    "contract": contract,
    "capture": capture,
    "discover": discover,
    "validate": validate,
    "synthesize": synthesize,
}


if __name__ == "__main__":
    STAGES[sys.argv[1]]()
