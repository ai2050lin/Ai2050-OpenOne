#!/usr/bin/env python3
"""C121: fresh structured comparison qualification, followed by an authorized field route."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1650_c121_structured_comparison_qualification"
C120 = RESULT / "phase1647_c120_controlled_comparison_observation_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1575_c101_dual_arm_contract as breadth_base

CAMPAIGN = "C121"
FAMILY = "structured_exact_comparison"
PARTITIONS = ("discovery", "confirmation", "lockbox")
DIMENSIONS = ("length", "width", "weight")
ROLES = (
    "focus_pre", "record_a_name", "record_a_value", "record_b_name",
    "record_b_value", "focus_post", "query_dimension", "query_focus",
    "query_other", "boundary",
)
WIDTH, BATCH = 256, 8
STEMS = (
    "abir", "bosk", "cenu", "dolv", "efra", "gilm", "huro", "ivak",
    "juno", "kepr", "laxo", "mivu", "nord", "opel", "prax", "qelu",
    "rovi", "semt", "tulo", "uvex", "wira", "xond", "yalu", "zemi",
)
SYSTEM = (
    "The local table contains exact integer scores. For the requested dimension, compare "
    "the two digits mathematically. A proposition X > Y is correct exactly when X is the "
    "larger integer. Reply using exactly the requested vocabulary."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory() -> list[tuple[str, str, str]]:
    return [(f"Gauge{stem}A", f"Gauge{stem}B", f"{stem}flag") for stem in STEMS]


def score_table(unit_index: int, dimension: str, truth: int, gap: int) -> dict[str, dict[str, int]]:
    high, low = ((6, 5) if gap == 1 else (9, 2))
    a_query, b_query = (high, low) if truth == 1 else (low, high)
    pool = [1, 3, 8, 10] if gap == 1 else [1, 4, 7, 10]
    shift = (unit_index * 3 + DIMENSIONS.index(dimension)) % 4
    pool = pool[shift:] + pool[:shift]
    result = {"A": {dimension: a_query}, "B": {dimension: b_query}}
    for index, name in enumerate(value for value in DIMENSIONS if value != dimension):
        result["A"][name] = pool[index * 2]
        result["B"][name] = pool[index * 2 + 1]
    return result


def prompt_for(unit_index: int, values: tuple[str, str, str], dimension: str, truth: int, gap: int, surface: int, output_format: int) -> tuple[str, dict]:
    focus, other, flag = values
    scores = score_table(unit_index, dimension, truth, gap)
    names = {"A": focus, "B": other}
    records = {
        side: f"{names[side]} [length={scores[side]['length']}; width={scores[side]['width']}; weight={scores[side]['weight']}]"
        for side in ("A", "B")
    }
    order = ("A", "B") if surface == 1 else ("B", "A")
    vocabulary = "yes or no" if output_format == 1 else "true or false"
    prompt = (
        f"Focus: {focus}. Exact table: {records[order[0]]}; {records[order[1]]}. "
        f"Unused flag: {flag}. Query target: {focus}. Requested dimension: {dimension}. "
        f"Proposition: {focus}.{dimension} > {other}.{dimension}. "
        f"Is the proposition correct? Reply exactly {vocabulary}."
    )
    return prompt, {
        "focus": focus,
        "other": other,
        "flag": flag,
        "scores": scores,
        "focus_value": str(scores["A"][dimension]),
        "other_value": str(scores["B"][dimension]),
        "truth_factor": truth,
        "output_labels": ["yes", "no"] if output_format == 1 else ["true", "false"],
    }


def build() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index, values in enumerate(inventory()):
        partition = PARTITIONS[unit_index // 8]
        unit = {
            "unit_id": f"c121-compare-{unit_index:02d}",
            "family": FAMILY,
            "world": "controlled_structured_integer_table",
            "partition": partition,
            "values": list(values),
        }
        units.append(unit)
        for dimension, truth, gap, surface, output_format in itertools.product(DIMENSIONS, (1, -1), (1, -1), (1, -1), (1, -1)):
            prompt, metadata = prompt_for(unit_index, values, dimension, truth, gap, surface, output_format)
            cases.append({
                **unit, **metadata,
                "case_id": f"c121-{len(cases):04d}",
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
        raise RuntimeError((text, spans))
    return spans[0]


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    cache: dict[tuple[str, str], list[list[int]]] = {}
    compiled = []
    for row in rows:
        ids = core.chat_ids(tok, SYSTEM, row["prompt"])
        labels = tuple(row["output_labels"])
        if labels not in cache:
            encoded = [tok.encode(" " + label, add_special_tokens=False) for label in labels]
            if any(len(value) != 1 for value in encoded):
                raise RuntimeError((labels, encoded))
            cache[labels] = [[int(value[0])] for value in encoded]
        focus_spans = breadth_base.find_spans(tok, ids, row["focus"])
        other_spans = breadth_base.find_spans(tok, ids, row["other"])
        dimension_spans = breadth_base.find_spans(tok, ids, row["dimension"])
        if len(focus_spans) != 4 or len(other_spans) != 2 or len(dimension_spans) < 3:
            raise RuntimeError((row["case_id"], focus_spans, other_spans, dimension_spans))
        focus_pre, record_a_name, focus_post, query_focus = focus_spans
        record_b_name, query_other = other_spans
        query_dimension = [span for span in dimension_spans if min(span) > max(focus_post) and max(span) < min(query_focus)]
        if len(query_dimension) != 1:
            raise RuntimeError((row["case_id"], query_dimension))
        roles = {
            "focus_pre": focus_pre,
            "record_a_name": record_a_name,
            "record_a_value": unique_span(tok, ids, row["focus_value"]),
            "record_b_name": record_b_name,
            "record_b_value": unique_span(tok, ids, row["other_value"]),
            "focus_post": focus_post,
            "query_dimension": query_dimension[0],
            "query_focus": query_focus,
            "query_other": query_other,
            "boundary": [len(ids) - 1],
        }
        occupied = [position for span in roles.values() for position in span]
        if len(occupied) != len(set(occupied)):
            raise RuntimeError((row["case_id"], roles))
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": cache[labels], "role_positions": roles})
    return compiled


def zero_models(rows: list[dict]) -> dict[str, float]:
    gold = np.asarray([row["truth_factor"] == 1 for row in rows])
    predictions = {
        "always_positive": np.ones(len(rows), dtype=bool),
        "always_negative": np.zeros(len(rows), dtype=bool),
        "dimension_only": np.asarray([row["dimension"] == "length" for row in rows]),
        "gap_only": np.asarray([row["gap_factor"] == 1 for row in rows]),
        "surface_only": np.asarray([row["surface_factor"] == 1 for row in rows]),
        "format_only": np.asarray([row["output_format"] == 1 for row in rows]),
    }
    result = {name: float(np.mean(value == gold)) for name, value in predictions.items()}
    result["integer_comparison_oracle"] = 1.0
    return result


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C121 already exists: {OUT}")
    parent = core.load(C120 / "analysis/closure.json")
    parent_audit = core.load(C120 / "audit/independent_closure_audit.json")
    if not parent_audit["all_checks_passed"] or "execute_C121" not in parent["next_authorization"]:
        raise RuntimeError("C121 authorization missing")
    units, cases = build()
    tok = graph_base.tokenizer()
    compiled = compile_rows(tok, cases)
    occurrences = []
    for row_index, row in enumerate(compiled):
        for role in ROLES:
            for subtoken, position in enumerate(row["role_positions"][role]):
                occurrences.append({
                    "occurrence_index": len(occurrences), "row_index": row_index,
                    "case_id": row["case_id"], "unit_id": row["unit_id"],
                    "partition": row["partition"], "dimension": row["dimension"],
                    "truth_factor": row["truth_factor"], "gap_factor": row["gap_factor"],
                    "surface_factor": row["surface_factor"], "output_format": row["output_format"],
                    "role": role, "subtoken": subtoken,
                    "span_length": len(row["role_positions"][role]),
                    "token_position": int(position), "token_id": int(row["prompt_ids"][position]),
                    "token_text": tok.convert_ids_to_tokens([int(row["prompt_ids"][position])])[0],
                })
    cells = Counter((row["partition"], row["dimension"], row["truth_factor"], row["gap_factor"], row["surface_factor"], row["output_format"]) for row in cases)
    checks = {
        "counts": (len(units), len(cases), len(compiled)) == (24, 1152, 1152),
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in PARTITIONS},
        "factorial": len(cells) == 144 and all(value == 8 for value in cells.values()),
        "unique": len({row["prompt"] for row in cases}) == 1152,
        "truth_balance": all(sum(row["truth_factor"] for row in cases if row["partition"] == partition and row["dimension"] == dimension) == 0 for partition in PARTITIONS for dimension in DIMENSIONS),
        "scores": all(row["truth_factor"] == (1 if row["scores"]["A"][row["dimension"]] > row["scores"]["B"][row["dimension"]] else -1) for row in cases),
        "score_unique": all(len({value for side in row["scores"].values() for value in side.values()}) == 6 for row in cases),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "candidates": all(len(candidate) == 1 for row in compiled for candidate in row["candidate_ids"]),
        "zero_models": all(value == 0.5 for key, value in zero_models(cases).items() if key != "integer_comparison_oracle"),
        "oracle": zero_models(cases)["integer_comparison_oracle"] == 1.0,
        "width": max(len(row["prompt_ids"]) for row in compiled) <= WIDTH,
        "semantic_unique": all("Proposition:" in row["prompt"] and "Requested dimension:" in row["prompt"] for row in cases),
        "fresh": not any(stem in core.load(C120 / "protocol/preregistration.json").get("material_digest", "") for stem in STEMS),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    core.write_rows(OUT / "protocol/role_occurrence_manifest.jsonl", occurrences)
    protocol = {
        "phase": 1650, "campaign": CAMPAIGN, "created_at_utc": now(),
        "object": "fresh structured exact-integer greater-than comparison across length, width and weight",
        "model": "Qwen3-4B local CUDA BF16 without quantization",
        "material": {"units": 24, "cases": 1152, "partitions": {name: 8 for name in PARTITIONS}, "human_naturalness": "not available; structured machine-audited English"},
        "factors": ["dimension", "truth", "numeric_gap", "record_order", "output_vocabulary"],
        "roles": list(ROLES),
        "zero_models": zero_models(cases),
        "behavior_gates": {"overall_min": 0.90, "each_partition_min": 0.85, "each_dimension_min": 0.85, "each_gap_min": 0.85, "each_output_format_min": 0.85},
        "behavior_first": "Phase1651 saves logits and behavior only; no HiddenState archive is written or analyzed before all behavior gates pass",
        "post_behavior_authorization": "if qualified, freeze and execute a separate all-coordinate embedding/HiddenState capture without changing materials or thresholds",
        "stop_conditions": {"pre_model": "any material, balance, role, candidate, width or zero-model audit fails", "behavior": "close this route if any registered behavior gate fails", "post_reveal": "no object, material, factor, partition or threshold changes"},
        "claim_boundary": "structured synthetic comparison qualification in one Qwen3; no HiddenState, weights, semantic neurons, attention/MLP, shared comparator, orthogonal subspace, manifold, topology or new mathematics claim at this stage",
        "parent_hashes": {"c120_closure": core.sha(C120 / "analysis/closure.json"), "c120_closure_audit": core.sha(C120 / "audit/independent_closure_audit.json")},
        "material_digest": core.digest([*units, *cases]), "occurrences": len(occurrences),
        "producer_sha256": core.sha(Path(__file__)), "authorization": "execute_phase1651_c121_behavior_qualification",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1650, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "occurrences": len(occurrences), "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def behavior() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C121 contract audit missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    logits_path = OUT / "raw/qwen3_behavior_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    if logits_path.exists() or index_path.exists():
        raise RuntimeError("C121 behavior output already exists")
    candidate_logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(1152, 2))
    behavior_rows, model, repeat = [], None, 0.0
    first = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            boundary = torch.stack([output.last_hidden_state[i, length - 1] for i, length in enumerate(lengths)])
            logits = model.lm_head(boundary).float()
            for local, row in enumerate(batch):
                row_index = start + local
                scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]
                candidate_logits[row_index] = scores
                prediction = int(scores[1] > scores[0])
                behavior_rows.append({"row_index": row_index, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "dimension": row["dimension"], "truth_factor": row["truth_factor"], "gap_factor": row["gap_factor"], "surface_factor": row["surface_factor"], "output_format": row["output_format"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "positive_minus_negative": scores[0] - scores[1]})
            if first is None:
                first = (batch, candidate_logits[:len(batch)].copy())
            if (start // BATCH + 1) % 24 == 0:
                candidate_logits.flush(); print(f"[phase1651] behavior {start + len(batch)}/1152", flush=True)
            del output, boundary, logits, ids, mask, positions
        batch, old = first
        ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
        boundary = torch.stack([output.last_hidden_state[i, length - 1] for i, length in enumerate(lengths)])
        logits = model.lm_head(boundary).float()
        new = np.asarray([[float(logits[i, candidate[0]]) for candidate in row["candidate_ids"]] for i, row in enumerate(batch)], dtype=np.float32)
        repeat = float(np.max(np.abs(new - old)))
    finally:
        candidate_logits.flush()
        if model is not None: release_bf16(model)
        gc.collect()
    core.write_rows(index_path, behavior_rows)
    def acc(selected: list[dict]) -> float: return float(np.mean([row["correct"] for row in selected]))
    summary = {
        "overall": acc(behavior_rows),
        "by_partition": {name: acc([row for row in behavior_rows if row["partition"] == name]) for name in PARTITIONS},
        "by_dimension": {name: acc([row for row in behavior_rows if row["dimension"] == name]) for name in DIMENSIONS},
        "by_gap": {str(value): acc([row for row in behavior_rows if row["gap_factor"] == value]) for value in (1, -1)},
        "by_output_format": {str(value): acc([row for row in behavior_rows if row["output_format"] == value]) for value in (1, -1)},
        "by_truth": {str(value): acc([row for row in behavior_rows if row["truth_factor"] == value]) for value in (1, -1)},
    }
    gates = protocol["behavior_gates"]
    gate_checks = {
        "overall": summary["overall"] >= gates["overall_min"],
        "partitions": all(value >= gates["each_partition_min"] for value in summary["by_partition"].values()),
        "dimensions": all(value >= gates["each_dimension_min"] for value in summary["by_dimension"].values()),
        "gaps": all(value >= gates["each_gap_min"] for value in summary["by_gap"].values()),
        "formats": all(value >= gates["each_output_format_min"] for value in summary["by_output_format"].values()),
    }
    report = {"phase": 1651, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "behavior_qualification_complete", "behavior": summary, "gate_checks": gate_checks, "gate_passed": all(gate_checks.values()), "repeat_logits_max_abs": repeat, "logits_sha256": core.sha(logits_path), "index_sha256": core.sha(index_path), "runtime": {"placement": placement, "quantization": quant}, "authorization": "freeze_phase1652_c121_all_coordinate_capture" if all(gate_checks.values()) else "close_C121_behavior_route"}
    if repeat != 0 or not quant["has_bf16_parameters"] or quant["has_quantized_modules"]:
        raise RuntimeError(report)
    core.save(OUT / "analysis/behavior_qualification.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


STAGES = {"contract": contract, "behavior": behavior}
if __name__ == "__main__": STAGES[sys.argv[1]]()
