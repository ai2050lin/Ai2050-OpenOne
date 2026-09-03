#!/usr/bin/env python3
"""C122: discovery-selected multi-output interface calibration for comparison behavior."""
from __future__ import annotations

import gc
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1653_c122_multi_interface_comparison_calibration"
C121 = RESULT / "phase1650_c121_structured_comparison_qualification"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

CAMPAIGN = "C122"
PARTITIONS = ("discovery", "confirmation", "lockbox")
DIMENSIONS = ("length", "width", "weight")
INTERFACES = ("yes_no", "true_false", "correct_incorrect", "A_B", "alpha_beta", "larger_smaller")
LABELS = {
    "yes_no": ("yes", "no"), "true_false": ("true", "false"),
    "correct_incorrect": ("correct", "incorrect"), "A_B": ("A", "B"),
    "alpha_beta": ("alpha", "beta"), "larger_smaller": ("larger", "smaller"),
}
WIDTH, BATCH = 256, 8
STEMS = (
    "alvo", "brin", "cusk", "daro", "epli", "furo", "gexa", "hilm",
    "ispa", "joru", "keln", "lavo", "mert", "nupa", "oski", "pavo",
    "qint", "rume", "savi", "tego", "ulmo", "vepi", "wori", "xalu",
)
SYSTEM = "Use the exact integer table. Compare only the requested dimension and obey the requested output key."


def now() -> str: return datetime.now(timezone.utc).isoformat()


def score_table(unit_index: int, dimension: str, truth: int, gap: int) -> dict[str, dict[str, int]]:
    high, low = ((6, 5) if gap == 1 else (9, 2))
    a_query, b_query = (high, low) if truth == 1 else (low, high)
    pool = [1, 3, 8, 10] if gap == 1 else [1, 4, 7, 10]
    shift = (unit_index + 2 * DIMENSIONS.index(dimension)) % 4
    pool = pool[shift:] + pool[:shift]
    result = {"A": {dimension: a_query}, "B": {dimension: b_query}}
    for index, name in enumerate(value for value in DIMENSIONS if value != dimension):
        result["A"][name] = pool[index * 2]; result["B"][name] = pool[index * 2 + 1]
    return result


def prompt_for(unit_index: int, values: tuple[str, str], dimension: str, truth: int, gap: int, surface: int, interface: str) -> tuple[str, dict]:
    focus, other = values
    scores = score_table(unit_index, dimension, truth, gap)
    records = {
        "A": f"Item A ({focus}) [length={scores['A']['length']}; width={scores['A']['width']}; weight={scores['A']['weight']}]",
        "B": f"Item B ({other}) [length={scores['B']['length']}; width={scores['B']['width']}; weight={scores['B']['weight']}]",
    }
    order = ("A", "B") if surface == 1 else ("B", "A")
    base = f"Exact table: {records[order[0]]}; {records[order[1]]}. Requested dimension: {dimension}. "
    if interface == "yes_no":
        ending = f"Is item A's {dimension} score greater than item B's {dimension} score? Reply exactly yes or no."
    elif interface == "true_false":
        ending = f"Proposition: item A's {dimension} score is greater than item B's {dimension} score. Reply exactly true or false."
    elif interface == "correct_incorrect":
        ending = f"Proposition: item A's {dimension} score is greater than item B's {dimension} score. Reply exactly correct or incorrect."
    elif interface == "A_B":
        ending = f"Which labeled item has the greater {dimension} score? Reply exactly A or B."
    elif interface == "alpha_beta":
        ending = f"Output key: alpha means item A; beta means item B. Which keyed item has the greater {dimension} score? Reply exactly alpha or beta."
    else:
        ending = f"Relative to item B, is item A's {dimension} score larger or smaller? Reply exactly larger or smaller."
    return base + ending, {"scores": scores, "truth_factor": truth, "output_labels": list(LABELS[interface])}


def build() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index, stem in enumerate(STEMS):
        partition = PARTITIONS[unit_index // 8]
        values = (f"Cal{stem}A", f"Cal{stem}B")
        unit = {"unit_id": f"c122-interface-{unit_index:02d}", "partition": partition, "values": list(values)}
        units.append(unit)
        for dimension, truth, gap, surface, interface in itertools.product(DIMENSIONS, (1, -1), (1, -1), (1, -1), INTERFACES):
            prompt, metadata = prompt_for(unit_index, values, dimension, truth, gap, surface, interface)
            cases.append({**unit, **metadata, "case_id": f"c122-{len(cases):04d}", "dimension": dimension, "gap_factor": gap, "surface_factor": surface, "interface": interface, "gold_position": 0 if truth == 1 else 1, "prompt": prompt})
    return units, cases


def compile_rows(tok, rows: list[dict]) -> list[dict]:
    cache = {}
    compiled = []
    for row in rows:
        labels = tuple(row["output_labels"])
        if labels not in cache:
            values = [tok.encode(" " + label, add_special_tokens=False) for label in labels]
            if any(len(value) != 1 for value in values): raise RuntimeError((labels, values))
            cache[labels] = [[int(value[0])] for value in values]
        compiled.append({**row, "prompt_ids": core.chat_ids(tok, SYSTEM, row["prompt"]), "candidate_ids": cache[labels]})
    return compiled


def contract() -> None:
    if OUT.exists(): raise RuntimeError(f"C122 exists: {OUT}")
    parent = core.load(C121 / "analysis/closure.json"); audit = core.load(C121 / "audit/independent_closure_audit.json")
    if not audit["all_checks_passed"] or not parent["next_authorization"].startswith("execute_C122"): raise RuntimeError("C122 authorization missing")
    units, cases = build(); tok = graph_base.tokenizer(); compiled = compile_rows(tok, cases)
    cells = Counter((row["partition"], row["dimension"], row["truth_factor"], row["gap_factor"], row["surface_factor"], row["interface"]) for row in cases)
    prior_values = {str(value).casefold() for row in core.rows(C121 / "material/units.jsonl") for value in row["values"]}
    current_values = {str(value).casefold() for row in units for value in row["values"]}
    checks = {
        "counts": (len(units), len(cases), len(compiled)) == (24, 3456, 3456),
        "partitions": Counter(row["partition"] for row in units) == {name: 8 for name in PARTITIONS},
        "factorial": len(cells) == 432 and all(value == 8 for value in cells.values()),
        "truth_balance": all(sum(row["truth_factor"] for row in cases if row["partition"] == p and row["interface"] == interface) == 0 for p in PARTITIONS for interface in INTERFACES),
        "scores": all(row["truth_factor"] == (1 if row["scores"]["A"][row["dimension"]] > row["scores"]["B"][row["dimension"]] else -1) for row in cases),
        "unique": len({row["prompt"] for row in cases}) == 3456,
        "fresh": not (prior_values & current_values),
        "candidates": all(len(candidate) == 1 for row in compiled for candidate in row["candidate_ids"]),
        "width": max(len(row["prompt_ids"]) for row in compiled) <= WIDTH,
    }
    if not all(checks.values()): raise RuntimeError(checks)
    core.write_rows(OUT / "material/units.jsonl", units); core.write_rows(OUT / "material/cases.jsonl", cases); core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {
        "phase": 1653, "campaign": CAMPAIGN, "created_at_utc": now(), "object": "comparison output-interface calibration with discovery-only selection", "model": "Qwen3-4B local CUDA BF16 without quantization",
        "material": {"units": 24, "cases": 3456, "partitions": {name: 8 for name in PARTITIONS}}, "factors": ["dimension", "truth", "gap", "record_order", "output_interface"], "interfaces": list(INTERFACES),
        "selection_rule": {"read_partition": "discovery only", "eligible_each_dimension_min": 0.80, "eligible_each_truth_min": 0.75, "eligible_each_gap_min": 0.75, "rank": "max min(dimension,truth,gap), then overall, then frozen interface order"},
        "holdout_gates": {"confirmation_overall_min": 0.85, "lockbox_overall_min": 0.85, "each_dimension_min": 0.80, "each_truth_min": 0.75, "each_gap_min": 0.75},
        "behavior_first": "capture logits for all frozen interfaces; selection script reads discovery rows only; holdout script reads only the frozen winner on confirmation and lockbox; no HiddenState archive",
        "stop_conditions": {"pre_model": "any audit fails", "selection": "no eligible discovery interface closes route", "holdout": "winner failing any holdout gate closes route", "post_reveal": "no interfaces, rules or gates change"},
        "claim_boundary": "output-interface calibration only; no HiddenState, weights, attention/MLP, semantic neuron, shared comparator, manifold, topology or new mathematics claim",
        "material_digest": core.digest([*units, *cases]), "producer_sha256": core.sha(Path(__file__)), "authorization": "execute_phase1654_c122_all_interface_behavior_capture",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1653, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report); print(json.dumps(report, indent=2))


@torch.inference_mode()
def capture() -> None:
    if not core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"]: raise RuntimeError("C122 contract audit missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl"); logits_path = OUT / "raw/qwen3_all_interface_logits.float32.npy"; index_path = OUT / "raw/qwen3_all_interface_behavior.jsonl"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    if logits_path.exists() or index_path.exists(): raise RuntimeError("C122 output exists")
    saved = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(3456, 2)); result = []; model = None; repeat = 0.0; first = None
    try:
        model, tok, device, placement = load_bf16("qwen3"); quant = quantization_audit(model); pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start+BATCH]; ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True); boundary = torch.stack([output.last_hidden_state[i, length-1] for i, length in enumerate(lengths)]); logits = model.lm_head(boundary).float()
            for local, row in enumerate(batch):
                idx = start + local; scores = [float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]]; saved[idx] = scores; prediction = int(scores[1] > scores[0]); result.append({"row_index": idx, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "dimension": row["dimension"], "truth_factor": row["truth_factor"], "gap_factor": row["gap_factor"], "surface_factor": row["surface_factor"], "interface": row["interface"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "candidate0_minus_candidate1": scores[0]-scores[1]})
            if first is None: first = (batch, saved[:len(batch)].copy())
            if (start // BATCH + 1) % 48 == 0: saved.flush(); print(f"[phase1654] {start+len(batch)}/3456", flush=True)
            del output, boundary, logits, ids, mask, positions
        batch, old = first; ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH); output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True); boundary = torch.stack([output.last_hidden_state[i, length-1] for i, length in enumerate(lengths)]); logits = model.lm_head(boundary).float(); new = np.asarray([[float(logits[i, c[0]]) for c in row["candidate_ids"]] for i,row in enumerate(batch)], dtype=np.float32); repeat = float(np.max(np.abs(new-old)))
    finally:
        saved.flush()
        if model is not None: release_bf16(model)
        gc.collect()
    core.write_rows(index_path, result)
    report = {"phase": 1654, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "all_interface_behavior_captured_holdouts_sealed", "rows": len(result), "repeat_logits_max_abs": repeat, "logits_sha256": core.sha(logits_path), "index_sha256": core.sha(index_path), "runtime": {"placement": placement, "quantization": quant}, "authorization": "execute_phase1655_c122_discovery_interface_selection"}
    if len(result) != 3456 or repeat != 0 or not quant["has_bf16_parameters"] or quant["has_quantized_modules"]: raise RuntimeError(report)
    core.save(OUT / "analysis/capture_summary.json", report); print(json.dumps({k:v for k,v in report.items() if k != "runtime"}, indent=2))


def acc(rows: list[dict]) -> float: return sum(row["correct"] for row in rows) / len(rows)


def selection() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_capture_audit.json")["all_checks_passed"]: raise RuntimeError("C122 capture audit missing")
    all_rows = core.rows(OUT / "raw/qwen3_all_interface_behavior.jsonl"); rows = [row for row in all_rows if row["partition"] == "discovery"]
    table = []
    for interface in INTERFACES:
        local = [row for row in rows if row["interface"] == interface]
        summary = {"interface": interface, "n": len(local), "overall": acc(local), "by_dimension": {name: acc([row for row in local if row["dimension"] == name]) for name in DIMENSIONS}, "by_truth": {str(v): acc([row for row in local if row["truth_factor"] == v]) for v in (1,-1)}, "by_gap": {str(v): acc([row for row in local if row["gap_factor"] == v]) for v in (1,-1)}}
        summary["minimum_slice"] = min(*summary["by_dimension"].values(), *summary["by_truth"].values(), *summary["by_gap"].values()); summary["eligible"] = min(summary["by_dimension"].values()) >= protocol["selection_rule"]["eligible_each_dimension_min"] and min(summary["by_truth"].values()) >= protocol["selection_rule"]["eligible_each_truth_min"] and min(summary["by_gap"].values()) >= protocol["selection_rule"]["eligible_each_gap_min"]; table.append(summary)
    eligible = [row for row in table if row["eligible"]]
    winner = None if not eligible else sorted(eligible, key=lambda row: (-row["minimum_slice"], -row["overall"], INTERFACES.index(row["interface"])))[0]
    freeze = {"phase": 1655, "campaign": CAMPAIGN, "created_at_utc": now(), "read_partition": "discovery", "table": table, "winner": winner, "source_index_sha256": core.sha(OUT / "raw/qwen3_all_interface_behavior.jsonl"), "authorization": "close_C122_no_eligible_interface" if winner is None else "execute_phase1656_c122_frozen_winner_holdout_validation"}
    core.save(OUT / "protocol/frozen_interface_selection.json", freeze); print(json.dumps(freeze, indent=2))


def validate() -> None:
    freeze = core.load(OUT / "protocol/frozen_interface_selection.json"); protocol = core.load(OUT / "protocol/preregistration.json")
    if not core.load(OUT / "audit/independent_selection_audit.json")["all_checks_passed"] or freeze["winner"] is None: raise RuntimeError("C122 eligible frozen winner missing")
    rows = [row for row in core.rows(OUT / "raw/qwen3_all_interface_behavior.jsonl") if row["interface"] == freeze["winner"]["interface"] and row["partition"] != "discovery"]
    summaries = []
    for partition in ("confirmation", "lockbox"):
        local = [row for row in rows if row["partition"] == partition]; summaries.append({"partition": partition, "n": len(local), "overall": acc(local), "by_dimension": {name: acc([row for row in local if row["dimension"] == name]) for name in DIMENSIONS}, "by_truth": {str(v): acc([row for row in local if row["truth_factor"] == v]) for v in (1,-1)}, "by_gap": {str(v): acc([row for row in local if row["gap_factor"] == v]) for v in (1,-1)}})
    gates = protocol["holdout_gates"]; checks = {row["partition"]: row["overall"] >= gates[f"{row['partition']}_overall_min"] and min(row["by_dimension"].values()) >= gates["each_dimension_min"] and min(row["by_truth"].values()) >= gates["each_truth_min"] and min(row["by_gap"].values()) >= gates["each_gap_min"] for row in summaries}
    report = {"phase": 1656, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "frozen_interface_holdout_revealed", "winner": freeze["winner"]["interface"], "summaries": summaries, "checks": checks, "passed": all(checks.values()), "authorization": "freeze_C123_all_coordinate_capture_on_C122_winner" if all(checks.values()) else "close_comparison_interface_campaign"}
    core.save(OUT / "analysis/holdout_validation.json", report); print(json.dumps(report, indent=2))


STAGES = {"contract": contract, "capture": capture, "selection": selection, "validate": validate}
if __name__ == "__main__": STAGES[sys.argv[1]]()
