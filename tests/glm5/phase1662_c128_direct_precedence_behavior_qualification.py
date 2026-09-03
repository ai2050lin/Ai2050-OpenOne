#!/usr/bin/env python3
"""C128 fresh direct-precedence language-interface qualification."""
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
OUT = RESULT / "phase1662_c128_direct_precedence_behavior_qualification"
C127 = RESULT / "phase1661_c127_typed_transition_language_family"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1571_c098_observation_first_graph_campaign as graph_base
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

FAMILY = "direct_precedence"
PARTITIONS = ("discovery", "confirmation")
ROLES = ("left_fact", "right_fact", "query_left", "query_right", "boundary")
WIDTH = 144
BATCH = 8
SYSTEM = "Use only the route record. Answer only yes or no."
SYLLABLES = ("bar", "cev", "dun", "fex", "gil", "hor", "jiv", "kar", "lup", "mor", "nis", "pev", "qim", "rus", "sav", "tor")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def values_for(index: int) -> tuple[str, str, str, str]:
    a = SYLLABLES[index % 16]
    b = SYLLABLES[(index * 7 + 5) % 16]
    return tuple(f"Octa{a}{b}{index:02d}{suffix}" for suffix in ("a", "b", "c", "d"))


def prompt_for(values: tuple[str, ...], truth: int, surface: int, distractor: int) -> tuple[str, str, str]:
    left, right, extra_left, extra_right = values
    relation = f"{left} comes before {right}" if surface == 1 else f"{left} appears earlier than {right}"
    extra_a, extra_b = (extra_left, extra_right) if distractor == 1 else (extra_right, extra_left)
    distractor_text = f"{extra_a} comes before {extra_b}" if surface == 1 else f"{extra_a} appears earlier than {extra_b}"
    query_left, query_right = (left, right) if truth == 1 else (right, left)
    return f"Route record: {relation}. Separate record: {distractor_text}. Question: Does {query_left} come before {query_right}? Reply exactly yes or no.", query_left, query_right


def material() -> tuple[list[dict], list[dict]]:
    units, cases = [], []
    for unit_index in range(32):
        values = values_for(unit_index)
        partition = PARTITIONS[unit_index // 16]
        unit = {"unit_id": f"c128-{unit_index:02d}", "family": FAMILY, "partition": partition, "world": "controlled_synthetic_direct_precedence", "values": list(values)}
        units.append(unit)
        for truth, surface, distractor in itertools.product((1, -1), repeat=3):
            prompt, query_left, query_right = prompt_for(values, truth, surface, distractor)
            cases.append({**unit, "case_id": f"c128-{len(cases):04d}", "truth_factor": truth, "surface_factor": surface, "distractor_factor": distractor, "truth": truth == 1, "output_yes": truth == 1, "gold_position": 0 if truth == 1 else 1, "query_left": query_left, "query_right": query_right, "prompt": prompt})
    return units, cases


def historical_values() -> set[str]:
    result = set()
    for path in RESULT.glob("phase*/material/units.jsonl"):
        for row in core.rows(path):
            result.update(str(value).casefold() for value in row.get("values", []))
    return result


def compile_rows(tokenizer, rows: list[dict]) -> list[dict]:
    candidate_ids = [[int(value) for value in tokenizer.encode(" " + candidate, add_special_tokens=False)] for candidate in ("yes", "no")]
    if any(len(value) != 1 for value in candidate_ids):
        raise RuntimeError(candidate_ids)
    compiled = []
    for row in rows:
        ids = core.chat_ids(tokenizer, SYSTEM, row["prompt"])
        left, right, _extra_left, _extra_right = row["values"]
        left_spans = graph_base.name_spans(tokenizer, ids, left)
        right_spans = graph_base.name_spans(tokenizer, ids, right)
        query_left_spans = graph_base.name_spans(tokenizer, ids, row["query_left"])
        query_right_spans = graph_base.name_spans(tokenizer, ids, row["query_right"])
        if min(len(left_spans), len(right_spans), len(query_left_spans), len(query_right_spans)) < 2:
            raise RuntimeError((row["case_id"], left_spans, right_spans))
        roles = {"left_fact": left_spans[0], "right_fact": right_spans[0], "query_left": query_left_spans[-1], "query_right": query_right_spans[-1], "boundary": [len(ids) - 1]}
        if max(roles["left_fact"]) >= min(roles["query_left"]) or max(roles["right_fact"]) >= min(roles["query_right"]):
            raise RuntimeError(("role_order", row["case_id"], roles))
        compiled.append({**row, "prompt_ids": ids, "candidate_ids": candidate_ids, "role_positions": roles})
    return compiled


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(f"C128 already exists: {OUT}")
    parent = core.load(C127 / "analysis/closure.json")
    parent_audit = core.load(C127 / "audit/independent_behavior_failure_audit.json")
    units, cases = material()
    compiled = compile_rows(graph_base.tokenizer(), cases)
    fresh = [str(value).casefold() for row in units for value in row["values"]]
    old = historical_values()
    cells = Counter((row["partition"], row["truth_factor"], row["surface_factor"], row["distractor_factor"]) for row in cases)
    zero = {"always_yes": float(np.mean([row["truth"] for row in cases])), "always_no": float(np.mean([not row["truth"] for row in cases])), "surface_only": float(np.mean([(row["surface_factor"] == 1) == row["truth"] for row in cases])), "distractor_only": float(np.mean([(row["distractor_factor"] == 1) == row["truth"] for row in cases]))}
    checks = {
        "authorization": parent_audit["all_checks_passed"] and parent["next_authorization"].startswith("C128 may freeze"),
        "units": len(units) == 32,
        "cases": len(cases) == 256,
        "factorial": cells == {(partition, *cell): 16 for partition in PARTITIONS for cell in itertools.product((1, -1), repeat=3)},
        "freshness": not (set(fresh) & old) and len(fresh) == len(set(fresh)),
        "unique_prompts": len({row["prompt"] for row in cases}) == 256,
        "zero_models": all(abs(value - 0.5) < 1e-12 for value in zero.values()),
        "candidate_ids": all(row["candidate_ids"] == [[9834], [902]] for row in compiled),
        "roles": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "width": max(len(row["prompt_ids"]) for row in compiled) < WIDTH,
        "machine_naturalness": all(row["prompt"].startswith("Route record:") and row["prompt"].endswith("Reply exactly yes or no.") for row in cases),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "overlap": sorted(set(fresh) & old)})
    core.write_rows(OUT / "material/units.jsonl", units)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    core.write_rows(OUT / "compiled/qwen3.jsonl", compiled)
    protocol = {"phase": 1662, "campaign": "C128", "created_at_utc": now(), "status": "direct_precedence_behavior_contract_frozen", "object": "fresh direct-precedence base language-interface qualification", "model": "Qwen3-4B local BF16 CUDA nonquantized", "units": 32, "cases": 256, "partitions": list(PARTITIONS), "factors": ["truth", "surface", "distractor"], "roles": list(ROLES), "zero_models": zero, "behavior_gate": {"global_accuracy_min": 0.95, "partition_accuracy_min": 0.90, "truth_accuracy_min": 0.90, "surface_accuracy_min": 0.90}, "stop_condition": "failure closes C128 without HiddenState authorization", "claim_boundary": "behavior-interface qualification only; no embedding, HiddenState, activation-coordinate, or mechanism claim", "parent_paths": {"closure": str(C127 / "analysis/closure.json"), "audit": str(C127 / "audit/independent_behavior_failure_audit.json")}, "parent_hashes": {"closure": core.sha(C127 / "analysis/closure.json"), "audit": core.sha(C127 / "audit/independent_behavior_failure_audit.json")}, "producer_sha256": core.sha(Path(__file__)), "authorization": "run_c128_behavior"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    report = {"phase": 1662, "campaign": "C128", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "zero_models": zero, "max_width": max(len(row["prompt_ids"]) for row in compiled), "authorization": protocol["authorization"]}
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def group_accuracy(rows: list[dict]) -> float:
    return float(np.mean([row["correct"] for row in rows]))


@torch.inference_mode()
def behavior() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "run_c128_behavior" or not core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"]:
        raise RuntimeError("C128 behavior authorization missing")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    logits_array = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(rows), 2))
    results = []
    model = None
    repeat = 0.0
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
            output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
            boundary = torch.stack([output.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)])
            logits = model.lm_head(boundary).float()
            for local, row in enumerate(batch):
                scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
                logits_array[start + local] = scores
                prediction = int(scores[1] > scores[0])
                results.append({"row_index": start + local, "case_id": row["case_id"], "unit_id": row["unit_id"], "partition": row["partition"], "truth_factor": row["truth_factor"], "surface_factor": row["surface_factor"], "distractor_factor": row["distractor_factor"], "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "yes_minus_no": float(scores[0] - scores[1])})
            del output, boundary, logits, ids, mask, positions
        logits_array.flush()
        batch = rows[:BATCH]
        ids, mask, positions, lengths = fixed_base.fixed_batch(batch, pad, device, WIDTH)
        output = model.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, output_hidden_states=False, return_dict=True)
        boundary = torch.stack([output.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)])
        logits = model.lm_head(boundary).float()
        for local, row in enumerate(batch):
            scores = np.asarray([float(logits[local, candidate[0]]) for candidate in row["candidate_ids"]], dtype=np.float32)
            repeat = max(repeat, float(np.max(np.abs(scores - logits_array[local]))))
    finally:
        logits_array.flush()
        if model is not None:
            release_bf16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    core.write_rows(OUT / "raw/qwen3_behavior_index.jsonl", results)
    summary = {"global_accuracy": group_accuracy(results), "by_partition": {key: group_accuracy([row for row in results if row["partition"] == key]) for key in PARTITIONS}, "by_truth": {str(key): group_accuracy([row for row in results if row["truth_factor"] == key]) for key in (1, -1)}, "by_surface": {str(key): group_accuracy([row for row in results if row["surface_factor"] == key]) for key in (1, -1)}}
    gates = protocol["behavior_gate"]
    gate = summary["global_accuracy"] >= gates["global_accuracy_min"] and min(summary["by_partition"].values()) >= gates["partition_accuracy_min"] and min(summary["by_truth"].values()) >= gates["truth_accuracy_min"] and min(summary["by_surface"].values()) >= gates["surface_accuracy_min"]
    checks = {"rows": len(results) == 256, "finite": bool(np.isfinite(logits_array).all()), "repeat": repeat == 0.0, "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"]}
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1662, "campaign": "C128", "created_at_utc": now(), "status": "behavior_qualified" if gate else "behavior_gate_failed", "checks": checks, "summary": summary, "gate_passed": gate, "repeat_logits_max_abs": repeat, "runtime": {"placement": placement, "quantization": quant}, "authorization": "close_c128_and_authorize_c129_typed_capture" if gate else "close_c128_failed"}
    core.save(OUT / "analysis/behavior_gate.json", report)
    core.save(OUT / "audit/internal_behavior_audit.json", {"phase": 1662, "campaign": "C128", "checks": checks, "all_integrity_checks_passed": all(checks.values()), "scientific_gate_passed": gate, "authorization": report["authorization"]})
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}, indent=2))


def close() -> None:
    behavior_report = core.load(OUT / "analysis/behavior_gate.json")
    closure = {"phase": 1662, "campaign": "C128", "created_at_utc": now(), "status": "direct_precedence_behavior_qualified" if behavior_report["gate_passed"] else "direct_precedence_behavior_failed", "headline": behavior_report["summary"], "result": "C128 is a behavior-interface qualification stage; no HiddenState was captured.", "theory_update": "No internal mechanism update.", "problems": ["controlled synthetic English", "direct one-hop relation only", "Qwen3 only", "no human naturalness audit"], "claim_boundary": "behavior only; no embedding, HiddenState, activation-coordinate, causal-path, or operator claim", "next_authorization": "C129 may capture uniformly typed embedding, all post-block pre-final-norm HiddenStates, and final-norm states on the frozen C128 material" if behavior_report["gate_passed"] else "No C129 internal capture is authorized from C128"}
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "behavior_integrity": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"], "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists()}
    core.save(OUT / "audit/internal_closure_audit.json", {"phase": 1662, "campaign": "C128", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": behavior_report["gate_passed"], "authorization": "independent_audit_then_continue"})
    print(json.dumps({"checks": checks, "closure": closure}, indent=2))


def main() -> None:
    modes = {"contract": contract, "behavior": behavior, "close": close}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(f"usage: {Path(__file__).name} {{{'|'.join(modes)}}}")
    modes[sys.argv[1]]()


if __name__ == "__main__":
    main()
