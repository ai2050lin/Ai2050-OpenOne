#!/usr/bin/env python3
"""Phase 1231: execute the frozen Qwen3 clock-to-compass behavior contract.

This phase consumes the exact Phase1230 token manifest, freezes an exact
length-bucket batch plan before model output, runs Qwen3-4B in full CUDA FP16,
and adjudicates the six preregistered Q0--Q5 ledgers.  It never requests or
stores hidden states or attentions and performs no intervention.

A protocol-side shortcut audit is also frozen before execution.  It does not
change Q0--Q5; it only limits interpretation if non-target records can recover
the answer under the bijective C4 material construction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1231
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1231_qwen3_clock_compass_behavior_execution_audit.py"

UPSTREAM_ROOT = TEST_ROOT / "result/phase1230_qwen3_clock_compass_behavior_protocol"
UPSTREAM_CONTRACT = UPSTREAM_ROOT / "protocol/preregistration.json"
UPSTREAM_MANIFEST = UPSTREAM_ROOT / "protocol/qwen3_manifest.jsonl"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_FINAL_AUDIT = UPSTREAM_ROOT / "audit/independent_final_audit.json"
MATERIAL_PATH = TEST_ROOT / "result/phase1229_deanswer_clock_compass_material_contract/material/clock_compass_binding.jsonl"

EXPECTED_UPSTREAM_CONTRACT = "9dd1b27cee49941ef2262602d2c231dfbdb2d998f8cc05e162edd2b2cf087d2c"
EXPECTED_UPSTREAM_FINAL = "ebee860364c036ff9700d4a8af30b6a2f7309a5d24c03fccd1e451f895d4923b"
EXPECTED_UPSTREAM_AUDIT = "00bc4facb62df8cfa4fe89d41f849bb6650dd27dcabadc9624e21cf8703e0003"
EXPECTED_MANIFEST_DIGEST = "15f82fb7092a3444b22290749d93f37c683d0be872b0e00bc461ed846eced4db"

OUT_ROOT = TEST_ROOT / "result/phase1231_qwen3_clock_compass_behavior_execution"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
SHORTCUT_PATH = OUT_ROOT / "protocol/preoutput_shortcut_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use")
PANELS = ("active", "matched_null", "surface_order")
CANDIDATES = ("north", "east", "south", "west")
ACTIVE_AXES = (
    "world_id",
    "template_id",
    "target_entity",
    "gold_candidate",
    "order_variant",
    "mapping_variant",
)
EXPECTED_ROWS = 9216
EXPECTED_ACTIVE = 3072
BATCH_SIZE = 16
TIE_TOLERANCE = 1e-7

THRESHOLDS = {
    "Q0_finite_rate": 1.0,
    "Q1_panel_accuracy": 0.90,
    "Q1_active_worst_marginal": 0.80,
    "Q2_active_quartet": 0.75,
    "Q3_control_invariant_bundle": 0.80,
    "Q4_template_pair": 0.85,
    "Q5_natural_first_token": 0.80,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def source_hashes() -> dict[str, str]:
    return {
        "execution": file_sha256(SCRIPT),
        "independent_audit": file_sha256(AUDIT_SCRIPT),
    }


def verify_upstream() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    upstream_contract = read_json(UPSTREAM_CONTRACT)
    upstream_final = read_json(UPSTREAM_FINAL)
    upstream_audit = read_json(UPSTREAM_FINAL_AUDIT)
    manifest = read_jsonl(UPSTREAM_MANIFEST)
    material = read_jsonl(MATERIAL_PATH)
    if upstream_contract.get("contract_digest") != EXPECTED_UPSTREAM_CONTRACT:
        raise RuntimeError("Phase1230 contract digest mismatch")
    if upstream_final.get("final_digest") != EXPECTED_UPSTREAM_FINAL:
        raise RuntimeError("Phase1230 final digest mismatch")
    if upstream_audit.get("audit_digest") != EXPECTED_UPSTREAM_AUDIT or not upstream_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1230 final audit mismatch")
    if upstream_final.get("manifest_digest") != EXPECTED_MANIFEST_DIGEST:
        raise RuntimeError("Phase1230 manifest digest mismatch")
    if digest(manifest) != EXPECTED_MANIFEST_DIGEST or len(manifest) != EXPECTED_ROWS:
        raise RuntimeError("frozen manifest changed")
    if sum(row["panel"] == "active" for row in material) != EXPECTED_ACTIVE:
        raise RuntimeError("Phase1229 material active count mismatch")
    if upstream_contract["future_execution"]["phase"] != PHASE:
        raise RuntimeError("Phase1230 did not name Phase1231 execution")
    return upstream_contract, manifest, material


def build_batch_plan(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        buckets[int(row["input_length"])].append(row)
    batches: list[dict[str, Any]] = []
    batch_index = 0
    for length in sorted(buckets):
        ordered = sorted(buckets[length], key=lambda row: int(row["execution_index"]))
        for start in range(0, len(ordered), BATCH_SIZE):
            members = ordered[start : start + BATCH_SIZE]
            batches.append({
                "batch_index": batch_index,
                "input_length": length,
                "runtime_batch_size": len(members),
                "execution_indices": [int(row["execution_index"]) for row in members],
                "item_ids": [row["item_id"] for row in members],
            })
            batch_index += 1
    flat = [item for batch in batches for item in batch["item_ids"]]
    if len(flat) != EXPECTED_ROWS or len(set(flat)) != EXPECTED_ROWS:
        raise RuntimeError("batch plan does not partition manifest")
    plan: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1231.qwen3.frozen_batch_plan.v1",
        "batch_size": BATCH_SIZE,
        "adaptive_fallback": False,
        "bucket_count": len(buckets),
        "batch_count": len(batches),
        "bucket_counts": {str(k): len(v) for k, v in sorted(buckets.items())},
        "batches": batches,
    }
    plan["plan_digest"] = digest(plan)
    return plan


def empirical_lookup_accuracy(rows: list[dict[str, Any]], feature: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[canonical_json(feature(row))][row["gold_candidate"]] += 1
    correct = sum(max(counts.values()) for counts in table.values())
    return {"accuracy": correct / len(rows), "cell_count": len(table), "row_count": len(rows)}


def clock_at(row: dict[str, Any], entity_index: int) -> str:
    return row["assignments"][row["entities"][entity_index]]


def non_target_witness(row: dict[str, Any], rank: int) -> tuple[int, str]:
    indices = [int(i) for i in row["record_order_indices"] if int(i) != int(row["target_index"])]
    index = indices[rank]
    return index, clock_at(row, index)


def build_shortcut_audit(material: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in material if row["panel"] == "active"]
    feature_specs: dict[str, Callable[[dict[str, Any]], Any]] = {
        "constant": lambda row: "constant",
        "first_clock_only": lambda row: clock_at(row, int(row["record_order_indices"][0])),
        "target_index_plus_first_record": lambda row: (
            int(row["target_index"]),
            int(row["record_order_indices"][0]),
            clock_at(row, int(row["record_order_indices"][0])),
        ),
        "target_index_plus_one_non_target": lambda row: (
            int(row["target_index"]),
            *non_target_witness(row, 0),
        ),
        "target_index_plus_two_non_targets": lambda row: (
            int(row["target_index"]),
            *non_target_witness(row, 0),
            *non_target_witness(row, 1),
        ),
        "target_index_plus_three_non_targets": lambda row: (
            int(row["target_index"]),
            *non_target_witness(row, 0),
            *non_target_witness(row, 1),
            *non_target_witness(row, 2),
        ),
        "intended_target_clock": lambda row: clock_at(row, int(row["target_index"])),
    }
    results = {name: empirical_lookup_accuracy(active, feature) for name, feature in feature_specs.items()}
    value = {
        "phase": PHASE,
        "schema_version": "phase1231.preoutput.shortcut_audit.v1",
        "created_at_utc": utc_now(),
        "scope": "Phase1229 active panel; computed before any Phase1231 model output",
        "method": "empirical Bayes lookup accuracy for preregistered surface-extractable feature tuples",
        "results": results,
        "c4_single_or_pair_risk": results["target_index_plus_two_non_targets"]["accuracy"] > 0.25,
        "global_complement_identifiability_failure": results["target_index_plus_three_non_targets"]["accuracy"] == 1.0,
        "interpretation": (
            "Q0-Q5 remain frozen. A behavior pass identifies task solution, not target-record use: "
            "because all four clock values occur bijectively, the three non-target records determine the missing target value. "
            "Expanding C4 to S4 removes some low-order cycle codes but does not remove the full complement strategy."
        ),
        "does_not_change_Q0_Q5": True,
    }
    value["shortcut_audit_digest"] = digest(value)
    return value


def preregister() -> None:
    if any(path.exists() for path in (CONTRACT_PATH, BATCH_PLAN_PATH, SHORTCUT_PATH)):
        raise RuntimeError("Phase1231 preregistration output already exists")
    upstream, manifest, material = verify_upstream()
    plan = build_batch_plan(manifest)
    shortcut = build_shortcut_audit(material)
    write_json(BATCH_PLAN_PATH, plan)
    write_json(SHORTCUT_PATH, shortcut)
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1231.qwen3.clock_compass.behavior_execution.v1",
        "created_at_utc": utc_now(),
        "objective": "Execute the frozen Phase1230 Qwen3 behavior interface and adjudicate Q0-Q5 without hidden states.",
        "source_hashes": source_hashes(),
        "upstream": {
            "contract_digest": EXPECTED_UPSTREAM_CONTRACT,
            "final_digest": EXPECTED_UPSTREAM_FINAL,
            "final_audit_digest": EXPECTED_UPSTREAM_AUDIT,
            "manifest_digest": EXPECTED_MANIFEST_DIGEST,
            "file_hashes": {
                "contract": file_sha256(UPSTREAM_CONTRACT),
                "manifest": file_sha256(UPSTREAM_MANIFEST),
                "final": file_sha256(UPSTREAM_FINAL),
                "final_audit": file_sha256(UPSTREAM_FINAL_AUDIT),
                "material": file_sha256(MATERIAL_PATH),
            },
        },
        "execution": {
            "model": "qwen3",
            "precision": "float16",
            "device": "cuda",
            "quantization": "none",
            "eval": True,
            "inference_mode": True,
            "hidden_states": False,
            "attentions": False,
            "intervention": False,
            "input": "exact frozen input_ids only",
            "batch_plan_digest": plan["plan_digest"],
            "batch_size": BATCH_SIZE,
            "adaptive_batch_fallback": False,
            "score": "FP32 log_softmax at final prompt position",
            "tie_tolerance": TIE_TOLERANCE,
        },
        "thresholds": THRESHOLDS,
        "ledgers": {
            "Q0": "overall and every split finite rate",
            "Q1": "every split x panel candidate accuracy and active one-factor worst cells",
            "Q2": "active four-state counterfactual quartet success per split",
            "Q3": "matched-null and surface-order invariant bundle success per split x control",
            "Q4": "paired-template correctness and prediction invariance per split x panel",
            "Q5": "natural_use full-vocabulary first-token top1 equals gold candidate token",
        },
        "shortcut_sidecar": {
            "digest": shortcut["shortcut_audit_digest"],
            "changes_formal_gates": False,
            "limits_target_binding_claim": shortcut["global_complement_identifiability_failure"],
        },
        "authorization_rule": {
            "candidate_behavior": "Q0 and Q1 and Q2 and Q3 and Q4",
            "natural_first_token": "candidate_behavior and Q5",
            "Q5_does_not_veto_candidate_behavior": True,
            "shortcut_audit_does_not_rewrite_Q0_Q5": True,
            "shortcut_audit_limits_mechanism_interpretation": True,
        },
        "forbidden": [
            "retokenize or rerender prompt at runtime",
            "change batch membership, thresholds, tie rule, candidates, split, panel, or denominator",
            "save hidden states or attentions",
            "perform activation or parameter intervention",
            "call a behavior pass evidence of target-record-specific neural routing",
            "run GLM4 or DS7B in this phase",
        ],
        "upstream_claim_boundary": upstream["claim_boundary"],
    }
    contract["contract_digest"] = digest(contract)
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({
        "status": "phase1231_preregistered",
        "contract_digest": contract["contract_digest"],
        "batch_plan_digest": plan["plan_digest"],
        "shortcut_audit_digest": shortcut["shortcut_audit_digest"],
    }))


def verify_frozen() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _upstream, manifest, material = verify_upstream()
    contract = read_json(CONTRACT_PATH)
    plan = read_json(BATCH_PLAN_PATH)
    shortcut = read_json(SHORTCUT_PATH)
    if contract.get("contract_digest") != digest(strip_digest(contract, "contract_digest")):
        raise RuntimeError("Phase1231 contract digest drift")
    if contract.get("source_hashes") != source_hashes():
        raise RuntimeError("Phase1231 source changed after preregistration")
    if plan.get("plan_digest") != digest(strip_digest(plan, "plan_digest")):
        raise RuntimeError("batch plan digest drift")
    if shortcut.get("shortcut_audit_digest") != digest(strip_digest(shortcut, "shortcut_audit_digest")):
        raise RuntimeError("shortcut audit digest drift")
    if contract["execution"]["batch_plan_digest"] != plan["plan_digest"]:
        raise RuntimeError("contract batch plan link mismatch")
    if contract["shortcut_sidecar"]["digest"] != shortcut["shortcut_audit_digest"]:
        raise RuntimeError("contract shortcut link mismatch")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    return contract, plan, manifest, material


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1231 behavior output already exists")
    contract, plan, manifest, _material = verify_frozen()
    manifest_by_id = {row["item_id"]: row for row in manifest}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    if device.type != "cuda":
        release_fp16(model)
        raise RuntimeError("Phase1231 requires CUDA")
    if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
        release_fp16(model)
        raise RuntimeError("Phase1231 precision contract failed")
    raw: list[dict[str, Any]] = []
    try:
        for batch_number, batch in enumerate(plan["batches"], start=1):
            members = [manifest_by_id[item_id] for item_id in batch["item_ids"]]
            expected_length = int(batch["input_length"])
            if any(len(row["input_ids"]) != expected_length for row in members):
                raise RuntimeError("runtime batch length drift")
            input_ids = torch.tensor([row["input_ids"] for row in members], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=1,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            logits = output.logits[:, -1, :].float()
            finite_rows = torch.isfinite(logits).all(dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            top1_ids = logits.argmax(dim=-1)
            for row_index, source in enumerate(members):
                score_map = {
                    candidate: float(log_probs[row_index, int(source["candidate_token_ids"][candidate][0])].item())
                    for candidate in CANDIDATES
                }
                order = sorted(CANDIDATES, key=lambda candidate: score_map[candidate], reverse=True)
                top_margin = score_map[order[0]] - score_map[order[1]]
                finite = bool(finite_rows[row_index].item()) and all(math.isfinite(value) for value in score_map.values())
                prediction = None if (not finite or top_margin <= TIE_TOLERANCE) else order[0]
                gold = source["gold_candidate"]
                wrong_best = max(score for candidate, score in score_map.items() if candidate != gold)
                full_top1_id = int(top1_ids[row_index].item())
                result: dict[str, Any] = {
                    "phase": PHASE,
                    "schema_version": "phase1231.qwen3.behavior.row.v1",
                    "contract_digest": contract["contract_digest"],
                    "item_id": source["item_id"],
                    "manifest_row_digest": source["manifest_row_digest"],
                    "execution_index": int(source["execution_index"]),
                    "split": source["split"],
                    "panel": source["panel"],
                    "bundle_id": source["bundle_id"],
                    "world_id": source["world_id"],
                    "template_id": source["template_id"],
                    "target_entity": source["target_entity"],
                    "gold_candidate": gold,
                    "all_vocab_logits_finite": finite,
                    "candidate_scores": score_map,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "unresolved_tie": finite and top_margin <= TIE_TOLERANCE,
                    "top_candidate_margin": top_margin,
                    "gold_margin": score_map[gold] - wrong_best,
                    "full_vocab_top1_id": full_top1_id,
                    "full_vocab_top1_text": tokenizer.decode([full_top1_id], skip_special_tokens=False),
                    "full_vocab_top1_is_gold_candidate": full_top1_id == int(source["gold_candidate_token_id"]),
                    "input_length": expected_length,
                    "runtime_batch_size": len(members),
                    "batch_index": int(batch["batch_index"]),
                }
                result["behavior_row_digest"] = digest(result)
                raw.append(result)
            del output, logits, log_probs, top1_ids, finite_rows, input_ids
            if batch_number % 50 == 0 or batch_number == len(plan["batches"]):
                print(f"[phase1231] {batch_number}/{len(plan['batches'])} batches", flush=True)
        raw.sort(key=lambda row: row["execution_index"])
        write_jsonl(RAW_PATH, raw)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1231.qwen3.run_summary.v1",
            "created_at_utc": utc_now(),
            "model": "qwen3",
            "contract_digest": contract["contract_digest"],
            "case_count": len(raw),
            "raw_digest": digest(raw),
            "precision_audit": precision,
            "placement": placement,
            "batch_plan_digest": plan["plan_digest"],
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hidden_states_saved": False,
            "attentions_saved": False,
            "interventions_performed": False,
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(canonical_json({"status": "behavior_complete", "rows": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def group_success(rows: list[dict[str, Any]], keys: tuple[str, ...], expected_size: int, invariant: bool) -> float:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    successes = []
    for values in groups.values():
        predictions = [row["prediction"] for row in values]
        success = len(values) == expected_size and all(row["all_vocab_logits_finite"] and row["correct"] for row in values)
        if invariant:
            success = success and len(set(predictions)) == 1
        else:
            success = success and set(predictions) == set(CANDIDATES)
        successes.append(success)
    return sum(successes) / len(successes) if successes else float("nan")


def adjudicate(raw: list[dict[str, Any]], material: list[dict[str, Any]]) -> dict[str, Any]:
    material_by_id = {row["item_id"]: row for row in material}
    enriched = [{**row, **{key: material_by_id[row["item_id"]][key] for key in (
        "binding_state", "mapping_variant", "order_variant", "target_record_position"
    )}} for row in raw]

    q0_split = {split: rate([row for row in raw if row["split"] == split], "all_vocab_logits_finite") for split in SPLITS}
    q0_overall = rate(raw, "all_vocab_logits_finite")
    q0_pass = q0_overall >= THRESHOLDS["Q0_finite_rate"] and min(q0_split.values()) >= THRESHOLDS["Q0_finite_rate"]

    q1_panel: dict[str, float] = {}
    for split in SPLITS:
        for panel in PANELS:
            selected = [row for row in raw if row["split"] == split and row["panel"] == panel]
            q1_panel[f"{split}|{panel}"] = rate(selected, "correct")
    active = [row for row in enriched if row["panel"] == "active"]
    marginal_cells: dict[str, dict[str, float]] = {}
    marginal_worst: dict[str, float] = {}
    for axis in ACTIVE_AXES:
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in active:
            cells[canonical_json(row[axis])].append(row)
        values = {key: rate(cell, "correct") for key, cell in cells.items()}
        marginal_cells[axis] = values
        marginal_worst[axis] = min(values.values())
    q1_pass = min(q1_panel.values()) >= THRESHOLDS["Q1_panel_accuracy"] and min(marginal_worst.values()) >= THRESHOLDS["Q1_active_worst_marginal"]

    q2_rates = {
        split: group_success(
            [row for row in raw if row["split"] == split and row["panel"] == "active"],
            ("bundle_id",), 4, False,
        ) for split in SPLITS
    }
    q2_pass = min(q2_rates.values()) >= THRESHOLDS["Q2_active_quartet"]

    q3_rates: dict[str, float] = {}
    for split in SPLITS:
        for panel in ("matched_null", "surface_order"):
            selected = [row for row in raw if row["split"] == split and row["panel"] == panel]
            q3_rates[f"{split}|{panel}"] = group_success(selected, ("bundle_id",), 4, True)
    q3_pass = min(q3_rates.values()) >= THRESHOLDS["Q3_control_invariant_bundle"]

    pair_keys = ("split", "panel", "world_id", "target_entity", "order_variant", "mapping_variant", "binding_state")
    q4_rates: dict[str, float] = {}
    for split in SPLITS:
        for panel in PANELS:
            selected = [row for row in enriched if row["split"] == split and row["panel"] == panel]
            q4_rates[f"{split}|{panel}"] = group_success(selected, pair_keys, 2, True)
    q4_pass = min(q4_rates.values()) >= THRESHOLDS["Q4_template_pair"]

    natural = [row for row in raw if row["split"] == "natural_use"]
    q5_overall = rate(natural, "full_vocab_top1_is_gold_candidate")
    q5_panel = {panel: rate([row for row in natural if row["panel"] == panel], "full_vocab_top1_is_gold_candidate") for panel in PANELS}
    q5_pass = q5_overall >= THRESHOLDS["Q5_natural_first_token"]

    candidate_gate = q0_pass and q1_pass and q2_pass and q3_pass and q4_pass
    first_token_gate = candidate_gate and q5_pass
    return {
        "Q0": {"overall_finite_rate": q0_overall, "split_finite_rates": q0_split, "passed": q0_pass},
        "Q1": {
            "split_panel_candidate_accuracy": q1_panel,
            "active_worst_marginal_by_axis": marginal_worst,
            "active_marginal_cells": marginal_cells,
            "passed": q1_pass,
        },
        "Q2": {"active_quartet_success_by_split": q2_rates, "passed": q2_pass},
        "Q3": {"control_invariant_bundle_success": q3_rates, "passed": q3_pass},
        "Q4": {"template_pair_success": q4_rates, "passed": q4_pass},
        "Q5": {"natural_first_token_accuracy": q5_overall, "by_panel": q5_panel, "passed": q5_pass},
        "candidate_behavior_gate": candidate_gate,
        "natural_first_token_gate": first_token_gate,
        "overall_candidate_accuracy": rate(raw, "correct"),
        "tie_count": sum(row["unresolved_tie"] for row in raw),
        "nonfinite_count": sum(not row["all_vocab_logits_finite"] for row in raw),
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in raw)),
    }


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1231 final already exists")
    contract, plan, manifest, material = verify_frozen()
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    if len(raw) != EXPECTED_ROWS or summary.get("raw_digest") != digest(raw):
        raise RuntimeError("raw behavior digest mismatch")
    if {row["item_id"] for row in raw} != {row["item_id"] for row in manifest}:
        raise RuntimeError("raw behavior does not cover frozen manifest")
    ledgers = adjudicate(raw, material)
    shortcut = read_json(SHORTCUT_PATH)
    candidate_gate = bool(ledgers["candidate_behavior_gate"])
    first_token_gate = bool(ledgers["natural_first_token_gate"])
    if candidate_gate:
        status = "candidate_behavior_passed_construct_ambiguous" if shortcut["global_complement_identifiability_failure"] else "candidate_behavior_passed"
        k_statement = (
            "Qwen3 FP16 passed the frozen Q0-Q4 candidate behavior contract on the de-answer-loaded clock-compass family; "
            "the C4/bijective material does not identify target-record use because three non-target records determine the complement."
        )
        grade = "E3-BEHAVIOR-CONSTRUCT-BOUNDARY"
    else:
        status = "candidate_behavior_gate_failed"
        k_statement = "Qwen3 FP16 did not pass all frozen Q0-Q4 ledgers for the de-answer-loaded clock-compass family."
        grade = "E3-NEGATIVE-BOUNDARY"
    final: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1231.qwen3.clock_compass.final.v1",
        "created_at_utc": utc_now(),
        "status": status,
        "contract_digest": contract["contract_digest"],
        "batch_plan_digest": plan["plan_digest"],
        "shortcut_audit_digest": shortcut["shortcut_audit_digest"],
        "run_summary_digest": summary["summary_digest"],
        "raw_digest": summary["raw_digest"],
        "ledgers": ledgers,
        "shortcut_boundary": {
            "two_non_target_lookup_accuracy": shortcut["results"]["target_index_plus_two_non_targets"]["accuracy"],
            "three_non_target_lookup_accuracy": shortcut["results"]["target_index_plus_three_non_targets"]["accuracy"],
            "global_complement_identifiability_failure": shortcut["global_complement_identifiability_failure"],
            "formal_gates_rewritten": False,
        },
        "k_item": {
            "identifier": "K206",
            "evidence_grade": grade,
            "statement": k_statement,
            "scope": "Qwen3-4B; CUDA FP16; frozen Phase1229-1230 artificial English clock-compass interface; behavior only",
        },
        "authorization": {
            "candidate_behavior_object": candidate_gate,
            "natural_first_token_claim": first_token_gate,
            "target_record_specific_mechanism_claim": False,
            "hidden_state_execution_in_this_phase": False,
            "next_experiment": (
                "Phase1232 preoutput freeze of a shortcut-aware, record-indexed future-response tensor protocol"
                if candidate_gate else None
            ),
            "automatic_next_protocol_freeze": candidate_gate,
            "automatic_hidden_scan": False,
            "cross_model_run": False,
        },
        "claim_boundary": [
            "A candidate behavior pass is not evidence that the target record was read.",
            "The three non-target records deterministically reveal the missing target clock under the bijective material.",
            "C4-to-S4 expansion alone does not remove the full-complement shortcut.",
            "Q5 concerns only the first generated token, not open multi-token generation.",
            "No hidden state, attention, causal necessity, rescue, or cross-model mechanism was measured.",
        ],
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    _upstream, manifest, material = verify_upstream()
    plan = build_batch_plan(manifest)
    shortcut = build_shortcut_audit(material)
    assert plan["batch_count"] > 0
    assert shortcut["results"]["constant"]["accuracy"] == 0.25
    assert shortcut["results"]["target_index_plus_three_non_targets"]["accuracy"] == 1.0
    print(canonical_json({
        "status": "selftest_passed",
        "rows": len(manifest),
        "batches": plan["batch_count"],
        "three_non_target_accuracy": shortcut["results"]["target_index_plus_three_non_targets"]["accuracy"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "preregister", "run", "finalize"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "preregister":
        preregister()
    elif args.stage == "run":
        run_qwen3()
    elif args.stage == "finalize":
        finalize()


if __name__ == "__main__":
    main()
