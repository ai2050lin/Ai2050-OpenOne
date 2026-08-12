#!/usr/bin/env python3
"""Run Phase1092 behavior calibration, one unquantized FP16 model at a time."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as generation
from phase1065_multimode_response_atlas_scan import strict_generated_answer
import phase1083_same_carrier_attribute_behavior as behavior_tools
import phase1092_natural_bilingual_attribute_protocol as protocol


BATCH_SIZE = {"qwen3": 32, "glm4": 32, "deepseek7b": 32}


def generation_selection(rows: list[dict]) -> list[dict]:
    """Select balanced natural generations without inflating decode cost."""
    selected = []
    by_cell_split: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell_split[(str(row["cell"]), str(row["split"]))].append(row)
    for key in sorted(by_cell_split):
        panel = by_cell_split[key]
        unit_ids = sorted({str(row["unit_id"]) for row in panel})
        choices = (
            (unit_ids[0], "t0_cactive_m0_q0_w0"),
            (unit_ids[1], "t0_cfield_null_m1_q0_w0"),
        )
        for unit_id, state in choices:
            row = next(
                value for value in panel
                if value["unit_id"] == unit_id and value["state"] == state
            )
            selected.append({**row, "semantic_case_index": int(row["case_index"])})
    return selected


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1092 static protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    started = time.time()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")

        by_length: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        candidate_detail = []
        totals: Counter = Counter()
        finite_counts: Counter = Counter()
        hits: Counter = Counter()
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), BATCH_SIZE[model_name]):
                    batch = panel[start:start + BATCH_SIZE[model_name]]
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits = output.logits[:, -1, :].float()
                    for slot, row in enumerate(batch):
                        values = logits[slot]
                        scores = {}
                        for answer_class in ("a0", "a1"):
                            ids = torch.tensor(
                                row["candidate_first_token_ids"][answer_class],
                                dtype=torch.long,
                                device=values.device,
                            )
                            scores[answer_class] = float(values[ids].max().item())
                        expected = str(row["expected_class"])
                        other = "a1" if expected == "a0" else "a0"
                        margin = scores[expected] - scores[other]
                        finite = all(math.isfinite(value) for value in scores.values()) \
                            and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        key = (
                            row["attribute"], row["operation"], row["surface"],
                            row["world"], row["panel"], row["split"],
                        )
                        totals[key] += 1
                        finite_counts[key] += int(finite)
                        hits[key] += int(hit)
                        candidate_detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "attribute": row["attribute"],
                            "operation": row["operation"],
                            "surface": row["surface"],
                            "world": row["world"],
                            "split": row["split"],
                            "panel": row["panel"],
                            "state": row["state"],
                            "expected_class": expected,
                            "target_answer": row["target_answer"],
                            "scores": {
                                name: value if math.isfinite(value) else None
                                for name, value in scores.items()
                            },
                            "margin": margin if math.isfinite(margin) else None,
                            "finite": finite,
                            "hit": hit,
                        })
                    del output, logits, input_ids, attention_mask
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "candidate_length_complete": length,
                }), flush=True)

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        generation_rows = generation_selection(rows)
        generated = generation.generate_case_outputs(
            model,
            device,
            generation_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["generation_steps"]),
        )
        generation_detail = []
        gen_totals: Counter = Counter()
        gen_hits: Counter = Counter()
        for row in generation_rows:
            output_ids = generated[int(row["case_index"])]
            generated_text = strict_generated_answer(tokenizer, output_ids, eos_ids)
            distractor = row["answer_labels"][1 - int(row["answer_index"])]
            classification = behavior_tools.classify_generation(
                generated_text, row["target_answer"], distractor
            )
            key = (row["attribute"], row["surface"], row["panel"])
            gen_totals[key] += 1
            gen_hits[key] += int(classification["target_before_distractor"])
            generation_detail.append({
                "case_index": int(row["case_index"]),
                "unit_id": row["unit_id"],
                "attribute": row["attribute"],
                "operation": row["operation"],
                "surface": row["surface"],
                "world": row["world"],
                "split": row["split"],
                "panel": row["panel"],
                "target_answer": row["target_answer"],
                "distractor_answer": distractor,
                "generated_token_ids": [int(value) for value in output_ids],
                "generated_text": generated_text,
                **classification,
            })

        per_cell = {}
        for attribute in protocol.ATTRIBUTES:
            operations = [
                value for value in protocol.OPERATIONS
                if value.startswith(f"{attribute}_")
            ]
            for operation in operations:
                for surface in protocol.SURFACES:
                    for world in protocol.BASE_WORLDS:
                        for panel in protocol.PANELS:
                            keys = [
                                (attribute, operation, surface, world, panel, split)
                                for split in protocol.SPLITS
                            ]
                            total = sum(totals[key] for key in keys)
                            finite = sum(finite_counts[key] for key in keys)
                            hit = sum(hits[key] for key in keys)
                            per_cell[
                                f"{attribute}__{operation}__{surface}__{world}__{panel}"
                            ] = {
                                "count": total,
                                "finite_fraction": finite / total if total else 0.0,
                                "accuracy": hit / total if total else 0.0,
                            }
        generation_by_attribute_surface = {}
        for attribute in protocol.ATTRIBUTES:
            for surface in protocol.SURFACES:
                panels = {}
                for panel in protocol.PANELS:
                    key = (attribute, surface, panel)
                    panels[panel] = {
                        "count": gen_totals[key],
                        "target_before_distractor_accuracy": (
                            gen_hits[key] / gen_totals[key] if gen_totals[key] else 0.0
                        ),
                    }
                generation_by_attribute_surface[
                    f"{attribute}__{surface}"
                ] = panels

        result = {
            "schema_version": "phase1092_behavior_result.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_case_count": len(candidate_detail),
            "candidate_finite_fraction": (
                sum(finite_counts.values()) / sum(totals.values())
            ),
            "generation_case_count": len(generation_detail),
            "per_cell": per_cell,
            "generation_by_attribute_surface": generation_by_attribute_surface,
            "elapsed_seconds": time.time() - started,
        }
        result["result_digest"] = protocol.digest(result)
        pilot_root = protocol.OUT_ROOT / "pilot"
        protocol.write_jsonl(
            pilot_root / f"candidate.{model_name}.jsonl", candidate_detail
        )
        protocol.write_jsonl(
            pilot_root / f"generation.{model_name}.jsonl", generation_detail
        )
        protocol.write_json(pilot_root / f"{model_name}.json", result)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "status": "behavior_complete",
            "candidate_finite_fraction": result["candidate_finite_fraction"],
            "elapsed_seconds": result["elapsed_seconds"],
            "result_digest": result["result_digest"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
