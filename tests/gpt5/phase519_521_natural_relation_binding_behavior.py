#!/usr/bin/env python3
"""Run one Phase518 behavior stage for one CUDA model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase518_world_query_platform_protocol import (  # noqa: E402
    parse_free_label,
    parse_mapping_event,
    parse_natural_event,
)


PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase518_world_query_platform_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase518_frozen_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase518_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase518_world_query_platform_protocol.py"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "natural_paraphrase")
LABEL_SYSTEMS = ("mapped_ab", "mapped_01")
Z = 1.96

STAGES = {
    "calibration": {
        "phase": 519,
        "split": "calibration",
        "authorization": None,
        "out": ROOT / "tests/gpt5/result/phase519_natural_relation_binding_calibration",
    },
    "confirmation": {
        "phase": 521,
        "split": "confirmation",
        "authorization": (
            ROOT
            / "tests/gpt5/result/phase520_behavior_authorization"
            / "phase520_behavior_authorization.json"
        ),
        "out": ROOT / "tests/gpt5/result/phase521_natural_relation_binding_confirmation",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(count: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    n = len(rows)
    count = sum(bool(row[field]) for row in rows)
    lower, upper = wilson(count, n)
    return {"n": n, "count": count, "rate": count / n if n else 0.0, "lcb95": lower, "ucb95": upper}


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def verify_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase518 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase518 protocol changed after freeze")
    return contract


def allowed_contracts(stage: str, model: str) -> set[str]:
    if stage == "calibration":
        return {"relation", "binding"}
    authorization = read_json(STAGES[stage]["authorization"])
    allowed = set()
    if model in authorization["relation_models"]:
        allowed.add("relation")
    if model in authorization["binding_models"]:
        allowed.add("binding")
    return allowed


def flatten_relation(samples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"invalid relation sample for {split}")
        for variant in sample["variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "source_pair_id": sample["source_pair_id"],
                "split": split,
                "pair_index": sample["pair_index"],
                "truth_value": sample["truth_value"],
                "world_topology": sample["world_topology"],
                "query_pattern": sample["query_pattern"],
                "length_control": sample["length_control"],
                "fact_order": sample["fact_order"],
                "relation_verb": sample["relation_verb"],
                "surface": variant["surface"],
                "candidate_prompt": variant["candidate_prompt"],
                "natural_prompt": variant["natural_prompt"],
            })
    return rows


def flatten_binding(samples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"invalid binding sample for {split}")
        for variant in sample["variants"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "mapping_reversal_id": sample["mapping_reversal_id"],
                "mapping_probe_id": sample["mapping_probe_id"],
                "split": split,
                "truth_value": sample["truth_value"],
                "label_system": sample["label_system"],
                "mapping_flip": sample["mapping_flip"],
                "mapping_order": sample["mapping_order"],
                "mapping_template": sample["mapping_template"],
                "holding_symbol": sample["holding_symbol"],
                "failing_symbol": sample["failing_symbol"],
                "expected_symbol": sample["expected_symbol"],
                "surface": variant["surface"],
                "candidate_prompt": variant["candidate_prompt"],
                "free_prompt": variant["free_prompt"],
                "mapping_prompt": variant["mapping_prompt"],
            })
    return rows


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"candidate {text!r} is not one token: {ids}")
    return int(ids[0])


def candidate_token(label_system: str, symbol: str) -> str:
    if label_system == "mapped_ab":
        return f" {symbol}"
    return symbol


def score_candidates(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    prompt_field: str,
    candidates: Callable[[dict[str, Any]], tuple[str, str]],
    batch_size: int,
    stage_label: str,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[candidates(row)].append(row)
    tokenizer.padding_side = "left"
    completed = 0
    for (positive, negative), group in sorted(groups.items()):
        positive_id = single_token_id(tokenizer, positive)
        negative_id = single_token_id(tokenizer, negative)
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            encoded = tokenizer(
                [row[prompt_field] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :].float()
            positive_scores = logits[:, positive_id]
            negative_scores = logits[:, negative_id]
            normalizer = torch.logsumexp(logits, dim=-1)
            positive_probs = torch.exp(positive_scores - normalizer)
            negative_probs = torch.exp(negative_scores - normalizer)
            for index, row in enumerate(batch):
                prediction = bool(positive_scores[index] > negative_scores[index])
                row.update({
                    "candidate_margin_positive_minus_negative": float(positive_scores[index] - negative_scores[index]),
                    "candidate_prediction": prediction,
                    "candidate_correct": prediction == row["truth_value"],
                    "candidate_probability_mass": float(positive_probs[index] + negative_probs[index]),
                    "non_candidate_probability_mass": float(1.0 - positive_probs[index] - negative_probs[index]),
                })
            completed += len(batch)
            if completed == len(rows) or completed % 512 < len(batch):
                log(f"{stage_label} candidate {completed}/{len(rows)}")


def score_binding_candidates(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        symbols = ("A", "B") if row["label_system"] == "mapped_ab" else ("0", "1")
        groups[(candidate_token(row["label_system"], symbols[0]), candidate_token(row["label_system"], symbols[1]))].append(row)
    tokenizer.padding_side = "left"
    completed = 0
    for (left, right), group in sorted(groups.items()):
        left_id = single_token_id(tokenizer, left)
        right_id = single_token_id(tokenizer, right)
        left_symbol = left.strip()
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            encoded = tokenizer(
                [row["candidate_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :].float()
            left_scores = logits[:, left_id]
            right_scores = logits[:, right_id]
            normalizer = torch.logsumexp(logits, dim=-1)
            mass = torch.exp(left_scores - normalizer) + torch.exp(right_scores - normalizer)
            for index, row in enumerate(batch):
                prediction = left_symbol if left_scores[index] > right_scores[index] else right.strip()
                row.update({
                    "candidate_prediction_symbol": prediction,
                    "candidate_correct": prediction == row["expected_symbol"],
                    "candidate_margin_left_minus_right": float(left_scores[index] - right_scores[index]),
                    "candidate_probability_mass": float(mass[index]),
                    "non_candidate_probability_mass": float(1.0 - mass[index]),
                })
            completed += len(batch)
            if completed == len(rows) or completed % 512 < len(batch):
                log(f"binding candidate {completed}/{len(rows)}")


def generate_texts(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    prompt_field: str,
    output_field: str,
    parser: Callable[[str], dict[str, Any]],
    parsed_prefix: str,
    batch_size: int,
    max_new_tokens: int,
    stage_label: str,
) -> None:
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        encoded = tokenizer(
            [row[prompt_field] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        prompt_width = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for index, row in enumerate(batch):
            text = tokenizer.decode(generated[index, prompt_width:], skip_special_tokens=True)
            parsed = parser(text)
            row[output_field] = text
            row.update({f"{parsed_prefix}_{key}": value for key, value in parsed.items()})
        if start == 0 or start + len(batch) == len(rows) or (start // batch_size) % 16 == 15:
            log(f"{stage_label} generation {min(start + len(batch), len(rows))}/{len(rows)}")


def paired_rate(
    rows: list[dict[str, Any]],
    field: str,
    group_field: str,
    expected_count: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(row)
    records = []
    for key, items in groups.items():
        if len(items) != expected_count:
            raise RuntimeError(f"incomplete group {key}: {len(items)}")
        records.append({"all_correct": all(bool(item[field]) for item in items)})
    return rate(records, "all_correct")


def relation_summary(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        row["natural_event_correct"] = (
            row["natural_semantic_value"] is not None
            and row["natural_semantic_value"] == row["truth_value"]
        )
        row["natural_unrecoverable"] = not row["natural_recoverable"]
    natural_by_surface = {
        surface: rate([row for row in rows if row["surface"] == surface], "natural_event_correct")
        for surface in SURFACES
    }
    candidate_by_surface = {
        surface: rate([row for row in rows if row["surface"] == surface], "candidate_correct")
        for surface in SURFACES
    }
    natural_intersection = paired_rate(rows, "natural_event_correct", "sample_id", 2)
    candidate_intersection = paired_rate(rows, "candidate_correct", "sample_id", 2)
    natural_four_way = paired_rate(rows, "natural_event_correct", "source_pair_id", 4)
    candidate_four_way = paired_rate(rows, "candidate_correct", "source_pair_id", 4)
    unrecoverable = rate(rows, "natural_unrecoverable")
    gate = contract["gates"]["natural_relation"]
    passed = (
        all(item["lcb95"] >= gate["surface_lcb95_min"] for item in natural_by_surface.values())
        and natural_intersection["lcb95"] >= gate["surface_intersection_lcb95_min"]
        and natural_four_way["lcb95"] >= gate["four_way_lcb95_min"]
        and all(item["lcb95"] >= gate["candidate_surface_lcb95_min"] for item in candidate_by_surface.values())
        and unrecoverable["ucb95"] <= gate["unrecoverable_ucb95_max"]
    )
    return {
        "natural_by_surface": natural_by_surface,
        "natural_surface_intersection": natural_intersection,
        "natural_four_way": natural_four_way,
        "natural_unrecoverable": unrecoverable,
        "candidate_by_surface": candidate_by_surface,
        "candidate_surface_intersection": candidate_intersection,
        "candidate_four_way": candidate_four_way,
        "mean_candidate_probability_mass": mean(rows, "candidate_probability_mass"),
        "natural_event_types": dict(sorted(Counter(row["natural_event_type"] for row in rows).items())),
        "gate_pass": passed,
    }


def mapping_reversal_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mapping_reversal_id"], row["surface"])].append(row)
    records = []
    for key, items in groups.items():
        if len(items) != 2 or {item["mapping_flip"] for item in items} != {False, True}:
            raise RuntimeError(f"incomplete reversal group {key}")
        records.append({"all_correct": all(item["candidate_correct"] for item in items)})
    return rate(records, "all_correct")


def binding_summary(
    rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    for row in rows:
        row["free_output_correct"] = row["free_symbol"] == row["expected_symbol"]
        row["free_output_unrecoverable"] = not row["free_recoverable"]
    for row in mapping_rows:
        row["mapping_comprehension_correct"] = row["mapping_symbol"] == row["holding_symbol"]
    mapping_report = rate(mapping_rows, "mapping_comprehension_correct")
    by_label = {}
    for label_system in LABEL_SYSTEMS:
        selected = [row for row in rows if row["label_system"] == label_system]
        candidate_by_surface = {
            surface: rate([row for row in selected if row["surface"] == surface], "candidate_correct")
            for surface in SURFACES
        }
        candidate_by_flip = {
            str(flip).lower(): rate(
                [row for row in selected if row["mapping_flip"] is flip], "candidate_correct"
            )
            for flip in (False, True)
        }
        by_label[label_system] = {
            "candidate_by_surface": candidate_by_surface,
            "candidate_by_mapping_flip": candidate_by_flip,
            "candidate_surface_intersection": paired_rate(selected, "candidate_correct", "sample_id", 2),
            "mapping_reversal": mapping_reversal_rate(selected),
            "strict_free_output": rate(selected, "free_output_correct"),
            "free_output_unrecoverable": rate(selected, "free_output_unrecoverable"),
            "mean_non_candidate_probability_mass": mean(selected, "non_candidate_probability_mass"),
            "free_event_types": dict(sorted(Counter(row["free_event_type"] for row in selected).items())),
        }
    gate = contract["gates"]["binding"]
    passed = (
        mapping_report["lcb95"] >= gate["mapping_comprehension_lcb95_min"]
        and all(
            all(item["lcb95"] >= gate["candidate_surface_lcb95_min"] for item in report["candidate_by_surface"].values())
            and report["mapping_reversal"]["lcb95"] >= gate["mapping_reversal_lcb95_min"]
            and report["strict_free_output"]["lcb95"] >= gate["strict_free_output_lcb95_min"]
            and report["mean_non_candidate_probability_mass"] <= gate["mean_non_candidate_mass_max"]
            for report in by_label.values()
        )
    )
    return {
        "mapping_comprehension": mapping_report,
        "mapping_comprehension_by_label": {
            label: rate([row for row in mapping_rows if row["label_system"] == label], "mapping_comprehension_correct")
            for label in LABEL_SYSTEMS
        },
        "mapping_event_types": dict(sorted(Counter(row["mapping_event_type"] for row in mapping_rows).items())),
        "by_label_system": by_label,
        "gate_pass": passed,
    }


def run_stage(model_name: str, stage: str, batch_size: int, max_new_tokens: int, use_8bit: bool) -> None:
    contract = verify_contract()
    config = STAGES[stage]
    split = config["split"]
    allowed = allowed_contracts(stage, model_name)
    out_dir = config["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"phase{config['phase']}_{model_name}_summary.json"
    started = time.monotonic()
    if not allowed:
        summary = {
            "schema_version": f"phase{config['phase']}_natural_relation_binding_behavior.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "no_authorized_contracts",
            "model": model_name,
            "stage": stage,
            "split": split,
            "row_count": 0,
            "passed_contracts": [],
            "contract_summaries": {},
            "cuda_used": False,
            "model_weights_loaded": False,
            "sealed_split_read": False,
            "runtime_seconds": time.monotonic() - started,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
        return

    relation_rows = []
    binding_rows = []
    if "relation" in allowed:
        relation_path = PROTOCOL_DIR / f"phase518_{split}_relation.jsonl"
        expected = contract["split_files"][f"{split}_relation"]["sha256"]
        if sha256_file(relation_path) != expected:
            raise RuntimeError("relation split hash drift")
        relation_rows = flatten_relation(read_jsonl(relation_path), split)
    if "binding" in allowed:
        binding_path = PROTOCOL_DIR / f"phase518_{split}_binding.jsonl"
        expected = contract["split_files"][f"{split}_binding"]["sha256"]
        if sha256_file(binding_path) != expected:
            raise RuntimeError("binding split hash drift")
        binding_rows = flatten_binding(read_jsonl(binding_path), split)

    loaded = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase519/521 requires CUDA")
        loaded, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        if relation_rows:
            score_candidates(
                loaded,
                tokenizer,
                device,
                relation_rows,
                "candidate_prompt",
                lambda _row: (" true", " false"),
                batch_size,
                "relation",
            )
            generate_texts(
                loaded,
                tokenizer,
                device,
                relation_rows,
                "natural_prompt",
                "generated_natural_text",
                parse_natural_event,
                "natural",
                batch_size,
                max_new_tokens,
                "relation",
            )
        mapping_rows = []
        if binding_rows:
            score_binding_candidates(loaded, tokenizer, device, binding_rows, batch_size)
            generate_texts(
                loaded,
                tokenizer,
                device,
                binding_rows,
                "free_prompt",
                "generated_free_text",
                parse_free_label,
                "free",
                batch_size,
                4,
                "binding-free",
            )
            unique_mapping: dict[str, dict[str, Any]] = {}
            for row in binding_rows:
                unique_mapping.setdefault(row["mapping_probe_id"], {
                    "mapping_probe_id": row["mapping_probe_id"],
                    "label_system": row["label_system"],
                    "mapping_flip": row["mapping_flip"],
                    "mapping_template": row["mapping_template"],
                    "holding_symbol": row["holding_symbol"],
                    "mapping_prompt": row["mapping_prompt"],
                })
            mapping_rows = list(unique_mapping.values())
            generate_texts(
                loaded,
                tokenizer,
                device,
                mapping_rows,
                "mapping_prompt",
                "generated_mapping_text",
                parse_mapping_event,
                "mapping",
                batch_size,
                max_new_tokens,
                "binding-mapping",
            )

        contract_summaries = {}
        passed_contracts = []
        if relation_rows:
            contract_summaries["R_natural"] = relation_summary(relation_rows, contract)
            if contract_summaries["R_natural"]["gate_pass"]:
                passed_contracts.append("R_natural")
            write_jsonl(out_dir / f"phase{config['phase']}_{model_name}_relation_rows.jsonl", relation_rows)
        if binding_rows:
            contract_summaries["B_ledger"] = binding_summary(binding_rows, mapping_rows, contract)
            if contract_summaries["B_ledger"]["gate_pass"]:
                passed_contracts.append("B_ledger")
            write_jsonl(out_dir / f"phase{config['phase']}_{model_name}_binding_rows.jsonl", binding_rows)
            write_jsonl(out_dir / f"phase{config['phase']}_{model_name}_mapping_rows.jsonl", mapping_rows)
        summary = {
            "schema_version": f"phase{config['phase']}_natural_relation_binding_behavior.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "model": model_name,
            "stage": stage,
            "split": split,
            "row_count": len(relation_rows) + len(binding_rows),
            "relation_row_count": len(relation_rows),
            "binding_row_count": len(binding_rows),
            "mapping_probe_count": len(mapping_rows),
            "passed_contracts": passed_contracts,
            "contract_summaries": contract_summaries,
            "cuda_used": True,
            "model_weights_loaded": True,
            "sealed_split_read": False,
            "runtime_seconds": time.monotonic() - started,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
    finally:
        if loaded is not None:
            release_model(loaded)
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--stage", choices=tuple(STAGES), default="calibration")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    run_stage(args.model, args.stage, args.batch_size, args.max_new_tokens, args.use_8bit)


if __name__ == "__main__":
    main()
