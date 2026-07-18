#!/usr/bin/env python3
"""Qualify Phase526 behavior before any role-normalized physical analysis."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
OPEN_SPLITS = (
    "world_fit",
    "world_entity_prediction",
    "world_relation_prediction",
    "bridge_open_prediction",
)
PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase526_role_normalized_world_geometry_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase526_frozen_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase526_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase526_role_normalized_world_geometry_protocol.py"
OUT_DIR = ROOT / "tests/gpt5/result/phase527_world_geometry_behavior_qualification"
Z = 1.96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wilson(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * ((p * (1 - p) + Z * Z / (4 * n)) / n) ** 0.5 / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(values: list[bool]) -> dict[str, Any]:
    n = len(values)
    count = sum(values)
    lower, upper = wilson(count, n)
    return {"n": n, "count": count, "rate": count / n if n else 0.0, "lcb95": lower, "ucb95": upper}


def parse_first_event(text: str) -> bool | None:
    normalized = " ".join(text.lstrip().split())
    match = re.match(
        r"^(The statement is supported\.|The statement is contradicted\.)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).lower() == "the statement is supported."


def verify() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase526 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase526 protocol source drift")
    for split in OPEN_SPLITS:
        spec = contract["split_files"][split]
        path = ROOT / spec["path"]
        if spec["sealed"] or sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"Phase526 split drift: {split}")
    return contract


def register_anchor_token_audit(tokenizer: Any, row: dict[str, Any]) -> bool:
    register_end = max(
        int(value)
        for key, value in row["role_char_ends"].items()
        if key.startswith("register_slot_")
    )
    prefix = tokenizer.encode(row["natural_prompt"][:register_end], add_special_tokens=True)
    full = tokenizer.encode(row["natural_prompt"], add_special_tokens=True)
    return len(full) >= len(prefix) and full[: len(prefix)] == prefix


def split_report(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    overall = rate([bool(row["first_event_correct"]) for row in rows])
    recoverable = rate([bool(row["first_event_recoverable"]) for row in rows])
    unrecoverable = rate([not bool(row["first_event_recoverable"]) for row in rows])
    strict = rate([bool(row["strict_whole_response_correct"]) for row in rows])
    by_surface = {}
    for surface in sorted({row["surface"] for row in rows}):
        local = [row for row in rows if row["surface"] == surface]
        by_surface[surface] = rate([bool(row["first_event_correct"]) for row in local])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_pair_id"]].append(row)
    four_way_values = [
        len(group) == 4 and all(bool(row["first_event_correct"]) for row in group)
        for group in groups.values()
    ]
    four_way = rate(four_way_values)
    prefix_exact = rate([bool(row["register_anchor_token_exact"]) for row in rows])
    gate_pass = (
        overall["lcb95"] >= float(gate["overall_lcb95_min"])
        and all(item["lcb95"] >= float(gate["surface_lcb95_min"]) for item in by_surface.values())
        and four_way["lcb95"] >= float(gate["four_way_lcb95_min"])
        and unrecoverable["ucb95"] <= float(gate["unrecoverable_ucb95_max"])
        and prefix_exact["rate"] == 1.0
    )
    return {
        "overall_first_event": overall,
        "recoverable": recoverable,
        "unrecoverable": unrecoverable,
        "strict_whole_response": strict,
        "by_surface": by_surface,
        "four_way_source_pair": four_way,
        "register_anchor_token_exact": prefix_exact,
        "gate_pass": gate_pass,
    }


def run_model(model_name: str, batch_size: int, max_new_tokens: int, use_8bit: bool) -> Path:
    contract = verify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"phase527_{model_name}_behavior_summary.json"
    all_rows = []
    for split in OPEN_SPLITS:
        all_rows.extend(read_jsonl(ROOT / contract["split_files"][split]["path"]))
    loaded = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase527 requires CUDA")
        loaded, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        tokenizer.padding_side = "left"
        for row in all_rows:
            row["register_anchor_token_exact"] = register_anchor_token_audit(tokenizer, row)
        for start in range(0, len(all_rows), batch_size):
            batch = all_rows[start : start + batch_size]
            encoded = tokenizer(
                [row["natural_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = loaded.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for index, row in enumerate(batch):
                text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
                event = parse_first_event(text)
                expected = (
                    "The statement is supported." if row["truth_value"] else "The statement is contradicted."
                )
                row["generated_natural_text"] = text
                row["first_event_value"] = event
                row["first_event_recoverable"] = event is not None
                row["first_event_correct"] = event is not None and event == row["truth_value"]
                row["strict_whole_response_correct"] = " ".join(text.strip().split()) == expected
            del generated, encoded
            if start == 0 or start + len(batch) == len(all_rows) or (start // batch_size) % 16 == 15:
                print(f"[{time.strftime('%H:%M:%S')}] {model_name} {min(start + len(batch), len(all_rows))}/{len(all_rows)}", flush=True)

        reports = {}
        for split in OPEN_SPLITS:
            local = [row for row in all_rows if row["split"] == split]
            reports[split] = split_report(local, contract["behavior_gate"])
        authorized = all(report["gate_pass"] for report in reports.values())
        rows_path = OUT_DIR / f"phase527_{model_name}_behavior_rows.jsonl"
        write_jsonl(rows_path, all_rows)
        summary = {
            "schema_version": "phase527_world_geometry_behavior_qualification.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "model": model_name,
            "row_count": len(all_rows),
            "split_reports": reports,
            "physical_authorized": authorized,
            "cuda_used": True,
            "model_weights_loaded": True,
            "register_anchor_token_exact_all": all(
                bool(row["register_anchor_token_exact"]) for row in all_rows
            ),
            "sealed_split_read": False,
            "rows_path": str(rows_path.relative_to(ROOT)),
            "rows_sha256": sha256_file(rows_path),
            "runtime_seconds": time.monotonic() - started,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
        return summary_path
    finally:
        if loaded is not None:
            release_model(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> Path:
    contract = verify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model in MODELS:
        path = OUT_DIR / f"phase527_{model}_behavior_summary.json"
        if not path.exists():
            raise RuntimeError(f"missing model summary: {model}")
        summaries[model] = read_json(path)
    authorized = [model for model in MODELS if summaries[model]["physical_authorized"]]
    output = OUT_DIR / "phase527_physical_authorization.json"
    payload = {
        "schema_version": "phase527_world_geometry_physical_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_required_model_order",
        "models_in_required_order": list(MODELS),
        "physical_authorized_models": authorized,
        "model_authorization": {model: summaries[model]["physical_authorized"] for model in MODELS},
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "sealed_split_read": False,
        "evidence_boundary": "Behavior qualification authorizes measurement only, not physical or causal claims.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS + ("aggregate",))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    if args.model == "aggregate":
        aggregate()
    else:
        run_model(args.model, args.batch_size, args.max_new_tokens, args.use_8bit)


if __name__ == "__main__":
    main()
