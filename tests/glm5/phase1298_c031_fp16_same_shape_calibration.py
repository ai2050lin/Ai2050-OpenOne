#!/usr/bin/env python3
"""Phase 1298: same-shape FP16 hidden-state camera calibration for C031."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))
from model_utils import MODEL_CONFIGS  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402


PHASE = 1298
CAMPAIGN = "C031"
SCRIPT = Path(__file__).resolve()
AUDITOR = TEST_ROOT / "phase1298_c031_fp16_same_shape_calibration_audit.py"
PARENT = TEST_ROOT / "result/phase1297_c031_event_interval_contract"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
MATERIAL = PARENT / "material/frozen_event_interval_cases.jsonl"
OUT = TEST_ROOT / "result/phase1298_c031_fp16_same_shape_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_calibration_manifest.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
POSTAUDIT = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/calibration_arrays.npz"
RUN_META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/calibration_summary.json"
TOLERANCE = OUT / "protocol/frozen_empirical_tolerance.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

SYSTEM_PROMPT = "Use only the supplied catalog. Reply exactly as requested and do not explain."
DEPTHS = tuple(range(37))
ROLES = ("record_slot0_entity", "record_slot0_value")
CALIBRATION_THRESHOLDS = {
    "case_count_min": 96,
    "finite_fraction_min": 1.0,
    "exact_duplicate_relative_max": 1e-6,
    "same_batch_prefix_relative_max": 0.0025,
    "cross_composition_prefix_relative_max": 0.005,
    "derived_tolerance_multiplier": 4.0,
    "derived_tolerance_floor": 1e-6,
    "derived_tolerance_cap": 0.01,
}
EPS = 1e-12
BATCH_GROUPS = 4


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def overlap(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    selected = [i for i, (a, b) in enumerate(offsets) if b > left and a < right and b > a]
    if not selected:
        raise RuntimeError((left, right))
    return selected


def state_spec(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    text = render(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    base = text.find(row["candidate_prompt"])
    record = row["typed_spans"]["records"][0]
    spans = {
        "record_slot0_entity": record["entity_spans"][0],
        "record_slot0_value": record["queried_attribute_value_spans"][0],
    }
    positions = {}
    for role, (left, right) in spans.items():
        positions[role] = overlap(offsets, base + left, base + right)[-1]
    return {"case_id": row["case_id"], "ids": ids, "positions": positions, "input_digest": digest(ids)}


def create_manifest(tokenizer: Any, material: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["profile_index"], row["attribute"], row["surface"], row["panel"], row["binding_state"], row["candidate_order"]): row for row in material if row["partition"] == "discovery"}
    result = []
    for profile in range(8):
        for attribute in ("color", "material", "location", "size", "shape", "status"):
            for surface in ("catalog_prose", "inventory_ledger"):
                active = index[(profile, attribute, surface, "active", 0, 0)]
                null = index[(profile, attribute, surface, "matched_null", 0, 0)]
                active_spec, null_spec = state_spec(tokenizer, active), state_spec(tokenizer, null)
                record_end_a = active["typed_spans"]["records"][0]["queried_attribute_value_spans"][0][1]
                record_end_n = null["typed_spans"]["records"][0]["queried_attribute_value_spans"][0][1]
                if active["candidate_prompt"][:record_end_a] != null["candidate_prompt"][:record_end_n]:
                    raise RuntimeError("causal prefix mismatch")
                result.append({
                    "calibration_id": f"p{profile:02d}|{attribute}|{surface}",
                    "profile_index": profile,
                    "attribute": attribute,
                    "surface": surface,
                    "active": active_spec,
                    "matched_null": null_spec,
                    "prefix_identity_through_record_value": True,
                })
    return result


def preregister(force: bool) -> None:
    if load(PARENT_FINAL).get("authorization") != "phase1298_numerical_calibration_only" or not load(PARENT_AUDIT).get("all_checks_passed"):
        raise RuntimeError("Phase1297 authorization missing")
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} exists")
    if OUT.exists():
        shutil.rmtree(OUT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True)
    manifest = create_manifest(tokenizer, read_jsonl(MATERIAL))
    write_jsonl(MANIFEST, manifest)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "schema_version": "phase1298.c031.calibration.v1",
        "object": "FP16 hidden-state reproducibility under frozen equal-shape execution",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "formal_run_budget": 1,
        "calibration_cases": len(manifest),
        "roles": list(ROLES),
        "depths": list(DEPTHS),
        "batch_groups": BATCH_GROUPS,
        "execution": {
            "global_fixed_sequence_length": True,
            "fixed_batch_size": 3 * BATCH_GROUPS,
            "explicit_position_ids": True,
            "comparisons": [
                "exact duplicate within the same batch",
                "causally identical active/matched-null prefix within the same batch",
                "same active prompt under a different fixed-shape batch composition",
            ],
        },
        "thresholds": CALIBRATION_THRESHOLDS,
        "tolerance_rule": "tau=max(floor,min(cap,multiplier*max_observed_noise))",
        "success": "all finite and all three frozen maximum-noise thresholds pass",
        "failure": "close_c031_as_numerically_unqualified",
        "success_authorization": "freeze_tau_and_authorize_phase1299_behavior_only",
        "dependencies": {
            "parent_protocol": sha(PARENT_PROTOCOL), "parent_final": sha(PARENT_FINAL),
            "parent_audit": sha(PARENT_AUDIT), "material": sha(MATERIAL), "manifest": sha(MANIFEST),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "model_weights_loaded": False,
    }
    protocol = {**timeless, "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_digest": digest(timeless)}
    save(PROTOCOL, protocol)
    print(canonical({"cases": len(manifest), "manifest": sha(MANIFEST), "digest": protocol["protocol_digest"]}))


def pack(tokenizer: Any, specs: list[dict[str, Any]], max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    ids = torch.full((len(specs), max_length), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(specs), max_length), dtype=torch.long, device=device)
    offsets = []
    for i, spec in enumerate(specs):
        values = spec["ids"]
        offset = max_length - len(values)
        offsets.append(offset)
        ids[i, offset:] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, offset:] = 1
    pos = mask.cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 0)
    return ids, mask, pos, offsets


def relative(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(left.float() - right.float())
    denominator = 0.5 * (torch.linalg.vector_norm(left.float()) + torch.linalg.vector_norm(right.float()))
    return float((numerator / (denominator + EPS)).item())


@torch.inference_mode()
def run() -> None:
    protocol = load(PROTOCOL)
    preaudit = load(PREAUDIT)
    if preaudit.get("authorization") != "run_phase1298_once" or not preaudit.get("all_checks_passed"):
        raise RuntimeError("preaudit authorization missing")
    if any(path.exists() for path in (ARRAYS, RUN_META, SUMMARY, TOLERANCE, FINAL, COMPLETE)):
        raise RuntimeError("formal run already consumed")
    manifest = read_jsonl(MANIFEST)
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(qa)
        max_length = max(len(side["ids"]) for item in manifest for side in (item["active"], item["matched_null"]))
        exact = np.empty((len(manifest), len(DEPTHS), len(ROLES)), dtype=np.float32)
        prefix = np.empty_like(exact)
        cross = np.empty_like(exact)
        baseline: dict[str, np.ndarray] = {}
        supports_last = "logits_to_keep" in inspect.signature(model.forward).parameters

        def forward_specs(specs: list[dict[str, Any]]) -> tuple[Any, list[int]]:
            ids, mask, pos, offsets = pack(tokenizer, specs, max_length, device)
            kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "use_cache": False, "output_hidden_states": True, "return_dict": True}
            if supports_last:
                kwargs["logits_to_keep"] = 1
            return model(**kwargs), offsets

        for start in range(0, len(manifest), BATCH_GROUPS):
            group = manifest[start:start + BATCH_GROUPS]
            specs = []
            for item in group:
                specs.extend((item["active"], item["active"], item["matched_null"]))
            output, offsets = forward_specs(specs)
            for local, item in enumerate(group):
                a, dup, null = 3 * local, 3 * local + 1, 3 * local + 2
                per_depth = np.empty((len(DEPTHS), len(ROLES), 1), dtype=np.float32)
                for depth, hidden in enumerate(output.hidden_states):
                    for role_index, role in enumerate(ROLES):
                        pa = offsets[a] + item["active"]["positions"][role]
                        pd = offsets[dup] + item["active"]["positions"][role]
                        pn = offsets[null] + item["matched_null"]["positions"][role]
                        exact[start + local, depth, role_index] = relative(hidden[a, pa], hidden[dup, pd])
                        prefix[start + local, depth, role_index] = relative(hidden[a, pa], hidden[null, pn])
                        per_depth[depth, role_index, 0] = 0.0
                baseline[item["calibration_id"]] = np.stack([
                    torch.stack([hidden[a, offsets[a] + item["active"]["positions"][role]].float().cpu() for role in ROLES]).numpy()
                    for hidden in output.hidden_states
                ])
            del output

        reversed_manifest = list(reversed(manifest))
        for start in range(0, len(reversed_manifest), BATCH_GROUPS):
            group = reversed_manifest[start:start + BATCH_GROUPS]
            specs = []
            for item in group:
                specs.extend((item["matched_null"], item["active"], item["matched_null"]))
            output, offsets = forward_specs(specs)
            for local, item in enumerate(group):
                a = 3 * local + 1
                original_index = next(index for index, entry in enumerate(manifest) if entry["calibration_id"] == item["calibration_id"])
                for depth, hidden in enumerate(output.hidden_states):
                    for role_index, role in enumerate(ROLES):
                        p = offsets[a] + item["active"]["positions"][role]
                        reference = torch.from_numpy(baseline[item["calibration_id"]][depth, role_index]).to(hidden.device)
                        cross[original_index, depth, role_index] = relative(hidden[a, p], reference)
            del output

        ARRAYS.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ARRAYS, exact_duplicate=exact, same_batch_prefix=prefix, cross_composition=cross, depths=np.asarray(DEPTHS), roles=np.asarray(ROLES))
        maxima = {
            "exact_duplicate_relative_max": float(exact.max()),
            "same_batch_prefix_relative_max": float(prefix.max()),
            "cross_composition_prefix_relative_max": float(cross.max()),
        }
        max_noise = max(maxima.values())
        tau = max(CALIBRATION_THRESHOLDS["derived_tolerance_floor"], min(CALIBRATION_THRESHOLDS["derived_tolerance_cap"], CALIBRATION_THRESHOLDS["derived_tolerance_multiplier"] * max_noise))
        gates = {
            "case_count": len(manifest) >= CALIBRATION_THRESHOLDS["case_count_min"],
            "finite": all(np.isfinite(array).all() for array in (exact, prefix, cross)),
            "exact_duplicate": maxima["exact_duplicate_relative_max"] <= CALIBRATION_THRESHOLDS["exact_duplicate_relative_max"],
            "same_batch_prefix": maxima["same_batch_prefix_relative_max"] <= CALIBRATION_THRESHOLDS["same_batch_prefix_relative_max"],
            "cross_composition": maxima["cross_composition_prefix_relative_max"] <= CALIBRATION_THRESHOLDS["cross_composition_prefix_relative_max"],
            "derived_tolerance_below_cap": tau < CALIBRATION_THRESHOLDS["derived_tolerance_cap"],
        }
        passed = all(gates.values())
        authorization = "phase1299_qwen3_behavior_only" if passed else "close_c031_as_numerically_unqualified"
        summary = {"case_count": len(manifest), "maxima": maxima, "max_observed_noise": max_noise, "frozen_tolerance": tau, "gates": gates, "all_gates_passed": passed}
        save(SUMMARY, summary)
        save(TOLERANCE, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"], "rule": protocol["tolerance_rule"], "max_observed_noise": max_noise, "tau": tau, "source_array_sha256": sha(ARRAYS), "frozen_before_semantic_hidden_run": True})
        save(RUN_META, {"phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"], "array_sha256": sha(ARRAYS), "model_audit": qa, "placement": placement, "runtime_seconds": time.time() - started, "fixed_sequence_length": max_length, "fixed_batch_size": 3 * BATCH_GROUPS, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
        save(FINAL, {"phase": PHASE, "campaign": CAMPAIGN, "verdict": "fp16_same_shape_camera_calibrated" if passed else "fp16_same_shape_camera_unqualified", "all_gates_passed": passed, "frozen_tolerance": tau, "authorization": authorization, "array_sha256": sha(ARRAYS), "protocol_digest": protocol["protocol_digest"]})
        save(COMPLETE, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "formal_runs_consumed": 1, "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"maxima": maxima, "tau": tau, "gates": gates, "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()


if __name__ == "__main__":
    main()
