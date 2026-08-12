#!/usr/bin/env python3
"""Run one frozen Pythia checkpoint for Phase1117."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import quantization_audit, release_fp16
import phase1117_pythia_training_dynamics_protocol as protocol


BATCH_SIZE = 64
DOWNLOAD_PATTERNS = tuple(
    [
        "config.json",
        "generation_config.json",
        protocol.WEIGHT_FORMAT,
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
)


def ensure_checkpoint(checkpoint: str) -> tuple[Path, str]:
    target = protocol.MODEL_ROOT / checkpoint
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    snapshot_download(
        protocol.MODEL_REPO,
        revision=checkpoint,
        local_dir=target,
        allow_patterns=list(DOWNLOAD_PATTERNS),
    )
    info = HfApi().model_info(protocol.MODEL_REPO, revision=checkpoint)
    return target, str(info.sha)


def model_manifest(root: Path) -> list[dict[str, Any]]:
    manifest = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.name.endswith(".metadata"):
            continue
        if not (
            path.name == protocol.WEIGHT_FORMAT
            or path.name in {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
        ):
            continue
        manifest.append(
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": protocol.file_sha256(path),
            }
        )
    return manifest


def parameter_probe(model: torch.nn.Module) -> dict[str, Any]:
    """Fingerprint tensor content independently of checkpoint file serialization."""
    samples: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        indices = sorted({0, flat.numel() // 3, (2 * flat.numel()) // 3, flat.numel() - 1})
        values = [float(flat[index].float().item()) for index in indices]
        samples.append({"name": name, "shape": list(parameter.shape), "indices": indices, "values": values})
    return {
        "method": "four deterministic FP32-read samples per named parameter",
        "parameter_count": len(samples),
        "digest": protocol.digest(samples),
    }


def run(checkpoint: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1117 protocol audit failed")
    integrity_path = protocol.OUT_ROOT / "protocol" / "checkpoint_integrity.json"
    if not integrity_path.exists():
        raise RuntimeError("all-checkpoint parameter integrity audit is missing")
    integrity = protocol.read_json(integrity_path)
    if not integrity["all_checks_passed"] or integrity["checkpoint_set"] != prereg["checkpoints"]:
        raise RuntimeError("all-checkpoint parameter integrity audit failed")
    if checkpoint not in prereg["checkpoints"]:
        raise RuntimeError(f"checkpoint not frozen: {checkpoint}")
    if checkpoint != protocol.FINAL_QUALIFICATION_CHECKPOINT:
        authorization_path = protocol.OUT_ROOT / "analysis" / "trajectory_authorization.json"
        if not authorization_path.exists():
            raise RuntimeError("trajectory authorization does not exist")
        authorization = protocol.read_json(authorization_path)
        if not authorization["trajectory_authorized"]:
            raise RuntimeError("final checkpoint did not authorize the trajectory")

    rows = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    if protocol.digest(rows) != prereg["case_digest"]:
        raise RuntimeError("case digest mismatch")

    started = time.time()
    local_path, repo_commit = ensure_checkpoint(checkpoint)
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=protocol.WEIGHT_FORMAT.endswith(".safetensors"),
        ).to("cuda")
        model.eval()
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        probe = parameter_probe(model)

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        detail: list[dict[str, Any]] = []
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), BATCH_SIZE):
                    batch = panel[start : start + BATCH_SIZE]
                    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device="cuda")
                    attention_mask = torch.ones_like(input_ids)
                    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                    logits = output.logits[:, -1, :].float()
                    top_ids = torch.argmax(logits, dim=-1)
                    for slot, row in enumerate(batch):
                        true_scores = [float(logits[slot, int(token_id)].item()) for token_id in row["true_candidate_ids"]]
                        control_scores = [float(logits[slot, int(token_id)].item()) for token_id in row["control_candidate_ids"]]
                        true_z = true_scores[0] - true_scores[1]
                        control_z = control_scores[0] - control_scores[1]
                        finite = all(math.isfinite(value) for value in [*true_scores, *control_scores, true_z, control_z])
                        expected_index = int(row["sense"])
                        expected_margin = true_scores[expected_index] - true_scores[1 - expected_index]
                        top_id = int(top_ids[slot].item())
                        detail.append(
                            {
                                "schema_version": "phase1117_pythia_training_detail.v1",
                                "phase": protocol.PHASE,
                                "checkpoint": checkpoint,
                                "case_index": row["case_index"],
                                "record_id": row["record_id"],
                                "pair_id": row["pair_id"],
                                "concept_id": row["concept_id"],
                                "split": row["split"],
                                "template": row["template"],
                                "sense": row["sense"],
                                "true_candidate_labels": row["true_candidate_labels"],
                                "control_concept_id": row["control_concept_id"],
                                "control_candidate_labels": row["control_candidate_labels"],
                                "true_scores": true_scores if finite else None,
                                "control_scores": control_scores if finite else None,
                                "true_z": true_z if finite else None,
                                "control_z": control_z if finite else None,
                                "expected_margin": expected_margin if finite else None,
                                "finite": finite,
                                "candidate_hit": finite and expected_margin > 0.0,
                                "top_token_id": top_id,
                                "top_token_text": tokenizer.decode([top_id]),
                                "direct_true_candidate": top_id in row["true_candidate_ids"],
                            }
                        )
                    del output, logits, top_ids, input_ids, attention_mask
                print(json.dumps({"phase": protocol.PHASE, "checkpoint": checkpoint, "length_complete": length}), flush=True)

        finite_rows = [row for row in detail if row["finite"]]
        manifest = model_manifest(local_path)
        summary_core = {
            "schema_version": "phase1117_pythia_training_behavior_summary.v1",
            "phase": protocol.PHASE,
            "checkpoint": checkpoint,
            "repo": protocol.MODEL_REPO,
            "repo_commit": repo_commit,
            "weight_format": protocol.WEIGHT_FORMAT,
            "parameter_probe": probe,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digest"],
            "precision": precision,
            "placement": "full_cuda",
            "quantization": "none",
            "case_count": len(detail),
            "detail_digest": protocol.digest(detail),
            "finite_fraction": len(finite_rows) / max(len(detail), 1),
            "candidate_accuracy": sum(bool(row["candidate_hit"]) for row in finite_rows) / max(len(finite_rows), 1),
            "direct_true_candidate_rate": sum(bool(row["direct_true_candidate"]) for row in detail) / max(len(detail), 1),
            "elapsed_seconds": time.time() - started,
            "model_manifest": manifest,
            "model_manifest_digest": protocol.digest(manifest),
        }
        summary = dict(summary_core)
        summary["summary_digest"] = protocol.digest(summary_core)
        output_root = protocol.OUT_ROOT / "behavior" / checkpoint
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", detail)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", choices=protocol.CHECKPOINTS)
    args = parser.parse_args()
    run(args.checkpoint)


if __name__ == "__main__":
    main()
