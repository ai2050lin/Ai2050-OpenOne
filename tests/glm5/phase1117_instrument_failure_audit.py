#!/usr/bin/env python3
"""Record why the Phase1117 safetensors trajectory is scientifically invalid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1117_pythia_training_dynamics"
MODEL_ROOT = ROOT / "models" / "hf" / "pythia-1.4b-deduped"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tensor_probe(path: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name).reshape(-1)
            indices = sorted({0, tensor.numel() // 3, (2 * tensor.numel()) // 3, tensor.numel() - 1})
            values = [float(tensor[index].float().item()) for index in indices]
            samples.append({"name": name, "indices": indices, "values": values})
    return {"parameter_count": len(samples), "digest": digest(samples)}


def main() -> None:
    left_name = "step16"
    right_name = "step143000"
    left_detail = read_jsonl(RESULT_ROOT / "behavior" / left_name / "candidate_detail.jsonl")
    right_detail = read_jsonl(RESULT_ROOT / "behavior" / right_name / "candidate_detail.jsonl")
    same_ids = sum(a["record_id"] == b["record_id"] for a, b in zip(left_detail, right_detail, strict=True))
    same_margins = sum(a["expected_margin"] == b["expected_margin"] for a, b in zip(left_detail, right_detail, strict=True))
    left_path = MODEL_ROOT / left_name / "model.safetensors"
    right_path = MODEL_ROOT / right_name / "model.safetensors"
    left_probe = tensor_probe(left_path)
    right_probe = tensor_probe(right_path)
    checks = {
        "case_count_684_each": len(left_detail) == len(right_detail) == 684,
        "record_alignment_complete": same_ids == 684,
        "all_expected_margins_identical": same_margins == 684,
        "weight_file_sha_differs": file_sha256(left_path) != file_sha256(right_path),
        "tensor_content_probe_collides": left_probe["digest"] == right_probe["digest"],
    }
    core = {
        "schema_version": "phase1117_instrument_failure_audit.v1",
        "phase": 1117,
        "invalid_revision": 1,
        "invalid_weight_format": "model.safetensors",
        "compared_checkpoints": [left_name, right_name],
        "checks": checks,
        "all_failure_checks_passed": all(checks.values()),
        "case_comparison": {
            "case_count": len(left_detail),
            "same_record_ids": same_ids,
            "identical_expected_margins": same_margins,
        },
        "file_sha256": {left_name: file_sha256(left_path), right_name: file_sha256(right_path)},
        "tensor_probes": {left_name: left_probe, right_name: right_probe},
        "decision": "invalidate all revision-1 trajectory conclusions and rerun unchanged scientific protocol with native pytorch_model.bin",
    }
    result = dict(core)
    result["audit_digest"] = digest(core)
    output = RESULT_ROOT / "audit" / "instrument_failure.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_failure_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
