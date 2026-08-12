#!/usr/bin/env python3
"""Independent audit for the Qwen3-14B FP16 offload smoke test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "hf" / "Qwen3-14B"
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1118_qwen3_14b_fp16_offload_smoke"
SHARDS = (
    ("model-00001-of-00008.safetensors", 3_841_788_544, "e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e"),
    ("model-00002-of-00008.safetensors", 3_963_750_816, "f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9"),
    ("model-00003-of-00008.safetensors", 3_963_750_880, "dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9"),
    ("model-00004-of-00008.safetensors", 3_963_750_880, "eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d"),
    ("model-00005-of-00008.safetensors", 3_963_750_880, "97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6"),
    ("model-00006-of-00008.safetensors", 3_963_750_880, "9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e"),
    ("model-00007-of-00008.safetensors", 3_963_750_880, "0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750"),
    ("model-00008-of-00008.safetensors", 1_912_371_880, "0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    result = read_json(OUT_ROOT / "result" / "smoke_result.json")
    prefetch = read_json(OUT_ROOT / "download" / "prefetch_audit.json")

    protocol_core = dict(protocol)
    recorded_protocol_digest = protocol_core.pop("protocol_digest")
    result_core = dict(result)
    recorded_result_digest = result_core.pop("result_digest")
    prefetch_core = dict(prefetch)
    recorded_prefetch_digest = prefetch_core.pop("audit_digest")

    shard_rows: list[dict[str, Any]] = []
    for name, expected_size, expected_sha256 in SHARDS:
        path = MODEL_ROOT / name
        actual_size = path.stat().st_size if path.exists() else None
        actual_sha256 = file_sha256(path) if actual_size == expected_size else None
        shard_rows.append(
            {
                "name": name,
                "actual_size": actual_size,
                "actual_sha256": actual_sha256,
                "passed": actual_size == expected_size and actual_sha256 == expected_sha256,
            }
        )

    rows = result.get("rows", [])
    precision = result.get("precision", {})
    checks = {
        "protocol_digest": digest(protocol_core) == recorded_protocol_digest,
        "result_digest": digest(result_core) == recorded_result_digest,
        "prefetch_digest": digest(prefetch_core) == recorded_prefetch_digest,
        "all_weight_shards_rehashed": all(row["passed"] for row in shard_rows),
        "prefetch_self_checks": prefetch.get("all_checks_passed") is True,
        "smoke_self_checks": result.get("all_checks_passed") is True,
        "protocol_link": result.get("protocol_digest") == recorded_protocol_digest,
        "parameter_count": result.get("parameter_count") == 14_768_307_200,
        "precision_fp16": precision.get("has_fp16_parameters") is True,
        "precision_not_bf16": precision.get("has_bf16_parameters") is False,
        "not_quantized": precision.get("has_quantized_modules") is False,
        "eight_finite_forwards": len(rows) == 8 and all(row.get("finite") is True for row in rows),
        "device_map_preserved": result.get("actual_device_map") == protocol.get("device_map"),
        "required_disk_offload": any(value == "disk" for value in result.get("actual_device_map", {}).values()),
        "engineering_only_claim": result.get("scientific_scale_effect_identified") is False,
    }
    core = {
        "schema_version": "phase1118_qwen3_14b_fp16_smoke_audit.v2",
        "phase": 1118,
        "protocol_digest": recorded_protocol_digest,
        "result_digest": recorded_result_digest,
        "prefetch_digest": recorded_prefetch_digest,
        "checks": checks,
        "shards": shard_rows,
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(core)
    audit["audit_digest"] = digest(core)
    output = OUT_ROOT / "audit" / "result_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
