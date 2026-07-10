from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BUNDLE_SCHEMA = "agi_research_bundle.v2"
REAL_UNIT_SCHEMA = "real_unit_evidence.v1"
TRACE_EVENT_SCHEMA = "real_trace_event.v1"
CLAIM_SCHEMA = "mechanism_claim.v1"

UNIT_KINDS = {
    "residual_dimension",
    "attention_head",
    "attention_head_channel",
    "mlp_gate_neuron",
    "mlp_up_neuron",
    "mlp_product_neuron",
    "unembedding_token",
}

EVIDENCE_LEVELS = {f"L{index}" for index in range(9)}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message}


def canonical_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(row: dict[str, Any], fields: Iterable[str], prefix: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in fields:
        if row.get(field) in (None, ""):
            issues.append(ValidationIssue(f"{prefix}.{field}", "required field is missing"))
    return issues


def validate_unit_row(row: dict[str, Any], model_snapshot: dict[str, Any] | None = None) -> list[ValidationIssue]:
    issues = _required(
        row,
        (
            "schema_version",
            "run_id",
            "model",
            "model_revision",
            "case_id",
            "token_position",
            "layer",
            "component",
            "unit_kind",
            "unit_index",
            "evidence_level",
            "source_artifact",
        ),
        "unit",
    )
    if row.get("schema_version") != REAL_UNIT_SCHEMA:
        issues.append(ValidationIssue("unit.schema_version", f"expected {REAL_UNIT_SCHEMA}"))
    if row.get("unit_kind") not in UNIT_KINDS:
        issues.append(ValidationIssue("unit.unit_kind", "unknown physical unit kind"))
    if row.get("evidence_level") not in EVIDENCE_LEVELS:
        issues.append(ValidationIssue("unit.evidence_level", "expected L0-L8"))

    if model_snapshot:
        layer = row.get("layer")
        unit_index = row.get("unit_index")
        layer_count = model_snapshot.get("num_hidden_layers")
        if isinstance(layer, int) and isinstance(layer_count, int) and not 0 <= layer < layer_count:
            issues.append(ValidationIssue("unit.layer", f"outside model range 0..{layer_count - 1}"))
        limit_by_kind = {
            "residual_dimension": model_snapshot.get("hidden_size"),
            "attention_head": model_snapshot.get("num_attention_heads"),
            "attention_head_channel": model_snapshot.get("head_dim"),
            "mlp_gate_neuron": model_snapshot.get("intermediate_size"),
            "mlp_up_neuron": model_snapshot.get("intermediate_size"),
            "mlp_product_neuron": model_snapshot.get("intermediate_size"),
            "unembedding_token": model_snapshot.get("vocab_size"),
        }
        limit = limit_by_kind.get(str(row.get("unit_kind")))
        if isinstance(unit_index, int) and isinstance(limit, int) and not 0 <= unit_index < limit:
            issues.append(ValidationIssue("unit.unit_index", f"outside unit range 0..{limit - 1}"))
    return issues


def validate_trace_event(row: dict[str, Any], model_snapshot: dict[str, Any] | None = None) -> list[ValidationIssue]:
    issues = _required(
        row,
        ("schema_version", "run_id", "event_index", "event_type", "model", "token_position", "source_artifact"),
        "trace_event",
    )
    if row.get("schema_version") != TRACE_EVENT_SCHEMA:
        issues.append(ValidationIssue("trace_event.schema_version", f"expected {TRACE_EVENT_SCHEMA}"))
    layer = row.get("layer")
    layer_count = (model_snapshot or {}).get("num_hidden_layers")
    if isinstance(layer, int) and layer >= 0 and isinstance(layer_count, int) and layer >= layer_count:
        issues.append(ValidationIssue("trace_event.layer", f"outside model range 0..{layer_count - 1}"))
    return issues


def validate_bundle_manifest(manifest: dict[str, Any], run_dir: Path) -> list[ValidationIssue]:
    issues = _required(
        manifest,
        ("schema_version", "run_id", "model", "model_revision", "created_at", "artifacts", "counts"),
        "manifest",
    )
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        issues.append(ValidationIssue("manifest.schema_version", f"expected {BUNDLE_SCHEMA}"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        issues.append(ValidationIssue("manifest.artifacts", "must be an object"))
        return issues
    for name, record in artifacts.items():
        if not isinstance(record, dict) or not record.get("path"):
            issues.append(ValidationIssue(f"manifest.artifacts.{name}", "artifact record needs a path"))
            continue
        target = (run_dir / str(record["path"])).resolve()
        try:
            target.relative_to(run_dir.resolve())
        except ValueError:
            issues.append(ValidationIssue(f"manifest.artifacts.{name}.path", "artifact escapes run directory"))
            continue
        if not target.exists():
            issues.append(ValidationIssue(f"manifest.artifacts.{name}.path", "artifact file does not exist"))
            continue
        expected_hash = record.get("sha256")
        if expected_hash and file_sha256(target) != expected_hash:
            issues.append(ValidationIssue(f"manifest.artifacts.{name}.sha256", "artifact checksum mismatch"))
    return issues
