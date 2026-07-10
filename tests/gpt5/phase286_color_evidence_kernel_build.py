#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = ROOT / "tests" / "result" / "research_kernel"
PUBLIC_ROOT = ROOT / "frontend" / "public" / "vis_data" / "research_kernel"
MODEL_ROOT = ROOT / "models" / "hf"

MODEL_DIRS = {
    "qwen3": "qwen3-4b",
    "glm4": "glm4-9b-chat-hf",
    "deepseek7b": "deepseek-r1-distill-qwen-7b",
}

PHASE941_ROOT = ROOT / "tests" / "result" / "phase941_color_feature_neuron_atlas"
PHASE941_ROUNDS = {
    model: f"color_feature_neuron_atlas_{model}_full" for model in MODEL_DIRS
}
PHASE944_TRACE_ROOT = ROOT / "tests" / "result" / "phase944_integrated_mechanism_trace"
PHASE944_AUDIT_ROOT = (
    ROOT
    / "tests"
    / "result"
    / "phase944_activation_weighted_mlp_channel_causal_audit"
    / "activation_weighted_mlp_channel_causal_audit"
)

BUNDLE_SCHEMA = "agi_research_bundle.v2"
UNIT_SCHEMA = "real_unit_evidence.v1"
TRACE_SCHEMA = "real_trace_event.v1"
CLAIM_SCHEMA = "mechanism_claim.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def build_model_snapshot(model: str) -> dict[str, Any]:
    config_path = MODEL_ROOT / MODEL_DIRS[model] / "config.json"
    config = read_json(config_path)
    hidden_size = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    model_revision = f"config-sha256:{sha256(config_path)}"
    return {
        "schema_version": "model_snapshot.v1",
        "model": model,
        "model_dir": relative(config_path.parent),
        "model_revision": model_revision,
        "architecture": (config.get("architectures") or [config.get("model_type")])[0],
        "model_type": config.get("model_type"),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "intermediate_size": int(config["intermediate_size"]),
        "num_attention_heads": heads,
        "num_key_value_heads": int(config.get("num_key_value_heads", heads)),
        "head_dim": int(config.get("head_dim") or hidden_size // heads),
        "vocab_size": int(config["vocab_size"]),
        "config_sha256": sha256(config_path),
        "captured_at": utc_now(),
    }


def phase941_paths(model: str) -> dict[str, Path]:
    directory = PHASE941_ROOT / PHASE941_ROUNDS[model]
    return {
        "dataset": directory / f"phase941_{model}_dataset.jsonl",
        "sample_rows": directory / f"phase941_{model}_sample_rows.jsonl",
        "channel_rows": directory / f"phase941_{model}_channel_rows.jsonl",
        "intervention_rows": directory / f"phase941_{model}_intervention_rows.jsonl",
        "summary": directory / f"phase941_{model}_summary.json",
    }


def trace_path(model: str) -> Path:
    directory = PHASE944_TRACE_ROOT / f"phase944_{model}_red_cube_trace"
    return directory / f"phase944_{model}_integrated_mechanism_trace.json"


def audit_path(model: str) -> Path:
    return PHASE944_AUDIT_ROOT / f"phase944_{model}_rows.jsonl"


def aggregate_group_interventions(rows: list[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    values: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row.get("factor", 1.0)) not in {0.0, 0.5}:
            continue
        layer = int(row["layer"])
        label = str(row["target_label"])
        for channel in row.get("channels") or []:
            values[(layer, int(channel), label)].append(row)

    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for key, matches in values.items():
        zero = [row for row in matches if float(row.get("factor", 1.0)) == 0.0]
        half = [row for row in matches if float(row.get("factor", 1.0)) == 0.5]
        zero_deltas = [float(row.get("margin_delta", 0.0)) for row in zero]
        half_deltas = [float(row.get("margin_delta", 0.0)) for row in half]
        result[key] = {
            "causal_scope": "channel_group_not_single_unit",
            "zero_scale_margin_delta_mean": mean(zero_deltas),
            "half_scale_margin_delta_mean": mean(half_deltas),
            "causal_supported": bool(zero_deltas and mean(zero_deltas) is not None and mean(zero_deltas) < -0.05),
            "group_test_rows": len(matches),
            "source_lines": sorted({int(row["_source_line"]) for row in matches}),
        }
    return result


def build_unit_rows(
    model: str,
    run_id: str,
    snapshot: dict[str, Any],
    channel_rows: list[dict[str, Any]],
    interventions: list[dict[str, Any]],
    max_per_label: int,
) -> list[dict[str, Any]]:
    causal_map = aggregate_group_interventions(interventions)
    stable = [
        row
        for row in channel_rows
        if int(row.get("coverage_objects", 0)) >= 3
        and int(row.get("coverage_templates", 0)) >= 3
        and int(row.get("count", 0)) >= 8
    ]
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stable:
        by_label[str(row["target_label"])].append(row)

    selected: list[dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        rows.sort(key=lambda row: float(row.get("effective_score", 0.0)), reverse=True)
        selected.extend(rows[:max_per_label])

    unit_rows: list[dict[str, Any]] = []
    source = phase941_paths(model)["channel_rows"]
    intervention_source = phase941_paths(model)["intervention_rows"]
    for row in selected:
        layer = int(row["layer"])
        channel = int(row["channel"])
        label = str(row["target_label"])
        group_causal = causal_map.get((layer, channel, label))
        unit_rows.append(
            {
                "schema_version": UNIT_SCHEMA,
                "run_id": run_id,
                "model": model,
                "model_revision": snapshot["model_revision"],
                "case_id": str(row.get("max_sample_id") or f"aggregate:color:{label}"),
                "family_id": "content_knowledge",
                "relation": "color",
                "target_label": label,
                "token_position": -1,
                "layer": layer,
                "component": "mlp",
                "unit_kind": "mlp_product_neuron",
                "unit_index": channel,
                "activation": row.get("mean_activation"),
                "activation_abs": row.get("mean_abs_activation"),
                "contrast_delta": row.get("selectivity_delta"),
                "readout_contribution": row.get("mean_contribution"),
                "candidate_score": row.get("effective_score"),
                "coverage_objects": row.get("coverage_objects"),
                "coverage_templates": row.get("coverage_templates"),
                "positive_rate": row.get("positive_rate"),
                "causal_supported": bool(group_causal and group_causal.get("causal_supported")),
                "causal_scope": (group_causal or {}).get("causal_scope", "not_tested"),
                "causal_margin_delta": (group_causal or {}).get("zero_scale_margin_delta_mean"),
                "half_scale_margin_delta": (group_causal or {}).get("half_scale_margin_delta_mean"),
                "side_effect": None,
                "evidence_level": "L4",
                "evidence_boundary": "real unit address; readout attribution plus optional group intervention; not single-unit causality",
                "source_artifact": f"{relative(source)}#line={row['_source_line']}",
                "causal_source_artifact": (
                    f"{relative(intervention_source)}#lines={','.join(map(str, group_causal['source_lines']))}"
                    if group_causal
                    else None
                ),
            }
        )
    unit_rows.sort(key=lambda row: (int(row["layer"]), -float(row.get("candidate_score") or 0.0)))
    return unit_rows


def component_payload(layer: dict[str, Any], component: str) -> dict[str, Any]:
    payload = ((layer.get("components") or {}).get(component) or {})
    field = payload.get("candidate_field") or {}
    return {
        "norm": payload.get("norm"),
        "cosine_to_residual": payload.get("cosine_to_residual"),
        "target_score": field.get("target_score"),
        "target_probability": field.get("target_probability"),
        "target_rank": field.get("target_rank_within_candidates"),
        "competitor_label": field.get("competitor_label"),
        "margin": field.get("margin_vs_competitor"),
        "candidate_gate_open": field.get("candidate_gate_open"),
    }


def build_trace_rows(
    model: str,
    run_id: str,
    snapshot: dict[str, Any],
    trace: dict[str, Any],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in units:
        if row.get("target_label") == trace.get("target", {}).get("attribute"):
            units_by_layer[int(row["layer"])].append(row)
    for rows in units_by_layer.values():
        rows.sort(key=lambda row: float(row.get("candidate_score") or 0.0), reverse=True)

    source = trace_path(model)
    token_position = int((trace.get("positions") or {}).get("answer", -1))
    events: list[dict[str, Any]] = []

    def add(event_type: str, layer: int, component: str, metrics: dict[str, Any], top_units: list[dict[str, Any]] | None = None) -> None:
        events.append(
            {
                "schema_version": TRACE_SCHEMA,
                "run_id": run_id,
                "event_index": len(events),
                "event_type": event_type,
                "model": model,
                "model_revision": snapshot["model_revision"],
                "layer": layer,
                "component": component,
                "token_position": token_position,
                "token": (trace.get("tokens") or [{}])[token_position].get("text") if token_position >= 0 else None,
                "metrics": metrics,
                "top_units": [
                    {
                        "unit_kind": row["unit_kind"],
                        "unit_index": row["unit_index"],
                        "candidate_score": row.get("candidate_score"),
                        "evidence_level": row.get("evidence_level"),
                        "causal_scope": row.get("causal_scope"),
                    }
                    for row in (top_units or [])[:24]
                ],
                "source_artifact": f"{relative(source)}#layers={layer}",
            }
        )

    for layer in trace.get("layers") or []:
        layer_index = int(layer.get("layer", -1))
        residual = layer.get("residual") or {}
        if layer_index < 0:
            add("embedding", -1, "embedding", residual)
            continue
        add("attention_output", layer_index, "attention", component_payload(layer, "attention"))
        add("mlp_product_write", layer_index, "mlp", component_payload(layer, "mlp"), units_by_layer.get(layer_index))
        answer_field = (((layer.get("positions") or {}).get("answer") or {}).get("candidate_field") or {})
        add(
            "residual_update",
            layer_index,
            "residual",
            {
                "norm": residual.get("norm"),
                "delta_norm": residual.get("delta_norm_from_previous"),
                "cosine_to_previous": residual.get("cosine_to_previous"),
                "target_score": answer_field.get("target_score"),
                "target_rank": answer_field.get("target_rank_within_candidates"),
                "competitor_label": answer_field.get("competitor_label"),
                "margin": answer_field.get("margin_vs_competitor"),
                "candidate_gate_open": answer_field.get("candidate_gate_open"),
            },
        )

    summary = trace.get("summary") or {}
    add(
        "unembedding_readout",
        int(snapshot["num_hidden_layers"]) - 1,
        "unembedding",
        {
            "target_token_id": summary.get("target_global_token_id"),
            "target_rank": summary.get("target_global_rank"),
            "target_logit": summary.get("target_global_logit"),
            "margin": summary.get("final_margin_vs_color_competitor"),
            "competitor_label": summary.get("final_competitor_label"),
            "next_token": summary.get("next_token"),
            "global_closed": summary.get("global_closed"),
        },
    )
    return events


def artifact_record(run_dir: Path, path: Path, row_count: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if row_count is not None:
        record["rows"] = row_count
    return record


def build_claim(model: str, run_id: str, unit_count: int, causal_group_count: int) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_SCHEMA,
        "claim_id": f"claim:color:mlp_product_channels:{model}",
        "claim_text": "Stable color-related readout contributions are physically localized to real MLP product channels at selected layers.",
        "scope": {"model": model, "family": "content_knowledge", "relation": "color"},
        "status": "supported_component_path_not_single_unit_causal",
        "evidence_level": "L4",
        "positive_runs": [run_id],
        "negative_evidence": [
            "Phase941 interventions scale channel groups, so they do not establish single-unit necessity.",
            "Only five layer bands were scanned in the source atlas.",
        ],
        "counterexamples": [],
        "next_test": "Run low-side-effect single-unit and matched random-unit interventions on heldout objects and templates.",
        "unit_count": unit_count,
        "group_causal_unit_count": causal_group_count,
    }


def build_run(model: str, max_units_per_label: int) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = f"phase286_color_real_units_{model}"
    run_dir = KERNEL_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_model_snapshot(model)
    paths = phase941_paths(model)
    summary = read_json(paths["summary"])
    cases = read_jsonl(paths["dataset"])
    channel_rows = read_jsonl(paths["channel_rows"])
    interventions = read_jsonl(paths["intervention_rows"])
    trace = read_json(trace_path(model))
    audit_rows = read_jsonl(audit_path(model)) if audit_path(model).exists() else []
    unit_rows = build_unit_rows(model, run_id, snapshot, channel_rows, interventions, max_units_per_label)
    trace_rows = build_trace_rows(model, run_id, snapshot, trace, unit_rows)

    experiment = {
        "schema_version": "experiment_spec.v2",
        "run_id": run_id,
        "phase": 286,
        "title": f"Color real-unit evidence normalization ({model})",
        "objective": "Publish real MLP product-unit addresses and a layer-synchronized trace without promoting group interventions to single-unit causality.",
        "model": model,
        "source_phases": [941, 944],
        "sample_count": len(cases),
        "acceptance": {
            "real_unit_addresses_valid": True,
            "three_model_source_required": True,
            "single_unit_causality_required_for_L5": True,
        },
        "created_at": utc_now(),
    }
    claim = build_claim(
        model,
        run_id,
        len(unit_rows),
        sum(1 for row in unit_rows if row.get("causal_supported")),
    )
    claims_delta = [claim]
    report = "\n".join(
        [
            f"# {experiment['title']}",
            "",
            f"- run_id: {run_id}",
            f"- source samples: {len(cases)}",
            f"- published real MLP product units: {len(unit_rows)}",
            f"- compact trace events: {len(trace_rows)}",
            f"- group-causal-supported unit addresses: {sum(1 for row in unit_rows if row.get('causal_supported'))}",
            f"- source global closure: {trace.get('summary', {}).get('global_closed')}",
            "",
            "Boundary: channel-group interventions remain L4 component attribution evidence. No row is labeled as single-unit necessity.",
            "",
        ]
    )

    write_json(run_dir / "experiment.json", experiment)
    write_json(run_dir / "model_snapshot.json", snapshot)
    write_jsonl(run_dir / "cases.jsonl", [{k: v for k, v in row.items() if not k.startswith("_")} for row in cases])
    write_jsonl(run_dir / "trace_events.jsonl", trace_rows)
    write_jsonl(run_dir / "unit_evidence.jsonl", unit_rows)
    write_jsonl(
        run_dir / "intervention_rows.jsonl",
        [
            {"source_phase": 941, **{k: v for k, v in row.items() if not k.startswith("_")}}
            for row in interventions
        ]
        + [
            {"source_phase": 944, **{k: v for k, v in row.items() if not k.startswith("_")}}
            for row in audit_rows
        ],
    )
    write_jsonl(run_dir / "claims_delta.jsonl", claims_delta)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    artifact_specs = {
        "experiment": (run_dir / "experiment.json", None),
        "model_snapshot": (run_dir / "model_snapshot.json", None),
        "cases": (run_dir / "cases.jsonl", len(cases)),
        "trace_events": (run_dir / "trace_events.jsonl", len(trace_rows)),
        "unit_evidence": (run_dir / "unit_evidence.jsonl", len(unit_rows)),
        "interventions": (run_dir / "intervention_rows.jsonl", len(interventions) + len(audit_rows)),
        "claims_delta": (run_dir / "claims_delta.jsonl", len(claims_delta)),
        "report": (run_dir / "report.md", None),
    }
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "run_id": run_id,
        "phase": 286,
        "model": model,
        "model_revision": snapshot["model_revision"],
        "created_at": utc_now(),
        "status": "complete",
        "evidence_boundary": "real-unit L4; group causal only; not single-unit closure",
        "counts": {
            "cases": len(cases),
            "trace_events": len(trace_rows),
            "unit_evidence": len(unit_rows),
            "interventions": len(interventions) + len(audit_rows),
            "claims": len(claims_delta),
        },
        "source_artifacts": [relative(path) for path in paths.values()] + [relative(trace_path(model)), relative(audit_path(model))],
        "artifacts": {name: artifact_record(run_dir, path, count) for name, (path, count) in artifact_specs.items()},
    }
    write_json(run_dir / "manifest.json", manifest)
    run_entry = {
        "run_id": run_id,
        "model": model,
        "phase": 286,
        "title": experiment["title"],
        "status": "complete",
        "evidence_level": "L4",
        "case_count": len(cases),
        "unit_count": len(unit_rows),
        "trace_event_count": len(trace_rows),
        "global_closed_single_trace": bool(trace.get("summary", {}).get("global_closed")),
        "manifest_path": f"runs/{run_id}/manifest.json",
        "trace_path": f"runs/{run_id}/trace_events.jsonl",
        "unit_path": f"runs/{run_id}/unit_evidence.jsonl",
    }
    return run_entry, claim


def build_kernel(max_units_per_label: int = 50) -> dict[str, Any]:
    KERNEL_ROOT.mkdir(parents=True, exist_ok=True)
    snapshots = [build_model_snapshot(model) for model in MODEL_DIRS]
    run_entries: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for model in MODEL_DIRS:
        run_entry, claim = build_run(model, max_units_per_label=max_units_per_label)
        run_entries.append(run_entry)
        claims.append(claim)

    total_cases = sum(int(run["case_count"]) for run in run_entries)
    target_cases = 1440 * len(MODEL_DIRS)
    progress = {
        "schema_version": "research_progress.v1",
        "dimensions": {
            "model_coverage": {"valid": 3, "required": 3, "ratio": 1.0},
            "color_case_coverage": {"valid": total_cases, "required": target_cases, "ratio": min(1.0, total_cases / target_cases)},
            "real_trace_coverage": {"valid": 3, "required": 3, "ratio": 1.0},
            "real_unit_address_coverage": {"valid": 3, "required": 3, "ratio": 1.0},
            "single_unit_causal_coverage": {"valid": 0, "required": 3, "ratio": 0.0},
            "heldout_prediction_coverage": {"valid": 0, "required": 3, "ratio": 0.0},
            "clean_closure_coverage": {"valid": 0, "required": 3, "ratio": 0.0},
        },
        "boundary": "Ratios use explicit numerators and denominators. Single prompt closure is not counted as mechanism closure.",
    }
    gaps = [
        {
            "gap_id": "gap:color:sample_expansion",
            "title": "Expand color cases to the declared 1440 cases per model",
            "status": "open",
            "priority": "P0",
            "next_test": "phase287_color_real_trace_expansion",
        },
        {
            "gap_id": "gap:color:single_unit_causal",
            "title": "Separate channel-group effects into low-side-effect single-unit necessity tests",
            "status": "open",
            "priority": "P0",
            "next_test": "phase287_color_single_unit_causal_audit",
        },
        {
            "gap_id": "gap:color:heldout_prediction",
            "title": "Predict units and layer path on unseen objects and templates before measuring them",
            "status": "open",
            "priority": "P0",
            "next_test": "phase288_color_heldout_prediction",
        },
        {
            "gap_id": "gap:color:trace_component_depth",
            "title": "Split MLP trace into gate, up, activation product and down-projection events",
            "status": "open",
            "priority": "P1",
            "next_test": "phase287_full_component_trace",
        },
    ]
    registry = {
        "schema_version": "real_model_registry.v1",
        "generated_at": utc_now(),
        "models": snapshots,
    }
    root_manifest = {
        "schema_version": "research_kernel_manifest.v1",
        "generated_at": utc_now(),
        "phase": 286,
        "title": "Color real-unit evidence kernel",
        "runs": run_entries,
        "claims": claims,
        "gaps": gaps,
        "progress": progress,
        "evidence_policy": {
            "real_unit_required_fields": ["model_revision", "layer", "component", "unit_kind", "unit_index", "token_position", "source_artifact"],
            "group_causal_is_not_single_unit": True,
            "demo_nodes_default_visible": False,
        },
    }
    write_json(KERNEL_ROOT / "model_registry.json", registry)
    write_json(KERNEL_ROOT / "manifest.json", root_manifest)
    write_json(KERNEL_ROOT / "progress.json", progress)
    write_jsonl(KERNEL_ROOT / "claims.jsonl", claims)
    write_jsonl(KERNEL_ROOT / "gaps.jsonl", gaps)

    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(KERNEL_ROOT, PUBLIC_ROOT, dirs_exist_ok=True)
    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase286 color evidence kernel from real Phase941/944 artifacts")
    parser.add_argument("--max-units-per-label", type=int, default=50)
    args = parser.parse_args()
    manifest = build_kernel(max_units_per_label=max(1, args.max_units_per_label))
    print(
        json.dumps(
            {
                "phase": manifest["phase"],
                "runs": len(manifest["runs"]),
                "claims": len(manifest["claims"]),
                "open_gaps": len(manifest["gaps"]),
                "public_root": relative(PUBLIC_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
