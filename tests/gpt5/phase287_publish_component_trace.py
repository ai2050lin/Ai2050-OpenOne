#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "tests" / "result" / "phase287_real_component_trace"
KERNEL_ROOT = ROOT / "tests" / "result" / "research_kernel"
PUBLIC_KERNEL_ROOT = ROOT / "frontend" / "public" / "vis_data" / "research_kernel"
PUBLIC_TRACE_ROOT = ROOT / "frontend" / "public" / "vis_data" / "real_component_trace"
RUN_IDS = {
    "qwen3": "phase287_qwen3_red_component_trace",
    "glm4": "phase287_glm4_red_component_trace",
    "deepseek7b": "phase287_deepseek7b_red_component_trace",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(run_dir: Path, path: Path, rows: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if rows is not None:
        payload["rows"] = rows
    return payload


def build_unit_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = trace["run_id"]
    snapshot = trace["model_snapshot"]
    seen: set[tuple[int, int]] = set()
    for event in trace.get("events") or []:
        if event.get("event_type") != "mlp_product":
            continue
        layer = int(event["layer"])
        for unit in event.get("top_units") or []:
            key = (layer, int(unit["unit_index"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "schema_version": "real_unit_evidence.v1",
                    "run_id": run_id,
                    "model": trace["model"],
                    "model_revision": snapshot["model_revision"],
                    "case_id": "color:red:cube:component_trace",
                    "family_id": "content_knowledge",
                    "relation": "color",
                    "target_label": trace["target_label"],
                    "token_position": int(trace["token_position"]),
                    "layer": layer,
                    "component": "mlp",
                    "unit_kind": "mlp_product_neuron",
                    "unit_index": int(unit["unit_index"]),
                    "activation": unit.get("value"),
                    "activation_abs": unit.get("magnitude"),
                    "contrast_delta": None,
                    "readout_contribution": None,
                    "candidate_score": unit.get("magnitude"),
                    "causal_supported": False,
                    "causal_scope": "not_tested",
                    "causal_margin_delta": None,
                    "side_effect": None,
                    "evidence_level": "L2",
                    "evidence_boundary": "exact activation at a real unit for one prompt; not stable and not causal",
                    "source_artifact": f"tests/result/phase287_real_component_trace/{run_id}/trace.json#event={event['event_index']}",
                }
            )
    return rows


def publish_run(model: str, run_id: str) -> dict[str, Any]:
    source_dir = SOURCE_ROOT / run_id
    trace = read_json(source_dir / "trace.json")
    run_dir = KERNEL_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = trace["model_snapshot"]
    experiment = {
        "schema_version": "experiment_spec.v2",
        "run_id": run_id,
        "phase": 287,
        "title": f"Full real-component color trace ({model})",
        "objective": "Capture the exact embedding-to-W_U component sequence and full intermediate vectors for one controlled color case.",
        "model": model,
        "case_count": 1,
        "created_at": trace["created_at"],
    }
    unit_rows = build_unit_rows(trace)
    trace_events = trace.get("events") or []
    claims_delta = [
        {
            "schema_version": "mechanism_claim.v1",
            "claim_id": f"claim:color:full_component_trace:{model}",
            "claim_text": "The controlled color prompt has a reproducible, physically addressed embedding-to-W_U trace in this model.",
            "scope": {"model": model, "relation": "color", "case": "red cube"},
            "status": "observed_single_case_not_generalized",
            "evidence_level": "L2",
            "positive_runs": [run_id],
            "negative_evidence": ["One prompt does not establish path stability or causality."],
            "counterexamples": ["DeepSeek7B opens the red candidate field but emits 'determined' as the global next token."] if model == "deepseek7b" else [],
            "next_test": "Run heldout color cases and single-unit causal controls.",
        }
    ]
    case = {
        "case_id": "color:red:cube:component_trace",
        "prompt": trace["prompt"],
        "target_label": trace["target_label"],
        "token_position": trace["token_position"],
    }
    write_json(run_dir / "experiment.json", experiment)
    write_json(run_dir / "model_snapshot.json", snapshot)
    write_jsonl(run_dir / "cases.jsonl", [case])
    write_jsonl(run_dir / "trace_events.jsonl", trace_events)
    write_jsonl(run_dir / "unit_evidence.jsonl", unit_rows)
    write_jsonl(run_dir / "intervention_rows.jsonl", [])
    write_jsonl(run_dir / "claims_delta.jsonl", claims_delta)
    shutil.copy2(source_dir / "full_vectors.pt", run_dir / "full_vectors.pt")
    report = "\n".join(
        [
            f"# Full real-component color trace ({model})",
            "",
            f"- events: {len(trace_events)}",
            f"- full vectors: {trace['summary']['vector_count']}",
            f"- layers: {trace['summary']['layer_count']}",
            f"- target color rank: {trace['summary']['target_color_rank']}",
            f"- target color margin: {trace['summary']['target_color_margin']}",
            f"- actual next token: {trace['summary']['next_token']['token']}",
            "",
            "This is exact L2 trace evidence for one prompt, not a causal or generalized mechanism claim.",
            "",
        ]
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    specs = {
        "experiment": (run_dir / "experiment.json", None),
        "model_snapshot": (run_dir / "model_snapshot.json", None),
        "cases": (run_dir / "cases.jsonl", 1),
        "trace_events": (run_dir / "trace_events.jsonl", len(trace_events)),
        "unit_evidence": (run_dir / "unit_evidence.jsonl", len(unit_rows)),
        "interventions": (run_dir / "intervention_rows.jsonl", 0),
        "claims_delta": (run_dir / "claims_delta.jsonl", 1),
        "full_vectors": (run_dir / "full_vectors.pt", trace["summary"]["vector_count"]),
        "report": (run_dir / "report.md", None),
    }
    manifest = {
        "schema_version": "agi_research_bundle.v2",
        "run_id": run_id,
        "phase": 287,
        "model": model,
        "model_revision": snapshot["model_revision"],
        "created_at": trace["created_at"],
        "status": "complete",
        "evidence_boundary": "exact one-case trace L2; no causal or closure promotion",
        "counts": {"cases": 1, "trace_events": len(trace_events), "unit_evidence": len(unit_rows), "interventions": 0, "claims": 1},
        "source_artifacts": [f"tests/result/phase287_real_component_trace/{run_id}/trace.json"],
        "artifacts": {name: artifact(run_dir, path, count) for name, (path, count) in specs.items()},
    }
    write_json(run_dir / "manifest.json", manifest)
    return {
        "run_id": run_id,
        "model": model,
        "phase": 287,
        "title": experiment["title"],
        "status": "complete",
        "evidence_level": "L2",
        "case_count": 1,
        "unit_count": len(unit_rows),
        "trace_event_count": len(trace_events),
        "target_color_rank": trace["summary"]["target_color_rank"],
        "target_color_margin": trace["summary"]["target_color_margin"],
        "next_token": trace["summary"]["next_token"],
        "manifest_path": f"runs/{run_id}/manifest.json",
        "trace_path": f"runs/{run_id}/trace_events.jsonl",
        "unit_path": f"runs/{run_id}/unit_evidence.jsonl",
        "public_trace_path": f"/vis_data/real_component_trace/{run_id}.json",
    }, claims_delta[0]


def main() -> None:
    kernel_manifest = read_json(KERNEL_ROOT / "manifest.json")
    existing_runs = {row["run_id"]: row for row in kernel_manifest.get("runs") or []}
    existing_claims = {row["claim_id"]: row for row in kernel_manifest.get("claims") or []}
    public_items = []
    for model, run_id in RUN_IDS.items():
        run, claim = publish_run(model, run_id)
        existing_runs[run_id] = run
        existing_claims[claim["claim_id"]] = claim
        public_items.append(
            {
                "run_id": run_id,
                "model": model,
                "label": f"{model} - real red color component trace",
                "path": run["public_trace_path"],
                "event_count": run["trace_event_count"],
                "unit_count": run["unit_count"],
                "target_color_rank": run["target_color_rank"],
                "target_color_margin": run["target_color_margin"],
                "next_token": run["next_token"],
                "evidence_level": "L2",
            }
        )

    kernel_manifest["runs"] = sorted(existing_runs.values(), key=lambda row: (int(row["phase"]), row["model"]))
    kernel_manifest["claims"] = sorted(existing_claims.values(), key=lambda row: row["claim_id"])
    kernel_manifest["generated_at"] = utc_now()
    for gap in kernel_manifest.get("gaps") or []:
        if gap.get("gap_id") == "gap:color:trace_component_depth":
            gap["status"] = "filled_phase287_single_case"
            gap["filled_by"] = list(RUN_IDS.values())
    progress = ((kernel_manifest.get("progress") or {}).get("dimensions") or {})
    progress["full_component_trace_coverage"] = {"valid": 3, "required": 3, "ratio": 1.0}
    write_json(KERNEL_ROOT / "manifest.json", kernel_manifest)
    write_jsonl(KERNEL_ROOT / "claims.jsonl", kernel_manifest["claims"])
    write_jsonl(KERNEL_ROOT / "gaps.jsonl", kernel_manifest.get("gaps") or [])

    PUBLIC_TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        PUBLIC_TRACE_ROOT / "manifest.json",
        {
            "schema_version": "real_component_trace_manifest.v1",
            "generated_at": utc_now(),
            "items": public_items,
        },
    )
    shutil.copytree(KERNEL_ROOT, PUBLIC_KERNEL_ROOT, dirs_exist_ok=True)
    print(json.dumps({"phase": 287, "published_runs": len(public_items), "items": public_items}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
