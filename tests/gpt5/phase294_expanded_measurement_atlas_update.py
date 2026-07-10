#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 294
SCHEMA_VERSION = "2.21.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
SRC = ROOT / "tests/result/phase293_expanded_queue_behavior_readout_runner/expanded_queue_behavior_readout"
OUT = ROOT / "tests/result/phase294_expanded_measurement_atlas_update"
MODELS = ["qwen3", "glm4", "deepseek7b"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def load_rows(kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        rows.extend(read_jsonl(SRC / f"phase293_{model}_expanded_{kind}_rows.jsonl"))
    return rows


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k) or "") for k in keys)].append(row)
    return out


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model")), str(row.get("case_id"))


def expanded_path_signatures(behavior: list[dict[str, Any]], readout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    readout_by_key = {key(r): r for r in readout}
    rows = []
    for b in behavior:
        r = readout_by_key.get(key(b), {})
        target_rank = safe_float(r.get("target_rank"), 999999.0)
        behavior_score = 1.0 if b.get("answer_correct_proxy") else 0.0
        readout_score = 1.0 if target_rank <= 5 else 0.5 if target_rank <= 100 else 0.0
        rollout_score = 1.0 if not b.get("has_drift_marker") else 0.25
        closure_score = (0.35 if b.get("answer_correct_proxy") else 0.0) + (0.25 if b.get("pattern_matched_proxy") else 0.0) + (0.25 if b.get("model_stop_executed") else 0.0) + (0.15 if r.get("competition_winner") == "stop" else 0.0)
        closure_score = clamp(closure_score)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase294",
                "created_at": now(),
                "signature_id": f"phase294:expanded_signature:{b.get('model')}:{b.get('case_id')}",
                "case_id": b.get("case_id"),
                "model": b.get("model"),
                "family_id": b.get("family_id"),
                "mode_id": b.get("mode_id"),
                "variant_id": b.get("variant_id"),
                "path_schema_id": b.get("path_schema_id"),
                "target": b.get("target"),
                "path_signature": {
                    "trigger": b.get("channel_focus"),
                    "state": ["S_content", "S_target", "S_protocol", "S_boundary"],
                    "dominant_layers": [],
                    "attention_route_score": 0.0,
                    "mlp_write_score": 0.0,
                    "compensation_score": 0.0,
                    "readout_winner": r.get("competition_winner"),
                    "top_competitor": r.get("top_continue_channel"),
                    "strict_protocol_clean_count": 1 if b.get("pattern_matched_proxy") else 0,
                },
                "scores": {
                    "behavior": behavior_score,
                    "readout": readout_score,
                    "layer_path": 0.0,
                    "component_path": 0.0,
                    "causal": 0.0,
                    "rollout": rollout_score,
                    "closure": closure_score,
                    "overall": round(0.22 * behavior_score + 0.20 * readout_score + 0.18 * rollout_score + 0.20 * closure_score, 6),
                },
                "status": "expanded_measured_partial",
                "detail_ref": None,
                "measurement_note": "expanded behavior/readout measurement only; no layer/component/causal path yet",
            }
        )
    return rows


def expanded_gap_rows(signatures: list[dict[str, Any]], behavior: list[dict[str, Any]], readout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    b_by_key = {key(r): r for r in behavior}
    r_by_key = {key(r): r for r in readout}
    rows = []
    for sig in signatures:
        b = b_by_key.get(key(sig), {})
        r = r_by_key.get(key(sig), {})
        flags = {
            "need_component_path": True,
            "need_layer_path": True,
            "need_causal_audit": True,
            "need_closure_quality": bool(b.get("answer_correct_proxy")),
            "need_readout_competition": r.get("competition_winner") != "stop",
            "candidate_not_closed": not bool(b.get("model_stop_executed") and r.get("competition_winner") == "stop" and b.get("pattern_matched_proxy")),
            "good_behavior_low_path": bool(b.get("answer_correct_proxy")),
            "good_readout_low_causal": safe_float(r.get("target_rank"), 999999.0) <= 5,
        }
        priority = 0.0
        priority += 3.0 if flags["need_readout_competition"] else 0.0
        priority += 2.0 if flags["need_component_path"] else 0.0
        priority += 2.0 if flags["need_causal_audit"] else 0.0
        priority += 1.5 if flags["need_closure_quality"] else 0.0
        priority += safe_float(r.get("top_continue_vs_stop_margin")) / 10.0
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase294",
                "created_at": now(),
                "gap_id": f"phase294:expanded_gap:{sig.get('model')}:{sig.get('case_id')}",
                "source_signature_id": sig.get("signature_id"),
                "case_id": sig.get("case_id"),
                "model": sig.get("model"),
                "family_id": sig.get("family_id"),
                "mode_id": sig.get("mode_id"),
                "variant_id": sig.get("variant_id"),
                "status": sig.get("status"),
                "scores": sig.get("scores"),
                "remaining_gap_flags": flags,
                "remaining_dimensions": [name for name, value in flags.items() if value and name.startswith("need_")],
                "priority_score": round(priority, 6),
                "recommended_next_test": {
                    "component_path": "layerwise attention/mlp/residual contribution sweep",
                    "causal": "low-side-effect channel-specific causal audit",
                    "closure_quality": "four-condition closure quality check after readout competition decomposition",
                },
            }
        )
    return rows


def family_model_rows(behavior: list[dict[str, Any]], readout: list[dict[str, Any]], signatures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    b_group = group(behavior, "family_id", "model")
    r_group = group(readout, "family_id", "model")
    s_group = group(signatures, "family_id", "model")
    rows = []
    for cell, brows in sorted(b_group.items()):
        family, model = cell
        rrows = r_group.get(cell, [])
        srows = s_group.get(cell, [])
        total = len(brows)
        continue_wins = sum(1 for r in rrows if r.get("competition_winner") == "continue")
        stop_wins = sum(1 for r in rrows if r.get("competition_winner") == "stop")
        stop_exec = sum(1 for b in brows if b.get("model_stop_executed"))
        answer = sum(1 for b in brows if b.get("answer_correct_proxy"))
        pattern = sum(1 for b in brows if b.get("pattern_matched_proxy"))
        completion = clamp(
            0.25 * rate(answer, total)
            + 0.20 * rate(pattern, total)
            + 0.20 * rate(stop_exec, total)
            + 0.15 * rate(stop_wins, len(rrows))
            + 0.10 * (1.0 - rate(continue_wins, len(rrows)))
            + 0.10 * mean_safe([safe_float((s.get("scores") or {}).get("overall")) for s in srows])
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase294",
                "created_at": now(),
                "family_model_update_id": f"phase294:expanded_family_model:{family}:{model}",
                "family_id": family,
                "model": model,
                "rows": total,
                "answer_correct_proxy_rate": rate(answer, total),
                "pattern_matched_proxy_rate": rate(pattern, total),
                "model_stop_executed_rate": rate(stop_exec, total),
                "continue_winner_rate": rate(continue_wins, len(rrows)),
                "stop_winner_rate": rate(stop_wins, len(rrows)),
                "mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in rrows]),
                "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in rrows)),
                "expanded_completion_score": completion,
            }
        )
    return rows


def cross_model_summary(behavior: list[dict[str, Any]], readout: list[dict[str, Any]], fm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_b = group(behavior, "model")
    by_model_r = group(readout, "model")
    model_summary = {}
    for model in MODELS:
        brows = by_model_b.get((model,), [])
        rrows = by_model_r.get((model,), [])
        model_summary[model] = {
            "rows": len(brows),
            "answer_correct_proxy_rate": rate(sum(1 for r in brows if r.get("answer_correct_proxy")), len(brows)),
            "pattern_matched_proxy_rate": rate(sum(1 for r in brows if r.get("pattern_matched_proxy")), len(brows)),
            "model_stop_executed_rate": rate(sum(1 for r in brows if r.get("model_stop_executed")), len(brows)),
            "continue_winner_rate": rate(sum(1 for r in rrows if r.get("competition_winner") == "continue"), len(rrows)),
            "stop_winner_rate": rate(sum(1 for r in rrows if r.get("competition_winner") == "stop"), len(rrows)),
            "mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in rrows]),
            "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in rrows)),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase294",
        "created_at": now(),
        "source_phase": "Phase293",
        "expanded_behavior_rows": len(behavior),
        "expanded_readout_rows": len(readout),
        "expanded_path_signature_rows": len(behavior),
        "expanded_gap_rows": len(behavior),
        "family_model_update_rows": len(fm_rows),
        "global_answer_correct_proxy_rate": rate(sum(1 for r in behavior if r.get("answer_correct_proxy")), len(behavior)),
        "global_pattern_matched_proxy_rate": rate(sum(1 for r in behavior if r.get("pattern_matched_proxy")), len(behavior)),
        "global_model_stop_executed_rate": rate(sum(1 for r in behavior if r.get("model_stop_executed")), len(behavior)),
        "global_continue_winner_rate": rate(sum(1 for r in readout if r.get("competition_winner") == "continue"), len(readout)),
        "global_stop_winner_rate": rate(sum(1 for r in readout if r.get("competition_winner") == "stop"), len(readout)),
        "global_mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in readout]),
        "model_summary": model_summary,
        "progress_estimate": {
            "pattern_family_atlas": 0.74,
            "sample_type_coverage": 0.68,
            "feature_mining": 0.54,
            "physical_distribution_puzzle": 0.66,
            "mechanism_audit": 0.44,
            "closure": 0.21,
        },
    }


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase294_expanded_behavior_rows",
        "phase294_expanded_readout_rows",
        "phase294_expanded_path_signature_rows",
        "phase294_expanded_gap_rows",
        "phase294_expanded_family_model_update_rows",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase294_cross_model_summary"] = "phase294_cross_model_summary.json"
    files["phase294_report"] = "phase294_report.md"
    manifest["latest_expanded_measurement_phase"] = "Phase294"
    manifest["phase294_summary"] = summary
    write_json(V2 / "manifest.json", manifest)

    client = read_json(V2 / "client_index.json")
    for item in [
        "phase294_cross_model_summary.json",
        "phase294_expanded_family_model_update_rows.jsonl",
        "phase294_expanded_gap_rows.jsonl",
    ]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase294_summary_ref"] = "phase294_cross_model_summary.json"
    write_json(V2 / "client_index.json", client)

    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase294_expanded_behavior_rows"] = "measured behavior rows for Phase291 expanded sample queue"
    tables["phase294_expanded_readout_rows"] = "measured readout rows for Phase291 expanded sample queue"
    tables["phase294_expanded_path_signature_rows"] = "partial path signatures from expanded behavior/readout measurement"
    tables["phase294_expanded_gap_rows"] = "gap rows for expanded measured samples"
    tables["phase294_expanded_family_model_update_rows"] = "family-model aggregate update from expanded measurement"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    behavior = load_rows("behavior")
    readout = load_rows("readout")
    signatures = expanded_path_signatures(behavior, readout)
    gaps = expanded_gap_rows(signatures, behavior, readout)
    fm_rows = family_model_rows(behavior, readout, signatures)
    summary = cross_model_summary(behavior, readout, fm_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase294_expanded_behavior_rows.jsonl": behavior,
        "phase294_expanded_readout_rows.jsonl": readout,
        "phase294_expanded_path_signature_rows.jsonl": signatures,
        "phase294_expanded_gap_rows.jsonl": gaps,
        "phase294_expanded_family_model_update_rows.jsonl": fm_rows,
    }
    for name, rows in outputs.items():
        write_jsonl(OUT / name, rows)
        write_jsonl(V2 / name, rows)
    write_json(OUT / "phase294_cross_model_summary.json", summary)
    write_json(V2 / "phase294_cross_model_summary.json", summary)
    report = "\n".join(
        [
            "# Phase294 Expanded Measurement Atlas Update",
            "",
            f"- expanded_behavior_rows: {summary['expanded_behavior_rows']}",
            f"- expanded_readout_rows: {summary['expanded_readout_rows']}",
            f"- global_answer_correct_proxy_rate: {summary['global_answer_correct_proxy_rate']}",
            f"- global_pattern_matched_proxy_rate: {summary['global_pattern_matched_proxy_rate']}",
            f"- global_model_stop_executed_rate: {summary['global_model_stop_executed_rate']}",
            f"- global_continue_winner_rate: {summary['global_continue_winner_rate']}",
            f"- global_stop_winner_rate: {summary['global_stop_winner_rate']}",
            f"- model_summary: {json.dumps(summary['model_summary'], ensure_ascii=False)}",
            "",
            "Expanded rows are measured behavior/readout evidence. Layer, component, and causal paths remain open.",
        ]
    ) + "\n"
    (OUT / "phase294_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase294_report.md").write_text(report, encoding="utf-8")
    update_v2(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
