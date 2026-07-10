#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 290
SCHEMA_VERSION = "2.17.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase290_readout_competition_channel_decomposition"
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def median_safe(values: list[float]) -> float:
    return round(median(values), 6) if values else 0.0


def rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def confidence_flag(coverage_rate: float) -> str:
    if coverage_rate >= 0.8:
        return "high_coverage"
    if coverage_rate >= 0.4:
        return "medium_coverage"
    if coverage_rate > 0:
        return "low_coverage"
    return "no_coverage"


def channel_family(channel: str) -> str:
    channel = (channel or "unknown").lower()
    if "json" in channel:
        return "protocol_json_continue"
    if "format" in channel:
        return "protocol_format_continue"
    if "list" in channel:
        return "list_structure_continue"
    if "next_sentence" in channel or channel.endswith("_the") or "continue_the" in channel:
        return "natural_language_continue"
    if "because" in channel or "for" in channel or "is" in channel:
        return "explanation_relation_continue"
    if "comma" in channel or "and" in channel:
        return "local_syntax_continue"
    if "answer_boundary" in channel or "boundary" in channel:
        return "answer_boundary_continue"
    if "stop" in channel or "eos" in channel:
        return "stop_channel"
    return "other_continue"


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    out: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(str(row.get(k) or "") for k in keys)].append(row)
    return out


def load_closure_quality_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in ["phase281", "phase285"]:
        rows.extend(read_jsonl(V2 / f"{phase}_closure_quality_rows.jsonl"))
    return rows


def latest_gap_index() -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(V2 / "phase286_recalibrated_gap_rows.jsonl")
    return {(str(r.get("model")), str(r.get("case_id"))): r for r in rows}


def closure_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("model")), str(r.get("case_id"))): r for r in rows}


def feature_priority_cells() -> set[tuple[str, str]]:
    cells: set[tuple[str, str]] = set()
    for row in read_jsonl(V2 / "phase289_feature_driven_audit_candidates.jsonl"):
        if row.get("recommended_next") == "readout_competition_channel_decomposition":
            cells.add((str(row.get("model")), str(row.get("family_id"))))
    return cells


def detail_for_signature(row: dict[str, Any]) -> dict[str, Any]:
    detail_ref = row.get("detail_ref")
    if not detail_ref:
        return {}
    return read_json(V2 / str(detail_ref))


def row_from_signature(
    sig: dict[str, Any],
    detail: dict[str, Any],
    gap: dict[str, Any] | None,
    closure: dict[str, Any] | None,
    priority_cells: set[tuple[str, str]],
) -> dict[str, Any]:
    readout = detail.get("readout") or {}
    behavior = detail.get("behavior") or {}
    case = detail.get("case") or {}
    scores = sig.get("scores") or {}
    path_signature = sig.get("path_signature") or {}
    model = str(sig.get("model"))
    family = str(sig.get("family_id"))
    top_channel = str(readout.get("top_continue_channel") or path_signature.get("top_competitor") or "unknown")
    top_margin = safe_float(readout.get("top_continue_vs_stop_margin"))
    stop_margin = safe_float(readout.get("stop_continue_margin"))
    winner = str(readout.get("competition_winner") or path_signature.get("readout_winner") or "unknown")
    closure_blockers = list((closure or {}).get("closure_blockers") or [])
    gap_flags = (gap or {}).get("remaining_gap_flags") or {}
    target_rank = safe_int(readout.get("target_rank"))
    channel_kind = channel_family(top_channel)
    bottlenecks: list[str] = []
    if winner != "stop":
        bottlenecks.append("stop_not_winner")
    if top_margin >= -0.25:
        bottlenecks.append("continue_not_suppressed")
    if target_rank > 100:
        bottlenecks.append("target_readout_weak")
    if channel_kind.startswith("protocol_") or channel_kind == "list_structure_continue":
        bottlenecks.append("protocol_or_structure_continue")
    if bool(gap_flags.get("need_readout_competition")):
        bottlenecks.append("gap_need_readout_competition")
    if closure_blockers:
        bottlenecks.extend([f"closure_{b}" for b in closure_blockers])
    priority_reasons: list[str] = []
    if (model, family) in priority_cells:
        priority_reasons.append("phase289_readout_competition_cell")
    if bool(gap_flags.get("need_readout_competition")):
        priority_reasons.append("gap_queue_need_readout_competition")
    if closure and not closure.get("four_condition_closed"):
        priority_reasons.append("closure_quality_rejected")
    priority_score = (
        abs(top_margin)
        + (2.0 if winner == "continue" else 0.0)
        + (1.5 if (model, family) in priority_cells else 0.0)
        + (1.0 if gap_flags.get("need_readout_competition") else 0.0)
        + (1.0 if closure_blockers else 0.0)
        + (0.5 if target_rank > 100 else 0.0)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase290",
        "created_at": now(),
        "readout_channel_id": f"phase290:readout_channel:{model}:{sig.get('case_id')}",
        "signature_id": sig.get("signature_id"),
        "case_id": sig.get("case_id"),
        "detail_ref": sig.get("detail_ref"),
        "model": model,
        "family_id": family,
        "mode_id": sig.get("mode_id"),
        "variant_id": sig.get("variant_id"),
        "output_protocol": case.get("output_protocol") or behavior.get("output_protocol"),
        "continuation_trigger": readout.get("continuation_trigger") or case.get("continuation_trigger"),
        "expected_pattern": case.get("expected_pattern") or behavior.get("expected_pattern"),
        "target": sig.get("target"),
        "target_rank": target_rank,
        "target_logit": safe_float(readout.get("target_logit")),
        "target_margin_vs_winner": safe_float(readout.get("target_margin_vs_winner")),
        "r_stop": safe_float(readout.get("r_stop")),
        "r_stop_name": readout.get("r_stop_name"),
        "r_continue": safe_float(readout.get("r_continue")),
        "r_continue_name": readout.get("r_continue_name"),
        "competition_winner": winner,
        "winning_regime": readout.get("winning_regime"),
        "top_token": readout.get("top_token"),
        "top_continue_channel": top_channel,
        "continue_channel_family": channel_kind,
        "second_continue_channel": readout.get("second_continue_channel"),
        "second_competitor": readout.get("second_competitor"),
        "top_continue_vs_stop_margin": round(top_margin, 6),
        "stop_continue_margin": round(stop_margin, 6),
        "scores": scores,
        "answer_correct_proxy": bool(behavior.get("answer_correct_proxy") or safe_float(scores.get("behavior")) >= 0.5),
        "model_stop_executed": bool(behavior.get("model_stop_executed")),
        "closure_quality_checked": closure is not None,
        "four_condition_closed": bool((closure or {}).get("four_condition_closed")),
        "closure_blockers": sorted(set(closure_blockers)),
        "remaining_gap_flags": gap_flags,
        "phase289_priority_cell": (model, family) in priority_cells,
        "readout_bottlenecks": sorted(set(bottlenecks)),
        "priority_reasons": sorted(set(priority_reasons)),
        "priority_score": round(priority_score, 6),
    }


def build_channel_rows(signatures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = latest_gap_index()
    closures = closure_index(load_closure_quality_rows())
    priority_cells = feature_priority_cells()
    rows = []
    for sig in signatures:
        detail = detail_for_signature(sig)
        rows.append(row_from_signature(sig, detail, gaps.get((str(sig.get("model")), str(sig.get("case_id")))), closures.get((str(sig.get("model")), str(sig.get("case_id")))), priority_cells))
    return rows


def channel_matrix(rows: list[dict[str, Any]], total_signatures: int) -> list[dict[str, Any]]:
    out = []
    for (family, model, channel_family_name, channel), bucket in sorted(group(rows, "family_id", "model", "continue_channel_family", "top_continue_channel").items()):
        margins = [safe_float(r.get("top_continue_vs_stop_margin")) for r in bucket]
        target_ranks = [safe_float(r.get("target_rank")) for r in bucket if safe_float(r.get("target_rank")) > 0]
        coverage = rate(len(bucket), total_signatures)
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase290",
                "created_at": now(),
                "channel_matrix_id": f"phase290:channel_matrix:{family}:{model}:{channel}",
                "family_id": family,
                "model": model,
                "continue_channel_family": channel_family_name,
                "top_continue_channel": channel,
                "rows": len(bucket),
                "coverage_count": len(bucket),
                "coverage_rate": coverage,
                "confidence_flag": confidence_flag(coverage),
                "continue_winner_rate": rate(sum(1 for r in bucket if r.get("competition_winner") == "continue"), len(bucket)),
                "stop_winner_rate": rate(sum(1 for r in bucket if r.get("competition_winner") == "stop"), len(bucket)),
                "mean_top_continue_vs_stop_margin": mean_safe(margins),
                "median_top_continue_vs_stop_margin": median_safe(margins),
                "mean_target_rank": mean_safe(target_ranks),
                "median_target_rank": median_safe(target_ranks),
                "answer_correct_proxy_rate": rate(sum(1 for r in bucket if r.get("answer_correct_proxy")), len(bucket)),
                "closure_checked_count": sum(1 for r in bucket if r.get("closure_quality_checked")),
                "closure_closed_count": sum(1 for r in bucket if r.get("four_condition_closed")),
                "readout_bottleneck_counts": dict(Counter(b for r in bucket for b in r.get("readout_bottlenecks", []))),
            }
        )
    return out


def bottleneck_rows(rows: list[dict[str, Any]], total_signatures: int) -> list[dict[str, Any]]:
    out = []
    expanded = []
    for row in rows:
        for bottleneck in row.get("readout_bottlenecks") or ["none"]:
            expanded.append({**row, "bottleneck": bottleneck})
    for (family, model, channel_family_name, bottleneck), bucket in sorted(group(expanded, "family_id", "model", "continue_channel_family", "bottleneck").items()):
        if bottleneck == "none":
            continue
        margins = [safe_float(r.get("top_continue_vs_stop_margin")) for r in bucket]
        coverage = rate(len(bucket), total_signatures)
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase290",
                "created_at": now(),
                "bottleneck_id": f"phase290:bottleneck:{family}:{model}:{channel_family_name}:{bottleneck}",
                "family_id": family,
                "model": model,
                "continue_channel_family": channel_family_name,
                "readout_bottleneck": bottleneck,
                "rows": len(bucket),
                "coverage_count": len(bucket),
                "coverage_rate": coverage,
                "confidence_flag": confidence_flag(coverage),
                "mean_priority_score": mean_safe([safe_float(r.get("priority_score")) for r in bucket]),
                "mean_top_continue_vs_stop_margin": mean_safe(margins),
                "max_top_continue_vs_stop_margin": round(max(margins), 6) if margins else 0.0,
                "phase289_priority_rows": sum(1 for r in bucket if r.get("phase289_priority_cell")),
                "closure_checked_rows": sum(1 for r in bucket if r.get("closure_quality_checked")),
                "case_ids": [r.get("case_id") for r in sorted(bucket, key=lambda x: -safe_float(x.get("priority_score")))[:8]],
            }
        )
    out.sort(key=lambda r: (-safe_float(r.get("mean_priority_score")), -int(r.get("rows")), str(r.get("model")), str(r.get("family_id"))))
    return out


def audit_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        r
        for r in rows
        if r.get("phase289_priority_cell")
        or "gap_queue_need_readout_competition" in (r.get("priority_reasons") or [])
        or "closure_quality_rejected" in (r.get("priority_reasons") or [])
    ]
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[(str(row.get("model")), str(row.get("family_id")), str(row.get("continue_channel_family")))].append(row)
    selected: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda r: (-safe_float(r.get("priority_score")), -abs(safe_float(r.get("top_continue_vs_stop_margin"))), str(r.get("case_id"))))
        selected.extend(bucket[:3])
    selected.sort(key=lambda r: (-safe_float(r.get("priority_score")), str(r.get("model")), str(r.get("family_id")), str(r.get("case_id"))))
    out = []
    for rank, row in enumerate(selected[:144], start=1):
        audit_kind = "readout_channel_decomposition"
        if "target_readout_weak" in row.get("readout_bottlenecks", []):
            audit_kind = "target_vs_continue_competition"
        if "protocol_or_structure_continue" in row.get("readout_bottlenecks", []):
            audit_kind = "protocol_continue_suppression"
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase290",
                "created_at": now(),
                "audit_queue_id": f"phase290:audit_queue:{rank:03d}:{row.get('model')}:{row.get('case_id')}",
                "phase290_rank": rank,
                "recommended_next": audit_kind,
                "model": row.get("model"),
                "family_id": row.get("family_id"),
                "case_id": row.get("case_id"),
                "detail_ref": row.get("detail_ref"),
                "top_continue_channel": row.get("top_continue_channel"),
                "continue_channel_family": row.get("continue_channel_family"),
                "competition_winner": row.get("competition_winner"),
                "top_continue_vs_stop_margin": row.get("top_continue_vs_stop_margin"),
                "target_rank": row.get("target_rank"),
                "readout_bottlenecks": row.get("readout_bottlenecks"),
                "priority_reasons": row.get("priority_reasons"),
                "priority_score": row.get("priority_score"),
                "test_instruction": "Use sequential qwen3 -> GLM4 -> DS7B CUDA audit only if moving from offline atlas mining to model intervention.",
            }
        )
    return out


def family_model_summary(rows: list[dict[str, Any]], total_by_cell: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    out = []
    for (family, model), bucket in sorted(group(rows, "family_id", "model").items()):
        total = total_by_cell.get((family, model), len(bucket))
        coverage = rate(len(bucket), total)
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase290",
                "created_at": now(),
                "family_model_summary_id": f"phase290:family_model_readout:{family}:{model}",
                "family_id": family,
                "model": model,
                "rows": len(bucket),
                "coverage_count": len(bucket),
                "coverage_rate": coverage,
                "confidence_flag": confidence_flag(coverage),
                "continue_winner_rate": rate(sum(1 for r in bucket if r.get("competition_winner") == "continue"), len(bucket)),
                "mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in bucket]),
                "channel_family_counts": dict(Counter(str(r.get("continue_channel_family")) for r in bucket)),
                "top_continue_channel_counts": dict(Counter(str(r.get("top_continue_channel")) for r in bucket).most_common(8)),
                "bottleneck_counts": dict(Counter(b for r in bucket for b in r.get("readout_bottlenecks", []))),
                "phase289_priority_rows": sum(1 for r in bucket if r.get("phase289_priority_cell")),
            }
        )
    return out


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase290_readout_channel_rows",
        "phase290_channel_family_model_matrix",
        "phase290_stop_continue_bottleneck_rows",
        "phase290_readout_competition_audit_queue",
        "phase290_family_model_readout_summary",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase290_summary"] = "phase290_summary.json"
    files["phase290_report"] = "phase290_report.md"
    manifest["latest_readout_decomposition_phase"] = "Phase290"
    manifest["phase290_summary"] = summary
    write_json(V2 / "manifest.json", manifest)

    client = read_json(V2 / "client_index.json")
    for item in [
        "phase290_summary.json",
        "phase290_channel_family_model_matrix.jsonl",
        "phase290_stop_continue_bottleneck_rows.jsonl",
        "phase290_readout_competition_audit_queue.jsonl",
        "phase290_family_model_readout_summary.jsonl",
    ]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase290_summary_ref"] = "phase290_summary.json"
    client["phase290_readout_audit_queue_ref"] = "phase290_readout_competition_audit_queue.jsonl"
    write_json(V2 / "client_index.json", client)

    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase290_readout_channel_rows"] = "one row per signature with decomposed stop/continue readout competition channel"
    tables["phase290_channel_family_model_matrix"] = "family x model x continue-channel readout competition matrix with coverage fields"
    tables["phase290_stop_continue_bottleneck_rows"] = "aggregated stop/continue bottlenecks by family, model, and channel family"
    tables["phase290_readout_competition_audit_queue"] = "case-level queue for the next channel-level readout competition audit"
    tables["phase290_family_model_readout_summary"] = "family x model readout channel summary"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    signatures = read_jsonl(V2 / "path_signature_rows.jsonl")
    rows = build_channel_rows(signatures)
    total_by_cell = Counter((str(r.get("family_id")), str(r.get("model"))) for r in signatures)
    matrix = channel_matrix(rows, len(signatures))
    bottlenecks = bottleneck_rows(rows, len(signatures))
    queue = audit_queue(rows)
    fm_summary = family_model_summary(rows, dict(total_by_cell))
    channel_counts = Counter(str(r.get("top_continue_channel")) for r in rows)
    family_counts = Counter(str(r.get("continue_channel_family")) for r in rows)
    bottleneck_counts = Counter(b for r in rows for b in r.get("readout_bottlenecks", []))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase290",
        "created_at": now(),
        "model_test_status": "not_run_offline_atlas_readout_decomposition",
        "source_signature_rows": len(signatures),
        "readout_channel_rows": len(rows),
        "channel_family_model_matrix_rows": len(matrix),
        "stop_continue_bottleneck_rows": len(bottlenecks),
        "readout_competition_audit_queue_rows": len(queue),
        "family_model_readout_summary_rows": len(fm_summary),
        "global_continue_winner_rate": rate(sum(1 for r in rows if r.get("competition_winner") == "continue"), len(rows)),
        "global_stop_winner_rate": rate(sum(1 for r in rows if r.get("competition_winner") == "stop"), len(rows)),
        "global_mean_top_continue_vs_stop_margin": mean_safe([safe_float(r.get("top_continue_vs_stop_margin")) for r in rows]),
        "top_continue_channel_counts": dict(channel_counts.most_common(12)),
        "continue_channel_family_counts": dict(family_counts.most_common()),
        "readout_bottleneck_counts": dict(bottleneck_counts.most_common()),
        "audit_queue_by_model": dict(Counter(str(r.get("model")) for r in queue)),
        "audit_queue_by_recommended_next": dict(Counter(str(r.get("recommended_next")) for r in queue)),
        "coverage_policy": "All rates include coverage_count, coverage_rate, and confidence_flag where aggregation is not full-table.",
        "progress_estimate": {
            "pattern_family_atlas": 0.66,
            "physical_distribution_puzzle": 0.63,
            "feature_mining": 0.42,
            "readout_competition_decomposition": 0.35,
            "mechanism_audit": 0.41,
            "closure": 0.20,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase290_readout_channel_rows.jsonl": rows,
        "phase290_channel_family_model_matrix.jsonl": matrix,
        "phase290_stop_continue_bottleneck_rows.jsonl": bottlenecks,
        "phase290_readout_competition_audit_queue.jsonl": queue,
        "phase290_family_model_readout_summary.jsonl": fm_summary,
    }
    for name, data in outputs.items():
        write_jsonl(OUT / name, data)
        write_jsonl(V2 / name, data)
    write_json(OUT / "phase290_summary.json", summary)
    write_json(V2 / "phase290_summary.json", summary)
    report = "\n".join(
        [
            "# Phase290 Readout Competition Channel Decomposition",
            "",
            f"- source_signature_rows: {summary['source_signature_rows']}",
            f"- readout_channel_rows: {summary['readout_channel_rows']}",
            f"- channel_family_model_matrix_rows: {summary['channel_family_model_matrix_rows']}",
            f"- stop_continue_bottleneck_rows: {summary['stop_continue_bottleneck_rows']}",
            f"- readout_competition_audit_queue_rows: {summary['readout_competition_audit_queue_rows']}",
            f"- global_continue_winner_rate: {summary['global_continue_winner_rate']}",
            f"- global_mean_top_continue_vs_stop_margin: {summary['global_mean_top_continue_vs_stop_margin']}",
            f"- continue_channel_family_counts: {json.dumps(summary['continue_channel_family_counts'], ensure_ascii=False)}",
            f"- readout_bottleneck_counts: {json.dumps(summary['readout_bottleneck_counts'], ensure_ascii=False)}",
            "",
            "This phase decomposes the stop/continue readout bottleneck into channel families and produces a case-level audit queue.",
            "It does not claim closure and does not run new model interventions.",
        ]
    ) + "\n"
    (OUT / "phase290_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase290_report.md").write_text(report, encoding="utf-8")
    update_v2(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
