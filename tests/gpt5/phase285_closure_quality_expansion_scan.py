#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase938_semantic_factor_causal_transfer_audit as p938  # noqa: E402
import phase266_multi_family_baseline_behavior_readout_scan as p266  # noqa: E402
import phase281_candidate_closure_quality_verification as p281  # noqa: E402


PHASE = 285
SCHEMA_VERSION = "2.12.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
RESULT_ROOT = ROOT / "tests/result/phase285_closure_quality_expansion_scan"
ROUND_DEFAULT = "closure_quality_expansion_scan"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def select_model_rows(model: str, max_cases: int, min_behavior: float) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(V2 / "phase284_recalibrated_gap_rows.jsonl"):
        if row.get("model") != model:
            continue
        flags = row.get("remaining_gap_flags") or {}
        scores = row.get("scores") or {}
        if not flags.get("need_closure_quality"):
            continue
        if safe_float(scores.get("behavior")) < min_behavior:
            continue
        if flags.get("candidate_not_closed"):
            continue
        rows.append(row)
    rows.sort(
        key=lambda r: (
            -safe_float((r.get("scores") or {}).get("behavior")),
            -safe_float((r.get("scores") or {}).get("readout")),
            -safe_float((r.get("scores") or {}).get("rollout")),
            str(r.get("family_id")),
            str(r.get("case_id")),
        )
    )
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for row in rows:
        family = str(row.get("family_id"))
        if family_counts[family] >= 2:
            continue
        selected.append(row)
        family_counts[family] += 1
        if len(selected) >= max_cases:
            break
    for row in rows:
        if len(selected) >= max_cases:
            break
        if row not in selected:
            selected.append(row)
    return selected[:max_cases]


def make_rows(model: str, row: dict[str, Any], case: dict[str, Any], readout: dict[str, Any], output: str, stopped: bool, new_tokens: int, rollout_tokens: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    q = p281.closure_row(model, row, case, readout, output, stopped, new_tokens, rollout_tokens)
    q["schema_version"] = SCHEMA_VERSION
    q["phase_id"] = "Phase285"
    q["created_at"] = utc_now()
    q["closure_quality_id"] = f"phase285:closure_quality:{model}:{row['case_id']}"
    q["source_phase284_gap_id"] = row.get("gap_id")
    q["source_phase284_status"] = row.get("phase284_status")
    q["source_phase284_priority"] = row.get("priority_score_after_phase283")
    q["rollout_tokens_limit"] = int(rollout_tokens)
    if q["four_condition_closed"]:
        reclass = "closure_test_ready"
    elif q["semantic_done"] and q["protocol_matched"]:
        reclass = "semantic_protocol_ok_but_not_closed"
    else:
        reclass = "closure_rejected"
    q["closure_reclassification"] = reclass
    four = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase285",
        "created_at": utc_now(),
        "four_condition_id": f"phase285:four_condition:{model}:{row['case_id']}",
        "model": model,
        "case_id": row["case_id"],
        "family_id": row.get("family_id"),
        "mode_id": row.get("mode_id"),
        "variant_id": row.get("variant_id"),
        "semantic_done": q["semantic_done"],
        "protocol_matched": q["protocol_matched"],
        "stop_wins": q["stop_wins"],
        "continue_suppressed": q["continue_suppressed"],
        "rollout_stable": q["rollout_stable"],
        "four_condition_closed": q["four_condition_closed"],
        "closure_reclassification": reclass,
        "closure_blockers": q["closure_blockers"],
    }
    rollout = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase285",
        "created_at": utc_now(),
        "rollout_stability_id": f"phase285:rollout_stability:{model}:{row['case_id']}",
        "model": model,
        "case_id": row["case_id"],
        "family_id": row.get("family_id"),
        "mode_id": row.get("mode_id"),
        "variant_id": row.get("variant_id"),
        "generated_text": output[:800],
        "generated_token_count": new_tokens,
        "rollout_tokens_limit": int(rollout_tokens),
        "model_stop_executed": stopped,
        "rollout_stable": q["rollout_stable"],
        "has_drift_marker": q["has_drift_marker"],
        "repeated_protocol_marker": q["repeated_protocol_marker"],
        "pattern_matched_proxy": q["pattern_matched_proxy"],
        "answer_correct_proxy": q["answer_correct_proxy"],
    }
    return q, four, rollout


def update_detail(model: str, case_id: str, row: dict[str, Any]) -> None:
    detail_ref = V2 / "case_details" / f"{model}__{case_id}.json"
    if not detail_ref.exists():
        return
    detail = read_json(detail_ref)
    detail["phase285_closure_quality"] = row
    detail["phase285_updated_at"] = utc_now()
    write_json(detail_ref, detail)


def evaluate_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_model_rows(model, int(args.max_cases_per_model), float(args.min_behavior))
    model_obj = None
    tokenizer = None
    quality_rows: list[dict[str, Any]] = []
    four_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    try:
        model_obj, tokenizer, device, _attn_impl = p938.p862.p844.p828.p796.load_model_bf16_prefer_flash(model, args.attn_implementations)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for idx, gap_row in enumerate(selected, start=1):
            try:
                case = p281.load_case(gap_row)
                prompt = str(case["prompt"])
                aliases = [str(x) for x in case.get("target_aliases") or [case.get("target", "")]]
                readout = p266.capture_readout(model_obj, tokenizer, device, prompt, aliases)
                output, stopped, new_tokens = p266.generate_probe(model_obj, tokenizer, device, prompt, int(args.rollout_tokens))
                qrow, frow, rrow = make_rows(model, gap_row, case, readout, output, stopped, new_tokens, int(args.rollout_tokens))
                quality_rows.append(qrow)
                four_rows.append(frow)
                rollout_rows.append(rrow)
                update_detail(model, str(gap_row["case_id"]), qrow)
            except Exception as exc:  # noqa: BLE001
                missing_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase285",
                        "created_at": utc_now(),
                        "missing_id": f"phase285:missing:{model}:{gap_row.get('case_id')}",
                        "model": model,
                        "case_id": gap_row.get("case_id"),
                        "family_id": gap_row.get("family_id"),
                        "reason": repr(exc),
                    }
                )
            log(f"{model}: phase285 closure quality scanned {idx}/{len(selected)} rows")
    finally:
        if model_obj is not None:
            p938.p862.p844.p828.release_model(model_obj)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    summary = summarize_model(model, selected, quality_rows, missing_rows)
    write_json(out_dir / f"phase285_{model}_summary.json", summary)
    write_jsonl(out_dir / f"phase285_{model}_closure_quality_rows.jsonl", quality_rows)
    write_jsonl(out_dir / f"phase285_{model}_four_condition_rows.jsonl", four_rows)
    write_jsonl(out_dir / f"phase285_{model}_rollout_stability_rows.jsonl", rollout_rows)
    write_jsonl(out_dir / f"phase285_{model}_missing_rows.jsonl", missing_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize_model(model: str, selected: list[dict[str, Any]], rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase285",
        "title": "closure quality expansion scan",
        "status": "complete",
        "created_at": utc_now(),
        "model": model,
        "selected_rows": len(selected),
        "closure_quality_rows": len(rows),
        "missing_rows": len(missing),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in rows)),
        "reclassification_counts": dict(Counter(str(r.get("closure_reclassification")) for r in rows)),
        "semantic_done_count": sum(1 for r in rows if r.get("semantic_done")),
        "stop_wins_count": sum(1 for r in rows if r.get("stop_wins")),
        "continue_suppressed_count": sum(1 for r in rows if r.get("continue_suppressed")),
        "rollout_stable_count": sum(1 for r in rows if r.get("rollout_stable")),
        "four_condition_closed_count": sum(1 for r in rows if r.get("four_condition_closed")),
        "blocker_counts": dict(Counter(b for r in rows for b in r.get("closure_blockers", []))),
        "mean_stop_continue_margin": mean_safe([safe_float(r.get("stop_continue_margin")) for r in rows]),
    }


def update_v2(round_name: str, payload: dict[str, Any]) -> None:
    out_dir = RESULT_ROOT / round_name
    table_names = ["closure_quality_rows", "four_condition_rows", "rollout_stability_rows", "missing_rows"]
    for table in table_names:
        rows: list[dict[str, Any]] = []
        for model in MODELS:
            rows.extend(read_jsonl(out_dir / f"phase285_{model}_{table}.jsonl"))
        write_jsonl(V2 / f"phase285_{table}.jsonl", rows)
    write_json(V2 / "phase285_cross_model_summary.json", payload)
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for table in table_names:
        files[f"phase285_{table}"] = f"phase285_{table}.jsonl"
    files["phase285_cross_model_summary"] = "phase285_cross_model_summary.json"
    files["phase285_report"] = "phase285_report.md"
    manifest["latest_closure_quality_phase"] = "Phase285"
    manifest["phase285_summary"] = payload
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase285_cross_model_summary.json", "phase285_closure_quality_rows.jsonl", "phase285_four_condition_rows.jsonl", "phase285_rollout_stability_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase285_summary_ref"] = "phase285_cross_model_summary.json"
    client["phase285_closure_quality_ref"] = "phase285_closure_quality_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase285_closure_quality_rows"] = "expanded closure-quality scan for high-behavior need_closure_quality rows"
    tables["phase285_four_condition_rows"] = "four-condition closure decomposition"
    tables["phase285_rollout_stability_rows"] = "32-token rollout stability probes"
    write_json(V2 / "schema.json", schema)


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase285_{model}_summary.json") for model in MODELS]
    summaries = [s for s in summaries if s]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model in MODELS:
        rows.extend(read_jsonl(out_dir / f"phase285_{model}_closure_quality_rows.jsonl"))
        missing.extend(read_jsonl(out_dir / f"phase285_{model}_missing_rows.jsonl"))
    payload = {
        "phase": PHASE,
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase285",
        "title": "closure quality expansion scan",
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "created_at": utc_now(),
        "round_name": round_name,
        "model_summaries": summaries,
        "closure_quality_rows": len(rows),
        "missing_rows": len(missing),
        "model_counts": dict(Counter(str(r.get("model")) for r in rows)),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in rows)),
        "reclassification_counts": dict(Counter(str(r.get("closure_reclassification")) for r in rows)),
        "semantic_done_rate": round(sum(1 for r in rows if r.get("semantic_done")) / len(rows), 6) if rows else 0.0,
        "stop_wins_rate": round(sum(1 for r in rows if r.get("stop_wins")) / len(rows), 6) if rows else 0.0,
        "continue_suppressed_rate": round(sum(1 for r in rows if r.get("continue_suppressed")) / len(rows), 6) if rows else 0.0,
        "rollout_stable_rate": round(sum(1 for r in rows if r.get("rollout_stable")) / len(rows), 6) if rows else 0.0,
        "four_condition_closed_count": sum(1 for r in rows if r.get("four_condition_closed")),
        "blocker_counts": dict(Counter(b for r in rows for b in r.get("closure_blockers", []))),
        "mean_stop_continue_margin": mean_safe([safe_float(r.get("stop_continue_margin")) for r in rows]),
    }
    write_json(out_dir / "phase285_cross_model_summary.json", payload)
    update_v2(round_name, payload)
    lines = [
        "# Phase285 Closure Quality Expansion Scan",
        "",
        f"- closure_quality_rows: {payload['closure_quality_rows']}",
        f"- missing_rows: {payload['missing_rows']}",
        f"- four_condition_closed_count: {payload['four_condition_closed_count']}",
        f"- reclassification_counts: {json.dumps(payload['reclassification_counts'], ensure_ascii=False)}",
        f"- blocker_counts: {json.dumps(payload['blocker_counts'], ensure_ascii=False)}",
        "",
        "This phase expands closure-quality measurement. It is not a closure claim.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "phase285_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase285_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases-per-model", type=int, default=9)
    parser.add_argument("--min-behavior", type=float, default=0.5)
    parser.add_argument("--rollout-tokens", type=int, default=32)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_round(args.round_name)
        return
    if args.model:
        evaluate_model(args, args.model)
        return
    for model in MODELS:
        evaluate_model(args, model)
    summarize_round(args.round_name)


if __name__ == "__main__":
    main()
